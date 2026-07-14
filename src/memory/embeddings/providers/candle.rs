//! Local candle BERT embedding provider (feature `ml-models`)
//!
//! Wraps a candle `BertModel` as a first-class [`EmbeddingProvider`], peer to
//! the API-backed providers and selectable through
//! [`MultiProvider`](crate::memory::embeddings::MultiProvider). Inference is
//! real: tokenize, BERT forward pass, mean-pool the token embeddings.
//!
//! Construction fails closed: if the model directory is missing the
//! tokenizer, config, or safetensors weights, `new` returns an error — there
//! is no fake fallback provider.

use crate::error::{MemoryError, Result};
use crate::memory::embeddings::provider::{
    compute_content_hash, EmbedOptions, Embedding, EmbeddingProvider, ProviderCapabilities,
};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use dashmap::DashMap;
use std::path::Path;
use tokenizers::Tokenizer;

/// Embedding provider backed by a local candle BERT model.
///
/// Holds the loaded model and tokenizer; `embed` runs a real forward pass,
/// attention-mask mean-pools the final hidden states (the
/// sentence-transformers pooling used by all-MiniLM-L6-v2) and L2-normalizes
/// the result into a fixed-dimension vector (`hidden_size`, 384 for
/// all-MiniLM-L6-v2). `embed_for_scoring` and `embed_for_scoring_batch` run
/// the same inference through a content-hash cache; the batch path pads the
/// whole set to one `[batch, seq]` tensor and runs a SINGLE forward pass
/// (semantic embeddings have no corpus-IDF distinction, so scoring == embed).
pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
    max_length: usize,
    pad_token_id: u32,
    model_id: String,
    /// Query-time scoring cache keyed by content hash: BERT embeddings are
    /// deterministic and carry no corpus-relative state, so a content's
    /// scoring vector never goes stale. Shared between the dense retriever
    /// and the reranker, only cache MISSES are sent to the batched forward
    /// pass. Bounded by [`SCORING_CACHE_CAP`].
    scoring_cache: DashMap<String, Vec<f32>>,
}

/// Bound on the scoring cache; on overflow the cache is cleared wholesale
/// (crude but O(1)-amortized; affects only hit rate, never correctness).
const SCORING_CACHE_CAP: usize = 8192;

impl CandleEmbeddingProvider {
    /// Load a BERT model from a local directory containing
    /// `tokenizer.json`, `config.json`, and `model.safetensors`.
    ///
    /// Fails closed: any missing or unreadable file is an error; no
    /// placeholder provider is ever constructed.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        Self::from_dir_on_device(model_dir, device)
    }

    /// Load from a local directory onto a specific device.
    pub fn from_dir_on_device(model_dir: &Path, device: Device) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            MemoryError::configuration(format!(
                "candle embedding tokenizer not loadable at {}: {}",
                tokenizer_path.display(),
                e
            ))
        })?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            MemoryError::configuration(format!(
                "candle embedding config not readable at {}: {}",
                config_path.display(),
                e
            ))
        })?;
        let config: BertConfig = serde_json::from_str(&config_str).map_err(|e| {
            MemoryError::configuration(format!(
                "candle embedding config not parseable at {}: {}",
                config_path.display(),
                e
            ))
        })?;

        let weights_path = model_dir.join("model.safetensors");
        let weights = std::fs::read(&weights_path).map_err(|e| {
            MemoryError::configuration(format!(
                "candle embedding weights not readable at {}: {}",
                weights_path.display(),
                e
            ))
        })?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;

        let max_length = config.max_position_embeddings;
        let mut provider = Self::from_var_builder(vb, &config, tokenizer, max_length)?;
        provider.model_id = format!("candle-bert({})", model_dir.display());
        Ok(provider)
    }

    /// Build a provider from an already-constructed [`VarBuilder`] and
    /// tokenizer. This is the shared trunk of [`Self::new`]; it also lets
    /// tests exercise the real inference path from in-memory weights without
    /// touching the network or filesystem.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: &BertConfig,
        tokenizer: Tokenizer,
        max_length: usize,
    ) -> Result<Self> {
        if max_length == 0 || max_length > config.max_position_embeddings {
            return Err(MemoryError::configuration(format!(
                "candle embedding max_length {} must be in 1..={}",
                max_length, config.max_position_embeddings
            )));
        }
        let device = vb.device().clone();
        let model = BertModel::load(vb, config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            hidden_size: config.hidden_size,
            max_length,
            pad_token_id: config.pad_token_id as u32,
            model_id: "candle-bert(in-memory)".to_string(),
            scoring_cache: DashMap::new(),
        })
    }

    /// Run the real BERT forward pass for one text and mean-pool the final
    /// hidden states into a `hidden_size` vector.
    ///
    /// Implemented as a batch of one through [`Self::embed_texts_batch`] so
    /// the single and batched paths share ONE tokenization/forward/pooling
    /// implementation and cannot drift apart.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_texts_batch(&[text])?;
        vectors.pop().ok_or_else(|| {
            MemoryError::processing_error(
                "candle embedding batch of one returned no vector (invariant violation)",
            )
        })
    }

    /// Tokenize ALL texts, pad to the batch max length, run ONE BERT forward
    /// pass over the `[batch, seq]` tensors and attention-masked mean-pool +
    /// L2-normalize each row.
    ///
    /// Correctness: padded positions carry attention-mask 0, which (a) the
    /// BERT forward turns into a large negative additive attention bias, so
    /// no real token ever attends to padding, and (b) zeroes those positions
    /// out of the mean-pool numerator and denominator. A text embedded in a
    /// padded batch is therefore identical (within float tolerance) to the
    /// same text embedded alone.
    fn embed_texts_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut encodings = Vec::with_capacity(texts.len());
        let mut max_len = 0usize;
        for text in texts {
            let encoding = self.tokenizer.encode(*text, true).map_err(|e| {
                MemoryError::processing_error(format!(
                    "candle embedding tokenization failed: {}",
                    e
                ))
            })?;
            let take = encoding.get_ids().len().min(self.max_length);
            if take == 0 {
                return Err(MemoryError::processing_error(
                    "candle embedding tokenization produced no tokens",
                ));
            }
            max_len = max_len.max(take);
            encodings.push((encoding, take));
        }

        let batch = encodings.len();
        let mut ids = vec![self.pad_token_id; batch * max_len];
        let mut type_ids = vec![0u32; batch * max_len];
        let mut mask = vec![0u32; batch * max_len];
        for (row, (encoding, take)) in encodings.iter().enumerate() {
            let offset = row * max_len;
            ids[offset..offset + take].copy_from_slice(&encoding.get_ids()[..*take]);
            type_ids[offset..offset + take].copy_from_slice(&encoding.get_type_ids()[..*take]);
            mask[offset..offset + take].copy_from_slice(&encoding.get_attention_mask()[..*take]);
        }

        let input_ids = Tensor::from_vec(ids, (batch, max_len), &self.device)?;
        let token_type_ids = Tensor::from_vec(type_ids, (batch, max_len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask, (batch, max_len), &self.device)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Sentence-transformers pooling for MiniLM: attention-masked mean of
        // the final hidden states, then L2 normalization per row.
        let mask_f = attention_mask
            .to_dtype(hidden.dtype())?
            .unsqueeze(2)?
            .broadcast_as(hidden.shape())?;
        let summed = (hidden * &mask_f)?.sum(1)?;
        let counts = mask_f.sum(1)?.clamp(1e-9, f64::INFINITY)?;
        let pooled = (summed / counts)?.to_dtype(DType::F32)?;
        let mut vectors = pooled.to_vec2::<f32>()?;
        for vector in vectors.iter_mut() {
            crate::memory::embeddings::provider::normalize_vector(vector);
        }
        Ok(vectors)
    }
}

#[async_trait]
impl EmbeddingProvider for CandleEmbeddingProvider {
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let vector = self.embed_text(text)?;
        Ok(Embedding::new(vector, self.model_id()))
    }

    /// Read-only scoring path, served from the content-hash cache when the
    /// text was already embedded (e.g. by the dense retriever before the
    /// reranker asks for the same content).
    async fn embed_for_scoring(
        &self,
        text: &str,
        _options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        let embeddings = self.embed_for_scoring_batch(&[text]).await?;
        embeddings.into_iter().next().ok_or_else(|| {
            MemoryError::processing_error(
                "candle scoring batch of one returned no embedding (invariant violation)",
            )
        })
    }

    /// TRUE batched scoring: one BERT forward pass over the whole candidate
    /// set instead of one full forward per text — this is the per-query hot
    /// path used by the dense retriever and the reranker.
    ///
    /// Content-hash cache discipline: cache hits are served directly and
    /// only the MISSES go into the single padded forward pass; computed
    /// vectors are inserted back so the reranker reuses the dense
    /// retriever's work.
    async fn embed_for_scoring_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let hashes: Vec<String> = texts.iter().map(|t| compute_content_hash(t)).collect();
        let mut vectors: Vec<Option<Vec<f32>>> = hashes
            .iter()
            .map(|hash| self.scoring_cache.get(hash).map(|v| v.clone()))
            .collect();

        let miss_indices: Vec<usize> = vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_none().then_some(i))
            .collect();

        if !miss_indices.is_empty() {
            let miss_texts: Vec<&str> = miss_indices.iter().map(|&i| texts[i]).collect();
            let computed = self.embed_texts_batch(&miss_texts)?;
            if self.scoring_cache.len() + computed.len() > SCORING_CACHE_CAP {
                self.scoring_cache.clear();
            }
            for (&i, vector) in miss_indices.iter().zip(computed) {
                self.scoring_cache.insert(hashes[i].clone(), vector.clone());
                vectors[i] = Some(vector);
            }
        }

        vectors
            .into_iter()
            .zip(hashes)
            .map(|(vector, hash)| {
                let vector = vector.ok_or_else(|| {
                    MemoryError::processing_error(
                        "candle scoring batch left a vector unfilled (invariant violation)",
                    )
                })?;
                Ok(Embedding::new(vector, self.model_id()).with_content_hash(hash))
            })
            .collect()
    }

    fn embedding_dimension(&self) -> usize {
        self.hidden_size
    }

    fn name(&self) -> &'static str {
        "CandleEmbeddingProvider"
    }

    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn is_available(&self) -> bool {
        // Construction fails closed: if this value exists, the model loaded.
        true
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_batch: true,
            max_batch_size: None,
            max_input_length: Some(self.max_length),
            supports_input_types: false,
            requires_api_key: false,
            is_local: true,
        }
    }
}
