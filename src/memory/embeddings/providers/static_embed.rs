//! model2vec static-embedding provider (feature `static-embeddings`)
//!
//! Loads a model2vec static embedding TABLE (e.g. `minishlab/potion-base-8M`,
//! ~30 MB, 256-dim) and embeds text with NO transformer forward pass:
//! tokenize → look up each token id's row in the matrix → masked mean-pool →
//! L2-normalize. That is O(tokens) row lookups — microseconds on any CPU —
//! while the vectors stay genuinely semantic because the table was distilled
//! from a real sentence-transformer.
//!
//! Runtime notes verified against the actual potion-base-8M artifacts:
//! - `model.safetensors` holds ONE tensor named `embeddings`, shape
//!   `[vocab_size, dim]` (`[29528, 256]`), dtype F32.
//! - `config.json` is model2vec-shaped (`model_type: "model2vec"`,
//!   `hidden_dim: 256`, `normalize: true`); we honor `normalize` and take the
//!   dimension from the matrix itself.
//! - model2vec encodes WITHOUT special tokens (its Python `StaticModel`
//!   passes `add_special_tokens=False`), so we do the same; every token id
//!   the tokenizer emits has a row in the table.
//!
//! Construction fails closed: a missing/unreadable model directory is an
//! error — no placeholder provider is ever constructed. Wire it through
//! [`build_retrieval_provider`](crate::memory::embeddings::build_retrieval_provider)
//! to get the TF-IDF fallback behavior.

use crate::error::{MemoryError, Result};
use crate::memory::embeddings::provider::{
    compute_content_hash, normalize_vector, EmbedOptions, Embedding, EmbeddingProvider,
    ProviderCapabilities,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

/// The tensor name model2vec uses for the static embedding matrix.
const EMBEDDINGS_TENSOR: &str = "embeddings";

/// Subset of the model2vec `config.json` we care about.
#[derive(Debug, Deserialize)]
struct StaticModelConfig {
    /// L2-normalize the pooled output (potion sets `true`).
    #[serde(default = "default_normalize")]
    normalize: bool,
    /// Expected dimension; cross-checked against the matrix when present.
    #[serde(default)]
    hidden_dim: Option<usize>,
}

fn default_normalize() -> bool {
    true
}

/// Embedding provider backed by a model2vec static embedding table.
///
/// `embed` = tokenize (no special tokens) → per-token row lookup in the
/// `[vocab, dim]` matrix → mean-pool → L2-normalize (per model config).
/// `embed_for_scoring` delegates to the same computation (static tables have
/// no corpus-relative state); the batch path is a loop — each text is already
/// microseconds.
pub struct StaticEmbeddingProvider {
    tokenizer: Tokenizer,
    /// Row-major `[vocab_size * dimension]` embedding table.
    table: Vec<f32>,
    vocab_size: usize,
    dimension: usize,
    normalize: bool,
    model_id: String,
}

impl StaticEmbeddingProvider {
    /// Load a model2vec static model from a local directory containing
    /// `config.json`, `tokenizer.json`, and `model.safetensors`.
    ///
    /// Fails closed: any missing or malformed file is an error.
    pub fn new(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding tokenizer not loadable at {}: {}",
                tokenizer_path.display(),
                e
            ))
        })?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding config not readable at {}: {}",
                config_path.display(),
                e
            ))
        })?;
        let config: StaticModelConfig = serde_json::from_str(&config_str).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding config not parseable at {}: {}",
                config_path.display(),
                e
            ))
        })?;

        let weights_path = model_dir.join("model.safetensors");
        let weights = std::fs::read(&weights_path).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding weights not readable at {}: {}",
                weights_path.display(),
                e
            ))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&weights).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding safetensors not parseable at {}: {}",
                weights_path.display(),
                e
            ))
        })?;
        let view = tensors.tensor(EMBEDDINGS_TENSOR).map_err(|e| {
            MemoryError::configuration(format!(
                "static embedding tensor '{}' missing in {} (tensors present: {:?}): {}",
                EMBEDDINGS_TENSOR,
                weights_path.display(),
                tensors.names(),
                e
            ))
        })?;
        if view.dtype() != safetensors::Dtype::F32 {
            return Err(MemoryError::configuration(format!(
                "static embedding tensor '{}' has dtype {:?}, expected F32",
                EMBEDDINGS_TENSOR,
                view.dtype()
            )));
        }
        let shape = view.shape();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(MemoryError::configuration(format!(
                "static embedding tensor '{}' has shape {:?}, expected [vocab, dim]",
                EMBEDDINGS_TENSOR, shape
            )));
        }
        let (vocab_size, dimension) = (shape[0], shape[1]);
        if let Some(expected) = config.hidden_dim {
            if expected != dimension {
                tracing::warn!(
                    configured = expected,
                    actual = dimension,
                    "model2vec config hidden_dim differs from matrix dimension; using the matrix"
                );
            }
        }

        // safetensors data is little-endian F32; decode into a native table.
        let data = view.data();
        if data.len() != vocab_size * dimension * 4 {
            return Err(MemoryError::configuration(format!(
                "static embedding tensor byte length {} does not match shape [{}, {}]",
                data.len(),
                vocab_size,
                dimension
            )));
        }
        let table: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self {
            tokenizer,
            table,
            vocab_size,
            dimension,
            normalize: config.normalize,
            model_id: format!("model2vec-static({})", model_dir.display()),
        })
    }

    /// Tokenize (no special tokens, matching model2vec) and mean-pool the
    /// token rows. Empty/unknown-only input yields the zero vector rather
    /// than an error — deterministic and comparable (cosine 0 to everything).
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| {
            MemoryError::processing_error(format!("static embedding tokenization failed: {}", e))
        })?;
        let mut pooled = vec![0.0f32; self.dimension];
        let mut count = 0usize;
        for (&id, &mask) in encoding.get_ids().iter().zip(encoding.get_attention_mask()) {
            if mask == 0 {
                continue;
            }
            let row = id as usize;
            if row >= self.vocab_size {
                // Token id outside the distilled table: skip it (masked out of
                // the mean) rather than reading out of bounds.
                continue;
            }
            let offset = row * self.dimension;
            for (acc, &v) in pooled
                .iter_mut()
                .zip(&self.table[offset..offset + self.dimension])
            {
                *acc += v;
            }
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            for v in pooled.iter_mut() {
                *v *= inv;
            }
        }
        if self.normalize {
            normalize_vector(&mut pooled);
        }
        Ok(pooled)
    }
}

#[async_trait]
impl EmbeddingProvider for StaticEmbeddingProvider {
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let vector = self.embed_text(text)?;
        Ok(Embedding::new(vector, self.model_id()))
    }

    /// Static tables carry no corpus-relative state: scoring == embed.
    async fn embed_for_scoring(
        &self,
        text: &str,
        options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        self.embed(text, options).await
    }

    /// Batch scoring is a plain loop: each lookup+pool is already
    /// microseconds, so there is nothing to batch across texts.
    async fn embed_for_scoring_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts
            .iter()
            .map(|text| {
                let vector = self.embed_text(text)?;
                Ok(Embedding::new(vector, self.model_id())
                    .with_content_hash(compute_content_hash(text)))
            })
            .collect()
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &'static str {
        "StaticEmbeddingProvider"
    }

    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn is_available(&self) -> bool {
        // Construction fails closed: if this value exists, the table loaded.
        true
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_batch: true,
            max_batch_size: None,
            max_input_length: None,
            supports_input_types: false,
            requires_api_key: false,
            is_local: true,
        }
    }
}
