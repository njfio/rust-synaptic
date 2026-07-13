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
    EmbedOptions, Embedding, EmbeddingProvider, ProviderCapabilities,
};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use tokenizers::Tokenizer;

/// Embedding provider backed by a local candle BERT model.
///
/// Holds the loaded model and tokenizer; `embed` runs a real forward pass
/// and mean-pools the final hidden states into a fixed-dimension vector
/// (`hidden_size`, 768 for bert-base).
pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    hidden_size: usize,
    max_length: usize,
    model_id: String,
}

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
            model_id: "candle-bert(in-memory)".to_string(),
        })
    }

    /// Run the real BERT forward pass for one text and mean-pool the final
    /// hidden states into a `hidden_size` vector.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            MemoryError::processing_error(format!("candle embedding tokenization failed: {}", e))
        })?;

        let take = encoding.get_ids().len().min(self.max_length);
        if take == 0 {
            return Err(MemoryError::processing_error(
                "candle embedding tokenization produced no tokens",
            ));
        }
        let ids = &encoding.get_ids()[..take];
        let type_ids = &encoding.get_type_ids()[..take];
        let mask = &encoding.get_attention_mask()[..take];

        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let (_, n_tokens, _) = hidden.dims3()?;
        let pooled = (hidden.sum(1)? / (n_tokens as f64))?
            .to_dtype(DType::F32)?
            .squeeze(0)?;
        Ok(pooled.to_vec1::<f32>()?)
    }
}

#[async_trait]
impl EmbeddingProvider for CandleEmbeddingProvider {
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let vector = self.embed_text(text)?;
        Ok(Embedding::new(vector, self.model_id()))
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
