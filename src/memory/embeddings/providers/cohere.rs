//! Cohere embedding provider
//!
//! Provides high-quality embeddings using Cohere's API,
//! supporting various embedding models optimized for different tasks.

use super::super::provider::{
    EmbeddingProvider, Embedding, EmbedOptions, ProviderCapabilities, compute_content_hash,
};
use crate::error::{MemoryError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "llm-integration")]
use reqwest::Client;

/// Cohere embedding provider
///
/// This provider uses Cohere's embedding API to generate high-quality
/// semantic embeddings. Supports multiple input types and model sizes.
pub struct CohereProvider {
    config: CohereConfig,
    #[cfg(feature = "llm-integration")]
    client: Arc<Client>,
}

/// Configuration for Cohere embeddings
#[derive(Debug, Clone)]
pub struct CohereConfig {
    /// Cohere API key
    pub api_key: String,
    /// Model to use
    pub model: CohereModel,
    /// API endpoint (allows overriding for testing)
    pub endpoint: String,
    /// Input type for embedding
    pub input_type: CohereInputType,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub retry_base_delay_ms: u64,
}

/// Supported Cohere embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CohereModel {
    /// English model (1024 dimensions)
    EmbedEnglishV3,
    /// Multilingual model (1024 dimensions)
    EmbedMultilingualV3,
    /// Lightweight English model (384 dimensions)
    EmbedEnglishLightV3,
    /// Lightweight multilingual model (384 dimensions)
    EmbedMultilingualLightV3,
}

impl CohereModel {
    /// Get the model identifier string
    pub fn as_str(&self) -> &str {
        match self {
            Self::EmbedEnglishV3 => "embed-english-v3.0",
            Self::EmbedMultilingualV3 => "embed-multilingual-v3.0",
            Self::EmbedEnglishLightV3 => "embed-english-light-v3.0",
            Self::EmbedMultilingualLightV3 => "embed-multilingual-light-v3.0",
        }
    }

    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            Self::EmbedEnglishV3 | Self::EmbedMultilingualV3 => 1024,
            Self::EmbedEnglishLightV3 | Self::EmbedMultilingualLightV3 => 384,
        }
    }
}

/// Input type for Cohere embeddings
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereInputType {
    /// For search queries
    SearchQuery,
    /// For documents to be searched
    SearchDocument,
    /// For classification
    Classification,
    /// For clustering
    Clustering,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: CohereModel::EmbedEnglishLightV3,
            endpoint: "https://api.cohere.ai/v1/embed".to_string(),
            input_type: CohereInputType::SearchDocument,
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_delay_ms: 1000,
        }
    }
}

impl CohereConfig {
    /// Create a new configuration with an API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: CohereModel) -> Self {
        self.model = model;
        self
    }

    /// Set the input type
    pub fn with_input_type(mut self, input_type: CohereInputType) -> Self {
        self.input_type = input_type;
        self
    }

    /// Set a custom endpoint
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Load API key from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| MemoryError::Configuration("COHERE_API_KEY not set".to_string()))?;
        Ok(Self::new(api_key))
    }
}

/// Cohere API request format
#[derive(Debug, Serialize)]
struct CohereRequest {
    texts: Vec<String>,
    model: String,
    input_type: CohereInputType,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

/// Cohere API response format
#[derive(Debug, Deserialize)]
struct CohereResponse {
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    meta: Option<ResponseMeta>,
}

#[derive(Debug, Deserialize)]
struct ResponseMeta {
    #[serde(default)]
    billed_units: Option<BilledUnits>,
}

#[derive(Debug, Deserialize)]
struct BilledUnits {
    #[serde(default)]
    input_tokens: Option<usize>,
}

impl CohereProvider {
    /// Create a new Cohere provider
    #[cfg(feature = "llm-integration")]
    pub fn new(config: CohereConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(MemoryError::Configuration(
                "Cohere API key is required".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| MemoryError::External(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client: Arc::new(client),
        })
    }

    #[cfg(not(feature = "llm-integration"))]
    pub fn new(_config: CohereConfig) -> Result<Self> {
        Err(MemoryError::Configuration(
            "Cohere provider requires 'llm-integration' feature".to_string(),
        ))
    }

    /// Create provider from environment variables
    pub fn from_env() -> Result<Self> {
        let config = CohereConfig::from_env()?;
        Self::new(config)
    }

    /// Make API request with retry logic
    #[cfg(feature = "llm-integration")]
    async fn make_request(&self, texts: Vec<String>) -> Result<CohereResponse> {
        let request = CohereRequest {
            texts,
            model: self.config.model.as_str().to_string(),
            input_type: self.config.input_type,
            truncate: Some("END".to_string()),
        };

        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff
                let delay = Duration::from_millis(
                    self.config.retry_base_delay_ms * 2_u64.pow(attempt - 1),
                );
                tokio::time::sleep(delay).await;
                tracing::debug!("Retrying Cohere request (attempt {})", attempt + 1);
            }

            let response = self
                .client
                .post(&self.config.endpoint)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.json::<CohereResponse>().await {
                            Ok(data) => {
                                tracing::debug!("Cohere embedding request successful");
                                if let Some(meta) = &data.meta {
                                    if let Some(billed) = &meta.billed_units {
                                        if let Some(tokens) = billed.input_tokens {
                                            tracing::debug!("Tokens used: {}", tokens);
                                        }
                                    }
                                }
                                return Ok(data);
                            }
                            Err(e) => {
                                last_error = Some(MemoryError::External(format!(
                                    "Failed to parse Cohere response: {}",
                                    e
                                )));
                            }
                        }
                    } else {
                        let status = resp.status();
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());

                        // Don't retry 4xx errors (except 429 rate limit)
                        if status.is_client_error() && status.as_u16() != 429 {
                            return Err(MemoryError::External(format!(
                                "Cohere API error {}: {}",
                                status, error_text
                            )));
                        }

                        last_error = Some(MemoryError::External(format!(
                            "Cohere API error {}: {}",
                            status, error_text
                        )));
                    }
                }
                Err(e) => {
                    last_error = Some(MemoryError::External(format!(
                        "Failed to send request to Cohere: {}",
                        e
                    )));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MemoryError::External("Cohere request failed after all retries".to_string())
        }))
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    #[cfg(feature = "llm-integration")]
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);

        // Make API request
        let response = self.make_request(vec![text.to_string()]).await?;

        if response.embeddings.is_empty() {
            return Err(MemoryError::External(
                "Cohere returned no embeddings".to_string(),
            ));
        }

        let vector = response.embeddings[0].clone();

        // Extract token count from metadata
        let token_count = response
            .meta
            .as_ref()
            .and_then(|m| m.billed_units.as_ref())
            .and_then(|b| b.input_tokens);

        // Create embedding with metadata
        let mut embedding = Embedding::new(vector, self.model_id())
            .with_content_hash(content_hash)
            .with_version("3.0".to_string());

        if let Some(tokens) = token_count {
            embedding = embedding.with_token_count(tokens);
        }

        Ok(embedding)
    }

    #[cfg(not(feature = "llm-integration"))]
    async fn embed(&self, _text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        Err(MemoryError::Configuration(
            "Cohere provider requires 'llm-integration' feature".to_string(),
        ))
    }

    #[cfg(feature = "llm-integration")]
    async fn embed_batch(
        &self,
        texts: &[String],
        _options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Cohere supports batch embedding (up to 96 inputs)
        const MAX_BATCH_SIZE: usize = 96;

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in chunks
        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let response = self.make_request(chunk.to_vec()).await?;

            if response.embeddings.len() != chunk.len() {
                return Err(MemoryError::External(format!(
                    "Cohere returned {} embeddings for {} texts",
                    response.embeddings.len(),
                    chunk.len()
                )));
            }

            // Extract token count
            let token_count = response
                .meta
                .as_ref()
                .and_then(|m| m.billed_units.as_ref())
                .and_then(|b| b.input_tokens);

            for (i, vector) in response.embeddings.into_iter().enumerate() {
                let text = &chunk[i];
                let content_hash = compute_content_hash(text);

                let mut embedding = Embedding::new(vector, self.model_id())
                    .with_content_hash(content_hash)
                    .with_version("3.0".to_string());

                if let Some(tokens) = token_count {
                    embedding = embedding.with_token_count(tokens / chunk.len());
                }

                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }

    #[cfg(not(feature = "llm-integration"))]
    async fn embed_batch(
        &self,
        _texts: &[String],
        _options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        Err(MemoryError::Configuration(
            "Cohere provider requires 'llm-integration' feature".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        self.config.model.dimension()
    }

    fn name(&self) -> &'static str {
        "CohereProvider"
    }

    fn model_id(&self) -> String {
        self.config.model.as_str().to_string()
    }

    #[cfg(feature = "llm-integration")]
    fn is_available(&self) -> bool {
        !self.config.api_key.is_empty()
    }

    #[cfg(not(feature = "llm-integration"))]
    fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_batch: true,
            max_batch_size: Some(96),
            max_input_length: Some(512), // tokens
            supports_input_types: true,
            requires_api_key: true,
            is_local: false,
        }
    }
}

#[cfg(all(test, feature = "llm-integration"))]
mod tests {
    use super::*;

    fn get_test_api_key() -> Option<String> {
        std::env::var("COHERE_API_KEY").ok()
    }

    #[test]
    fn test_cohere_config() {
        let config = CohereConfig::new("test-key".to_string())
            .with_model(CohereModel::EmbedEnglishV3)
            .with_input_type(CohereInputType::SearchQuery)
            .with_timeout(60);

        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model.as_str(), "embed-english-v3.0");
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(CohereModel::EmbedEnglishV3.dimension(), 1024);
        assert_eq!(CohereModel::EmbedMultilingualV3.dimension(), 1024);
        assert_eq!(CohereModel::EmbedEnglishLightV3.dimension(), 384);
        assert_eq!(CohereModel::EmbedMultilingualLightV3.dimension(), 384);
    }

    #[tokio::test]
    async fn test_cohere_provider_creation() {
        let config = CohereConfig::new("test-key".to_string());
        let provider = CohereProvider::new(config);
        assert!(provider.is_ok());

        let provider = provider.unwrap();
        assert_eq!(provider.name(), "CohereProvider");
        assert_eq!(provider.embedding_dimension(), 384);
    }

    #[tokio::test]
    async fn test_cohere_provider_no_key() {
        let config = CohereConfig::default();
        let provider = CohereProvider::new(config);
        assert!(provider.is_err());
    }

    // Integration tests (require real API key)
    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_cohere_embedding_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: COHERE_API_KEY not set");
            return;
        };

        let config = CohereConfig::new(api_key)
            .with_model(CohereModel::EmbedEnglishLightV3)
            .with_input_type(CohereInputType::SearchDocument);
        let provider = CohereProvider::new(config).unwrap();

        let text = "This is a test of Cohere embeddings";
        let embedding = provider.embed(text, None).await;

        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.dimension(), 384);
        assert_eq!(embedding.model, "embed-english-light-v3.0");
        assert!(!embedding.content_hash.is_empty());
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_cohere_batch_embedding_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: COHERE_API_KEY not set");
            return;
        };

        let config = CohereConfig::new(api_key)
            .with_model(CohereModel::EmbedEnglishLightV3);
        let provider = CohereProvider::new(config).unwrap();

        let texts = vec![
            "First test text".to_string(),
            "Second test text".to_string(),
            "Third test text".to_string(),
        ];

        let embeddings = provider.embed_batch(&texts, None).await;

        assert!(embeddings.is_ok());
        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 3);

        for embedding in embeddings {
            assert_eq!(embedding.dimension(), 384);
            assert_eq!(embedding.model, "embed-english-light-v3.0");
        }
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_cohere_similarity_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: COHERE_API_KEY not set");
            return;
        };

        let config = CohereConfig::new(api_key)
            .with_model(CohereModel::EmbedEnglishLightV3)
            .with_input_type(CohereInputType::SearchDocument);
        let provider = CohereProvider::new(config).unwrap();

        let emb1 = provider.embed("machine learning and artificial intelligence", None).await.unwrap();
        let emb2 = provider.embed("deep learning and neural networks", None).await.unwrap();
        let emb3 = provider.embed("cooking Italian pasta recipes", None).await.unwrap();

        let sim_related = emb1.cosine_similarity(&emb2);
        let sim_unrelated = emb1.cosine_similarity(&emb3);

        // Related concepts should have higher similarity
        assert!(sim_related > sim_unrelated);
        println!("Related similarity: {}, Unrelated similarity: {}", sim_related, sim_unrelated);
    }
}
