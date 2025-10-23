//! OpenAI embedding provider
//!
//! Provides production-ready embeddings using OpenAI's API,
//! supporting text-embedding-ada-002 and text-embedding-3-* models.

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

/// OpenAI embedding provider
///
/// This provider uses OpenAI's embedding API to generate high-quality
/// semantic embeddings for text. Requires an OpenAI API key.
pub struct OpenAIProvider {
    config: OpenAIConfig,
    #[cfg(feature = "llm-integration")]
    client: Arc<Client>,
}

/// Configuration for OpenAI embeddings
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// OpenAI API key
    pub api_key: String,
    /// Model to use (e.g., "text-embedding-ada-002", "text-embedding-3-small")
    pub model: OpenAIModel,
    /// API endpoint (allows overriding for Azure or compatible APIs)
    pub endpoint: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Base delay for exponential backoff (milliseconds)
    pub retry_base_delay_ms: u64,
}

/// Supported OpenAI embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAIModel {
    /// Legacy ada-002 model (1536 dimensions)
    TextEmbeddingAda002,
    /// New small model (512 dimensions, cost-efficient)
    TextEmbedding3Small,
    /// New large model (3072 dimensions, highest quality)
    TextEmbedding3Large,
}

impl OpenAIModel {
    /// Get the model identifier string
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
        }
    }

    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        match self {
            Self::TextEmbeddingAda002 => 1536,
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
        }
    }
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: OpenAIModel::TextEmbedding3Small,
            endpoint: "https://api.openai.com/v1/embeddings".to_string(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_base_delay_ms: 1000,
        }
    }
}

impl OpenAIConfig {
    /// Create a new configuration with an API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Set the model to use
    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.model = model;
        self
    }

    /// Set a custom endpoint (for Azure OpenAI or compatible APIs)
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
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| MemoryError::Configuration("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(api_key))
    }
}

/// OpenAI API request format
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

/// OpenAI API response format
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    data: Vec<EmbeddingData>,
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    prompt_tokens: usize,
    total_tokens: usize,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    #[cfg(feature = "llm-integration")]
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(MemoryError::Configuration(
                "OpenAI API key is required".to_string(),
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
    pub fn new(_config: OpenAIConfig) -> Result<Self> {
        Err(MemoryError::Configuration(
            "OpenAI provider requires 'llm-integration' feature".to_string(),
        ))
    }

    /// Create provider from environment variables
    pub fn from_env() -> Result<Self> {
        let config = OpenAIConfig::from_env()?;
        Self::new(config)
    }

    /// Make API request with retry logic
    #[cfg(feature = "llm-integration")]
    async fn make_request(&self, texts: Vec<String>) -> Result<OpenAIResponse> {
        let request = OpenAIRequest {
            input: texts,
            model: self.config.model.as_str().to_string(),
            encoding_format: Some("float".to_string()),
        };

        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff
                let delay = Duration::from_millis(
                    self.config.retry_base_delay_ms * 2_u64.pow(attempt - 1),
                );
                tokio::time::sleep(delay).await;
                tracing::debug!("Retrying OpenAI request (attempt {})", attempt + 1);
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
                        match resp.json::<OpenAIResponse>().await {
                            Ok(data) => {
                                tracing::debug!(
                                    "OpenAI embedding request successful. Tokens used: {}",
                                    data.usage.total_tokens
                                );
                                return Ok(data);
                            }
                            Err(e) => {
                                last_error = Some(MemoryError::External(format!(
                                    "Failed to parse OpenAI response: {}",
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
                                "OpenAI API error {}: {}",
                                status, error_text
                            )));
                        }

                        last_error = Some(MemoryError::External(format!(
                            "OpenAI API error {}: {}",
                            status, error_text
                        )));
                    }
                }
                Err(e) => {
                    last_error = Some(MemoryError::External(format!(
                        "Failed to send request to OpenAI: {}",
                        e
                    )));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MemoryError::External("OpenAI request failed after all retries".to_string())
        }))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    #[cfg(feature = "llm-integration")]
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);

        // Make API request
        let response = self.make_request(vec![text.to_string()]).await?;

        if response.data.is_empty() {
            return Err(MemoryError::External(
                "OpenAI returned no embeddings".to_string(),
            ));
        }

        let vector = response.data[0].embedding.clone();

        // Create embedding with metadata
        let embedding = Embedding::new(vector, self.model_id())
            .with_content_hash(content_hash)
            .with_version("1".to_string())
            .with_token_count(response.usage.prompt_tokens);

        Ok(embedding)
    }

    #[cfg(not(feature = "llm-integration"))]
    async fn embed(&self, _text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        Err(MemoryError::Configuration(
            "OpenAI provider requires 'llm-integration' feature".to_string(),
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

        // OpenAI supports batch embedding (up to 2048 inputs)
        const MAX_BATCH_SIZE: usize = 100; // Conservative limit to avoid rate limits

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in chunks
        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let response = self.make_request(chunk.to_vec()).await?;

            if response.data.len() != chunk.len() {
                return Err(MemoryError::External(format!(
                    "OpenAI returned {} embeddings for {} texts",
                    response.data.len(),
                    chunk.len()
                )));
            }

            // Sort by index to ensure correct order
            let mut sorted_data = response.data;
            sorted_data.sort_by_key(|d| d.index);

            for (i, data) in sorted_data.into_iter().enumerate() {
                let text = &chunk[i];
                let content_hash = compute_content_hash(text);

                let embedding = Embedding::new(data.embedding, self.model_id())
                    .with_content_hash(content_hash)
                    .with_version("1".to_string())
                    .with_token_count(response.usage.prompt_tokens / chunk.len());

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
            "OpenAI provider requires 'llm-integration' feature".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        self.config.model.dimension()
    }

    fn name(&self) -> &'static str {
        "OpenAIProvider"
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
            max_batch_size: Some(100),
            max_input_length: Some(8191), // tokens
            supports_input_types: false,
            requires_api_key: true,
            is_local: false,
        }
    }
}

#[cfg(all(test, feature = "llm-integration"))]
mod tests {
    use super::*;

    fn get_test_api_key() -> Option<String> {
        std::env::var("OPENAI_API_KEY").ok()
    }

    #[test]
    fn test_openai_config() {
        let config = OpenAIConfig::new("test-key".to_string())
            .with_model(OpenAIModel::TextEmbedding3Small)
            .with_timeout(60);

        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model.as_str(), "text-embedding-3-small");
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(OpenAIModel::TextEmbeddingAda002.dimension(), 1536);
        assert_eq!(OpenAIModel::TextEmbedding3Small.dimension(), 1536);
        assert_eq!(OpenAIModel::TextEmbedding3Large.dimension(), 3072);
    }

    #[tokio::test]
    async fn test_openai_provider_creation() {
        let config = OpenAIConfig::new("test-key".to_string());
        let provider = OpenAIProvider::new(config);
        assert!(provider.is_ok());

        let provider = provider.unwrap();
        assert_eq!(provider.name(), "OpenAIProvider");
        assert_eq!(provider.embedding_dimension(), 1536);
    }

    #[tokio::test]
    async fn test_openai_provider_no_key() {
        let config = OpenAIConfig::default();
        let provider = OpenAIProvider::new(config);
        assert!(provider.is_err());
    }

    // Integration tests (require real API key)
    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_openai_embedding_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: OPENAI_API_KEY not set");
            return;
        };

        let config = OpenAIConfig::new(api_key)
            .with_model(OpenAIModel::TextEmbedding3Small);
        let provider = OpenAIProvider::new(config).unwrap();

        let text = "This is a test of OpenAI embeddings";
        let embedding = provider.embed(text, None).await;

        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.dimension(), 1536);
        assert_eq!(embedding.model, "text-embedding-3-small");
        assert!(!embedding.content_hash.is_empty());
        assert!(embedding.token_count.is_some());
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_openai_batch_embedding_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: OPENAI_API_KEY not set");
            return;
        };

        let config = OpenAIConfig::new(api_key)
            .with_model(OpenAIModel::TextEmbedding3Small);
        let provider = OpenAIProvider::new(config).unwrap();

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
            assert_eq!(embedding.dimension(), 1536);
            assert_eq!(embedding.model, "text-embedding-3-small");
        }
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when testing with real API
    async fn test_openai_similarity_integration() {
        let Some(api_key) = get_test_api_key() else {
            println!("Skipping integration test: OPENAI_API_KEY not set");
            return;
        };

        let config = OpenAIConfig::new(api_key)
            .with_model(OpenAIModel::TextEmbedding3Small);
        let provider = OpenAIProvider::new(config).unwrap();

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
