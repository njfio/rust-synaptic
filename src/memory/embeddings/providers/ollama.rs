//! Ollama embedding provider
//!
//! Provides local embeddings using Ollama models, enabling
//! fully private and self-hosted embedding generation.

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

/// Ollama embedding provider
///
/// This provider uses locally running Ollama models to generate embeddings.
/// No API key required, fully private, supports various model sizes.
pub struct OllamaProvider {
    config: OllamaConfig,
    #[cfg(feature = "llm-integration")]
    client: Arc<Client>,
}

/// Configuration for Ollama embeddings
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Ollama server endpoint
    pub endpoint: String,
    /// Model to use (e.g., "nomic-embed-text", "all-minilm", "mxbai-embed-large")
    pub model: String,
    /// Expected embedding dimension (model-specific)
    pub embedding_dim: usize,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            embedding_dim: 768, // nomic-embed-text default
            timeout_seconds: 30,
            max_retries: 2,
        }
    }
}

impl OllamaConfig {
    /// Create a new configuration with a model
    pub fn new(model: String, embedding_dim: usize) -> Self {
        Self {
            model,
            embedding_dim,
            ..Default::default()
        }
    }

    /// Set the server endpoint
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let endpoint = std::env::var("OLLAMA_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());
        let embedding_dim = std::env::var("OLLAMA_EMBEDDING_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(768);

        Self {
            endpoint,
            model,
            embedding_dim,
            timeout_seconds: 30,
            max_retries: 2,
        }
    }

    /// Configuration for nomic-embed-text model (768d, optimized for search)
    pub fn nomic_embed_text() -> Self {
        Self::new("nomic-embed-text".to_string(), 768)
    }

    /// Configuration for all-minilm model (384d, fast and lightweight)
    pub fn all_minilm() -> Self {
        Self::new("all-minilm".to_string(), 384)
    }

    /// Configuration for mxbai-embed-large model (1024d, high quality)
    pub fn mxbai_embed_large() -> Self {
        Self::new("mxbai-embed-large".to_string(), 1024)
    }
}

/// Ollama API request format
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
}

/// Ollama API response format
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    embedding: Vec<f32>,
}

impl OllamaProvider {
    /// Create a new Ollama provider
    #[cfg(feature = "llm-integration")]
    pub fn new(config: OllamaConfig) -> Result<Self> {
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
    pub fn new(_config: OllamaConfig) -> Result<Self> {
        Err(MemoryError::Configuration(
            "Ollama provider requires 'llm-integration' feature".to_string(),
        ))
    }

    /// Create provider from environment variables
    pub fn from_env() -> Result<Self> {
        let config = OllamaConfig::from_env();
        Self::new(config)
    }

    /// Create provider with default nomic-embed-text model
    pub fn default() -> Result<Self> {
        Self::new(OllamaConfig::default())
    }

    /// Check if Ollama server is available
    #[cfg(feature = "llm-integration")]
    pub async fn check_availability(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.config.endpoint);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    #[cfg(not(feature = "llm-integration"))]
    pub async fn check_availability(&self) -> Result<bool> {
        Ok(false)
    }

    /// Make API request with retry logic
    #[cfg(feature = "llm-integration")]
    async fn make_request(&self, text: &str) -> Result<OllamaResponse> {
        let request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", self.config.endpoint);

        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Simple fixed delay for retries
                tokio::time::sleep(Duration::from_millis(500)).await;
                tracing::debug!("Retrying Ollama request (attempt {})", attempt + 1);
            }

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        match resp.json::<OllamaResponse>().await {
                            Ok(data) => {
                                tracing::debug!("Ollama embedding request successful");
                                return Ok(data);
                            }
                            Err(e) => {
                                last_error = Some(MemoryError::External(format!(
                                    "Failed to parse Ollama response: {}",
                                    e
                                )));
                            }
                        }
                    } else {
                        let status = resp.status();
                        let error_text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                        last_error = Some(MemoryError::External(format!(
                            "Ollama API error {}: {}",
                            status, error_text
                        )));
                    }
                }
                Err(e) => {
                    last_error = Some(MemoryError::External(format!(
                        "Failed to connect to Ollama at {}: {}. Make sure Ollama is running.",
                        self.config.endpoint, e
                    )));
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MemoryError::External("Ollama request failed after all retries".to_string())
        }))
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    #[cfg(feature = "llm-integration")]
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);

        // Make API request
        let response = self.make_request(text).await?;

        if response.embedding.len() != self.config.embedding_dim {
            tracing::warn!(
                "Expected embedding dimension {}, got {}",
                self.config.embedding_dim,
                response.embedding.len()
            );
        }

        // Create embedding with metadata
        let embedding = Embedding::new(response.embedding, self.model_id())
            .with_content_hash(content_hash)
            .with_version("1".to_string());

        Ok(embedding)
    }

    #[cfg(not(feature = "llm-integration"))]
    async fn embed(&self, _text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        Err(MemoryError::Configuration(
            "Ollama provider requires 'llm-integration' feature".to_string(),
        ))
    }

    #[cfg(feature = "llm-integration")]
    async fn embed_batch(
        &self,
        texts: &[String],
        options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        // Ollama doesn't support native batch embedding, so we process sequentially
        // Could be parallelized but would increase load on local server
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            embeddings.push(self.embed(text, options).await?);
        }

        Ok(embeddings)
    }

    #[cfg(not(feature = "llm-integration"))]
    async fn embed_batch(
        &self,
        _texts: &[String],
        _options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        Err(MemoryError::Configuration(
            "Ollama provider requires 'llm-integration' feature".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dim
    }

    fn name(&self) -> &'static str {
        "OllamaProvider"
    }

    fn model_id(&self) -> String {
        format!("ollama:{}", self.config.model)
    }

    fn is_available(&self) -> bool {
        // Cannot check async availability in sync method
        // User should call check_availability() explicitly
        true
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_batch: true,
            max_batch_size: None, // No hard limit, but sequential processing
            max_input_length: None, // Model-dependent
            supports_input_types: false,
            requires_api_key: false,
            is_local: true,
        }
    }
}

#[cfg(all(test, feature = "llm-integration"))]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_config() {
        let config = OllamaConfig::new("test-model".to_string(), 512)
            .with_endpoint("http://localhost:11434".to_string())
            .with_timeout(60);

        assert_eq!(config.model, "test-model");
        assert_eq!(config.embedding_dim, 512);
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_predefined_configs() {
        let nomic = OllamaConfig::nomic_embed_text();
        assert_eq!(nomic.model, "nomic-embed-text");
        assert_eq!(nomic.embedding_dim, 768);

        let minilm = OllamaConfig::all_minilm();
        assert_eq!(minilm.model, "all-minilm");
        assert_eq!(minilm.embedding_dim, 384);

        let mxbai = OllamaConfig::mxbai_embed_large();
        assert_eq!(mxbai.model, "mxbai-embed-large");
        assert_eq!(mxbai.embedding_dim, 1024);
    }

    #[tokio::test]
    async fn test_ollama_provider_creation() {
        let config = OllamaConfig::default();
        let provider = OllamaProvider::new(config);
        assert!(provider.is_ok());

        let provider = provider.unwrap();
        assert_eq!(provider.name(), "OllamaProvider");
        assert_eq!(provider.embedding_dimension(), 768);
        assert!(provider.is_available());
    }

    // Integration tests (require Ollama running locally)
    #[tokio::test]
    #[ignore] // Run with --ignored flag when Ollama is running
    async fn test_ollama_availability() {
        let provider = OllamaProvider::default().unwrap();
        let available = provider.check_availability().await;
        assert!(available.is_ok());

        if available.unwrap() {
            println!("Ollama server is available");
        } else {
            println!("Ollama server is not running");
        }
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when Ollama is running
    async fn test_ollama_embedding_integration() {
        let provider = OllamaProvider::default().unwrap();

        // Check if server is available
        if !provider.check_availability().await.unwrap_or(false) {
            println!("Skipping integration test: Ollama server not available");
            return;
        }

        let text = "This is a test of Ollama embeddings";
        let embedding = provider.embed(text, None).await;

        assert!(embedding.is_ok());
        let embedding = embedding.unwrap();
        assert_eq!(embedding.dimension(), 768);
        assert!(embedding.model.starts_with("ollama:"));
        assert!(!embedding.content_hash.is_empty());
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when Ollama is running
    async fn test_ollama_batch_embedding_integration() {
        let provider = OllamaProvider::default().unwrap();

        if !provider.check_availability().await.unwrap_or(false) {
            println!("Skipping integration test: Ollama server not available");
            return;
        }

        let texts = vec![
            "First test text".to_string(),
            "Second test text".to_string(),
        ];

        let embeddings = provider.embed_batch(&texts, None).await;

        assert!(embeddings.is_ok());
        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 2);

        for embedding in embeddings {
            assert_eq!(embedding.dimension(), 768);
        }
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag when Ollama is running
    async fn test_ollama_similarity_integration() {
        let provider = OllamaProvider::default().unwrap();

        if !provider.check_availability().await.unwrap_or(false) {
            println!("Skipping integration test: Ollama server not available");
            return;
        }

        let emb1 = provider.embed("machine learning and AI", None).await.unwrap();
        let emb2 = provider.embed("deep learning and neural networks", None).await.unwrap();
        let emb3 = provider.embed("cooking pasta recipes", None).await.unwrap();

        let sim_related = emb1.cosine_similarity(&emb2);
        let sim_unrelated = emb1.cosine_similarity(&emb3);

        // Related concepts should have higher similarity
        assert!(sim_related > sim_unrelated);
        println!("Related similarity: {}, Unrelated similarity: {}", sim_related, sim_unrelated);
    }
}
