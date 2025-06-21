//! Real OpenAI Embeddings Implementation
//! 
//! This module provides state-of-the-art embeddings using OpenAI's text-embedding-3-small
//! and text-embedding-3-large models for semantic similarity and search.

use crate::error::{Result, MemoryError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


#[cfg(feature = "openai-embeddings")]
use reqwest::Client;

/// Configuration for OpenAI embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingConfig {
    /// OpenAI API key
    pub api_key: String,
    /// Model to use for embeddings
    pub model: String,
    /// Dimension of embeddings (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)
    pub embedding_dim: usize,
    /// API base URL
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable caching of embeddings
    pub enable_cache: bool,
    /// Maximum cache size
    pub cache_size: usize,
}

impl Default for OpenAIEmbeddingConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "text-embedding-3-large".to_string(), // Updated to use the larger, better model
            embedding_dim: 3072, // text-embedding-3-large dimension (can be reduced to 1536 if needed)
            base_url: "https://api.openai.com/v1/embeddings".to_string(),
            timeout_secs: 30,
            enable_cache: true,
            cache_size: 10000,
        }
    }
}

/// OpenAI embeddings client
#[derive(Debug)]
pub struct OpenAIEmbedder {
    config: OpenAIEmbeddingConfig,
    #[cfg(feature = "openai-embeddings")]
    client: Client,
    embedding_cache: HashMap<String, Vec<f32>>,
    metrics: EmbeddingMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddingMetrics {
    pub embeddings_generated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub api_calls: u64,
    pub total_tokens: u64,
    pub total_api_time_ms: u64,
    pub errors: u64,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingRequest {
    input: String,
    model: String,
    encoding_format: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    #[allow(dead_code)]
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    #[allow(dead_code)]
    embedding: Vec<f32>,
    #[allow(dead_code)]
    index: usize,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[allow(dead_code)]
    prompt_tokens: u64,
    #[allow(dead_code)]
    total_tokens: u64,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder
    pub fn new(config: OpenAIEmbeddingConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(MemoryError::configuration(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            ));
        }

        #[cfg(feature = "openai-embeddings")]
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MemoryError::storage(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            #[cfg(feature = "openai-embeddings")]
            client,
            embedding_cache: HashMap::new(),
            metrics: EmbeddingMetrics::default(),
        })
    }

    /// Generate embedding for text using OpenAI API
    pub async fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let start_time = std::time::Instant::now();

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.embedding_cache.get(text) {
                self.metrics.cache_hits += 1;
                return Ok(cached.clone());
            }
            self.metrics.cache_misses += 1;
        }

        // Make API call
        let embedding = self.call_openai_api(text).await?;

        // Cache the result
        if self.config.enable_cache && self.embedding_cache.len() < self.config.cache_size {
            self.embedding_cache.insert(text.to_string(), embedding.clone());
        }

        self.metrics.embeddings_generated += 1;
        self.metrics.total_api_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch
    pub async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        
        // For now, process sequentially to avoid rate limits
        // TODO: Implement proper batching with OpenAI batch API
        for text in texts {
            let embedding = self.embed_text(text).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Get embedding metrics
    pub fn get_metrics(&self) -> &EmbeddingMetrics {
        &self.metrics
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.embedding_cache.len(), self.config.cache_size)
    }

    #[cfg(feature = "openai-embeddings")]
    async fn call_openai_api(&mut self, text: &str) -> Result<Vec<f32>> {
        let request = OpenAIEmbeddingRequest {
            input: text.to_string(),
            model: self.config.model.clone(),
            encoding_format: "float".to_string(),
        };

        let response = self.client
            .post(&self.config.base_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                self.metrics.errors += 1;
                MemoryError::storage(format!("OpenAI API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            self.metrics.errors += 1;
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemoryError::storage(format!("OpenAI API error: {}", error_text)));
        }

        let response_data: OpenAIEmbeddingResponse = response.json().await
            .map_err(|e| {
                self.metrics.errors += 1;
                MemoryError::storage(format!("Failed to parse OpenAI response: {}", e))
            })?;

        if response_data.data.is_empty() {
            self.metrics.errors += 1;
            return Err(MemoryError::storage("No embedding data in OpenAI response"));
        }

        self.metrics.api_calls += 1;
        self.metrics.total_tokens += response_data.usage.total_tokens;

        Ok(response_data.data[0].embedding.clone())
    }

    #[cfg(not(feature = "openai-embeddings"))]
    async fn call_openai_api(&mut self, _text: &str) -> Result<Vec<f32>> {
        Err(MemoryError::configuration(
            "OpenAI embeddings feature not enabled. Enable with --features openai-embeddings"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenAIEmbeddingConfig::default();
        assert_eq!(config.model, "text-embedding-3-large");
        assert_eq!(config.embedding_dim, 3072);
        assert_eq!(config.base_url, "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn test_embedder_creation_without_api_key() {
        let config = OpenAIEmbeddingConfig {
            api_key: String::new(),
            ..Default::default()
        };
        
        let result = OpenAIEmbedder::new(config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embedder_creation_with_api_key() {
        let config = OpenAIEmbeddingConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        
        let embedder = OpenAIEmbedder::new(config);
        assert!(embedder.is_ok());
    }
}
