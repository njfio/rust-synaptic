//! Voyage AI embeddings implementation
//!
//! Provides real Voyage AI embeddings integration with support for code-optimized models.
//! Voyage AI currently leads MTEB benchmarks and offers specialized models for code.

#[cfg(feature = "reqwest")]
mod implementation {
    use crate::error::{Result, MemoryError};
    use crate::memory::embeddings::provider_configs::VoyageAIConfig;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::time::Duration;
    use reqwest::Client;
    use lru::LruCache;
    use std::num::NonZeroUsize;

    /// Voyage AI API request structure
    #[derive(Debug, Serialize)]
struct VoyageEmbeddingRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
}

/// Voyage AI API response structure
#[derive(Debug, Deserialize)]
struct VoyageEmbeddingResponse {
    data: Vec<VoyageEmbeddingData>,
    model: String,
    usage: VoyageUsage,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct VoyageUsage {
    total_tokens: u32,
}

/// Voyage AI embedder with caching and performance optimization
pub struct VoyageAIEmbedder {
    config: VoyageAIConfig,
    client: Client,
    cache: Option<LruCache<String, Vec<f32>>>,
    metrics: VoyageMetrics,
}

/// Performance metrics for Voyage AI embeddings
#[derive(Debug, Clone, Default)]
pub struct VoyageMetrics {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_response_time_ms: f64,
    pub error_count: u64,
}

impl VoyageAIEmbedder {
    /// Create a new Voyage AI embedder
    pub fn new(config: VoyageAIConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(MemoryError::configuration("Voyage AI API key is required"));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| MemoryError::storage(format!("Failed to create HTTP client: {}", e)))?;

        let cache = if config.enable_cache {
            Some(LruCache::new(NonZeroUsize::new(config.cache_size).unwrap()))
        } else {
            None
        };

        Ok(Self {
            config,
            client,
            cache,
            metrics: VoyageMetrics::default(),
        })
    }

    /// Generate embedding for a single text
    pub async fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cache) = &mut self.cache {
            if let Some(cached) = cache.get(text) {
                self.metrics.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        self.metrics.cache_misses += 1;
        let start_time = std::time::Instant::now();

        let embeddings = self.embed_batch(&[text.to_string()]).await?;
        let embedding = embeddings.into_iter().next()
            .ok_or_else(|| MemoryError::storage("No embedding returned"))?;

        // Cache the result
        if let Some(cache) = &mut self.cache {
            cache.put(text.to_string(), embedding.clone());
        }

        self.metrics.average_response_time_ms = 
            (self.metrics.average_response_time_ms * self.metrics.total_requests as f64 + 
             start_time.elapsed().as_millis() as f64) / (self.metrics.total_requests + 1) as f64;
        self.metrics.total_requests += 1;

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let request = VoyageEmbeddingRequest {
            input: texts.to_vec(),
            model: self.config.model.clone(),
            input_type: Some("document".to_string()), // For storing documents
        };

        let response = self.client
            .post(&self.config.base_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                self.metrics.error_count += 1;
                MemoryError::storage(format!("Voyage AI API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            self.metrics.error_count += 1;
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(MemoryError::storage(format!(
                "Voyage AI API error {}: {}", status, error_text
            )));
        }

        let voyage_response: VoyageEmbeddingResponse = response.json().await
            .map_err(|e| {
                self.metrics.error_count += 1;
                MemoryError::storage(format!("Failed to parse Voyage AI response: {}", e))
            })?;

        self.metrics.total_tokens += voyage_response.usage.total_tokens as u64;

        // Sort embeddings by index to maintain order
        let mut embeddings: Vec<_> = voyage_response.data.into_iter().collect();
        embeddings.sort_by_key(|d| d.index);

        Ok(embeddings.into_iter().map(|d| d.embedding).collect())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> VoyageMetrics {
        self.metrics.clone()
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &mut self.cache {
            cache.clear();
        }
    }

    /// Calculate embedding quality score based on vector properties
    pub fn calculate_quality_score(&self, embedding: &[f32]) -> f64 {
        if embedding.is_empty() {
            return 0.0;
        }

        // Calculate vector magnitude
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Calculate variance (diversity of values)
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance: f32 = embedding.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / embedding.len() as f32;

        // Quality score based on magnitude and variance
        // Good embeddings should have reasonable magnitude and variance
        let magnitude_score = (magnitude / embedding.len() as f32).min(1.0);
        let variance_score = (variance * 10.0).min(1.0);

        ((magnitude_score + variance_score) / 2.0) as f64
    }

    /// Get model information
    pub fn get_model_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("provider".to_string(), "Voyage AI".to_string());
        info.insert("model".to_string(), self.config.model.clone());
        info.insert("dimensions".to_string(), self.config.embedding_dim.to_string());
        info.insert("optimized_for".to_string(), 
            if self.config.model.contains("code") {
                "Code and programming languages".to_string()
            } else {
                "General text and retrieval".to_string()
            }
        );
        info.insert("api_version".to_string(), "v1".to_string());
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> VoyageAIConfig {
        VoyageAIConfig {
            api_key: "test-key".to_string(),
            model: "voyage-code-2".to_string(),
            embedding_dim: 1536,
            base_url: "https://api.voyageai.com/v1/embeddings".to_string(),
            timeout_secs: 30,
            enable_cache: true,
            cache_size: 100,
        }
    }

    #[test]
    fn test_embedder_creation() {
        let config = create_test_config();
        let embedder = VoyageAIEmbedder::new(config);
        assert!(embedder.is_ok());
    }

    #[test]
    fn test_embedder_creation_empty_key() {
        let mut config = create_test_config();
        config.api_key = String::new();
        let embedder = VoyageAIEmbedder::new(config);
        assert!(embedder.is_err());
    }

    #[test]
    fn test_quality_score_calculation() {
        let config = create_test_config();
        let embedder = VoyageAIEmbedder::new(config).unwrap();
        
        let embedding = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let score = embedder.calculate_quality_score(&embedding);
        assert!(score > 0.0 && score <= 1.0);
        
        let empty_embedding = vec![];
        let empty_score = embedder.calculate_quality_score(&empty_embedding);
        assert_eq!(empty_score, 0.0);
    }

    #[test]
    fn test_model_info() {
        let config = create_test_config();
        let embedder = VoyageAIEmbedder::new(config).unwrap();
        
        let info = embedder.get_model_info();
        assert_eq!(info.get("provider").unwrap(), "Voyage AI");
        assert_eq!(info.get("model").unwrap(), "voyage-code-2");
        assert!(info.get("optimized_for").unwrap().contains("Code"));
    }

    #[test]
    fn test_metrics_initialization() {
        let config = create_test_config();
        let embedder = VoyageAIEmbedder::new(config).unwrap();
        
        let metrics = embedder.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
    }
}

} // Close implementation module

#[cfg(feature = "reqwest")]
pub use implementation::*;
