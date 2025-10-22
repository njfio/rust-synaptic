//! Pluggable embedding provider system
//!
//! This module defines the EmbeddingProvider trait and implements various
//! embedding strategies for semantic search.

use crate::error::{MemoryError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Trait for embedding providers
///
/// Implementors provide different strategies for generating vector embeddings
/// from text, enabling semantic search with various models and approaches.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single text
    ///
    /// # Arguments
    /// * `text` - The text to embed
    /// * `options` - Optional embedding options
    ///
    /// # Returns
    /// * Embedding vector with metadata
    async fn embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding>;

    /// Generate embeddings for multiple texts (batch operation)
    ///
    /// # Arguments
    /// * `texts` - Vector of texts to embed
    /// * `options` - Optional embedding options
    ///
    /// # Returns
    /// * Vector of embeddings corresponding to input texts
    async fn embed_batch(
        &self,
        texts: &[String],
        options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        // Default implementation: sequential embedding
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text, options).await?);
        }
        Ok(embeddings)
    }

    /// Get the dimension of embeddings produced by this provider
    fn embedding_dimension(&self) -> usize;

    /// Get the name of this embedding provider
    fn name(&self) -> &'static str;

    /// Get the model identifier (e.g., "text-embedding-ada-002", "all-MiniLM-L6-v2")
    fn model_id(&self) -> String;

    /// Check if this provider is available/initialized
    fn is_available(&self) -> bool {
        true
    }

    /// Get provider capabilities
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
}

/// Embedding with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Model that generated this embedding
    pub model: String,
    /// Model version
    pub version: Option<String>,
    /// Timestamp when created
    pub created_at: DateTime<Utc>,
    /// Content hash for cache validation
    pub content_hash: String,
    /// Token count (if applicable)
    pub token_count: Option<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Embedding {
    /// Create a new embedding
    pub fn new(vector: Vec<f32>, model: String) -> Self {
        Self {
            vector,
            model,
            version: None,
            created_at: Utc::now(),
            content_hash: String::new(),
            token_count: None,
            metadata: HashMap::new(),
        }
    }

    /// Set content hash
    pub fn with_content_hash(mut self, hash: String) -> Self {
        self.content_hash = hash;
        self
    }

    /// Set version
    pub fn with_version(mut self, version: String) -> Self {
        self.version = Some(version);
        self
    }

    /// Set token count
    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> f64 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot_product / (norm_a * norm_b)) as f64
        }
    }
}

/// Options for embedding generation
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Truncate input to maximum length
    pub truncate: bool,
    /// Normalize the output vector
    pub normalize: bool,
    /// Custom metadata to attach
    pub metadata: HashMap<String, String>,
}

impl EmbedOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Provider capabilities
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    /// Supports batch embedding
    pub supports_batch: bool,
    /// Maximum batch size
    pub max_batch_size: Option<usize>,
    /// Maximum input length (in tokens or characters)
    pub max_input_length: Option<usize>,
    /// Supports different input types (code, queries, documents)
    pub supports_input_types: bool,
    /// Requires API key
    pub requires_api_key: bool,
    /// Is local (doesn't require network)
    pub is_local: bool,
}

/// Embedding cache for avoiding re-computation
pub struct EmbeddingCache {
    cache: HashMap<String, Embedding>,
    max_size: usize,
    ttl_seconds: u64,
}

impl EmbeddingCache {
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            ttl_seconds,
        }
    }

    /// Get cached embedding
    pub fn get(&self, content_hash: &str) -> Option<&Embedding> {
        if let Some(embedding) = self.cache.get(content_hash) {
            // Check if expired
            let age = (Utc::now() - embedding.created_at).num_seconds() as u64;
            if age < self.ttl_seconds {
                return Some(embedding);
            }
        }
        None
    }

    /// Insert embedding into cache
    pub fn insert(&mut self, content_hash: String, embedding: Embedding) {
        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            // Simple LRU: remove oldest by created_at
            if let Some(oldest_key) = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone())
            {
                self.cache.remove(&oldest_key);
            }
        }

        self.cache.insert(content_hash, embedding);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.cache.len(),
            max_size: self.max_size,
            ttl_seconds: self.ttl_seconds,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entry_count: usize,
    pub max_size: usize,
    pub ttl_seconds: u64,
}

/// Compute content hash for caching
pub fn compute_content_hash(text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Normalize a vector to unit length
pub fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let vector = vec![0.1, 0.2, 0.3];
        let embedding = Embedding::new(vector.clone(), "test-model".to_string())
            .with_content_hash("hash123".to_string())
            .with_version("v1".to_string())
            .with_token_count(10);

        assert_eq!(embedding.dimension(), 3);
        assert_eq!(embedding.model, "test-model");
        assert_eq!(embedding.content_hash, "hash123");
        assert_eq!(embedding.version, Some("v1".to_string()));
        assert_eq!(embedding.token_count, Some(10));
    }

    #[test]
    fn test_cosine_similarity() {
        let emb1 = Embedding::new(vec![1.0, 0.0, 0.0], "model".to_string());
        let emb2 = Embedding::new(vec![1.0, 0.0, 0.0], "model".to_string());
        let emb3 = Embedding::new(vec![0.0, 1.0, 0.0], "model".to_string());

        // Identical vectors should have similarity 1.0
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.001);

        // Orthogonal vectors should have similarity 0.0
        assert!(emb1.cosine_similarity(&emb3).abs() < 0.001);
    }

    #[test]
    fn test_normalize_vector() {
        let mut vector = vec![3.0, 4.0];
        normalize_vector(&mut vector);

        // Length should be 1.0
        let length: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 0.001);

        // Values should be 0.6 and 0.8
        assert!((vector[0] - 0.6).abs() < 0.001);
        assert!((vector[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(2, 3600);

        let emb1 = Embedding::new(vec![1.0, 2.0], "model".to_string())
            .with_content_hash("hash1".to_string());
        let emb2 = Embedding::new(vec![3.0, 4.0], "model".to_string())
            .with_content_hash("hash2".to_string());

        cache.insert("hash1".to_string(), emb1);
        cache.insert("hash2".to_string(), emb2);

        assert!(cache.get("hash1").is_some());
        assert!(cache.get("hash2").is_some());
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = EmbeddingCache::new(2, 3600);

        let emb1 = Embedding::new(vec![1.0], "model".to_string());
        let emb2 = Embedding::new(vec![2.0], "model".to_string());
        let emb3 = Embedding::new(vec![3.0], "model".to_string());

        cache.insert("hash1".to_string(), emb1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.insert("hash2".to_string(), emb2);
        std::thread::sleep(std::time::Duration::from_millis(10));

        // This should evict hash1 (oldest)
        cache.insert("hash3".to_string(), emb3);

        assert_eq!(cache.stats().entry_count, 2);
        assert!(cache.get("hash1").is_none());
        assert!(cache.get("hash2").is_some());
        assert!(cache.get("hash3").is_some());
    }

    #[test]
    fn test_content_hash() {
        let hash1 = compute_content_hash("test content");
        let hash2 = compute_content_hash("test content");
        let hash3 = compute_content_hash("different content");

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_embed_options_builder() {
        let options = EmbedOptions::new()
            .with_truncate(true)
            .with_normalize(false);

        assert!(options.truncate);
        assert!(!options.normalize);
    }
}
