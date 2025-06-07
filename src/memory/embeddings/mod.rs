//! Vector embeddings for semantic memory search
//! 
//! This module provides semantic understanding through vector embeddings,
//! enabling similarity search and semantic relationships between memories.

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub mod simple_embeddings;
pub mod similarity;

/// Configuration for embedding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimension of the embedding vectors
    pub embedding_dim: usize,
    /// Similarity threshold for related memories (0.0 to 1.0)
    pub similarity_threshold: f64,
    /// Maximum number of similar memories to return
    pub max_similar: usize,
    /// Enable caching of embeddings
    pub enable_cache: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            similarity_threshold: 0.7,
            max_similar: 10,
            enable_cache: true,
        }
    }
}

/// Vector embedding for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbedding {
    /// Memory ID this embedding represents
    pub memory_id: Uuid,
    /// The embedding vector
    pub vector: Vec<f64>,
    /// Metadata about the embedding
    pub metadata: EmbeddingMetadata,
}

/// Metadata for embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Timestamp when embedding was created
    pub created_at: DateTime<Utc>,
    /// Content hash for cache validation
    pub content_hash: String,
    /// Embedding quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Method used to generate embedding
    pub method: String,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilarMemory {
    /// The similar memory
    pub memory: MemoryEntry,
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f64,
    /// Euclidean distance in embedding space
    pub distance: f64,
}

/// Main embedding manager
pub struct EmbeddingManager {
    config: EmbeddingConfig,
    embedder: simple_embeddings::SimpleEmbedder,
    embedding_cache: HashMap<String, MemoryEmbedding>,
    memory_store: HashMap<Uuid, MemoryEntry>,
}

impl EmbeddingManager {
    /// Create a new embedding manager
    pub fn new(config: EmbeddingConfig) -> Self {
        let embedder = simple_embeddings::SimpleEmbedder::new(config.embedding_dim);
        
        Self {
            config,
            embedder,
            embedding_cache: HashMap::new(),
            memory_store: HashMap::new(),
        }
    }

    /// Add a memory and generate its embedding
    pub fn add_memory(&mut self, memory: MemoryEntry) -> Result<MemoryEmbedding> {
        // Store the memory
        self.memory_store.insert(memory.id(), memory.clone());

        // Generate embedding
        let embedding = self.generate_embedding(&memory)?;
        
        // Cache if enabled
        if self.config.enable_cache {
            let content_hash = self.calculate_content_hash(&memory.value);
            self.embedding_cache.insert(content_hash, embedding.clone());
        }

        Ok(embedding)
    }

    /// Update a memory's embedding
    pub fn update_memory(&mut self, memory: MemoryEntry) -> Result<MemoryEmbedding> {
        // Remove old cache entry if it exists
        if let Some(old_memory) = self.memory_store.get(&memory.id()) {
            let old_hash = self.calculate_content_hash(&old_memory.value);
            self.embedding_cache.remove(&old_hash);
        }

        // Add updated memory
        self.add_memory(memory)
    }

    /// Find memories similar to a query string
    pub fn find_similar_to_query(&mut self, query: &str, limit: Option<usize>) -> Result<Vec<SimilarMemory>> {
        let query_embedding = self.embedder.embed_text(query)?;
        let limit = limit.unwrap_or(self.config.max_similar);
        
        let mut similarities = Vec::new();
        
        for embedding in self.embedding_cache.values() {
            if let Some(memory) = self.memory_store.get(&embedding.memory_id) {
                let similarity = similarity::cosine_similarity(&query_embedding, &embedding.vector);
                let distance = similarity::euclidean_distance(&query_embedding, &embedding.vector);
                
                if similarity >= self.config.similarity_threshold {
                    similarities.push(SimilarMemory {
                        memory: memory.clone(),
                        similarity,
                        distance,
                    });
                }
            }
        }

        // Sort by similarity (highest first) and limit results
        similarities.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        similarities.truncate(limit);
        
        Ok(similarities)
    }

    /// Find memories similar to a given memory
    pub fn find_similar_to_memory(&mut self, memory_id: Uuid, limit: Option<usize>) -> Result<Vec<SimilarMemory>> {
        let memory_value = self.memory_store.get(&memory_id)
            .ok_or_else(|| MemoryError::NotFound { key: memory_id.to_string() })?
            .value.clone();

        self.find_similar_to_query(&memory_value, limit)
    }

    /// Get all memories with their similarity to a query
    pub fn get_memory_similarities(&mut self, query: &str) -> Result<Vec<(Uuid, f64)>> {
        let query_embedding = self.embedder.embed_text(query)?;
        let mut similarities = Vec::new();
        
        for embedding in self.embedding_cache.values() {
            let similarity = similarity::cosine_similarity(&query_embedding, &embedding.vector);
            similarities.push((embedding.memory_id, similarity));
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(similarities)
    }

    /// Get embedding statistics
    pub fn get_stats(&self) -> EmbeddingStats {
        let total_embeddings = self.embedding_cache.len();
        let average_quality = if total_embeddings > 0 {
            self.embedding_cache.values()
                .map(|e| e.metadata.quality_score)
                .sum::<f64>() / total_embeddings as f64
        } else {
            0.0
        };

        EmbeddingStats {
            total_embeddings,
            total_memories: self.memory_store.len(),
            embedding_dimension: self.config.embedding_dim,
            average_quality_score: average_quality,
            cache_enabled: self.config.enable_cache,
            similarity_threshold: self.config.similarity_threshold,
        }
    }

    /// Clear all embeddings and memories
    pub fn clear(&mut self) {
        self.embedding_cache.clear();
        self.memory_store.clear();
    }

    /// Generate embedding for a memory
    pub fn generate_embedding(&mut self, memory: &MemoryEntry) -> Result<MemoryEmbedding> {
        let vector = self.embedder.embed_text(&memory.value)?;
        let content_hash = self.calculate_content_hash(&memory.value);
        let quality_score = self.calculate_quality_score(&vector);

        Ok(MemoryEmbedding {
            memory_id: memory.id(),
            vector,
            metadata: EmbeddingMetadata {
                created_at: Utc::now(),
                content_hash,
                quality_score,
                method: "simple_tfidf".to_string(),
            },
        })
    }

    /// Calculate content hash for caching
    fn calculate_content_hash(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Calculate quality score for an embedding
    fn calculate_quality_score(&self, vector: &[f64]) -> f64 {
        if vector.is_empty() {
            return 0.0;
        }

        // Calculate vector magnitude
        let magnitude = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        // Calculate variance (measure of information content)
        let mean = vector.iter().sum::<f64>() / vector.len() as f64;
        let variance = vector.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / vector.len() as f64;
        
        // Combine magnitude and variance for quality score
        let quality = (magnitude * variance.sqrt()).min(1.0).max(0.0);
        quality
    }
}

/// Statistics about the embedding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    pub total_embeddings: usize,
    pub total_memories: usize,
    pub embedding_dimension: usize,
    pub average_quality_score: f64,
    pub cache_enabled: bool,
    pub similarity_threshold: f64,
}
