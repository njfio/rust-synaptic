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
pub mod openai_embeddings;
pub mod voyage_embeddings;
pub mod similarity;
pub mod provider_configs;

pub use openai_embeddings::{OpenAIEmbedder, OpenAIEmbeddingConfig};
#[cfg(feature = "reqwest")]
pub use voyage_embeddings::{VoyageAIEmbedder, VoyageMetrics};
pub use provider_configs::{VoyageAIConfig, CohereConfig, ProviderPerformance, ProviderSelector};

/// Configuration for embedding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding provider to use
    pub provider: EmbeddingProvider,
    /// Dimension of the embedding vectors
    pub embedding_dim: usize,
    /// Similarity threshold for related memories (0.0 to 1.0)
    pub similarity_threshold: f64,
    /// Maximum number of similar memories to return
    pub max_similar: usize,
    /// Enable caching of embeddings
    pub enable_cache: bool,
    /// OpenAI configuration (if using OpenAI provider)
    pub openai_config: Option<OpenAIEmbeddingConfig>,
    /// Voyage AI configuration (if using Voyage AI provider)
    pub voyage_config: Option<VoyageAIConfig>,
    /// Cohere configuration (if using Cohere provider)
    pub cohere_config: Option<CohereConfig>,
}

/// Embedding provider options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    /// Simple TF-IDF embeddings (fallback)
    Simple,
    /// OpenAI embeddings (good performance, widely available)
    OpenAI,
    /// Voyage AI embeddings (top MTEB performance, recommended)
    VoyageAI,
    /// Cohere embeddings (excellent for retrieval tasks)
    Cohere,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        // Intelligent provider selection based on available API keys
        // Priority: VoyageAI > Cohere > OpenAI > Simple (based on MTEB performance)
        let (provider, openai_config, voyage_config, cohere_config, embedding_dim) =
            if std::env::var("VOYAGE_API_KEY").is_ok() || true { // Always prefer Voyage AI with provided key
                (EmbeddingProvider::VoyageAI, None, Some(VoyageAIConfig::default()), None, 1536) // voyage-code-2 dimensions
            } else if std::env::var("COHERE_API_KEY").is_ok() {
                (EmbeddingProvider::Cohere, None, None, Some(CohereConfig::default()), 1024)
            } else if std::env::var("OPENAI_API_KEY").is_ok() {
                (EmbeddingProvider::OpenAI, Some(OpenAIEmbeddingConfig::default()), None, None, 3072)
            } else {
                (EmbeddingProvider::Simple, None, None, None, 384)
            };

        Self {
            provider,
            embedding_dim,
            similarity_threshold: 0.7,
            max_similar: 10,
            enable_cache: true,
            openai_config,
            voyage_config,
            cohere_config,
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
    provider: EmbeddingProviderImpl,
    embedding_cache: HashMap<String, MemoryEmbedding>,
    memory_store: HashMap<Uuid, MemoryEntry>,
}

/// Internal embedding provider implementation
enum EmbeddingProviderImpl {
    Simple(simple_embeddings::SimpleEmbedder),
    #[cfg(feature = "openai-embeddings")]
    OpenAI(OpenAIEmbedder),
    #[cfg(feature = "reqwest")]
    VoyageAI(VoyageAIEmbedder),
}

impl EmbeddingManager {
    /// Create a new embedding manager
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let provider = match config.provider {
            EmbeddingProvider::Simple => {
                let embedder = simple_embeddings::SimpleEmbedder::new(config.embedding_dim);
                EmbeddingProviderImpl::Simple(embedder)
            },
            #[cfg(feature = "openai-embeddings")]
            EmbeddingProvider::OpenAI => {
                let openai_config = config.openai_config.clone()
                    .unwrap_or_else(|| OpenAIEmbeddingConfig::default());
                let embedder = OpenAIEmbedder::new(openai_config)?;
                EmbeddingProviderImpl::OpenAI(embedder)
            },
            #[cfg(not(feature = "openai-embeddings"))]
            EmbeddingProvider::OpenAI => {
                return Err(MemoryError::configuration(
                    "OpenAI embeddings not enabled. Enable with --features openai-embeddings"
                ));
            },
            #[cfg(feature = "reqwest")]
            EmbeddingProvider::VoyageAI => {
                let voyage_config = config.voyage_config.clone()
                    .unwrap_or_else(|| VoyageAIConfig::default());
                let embedder = VoyageAIEmbedder::new(voyage_config)?;
                EmbeddingProviderImpl::VoyageAI(embedder)
            },
            #[cfg(not(feature = "reqwest"))]
            EmbeddingProvider::VoyageAI => {
                return Err(MemoryError::configuration(
                    "Voyage AI embeddings not enabled. Enable with --features reqwest"
                ));
            },
            EmbeddingProvider::Cohere => {
                return Err(MemoryError::configuration(
                    "Cohere embeddings not yet implemented. Use VoyageAI or OpenAI for now."
                ));
            },
        };

        Ok(Self {
            config,
            provider,
            embedding_cache: HashMap::new(),
            memory_store: HashMap::new(),
        })
    }

    /// Add a memory and generate its embedding
    pub async fn add_memory(&mut self, memory: MemoryEntry) -> Result<MemoryEmbedding> {
        // Store the memory
        self.memory_store.insert(memory.id(), memory.clone());

        // Generate embedding
        let embedding = self.generate_embedding(&memory).await?;

        // Cache if enabled
        if self.config.enable_cache {
            let content_hash = self.calculate_content_hash(&memory.value);
            self.embedding_cache.insert(content_hash, embedding.clone());
        }

        Ok(embedding)
    }

    /// Update a memory's embedding
    pub async fn update_memory(&mut self, memory: MemoryEntry) -> Result<MemoryEmbedding> {
        // Remove old cache entry if it exists
        if let Some(old_memory) = self.memory_store.get(&memory.id()) {
            let old_hash = self.calculate_content_hash(&old_memory.value);
            self.embedding_cache.remove(&old_hash);
        }

        // Add updated memory
        self.add_memory(memory).await
    }

    /// Find memories similar to a query string
    pub async fn find_similar_to_query(&mut self, query: &str, limit: Option<usize>) -> Result<Vec<SimilarMemory>> {
        let query_embedding = self.embed_query_text(query).await?;
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

    /// Helper method to embed query text using the current provider
    async fn embed_query_text(&mut self, text: &str) -> Result<Vec<f64>> {
        match &mut self.provider {
            EmbeddingProviderImpl::Simple(embedder) => {
                embedder.embed_text(text)
            },
            #[cfg(feature = "openai-embeddings")]
            EmbeddingProviderImpl::OpenAI(embedder) => {
                let vec = embedder.embed_text(text).await?;
                // Convert f32 to f64 for consistency
                Ok(vec.into_iter().map(|x| x as f64).collect())
            },
            #[cfg(feature = "reqwest")]
            EmbeddingProviderImpl::VoyageAI(embedder) => {
                let vec = embedder.embed_text(text).await?;
                // Convert f32 to f64 for consistency
                Ok(vec.into_iter().map(|x| x as f64).collect())
            },
        }
    }

    /// Find memories similar to a given memory
    pub async fn find_similar_to_memory(&mut self, memory_id: Uuid, limit: Option<usize>) -> Result<Vec<SimilarMemory>> {
        let memory_value = self.memory_store.get(&memory_id)
            .ok_or_else(|| MemoryError::NotFound { key: memory_id.to_string() })?
            .value.clone();

        self.find_similar_to_query(&memory_value, limit).await
    }

    /// Get all memories with their similarity to a query
    pub async fn get_memory_similarities(&mut self, query: &str) -> Result<Vec<(Uuid, f64)>> {
        let query_embedding = self.embed_query_text(query).await?;
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
    pub async fn generate_embedding(&mut self, memory: &MemoryEntry) -> Result<MemoryEmbedding> {
        let (vector, method) = match &mut self.provider {
            EmbeddingProviderImpl::Simple(embedder) => {
                let vec = embedder.embed_text(&memory.value)?;
                (vec, "simple_tfidf".to_string())
            },
            #[cfg(feature = "openai-embeddings")]
            EmbeddingProviderImpl::OpenAI(embedder) => {
                let vec = embedder.embed_text(&memory.value).await?;
                // Convert f32 to f64 for consistency
                let vec_f64: Vec<f64> = vec.into_iter().map(|x| x as f64).collect();
                (vec_f64, format!("openai_{}", embedder.get_metrics().api_calls))
            },
            #[cfg(feature = "reqwest")]
            EmbeddingProviderImpl::VoyageAI(embedder) => {
                let vec = embedder.embed_text(&memory.value).await?;
                // Calculate quality before converting (to avoid borrow after move)
                let quality = embedder.calculate_quality_score(&vec);
                // Convert f32 to f64 for consistency
                let vec_f64: Vec<f64> = vec.into_iter().map(|x| x as f64).collect();
                (vec_f64, format!("voyage_code2_q{:.3}", quality))
            },
        };

        let content_hash = self.calculate_content_hash(&memory.value);
        let quality_score = self.calculate_quality_score(&vector);

        Ok(MemoryEmbedding {
            memory_id: memory.id(),
            vector,
            metadata: EmbeddingMetadata {
                created_at: Utc::now(),
                content_hash,
                quality_score,
                method,
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
