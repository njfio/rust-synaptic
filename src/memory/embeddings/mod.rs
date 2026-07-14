//! Vector embeddings for semantic memory search
//!
//! This module provides semantic understanding through vector embeddings,
//! enabling similarity search and semantic relationships between memories.

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub mod config;
pub mod multi_provider;
pub mod provider;
pub mod providers;
pub mod similarity;
pub mod simple_embeddings;

// Re-export provider types
pub use provider::{
    compute_content_hash, normalize_vector, CacheStats, EmbedOptions, Embedding, EmbeddingCache,
    EmbeddingProvider, ProviderCapabilities,
};

// Re-export concrete providers
pub use providers::{FallbackEmbeddingProvider, TfIdfConfig, TfIdfProvider};

// Re-export multi-provider system
pub use multi_provider::{MultiProvider, MultiProviderBuilder, MultiProviderConfig};

// Re-export configuration
pub use config::{EmbeddingProviderConfig, GlobalConfig, ProviderConfig, ProviderType};

/// Selects the embedding provider used by the RETRIEVAL pipeline (dense
/// retriever + reranker + the store-time corpus feed).
///
/// The default is [`RetrievalEmbeddingConfig::TfIdf`]: offline, no extra
/// dependency, no network — `cargo build` / `cargo test --lib` need nothing.
/// Semantic providers are opt-in; if the configured semantic provider cannot
/// be constructed or fails at first use, the pipeline FALLS BACK to TF-IDF
/// with a `tracing::warn` (see [`FallbackEmbeddingProvider`]) — it never
/// hard-fails and never fabricates embeddings.
///
/// Note: semantic providers ignore the TF-IDF document-vs-scoring (IDF)
/// distinction — their `embed_for_scoring` delegates to `embed`, which is
/// already read-only (no corpus-relative statistics).
#[derive(Clone, Default)]
pub enum RetrievalEmbeddingConfig {
    /// Offline TF-IDF (hashed TF with live corpus IDF). The default.
    #[default]
    TfIdf,
    /// A local Ollama embedding model (e.g. `nomic-embed-text`, 768-dim).
    /// Requires the `llm-integration` feature (reqwest client); without it,
    /// selection warns and falls back to TF-IDF at construction.
    Ollama {
        /// Ollama server endpoint, e.g. `http://localhost:11434`.
        endpoint: String,
        /// Embedding model name, e.g. `nomic-embed-text`.
        model: String,
        /// Expected embedding dimension (768 for nomic-embed-text).
        dimension: usize,
    },
    /// Any user-supplied provider implementation (also used by tests to
    /// inject mock semantic providers). Wrapped with the TF-IDF fallback.
    Custom(std::sync::Arc<dyn EmbeddingProvider>),
}

impl std::fmt::Debug for RetrievalEmbeddingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TfIdf => write!(f, "RetrievalEmbeddingConfig::TfIdf"),
            Self::Ollama {
                endpoint,
                model,
                dimension,
            } => f
                .debug_struct("RetrievalEmbeddingConfig::Ollama")
                .field("endpoint", endpoint)
                .field("model", model)
                .field("dimension", dimension)
                .finish(),
            Self::Custom(provider) => f
                .debug_struct("RetrievalEmbeddingConfig::Custom")
                .field("provider", &provider.name())
                .finish(),
        }
    }
}

impl RetrievalEmbeddingConfig {
    /// Ollama with the default local endpoint and `nomic-embed-text` (768-d).
    pub fn ollama_default() -> Self {
        Self::Ollama {
            endpoint: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            dimension: 768,
        }
    }

    /// Read the provider selection from the environment, for measurement
    /// runs without code changes: `SYNAPTIC_RETRIEVAL_EMBEDDER=ollama`
    /// selects Ollama (endpoint/model/dimension overridable via
    /// `OLLAMA_ENDPOINT`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_DIM`);
    /// `tfidf` selects TF-IDF explicitly; unset/unknown returns `None`.
    pub fn from_env() -> Option<Self> {
        match std::env::var("SYNAPTIC_RETRIEVAL_EMBEDDER").ok()?.as_str() {
            "ollama" => {
                let endpoint = std::env::var("OLLAMA_ENDPOINT")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string());
                let model = std::env::var("OLLAMA_MODEL")
                    .unwrap_or_else(|_| "nomic-embed-text".to_string());
                let dimension = std::env::var("OLLAMA_EMBEDDING_DIM")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(768);
                Some(Self::Ollama {
                    endpoint,
                    model,
                    dimension,
                })
            }
            "tfidf" => Some(Self::TfIdf),
            other => {
                tracing::warn!(
                    value = other,
                    "unknown SYNAPTIC_RETRIEVAL_EMBEDDER value; using the TF-IDF default"
                );
                None
            }
        }
    }
}

/// Build the shared retrieval embedding provider from its configuration.
///
/// Returns `(provider, tfidf)` where `provider` is the `Arc<dyn
/// EmbeddingProvider>` shared by the dense retriever, the reranker and the
/// store-time corpus feed, and `tfidf` is the concrete TF-IDF instance
/// backing it (either AS the provider, or as the fallback inside a
/// [`FallbackEmbeddingProvider`]) — retained for corpus-statistics
/// introspection.
///
/// Never fails: an unbuildable semantic provider (feature off, client
/// construction error) degrades to TF-IDF with a `tracing::warn`; a
/// reachable-at-construction but failing-at-use provider degrades at first
/// use via the fallback wrapper.
pub fn build_retrieval_provider(
    config: &RetrievalEmbeddingConfig,
) -> (
    std::sync::Arc<dyn EmbeddingProvider>,
    std::sync::Arc<TfIdfProvider>,
) {
    use std::sync::Arc;
    let tfidf = Arc::new(TfIdfProvider::default());
    match config {
        RetrievalEmbeddingConfig::TfIdf => (Arc::clone(&tfidf) as _, tfidf),
        RetrievalEmbeddingConfig::Ollama {
            endpoint,
            model,
            dimension,
        } => {
            #[cfg(feature = "llm-integration")]
            {
                let ollama_config = providers::OllamaConfig::new(model.clone(), *dimension)
                    .with_endpoint(endpoint.clone());
                match providers::OllamaProvider::new(ollama_config) {
                    Ok(ollama) => {
                        tracing::info!(
                            endpoint = %endpoint,
                            model = %model,
                            dimension = dimension,
                            "retrieval embedding provider: Ollama (TF-IDF fallback armed)"
                        );
                        let wrapped =
                            FallbackEmbeddingProvider::new(Arc::new(ollama), Arc::clone(&tfidf));
                        (Arc::new(wrapped) as _, tfidf)
                    }
                    Err(e) => {
                        tracing::warn!(
                            endpoint = %endpoint,
                            model = %model,
                            error = %e,
                            "failed to construct Ollama embedding provider; falling back to TF-IDF"
                        );
                        (Arc::clone(&tfidf) as _, tfidf)
                    }
                }
            }
            #[cfg(not(feature = "llm-integration"))]
            {
                tracing::warn!(
                    endpoint = %endpoint,
                    model = %model,
                    dimension = dimension,
                    "Ollama retrieval embedding requires the 'llm-integration' feature; falling back to TF-IDF"
                );
                (Arc::clone(&tfidf) as _, tfidf)
            }
        }
        RetrievalEmbeddingConfig::Custom(provider) => {
            let wrapped = FallbackEmbeddingProvider::new(Arc::clone(provider), Arc::clone(&tfidf));
            (Arc::new(wrapped) as _, tfidf)
        }
    }
}

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
    pub fn find_similar_to_query(
        &mut self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SimilarMemory>> {
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
        similarities.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// Find memories similar to a given memory
    pub fn find_similar_to_memory(
        &mut self,
        memory_id: Uuid,
        limit: Option<usize>,
    ) -> Result<Vec<SimilarMemory>> {
        let memory_value = self
            .memory_store
            .get(&memory_id)
            .ok_or_else(|| MemoryError::NotFound {
                key: memory_id.to_string(),
            })?
            .value
            .clone();

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
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities)
    }

    /// Get embedding statistics
    pub fn get_stats(&self) -> EmbeddingStats {
        let total_embeddings = self.embedding_cache.len();
        let average_quality = if total_embeddings > 0 {
            self.embedding_cache
                .values()
                .map(|e| e.metadata.quality_score)
                .sum::<f64>()
                / total_embeddings as f64
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
        let variance = vector.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vector.len() as f64;

        // Combine magnitude and variance for quality score

        (magnitude * variance.sqrt()).clamp(0.0, 1.0)
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
