//! Vector index trait for Approximate Nearest Neighbor (ANN) search.
//!
//! This module defines the generic interface for vector similarity search indexes,
//! enabling efficient retrieval of similar embeddings at scale.

use crate::memory::types::MemoryEntry;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Error types for vector indexing operations
#[derive(Debug, Error)]
pub enum IndexError {
    /// Index build failed
    #[error("Failed to build index: {0}")]
    BuildFailed(String),

    /// Index persistence failed
    #[error("Failed to save index: {0}")]
    SaveFailed(String),

    /// Index loading failed
    #[error("Failed to load index: {0}")]
    LoadFailed(String),

    /// Index query failed
    #[error("Failed to query index: {0}")]
    QueryFailed(String),

    /// Index update failed
    #[error("Failed to update index: {0}")]
    UpdateFailed(String),

    /// Invalid dimension
    #[error("Invalid dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Empty index
    #[error("Cannot query empty index")]
    EmptyIndex,

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for index operations
pub type IndexResult<T> = Result<T, IndexError>;

/// Search result containing a memory entry and its similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The memory entry
    pub entry: MemoryEntry,
    /// Similarity score (higher is more similar)
    pub score: f64,
    /// Distance metric value (lower is more similar)
    pub distance: f32,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(entry: MemoryEntry, score: f64, distance: f32) -> Self {
        Self {
            entry,
            score,
            distance,
        }
    }
}

/// Configuration for vector index construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Maximum number of connections per layer
    pub max_connections: usize,
    /// Maximum number of connections in layer 0
    pub max_connections_0: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search
    pub ef_search: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384, // Common embedding dimension (sentence-transformers)
            max_connections: 16,
            max_connections_0: 32,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl IndexConfig {
    /// Create a new config with specified dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set maximum connections per layer
    pub fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.max_connections = max_connections;
        self
    }

    /// Set ef_construction (build quality)
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set ef_search (search quality)
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Distance metrics for similarity computation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Euclidean distance (L2)
    Euclidean,
    /// Manhattan distance (L1)
    Manhattan,
    /// Dot product similarity
    DotProduct,
}

/// Generic vector index trait for ANN search
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Get the dimension of vectors in this index
    fn dimension(&self) -> usize;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the distance metric used by this index
    fn metric(&self) -> DistanceMetric;

    /// Add a vector with associated memory entry to the index
    async fn add(&mut self, vector: &[f32], entry: MemoryEntry) -> IndexResult<()>;

    /// Add multiple vectors with associated entries (batch operation)
    async fn add_batch(&mut self, vectors: &[Vec<f32>], entries: Vec<MemoryEntry>) -> IndexResult<()>;

    /// Search for k nearest neighbors
    async fn search(&self, query: &[f32], k: usize) -> IndexResult<Vec<SearchResult>>;

    /// Search with a minimum similarity threshold
    async fn search_threshold(
        &self,
        query: &[f32],
        k: usize,
        min_score: f64,
    ) -> IndexResult<Vec<SearchResult>>;

    /// Remove a vector by memory key
    async fn remove(&mut self, key: &str) -> IndexResult<bool>;

    /// Clear all vectors from the index
    async fn clear(&mut self) -> IndexResult<()>;

    /// Save the index to disk
    async fn save<P: AsRef<Path> + Send>(&self, path: P) -> IndexResult<()>;

    /// Load the index from disk
    async fn load<P: AsRef<Path> + Send>(&mut self, path: P) -> IndexResult<()>;

    /// Rebuild the index from scratch (useful for optimization)
    async fn rebuild(&mut self) -> IndexResult<()>;

    /// Get index statistics
    fn stats(&self) -> IndexStats;
}

/// Statistics about the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of vectors in index
    pub num_vectors: usize,
    /// Dimension of vectors
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Memory usage estimate (bytes)
    pub memory_bytes: usize,
    /// Number of layers (for HNSW)
    pub num_layers: Option<usize>,
    /// Average connections per node
    pub avg_connections: Option<f64>,
}

impl IndexStats {
    /// Create basic stats
    pub fn new(num_vectors: usize, dimension: usize, metric: DistanceMetric) -> Self {
        Self {
            num_vectors,
            dimension,
            metric,
            memory_bytes: 0,
            num_layers: None,
            avg_connections: None,
        }
    }

    /// Estimate memory usage
    pub fn with_memory_estimate(mut self, memory_bytes: usize) -> Self {
        self.memory_bytes = memory_bytes;
        self
    }

    /// Add HNSW-specific stats
    pub fn with_hnsw_stats(mut self, num_layers: usize, avg_connections: f64) -> Self {
        self.num_layers = Some(num_layers);
        self.avg_connections = Some(avg_connections);
        self
    }
}
