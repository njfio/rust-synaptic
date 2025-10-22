//! Vector indexing for approximate nearest neighbor search
//!
//! This module provides high-performance vector similarity search using
//! approximate nearest neighbor (ANN) algorithms.

use crate::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// Trait for vector index implementations
///
/// Provides a common interface for various ANN (Approximate Nearest Neighbor)
/// index implementations, enabling efficient similarity search at scale.
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The embedding vector to index
    fn add(&mut self, id: Uuid, vector: &[f32]) -> Result<()>;

    /// Add multiple vectors to the index in batch
    ///
    /// # Arguments
    /// * `items` - Vector of (id, vector) pairs to add
    fn add_batch(&mut self, items: Vec<(Uuid, Vec<f32>)>) -> Result<()> {
        for (id, vector) in items {
            self.add(id, &vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    /// * Vector of (id, distance) pairs sorted by distance (closest first)
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>>;

    /// Search with a distance threshold
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `max_distance` - Maximum distance threshold
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    /// * Vector of (id, distance) pairs within the threshold
    fn search_threshold(
        &self,
        query: &[f32],
        max_distance: f32,
        k: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let results = self.search(query, k)?;
        Ok(results
            .into_iter()
            .filter(|(_, dist)| *dist <= max_distance)
            .collect())
    }

    /// Remove a vector from the index
    ///
    /// # Arguments
    /// * `id` - Identifier of the vector to remove
    fn remove(&mut self, id: Uuid) -> Result<()>;

    /// Update a vector in the index
    ///
    /// # Arguments
    /// * `id` - Identifier of the vector to update
    /// * `vector` - New embedding vector
    fn update(&mut self, id: Uuid, vector: &[f32]) -> Result<()> {
        self.remove(id)?;
        self.add(id, vector)?;
        Ok(())
    }

    /// Clear all vectors from the index
    fn clear(&mut self) -> Result<()>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimension of vectors in this index
    fn dimension(&self) -> usize;

    /// Save the index to disk
    ///
    /// # Arguments
    /// * `path` - Path where to save the index
    fn save(&self, path: &Path) -> Result<()>;

    /// Load the index from disk
    ///
    /// # Arguments
    /// * `path` - Path to load the index from
    fn load(&mut self, path: &Path) -> Result<()>;

    /// Get index statistics
    fn stats(&self) -> IndexStats;

    /// Rebuild the index (optimize structure)
    fn rebuild(&mut self) -> Result<()> {
        // Default implementation: no-op
        Ok(())
    }
}

/// Statistics about a vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of vectors in the index
    pub vector_count: usize,
    /// Dimension of vectors
    pub dimension: usize,
    /// Index type (e.g., "HNSW", "IVF", "Flat")
    pub index_type: String,
    /// Memory usage in bytes (approximate)
    pub memory_bytes: usize,
    /// Average search time in microseconds
    pub avg_search_time_us: Option<f64>,
    /// Index-specific parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Search result with additional metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Memory ID
    pub id: Uuid,
    /// Distance to query vector (lower is more similar)
    pub distance: f32,
    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub similarity: f64,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: Uuid, distance: f32) -> Self {
        // Convert distance to similarity score (assuming cosine distance)
        // distance = 1 - cosine_similarity
        // similarity = 1 - distance
        let similarity = 1.0 - distance as f64;
        Self {
            id,
            distance,
            similarity: similarity.max(0.0).min(1.0),
        }
    }
}

/// Configuration for vector indexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Dimension of vectors
    pub dimension: usize,
    /// Maximum number of vectors (for capacity planning)
    pub max_vectors: Option<usize>,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Index-specific parameters
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// Distance metrics for vector comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Dot product (inner product)
    DotProduct,
}

impl DistanceMetric {
    /// Compute distance between two vectors
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        match self {
            Self::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            Self::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            Self::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            Self::DotProduct => {
                // Negative dot product (to maintain "lower is better" convention)
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_vectors: None,
            distance_metric: DistanceMetric::Cosine,
            parameters: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        // Identical vectors should have 0 distance (except cosine which returns 0 distance as 0)
        assert!((DistanceMetric::Cosine.compute(&a, &b)).abs() < 0.001);
        assert!((DistanceMetric::Euclidean.compute(&a, &b)).abs() < 0.001);

        // Orthogonal vectors
        let cos_dist = DistanceMetric::Cosine.compute(&a, &c);
        assert!((cos_dist - 1.0).abs() < 0.001); // Cosine distance = 1 for orthogonal

        let eucl_dist = DistanceMetric::Euclidean.compute(&a, &c);
        assert!((eucl_dist - 1.414).abs() < 0.01); // sqrt(2)
    }

    #[test]
    fn test_search_result() {
        let result = SearchResult::new(Uuid::new_v4(), 0.2);
        assert!((result.distance - 0.2).abs() < 0.001);
        assert!((result.similarity - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.dimension, 384);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
    }
}
