//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! This module provides an efficient ANN (Approximate Nearest Neighbor) search
//! using the HNSW algorithm, which provides excellent search quality with
//! logarithmic complexity.

use super::vector::{
    DistanceMetric, IndexConfig, IndexError, IndexResult, IndexStats, SearchResult, VectorIndex,
};
use crate::memory::types::MemoryEntry;
use async_trait::async_trait;
use dashmap::DashMap;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// HNSW index for vector similarity search
pub struct HnswIndex {
    /// The actual HNSW index from hnsw_rs
    index: Arc<RwLock<Option<Hnsw<f32, DistL2>>>>,
    /// Configuration
    config: IndexConfig,
    /// Map from internal ID to memory key
    id_to_key: Arc<DashMap<usize, String>>,
    /// Map from memory key to memory entry
    key_to_entry: Arc<DashMap<String, MemoryEntry>>,
    /// Map from memory key to internal ID
    key_to_id: Arc<DashMap<String, usize>>,
    /// Next available ID
    next_id: Arc<RwLock<usize>>,
    /// All vectors for rebuild
    vectors: Arc<DashMap<String, Vec<f32>>>,
}

impl HnswIndex {
    /// Create a new HNSW index with the given configuration
    #[instrument(skip_all, fields(dimension = config.dimension))]
    pub fn new(config: IndexConfig) -> Self {
        info!(
            dimension = config.dimension,
            max_connections = config.max_connections,
            ef_construction = config.ef_construction,
            "Creating new HNSW index"
        );

        Self {
            index: Arc::new(RwLock::new(None)),
            config,
            id_to_key: Arc::new(DashMap::new()),
            key_to_entry: Arc::new(DashMap::new()),
            key_to_id: Arc::new(DashMap::new()),
            next_id: Arc::new(RwLock::new(0)),
            vectors: Arc::new(DashMap::new()),
        }
    }

    /// Initialize the HNSW index if not already initialized
    fn ensure_initialized(&self) {
        let index_guard = self.index.read();
        if index_guard.is_none() {
            drop(index_guard);
            let mut index_guard = self.index.write();
            if index_guard.is_none() {
                debug!(
                    "Initializing HNSW index with dimension {}",
                    self.config.dimension
                );
                let hnsw = Hnsw::<f32, DistL2>::new(
                    self.config.max_connections,
                    1000, // Initial capacity, will grow as needed
                    self.config.max_connections_0,
                    self.config.ef_construction,
                    DistL2 {},
                );
                *index_guard = Some(hnsw);
            }
        }
    }

    /// Get the next ID and increment counter
    fn get_next_id(&self) -> usize {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;
        id
    }

    /// Convert distance to similarity score (0.0 to 1.0, higher is better)
    fn distance_to_score(&self, distance: f32) -> f64 {
        match self.config.metric {
            DistanceMetric::Cosine => {
                // For cosine distance: similarity = 1 - distance
                // Since HNSW uses L2, we approximate
                let sim = 1.0 / (1.0 + distance as f64);
                sim.clamp(0.0, 1.0)
            }
            DistanceMetric::Euclidean => {
                // Convert L2 distance to similarity score
                let sim = 1.0 / (1.0 + distance as f64);
                sim.clamp(0.0, 1.0)
            }
            DistanceMetric::Manhattan | DistanceMetric::DotProduct => {
                // Similar conversion for other metrics
                let sim = 1.0 / (1.0 + distance as f64);
                sim.clamp(0.0, 1.0)
            }
        }
    }

    /// Calculate actual cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)) as f64
    }
}

#[async_trait]
impl VectorIndex for HnswIndex {
    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn len(&self) -> usize {
        self.key_to_entry.len()
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }

    #[instrument(skip(self, vector, entry), fields(key = %entry.key))]
    async fn add(&mut self, vector: &[f32], entry: MemoryEntry) -> IndexResult<()> {
        if vector.len() != self.config.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        self.ensure_initialized();

        let key = entry.key.clone();
        let id = self.get_next_id();

        debug!(key = %key, id = id, "Adding vector to HNSW index");

        // Store the vector for potential rebuild
        self.vectors.insert(key.clone(), vector.to_vec());

        // Insert into HNSW
        let index_guard = self.index.read();
        if let Some(ref hnsw) = *index_guard {
            hnsw.insert((vector, id));
        } else {
            return Err(IndexError::BuildFailed(
                "Index not initialized".to_string(),
            ));
        }
        drop(index_guard);

        // Store mappings
        self.id_to_key.insert(id, key.clone());
        self.key_to_id.insert(key.clone(), id);
        self.key_to_entry.insert(key, entry);

        Ok(())
    }

    #[instrument(skip(self, vectors, entries), fields(count = vectors.len()))]
    async fn add_batch(
        &mut self,
        vectors: &[Vec<f32>],
        entries: Vec<MemoryEntry>,
    ) -> IndexResult<()> {
        if vectors.len() != entries.len() {
            return Err(IndexError::BuildFailed(
                "Vectors and entries length mismatch".to_string(),
            ));
        }

        info!("Adding batch of {} vectors", vectors.len());

        for (vector, entry) in vectors.iter().zip(entries.into_iter()) {
            self.add(vector, entry).await?;
        }

        Ok(())
    }

    #[instrument(skip(self, query), fields(k = k))]
    async fn search(&self, query: &[f32], k: usize) -> IndexResult<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        if self.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_initialized();

        debug!("Searching for {} nearest neighbors", k);

        let index_guard = self.index.read();
        let results = if let Some(ref hnsw) = *index_guard {
            hnsw.search(query, k, self.config.ef_search)
        } else {
            return Err(IndexError::QueryFailed("Index not initialized".to_string()));
        };
        drop(index_guard);

        let mut search_results = Vec::new();

        for neighbor in results {
            let id = neighbor.d_id;
            if let Some(key) = self.id_to_key.get(&id) {
                if let Some(entry) = self.key_to_entry.get(key.value()) {
                    let distance = neighbor.distance;

                    // Calculate actual similarity score
                    let score = if let Some(stored_vec) = self.vectors.get(key.value()) {
                        Self::cosine_similarity(query, &stored_vec)
                    } else {
                        self.distance_to_score(distance)
                    };

                    search_results.push(SearchResult::new(entry.clone(), score, distance));
                }
            }
        }

        // Sort by score descending
        search_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        debug!("Found {} results", search_results.len());
        Ok(search_results)
    }

    #[instrument(skip(self, query), fields(k = k, min_score = min_score))]
    async fn search_threshold(
        &self,
        query: &[f32],
        k: usize,
        min_score: f64,
    ) -> IndexResult<Vec<SearchResult>> {
        let results = self.search(query, k).await?;

        // Filter by minimum score
        let filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| r.score >= min_score)
            .collect();

        debug!(
            "Filtered to {} results above threshold {}",
            filtered.len(),
            min_score
        );

        Ok(filtered)
    }

    #[instrument(skip(self), fields(key = %key))]
    async fn remove(&mut self, key: &str) -> IndexResult<bool> {
        if let Some((_, id)) = self.key_to_id.remove(key) {
            self.id_to_key.remove(&id);
            self.key_to_entry.remove(key);
            self.vectors.remove(key);

            // Note: hnsw_rs doesn't support removal, so we need to rebuild
            // We'll mark this for rebuild on next optimization
            warn!(
                key = %key,
                "Vector removed from mappings; full rebuild required for index optimization"
            );

            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[instrument(skip(self))]
    async fn clear(&mut self) -> IndexResult<()> {
        info!("Clearing HNSW index");

        self.id_to_key.clear();
        self.key_to_entry.clear();
        self.key_to_id.clear();
        self.vectors.clear();
        *self.next_id.write() = 0;

        let mut index_guard = self.index.write();
        *index_guard = None;

        Ok(())
    }

    #[instrument(skip(self, path))]
    async fn save<P: AsRef<Path> + Send>(&self, path: P) -> IndexResult<()> {
        let path = path.as_ref();
        info!("Saving HNSW index to {:?}", path);

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| IndexError::SaveFailed(format!("Failed to create directory: {}", e)))?;
        }

        // Serialize the index data
        let index_data = IndexData {
            config: self.config.clone(),
            id_to_key: self.id_to_key.iter().map(|e| (*e.key(), e.value().clone())).collect(),
            key_to_entry: self.key_to_entry.iter().map(|e| (e.key().clone(), e.value().clone())).collect(),
            key_to_id: self.key_to_id.iter().map(|e| (e.key().clone(), *e.value())).collect(),
            next_id: *self.next_id.read(),
            vectors: self.vectors.iter().map(|e| (e.key().clone(), e.value().clone())).collect(),
        };

        let json = serde_json::to_string_pretty(&index_data)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;

        tokio::fs::write(path, json)
            .await
            .map_err(|e| IndexError::SaveFailed(e.to_string()))?;

        info!("HNSW index saved successfully");
        Ok(())
    }

    #[instrument(skip(self, path))]
    async fn load<P: AsRef<Path> + Send>(&mut self, path: P) -> IndexResult<()> {
        let path = path.as_ref();
        info!("Loading HNSW index from {:?}", path);

        let json = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| IndexError::LoadFailed(e.to_string()))?;

        let index_data: IndexData = serde_json::from_str(&json)
            .map_err(|e| IndexError::Serialization(e.to_string()))?;

        // Restore configuration
        self.config = index_data.config;

        // Clear existing data
        self.clear().await?;

        // Restore mappings
        for (id, key) in index_data.id_to_key {
            self.id_to_key.insert(id, key);
        }
        for (key, entry) in index_data.key_to_entry {
            self.key_to_entry.insert(key, entry);
        }
        for (key, id) in index_data.key_to_id {
            self.key_to_id.insert(key, id);
        }
        *self.next_id.write() = index_data.next_id;
        for (key, vec) in index_data.vectors {
            self.vectors.insert(key, vec);
        }

        // Rebuild HNSW from vectors
        self.rebuild().await?;

        info!("HNSW index loaded successfully");
        Ok(())
    }

    #[instrument(skip(self))]
    async fn rebuild(&mut self) -> IndexResult<()> {
        info!("Rebuilding HNSW index from {} vectors", self.vectors.len());

        // Create new HNSW
        let mut index_guard = self.index.write();
        *index_guard = Some(Hnsw::<f32, DistL2>::new(
            self.config.max_connections,
            self.vectors.len().max(1000),
            self.config.max_connections_0,
            self.config.ef_construction,
            DistL2 {},
        ));
        drop(index_guard);

        // Re-insert all vectors
        for entry in self.vectors.iter() {
            let key = entry.key();
            let vector = entry.value();
            if let Some(id) = self.key_to_id.get(key) {
                let index_guard = self.index.read();
                if let Some(ref hnsw) = *index_guard {
                    hnsw.insert((vector.as_slice(), *id));
                }
            }
        }

        info!("HNSW index rebuild complete");
        Ok(())
    }

    fn stats(&self) -> IndexStats {
        let num_vectors = self.len();
        let dimension = self.config.dimension;

        // Estimate memory usage
        let vector_memory = num_vectors * dimension * std::mem::size_of::<f32>();
        let mapping_memory = num_vectors * (std::mem::size_of::<usize>() + 64); // Rough estimate
        let entry_memory = num_vectors * 256; // Rough estimate for MemoryEntry
        let hnsw_memory = num_vectors * self.config.max_connections * std::mem::size_of::<usize>() * 2;

        let memory_bytes = vector_memory + mapping_memory + entry_memory + hnsw_memory;

        // Calculate average connections (approximation)
        let avg_connections = self.config.max_connections as f64;

        IndexStats::new(num_vectors, dimension, self.config.metric)
            .with_memory_estimate(memory_bytes)
            .with_hnsw_stats(4, avg_connections) // HNSW typically has ~4 layers
    }
}

/// Serializable index data for persistence
#[derive(Serialize, Deserialize)]
struct IndexData {
    config: IndexConfig,
    id_to_key: HashMap<usize, String>,
    key_to_entry: HashMap<String, MemoryEntry>,
    key_to_id: HashMap<String, usize>,
    next_id: usize,
    vectors: HashMap<String, Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{MemoryType, Metadata};
    use chrono::Utc;

    fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            key: key.to_string(),
            content: content.to_string(),
            memory_type: MemoryType::ShortTerm,
            metadata: Metadata::default(),
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 0,
        }
    }

    fn create_test_vector(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    #[tokio::test]
    async fn test_hnsw_basic_operations() {
        let config = IndexConfig::new(128);
        let mut index = HnswIndex::new(config);

        assert_eq!(index.dimension(), 128);
        assert!(index.is_empty());

        // Add a vector
        let vec1 = create_test_vector(128, 1.0);
        let entry1 = create_test_entry("key1", "content1");
        index.add(&vec1, entry1).await.expect("Failed to add");

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_search() {
        let config = IndexConfig::new(128).with_ef_search(10);
        let mut index = HnswIndex::new(config);

        // Add test vectors
        for i in 0..10 {
            let vec = create_test_vector(128, i as f32);
            let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
            index.add(&vec, entry).await.expect("Failed to add");
        }

        // Search for similar vector
        let query = create_test_vector(128, 5.0);
        let results = index.search(&query, 3).await.expect("Search failed");

        assert_eq!(results.len(), 3);
        // The closest should be key5
        assert_eq!(results[0].entry.key, "key5");
    }
}
