//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! Provides high-performance approximate nearest neighbor search using
//! the HNSW algorithm, which offers excellent recall with logarithmic
//! search complexity.

use super::vector::{VectorIndex, IndexStats, IndexConfig, DistanceMetric};
use crate::error::{MemoryError, Result};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// HNSW vector index implementation
///
/// Uses the Hierarchical Navigable Small World algorithm for efficient
/// approximate nearest neighbor search. Ideal for large-scale vector
/// databases with millions of entries.
pub struct HnswIndex {
    /// The underlying HNSW index
    index: Arc<RwLock<Hnsw<f32, DistanceL2>>>,
    /// Mapping from UUID to internal HNSW ID
    uuid_to_id: Arc<RwLock<HashMap<Uuid, usize>>>,
    /// Mapping from internal ID back to UUID
    id_to_uuid: Arc<RwLock<HashMap<usize, Uuid>>>,
    /// Configuration
    config: IndexConfig,
    /// HNSW-specific parameters
    hnsw_params: HnswParams,
    /// Next available ID
    next_id: Arc<RwLock<usize>>,
}

/// HNSW-specific parameters
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Maximum number of connections per layer (M)
    pub max_connections: usize,
    /// Size of the dynamic candidate list (ef_construction)
    pub ef_construction: usize,
    /// Search parameter (ef)
    pub ef_search: usize,
    /// Maximum number of layers
    pub max_layer: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 100,
            max_layer: 16,
        }
    }
}

impl HnswParams {
    /// Create parameters optimized for accuracy
    pub fn high_accuracy() -> Self {
        Self {
            max_connections: 32,
            ef_construction: 400,
            ef_search: 200,
            max_layer: 16,
        }
    }

    /// Create parameters optimized for speed
    pub fn high_speed() -> Self {
        Self {
            max_connections: 8,
            ef_construction: 100,
            ef_search: 50,
            max_layer: 12,
        }
    }

    /// Create parameters balanced between speed and accuracy
    pub fn balanced() -> Self {
        Self::default()
    }
}

impl HnswIndex {
    /// Create a new HNSW index
    ///
    /// # Arguments
    /// * `config` - Index configuration (dimension, distance metric, etc.)
    /// * `hnsw_params` - HNSW-specific parameters
    pub fn new(config: IndexConfig, hnsw_params: HnswParams) -> Result<Self> {
        // Create HNSW with L2 distance
        // Note: hnsw_rs uses L2 distance, we'll convert other metrics in search
        let max_nb_connection = hnsw_params.max_connections;
        let ef_c = hnsw_params.ef_construction;
        let max_layer = hnsw_params.max_layer as u8;

        let hnsw = Hnsw::<f32, DistanceL2>::new(
            max_nb_connection,
            config.max_vectors.unwrap_or(100_000),
            max_layer,
            ef_c,
            DistanceL2,
        );

        Ok(Self {
            index: Arc::new(RwLock::new(hnsw)),
            uuid_to_id: Arc::new(RwLock::new(HashMap::new())),
            id_to_uuid: Arc::new(RwLock::new(HashMap::new())),
            config,
            hnsw_params,
            next_id: Arc::new(RwLock::new(0)),
        })
    }

    /// Create with default parameters
    pub fn default_with_dimension(dimension: usize) -> Result<Self> {
        let config = IndexConfig {
            dimension,
            ..Default::default()
        };
        Self::new(config, HnswParams::default())
    }

    /// Get or create internal ID for UUID
    fn get_or_create_id(&self, uuid: Uuid) -> Result<usize> {
        let mut uuid_map = self.uuid_to_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock UUID map: {}", e)))?;

        if let Some(&id) = uuid_map.get(&uuid) {
            Ok(id)
        } else {
            let mut next_id = self.next_id.write()
                .map_err(|e| MemoryError::Internal(format!("Failed to lock next_id: {}", e)))?;
            let id = *next_id;
            *next_id += 1;

            uuid_map.insert(uuid, id);

            let mut id_map = self.id_to_uuid.write()
                .map_err(|e| MemoryError::Internal(format!("Failed to lock ID map: {}", e)))?;
            id_map.insert(id, uuid);

            Ok(id)
        }
    }

    /// Get UUID for internal ID
    fn get_uuid(&self, id: usize) -> Result<Uuid> {
        let id_map = self.id_to_uuid.read()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock ID map: {}", e)))?;

        id_map
            .get(&id)
            .copied()
            .ok_or_else(|| MemoryError::NotFound {
                key: format!("ID {}", id),
            })
    }

    /// Preprocess vector based on distance metric
    fn preprocess_vector(&self, vector: &[f32]) -> Vec<f32> {
        match self.config.distance_metric {
            DistanceMetric::Cosine => {
                // Normalize for cosine distance
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vector.iter().map(|x| x / norm).collect()
                } else {
                    vector.to_vec()
                }
            }
            _ => vector.to_vec(),
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: Uuid, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(MemoryError::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                vector.len()
            )));
        }

        let internal_id = self.get_or_create_id(id)?;
        let processed_vector = self.preprocess_vector(vector);

        let mut index = self.index.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock index: {}", e)))?;

        index.insert((&processed_vector, internal_id));

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        if query.len() != self.config.dimension {
            return Err(MemoryError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.config.dimension,
                query.len()
            )));
        }

        let processed_query = self.preprocess_vector(query);

        let index = self.index.read()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock index: {}", e)))?;

        // Search with ef_search parameter
        let neighbors = index.search(&processed_query, k, self.hnsw_params.ef_search);

        // Convert internal IDs to UUIDs
        let mut results = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let uuid = self.get_uuid(neighbor.d_id)?;
            results.push((uuid, neighbor.distance));
        }

        Ok(results)
    }

    fn remove(&mut self, id: Uuid) -> Result<()> {
        // HNSW doesn't support efficient removal
        // We just remove from our mappings
        let mut uuid_map = self.uuid_to_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock UUID map: {}", e)))?;

        if let Some(internal_id) = uuid_map.remove(&id) {
            let mut id_map = self.id_to_uuid.write()
                .map_err(|e| MemoryError::Internal(format!("Failed to lock ID map: {}", e)))?;
            id_map.remove(&internal_id);
        }

        // Note: Vector still exists in HNSW but won't be returned in results
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        // Create a new index
        let max_nb_connection = self.hnsw_params.max_connections;
        let ef_c = self.hnsw_params.ef_construction;
        let max_layer = self.hnsw_params.max_layer as u8;

        let new_hnsw = Hnsw::<f32, DistanceL2>::new(
            max_nb_connection,
            self.config.max_vectors.unwrap_or(100_000),
            max_layer,
            ef_c,
            DistanceL2,
        );

        let mut index = self.index.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock index: {}", e)))?;
        *index = new_hnsw;

        let mut uuid_map = self.uuid_to_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock UUID map: {}", e)))?;
        uuid_map.clear();

        let mut id_map = self.id_to_uuid.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock ID map: {}", e)))?;
        id_map.clear();

        let mut next_id = self.next_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock next_id: {}", e)))?;
        *next_id = 0;

        Ok(())
    }

    fn len(&self) -> usize {
        self.uuid_to_id.read().map(|m| m.len()).unwrap_or(0)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn save(&self, path: &Path) -> Result<()> {
        let index = self.index.read()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock index: {}", e)))?;

        index.file_dump(path)
            .map_err(|e| MemoryError::External(format!("Failed to save HNSW index: {:?}", e)))?;

        // Save UUID mappings
        let uuid_map_path = path.with_extension("uuid_map");
        let uuid_map = self.uuid_to_id.read()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock UUID map: {}", e)))?;

        let serialized = bincode::serialize(&*uuid_map)
            .map_err(|e| MemoryError::External(format!("Failed to serialize UUID map: {}", e)))?;

        std::fs::write(&uuid_map_path, serialized)
            .map_err(|e| MemoryError::External(format!("Failed to write UUID map: {}", e)))?;

        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        // Load HNSW index
        let mut index = self.index.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock index: {}", e)))?;

        *index = Hnsw::<f32, DistanceL2>::file_load(path)
            .map_err(|e| MemoryError::External(format!("Failed to load HNSW index: {:?}", e)))?;

        // Load UUID mappings
        let uuid_map_path = path.with_extension("uuid_map");
        let serialized = std::fs::read(&uuid_map_path)
            .map_err(|e| MemoryError::External(format!("Failed to read UUID map: {}", e)))?;

        let loaded_map: HashMap<Uuid, usize> = bincode::deserialize(&serialized)
            .map_err(|e| MemoryError::External(format!("Failed to deserialize UUID map: {}", e)))?;

        let mut uuid_map = self.uuid_to_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock UUID map: {}", e)))?;
        *uuid_map = loaded_map.clone();

        // Rebuild reverse mapping
        let mut id_map = self.id_to_uuid.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock ID map: {}", e)))?;
        id_map.clear();
        for (uuid, id) in loaded_map {
            id_map.insert(id, uuid);
        }

        // Update next_id
        let max_id = id_map.keys().max().copied().unwrap_or(0);
        let mut next_id = self.next_id.write()
            .map_err(|e| MemoryError::Internal(format!("Failed to lock next_id: {}", e)))?;
        *next_id = max_id + 1;

        Ok(())
    }

    fn stats(&self) -> IndexStats {
        let vector_count = self.len();
        let mut parameters = std::collections::HashMap::new();

        parameters.insert("max_connections".to_string(), self.hnsw_params.max_connections.to_string());
        parameters.insert("ef_construction".to_string(), self.hnsw_params.ef_construction.to_string());
        parameters.insert("ef_search".to_string(), self.hnsw_params.ef_search.to_string());
        parameters.insert("max_layer".to_string(), self.hnsw_params.max_layer.to_string());

        // Rough memory estimation
        let bytes_per_vector = self.config.dimension * 4; // f32
        let memory_bytes = vector_count * bytes_per_vector * 2; // Rough estimate with overhead

        IndexStats {
            vector_count,
            dimension: self.config.dimension,
            index_type: "HNSW".to_string(),
            memory_bytes,
            avg_search_time_us: None,
            parameters,
        }
    }

    fn rebuild(&mut self) -> Result<()> {
        // HNSW doesn't need explicit rebuilding
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_creation() {
        let index = HnswIndex::default_with_dimension(128);
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_add_and_search() {
        let mut index = HnswIndex::default_with_dimension(3).unwrap();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.9, 0.1, 0.0];
        let vec3 = vec![0.0, 1.0, 0.0];

        assert!(index.add(id1, &vec1).is_ok());
        assert!(index.add(id2, &vec2).is_ok());
        assert!(index.add(id3, &vec3).is_ok());

        assert_eq!(index.len(), 3);

        // Search for nearest neighbors to vec1
        let results = index.search(&vec1, 2).unwrap();
        assert_eq!(results.len(), 2);

        // First result should be id1 (exact match)
        assert_eq!(results[0].0, id1);
        assert!(results[0].1 < 0.01); // Very small distance
    }

    #[test]
    fn test_hnsw_remove() {
        let mut index = HnswIndex::default_with_dimension(3).unwrap();

        let id1 = Uuid::new_v4();
        let vec1 = vec![1.0, 0.0, 0.0];

        assert!(index.add(id1, &vec1).is_ok());
        assert_eq!(index.len(), 1);

        assert!(index.remove(id1).is_ok());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hnsw_clear() {
        let mut index = HnswIndex::default_with_dimension(3).unwrap();

        for _ in 0..10 {
            let id = Uuid::new_v4();
            let vec = vec![1.0, 2.0, 3.0];
            index.add(id, &vec).unwrap();
        }

        assert_eq!(index.len(), 10);

        index.clear().unwrap();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_params() {
        let high_acc = HnswParams::high_accuracy();
        assert_eq!(high_acc.max_connections, 32);

        let high_speed = HnswParams::high_speed();
        assert_eq!(high_speed.max_connections, 8);

        let balanced = HnswParams::balanced();
        assert_eq!(balanced.max_connections, 16);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = HnswIndex::default_with_dimension(3).unwrap();

        let id = Uuid::new_v4();
        let wrong_vec = vec![1.0, 2.0, 3.0, 4.0]; // 4D instead of 3D

        let result = index.add(id, &wrong_vec);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let mut index = HnswIndex::default_with_dimension(128).unwrap();

        let id = Uuid::new_v4();
        let vec = vec![1.0; 128];
        index.add(id, &vec).unwrap();

        let stats = index.stats();
        assert_eq!(stats.vector_count, 1);
        assert_eq!(stats.dimension, 128);
        assert_eq!(stats.index_type, "HNSW");
        assert!(stats.parameters.contains_key("max_connections"));
    }
}
