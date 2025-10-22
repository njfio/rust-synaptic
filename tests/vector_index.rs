//! Comprehensive tests for vector index (ANN) functionality
//!
//! Tests cover:
//! - HNSW index operations
//! - Index persistence and loading
//! - Search accuracy and performance
//! - Index manager lifecycle hooks
//! - Integration with embeddings

use synaptic::memory::embeddings::{EmbeddingProvider, TfIdfProvider};
use synaptic::memory::indexing::{
    DistanceMetric, HnswIndex, IndexConfig, IndexManager, VectorIndex,
};
use synaptic::memory::types::{MemoryEntry, MemoryType, Metadata};
use chrono::Utc;
use std::sync::Arc;
use tempfile::TempDir;

// Helper function to create test memory entries
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

// Helper to create a test vector
fn create_test_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| seed + (i as f32 * 0.01)).collect()
}

#[tokio::test]
async fn test_hnsw_index_creation() {
    let config = IndexConfig::new(128);
    let index = HnswIndex::new(config);

    assert_eq!(index.dimension(), 128);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.metric(), DistanceMetric::Cosine);
}

#[tokio::test]
async fn test_hnsw_add_vector() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    let vector = create_test_vector(128, 1.0);
    let entry = create_test_entry("key1", "content1");

    let result = index.add(&vector, entry).await;
    assert!(result.is_ok());
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[tokio::test]
async fn test_hnsw_add_multiple_vectors() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    for i in 0..10 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    assert_eq!(index.len(), 10);
}

#[tokio::test]
async fn test_hnsw_dimension_validation() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Try to add vector with wrong dimension
    let wrong_vector = create_test_vector(64, 1.0); // Wrong dimension
    let entry = create_test_entry("key1", "content1");

    let result = index.add(&wrong_vector, entry).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_hnsw_search() {
    let config = IndexConfig::new(128).with_ef_search(50);
    let mut index = HnswIndex::new(config);

    // Add test vectors with known similarities
    for i in 0..20 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    // Search for a vector similar to key10
    let query = create_test_vector(128, 10.0);
    let results = index.search(&query, 5).await.expect("Search failed");

    assert!(!results.is_empty());
    assert!(results.len() <= 5);

    // The closest result should be key10
    assert_eq!(results[0].entry.key, "key10");
}

#[tokio::test]
async fn test_hnsw_search_threshold() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Add vectors
    for i in 0..10 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    // Search with high threshold
    let query = create_test_vector(128, 5.0);
    let results = index
        .search_threshold(&query, 10, 0.9)
        .await
        .expect("Search failed");

    // High threshold should filter out most results
    assert!(results.len() <= 3);

    // All results should meet threshold
    for result in results {
        assert!(result.score >= 0.9);
    }
}

#[tokio::test]
async fn test_hnsw_batch_add() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    let mut vectors = Vec::new();
    let mut entries = Vec::new();

    for i in 0..50 {
        vectors.push(create_test_vector(128, i as f32));
        entries.push(create_test_entry(&format!("key{}", i), &format!("content{}", i)));
    }

    index
        .add_batch(&vectors, entries)
        .await
        .expect("Batch add failed");

    assert_eq!(index.len(), 50);
}

#[tokio::test]
async fn test_hnsw_remove() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Add vectors
    for i in 0..5 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    assert_eq!(index.len(), 5);

    // Remove one
    let removed = index.remove("key2").await.expect("Remove failed");
    assert!(removed);
    assert_eq!(index.len(), 4);

    // Try to remove non-existent
    let not_removed = index.remove("nonexistent").await.expect("Remove failed");
    assert!(!not_removed);
}

#[tokio::test]
async fn test_hnsw_clear() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Add vectors
    for i in 0..10 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    assert_eq!(index.len(), 10);

    // Clear
    index.clear().await.expect("Clear failed");
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[tokio::test]
async fn test_hnsw_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("test_index.json");

    // Create and populate index
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config.clone());

    for i in 0..20 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    // Save
    index
        .save(&index_path)
        .await
        .expect("Failed to save index");

    // Create new index and load
    let mut new_index = HnswIndex::new(config);
    new_index
        .load(&index_path)
        .await
        .expect("Failed to load index");

    // Verify loaded index
    assert_eq!(new_index.len(), 20);

    // Test search on loaded index
    let query = create_test_vector(128, 10.0);
    let results = new_index.search(&query, 3).await.expect("Search failed");
    assert!(!results.is_empty());
    assert_eq!(results[0].entry.key, "key10");
}

#[tokio::test]
async fn test_hnsw_rebuild() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Add vectors
    for i in 0..10 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    // Remove some vectors (creates fragmentation)
    index.remove("key2").await.expect("Remove failed");
    index.remove("key5").await.expect("Remove failed");

    assert_eq!(index.len(), 8);

    // Rebuild
    index.rebuild().await.expect("Rebuild failed");

    // Verify still works
    let query = create_test_vector(128, 3.0);
    let results = index.search(&query, 3).await.expect("Search failed");
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_hnsw_stats() {
    let config = IndexConfig::new(128);
    let mut index = HnswIndex::new(config);

    // Initially empty
    let stats = index.stats();
    assert_eq!(stats.num_vectors, 0);
    assert_eq!(stats.dimension, 128);

    // Add vectors
    for i in 0..100 {
        let vector = create_test_vector(128, i as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    let stats = index.stats();
    assert_eq!(stats.num_vectors, 100);
    assert_eq!(stats.dimension, 128);
    assert!(stats.memory_bytes > 0);
    assert!(stats.num_layers.is_some());
    assert!(stats.avg_connections.is_some());
}

#[tokio::test]
async fn test_index_manager_creation() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    assert_eq!(manager.deletion_count(), 0);
    let stats = manager.stats();
    assert_eq!(stats.num_vectors, 0);
}

#[tokio::test]
async fn test_index_manager_add_hook() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    let entry = create_test_entry("key1", "test content for indexing");
    manager
        .on_memory_added(&entry)
        .await
        .expect("Failed to add");

    let stats = manager.stats();
    assert_eq!(stats.num_vectors, 1);
}

#[tokio::test]
async fn test_index_manager_update_hook() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    let old_entry = create_test_entry("key1", "old content");
    let new_entry = create_test_entry("key1", "new updated content");

    manager
        .on_memory_added(&old_entry)
        .await
        .expect("Failed to add");

    // Should still have 1 entry after update
    manager
        .on_memory_updated(&old_entry, &new_entry)
        .await
        .expect("Failed to update");

    let stats = manager.stats();
    assert_eq!(stats.num_vectors, 1);
}

#[tokio::test]
async fn test_index_manager_delete_hook() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider).with_auto_rebuild(false);

    let entry = create_test_entry("key1", "test content");
    manager
        .on_memory_added(&entry)
        .await
        .expect("Failed to add");

    assert_eq!(manager.stats().num_vectors, 1);

    manager
        .on_memory_deleted(&entry)
        .await
        .expect("Failed to delete");

    assert_eq!(manager.deletion_count(), 1);
}

#[tokio::test]
async fn test_index_manager_batch_add() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    let entries: Vec<MemoryEntry> = (0..10)
        .map(|i| create_test_entry(&format!("key{}", i), &format!("content {}", i)))
        .collect();

    manager
        .on_batch_added(&entries)
        .await
        .expect("Batch add failed");

    let stats = manager.stats();
    assert_eq!(stats.num_vectors, 10);
}

#[tokio::test]
async fn test_index_manager_auto_rebuild() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider)
        .with_auto_rebuild(true)
        .with_deletion_threshold(3);

    // Add and delete to trigger rebuild
    for i in 0..5 {
        let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
        manager
            .on_memory_added(&entry)
            .await
            .expect("Failed to add");
        manager
            .on_memory_deleted(&entry)
            .await
            .expect("Failed to delete");
    }

    // After 3 deletions, should rebuild and reset counter
    // So we should have 2 more deletions counted
    assert_eq!(manager.deletion_count(), 2);
}

#[tokio::test]
async fn test_index_manager_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("manager_index.json");

    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    // Add some entries
    for i in 0..5 {
        let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
        manager
            .on_memory_added(&entry)
            .await
            .expect("Failed to add");
    }

    // Save
    manager.save(&index_path).await.expect("Failed to save");

    // Create new manager and load
    let config2 = IndexConfig::new(128);
    let index2 = Box::new(HnswIndex::new(config2));
    let provider2 = Arc::new(TfIdfProvider::default());
    let manager2 = IndexManager::new(index2, provider2);

    manager2.load(&index_path).await.expect("Failed to load");

    // Verify loaded
    let stats = manager2.stats();
    assert_eq!(stats.num_vectors, 5);
}

#[tokio::test]
async fn test_index_manager_clear() {
    let config = IndexConfig::new(128);
    let index = Box::new(HnswIndex::new(config));
    let provider = Arc::new(TfIdfProvider::default());
    let manager = IndexManager::new(index, provider);

    // Add entries
    for i in 0..5 {
        let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
        manager
            .on_memory_added(&entry)
            .await
            .expect("Failed to add");
    }

    assert_eq!(manager.stats().num_vectors, 5);

    // Clear
    manager.clear().await.expect("Failed to clear");

    assert_eq!(manager.stats().num_vectors, 0);
    assert_eq!(manager.deletion_count(), 0);
}

#[tokio::test]
async fn test_different_distance_metrics() {
    for metric in &[
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ] {
        let config = IndexConfig::new(128).with_metric(*metric);
        let mut index = HnswIndex::new(config);

        // Add test data
        for i in 0..10 {
            let vector = create_test_vector(128, i as f32);
            let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
            index.add(&vector, entry).await.expect("Failed to add");
        }

        // Search should work with any metric
        let query = create_test_vector(128, 5.0);
        let results = index.search(&query, 3).await.expect("Search failed");
        assert!(!results.is_empty());
    }
}

#[tokio::test]
async fn test_large_scale_index() {
    let config = IndexConfig::new(128)
        .with_ef_construction(100)
        .with_ef_search(50);
    let mut index = HnswIndex::new(config);

    // Add 1000 vectors
    for i in 0..1000 {
        let vector = create_test_vector(128, (i % 100) as f32);
        let entry = create_test_entry(&format!("key{}", i), &format!("content{}", i));
        index.add(&vector, entry).await.expect("Failed to add");
    }

    assert_eq!(index.len(), 1000);

    // Search should still be fast and accurate
    let query = create_test_vector(128, 50.0);
    let results = index.search(&query, 10).await.expect("Search failed");
    assert_eq!(results.len(), 10);

    // Verify results are ranked
    for i in 1..results.len() {
        assert!(results[i - 1].score >= results[i].score);
    }
}
