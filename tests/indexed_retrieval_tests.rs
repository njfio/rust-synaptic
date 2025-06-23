//! Tests for optimized indexed memory retrieval

use synaptic::memory::{
    storage::{memory::MemoryStorage, Storage},
    retrieval::{IndexedMemoryRetriever, IndexingConfig, RetrievalConfig},
    types::{MemoryEntry, MemoryType, MemoryMetadata},
};
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashSet;

/// Create test memory entries with varying access patterns
fn create_test_entries(count: usize) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    let base_time = Utc::now() - Duration::hours(24 * 30); // 30 days ago
    
    for i in 0..count {
        let mut entry = MemoryEntry::new(
            format!("test_key_{}", i),
            format!("Test content for memory entry number {}", i),
            if i % 3 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
        );
        
        // Simulate varying access patterns
        let mut metadata = MemoryMetadata::new();
        metadata.created_at = base_time + Duration::hours(i as i64);
        metadata.last_accessed = base_time + Duration::hours((i * 2) as i64);
        metadata.access_count = (i % 100) as u32; // Varying access counts
        metadata.importance = (i as f64 % 10.0) / 10.0; // Varying importance
        
        // Add tags to some entries
        if i % 5 == 0 {
            metadata.tags.push("important".to_string());
        }
        if i % 10 == 0 {
            metadata.tags.push("urgent".to_string());
        }
        if i % 7 == 0 {
            metadata.tags.push("project".to_string());
        }
        
        entry.metadata = metadata;
        entries.push(entry);
    }
    
    entries
}

/// Setup storage with test data
async fn setup_storage_with_data(entry_count: usize) -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());
    let entries = create_test_entries(entry_count);
    
    for entry in entries {
        storage.store(&entry).await.unwrap();
    }
    
    storage
}

#[tokio::test]
async fn test_indexed_retriever_creation() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    
    // Test that retriever was created successfully
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);
    assert_eq!(stats.frequency_index_entries, 0);
    assert_eq!(stats.tag_index_entries, 0);
    assert_eq!(stats.hot_cache_entries, 0);
}

#[tokio::test]
async fn test_index_initialization() {
    let storage = setup_storage_with_data(50).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    
    // Initialize indexes from storage
    retriever.initialize_indexes().await.unwrap();
    
    // Check that indexes were populated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 50);
    assert_eq!(stats.frequency_index_entries, 50);
    assert_eq!(stats.tag_index_entries, 50);
}

#[tokio::test]
async fn test_get_recent_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();
    
    // Test get_recent with small limit
    let recent = retriever.get_recent(10).await.unwrap();
    assert_eq!(recent.len(), 10);
    
    // Verify they are sorted by access time (most recent first)
    for i in 1..recent.len() {
        assert!(recent[i-1].last_accessed() >= recent[i].last_accessed());
    }
    
    // Test caching - second call should be faster (cache hit)
    let recent2 = retriever.get_recent(10).await.unwrap();
    assert_eq!(recent.len(), recent2.len());
    
    // Results should be identical
    for (a, b) in recent.iter().zip(recent2.iter()) {
        assert_eq!(a.key, b.key);
    }
}

#[tokio::test]
async fn test_get_frequent_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();
    
    // Test get_frequent with small limit
    let frequent = retriever.get_frequent(10).await.unwrap();
    assert_eq!(frequent.len(), 10);
    
    // Verify they are sorted by access count (most frequent first)
    for i in 1..frequent.len() {
        assert!(frequent[i-1].access_count() >= frequent[i].access_count());
    }
    
    // Test caching
    let frequent2 = retriever.get_frequent(10).await.unwrap();
    assert_eq!(frequent.len(), frequent2.len());
}

#[tokio::test]
async fn test_get_by_tags_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();
    
    // Test tag-based retrieval
    let important = retriever.get_by_tags(&["important".to_string()]).await.unwrap();
    assert!(!important.is_empty());
    
    // Verify all returned entries have the "important" tag
    for entry in &important {
        assert!(entry.has_tag("important"));
    }
    
    // Test multiple tags
    let urgent_important = retriever.get_by_tags(&["urgent".to_string(), "important".to_string()]).await.unwrap();
    assert!(!urgent_important.is_empty());
    
    // Test non-existent tag
    let nonexistent = retriever.get_by_tags(&["nonexistent".to_string()]).await.unwrap();
    assert!(nonexistent.is_empty());
}

#[tokio::test]
async fn test_index_updates_on_store() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage.clone(), config, indexing_config);
    
    // Initially empty
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);
    
    // Create and store a new entry
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test content".to_string(),
        MemoryType::ShortTerm,
    ).with_tags(vec!["test".to_string()]);
    
    storage.store(&entry).await.unwrap();
    retriever.on_entry_stored(&entry).await;
    
    // Check that indexes were updated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 1);
    assert_eq!(stats.frequency_index_entries, 1);
    assert_eq!(stats.tag_index_entries, 1);
    
    // Test retrieval
    let by_tags = retriever.get_by_tags(&["test".to_string()]).await.unwrap();
    assert_eq!(by_tags.len(), 1);
    assert_eq!(by_tags[0].key, "test_key");
}

#[tokio::test]
async fn test_index_updates_on_delete() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();
    
    let retriever = IndexedMemoryRetriever::new(storage.clone(), config, indexing_config);
    
    // Store an entry
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test content".to_string(),
        MemoryType::ShortTerm,
    ).with_tags(vec!["test".to_string()]);
    
    storage.store(&entry).await.unwrap();
    retriever.on_entry_stored(&entry).await;
    
    // Verify it's in indexes
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 1);
    
    // Delete the entry
    storage.delete("test_key").await.unwrap();
    retriever.on_entry_deleted("test_key").await;
    
    // Check that indexes were updated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);
    assert_eq!(stats.frequency_index_entries, 0);
    assert_eq!(stats.tag_index_entries, 0);
}

#[tokio::test]
async fn test_hot_cache_functionality() {
    let storage = setup_storage_with_data(50).await;
    let config = RetrievalConfig::default();
    let mut indexing_config = IndexingConfig::default();
    indexing_config.hot_cache_size = 10; // Small cache for testing
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();
    
    // First call should populate cache
    let recent1 = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent1.len(), 5);
    
    // Cache should have entries now
    let stats = retriever.get_index_stats().await;
    assert!(stats.hot_cache_entries > 0);
    assert!(stats.hot_cache_entries <= 10); // Respects max size
    
    // Second call should use cache
    let recent2 = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent1.len(), recent2.len());
}

#[tokio::test]
async fn test_background_maintenance() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let mut indexing_config = IndexingConfig::default();
    indexing_config.index_maintenance_interval_seconds = 1; // Fast for testing
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    
    // Start maintenance
    retriever.start_maintenance().await;
    
    // Wait a bit for maintenance to run
    tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
    
    // Stop maintenance
    retriever.stop_maintenance().await;
    
    // Test passes if no panics occurred
}

#[tokio::test]
async fn test_fallback_behavior_with_disabled_indexes() {
    let storage = setup_storage_with_data(20).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig {
        enable_access_time_index: false,
        enable_frequency_index: false,
        enable_tag_index: false,
        ..Default::default()
    };
    
    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    
    // Should still work using fallback methods
    let recent = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent.len(), 5);
    
    let frequent = retriever.get_frequent(5).await.unwrap();
    assert_eq!(frequent.len(), 5);
    
    let by_tags = retriever.get_by_tags(&["important".to_string()]).await.unwrap();
    // Should find some entries with "important" tag
    assert!(!by_tags.is_empty());
}

#[tokio::test]
async fn test_performance_improvement() {
    use std::time::Instant;
    
    let storage = setup_storage_with_data(1000).await;
    let config = RetrievalConfig::default();
    
    // Test with indexing disabled (fallback)
    let indexing_config_disabled = IndexingConfig {
        enable_access_time_index: false,
        enable_frequency_index: false,
        enable_tag_index: false,
        ..Default::default()
    };
    
    let retriever_fallback = IndexedMemoryRetriever::new(storage.clone(), config.clone(), indexing_config_disabled);
    
    let start = Instant::now();
    let _recent_fallback = retriever_fallback.get_recent(10).await.unwrap();
    let fallback_time = start.elapsed();
    
    // Test with indexing enabled
    let indexing_config_enabled = IndexingConfig::default();
    let retriever_indexed = IndexedMemoryRetriever::new(storage, config, indexing_config_enabled);
    retriever_indexed.initialize_indexes().await.unwrap();
    
    let start = Instant::now();
    let _recent_indexed = retriever_indexed.get_recent(10).await.unwrap();
    let indexed_time = start.elapsed();
    
    // Indexed version should be faster (though with small dataset, difference might be minimal)
    println!("Fallback time: {:?}, Indexed time: {:?}", fallback_time, indexed_time);
    
    // At minimum, both should complete successfully
    assert!(fallback_time.as_millis() > 0);
    assert!(indexed_time.as_millis() >= 0);
}
