//! Integration tests for the AI Agent Memory system

use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType, StorageBackend,
    memory::types::MemoryMetadata,
    error::{MemoryError, Result},
};
use tempfile::TempDir;
use uuid::Uuid;

#[tokio::test]
async fn test_basic_memory_operations() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Test storing and retrieving
    memory.store("test_key", "test_value").await?;
    let retrieved = memory.retrieve("test_key").await?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().value, "test_value");

    // Test non-existent key
    let non_existent = memory.retrieve("non_existent").await?;
    assert!(non_existent.is_none());

    Ok(())
}

#[tokio::test]
async fn test_memory_search() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store test data
    memory.store("coffee_preference", "I love coffee in the morning").await?;
    memory.store("tea_preference", "Tea is great for afternoon").await?;
    memory.store("water_intake", "Drink 8 glasses of water daily").await?;

    // Search for coffee
    let results = memory.search("coffee", 10).await?;
    assert_eq!(results.len(), 1);
    assert!(results[0].entry.value.contains("coffee"));

    // Search for a common word
    let results = memory.search("great", 10).await?;
    assert_eq!(results.len(), 1);
    assert!(results[0].entry.value.contains("great"));

    Ok(())
}

#[tokio::test]
async fn test_checkpointing() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store initial state
    memory.store("initial_key", "initial_value").await?;
    memory.store("shared_key", "original_value").await?;

    // Create checkpoint
    let checkpoint_id = memory.checkpoint().await?;

    // Modify state
    memory.store("new_key", "new_value").await?;
    memory.store("shared_key", "modified_value").await?;

    // Verify modified state
    let modified = memory.retrieve("shared_key").await?;
    assert_eq!(modified.unwrap().value, "modified_value");
    
    let new_entry = memory.retrieve("new_key").await?;
    assert!(new_entry.is_some());

    // Restore checkpoint
    memory.restore_checkpoint(checkpoint_id).await?;

    // Verify restored state
    let restored = memory.retrieve("shared_key").await?;
    assert_eq!(restored.unwrap().value, "original_value");
    
    let missing_entry = memory.retrieve("new_key").await?;
    assert!(missing_entry.is_none());

    Ok(())
}

#[tokio::test]
async fn test_file_storage_persistence() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    // Create first memory instance
    {
        let config = MemoryConfig {
            storage_backend: StorageBackend::File {
                path: db_path.to_string_lossy().to_string(),
            },
            ..Default::default()
        };
        let mut memory = AgentMemory::new(config).await?;
        memory.store("persistent_key", "persistent_value").await?;
    }

    // Create second memory instance with same path
    {
        let config = MemoryConfig {
            storage_backend: StorageBackend::File {
                path: db_path.to_string_lossy().to_string(),
            },
            ..Default::default()
        };
        let mut memory = AgentMemory::new(config).await?;
        let retrieved = memory.retrieve("persistent_key").await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "persistent_value");
    }

    Ok(())
}

#[tokio::test]
async fn test_memory_statistics() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Initial stats should be empty
    let initial_stats = memory.stats();
    assert_eq!(initial_stats.short_term_count, 0);
    assert_eq!(initial_stats.long_term_count, 0);

    // Add some memories
    memory.store("key1", "value1").await?;
    memory.store("key2", "value2").await?;
    memory.store("key3", "value3").await?;

    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 3); // Default is short-term
    assert_eq!(stats.long_term_count, 0);
    assert!(stats.total_size > 0);

    Ok(())
}

#[tokio::test]
async fn test_memory_clearing() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Add some memories
    memory.store("key1", "value1").await?;
    memory.store("key2", "value2").await?;

    // Verify they exist
    assert!(memory.retrieve("key1").await?.is_some());
    assert!(memory.retrieve("key2").await?.is_some());

    // Clear all memories
    memory.clear().await?;

    // Verify they're gone
    assert!(memory.retrieve("key1").await?.is_none());
    assert!(memory.retrieve("key2").await?.is_none());

    // Stats should be reset
    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 0);
    assert_eq!(stats.long_term_count, 0);

    Ok(())
}

#[tokio::test]
async fn test_memory_entry_metadata() -> Result<()> {
    let mut entry = MemoryEntry::new(
        "test_key".to_string(),
        "test_value".to_string(),
        MemoryType::LongTerm,
    );

    // Test metadata operations
    entry.metadata = entry.metadata
        .with_tags(vec!["tag1".to_string(), "tag2".to_string()])
        .with_importance(0.8);

    assert!(entry.has_tag("tag1"));
    assert!(entry.has_tag("tag2"));
    assert!(!entry.has_tag("tag3"));
    assert_eq!(entry.metadata.importance, 0.8);

    // Test custom fields
    entry.metadata.set_custom_field("category".to_string(), "test".to_string());
    assert_eq!(entry.metadata.get_custom_field("category"), Some(&"test".to_string()));
    assert_eq!(entry.metadata.get_custom_field("missing"), None);

    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Test restoring from non-existent checkpoint
    let fake_id = Uuid::new_v4();
    let result = memory.restore_checkpoint(fake_id).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        MemoryError::NotFound { .. } => {}, // Expected
        other => assert!(false, "Expected NotFound error, got: {:?}", other),
    }

    Ok(())
}

#[tokio::test]
async fn test_concurrent_access() -> Result<()> {
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio::task::JoinSet;

    let config = MemoryConfig::default();
    let memory = Arc::new(Mutex::new(AgentMemory::new(config).await?));

    let mut join_set = JoinSet::new();

    // Spawn multiple tasks that store memories concurrently
    for i in 0..10 {
        let memory_clone = Arc::clone(&memory);
        join_set.spawn(async move {
            let key = format!("concurrent_key_{}", i);
            let value = format!("concurrent_value_{}", i);
            let mut mem = memory_clone.lock().await;
            mem.store(&key, &value).await
        });
    }

    // Wait for all tasks to complete
    while let Some(result) = join_set.join_next().await {
        result.unwrap()?; // Unwrap the JoinResult and then the Result<()>
    }

    // Verify all memories were stored
    for i in 0..10 {
        let key = format!("concurrent_key_{}", i);
        let mut mem = memory.lock().await;
        let retrieved = mem.retrieve(&key).await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, format!("concurrent_value_{}", i));
    }

    Ok(())
}

#[tokio::test]
async fn test_memory_types() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store different types of memories
    memory.store("short_term_key", "short_term_value").await?; // Default is short-term
    
    // For now, we'll test with the basic store method
    // In a full implementation, we'd have methods to specify memory type
    
    let stats = memory.stats();
    assert!(stats.short_term_count > 0);

    Ok(())
}

#[tokio::test]
async fn test_large_memory_values() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Test with a large value
    let large_value = "x".repeat(10000); // 10KB string
    memory.store("large_key", &large_value).await?;

    let retrieved = memory.retrieve("large_key").await?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().value.len(), 10000);

    Ok(())
}

#[tokio::test]
async fn test_special_characters() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Test with special characters and Unicode
    let special_value = "Hello 🌍! Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?";
    memory.store("special_key", special_value).await?;

    let retrieved = memory.retrieve("special_key").await?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().value, special_value);

    Ok(())
}

#[tokio::test]
async fn test_temporal_metrics_accumulation() -> Result<()> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    memory.store("metric1", "v1").await?;
    memory.store("metric1", "v2").await?;
    memory.store("metric2", "v1").await?;

    let usage = memory
        .get_temporal_usage_stats()
        .await
        .expect("usage stats");
    assert!(usage.total_versions >= 3);
    assert!(usage.total_diffs >= 1);

    let diff_metrics = memory.get_temporal_diff_metrics().expect("diff metrics");
    assert!(diff_metrics.total_diffs >= 1);

    Ok(())
}

#[cfg(feature = "analytics")]
#[tokio::test]
async fn test_analytics_metrics_access() -> Result<()> {
    let mut config = MemoryConfig::default();
    config.enable_analytics = true;
    let mut memory = AgentMemory::new(config).await?;

    memory.store("a_key", "a_val").await?;

    let metrics = memory
        .get_analytics_metrics()
        .expect("analytics metrics");
    assert!(metrics.events_processed >= 1);
    Ok(())
}
