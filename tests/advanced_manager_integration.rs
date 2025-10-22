//! Integration tests for AdvancedMemoryManager with real storage backends
//!
//! These tests verify that AdvancedMemoryManager correctly integrates with
//! storage backends and performs operations on the real storage instead of
//! creating isolated instances.

use synaptic::memory::management::{AdvancedMemoryManager, MemoryManagementConfig};
use synaptic::memory::storage::{Storage, memory::MemoryStorage};
use synaptic::memory::{MemoryEntry, MemoryType};
use std::sync::Arc;

#[tokio::test]
async fn test_manager_stores_to_real_storage() {
    // Create a shared storage backend
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();

    // Create manager with the storage
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory through the manager
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "test_value".to_string(),
        MemoryType::ShortTerm,
    );

    manager.add_memory(entry.clone(), None).await.unwrap();

    // Verify it's actually in the storage
    let retrieved = storage.retrieve("test_key").await.unwrap();
    assert!(retrieved.is_some(), "Memory should be in storage");
    assert_eq!(retrieved.unwrap().content, "test_value");
}

#[tokio::test]
async fn test_manager_update_fetches_real_entry() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Store a LongTerm memory directly in storage
    let entry = MemoryEntry::new(
        "long_term_key".to_string(),
        "original_value".to_string(),
        MemoryType::LongTerm,  // Important: LongTerm type
    );
    storage.store(&entry).await.unwrap();

    // Update through manager
    manager.update_memory("long_term_key", "updated_value".to_string(), None)
        .await
        .unwrap();

    // Verify the update preserved the LongTerm type
    let updated = storage.retrieve("long_term_key").await.unwrap().unwrap();
    assert_eq!(updated.content, "updated_value");
    assert_eq!(updated.memory_type, MemoryType::LongTerm,
        "CRITICAL: Update should preserve original memory type");
}

#[tokio::test]
async fn test_manager_update_preserves_metadata() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Store a memory with specific metadata
    let mut entry = MemoryEntry::new(
        "metadata_key".to_string(),
        "original_content".to_string(),
        MemoryType::LongTerm,
    );
    entry.importance = 0.9;
    entry.tags = vec!["important".to_string(), "urgent".to_string()];

    storage.store(&entry).await.unwrap();

    let original_created_at = entry.created_at;

    // Update through manager
    manager.update_memory("metadata_key", "new_content".to_string(), None)
        .await
        .unwrap();

    // Verify metadata is preserved
    let updated = storage.retrieve("metadata_key").await.unwrap().unwrap();
    assert_eq!(updated.content, "new_content");
    assert_eq!(updated.importance, 0.9, "Importance should be preserved");
    assert_eq!(updated.tags, vec!["important", "urgent"], "Tags should be preserved");
    assert_eq!(updated.created_at, original_created_at, "Created timestamp should be preserved");
    assert_eq!(updated.memory_type, MemoryType::LongTerm, "Memory type should be preserved");
}

#[tokio::test]
async fn test_manager_update_nonexistent_fails() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Try to update non-existent memory
    let result = manager.update_memory("nonexistent", "value".to_string(), None).await;

    assert!(result.is_err(), "Updating non-existent memory should fail");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("non-existent") || err_msg.contains("not found"),
        "Error should indicate memory not found: {}", err_msg);
}

#[tokio::test]
async fn test_manager_delete_removes_from_storage() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory
    let entry = MemoryEntry::new(
        "delete_test".to_string(),
        "to_be_deleted".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry).await.unwrap();

    // Verify it exists
    assert!(storage.retrieve("delete_test").await.unwrap().is_some());

    // Delete through manager
    let result = manager.delete_memory("delete_test", None).await.unwrap();

    assert!(result.success, "Delete operation should succeed");
    assert_eq!(result.affected_count, 1, "Should affect 1 memory");

    // Verify it's actually deleted from storage
    let retrieved = storage.retrieve("delete_test").await.unwrap();
    assert!(retrieved.is_none(), "Memory should be deleted from storage");
}

#[tokio::test]
async fn test_manager_delete_nonexistent_fails() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Try to delete non-existent memory
    let result = manager.delete_memory("nonexistent", None).await;

    assert!(result.is_err(), "Deleting non-existent memory should fail");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("non-existent") || err_msg.contains("not found"),
        "Error should indicate memory not found: {}", err_msg);
}

#[tokio::test]
async fn test_manager_delete_fetches_real_entry_for_tracking() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Store a LongTerm memory with metadata
    let mut entry = MemoryEntry::new(
        "tracked_delete".to_string(),
        "important content".to_string(),
        MemoryType::LongTerm,
    );
    entry.importance = 0.95;
    storage.store(&entry).await.unwrap();

    // Delete through manager
    let result = manager.delete_memory("tracked_delete", None).await.unwrap();

    // Verify result contains information from the real entry
    let result_data = result.result_data.unwrap();
    assert_eq!(result_data["deleted_memory_key"], "tracked_delete");
    assert_eq!(result_data["original_memory_type"], "LongTerm");
    assert_eq!(result_data["original_value_length"], "important content".len());
}

#[tokio::test]
async fn test_multiple_managers_share_storage() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();

    // Create two managers sharing the same storage
    let mut manager1 = AdvancedMemoryManager::new(Arc::clone(&storage), config.clone());
    let mut manager2 = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Manager 1 adds a memory
    let entry1 = MemoryEntry::new(
        "shared_key_1".to_string(),
        "from_manager1".to_string(),
        MemoryType::ShortTerm,
    );
    manager1.add_memory(entry1, None).await.unwrap();

    // Manager 2 should be able to update it
    manager2.update_memory("shared_key_1", "updated_by_manager2".to_string(), None)
        .await
        .unwrap();

    // Verify both managers see the same data
    let retrieved = storage.retrieve("shared_key_1").await.unwrap().unwrap();
    assert_eq!(retrieved.content, "updated_by_manager2");
}

#[tokio::test]
async fn test_manager_add_multiple_memories() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add multiple memories
    for i in 0..10 {
        let entry = MemoryEntry::new(
            format!("key_{}", i),
            format!("value_{}", i),
            if i % 2 == 0 { MemoryType::ShortTerm } else { MemoryType::LongTerm },
        );
        manager.add_memory(entry, None).await.unwrap();
    }

    // Verify all are in storage
    for i in 0..10 {
        let retrieved = storage.retrieve(&format!("key_{}", i)).await.unwrap();
        assert!(retrieved.is_some(), "Memory {} should exist", i);
        assert_eq!(retrieved.unwrap().content, format!("value_{}", i));
    }

    // Verify count
    let count = storage.count().await.unwrap();
    assert_eq!(count, 10, "Storage should contain all 10 memories");
}

#[tokio::test]
async fn test_manager_lifecycle_integration() {
    let storage = Arc::new(MemoryStorage::new());
    let mut config = MemoryManagementConfig::default();
    config.enable_lifecycle_management = true;

    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory with lifecycle tracking
    let entry = MemoryEntry::new(
        "lifecycle_test".to_string(),
        "content".to_string(),
        MemoryType::LongTerm,
    );

    let result = manager.add_memory(entry, None).await.unwrap();

    // Verify lifecycle tracking was performed
    assert!(result.messages.iter().any(|m| m.contains("Lifecycle")),
        "Lifecycle tracking should be mentioned in messages");
}

#[tokio::test]
async fn test_manager_analytics_integration() {
    let storage = Arc::new(MemoryStorage::new());
    let mut config = MemoryManagementConfig::default();
    config.enable_analytics = true;

    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory with analytics
    let entry = MemoryEntry::new(
        "analytics_test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );

    let result = manager.add_memory(entry, None).await.unwrap();

    // Verify analytics was recorded
    assert!(result.messages.iter().any(|m| m.contains("Analytics")),
        "Analytics should be mentioned in messages");
}

#[tokio::test]
async fn test_manager_temporal_tracking() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory
    let entry = MemoryEntry::new(
        "temporal_test".to_string(),
        "original".to_string(),
        MemoryType::LongTerm,
    );
    manager.add_memory(entry, None).await.unwrap();

    // Update it
    manager.update_memory("temporal_test", "updated".to_string(), None)
        .await
        .unwrap();

    // Delete it
    let result = manager.delete_memory("temporal_test", None).await.unwrap();

    // Verify temporal tracking happened (version_id in result)
    let result_data = result.result_data.unwrap();
    assert!(result_data.get("version_id").is_some(),
        "Version ID should be tracked for temporal changes");
}

#[tokio::test]
async fn test_manager_update_multiple_times() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add initial memory
    let entry = MemoryEntry::new(
        "multi_update".to_string(),
        "version_1".to_string(),
        MemoryType::LongTerm,
    );
    manager.add_memory(entry, None).await.unwrap();

    // Perform multiple updates
    for i in 2..=5 {
        manager.update_memory("multi_update", format!("version_{}", i), None)
            .await
            .unwrap();
    }

    // Verify final state
    let final_entry = storage.retrieve("multi_update").await.unwrap().unwrap();
    assert_eq!(final_entry.content, "version_5");
    assert_eq!(final_entry.memory_type, MemoryType::LongTerm,
        "Memory type should remain consistent through all updates");
}

#[tokio::test]
async fn test_manager_preserves_access_count() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();
    let mut manager = AdvancedMemoryManager::new(Arc::clone(&storage), config);

    // Add a memory
    let entry = MemoryEntry::new(
        "access_test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry).await.unwrap();

    // Access it multiple times through retrieve
    for _ in 0..3 {
        storage.retrieve("access_test").await.unwrap();
    }

    let before_update = storage.retrieve("access_test").await.unwrap().unwrap();
    let initial_access_count = before_update.access_count;

    // Update through manager
    manager.update_memory("access_test", "updated".to_string(), None)
        .await
        .unwrap();

    // Verify access count was updated (incremented)
    let after_update = storage.retrieve("access_test").await.unwrap().unwrap();
    assert!(after_update.access_count > initial_access_count,
        "Access count should be incremented through update");
}

#[tokio::test]
async fn test_manager_concurrent_operations() {
    let storage = Arc::new(MemoryStorage::new());
    let config = MemoryManagementConfig::default();

    // Pre-populate storage
    for i in 0..20 {
        let entry = MemoryEntry::new(
            format!("concurrent_{}", i),
            format!("value_{}", i),
            MemoryType::ShortTerm,
        );
        storage.store(&entry).await.unwrap();
    }

    // Create manager and perform concurrent updates
    let manager = Arc::new(tokio::sync::Mutex::new(
        AdvancedMemoryManager::new(Arc::clone(&storage), config)
    ));

    let mut handles = vec![];
    for i in 0..20 {
        let mgr = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let mut m = mgr.lock().await;
            m.update_memory(
                &format!("concurrent_{}", i),
                format!("updated_{}", i),
                None
            ).await
        });
        handles.push(handle);
    }

    // Wait for all updates
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Verify all updates succeeded
    for i in 0..20 {
        let retrieved = storage.retrieve(&format!("concurrent_{}", i))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.content, format!("updated_{}", i));
    }
}
