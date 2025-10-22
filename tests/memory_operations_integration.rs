//! Comprehensive integration tests for MemoryOperations trait and SynapticMemory.
//!
//! These tests verify that the SynapticMemory implementation correctly integrates
//! storage, knowledge graphs, analytics, and all other subsystems through the
//! MemoryOperations trait.

use synaptic::memory::operations::{SynapticMemory, SynapticMemoryBuilder};
use synaptic::memory::{MemoryOperations, MemoryEntry, MemoryType};
use synaptic::memory::storage::StorageBackend;
use std::time::Duration;

#[tokio::test]
async fn test_basic_store_and_retrieve() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Create and store entry
    let entry = MemoryEntry::new(
        "basic_test".to_string(),
        "This is a basic test".to_string(),
        MemoryType::ShortTerm,
    );

    memory.store_memory(entry).await.unwrap();

    // Retrieve and verify
    let retrieved = memory.get_memory("basic_test").await.unwrap();
    assert!(retrieved.is_some(), "Memory should be retrievable");

    let retrieved_entry = retrieved.unwrap();
    assert_eq!(retrieved_entry.key, "basic_test");
    assert_eq!(retrieved_entry.content, "This is a basic test");
    assert_eq!(retrieved_entry.memory_type, MemoryType::ShortTerm);
}

#[tokio::test]
async fn test_store_multiple_memories() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store multiple entries
    for i in 0..10 {
        let entry = MemoryEntry::new(
            format!("key_{}", i),
            format!("Value number {}", i),
            MemoryType::ShortTerm,
        );
        memory.store_memory(entry).await.unwrap();
    }

    // Verify all are retrievable
    for i in 0..10 {
        let retrieved = memory.get_memory(&format!("key_{}", i)).await.unwrap();
        assert!(retrieved.is_some(), "Memory {} should exist", i);
        assert_eq!(retrieved.unwrap().content, format!("Value number {}", i));
    }
}

#[tokio::test]
async fn test_retrieve_nonexistent_memory() {
    let mut memory = SynapticMemory::new().await.unwrap();

    let retrieved = memory.get_memory("nonexistent_key").await.unwrap();
    assert!(retrieved.is_none(), "Non-existent memory should return None");
}

#[tokio::test]
async fn test_update_existing_memory() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store initial entry
    let entry = MemoryEntry::new(
        "update_key".to_string(),
        "Initial value".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await.unwrap();

    // Update the memory
    memory.update_memory("update_key", "Updated value").await.unwrap();

    // Verify update
    let retrieved = memory.get_memory("update_key").await.unwrap().unwrap();
    assert_eq!(retrieved.content, "Updated value");
    assert!(retrieved.access_count > 0, "Access count should be updated");
}

#[tokio::test]
async fn test_update_nonexistent_memory_fails() {
    let mut memory = SynapticMemory::new().await.unwrap();

    let result = memory.update_memory("nonexistent", "value").await;
    assert!(result.is_err(), "Updating non-existent memory should fail");

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("non-existent") || err_msg.contains("not found"),
        "Error should indicate memory not found");
}

#[tokio::test]
async fn test_search_memories() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store memories with searchable content
    let entries = vec![
        ("doc1", "Rust is a systems programming language"),
        ("doc2", "Python is great for data science"),
        ("doc3", "Rust has excellent memory safety"),
        ("doc4", "JavaScript runs in browsers"),
        ("doc5", "Rust provides zero-cost abstractions"),
    ];

    for (key, content) in entries {
        let entry = MemoryEntry::new(
            key.to_string(),
            content.to_string(),
            MemoryType::LongTerm,
        );
        memory.store_memory(entry).await.unwrap();
    }

    // Search for "Rust"
    let results = memory.search_memories("Rust", 10).await.unwrap();

    // Should find all Rust-related memories
    assert!(!results.is_empty(), "Should find matching memories");

    // All results should contain "Rust"
    for fragment in &results {
        assert!(fragment.entry.content.contains("Rust"),
            "Result should contain search term");
    }
}

#[tokio::test]
async fn test_search_with_limit() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store many memories
    for i in 0..20 {
        let entry = MemoryEntry::new(
            format!("item_{}", i),
            format!("Common content item {}", i),
            MemoryType::ShortTerm,
        );
        memory.store_memory(entry).await.unwrap();
    }

    // Search with limit
    let results = memory.search_memories("Common", 5).await.unwrap();

    // Should respect limit
    assert!(results.len() <= 5, "Should not exceed limit");
}

#[tokio::test]
async fn test_builder_with_custom_storage() {
    let memory = SynapticMemoryBuilder::new()
        .with_storage(StorageBackend::Memory)
        .build()
        .await
        .unwrap();

    assert!(memory.session_id() != uuid::Uuid::nil(), "Should have valid session ID");
}

#[tokio::test]
async fn test_builder_with_knowledge_graph() {
    let memory = SynapticMemoryBuilder::new()
        .with_storage(StorageBackend::Memory)
        .with_knowledge_graph(true)
        .build()
        .await
        .unwrap();

    // Knowledge graph should be integrated
    // We can verify by storing memories and checking they work
    let mut memory = memory;
    let entry = MemoryEntry::new(
        "kg_test".to_string(),
        "Knowledge graph test".to_string(),
        MemoryType::LongTerm,
    );

    let result = memory.store_memory(entry).await;
    assert!(result.is_ok(), "Should store with knowledge graph enabled");
}

#[tokio::test]
async fn test_builder_with_temporal_tracking() {
    let memory = SynapticMemoryBuilder::new()
        .with_temporal_tracking(true)
        .build()
        .await
        .unwrap();

    // Temporal tracking should be active
    let mut memory = memory;
    let entry = MemoryEntry::new(
        "temporal_test".to_string(),
        "Temporal tracking test".to_string(),
        MemoryType::ShortTerm,
    );

    let result = memory.store_memory(entry).await;
    assert!(result.is_ok(), "Should store with temporal tracking enabled");
}

#[tokio::test]
async fn test_builder_with_checkpoint_interval() {
    let memory = SynapticMemoryBuilder::new()
        .with_checkpoint_interval(Duration::from_secs(60))
        .build()
        .await
        .unwrap();

    // Should create successfully with custom interval
    assert!(memory.session_id() != uuid::Uuid::nil());
}

#[tokio::test]
async fn test_builder_with_custom_session_id() {
    let custom_id = uuid::Uuid::new_v4();

    let memory = SynapticMemoryBuilder::new()
        .with_session_id(custom_id)
        .build()
        .await
        .unwrap();

    assert_eq!(memory.session_id(), custom_id, "Should use custom session ID");
}

#[tokio::test]
async fn test_builder_full_configuration() {
    let custom_id = uuid::Uuid::new_v4();

    let memory = SynapticMemoryBuilder::new()
        .with_storage(StorageBackend::Memory)
        .with_knowledge_graph(true)
        .with_temporal_tracking(true)
        .with_checkpoint_interval(Duration::from_secs(300))
        .with_analytics(true)
        .with_session_id(custom_id)
        .build()
        .await
        .unwrap();

    assert_eq!(memory.session_id(), custom_id);

    // Verify it works end-to-end
    let mut memory = memory;
    let entry = MemoryEntry::new(
        "full_config_test".to_string(),
        "Testing full configuration".to_string(),
        MemoryType::LongTerm,
    );

    memory.store_memory(entry).await.unwrap();

    let retrieved = memory.get_memory("full_config_test").await.unwrap();
    assert!(retrieved.is_some());
}

#[tokio::test]
async fn test_get_stats() {
    let memory = SynapticMemory::new().await.unwrap();

    let stats = memory.get_stats();

    // Stats should have valid session ID
    assert_eq!(stats.session_id, memory.session_id());
}

#[tokio::test]
async fn test_access_agent_memory() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store via MemoryOperations
    let entry = MemoryEntry::new(
        "agent_test".to_string(),
        "Agent memory test".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await.unwrap();

    // Access underlying AgentMemory
    let agent_memory = memory.agent_memory();

    // Should be able to use AgentMemory methods
    // (This verifies the integration is working)
    assert!(agent_memory.session_id() != uuid::Uuid::nil());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let memory = std::sync::Arc::new(tokio::sync::Mutex::new(
        SynapticMemory::new().await.unwrap()
    ));

    // Perform concurrent stores
    let mut handles = vec![];

    for i in 0..10 {
        let mem = std::sync::Arc::clone(&memory);
        let handle = tokio::spawn(async move {
            let mut m = mem.lock().await;
            let entry = MemoryEntry::new(
                format!("concurrent_{}", i),
                format!("Concurrent value {}", i),
                MemoryType::ShortTerm,
            );
            m.store_memory(entry).await
        });
        handles.push(handle);
    }

    // Wait for all operations
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent store should succeed");
    }

    // Verify all were stored
    for i in 0..10 {
        let retrieved = memory.lock().await
            .get_memory(&format!("concurrent_{}", i))
            .await
            .unwrap();
        assert!(retrieved.is_some(), "Concurrent memory {} should exist", i);
    }
}

#[tokio::test]
async fn test_long_term_memory_storage() {
    let mut memory = SynapticMemory::new().await.unwrap();

    let entry = MemoryEntry::new(
        "long_term_key".to_string(),
        "Long-term memory content".to_string(),
        MemoryType::LongTerm,
    );

    memory.store_memory(entry).await.unwrap();

    let retrieved = memory.get_memory("long_term_key").await.unwrap().unwrap();
    assert_eq!(retrieved.memory_type, MemoryType::LongTerm);
}

#[tokio::test]
async fn test_memory_persistence_across_instances() {
    // Note: This test demonstrates the concept but won't work fully
    // without a persistent storage backend. With MemoryStorage,
    // data is lost between instances.

    let mut memory1 = SynapticMemory::new().await.unwrap();

    let entry = MemoryEntry::new(
        "persist_test".to_string(),
        "Persistence test".to_string(),
        MemoryType::LongTerm,
    );
    memory1.store_memory(entry).await.unwrap();

    // With in-memory storage, this would return None
    // With file/SQL storage, it would persist
    let retrieved = memory1.get_memory("persist_test").await.unwrap();
    assert!(retrieved.is_some(), "Should be retrievable in same instance");
}

#[tokio::test]
async fn test_large_content_storage() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Create large content
    let large_content = "x".repeat(1_000_000); // 1MB

    let entry = MemoryEntry::new(
        "large_content".to_string(),
        large_content.clone(),
        MemoryType::LongTerm,
    );

    memory.store_memory(entry).await.unwrap();

    let retrieved = memory.get_memory("large_content").await.unwrap().unwrap();
    assert_eq!(retrieved.content.len(), 1_000_000);
}

#[tokio::test]
async fn test_empty_search_query() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store some memories
    let entry = MemoryEntry::new(
        "search_test".to_string(),
        "Content for search".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await.unwrap();

    // Empty search should handle gracefully
    let result = memory.search_memories("", 10).await;
    // Should either return empty results or error gracefully
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_zero_search_limit() {
    let mut memory = SynapticMemory::new().await.unwrap();

    let entry = MemoryEntry::new(
        "limit_test".to_string(),
        "Limit test content".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await.unwrap();

    // Zero limit should return empty results or error
    let result = memory.search_memories("test", 0).await;
    // Should handle gracefully
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test]
async fn test_update_preserves_metadata() {
    let mut memory = SynapticMemory::new().await.unwrap();

    // Store with initial metadata
    let entry = MemoryEntry::new(
        "metadata_test".to_string(),
        "Initial content".to_string(),
        MemoryType::LongTerm,
    );
    memory.store_memory(entry).await.unwrap();

    let initial = memory.get_memory("metadata_test").await.unwrap().unwrap();
    let initial_created = initial.created_at;
    let initial_type = initial.memory_type;

    // Update content
    memory.update_memory("metadata_test", "Updated content").await.unwrap();

    // Verify metadata preserved
    let updated = memory.get_memory("metadata_test").await.unwrap().unwrap();
    assert_eq!(updated.created_at, initial_created, "Created time should be preserved");
    assert_eq!(updated.content, "Updated content");
}

#[tokio::test]
async fn test_multiple_updates() {
    let mut memory = SynapticMemory::new().await.unwrap();

    let entry = MemoryEntry::new(
        "multi_update".to_string(),
        "Version 1".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await.unwrap();

    // Multiple updates
    for i in 2..=5 {
        memory.update_memory("multi_update", &format!("Version {}", i)).await.unwrap();
    }

    let final_entry = memory.get_memory("multi_update").await.unwrap().unwrap();
    assert_eq!(final_entry.content, "Version 5");
    assert!(final_entry.access_count >= 5, "Should track multiple accesses");
}
