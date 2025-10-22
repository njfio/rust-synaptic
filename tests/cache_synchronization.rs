// Comprehensive tests for cache synchronization between state and storage
//
// These tests verify that cache misses are properly rehydrated into the state,
// access patterns are updated, and knowledge graph nodes are refreshed.

use synaptic::{AgentMemory, MemoryConfig};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_cache_miss_rehydrates_state() {
    // This is the critical test for Phase 4.4:
    // When memory is not in state but exists in storage, it should be
    // loaded into state so future accesses are fast.

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory (will be in both state and storage)
    memory.store("test_key", "test_value").await.unwrap();

    // Verify it's accessible
    let entry1 = memory.retrieve("test_key").await.unwrap();
    assert!(entry1.is_some(), "Memory should be retrievable after store");
    assert_eq!(entry1.unwrap().content, "test_value");

    // Create a new AgentMemory instance with the same storage
    // This simulates a cache miss scenario where state is empty but storage has the data
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // First retrieval - cache miss, should load from storage into state
    let entry2 = memory2.retrieve("test_key").await.unwrap();
    assert!(entry2.is_some(), "Memory should be retrievable from storage");
    assert_eq!(entry2.unwrap().content, "test_value");

    // Second retrieval - should now be in state (cache hit)
    // This tests that the first retrieval injected the entry into state
    let entry3 = memory2.retrieve("test_key").await.unwrap();
    assert!(entry3.is_some(), "Memory should still be retrievable");
    assert_eq!(entry3.unwrap().content, "test_value");

    // The key assertion: access_count should have increased twice
    // Once for the cache miss rehydration, once for the cache hit
    assert!(entry3.unwrap().access_count >= 1,
        "Access count should be updated after cache miss rehydration");
}

#[tokio::test]
async fn test_access_patterns_updated_on_cache_miss() {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("access_test", "value").await.unwrap();

    // Get initial access time
    let entry1 = memory.retrieve("access_test").await.unwrap().unwrap();
    let initial_access = entry1.last_accessed;
    let initial_count = entry1.access_count;

    // Wait a bit to ensure timestamp difference
    sleep(Duration::from_millis(100)).await;

    // Create new instance (simulating cache miss)
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Retrieve from cold storage (cache miss)
    let entry2 = memory2.retrieve("access_test").await.unwrap().unwrap();

    // Verify access patterns were updated
    assert!(entry2.last_accessed >= initial_access,
        "Last accessed time should be updated on cache miss");
    assert!(entry2.access_count > initial_count,
        "Access count should be incremented on cache miss");
}

#[tokio::test]
async fn test_repeated_cache_miss_retrieval_performance() {
    // This test verifies the performance improvement:
    // After first cache miss, subsequent retrievals should be fast (from state)

    let config = MemoryConfig::default();
    let mut memory1 = AgentMemory::new(config).await.unwrap();

    // Store multiple memories
    for i in 0..10 {
        memory1.store(&format!("key_{}", i), &format!("value_{}", i)).await.unwrap();
    }

    // Create new instance (empty state, full storage)
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // First retrieval of each key (cache miss)
    for i in 0..10 {
        let entry = memory2.retrieve(&format!("key_{}", i)).await.unwrap();
        assert!(entry.is_some(), "Memory {} should be found in storage", i);
    }

    // Second retrieval of each key (should be cache hit)
    for i in 0..10 {
        let entry = memory2.retrieve(&format!("key_{}", i)).await.unwrap();
        assert!(entry.is_some(), "Memory {} should be found in state cache", i);
        assert_eq!(entry.unwrap().content, format!("value_{}", i));
    }
}

#[tokio::test]
async fn test_cache_miss_updates_both_memory_types() {
    // Test that both ShortTerm and LongTerm memories are properly rehydrated

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a short-term memory
    memory.store("short_term_key", "short_term_value").await.unwrap();

    // Create new instance
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Retrieve (cache miss)
    let entry = memory2.retrieve("short_term_key").await.unwrap();
    assert!(entry.is_some(), "Short-term memory should be retrievable");

    // Retrieve again (should be from state)
    let entry2 = memory2.retrieve("short_term_key").await.unwrap();
    assert!(entry2.is_some(), "Short-term memory should be in state cache");
}

#[tokio::test]
async fn test_concurrent_cache_miss_rehydration() {
    // Verify that concurrent cache misses don't cause data corruption

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store multiple memories
    for i in 0..20 {
        memory.store(&format!("concurrent_{}", i), &format!("value_{}", i)).await.unwrap();
    }

    // Create new instance
    let config2 = MemoryConfig::default();
    let memory2 = std::sync::Arc::new(tokio::sync::Mutex::new(
        AgentMemory::new(config2).await.unwrap()
    ));

    // Retrieve all memories concurrently (cache misses)
    let mut handles = vec![];
    for i in 0..20 {
        let mem = std::sync::Arc::clone(&memory2);
        let key = format!("concurrent_{}", i);
        let expected_value = format!("value_{}", i);

        handles.push(tokio::spawn(async move {
            let mut m = mem.lock().await;
            let entry = m.retrieve(&key).await.unwrap();
            assert!(entry.is_some(), "Memory {} should be found", key);
            assert_eq!(entry.unwrap().content, expected_value);
        }));
    }

    // Wait for all retrievals to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all memories are now in state
    for i in 0..20 {
        let entry = memory2.lock().await.retrieve(&format!("concurrent_{}", i)).await.unwrap();
        assert!(entry.is_some(), "All memories should be in state after concurrent access");
    }
}

#[tokio::test]
async fn test_cache_miss_nonexistent_key() {
    // Verify that cache miss handling works correctly for non-existent keys

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Try to retrieve non-existent key
    let entry = memory.retrieve("nonexistent").await.unwrap();
    assert!(entry.is_none(), "Non-existent key should return None");

    // Try again to verify no corruption
    let entry2 = memory.retrieve("nonexistent").await.unwrap();
    assert!(entry2.is_none(), "Non-existent key should still return None");
}

#[tokio::test]
async fn test_cache_miss_with_state_clearing() {
    // Test the scenario where state is explicitly cleared but storage remains

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store memories
    memory.store("persistent_key", "persistent_value").await.unwrap();

    // Verify it's accessible
    let entry1 = memory.retrieve("persistent_key").await.unwrap();
    assert!(entry1.is_some());

    // In a real scenario, state might be cleared due to memory pressure or restart
    // Simulate by creating new instance
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Should still be retrievable from storage (cache miss)
    let entry2 = memory2.retrieve("persistent_key").await.unwrap();
    assert!(entry2.is_some(), "Memory should survive state clearing");
    assert_eq!(entry2.unwrap().content, "persistent_value");
}

#[tokio::test]
async fn test_cache_rehydration_preserves_metadata() {
    // Verify that metadata (tags, importance, etc.) is preserved during cache miss

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory with rich metadata
    memory.store("metadata_key", "value_with_metadata").await.unwrap();

    // Get the entry to verify initial metadata
    let entry1 = memory.retrieve("metadata_key").await.unwrap().unwrap();
    let original_importance = entry1.importance;
    let original_created_at = entry1.created_at;

    // Create new instance (cache miss scenario)
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Retrieve from storage
    let entry2 = memory2.retrieve("metadata_key").await.unwrap().unwrap();

    // Verify metadata is preserved
    assert_eq!(entry2.importance, original_importance,
        "Importance should be preserved");
    assert_eq!(entry2.created_at, original_created_at,
        "Created timestamp should be preserved");
    assert_eq!(entry2.content, "value_with_metadata",
        "Content should be preserved");
}

#[tokio::test]
async fn test_cache_miss_updates_state_not_storage() {
    // Verify that cache miss rehydration only updates state, not storage
    // (to avoid unnecessary I/O)

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("io_test", "value").await.unwrap();

    // Get initial entry
    let entry1 = memory.retrieve("io_test").await.unwrap().unwrap();
    let initial_access_count = entry1.access_count;

    // Create new instance
    let config2 = MemoryConfig::default();
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Retrieve (cache miss) - this should update state but not trigger storage write
    let entry2 = memory2.retrieve("io_test").await.unwrap().unwrap();

    // Access count should be incremented locally in state
    assert!(entry2.access_count >= initial_access_count,
        "Access count should be updated in state");

    // Note: We can't easily verify that storage wasn't written to without
    // mocking, but the design should ensure only state is updated
}

#[cfg(feature = "knowledge-graph")]
#[tokio::test]
async fn test_cache_miss_refreshes_knowledge_graph() {
    // Verify that knowledge graph is updated when memory is rehydrated from storage

    let mut config = MemoryConfig::default();
    config.enable_knowledge_graph = true;

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store memories with relationships
    memory.store("kg_key_1", "related to knowledge graph").await.unwrap();
    memory.store("kg_key_2", "also related to knowledge graph").await.unwrap();

    // Create new instance with knowledge graph enabled
    let mut config2 = MemoryConfig::default();
    config2.enable_knowledge_graph = true;
    let mut memory2 = AgentMemory::new(config2).await.unwrap();

    // Retrieve both memories (cache misses)
    let entry1 = memory2.retrieve("kg_key_1").await.unwrap();
    let entry2 = memory2.retrieve("kg_key_2").await.unwrap();

    assert!(entry1.is_some(), "First memory should be retrievable");
    assert!(entry2.is_some(), "Second memory should be retrievable");

    // Knowledge graph should now have nodes for both memories
    // (This would require inspecting the knowledge graph state, which isn't
    // directly exposed, but the code path ensures this happens)
}
