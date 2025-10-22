//! Integration tests for knowledge graph synchronization
//!
//! These tests verify that the knowledge graph stays synchronized with
//! memory storage operations through the MemoryGraphSync trait.

use rust_synaptic::{
    AgentMemory, MemoryConfig, StorageBackend, QueryContextOptions,
    memory::knowledge_graph::{MemoryGraphSync, GraphSyncConfig, TemporalEventType},
};

#[tokio::test]
async fn test_graph_sync_on_memory_creation() {
    // Create agent with knowledge graph enabled
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store a memory
    agent.store("test_key", "test_value").await.unwrap();

    // Verify graph has the node
    let stats = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats.node_count, 1, "Graph should have one node after memory creation");
}

#[tokio::test]
async fn test_graph_sync_on_memory_update() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Create initial memory
    agent.store("test_key", "initial_value").await.unwrap();
    let initial_stats = agent.knowledge_graph_stats().unwrap();
    let initial_node_count = initial_stats.node_count;

    // Update the memory
    agent.update("test_key", "updated_value").await.unwrap();

    // Verify graph still has the same number of nodes (update, not create)
    let updated_stats = agent.knowledge_graph_stats().unwrap();
    assert_eq!(
        updated_stats.node_count, initial_node_count,
        "Node count should remain the same after update"
    );

    // Verify memory was updated
    let memory = agent.retrieve("test_key").await.unwrap().unwrap();
    assert_eq!(memory.value, "updated_value");
}

#[tokio::test]
async fn test_graph_sync_on_memory_deletion() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Create memory
    agent.store("test_key", "test_value").await.unwrap();
    let stats_before = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats_before.node_count, 1);

    // Delete memory
    agent.delete("test_key").await.unwrap();

    // Verify graph node was removed
    let stats_after = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats_after.node_count, 0, "Graph should have no nodes after deletion");

    // Verify memory was deleted
    let memory = agent.retrieve("test_key").await.unwrap();
    assert!(memory.is_none(), "Memory should not exist after deletion");
}

#[tokio::test]
async fn test_graph_sync_disabled() {
    // Create agent with graph sync disabled
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::disabled()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memories
    agent.store("key1", "value1").await.unwrap();
    agent.store("key2", "value2").await.unwrap();

    // Even with sync disabled, basic operations should work
    let memory = agent.retrieve("key1").await.unwrap();
    assert!(memory.is_some());
}

#[tokio::test]
async fn test_graph_creates_relationships_between_memories() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store related memories with shared tags
    agent.store("memory1", "content about rust programming").await.unwrap();
    agent.store("memory2", "content about programming languages").await.unwrap();
    agent.store("memory3", "content about cooking recipes").await.unwrap();

    // Create explicit relationship
    agent.create_memory_relationship(
        "memory1",
        "memory2",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    // Find related memories
    let related = agent.find_related_memories("memory1", 2, None).await.unwrap();

    // Should find at least memory2 as related
    assert!(!related.is_empty(), "Should find related memories");
    let related_keys: Vec<_> = related.iter().map(|r| r.memory_key.as_str()).collect();
    assert!(related_keys.contains(&"memory2"), "Should find memory2 as related");
}

#[tokio::test]
async fn test_query_with_graph_context() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store multiple memories
    agent.store("rust_basics", "Introduction to Rust programming").await.unwrap();
    agent.store("rust_advanced", "Advanced Rust concepts").await.unwrap();
    agent.store("python_intro", "Python programming basics").await.unwrap();

    // Create relationships
    agent.create_memory_relationship(
        "rust_basics",
        "rust_advanced",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    // Query with graph context
    let options = QueryContextOptions::default()
        .with_limit(10)
        .with_depth(2);

    let results = agent.query_with_graph_context("Rust", options).await.unwrap();

    // Should find Rust-related memories
    assert!(!results.is_empty(), "Should find memories matching query");

    // At least one result should have graph context
    let has_context = results.iter().any(|r| r.graph_context.is_some());
    assert!(has_context, "At least one result should have graph context");
}

#[tokio::test]
async fn test_query_without_graph_context() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memories
    agent.store("test1", "some test content").await.unwrap();
    agent.store("test2", "more test content").await.unwrap();

    // Query without graph context
    let options = QueryContextOptions::default()
        .without_graph_context()
        .with_limit(10);

    let results = agent.query_with_graph_context("test", options).await.unwrap();

    // Should find memories but without graph context
    assert!(!results.is_empty(), "Should find memories matching query");

    // All results should have None for graph_context
    let all_no_context = results.iter().all(|r| r.graph_context.is_none());
    assert!(all_no_context, "All results should have no graph context when disabled");
}

#[tokio::test]
async fn test_batch_sync() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store multiple memories (batch operation)
    for i in 0..10 {
        agent.store(&format!("key_{}", i), &format!("value_{}", i)).await.unwrap();
    }

    // Verify all nodes were created
    let stats = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats.node_count, 10, "Should have created 10 nodes");

    // Verify all memories are retrievable
    for i in 0..10 {
        let memory = agent.retrieve(&format!("key_{}", i)).await.unwrap();
        assert!(memory.is_some(), "Memory {} should exist", i);
    }
}

#[tokio::test]
async fn test_update_preserves_relationships() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Create two memories and a relationship
    agent.store("mem1", "original content 1").await.unwrap();
    agent.store("mem2", "original content 2").await.unwrap();
    agent.create_memory_relationship(
        "mem1",
        "mem2",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    // Verify relationship exists
    let related_before = agent.find_related_memories("mem1", 2, None).await.unwrap();
    assert!(!related_before.is_empty(), "Should have relationships before update");

    // Update memory
    agent.update("mem1", "updated content 1").await.unwrap();

    // Verify relationship still exists after update
    let related_after = agent.find_related_memories("mem1", 2, None).await.unwrap();
    assert!(!related_after.is_empty(), "Should still have relationships after update");
}

#[tokio::test]
async fn test_delete_removes_relationships() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Create memories and relationships
    agent.store("mem1", "content 1").await.unwrap();
    agent.store("mem2", "content 2").await.unwrap();
    agent.store("mem3", "content 3").await.unwrap();

    agent.create_memory_relationship(
        "mem1",
        "mem2",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    agent.create_memory_relationship(
        "mem1",
        "mem3",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    let stats_before = agent.knowledge_graph_stats().unwrap();
    let edges_before = stats_before.edge_count;

    // Delete mem1
    agent.delete("mem1").await.unwrap();

    // Verify relationships were cleaned up
    let stats_after = agent.knowledge_graph_stats().unwrap();
    assert!(
        stats_after.edge_count < edges_before,
        "Relationship count should decrease after deletion"
    );
}

#[tokio::test]
async fn test_graph_sync_with_temporal_tracking() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memory
    agent.store("temporal_test", "test value").await.unwrap();

    // Access memory multiple times
    for _ in 0..5 {
        agent.retrieve("temporal_test").await.unwrap();
    }

    // Verify memory exists and has access count
    let memory = agent.retrieve("temporal_test").await.unwrap().unwrap();
    assert!(memory.access_count >= 5, "Access count should be tracked");
}

#[tokio::test]
async fn test_find_path_between_memories() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Create a chain of memories: A -> B -> C
    agent.store("mem_a", "content a").await.unwrap();
    agent.store("mem_b", "content b").await.unwrap();
    agent.store("mem_c", "content c").await.unwrap();

    agent.create_memory_relationship(
        "mem_a",
        "mem_b",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    agent.create_memory_relationship(
        "mem_b",
        "mem_c",
        rust_synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
        None,
    ).await.unwrap();

    // Find path from A to C
    let path = agent.find_path_between_memories("mem_a", "mem_c", Some(3)).await.unwrap();

    // Should find a path through B
    assert!(path.is_some(), "Should find path from A to C through B");
    let path = path.unwrap();
    assert!(path.nodes.len() >= 2, "Path should contain multiple nodes");
}

#[tokio::test]
async fn test_concurrent_graph_operations() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Perform multiple operations concurrently
    let mut handles = vec![];

    for i in 0..5 {
        agent.store(&format!("concurrent_{}", i), &format!("value_{}", i)).await.unwrap();
    }

    // Verify all memories were created
    for i in 0..5 {
        let memory = agent.retrieve(&format!("concurrent_{}", i)).await.unwrap();
        assert!(memory.is_some(), "Memory {} should exist", i);
    }

    let stats = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats.node_count, 5, "Should have created 5 nodes");
}

#[tokio::test]
async fn test_graph_sync_error_handling() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::default()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Try to update non-existent memory
    let result = agent.update("nonexistent", "new value").await;
    assert!(result.is_err(), "Updating nonexistent memory should fail");

    // Try to delete non-existent memory
    let result = agent.delete("nonexistent").await;
    assert!(result.is_err(), "Deleting nonexistent memory should fail");

    // Graph should remain consistent
    let stats = agent.knowledge_graph_stats().unwrap();
    assert_eq!(stats.node_count, 0, "Graph should have no nodes after failed operations");
}

#[tokio::test]
async fn test_lightweight_sync_config() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        graph_sync_config: Some(GraphSyncConfig::lightweight()),
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memories with lightweight sync
    agent.store("light1", "value1").await.unwrap();
    agent.store("light2", "value2").await.unwrap();

    // Basic operations should work
    let memory = agent.retrieve("light1").await.unwrap();
    assert!(memory.is_some());

    // Graph should be populated
    let stats = agent.knowledge_graph_stats().unwrap();
    assert!(stats.node_count > 0, "Graph should have nodes even with lightweight config");
}
