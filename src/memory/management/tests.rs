//! Comprehensive tests for memory management functionality

#[cfg(test)]
mod tests {
    use crate::memory::management::MemoryManager;
    use crate::error::Result;
    use crate::memory::storage::memory::MemoryStorage;
    use crate::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
    use crate::memory::knowledge_graph::{MemoryKnowledgeGraph, GraphConfig};
    use std::sync::Arc;
    use chrono::{Utc, Duration};

    /// Create a test memory manager with knowledge graph
    async fn create_test_memory_manager() -> Result<MemoryManager> {
        let storage = Arc::new(MemoryStorage::new());
        let kg_config = GraphConfig::default();
        let knowledge_graph = Some(MemoryKnowledgeGraph::new(kg_config));
        
        MemoryManager::new(storage, knowledge_graph, None, None).await
    }

    /// Create a test memory entry with specified properties
    fn create_test_memory(
        key: &str,
        content: &str,
        tags: Vec<String>,
        importance: f64,
        created_offset_hours: i64,
    ) -> MemoryEntry {
        let mut metadata = MemoryMetadata::new()
            .with_tags(tags)
            .with_importance(importance);
        
        // Adjust creation time
        metadata.created_at = Utc::now() - Duration::hours(created_offset_hours);
        
        MemoryEntry {
            key: key.to_string(),
            value: content.to_string(),
            memory_type: MemoryType::LongTerm,
            metadata,
            embedding: None,
        }
    }

    /// Create a test memory entry with embedding
    fn create_test_memory_with_embedding(
        key: &str,
        content: &str,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> MemoryEntry {
        let metadata = MemoryMetadata::new().with_tags(tags);
        
        MemoryEntry {
            key: key.to_string(),
            value: content.to_string(),
            memory_type: MemoryType::LongTerm,
            metadata,
            embedding: Some(embedding),
        }
    }

    #[tokio::test]
    async fn test_count_related_memories_empty_storage() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        let memory = create_test_memory(
            "test_memory",
            "This is a test memory",
            vec!["test".to_string()],
            0.5,
            0,
        );
        
        let count = manager.count_related_memories(&memory).await?;
        assert_eq!(count, 0, "Empty storage should return 0 related memories");
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_tag_based() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Store some memories with overlapping tags
        let memory1 = create_test_memory(
            "memory1",
            "Content about machine learning",
            vec!["ai".to_string(), "ml".to_string()],
            0.7,
            2,
        );
        
        let memory2 = create_test_memory(
            "memory2",
            "Content about artificial intelligence",
            vec!["ai".to_string(), "research".to_string()],
            0.6,
            3,
        );
        
        let memory3 = create_test_memory(
            "memory3",
            "Content about cooking",
            vec!["food".to_string(), "recipe".to_string()],
            0.5,
            4,
        );
        
        // Store memories
        manager.store_memory(&memory1).await?;
        manager.store_memory(&memory2).await?;
        manager.store_memory(&memory3).await?;
        
        // Test memory with overlapping tags
        let test_memory = create_test_memory(
            "test_memory",
            "Content about AI and machine learning",
            vec!["ai".to_string(), "ml".to_string(), "research".to_string()],
            0.8,
            0,
        );
        
        let count = manager.count_related_memories(&test_memory).await?;
        
        // Should find memory1 (shares ai, ml) and memory2 (shares ai, research)
        // memory3 has no overlapping tags
        assert!(count >= 2, "Should find at least 2 related memories based on tags, found: {}", count);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_similarity_based() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Create memories with similar embeddings
        let similar_embedding1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let similar_embedding2 = vec![0.15, 0.25, 0.35, 0.45, 0.55]; // Very similar
        let different_embedding = vec![0.9, 0.8, 0.7, 0.6, 0.5]; // Different
        
        let memory1 = create_test_memory_with_embedding(
            "memory1",
            "Content about machine learning algorithms",
            vec!["ml".to_string()],
            similar_embedding1.clone(),
        );
        
        let memory2 = create_test_memory_with_embedding(
            "memory2",
            "Content about deep learning models",
            vec!["dl".to_string()],
            similar_embedding2,
        );
        
        let memory3 = create_test_memory_with_embedding(
            "memory3",
            "Content about cooking recipes",
            vec!["food".to_string()],
            different_embedding,
        );
        
        // Store memories
        manager.store_memory(&memory1).await?;
        manager.store_memory(&memory2).await?;
        manager.store_memory(&memory3).await?;
        
        // Test memory with similar embedding to memory1
        let test_memory = create_test_memory_with_embedding(
            "test_memory",
            "Content about neural networks",
            vec!["nn".to_string()],
            similar_embedding1,
        );
        
        let count = manager.count_related_memories(&test_memory).await?;
        
        // Should find memory2 due to high similarity (memory1 has identical embedding)
        assert!(count >= 1, "Should find at least 1 related memory based on similarity, found: {}", count);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_temporal_proximity() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Create memories with temporal proximity
        let memory1 = create_test_memory(
            "memory1",
            "Meeting notes from morning standup",
            vec!["meeting".to_string()],
            0.6,
            0, // Created now
        );
        
        let memory2 = create_test_memory(
            "memory2",
            "Follow-up tasks from standup meeting",
            vec!["tasks".to_string()],
            0.7,
            0, // Created now (within 1 hour window)
        );
        
        let memory3 = create_test_memory(
            "memory3",
            "Old project documentation",
            vec!["docs".to_string()],
            0.5,
            25, // Created 25 hours ago (outside window)
        );
        
        // Store memories
        manager.store_memory(&memory1).await?;
        manager.store_memory(&memory2).await?;
        manager.store_memory(&memory3).await?;
        
        // Test memory created around the same time
        let test_memory = create_test_memory(
            "test_memory",
            "Action items from the standup",
            vec!["action".to_string()],
            0.8,
            0, // Created now
        );
        
        let count = manager.count_related_memories(&test_memory).await?;
        
        // Should find memory1 and memory2 due to temporal proximity and content similarity
        assert!(count >= 1, "Should find at least 1 related memory based on temporal proximity, found: {}", count);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_content_similarity() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Create memories with similar content
        let memory1 = create_test_memory(
            "memory1",
            "machine learning algorithms neural networks deep learning",
            vec!["tech".to_string()],
            0.6,
            2,
        );
        
        let memory2 = create_test_memory(
            "memory2",
            "cooking recipes ingredients preparation techniques",
            vec!["food".to_string()],
            0.5,
            3,
        );
        
        // Store memories
        manager.store_memory(&memory1).await?;
        manager.store_memory(&memory2).await?;
        
        // Test memory with content similar to memory1
        let test_memory = create_test_memory(
            "test_memory",
            "neural networks machine learning algorithms",
            vec!["ai".to_string()],
            0.8,
            0,
        );
        
        let count = manager.count_related_memories(&test_memory).await?;
        
        // Should find memory1 due to content similarity
        assert!(count >= 1, "Should find at least 1 related memory based on content similarity, found: {}", count);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_multiple_strategies() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Create a comprehensive test with multiple relationship types
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        
        let memory1 = create_test_memory_with_embedding(
            "memory1",
            "machine learning neural networks algorithms",
            vec!["ai".to_string(), "ml".to_string()],
            embedding.clone(),
        );
        
        let memory2 = create_test_memory(
            "memory2",
            "deep learning neural networks training",
            vec!["ai".to_string(), "dl".to_string()],
            0.7,
            0, // Same time
        );
        
        let memory3 = create_test_memory(
            "memory3",
            "completely different content about cooking",
            vec!["food".to_string()],
            0.5,
            10,
        );
        
        // Store memories
        manager.store_memory(&memory1).await?;
        manager.store_memory(&memory2).await?;
        manager.store_memory(&memory3).await?;
        
        // Test memory that should relate to memory1 and memory2 through multiple strategies
        let test_memory = create_test_memory_with_embedding(
            "test_memory",
            "artificial intelligence neural networks",
            vec!["ai".to_string(), "research".to_string()],
            embedding,
        );
        
        let count = manager.count_related_memories(&test_memory).await?;
        
        // Should find memory1 (tag overlap + embedding similarity + content similarity)
        // Should find memory2 (tag overlap + temporal proximity + content similarity)
        // Should NOT find memory3 (no relationships)
        assert!(count >= 2, "Should find at least 2 related memories through multiple strategies, found: {}", count);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_count_related_memories_performance() -> Result<()> {
        let manager = create_test_memory_manager().await?;
        
        // Store many memories to test performance
        for i in 0..100 {
            let memory = create_test_memory(
                &format!("memory_{}", i),
                &format!("Content for memory number {}", i),
                vec![format!("tag_{}", i % 10)], // 10 different tags
                0.5,
                i % 24, // Spread across 24 hours
            );
            manager.store_memory(&memory).await?;
        }
        
        let test_memory = create_test_memory(
            "test_memory",
            "Test content for performance",
            vec!["tag_5".to_string()], // Should match some memories
            0.8,
            0,
        );
        
        let start_time = std::time::Instant::now();
        let count = manager.count_related_memories(&test_memory).await?;
        let duration = start_time.elapsed();
        
        // Should complete within reasonable time (< 1 second)
        assert!(duration.as_millis() < 1000, "Related memory counting should complete within 1 second, took: {}ms", duration.as_millis());
        
        // Should find some related memories
        assert!(count > 0, "Should find some related memories in large dataset, found: {}", count);
        
        Ok(())
    }
}
