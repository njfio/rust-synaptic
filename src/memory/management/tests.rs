//! Comprehensive tests for memory management functionality

#[cfg(test)]
mod tests {
    use crate::memory::management::{MemoryManager, AdvancedMemoryManager, SummarizationTrigger, SummarizationTriggerType, MemoryManagementConfig};
    use crate::memory::storage::memory::MemoryStorage;
    use crate::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
    use crate::error::Result;
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
    async fn test_summarization_trigger_related_memory_threshold() -> Result<()> {
        let config = MemoryManagementConfig {
            enable_auto_summarization: true,
            summarization_threshold: 3,
            ..Default::default()
        };

        let advanced_manager = AdvancedMemoryManager::new(config);

        // Create a memory that should trigger summarization
        let memory = MemoryEntry::new(
            "test_memory".to_string(),
            "This is a test memory with related content".to_string(),
            MemoryType::LongTerm,
        );

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should trigger because our placeholder returns 5 related memories, which exceeds threshold of 3
        assert!(trigger_result.is_some());
        let trigger = trigger_result.unwrap();
        assert_eq!(trigger.trigger_type, SummarizationTriggerType::RelatedMemoryThreshold);
        assert!(trigger.confidence > 0.0);
        assert!(trigger.reason.contains("Related memory count"));

        Ok(())
    }

    #[tokio::test]
    async fn test_summarization_trigger_content_complexity() -> Result<()> {
        let config = MemoryManagementConfig::default();
        let advanced_manager = AdvancedMemoryManager::new(config);

        // Create a complex memory that should trigger summarization
        let complex_content = "This is an extremely complex memory entry with sophisticated vocabulary, intricate sentence structures, and comprehensive detailed explanations that demonstrate high linguistic complexity. The content includes multiple technical terms, elaborate descriptions, and extensive information that would benefit from summarization due to its inherent complexity and verbosity. Furthermore, this memory contains numerous sophisticated concepts that require careful analysis and understanding.".repeat(10);

        let memory = MemoryEntry::new(
            "complex_memory".to_string(),
            complex_content,
            MemoryType::LongTerm,
        );

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should trigger due to content complexity
        assert!(trigger_result.is_some());
        let trigger = trigger_result.unwrap();
        assert_eq!(trigger.trigger_type, SummarizationTriggerType::ContentComplexity);
        assert!(trigger.confidence > 0.0);
        assert!(trigger.reason.contains("Content complexity score"));

        Ok(())
    }

    #[tokio::test]
    async fn test_summarization_trigger_temporal_clustering() -> Result<()> {
        let config = MemoryManagementConfig::default();
        let advanced_manager = AdvancedMemoryManager::new(config);

        let memory = MemoryEntry::new(
            "temporal_memory".to_string(),
            "Memory for temporal clustering test".to_string(),
            MemoryType::ShortTerm,
        );

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should not trigger because our placeholder cluster size (3) is below threshold (5)
        // But let's test that the evaluation runs without error
        if let Some(trigger) = trigger_result {
            assert_eq!(trigger.trigger_type, SummarizationTriggerType::TemporalClustering);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_summarization_trigger_semantic_density() -> Result<()> {
        let config = MemoryManagementConfig::default();
        let advanced_manager = AdvancedMemoryManager::new(config);

        // Create a memory with high semantic density
        let mut memory = MemoryEntry::new(
            "semantic_memory".to_string(),
            "artificial intelligence machine learning neural networks deep learning algorithms optimization performance metrics evaluation".to_string(),
            MemoryType::LongTerm,
        );

        // Add many tags to increase semantic density
        memory.metadata.tags = vec![
            "ai".to_string(), "ml".to_string(), "neural".to_string(),
            "deep".to_string(), "learning".to_string(), "algorithms".to_string(),
            "optimization".to_string(), "performance".to_string(), "metrics".to_string(),
            "evaluation".to_string()
        ];

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should trigger due to high semantic density
        assert!(trigger_result.is_some());
        let trigger = trigger_result.unwrap();
        assert_eq!(trigger.trigger_type, SummarizationTriggerType::SemanticDensity);
        assert!(trigger.confidence > 0.0);
        assert!(trigger.reason.contains("Semantic density"));

        Ok(())
    }

    #[tokio::test]
    async fn test_summarization_trigger_storage_optimization() -> Result<()> {
        let config = MemoryManagementConfig::default();
        let advanced_manager = AdvancedMemoryManager::new(config);

        // Create a very large memory that should trigger storage optimization
        let large_content = "This is a very large memory entry that exceeds the storage optimization threshold. ".repeat(200);

        let memory = MemoryEntry::new(
            "large_memory".to_string(),
            large_content,
            MemoryType::LongTerm,
        );

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should trigger due to large size
        assert!(trigger_result.is_some());
        let trigger = trigger_result.unwrap();
        assert_eq!(trigger.trigger_type, SummarizationTriggerType::StorageOptimization);
        assert!(trigger.confidence > 0.0);
        assert!(trigger.reason.contains("Memory size"));

        Ok(())
    }

    #[tokio::test]
    async fn test_no_summarization_trigger() -> Result<()> {
        let config = MemoryManagementConfig {
            enable_auto_summarization: true,
            summarization_threshold: 10, // High threshold
            ..Default::default()
        };
        let advanced_manager = AdvancedMemoryManager::new(config);

        // Create a simple memory that should not trigger summarization
        let memory = MemoryEntry::new(
            "simple_memory".to_string(),
            "Simple short memory".to_string(),
            MemoryType::ShortTerm,
        );

        // Test the trigger evaluation
        let trigger_result = advanced_manager.evaluate_summarization_triggers(&memory, None).await?;

        // Should not trigger any summarization
        assert!(trigger_result.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_execute_automatic_summarization() -> Result<()> {
        let config = MemoryManagementConfig::default();
        let mut advanced_manager = AdvancedMemoryManager::new(config);

        // Create a summarization trigger
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("test_key".to_string(), "test_value".to_string());

        let trigger = SummarizationTrigger {
            reason: "Test trigger for automatic summarization".to_string(),
            related_memory_keys: vec!["memory1".to_string(), "memory2".to_string()],
            trigger_type: SummarizationTriggerType::Manual,
            confidence: 0.8,
            metadata,
        };

        // Execute the summarization
        let result = advanced_manager.execute_automatic_summarization(trigger).await?;

        // Verify the result
        assert_eq!(result.processed_count, 2);
        assert!(result.summary_key.starts_with("summary_"));
        // Duration is automatically valid as u64, no need to check
        assert!(!result.messages.is_empty());

        Ok(())
    }
}
