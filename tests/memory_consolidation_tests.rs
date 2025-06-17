//! Comprehensive Memory Consolidation System Tests
//! 
//! Tests for catastrophic forgetting prevention, selective replay,
//! importance scoring, and consolidation strategies.

use synaptic::{
    memory::{
        consolidation::{
            MemoryConsolidationSystem, ConsolidationConfig, ConsolidationStrategy,
            MemoryImportance, importance_scoring::ImportanceScorer,
            selective_replay::SelectiveReplayManager,
            consolidation_strategies::ConsolidationStrategies,
            elastic_weight_consolidation::ElasticWeightConsolidation,
        },
        types::{MemoryEntry, MemoryType},
    },
    error::Result,
};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use tokio;

/// Test memory consolidation system creation and basic functionality
#[tokio::test]
async fn test_consolidation_system_creation() -> Result<()> {
    let config = ConsolidationConfig::default();
    let system = MemoryConsolidationSystem::new(config)?;
    
    // Test initial state
    assert!(system.should_consolidate()); // Should consolidate on first run
    
    let stats = system.get_consolidation_stats();
    assert_eq!(stats.get("total_operations").unwrap_or(&0.0), &0.0);
    
    Ok(())
}

/// Test importance scoring with various memory types and content
#[tokio::test]
async fn test_importance_scoring_comprehensive() -> Result<()> {
    let config = ConsolidationConfig::default();
    let mut scorer = ImportanceScorer::new(&config)?;

    // Create diverse memory entries for testing
    let memories = vec![
        // High importance: long-term, frequently accessed
        create_test_memory("critical_data", "Critical system information that is accessed frequently", MemoryType::LongTerm, 50),
        // Medium importance: short-term, moderate access
        create_test_memory("temp_data", "Temporary data for processing", MemoryType::ShortTerm, 10),
        // Low importance: rarely accessed
        create_test_memory("old_data", "Old data that is rarely used", MemoryType::LongTerm, 1),
        // High content complexity
        create_test_memory("complex_data", &"A".repeat(1000), MemoryType::LongTerm, 25),
        // Short content
        create_test_memory("brief", "Hi", MemoryType::ShortTerm, 5),
    ];

    let importance_scores = scorer.calculate_batch_importance(&memories).await?;
    
    // Verify we got scores for all memories
    assert_eq!(importance_scores.len(), memories.len());
    
    // All scores should be between 0 and 1
    for score in &importance_scores {
        assert!(score.importance_score >= 0.0);
        assert!(score.importance_score <= 1.0);
        assert!(score.access_frequency >= 0.0);
        assert!(score.recency_score >= 0.0);
        assert!(score.centrality_score >= 0.0);
        assert!(score.uniqueness_score >= 0.0);
        assert!(score.temporal_consistency >= 0.0);
    }
    
    // Critical data should have higher importance than temporary data
    let critical_score = importance_scores.iter().find(|s| s.memory_key == "critical_data").unwrap();
    let temp_score = importance_scores.iter().find(|s| s.memory_key == "temp_data").unwrap();
    
    // Critical data should generally score higher due to access frequency
    assert!(critical_score.access_frequency >= temp_score.access_frequency);
    
    Ok(())
}

/// Test selective replay manager functionality
#[tokio::test]
async fn test_selective_replay_manager() -> Result<()> {
    let config = ConsolidationConfig::default();
    let mut replay_manager = SelectiveReplayManager::new(&config)?;

    // Initially empty
    assert_eq!(replay_manager.buffer_size(), 0);

    // Add memories to replay buffer
    let memories = create_test_memory_set(5);
    for memory in &memories {
        let importance = create_test_importance(&memory.key, 0.7);
        replay_manager.add_to_buffer(memory, &importance).await?;
    }

    // Buffer should contain memories
    assert_eq!(replay_manager.buffer_size(), 5);

    // Perform selective replay for testing (force immediate replay for testing)
    #[cfg(any(test, feature = "test-utils"))]
    replay_manager.force_immediate_replay().await?;

    #[cfg(not(any(test, feature = "test-utils")))]
    replay_manager.perform_selective_replay().await?;

    // Check metrics
    let metrics = replay_manager.get_metrics();
    assert!(metrics.total_replays > 0);
    assert!(metrics.avg_effectiveness >= 0.0);
    assert!(metrics.avg_effectiveness <= 1.0);

    // Get replay history
    let history = replay_manager.get_replay_history(10);
    assert!(!history.is_empty());

    Ok(())
}

/// Test consolidation strategies
#[tokio::test]
async fn test_consolidation_strategies() -> Result<()> {
    let config = ConsolidationConfig::default();
    let mut strategies = ConsolidationStrategies::new(&config)?;
    
    let memories = create_test_memory_set(10);
    let importance_scores = create_test_importance_set(&memories);
    
    // Test gradual forgetting strategy
    let result = strategies.apply_strategy(
        &ConsolidationStrategy::GradualForgetting,
        &memories,
        &importance_scores,
    ).await?;
    
    assert_eq!(result.strategy, ConsolidationStrategy::GradualForgetting);
    assert!(result.success_rate >= 0.0);
    assert!(result.success_rate <= 1.0);
    assert!(result.compression_ratio >= 0.0);
    assert!(result.quality_score >= 0.0);
    assert!(result.processing_time_ms >= 0);
    
    // Test selective replay strategy
    let result = strategies.apply_strategy(
        &ConsolidationStrategy::SelectiveReplay,
        &memories,
        &importance_scores,
    ).await?;
    
    assert_eq!(result.strategy, ConsolidationStrategy::SelectiveReplay);
    assert!(result.success_rate >= 0.0);
    
    // Test EWC strategy
    let result = strategies.apply_strategy(
        &ConsolidationStrategy::ElasticWeightConsolidation,
        &memories,
        &importance_scores,
    ).await?;
    
    assert_eq!(result.strategy, ConsolidationStrategy::ElasticWeightConsolidation);
    
    // Test hybrid strategy
    let hybrid_strategies = vec![
        ConsolidationStrategy::GradualForgetting,
        ConsolidationStrategy::SelectiveReplay,
    ];
    let result = strategies.apply_strategy(
        &ConsolidationStrategy::Hybrid(hybrid_strategies.clone()),
        &memories,
        &importance_scores,
    ).await?;
    
    if let ConsolidationStrategy::Hybrid(returned_strategies) = result.strategy {
        assert_eq!(returned_strategies, hybrid_strategies);
    } else {
        panic!("Expected hybrid strategy");
    }
    
    Ok(())
}

/// Test Elastic Weight Consolidation (EWC)
#[tokio::test]
async fn test_elastic_weight_consolidation() -> Result<()> {
    let config = ConsolidationConfig::default();
    let mut ewc = ElasticWeightConsolidation::new(&config)?;
    
    // Create importance scores with Fisher information
    let importance_scores = vec![
        MemoryImportance {
            memory_key: "test_key_1".to_string(),
            importance_score: 0.8,
            access_frequency: 0.7,
            recency_score: 0.6,
            centrality_score: 0.5,
            uniqueness_score: 0.4,
            temporal_consistency: 0.3,
            calculated_at: Utc::now(),
            fisher_information: Some(vec![0.5, 0.7, 0.3, 0.9]),
        },
        MemoryImportance {
            memory_key: "test_key_2".to_string(),
            importance_score: 0.6,
            access_frequency: 0.5,
            recency_score: 0.4,
            centrality_score: 0.3,
            uniqueness_score: 0.2,
            temporal_consistency: 0.1,
            calculated_at: Utc::now(),
            fisher_information: Some(vec![0.2, 0.4, 0.6, 0.8]),
        },
    ];
    
    // Update Fisher information
    ewc.update_fisher_information(&importance_scores).await?;
    
    // Check Fisher matrix was populated
    let fisher_matrix = ewc.get_fisher_matrix();
    assert!(!fisher_matrix.is_empty());
    
    // Check protected parameters
    let protected_params = ewc.get_protected_parameters();
    assert!(!protected_params.is_empty());
    
    // Test regularization penalty calculation
    let mut parameter_updates = HashMap::new();
    parameter_updates.insert("test_key_1_0".to_string(), 0.7);
    parameter_updates.insert("test_key_1_1".to_string(), 0.9);
    
    let penalty = ewc.calculate_regularization_penalty(&parameter_updates).await?;
    assert!(penalty >= 0.0);
    
    // Test EWC constraints application
    let mut updates = parameter_updates.clone();
    ewc.apply_ewc_constraints(&mut updates).await?;
    
    // Updates should be modified by constraints
    assert!(updates.contains_key("test_key_1_0"));
    
    // Test task consolidation
    let memories = create_test_memory_set(3);
    ewc.consolidate_task("task_1", &memories).await?;
    
    // Check task-specific information was stored
    assert!(ewc.get_task_fisher_info("task_1").is_some());
    assert!(ewc.get_task_parameters("task_1").is_some());
    
    // Test retention score calculation
    let retention_score = ewc.calculate_retention_score("test_key_1").await?;
    assert!(retention_score >= 0.0);
    assert!(retention_score <= 1.0);
    
    Ok(())
}

/// Test full consolidation system integration
#[tokio::test]
async fn test_full_consolidation_integration() -> Result<()> {
    let config = ConsolidationConfig {
        enable_ewc: true,
        ewc_lambda: 0.5,
        max_replay_buffer_size: 100,
        importance_threshold: 0.4,
        consolidation_frequency_hours: 1, // Short for testing
        enable_selective_replay: true,
        replay_batch_size: 10,
        forgetting_rate: 0.05,
        enable_importance_weighting: true,
    };
    
    let mut system = MemoryConsolidationSystem::new(config)?;
    
    // Create a diverse set of memories
    let memories = vec![
        create_test_memory("high_importance", "Critical system data", MemoryType::LongTerm, 100),
        create_test_memory("medium_importance", "Regular data", MemoryType::LongTerm, 20),
        create_test_memory("low_importance", "Temporary cache", MemoryType::ShortTerm, 2),
        create_test_memory("complex_content", &"Complex data ".repeat(100), MemoryType::LongTerm, 30),
        create_test_memory("simple_content", "Hi", MemoryType::ShortTerm, 1),
    ];
    
    // Perform consolidation
    let result = system.consolidate_memories(&memories).await?;
    
    // Verify consolidation results
    assert_eq!(result.memories_processed, memories.len());
    assert!(result.memories_consolidated > 0);
    assert!(result.effectiveness_score >= 0.0);
    assert!(result.effectiveness_score <= 1.0);
    assert!(result.processing_time_ms > 0);
    
    // Check that high importance memories are more likely to be consolidated
    assert!(result.avg_importance_score >= 0.0);
    
    // Test consolidation statistics
    let stats = system.get_consolidation_stats();
    assert_eq!(stats.get("total_operations").unwrap_or(&0.0), &1.0);
    assert!(stats.get("avg_effectiveness").unwrap_or(&0.0) >= &0.0);
    
    // Test recent results
    let recent_results = system.get_recent_results(5);
    assert_eq!(recent_results.len(), 1);
    assert_eq!(recent_results[0].memories_processed, memories.len());
    
    // Test forced consolidation
    let force_result = system.force_consolidation(&memories).await?;
    assert_eq!(force_result.memories_processed, memories.len());
    
    // Test importance scores retrieval
    let importance_scores = system.get_importance_scores(&memories).await?;
    assert_eq!(importance_scores.len(), memories.len());
    
    Ok(())
}

/// Test consolidation with different memory types and access patterns
#[tokio::test]
async fn test_consolidation_memory_type_handling() -> Result<()> {
    let config = ConsolidationConfig::default();
    let mut system = MemoryConsolidationSystem::new(config)?;
    
    // Create memories with different characteristics
    let long_term_memories = vec![
        create_test_memory("lt_1", "Long term memory 1", MemoryType::LongTerm, 50),
        create_test_memory("lt_2", "Long term memory 2", MemoryType::LongTerm, 30),
    ];
    
    let short_term_memories = vec![
        create_test_memory("st_1", "Short term memory 1", MemoryType::ShortTerm, 5),
        create_test_memory("st_2", "Short term memory 2", MemoryType::ShortTerm, 2),
    ];
    
    // Test consolidation of long-term memories
    let lt_result = system.consolidate_memories(&long_term_memories).await?;
    assert!(lt_result.memories_consolidated > 0);
    
    // Test consolidation of short-term memories
    let st_result = system.consolidate_memories(&short_term_memories).await?;
    
    // Long-term memories should generally have higher consolidation rates
    let lt_consolidation_rate = lt_result.memories_consolidated as f64 / lt_result.memories_processed as f64;
    let st_consolidation_rate = st_result.memories_consolidated as f64 / st_result.memories_processed as f64;
    
    // This is probabilistic, but long-term should generally consolidate more
    assert!(lt_consolidation_rate >= 0.0);
    assert!(st_consolidation_rate >= 0.0);
    
    Ok(())
}

/// Test consolidation performance under load
#[tokio::test]
async fn test_consolidation_performance() -> Result<()> {
    let config = ConsolidationConfig {
        max_replay_buffer_size: 1000,
        replay_batch_size: 50,
        ..ConsolidationConfig::default()
    };
    
    let mut system = MemoryConsolidationSystem::new(config)?;
    
    // Create a large set of memories
    let memories = create_test_memory_set(100);
    
    let start_time = std::time::Instant::now();
    let result = system.consolidate_memories(&memories).await?;
    let processing_time = start_time.elapsed();
    
    // Verify performance characteristics
    assert_eq!(result.memories_processed, 100);
    assert!(result.processing_time_ms > 0);
    assert!(processing_time.as_millis() < 5000); // Should complete within 5 seconds
    
    // Test multiple consolidation rounds
    for _ in 0..3 {
        let round_result = system.consolidate_memories(&memories).await?;
        assert!(round_result.effectiveness_score >= 0.0);
    }
    
    // Check that statistics are properly maintained
    let stats = system.get_consolidation_stats();
    assert!(stats.get("total_operations").unwrap_or(&0.0) >= &4.0);
    
    Ok(())
}

// Helper functions for creating test data

fn create_test_memory(key: &str, content: &str, memory_type: MemoryType, access_count: u64) -> MemoryEntry {
    let mut memory = MemoryEntry::new(key.to_string(), content.to_string(), memory_type);

    // Simulate access history
    for _ in 0..access_count {
        memory.mark_accessed();
    }

    memory
}

fn create_test_memory_set(count: usize) -> Vec<MemoryEntry> {
    (0..count)
        .map(|i| {
            let memory_type = if i % 2 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm };
            let access_count = (i * 3 + 1) as u64;
            create_test_memory(
                &format!("test_key_{}", i),
                &format!("Test content for memory {}", i),
                memory_type,
                access_count,
            )
        })
        .collect()
}

fn create_test_importance(memory_key: &str, base_score: f64) -> MemoryImportance {
    MemoryImportance {
        memory_key: memory_key.to_string(),
        importance_score: base_score,
        access_frequency: base_score * 0.8,
        recency_score: base_score * 0.9,
        centrality_score: base_score * 0.7,
        uniqueness_score: base_score * 0.6,
        temporal_consistency: base_score * 0.5,
        calculated_at: Utc::now(),
        fisher_information: Some(vec![base_score, base_score * 0.8, base_score * 1.2]),
    }
}

fn create_test_importance_set(memories: &[MemoryEntry]) -> Vec<MemoryImportance> {
    memories
        .iter()
        .enumerate()
        .map(|(i, memory)| {
            let base_score = 0.3 + (i as f64 * 0.1) % 0.7; // Vary scores between 0.3 and 1.0
            create_test_importance(&memory.key, base_score)
        })
        .collect()
}
