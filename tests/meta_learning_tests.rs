//! Comprehensive tests for meta-learning algorithms
//! 
//! This module tests MAML, Reptile, and Prototypical Networks implementations
//! for few-shot learning in memory management tasks.

use synaptic::memory::meta_learning::{
    MetaLearningSystem, MetaLearningConfig, MetaTask, TaskType, MetaAlgorithm,
    AdaptationResult, MetaLearningMetrics, task_distribution::{TaskDistribution, SamplingStrategy},
};
use synaptic::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
use synaptic::error::Result;
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

/// Create test memory entries for meta-learning tasks
fn create_test_memories(count: usize, memory_type: MemoryType, domain: &str) -> Vec<MemoryEntry> {
    let mut memories = Vec::new();
    
    for i in 0..count {
        let content = match domain {
            "technical" => format!("Technical documentation about API endpoint #{}: This endpoint handles user authentication and returns JWT tokens for secure access.", i),
            "personal" => format!("Personal note #{}: Remember to call mom on Sunday and discuss vacation plans for next month.", i),
            "business" => format!("Business meeting #{}: Quarterly review with stakeholders to discuss revenue targets and market expansion strategies.", i),
            _ => format!("General content #{}: This is a sample memory entry for testing purposes with various content patterns.", i),
        };
        
        let mut metadata = MemoryMetadata::new();
        metadata.access_count = (i % 10) as u64;
        metadata.tags = vec![domain.to_string(), format!("test_{}", i)];
        metadata.importance = 0.5 + (i as f64 / count as f64) * 0.5;
        metadata.set_custom_field("context".to_string(), format!("{}_context", domain));
        
        let entry = MemoryEntry::new(
            format!("{}_{}", domain, i),
            content,
            memory_type.clone(),
        ).with_metadata(metadata);
        
        memories.push(entry);
    }
    
    memories
}

/// Create a test meta-learning task
fn create_test_task(
    task_id: &str,
    task_type: TaskType,
    domain: &str,
    support_size: usize,
    query_size: usize,
) -> MetaTask {
    let total_memories = support_size + query_size;
    let memories = create_test_memories(total_memories, MemoryType::LongTerm, domain);
    
    let support_set = memories[..support_size].to_vec();
    let query_set = memories[support_size..].to_vec();
    
    MetaTask {
        id: task_id.to_string(),
        task_type,
        support_set,
        query_set,
        metadata: HashMap::new(),
        created_at: Utc::now(),
        difficulty: 0.5,
        domain: domain.to_string(),
    }
}

#[tokio::test]
async fn test_maml_meta_learning() -> Result<()> {
    // Create MAML configuration
    let config = MetaLearningConfig {
        inner_learning_rate: 0.01,
        outer_learning_rate: 0.001,
        inner_steps: 3,
        meta_batch_size: 2,
        support_set_size: 3,
        query_set_size: 5,
        max_meta_iterations: 10,
        convergence_threshold: 0.1,
        second_order: false, // Use first-order for faster testing
        adaptation_timeout_ms: 1000,
    };
    
    // Create MAML system
    let mut meta_system = MetaLearningSystem::new(config, MetaAlgorithm::MAML)?;
    
    // Create training tasks
    let training_tasks = vec![
        create_test_task("task_1", TaskType::Classification, "technical", 3, 5),
        create_test_task("task_2", TaskType::Classification, "personal", 3, 5),
        create_test_task("task_3", TaskType::Classification, "business", 3, 5),
        create_test_task("task_4", TaskType::Classification, "technical", 3, 5),
    ];
    
    // Train the meta-learner
    let training_metrics = meta_system.train(&training_tasks).await?;
    
    // Verify training metrics
    assert!(training_metrics.meta_iterations > 0);
    assert!(training_metrics.avg_adaptation_loss >= 0.0);
    assert!(training_metrics.adaptation_success_rate >= 0.0);
    assert!(training_metrics.adaptation_success_rate <= 1.0);
    
    println!("MAML Training completed:");
    println!("  Meta-iterations: {}", training_metrics.meta_iterations);
    println!("  Average loss: {:.4}", training_metrics.avg_adaptation_loss);
    println!("  Success rate: {:.2}%", training_metrics.adaptation_success_rate * 100.0);
    
    // Test adaptation to new task
    let new_task = create_test_task("new_task", TaskType::Classification, "technical", 3, 5);
    let adaptation_result = meta_system.adapt_to_new_task(&new_task).await?;
    
    // Verify adaptation result
    assert_eq!(adaptation_result.task_id, "new_task");
    assert!(adaptation_result.adaptation_steps > 0);
    assert!(adaptation_result.final_loss >= 0.0);
    assert!(adaptation_result.confidence >= 0.0);
    assert!(adaptation_result.confidence <= 1.0);
    
    println!("MAML Adaptation result:");
    println!("  Final loss: {:.4}", adaptation_result.final_loss);
    println!("  Confidence: {:.4}", adaptation_result.confidence);
    println!("  Success: {}", adaptation_result.success);
    
    Ok(())
}

#[tokio::test]
async fn test_reptile_meta_learning() -> Result<()> {
    // Create Reptile configuration
    let config = MetaLearningConfig {
        inner_learning_rate: 0.02,
        outer_learning_rate: 0.001,
        inner_steps: 5,
        meta_batch_size: 1, // Reptile typically uses batch size 1
        support_set_size: 4,
        query_set_size: 6,
        max_meta_iterations: 15,
        convergence_threshold: 0.1,
        second_order: false,
        adaptation_timeout_ms: 1000,
    };
    
    // Create Reptile system
    let mut meta_system = MetaLearningSystem::new(config, MetaAlgorithm::Reptile)?;
    
    // Create training tasks
    let training_tasks = vec![
        create_test_task("reptile_1", TaskType::Regression, "technical", 4, 6),
        create_test_task("reptile_2", TaskType::Regression, "personal", 4, 6),
        create_test_task("reptile_3", TaskType::Regression, "business", 4, 6),
    ];
    
    // Train the meta-learner
    let training_metrics = meta_system.train(&training_tasks).await?;
    
    // Verify training metrics
    assert!(training_metrics.meta_iterations > 0);
    assert!(training_metrics.memory_efficiency > 0.0);
    
    println!("Reptile Training completed:");
    println!("  Meta-iterations: {}", training_metrics.meta_iterations);
    println!("  Memory efficiency: {:.4}", training_metrics.memory_efficiency);
    println!("  Generalization score: {:.4}", training_metrics.generalization_score);
    
    // Test adaptation
    let new_task = create_test_task("reptile_new", TaskType::Regression, "technical", 4, 6);
    let adaptation_result = meta_system.adapt_to_new_task(&new_task).await?;
    
    // Verify adaptation
    assert!(adaptation_result.adaptation_time_ms > 0);
    assert!(adaptation_result.metrics.contains_key("final_loss"));
    
    println!("Reptile Adaptation result:");
    println!("  Adaptation time: {}ms", adaptation_result.adaptation_time_ms);
    println!("  Final loss: {:.4}", adaptation_result.final_loss);
    
    Ok(())
}

#[tokio::test]
async fn test_prototypical_networks() -> Result<()> {
    // Create Prototypical Networks configuration
    let config = MetaLearningConfig {
        inner_learning_rate: 0.001, // Not used for prototypical networks
        outer_learning_rate: 0.001,
        inner_steps: 1, // Prototypical networks adapt in one step
        meta_batch_size: 3,
        support_set_size: 2,
        query_set_size: 8,
        max_meta_iterations: 20,
        convergence_threshold: 0.1,
        second_order: false,
        adaptation_timeout_ms: 500,
    };
    
    // Create Prototypical Networks system
    let mut meta_system = MetaLearningSystem::new(config, MetaAlgorithm::Prototypical)?;
    
    // Create training tasks with different domains
    let training_tasks = vec![
        create_test_task("proto_1", TaskType::Classification, "technical", 2, 8),
        create_test_task("proto_2", TaskType::Classification, "personal", 2, 8),
        create_test_task("proto_3", TaskType::Classification, "business", 2, 8),
        create_test_task("proto_4", TaskType::Classification, "technical", 2, 8),
        create_test_task("proto_5", TaskType::Classification, "personal", 2, 8),
    ];
    
    // Train the meta-learner
    let training_metrics = meta_system.train(&training_tasks).await?;
    
    // Verify training metrics
    assert!(training_metrics.meta_iterations > 0);
    assert!(training_metrics.avg_adaptation_time_ms > 0.0);
    
    println!("Prototypical Networks Training completed:");
    println!("  Meta-iterations: {}", training_metrics.meta_iterations);
    println!("  Average adaptation time: {:.2}ms", training_metrics.avg_adaptation_time_ms);
    println!("  Success rate: {:.2}%", training_metrics.adaptation_success_rate * 100.0);
    
    // Test fast adaptation
    let new_task = create_test_task("proto_new", TaskType::Classification, "business", 2, 8);
    let adaptation_result = meta_system.adapt_to_new_task(&new_task).await?;
    
    // Verify fast adaptation (should be 1 step)
    assert_eq!(adaptation_result.adaptation_steps, 1);
    assert!(adaptation_result.metrics.contains_key("num_prototypes"));
    
    println!("Prototypical Networks Adaptation result:");
    println!("  Adaptation steps: {}", adaptation_result.adaptation_steps);
    println!("  Confidence: {:.4}", adaptation_result.confidence);
    
    Ok(())
}

#[tokio::test]
async fn test_task_distribution_management() -> Result<()> {
    let mut task_distribution = TaskDistribution::new();
    
    // Create diverse tasks
    let tasks = vec![
        create_test_task("dist_1", TaskType::Classification, "technical", 3, 7),
        create_test_task("dist_2", TaskType::Regression, "personal", 4, 6),
        create_test_task("dist_3", TaskType::Ranking, "business", 2, 8),
        create_test_task("dist_4", TaskType::Classification, "technical", 5, 5),
        create_test_task("dist_5", TaskType::PatternRecognition, "personal", 3, 7),
    ];
    
    // Update distribution
    task_distribution.update_distribution(&tasks).await?;
    
    // Test statistics
    let stats = task_distribution.get_statistics();
    assert_eq!(stats.total_created, 5);
    assert!(stats.by_type.len() > 0);
    assert!(stats.by_domain.len() > 0);

    println!("Task Distribution Statistics:");
    println!("  Total tasks: {}", stats.total_created);
    println!("  Average difficulty: {:.4}", stats.avg_difficulty);
    
    // Test different sampling strategies
    let uniform_samples = task_distribution.sample_tasks(3, SamplingStrategy::Uniform).await?;
    assert_eq!(uniform_samples.len(), 3);
    
    let curriculum_samples = task_distribution.sample_tasks(3, SamplingStrategy::Curriculum).await?;
    assert_eq!(curriculum_samples.len(), 3);
    
    let domain_balanced_samples = task_distribution.sample_tasks(3, SamplingStrategy::DomainBalanced).await?;
    assert!(domain_balanced_samples.len() <= 3);
    
    println!("Sampling strategies tested successfully");
    
    // Test task creation
    let memories = create_test_memories(10, MemoryType::LongTerm, "test_domain");
    let created_task = task_distribution.create_task(
        "created_task".to_string(),
        TaskType::Classification,
        memories,
        "test_domain".to_string(),
        0.3, // 30% support, 70% query
    ).await?;
    
    assert_eq!(created_task.id, "created_task");
    assert_eq!(created_task.domain, "test_domain");
    assert!(created_task.support_set.len() > 0);
    assert!(created_task.query_set.len() > 0);
    assert!(created_task.difficulty >= 0.0 && created_task.difficulty <= 1.0);
    
    println!("Task creation tested successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_meta_learning_evaluation() -> Result<()> {
    // Create a simple meta-learning system for evaluation
    let config = MetaLearningConfig {
        inner_learning_rate: 0.01,
        outer_learning_rate: 0.001,
        inner_steps: 2,
        meta_batch_size: 2,
        support_set_size: 3,
        query_set_size: 5,
        max_meta_iterations: 5,
        convergence_threshold: 0.2,
        second_order: false,
        adaptation_timeout_ms: 1000,
    };
    
    let mut meta_system = MetaLearningSystem::new(config, MetaAlgorithm::MAML)?;
    
    // Create training tasks
    let training_tasks = vec![
        create_test_task("eval_train_1", TaskType::Classification, "technical", 3, 5),
        create_test_task("eval_train_2", TaskType::Classification, "personal", 3, 5),
    ];
    
    // Train briefly
    let _training_metrics = meta_system.train(&training_tasks).await?;
    
    // Create evaluation tasks
    let eval_tasks = vec![
        create_test_task("eval_test_1", TaskType::Classification, "business", 3, 5),
        create_test_task("eval_test_2", TaskType::Classification, "technical", 3, 5),
    ];
    
    // Evaluate performance
    let eval_metrics = meta_system.evaluate(&eval_tasks).await?;
    
    // Verify evaluation metrics
    assert!(eval_metrics.avg_adaptation_loss >= 0.0);
    assert!(eval_metrics.adaptation_success_rate >= 0.0);
    assert!(eval_metrics.adaptation_success_rate <= 1.0);
    assert!(eval_metrics.avg_adaptation_time_ms > 0.0);
    
    println!("Meta-Learning Evaluation:");
    println!("  Average loss: {:.4}", eval_metrics.avg_adaptation_loss);
    println!("  Success rate: {:.2}%", eval_metrics.adaptation_success_rate * 100.0);
    println!("  Average time: {:.2}ms", eval_metrics.avg_adaptation_time_ms);
    
    // Test adaptation history
    let history = meta_system.get_adaptation_history().await;
    assert!(history.len() >= eval_tasks.len());
    
    // Test metrics retrieval
    let current_metrics = meta_system.get_metrics().await;
    assert!(current_metrics.meta_iterations > 0);
    
    // Test parameter save/load
    let saved_params = meta_system.save_meta_parameters().await?;
    assert!(!saved_params.is_empty());
    
    let mut new_system = MetaLearningSystem::new(
        MetaLearningConfig::default(),
        MetaAlgorithm::MAML
    )?;
    new_system.load_meta_parameters(saved_params).await?;
    
    println!("Parameter save/load tested successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_meta_learning_performance_comparison() -> Result<()> {
    let config = MetaLearningConfig {
        inner_learning_rate: 0.01,
        outer_learning_rate: 0.001,
        inner_steps: 3,
        meta_batch_size: 2,
        support_set_size: 3,
        query_set_size: 5,
        max_meta_iterations: 8,
        convergence_threshold: 0.15,
        second_order: false,
        adaptation_timeout_ms: 1000,
    };
    
    // Test all three algorithms
    let algorithms = vec![
        ("MAML", MetaAlgorithm::MAML),
        ("Reptile", MetaAlgorithm::Reptile),
        ("Prototypical", MetaAlgorithm::Prototypical),
    ];
    
    let training_tasks = vec![
        create_test_task("comp_1", TaskType::Classification, "technical", 3, 5),
        create_test_task("comp_2", TaskType::Classification, "personal", 3, 5),
        create_test_task("comp_3", TaskType::Classification, "business", 3, 5),
    ];
    
    let test_task = create_test_task("comp_test", TaskType::Classification, "technical", 3, 5);
    
    println!("Meta-Learning Algorithm Comparison:");
    
    for (name, algorithm) in algorithms {
        let mut system = MetaLearningSystem::new(config.clone(), algorithm)?;
        
        let start_time = std::time::Instant::now();
        let training_metrics = system.train(&training_tasks).await?;
        let training_time = start_time.elapsed();
        
        let adaptation_result = system.adapt_to_new_task(&test_task).await?;
        
        println!("  {}:", name);
        println!("    Training time: {:?}", training_time);
        println!("    Meta-iterations: {}", training_metrics.meta_iterations);
        println!("    Final loss: {:.4}", adaptation_result.final_loss);
        println!("    Adaptation time: {}ms", adaptation_result.adaptation_time_ms);
        println!("    Success: {}", adaptation_result.success);
        println!("    Memory efficiency: {:.4}", training_metrics.memory_efficiency);
    }
    
    Ok(())
}
