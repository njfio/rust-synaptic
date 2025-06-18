//! Comprehensive tests for Few-Shot Learning System
//! 
//! Tests prototype networks, matching networks, relation networks, and memory-augmented
//! few-shot learning capabilities with comprehensive validation.

use synaptic::memory::meta_learning::{
    FewShotLearningEngine, FewShotConfig, FewShotAlgorithm, FewShotEpisode,
    SupportExample, QueryExample, DistanceMetric, AttentionType, ActivationFunction,
    PrototypeUpdateStrategy
};
use std::collections::HashMap;
use chrono::Utc;

/// Create test few-shot learning configuration
fn create_test_config() -> FewShotConfig {
    FewShotConfig {
        support_shots: 3,
        num_ways: 3,
        query_shots: 5,
        embedding_dim: 64,
        adaptation_lr: 0.01,
        adaptation_steps: 5,
        temperature: 1.0,
        use_memory_augmentation: true,
        memory_bank_size: 100,
    }
}

/// Create test support examples
fn create_support_examples(num_classes: usize, shots_per_class: usize, feature_dim: usize) -> Vec<SupportExample> {
    let mut examples = Vec::new();
    
    for class_id in 0..num_classes {
        for shot in 0..shots_per_class {
            let mut features = vec![0.0; feature_dim];
            
            // Create class-specific patterns
            for i in 0..feature_dim {
                features[i] = (class_id as f64 + 1.0) * 0.5 + (shot as f64 * 0.1) + (i as f64 * 0.01);
            }
            
            examples.push(SupportExample {
                id: format!("support_{}_{}", class_id, shot),
                features,
                label: format!("class_{}", class_id),
                confidence: 0.8 + (shot as f64 * 0.05),
                metadata: HashMap::from([
                    ("class_id".to_string(), class_id.to_string()),
                    ("shot_id".to_string(), shot.to_string()),
                ]),
            });
        }
    }
    
    examples
}

/// Create test query examples
fn create_query_examples(num_classes: usize, queries_per_class: usize, feature_dim: usize) -> Vec<QueryExample> {
    let mut examples = Vec::new();
    
    for class_id in 0..num_classes {
        for query in 0..queries_per_class {
            let mut features = vec![0.0; feature_dim];
            
            // Create class-specific patterns with some noise
            for i in 0..feature_dim {
                features[i] = (class_id as f64 + 1.0) * 0.5 + (query as f64 * 0.05) + (i as f64 * 0.01) + 
                             (rand::random::<f64>() - 0.5) * 0.1; // Add noise
            }
            
            examples.push(QueryExample {
                id: format!("query_{}_{}", class_id, query),
                features,
                true_label: Some(format!("class_{}", class_id)),
                metadata: HashMap::from([
                    ("class_id".to_string(), class_id.to_string()),
                    ("query_id".to_string(), query.to_string()),
                ]),
            });
        }
    }
    
    examples
}

/// Create test episode
fn create_test_episode(config: &FewShotConfig) -> FewShotEpisode {
    let support_set = create_support_examples(config.num_ways, config.support_shots, config.embedding_dim);
    let query_set = create_query_examples(config.num_ways, config.query_shots, config.embedding_dim);
    
    FewShotEpisode {
        id: "test_episode_001".to_string(),
        support_set,
        query_set,
        metadata: HashMap::from([
            ("episode_type".to_string(), "test".to_string()),
            ("difficulty".to_string(), "medium".to_string()),
        ]),
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn test_few_shot_engine_creation() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let engine = FewShotLearningEngine::new(config)?;
    
    // Verify engine initialization
    assert_eq!(engine.get_metrics().total_episodes, 0);
    assert_eq!(engine.get_episode_history().len(), 0);
    assert_eq!(engine.get_memory_bank().examples.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_prototype_network_learning() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;
    
    // Set prototype network algorithm
    let algorithm = FewShotAlgorithm::PrototypeNetwork {
        distance_metric: DistanceMetric::Euclidean,
        prototype_update_strategy: PrototypeUpdateStrategy::Mean,
    };
    engine.set_algorithm(algorithm);
    
    // Create and process episode
    let episode = create_test_episode(&config);
    let result = engine.process_episode(episode).await?;
    
    // Verify results
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    assert!(result.adaptation_time_ms > 0);
    assert!(result.inference_time_ms > 0);
    
    // Check that prototypes were created
    assert!(engine.get_memory_bank().prototypes.len() > 0);
    
    // Verify per-class accuracy
    assert_eq!(result.per_class_accuracy.len(), config.num_ways);
    
    Ok(())
}

#[tokio::test]
async fn test_matching_network_learning() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;
    
    // Set matching network algorithm
    let algorithm = FewShotAlgorithm::MatchingNetwork {
        attention_type: AttentionType::ScaledDotProduct,
        bidirectional_encoding: true,
        context_encoding: true,
    };
    engine.set_algorithm(algorithm);
    
    // Create and process episode
    let episode = create_test_episode(&config);
    let result = engine.process_episode(episode).await?;
    
    // Verify results
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    assert!(result.adaptation_time_ms > 0);
    assert!(result.inference_time_ms > 0);
    
    // Check predictions have confidence scores
    for prediction in &result.predictions {
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(!prediction.class_probabilities.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_relation_network_learning() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;
    
    // Set relation network algorithm
    let algorithm = FewShotAlgorithm::RelationNetwork {
        relation_module_layers: vec![256, 128, 64, 1],
        embedding_layers: vec![128, 64],
        activation_function: ActivationFunction::ReLU,
    };
    engine.set_algorithm(algorithm);
    
    // Create and process episode
    let episode = create_test_episode(&config);
    let result = engine.process_episode(episode).await?;
    
    // Verify results
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
    assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    assert!(result.adaptation_time_ms > 0);
    assert!(result.inference_time_ms > 0);
    
    // Check that relation scores are reasonable
    for prediction in &result.predictions {
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.metadata.contains_key("max_relation_score"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_memory_augmentation() -> synaptic::error::Result<()> {
    let mut config = create_test_config();
    config.use_memory_augmentation = true;
    config.memory_bank_size = 50;
    
    let mut engine = FewShotLearningEngine::new(config.clone())?;
    
    // Process multiple episodes to build memory
    for i in 0..3 {
        let mut episode = create_test_episode(&config);
        episode.id = format!("episode_{}", i);
        
        let result = engine.process_episode(episode).await?;
        assert!(result.accuracy >= 0.0);
    }
    
    // Check memory bank utilization
    let metrics = engine.get_metrics();
    assert!(metrics.memory_utilization > 0.0);
    assert!(engine.get_memory_bank().examples.len() > 0);
    
    // Test memory bank clearing
    engine.clear_memory_bank().await?;
    assert_eq!(engine.get_memory_bank().examples.len(), 0);
    assert_eq!(engine.get_memory_bank().prototypes.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_distance_metrics() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let distance_metrics = vec![
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Manhattan,
        DistanceMetric::Mahalanobis,
    ];
    
    for metric in distance_metrics {
        let mut engine = FewShotLearningEngine::new(config.clone())?;
        
        let algorithm = FewShotAlgorithm::PrototypeNetwork {
            distance_metric: metric,
            prototype_update_strategy: PrototypeUpdateStrategy::Mean,
        };
        engine.set_algorithm(algorithm);
        
        let episode = create_test_episode(&config);
        let result = engine.process_episode(episode).await?;
        
        // All distance metrics should produce valid results
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_attention_mechanisms() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let attention_types = vec![
        AttentionType::Additive,
        AttentionType::Multiplicative,
        AttentionType::ScaledDotProduct,
        AttentionType::MultiHead { num_heads: 4 },
    ];
    
    for attention_type in attention_types {
        let mut engine = FewShotLearningEngine::new(config.clone())?;
        
        let algorithm = FewShotAlgorithm::MatchingNetwork {
            attention_type,
            bidirectional_encoding: false,
            context_encoding: false,
        };
        engine.set_algorithm(algorithm);
        
        let episode = create_test_episode(&config);
        let result = engine.process_episode(episode).await?;
        
        // All attention mechanisms should produce valid results
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_parameter_export_import() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine1 = FewShotLearningEngine::new(config.clone())?;

    // Process an episode to train the model
    let episode = create_test_episode(&config);
    let result1 = engine1.process_episode(episode.clone()).await?;

    // Export parameters
    let parameters = engine1.export_parameters();
    assert!(!parameters.is_empty());

    // Create new engine and import parameters
    let mut engine2 = FewShotLearningEngine::new(config)?;
    engine2.import_parameters(parameters);

    // Process same episode with imported parameters
    let result2 = engine2.process_episode(episode).await?;

    // Results should be similar (allowing for some numerical differences)
    let accuracy_diff = (result1.accuracy - result2.accuracy).abs();
    assert!(accuracy_diff < 0.1, "Accuracy difference too large: {}", accuracy_diff);

    Ok(())
}

#[tokio::test]
async fn test_metrics_tracking() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;

    // Process multiple episodes
    let num_episodes = 5;
    for i in 0..num_episodes {
        let mut episode = create_test_episode(&config);
        episode.id = format!("metrics_episode_{}", i);

        let result = engine.process_episode(episode).await?;
        assert!(result.accuracy >= 0.0);
    }

    // Check metrics
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_episodes, num_episodes);
    assert!(metrics.avg_accuracy >= 0.0 && metrics.avg_accuracy <= 1.0);
    assert!(metrics.avg_adaptation_time_ms > 0.0);
    assert!(metrics.avg_inference_time_ms > 0.0);

    // Check episode history
    let history = engine.get_episode_history();
    assert_eq!(history.len(), num_episodes);

    Ok(())
}

#[tokio::test]
async fn test_prototype_update_strategies() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let strategies = vec![
        PrototypeUpdateStrategy::Mean,
        PrototypeUpdateStrategy::WeightedMean,
        PrototypeUpdateStrategy::ExponentialMovingAverage { alpha: 0.7 },
        PrototypeUpdateStrategy::AttentionWeighted,
    ];

    for strategy in strategies {
        let mut engine = FewShotLearningEngine::new(config.clone())?;

        let algorithm = FewShotAlgorithm::PrototypeNetwork {
            distance_metric: DistanceMetric::Euclidean,
            prototype_update_strategy: strategy,
        };
        engine.set_algorithm(algorithm);

        let episode = create_test_episode(&config);
        let result = engine.process_episode(episode).await?;

        // All strategies should produce valid results
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert!(!engine.get_memory_bank().prototypes.is_empty());
    }

    Ok(())
}

#[tokio::test]
async fn test_activation_functions() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let activations = vec![
        ActivationFunction::ReLU,
        ActivationFunction::LeakyReLU { negative_slope: 0.01 },
        ActivationFunction::Tanh,
        ActivationFunction::Sigmoid,
        ActivationFunction::Swish,
    ];

    for activation in activations {
        let mut engine = FewShotLearningEngine::new(config.clone())?;

        let algorithm = FewShotAlgorithm::RelationNetwork {
            relation_module_layers: vec![128, 64, 1],
            embedding_layers: vec![64],
            activation_function: activation,
        };
        engine.set_algorithm(algorithm);

        let episode = create_test_episode(&config);
        let result = engine.process_episode(episode).await?;

        // All activation functions should produce valid results
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    }

    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;

    // Test empty support set
    let mut episode = create_test_episode(&config);
    episode.support_set.clear();

    let result = engine.process_episode(episode).await;
    assert!(result.is_err());

    // Test empty query set
    let mut episode = create_test_episode(&config);
    episode.query_set.clear();

    let result = engine.process_episode(episode).await;
    assert!(result.is_err());

    // Test mismatched feature dimensions
    let mut episode = create_test_episode(&config);
    episode.support_set[0].features = vec![0.0; 32]; // Wrong dimension

    let result = engine.process_episode(episode).await;
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_performance_consistency() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;

    // Set deterministic algorithm
    let algorithm = FewShotAlgorithm::PrototypeNetwork {
        distance_metric: DistanceMetric::Euclidean,
        prototype_update_strategy: PrototypeUpdateStrategy::Mean,
    };
    engine.set_algorithm(algorithm);

    // Process same episode multiple times
    let episode = create_test_episode(&config);
    let mut accuracies = Vec::new();

    for _ in 0..3 {
        // Clear memory bank to ensure consistent starting state
        engine.clear_memory_bank().await?;

        let result = engine.process_episode(episode.clone()).await?;
        accuracies.push(result.accuracy);
    }

    // Results should be consistent for deterministic algorithm
    let accuracy_variance = accuracies.iter()
        .map(|&acc| (acc - accuracies[0]).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    assert!(accuracy_variance < 0.01, "Accuracy variance too high: {}", accuracy_variance);

    Ok(())
}

#[tokio::test]
async fn test_memory_bank_size_limits() -> synaptic::error::Result<()> {
    let mut config = create_test_config();
    config.memory_bank_size = 10; // Small memory bank
    config.use_memory_augmentation = true;

    let mut engine = FewShotLearningEngine::new(config.clone())?;

    // Process many episodes to exceed memory bank size
    for i in 0..5 {
        let mut episode = create_test_episode(&config);
        episode.id = format!("memory_test_{}", i);

        let result = engine.process_episode(episode).await?;
        assert!(result.accuracy >= 0.0);
    }

    // Memory bank should not exceed size limit
    assert!(engine.get_memory_bank().examples.len() <= config.memory_bank_size);

    Ok(())
}

#[tokio::test]
async fn test_algorithm_switching() -> synaptic::error::Result<()> {
    let config = create_test_config();
    let mut engine = FewShotLearningEngine::new(config.clone())?;

    let algorithms = vec![
        FewShotAlgorithm::PrototypeNetwork {
            distance_metric: DistanceMetric::Euclidean,
            prototype_update_strategy: PrototypeUpdateStrategy::Mean,
        },
        FewShotAlgorithm::MatchingNetwork {
            attention_type: AttentionType::ScaledDotProduct,
            bidirectional_encoding: false,
            context_encoding: false,
        },
        FewShotAlgorithm::RelationNetwork {
            relation_module_layers: vec![128, 64, 1],
            embedding_layers: vec![64],
            activation_function: ActivationFunction::ReLU,
        },
    ];

    let episode = create_test_episode(&config);

    // Test switching between algorithms
    for algorithm in algorithms {
        engine.set_algorithm(algorithm);

        let result = engine.process_episode(episode.clone()).await?;
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
        assert_eq!(result.predictions.len(), config.num_ways * config.query_shots);
    }

    Ok(())
}
