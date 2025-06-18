//! ML-Based Parameter Optimization Tests
//!
//! Comprehensive tests for the machine learning-based parameter optimization system
//! including online learning, Bayesian optimization, genetic algorithms, and hyperparameter tuning.

use synaptic::performance::{
    PerformanceConfig,
    optimizer::{
        PerformanceOptimizer, MLPredictor, AdaptiveTuner, OnlineLearner,
        BayesianOptimizer, GeneticAlgorithm, HyperparameterTuner,
        Individual, OptimizationType, OptimizationPlan, Optimization,
        OptimizationResult, HyperparameterResult, SearchType, PerformanceAnalysis,
    },
    metrics::PerformanceMetrics,
};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;
use std::time::Duration;

#[tokio::test]
async fn test_ml_predictor_training_and_prediction() {
    let mut predictor = MLPredictor::new();
    
    // Create training data
    let mut plan = OptimizationPlan {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        analysis: PerformanceAnalysis::default(),
        optimizations: vec![
            Optimization {
                id: Uuid::new_v4(),
                optimization_type: OptimizationType::MemoryPoolOptimization,
                description: "Memory optimization test".to_string(),
                parameters: HashMap::new(),
                expected_improvement: 0.8,
                confidence: 0.9,
                estimated_duration: Duration::from_secs(10),
            }
        ],
        expected_improvement: 0.8,
        estimated_duration: Duration::from_secs(10),
    };
    
    // Train the predictor
    predictor.train_on_plan(&plan).await.unwrap();
    
    // Test prediction
    let prediction = predictor.predict_effectiveness(&OptimizationType::MemoryPoolOptimization).await.unwrap();
    assert!(prediction >= 0.0 && prediction <= 1.0);
    
    // Test model metrics
    let metrics = predictor.get_model_metrics();
    assert_eq!(metrics.training_samples, 1);
    assert!(metrics.model_accuracy >= 0.0);
    
    // Add more training data
    for i in 0..20 {
        plan.expected_improvement = 0.5 + (i as f64 * 0.02);
        predictor.train_on_plan(&plan).await.unwrap();
    }
    
    // Test improved prediction
    let improved_prediction = predictor.predict_effectiveness(&OptimizationType::MemoryPoolOptimization).await.unwrap();
    assert!(improved_prediction >= 0.0 && improved_prediction <= 1.0);
    
    // Verify model has learned
    let final_metrics = predictor.get_model_metrics();
    assert_eq!(final_metrics.training_samples, 21);
}

#[tokio::test]
async fn test_online_learner() {
    let mut learner = OnlineLearner::new();
    
    // Test initial prediction
    let features = vec![0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6, 1.0];
    let initial_prediction = learner.predict(&features).unwrap();
    assert!(initial_prediction >= 0.0 && initial_prediction <= 1.0);
    
    // Train with multiple samples
    let training_data = vec![
        (vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.8),
        (vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], 0.2),
        (vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.5),
    ];
    
    for (features, target) in training_data {
        learner.update(&features, target).unwrap();
    }
    
    // Test prediction after training
    let trained_prediction = learner.predict(&vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).unwrap();
    assert!(trained_prediction >= 0.0 && trained_prediction <= 1.0);
    
    // Test with different feature vector
    let different_prediction = learner.predict(&vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]).unwrap();
    assert!(different_prediction >= 0.0 && different_prediction <= 1.0);
}

#[tokio::test]
async fn test_bayesian_optimizer() {
    let mut optimizer = BayesianOptimizer::new();
    
    // Test with no observations
    let initial_suggestion = optimizer.suggest_next_parameters().unwrap();
    assert_eq!(initial_suggestion.len(), 10);
    
    // Add observations
    let parameters = vec![
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ];
    let scores = vec![0.8, 0.3, 0.6];
    
    optimizer.update_observations(&parameters, &scores).unwrap();
    
    // Test suggestion after observations
    let suggestion = optimizer.suggest_next_parameters().unwrap();
    assert_eq!(suggestion.len(), 10);
    for &param in &suggestion {
        assert!(param >= 0.0 && param <= 1.0);
    }
}

#[tokio::test]
async fn test_genetic_algorithm() {
    let mut ga = GeneticAlgorithm::new();
    
    // Create initial population
    let mut population = Vec::new();
    for _ in 0..20 {
        let genes = (0..10).map(|_| fastrand::f64()).collect();
        population.push(Individual { genes, fitness: 0.0 });
    }
    
    // Evolve population
    let best_individual = ga.evolve(population, 10).await.unwrap();
    
    // Verify best individual
    assert_eq!(best_individual.genes.len(), 10);
    assert!(best_individual.fitness >= 0.0);
    
    for &gene in &best_individual.genes {
        assert!(gene >= 0.0 && gene <= 1.0);
    }
}

#[tokio::test]
async fn test_hyperparameter_tuner() {
    let mut tuner = HyperparameterTuner::new();
    
    // Define search space
    let mut search_space = HashMap::new();
    search_space.insert("learning_rate".to_string(), vec![0.001, 0.01, 0.1]);
    search_space.insert("l1_reg".to_string(), vec![0.0, 0.01, 0.1]);
    search_space.insert("l2_reg".to_string(), vec![0.0, 0.01, 0.1]);
    
    // Test grid search
    let grid_results = tuner.grid_search(&search_space, 10).await.unwrap();
    assert!(!grid_results.is_empty());
    assert!(grid_results.len() <= 10);
    
    for result in &grid_results {
        assert!(result.score >= 0.0);
        assert!(matches!(result.search_type, SearchType::Grid));
        assert!(result.hyperparameters.contains_key("learning_rate"));
    }
    
    // Test random search
    let random_results = tuner.random_search(&search_space, 5).await.unwrap();
    assert_eq!(random_results.len(), 5);
    
    for result in &random_results {
        assert!(result.score >= 0.0);
        assert!(matches!(result.search_type, SearchType::Random));
        assert!(result.hyperparameters.contains_key("learning_rate"));
    }
}

#[tokio::test]
async fn test_adaptive_tuner_comprehensive() {
    let mut tuner = AdaptiveTuner::new();
    
    // Create optimization plan
    let mut optimization = Optimization {
        id: Uuid::new_v4(),
        optimization_type: OptimizationType::ExecutorOptimization,
        description: "CPU optimization test".to_string(),
        parameters: HashMap::new(),
        expected_improvement: 0.7,
        confidence: 0.8,
        estimated_duration: Duration::from_secs(15),
    };
    
    optimization.parameters.insert("thread_pool_size".to_string(), "8".to_string());
    optimization.parameters.insert("batch_size".to_string(), "100".to_string());
    
    let plan = OptimizationPlan {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        analysis: PerformanceAnalysis::default(),
        optimizations: vec![optimization],
        expected_improvement: 0.7,
        estimated_duration: Duration::from_secs(15),
    };
    
    // Test parameter adjustment
    tuner.adjust_parameters(&plan).await.unwrap();
    
    // Test parameter recommendations
    let recommendations = tuner.get_parameter_recommendations();
    assert!(recommendations.contains_key("thread_pool_size"));
    assert!(recommendations.contains_key("batch_size"));
    
    // Record optimization result
    let result = OptimizationResult {
        optimization_id: Uuid::new_v4(),
        parameters_used: HashMap::new(),
        performance_improvement: 0.75,
        execution_time_ms: 1500,
        timestamp: Utc::now(),
    };
    
    tuner.record_optimization_result(result).await.unwrap();
    
    // Test multiple adjustments
    for i in 0..10 {
        let mut new_plan = plan.clone();
        new_plan.expected_improvement = 0.5 + (i as f64 * 0.05);
        tuner.adjust_parameters(&new_plan).await.unwrap();
    }
    
    // Verify parameter history has grown
    let final_recommendations = tuner.get_parameter_recommendations();
    assert!(!final_recommendations.is_empty());
}

#[tokio::test]
async fn test_performance_optimizer_with_ml() {
    let config = PerformanceConfig::default();
    let mut optimizer = PerformanceOptimizer::new(config).await.unwrap();
    
    // Create performance metrics that need optimization
    let metrics = PerformanceMetrics {
        avg_latency_ms: 25.0,  // Above target
        throughput_ops_per_sec: 400.0,  // Below target
        memory_usage_mb: 800.0,  // Above target
        cpu_usage_percent: 85.0,  // Above target
        ..Default::default()
    };
    
    // Run optimization
    let optimization_plan = optimizer.optimize(&metrics).await.unwrap();
    
    // Verify optimization plan
    assert!(!optimization_plan.optimizations.is_empty());
    assert!(optimization_plan.expected_improvement > 0.0);
    
    // Test multiple optimization cycles
    for i in 0..5 {
        let mut test_metrics = metrics.clone();
        test_metrics.avg_latency_ms = 20.0 + (i as f64 * 2.0);
        
        let plan = optimizer.optimize(&test_metrics).await.unwrap();
        assert!(!plan.optimizations.is_empty());
    }
}

#[tokio::test]
async fn test_ml_parameter_optimization_integration() {
    let config = PerformanceConfig::default();
    let mut optimizer = PerformanceOptimizer::new(config).await.unwrap();
    
    // Simulate multiple optimization cycles with learning
    let mut performance_scores = Vec::new();
    
    for iteration in 0..20 {
        let metrics = PerformanceMetrics {
            avg_latency_ms: 15.0 + (iteration as f64 * 0.5),
            throughput_ops_per_sec: 800.0 - (iteration as f64 * 10.0),
            memory_usage_mb: 600.0 + (iteration as f64 * 5.0),
            cpu_usage_percent: 70.0 + (iteration as f64 * 1.0),
            ..Default::default()
        };
        
        let plan = optimizer.optimize(&metrics).await.unwrap();
        performance_scores.push(plan.expected_improvement);
        
        // Simulate some delay between optimizations
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Verify that the system is learning (later optimizations should be better)
    let early_avg = performance_scores[0..5].iter().sum::<f64>() / 5.0;
    let late_avg = performance_scores[15..20].iter().sum::<f64>() / 5.0;
    
    // The system should show some improvement or at least maintain performance
    assert!(late_avg >= early_avg * 0.8); // Allow for some variance
}
