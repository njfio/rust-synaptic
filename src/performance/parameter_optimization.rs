//! Parameter Optimization Module
//!
//! This module implements ML-based parameter optimization with online learning,
//! hyperparameter tuning, and performance prediction.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};

/// Parameter optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Exploration rate for parameter space
    pub exploration_rate: f64,
    /// Minimum parameter bounds
    pub min_bounds: HashMap<String, f64>,
    /// Maximum parameter bounds
    pub max_bounds: HashMap<String, f64>,
}

/// Parameter optimizer using ML-based optimization
pub struct ParameterOptimizer {
    /// Configuration
    config: OptimizerConfig,
    /// Parameter history
    parameter_history: Vec<ParameterSnapshot>,
    /// Performance history
    performance_history: Vec<f64>,
    /// Current best parameters
    best_parameters: HashMap<String, f64>,
    /// Best performance achieved
    best_performance: f64,
}

/// Parameter snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSnapshot {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Performance score
    pub performance: f64,
    /// Optimization iteration
    pub iteration: usize,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Success status
    pub success: bool,
    /// Performance improvement
    pub improvement: f64,
    /// Updated parameters
    pub parameters_updated: HashMap<String, f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Optimization duration
    pub duration: Duration,
    /// Convergence achieved
    pub converged: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        let mut min_bounds = HashMap::new();
        min_bounds.insert("cache_size".to_string(), 1.0);
        min_bounds.insert("thread_count".to_string(), 1.0);
        min_bounds.insert("batch_size".to_string(), 1.0);
        
        let mut max_bounds = HashMap::new();
        max_bounds.insert("cache_size".to_string(), 1000.0);
        max_bounds.insert("thread_count".to_string(), 32.0);
        max_bounds.insert("batch_size".to_string(), 1000.0);
        
        Self {
            learning_rate: 0.01,
            max_iterations: 100,
            convergence_threshold: 0.001,
            exploration_rate: 0.1,
            min_bounds,
            max_bounds,
        }
    }
}

impl ParameterOptimizer {
    /// Create a new parameter optimizer
    pub async fn new(config: OptimizerConfig) -> Result<Self> {
        Ok(Self {
            config,
            parameter_history: Vec::new(),
            performance_history: Vec::new(),
            best_parameters: HashMap::new(),
            best_performance: 0.0,
        })
    }

    /// Optimize parameters using gradient-based optimization
    pub async fn optimize_parameters(
        &mut self,
        current_parameters: HashMap<String, f64>,
        context: HashMap<String, f64>,
    ) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        let mut parameters = current_parameters.clone();
        let mut best_performance = self.evaluate_parameters(&parameters, &context).await?;
        let mut best_params = parameters.clone();
        let mut converged = false;
        
        for iteration in 0..self.config.max_iterations {
            // Calculate gradients using finite differences
            let gradients = self.calculate_gradients(&parameters, &context).await?;
            
            // Update parameters using gradient descent
            let mut updated_parameters = HashMap::new();
            for (param_name, current_value) in &parameters {
                if let Some(gradient) = gradients.get(param_name) {
                    let new_value = current_value - self.config.learning_rate * gradient;
                    let bounded_value = self.apply_bounds(param_name, new_value);
                    updated_parameters.insert(param_name.clone(), bounded_value);
                } else {
                    updated_parameters.insert(param_name.clone(), *current_value);
                }
            }
            
            // Evaluate new parameters
            let performance = self.evaluate_parameters(&updated_parameters, &context).await?;
            
            // Record snapshot
            self.parameter_history.push(ParameterSnapshot {
                timestamp: Utc::now(),
                parameters: updated_parameters.clone(),
                performance,
                iteration,
            });
            self.performance_history.push(performance);
            
            // Update best if improved
            if performance > best_performance {
                best_performance = performance;
                best_params = updated_parameters.clone();
                self.best_parameters = best_params.clone();
                self.best_performance = best_performance;
            }
            
            // Check for convergence
            if iteration > 0 {
                let improvement = performance - self.performance_history[self.performance_history.len() - 2];
                if improvement.abs() < self.config.convergence_threshold {
                    converged = true;
                    break;
                }
            }
            
            parameters = updated_parameters;
        }
        
        let improvement = best_performance - self.evaluate_parameters(&current_parameters, &context).await?;
        
        Ok(OptimizationResult {
            success: improvement > 0.0,
            improvement,
            parameters_updated: best_params,
            iterations: self.parameter_history.len(),
            duration: start_time.elapsed(),
            converged,
        })
    }

    /// Calculate gradients using finite differences
    async fn calculate_gradients(
        &self,
        parameters: &HashMap<String, f64>,
        context: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut gradients = HashMap::new();
        let epsilon = 0.01; // Small perturbation for finite differences
        
        let base_performance = self.evaluate_parameters(parameters, context).await?;
        
        for (param_name, param_value) in parameters {
            // Create perturbed parameters
            let mut perturbed_params = parameters.clone();
            perturbed_params.insert(param_name.clone(), param_value + epsilon);
            
            // Evaluate perturbed parameters
            let perturbed_performance = self.evaluate_parameters(&perturbed_params, context).await?;
            
            // Calculate gradient
            let gradient = (perturbed_performance - base_performance) / epsilon;
            gradients.insert(param_name.clone(), gradient);
        }
        
        Ok(gradients)
    }

    /// Evaluate parameters and return performance score
    async fn evaluate_parameters(
        &self,
        parameters: &HashMap<String, f64>,
        context: &HashMap<String, f64>,
    ) -> Result<f64> {
        // Simplified performance evaluation - in a real implementation,
        // this would run actual performance tests
        let mut score = 0.0;
        
        // Cache size optimization
        if let Some(cache_size) = parameters.get("cache_size") {
            let optimal_cache = context.get("memory_pressure").unwrap_or(&50.0);
            let cache_score = 1.0 - (cache_size - optimal_cache).abs() / 100.0;
            score += cache_score * 0.3;
        }
        
        // Thread count optimization
        if let Some(thread_count) = parameters.get("thread_count") {
            let cpu_cores = context.get("cpu_cores").unwrap_or(&4.0);
            let thread_score = 1.0 - (thread_count - cpu_cores).abs() / 10.0;
            score += thread_score * 0.4;
        }
        
        // Batch size optimization
        if let Some(batch_size) = parameters.get("batch_size") {
            let workload_size = context.get("avg_workload_size").unwrap_or(&100.0);
            let batch_score = 1.0 - (batch_size - workload_size).abs() / 200.0;
            score += batch_score * 0.3;
        }
        
        // Add some noise to simulate real-world variability
        let noise = (fastrand::f64() - 0.5) * 0.1;
        score += noise;
        
        Ok(score.max(0.0).min(1.0))
    }

    /// Apply parameter bounds
    fn apply_bounds(&self, param_name: &str, value: f64) -> f64 {
        let min_bound = self.config.min_bounds.get(param_name).unwrap_or(&0.0);
        let max_bound = self.config.max_bounds.get(param_name).unwrap_or(&1000.0);
        
        value.max(*min_bound).min(*max_bound)
    }

    /// Get optimization history
    pub fn get_history(&self) -> &[ParameterSnapshot] {
        &self.parameter_history
    }

    /// Get best parameters
    pub fn get_best_parameters(&self) -> &HashMap<String, f64> {
        &self.best_parameters
    }

    /// Get best performance
    pub fn get_best_performance(&self) -> f64 {
        self.best_performance
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.parameter_history.clear();
        self.performance_history.clear();
        self.best_parameters.clear();
        self.best_performance = 0.0;
    }

    /// Predict performance for given parameters
    pub async fn predict_performance(
        &self,
        parameters: &HashMap<String, f64>,
        context: &HashMap<String, f64>,
    ) -> Result<f64> {
        // Use the evaluation function for prediction
        self.evaluate_parameters(parameters, context).await
    }

    /// Get parameter recommendations based on context
    pub async fn get_parameter_recommendations(
        &self,
        context: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut recommendations = HashMap::new();
        
        // Recommend cache size based on memory pressure
        if let Some(memory_pressure) = context.get("memory_pressure") {
            let recommended_cache = if *memory_pressure > 80.0 {
                50.0 // Reduce cache size under high memory pressure
            } else if *memory_pressure < 30.0 {
                200.0 // Increase cache size when memory is available
            } else {
                100.0 // Default cache size
            };
            recommendations.insert("cache_size".to_string(), recommended_cache);
        }
        
        // Recommend thread count based on CPU utilization
        if let Some(cpu_utilization) = context.get("cpu_utilization") {
            let cpu_cores = context.get("cpu_cores").unwrap_or(&4.0);
            let recommended_threads = if *cpu_utilization > 80.0 {
                cpu_cores * 0.8 // Reduce threads under high CPU load
            } else if *cpu_utilization < 30.0 {
                cpu_cores * 1.2 // Increase threads when CPU is underutilized
            } else {
                *cpu_cores // Match CPU cores
            };
            recommendations.insert("thread_count".to_string(), recommended_threads);
        }
        
        // Recommend batch size based on workload characteristics
        if let Some(avg_workload) = context.get("avg_workload_size") {
            let recommended_batch = if *avg_workload > 500.0 {
                200.0 // Larger batches for large workloads
            } else if *avg_workload < 50.0 {
                25.0 // Smaller batches for small workloads
            } else {
                100.0 // Default batch size
            };
            recommendations.insert("batch_size".to_string(), recommended_batch);
        }
        
        Ok(recommendations)
    }
}
