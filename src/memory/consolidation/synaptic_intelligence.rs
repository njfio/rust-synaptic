//! Synaptic Intelligence (SI) Implementation
//! 
//! Implements the Synaptic Intelligence algorithm for continual learning and
//! catastrophic forgetting prevention through path integral-based parameter importance tracking.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::ConsolidationConfig;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Path integral accumulator for parameter importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathIntegral {
    /// Parameter identifier
    pub parameter_id: String,
    /// Accumulated path integral value
    pub integral_value: f64,
    /// Current parameter value
    pub current_value: f64,
    /// Previous parameter value
    pub previous_value: f64,
    /// Gradient accumulator
    pub gradient_accumulator: f64,
    /// Number of updates
    pub update_count: u64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Parameter importance based on path integral
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterImportance {
    /// Parameter identifier
    pub parameter_id: String,
    /// Normalized importance score (0.0 to 1.0)
    pub importance_score: f64,
    /// Raw path integral value
    pub path_integral: f64,
    /// Parameter change magnitude
    pub change_magnitude: f64,
    /// Contribution to loss reduction
    pub loss_contribution: f64,
    /// Task-specific importance
    pub task_importance: HashMap<String, f64>,
    /// Calculated at timestamp
    pub calculated_at: DateTime<Utc>,
}

/// Synaptic Intelligence regularization term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIRegularizationTerm {
    /// Parameter identifier
    pub parameter_id: String,
    /// Regularization penalty
    pub penalty: f64,
    /// Importance weight
    pub importance_weight: f64,
    /// Parameter deviation from consolidated value
    pub deviation: f64,
    /// Damping factor
    pub damping_factor: f64,
}

/// Synaptic Intelligence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIMetrics {
    /// Total number of tracked parameters
    pub tracked_parameters: usize,
    /// Average path integral value
    pub avg_path_integral: f64,
    /// Total regularization penalty
    pub total_penalty: f64,
    /// Forgetting prevention rate
    pub prevention_rate: f64,
    /// Task consolidation count
    pub task_count: usize,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Main Synaptic Intelligence implementation
#[derive(Debug)]
pub struct SynapticIntelligence {
    /// Configuration
    config: ConsolidationConfig,
    /// Path integral accumulators for each parameter
    path_integrals: HashMap<String, PathIntegral>,
    /// Parameter importance scores
    parameter_importance: HashMap<String, ParameterImportance>,
    /// Consolidated parameter values per task
    task_parameters: HashMap<String, HashMap<String, f64>>,
    /// Task-specific importance weights
    task_importance_weights: HashMap<String, HashMap<String, f64>>,
    /// Regularization terms
    regularization_terms: Vec<SIRegularizationTerm>,
    /// Performance metrics
    metrics: SIMetrics,
    /// Damping factor for importance calculation
    damping_factor: f64,
}

impl SynapticIntelligence {
    /// Create new Synaptic Intelligence instance
    pub fn new(config: &ConsolidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            path_integrals: HashMap::new(),
            parameter_importance: HashMap::new(),
            task_parameters: HashMap::new(),
            task_importance_weights: HashMap::new(),
            regularization_terms: Vec::new(),
            metrics: SIMetrics {
                tracked_parameters: 0,
                avg_path_integral: 0.0,
                total_penalty: 0.0,
                prevention_rate: 0.0,
                task_count: 0,
                last_updated: Utc::now(),
            },
            damping_factor: 0.1, // Default damping factor
        })
    }

    /// Update path integrals for parameter changes
    pub async fn update_path_integrals(
        &mut self,
        parameter_updates: &HashMap<String, f64>,
        gradients: &HashMap<String, f64>,
    ) -> Result<()> {
        tracing::debug!("Updating path integrals for {} parameters", parameter_updates.len());

        for (param_id, &new_value) in parameter_updates {
            let gradient = gradients.get(param_id).copied().unwrap_or(0.0);
            
            let path_integral = self.path_integrals.entry(param_id.clone())
                .or_insert_with(|| PathIntegral {
                    parameter_id: param_id.clone(),
                    integral_value: 0.0,
                    current_value: new_value,
                    previous_value: new_value,
                    gradient_accumulator: 0.0,
                    update_count: 0,
                    last_updated: Utc::now(),
                });

            // Calculate parameter change
            let param_change = new_value - path_integral.current_value;
            
            // Update path integral using: Ω += -∇L * Δθ
            let integral_update = -gradient * param_change;
            path_integral.integral_value += integral_update;
            
            // Update gradient accumulator
            path_integral.gradient_accumulator += gradient.abs();
            
            // Update parameter values
            path_integral.previous_value = path_integral.current_value;
            path_integral.current_value = new_value;
            path_integral.update_count += 1;
            path_integral.last_updated = Utc::now();
        }

        // Update metrics
        self.update_metrics().await?;
        
        Ok(())
    }

    /// Calculate parameter importance from path integrals
    pub async fn calculate_parameter_importance(&mut self) -> Result<()> {
        tracing::debug!("Calculating parameter importance for {} parameters", self.path_integrals.len());

        for (param_id, path_integral) in &self.path_integrals {
            // Calculate importance using: ω = Ω / (ξ + ||Δθ||²)
            let change_magnitude = (path_integral.current_value - path_integral.previous_value).abs();
            let denominator = self.damping_factor + change_magnitude.powi(2);
            
            let raw_importance = if denominator > 0.0 {
                path_integral.integral_value.abs() / denominator
            } else {
                0.0
            };

            // Normalize importance score
            let importance_score = (raw_importance / (1.0 + raw_importance)).min(1.0).max(0.0);

            // Calculate loss contribution (simplified)
            let loss_contribution = path_integral.gradient_accumulator / (path_integral.update_count as f64 + 1.0);

            let importance = ParameterImportance {
                parameter_id: param_id.clone(),
                importance_score,
                path_integral: path_integral.integral_value,
                change_magnitude,
                loss_contribution,
                task_importance: HashMap::new(),
                calculated_at: Utc::now(),
            };

            self.parameter_importance.insert(param_id.clone(), importance);
        }

        Ok(())
    }

    /// Consolidate parameters for a completed task
    pub async fn consolidate_task(&mut self, task_id: &str, memories: &[MemoryEntry]) -> Result<()> {
        tracing::info!("Consolidating SI parameters for task: {}", task_id);

        // Calculate current parameter importance
        self.calculate_parameter_importance().await?;

        let mut task_params = HashMap::new();
        let mut task_weights = HashMap::new();

        // Store consolidated parameters and importance weights
        for memory in memories {
            let param_id = format!("task_{}_{}", task_id, memory.key);
            
            // Get current parameter value (simulated)
            let param_value = self.get_current_parameter_value(&param_id).await?;
            task_params.insert(param_id.clone(), param_value);

            // Get importance weight
            if let Some(importance) = self.parameter_importance.get(&param_id) {
                task_weights.insert(param_id, importance.importance_score);
            }
        }

        self.task_parameters.insert(task_id.to_string(), task_params);
        self.task_importance_weights.insert(task_id.to_string(), task_weights);
        
        // Reset path integrals for next task
        self.reset_path_integrals().await?;
        
        self.metrics.task_count += 1;
        tracing::info!("Task {} consolidated with {} parameters", task_id, memories.len());
        
        Ok(())
    }

    /// Calculate SI regularization penalty
    pub async fn calculate_regularization_penalty(
        &self,
        parameter_updates: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut total_penalty = 0.0;

        for (param_id, &new_value) in parameter_updates {
            // Find consolidated value from all previous tasks
            let mut consolidated_value = 0.0;
            let mut total_importance = 0.0;

            for (task_id, task_params) in &self.task_parameters {
                if let Some(&task_param_value) = task_params.get(param_id) {
                    if let Some(task_weights) = self.task_importance_weights.get(task_id) {
                        if let Some(&importance_weight) = task_weights.get(param_id) {
                            consolidated_value += task_param_value * importance_weight;
                            total_importance += importance_weight;
                        }
                    }
                }
            }

            if total_importance > 0.0 {
                consolidated_value /= total_importance;
                
                // Calculate SI penalty: c/2 * ω * (θ - θ*)²
                let deviation = new_value - consolidated_value;
                let penalty = 0.5 * self.config.ewc_lambda * total_importance * deviation.powi(2);
                
                total_penalty += penalty;
            }
        }

        Ok(total_penalty)
    }

    /// Get task-specific parameter values
    pub fn get_task_parameters(&self, task_id: &str) -> Option<&HashMap<String, f64>> {
        self.task_parameters.get(task_id)
    }

    /// Get task-specific importance weights
    pub fn get_task_importance_weights(&self, task_id: &str) -> Option<&HashMap<String, f64>> {
        self.task_importance_weights.get(task_id)
    }

    /// Get current SI metrics
    pub fn get_metrics(&self) -> &SIMetrics {
        &self.metrics
    }

    /// Reset path integrals for new task
    async fn reset_path_integrals(&mut self) -> Result<()> {
        for path_integral in self.path_integrals.values_mut() {
            path_integral.integral_value = 0.0;
            path_integral.gradient_accumulator = 0.0;
            path_integral.update_count = 0;
            path_integral.last_updated = Utc::now();
        }
        Ok(())
    }

    /// Update performance metrics
    async fn update_metrics(&mut self) -> Result<()> {
        self.metrics.tracked_parameters = self.path_integrals.len();
        
        if !self.path_integrals.is_empty() {
            self.metrics.avg_path_integral = self.path_integrals.values()
                .map(|pi| pi.integral_value.abs())
                .sum::<f64>() / self.path_integrals.len() as f64;
        }

        // Calculate prevention rate based on protected parameters
        let protected_params = self.parameter_importance.values()
            .filter(|imp| imp.importance_score > 0.5)
            .count();
        
        self.metrics.prevention_rate = if self.parameter_importance.len() > 0 {
            protected_params as f64 / self.parameter_importance.len() as f64
        } else {
            0.0
        };

        self.metrics.last_updated = Utc::now();
        Ok(())
    }

    /// Simulate getting current parameter value
    async fn get_current_parameter_value(&self, _param_id: &str) -> Result<f64> {
        // In a real implementation, this would retrieve the actual parameter value
        use rand::Rng;
        Ok(rand::thread_rng().gen::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_synaptic_intelligence_creation() {
        let config = ConsolidationConfig::default();
        let si = SynapticIntelligence::new(&config);
        assert!(si.is_ok());
    }

    #[tokio::test]
    async fn test_path_integral_updates() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.5);
        parameter_updates.insert("param2".to_string(), -0.3);

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 0.1);
        gradients.insert("param2".to_string(), -0.2);

        let result = si.update_path_integrals(&parameter_updates, &gradients).await;
        assert!(result.is_ok());
        assert_eq!(si.path_integrals.len(), 2);
    }

    #[tokio::test]
    async fn test_parameter_importance_calculation() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        // Add some path integrals first
        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.5);
        
        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 0.1);

        si.update_path_integrals(&parameter_updates, &gradients).await.unwrap();
        
        let result = si.calculate_parameter_importance().await;
        assert!(result.is_ok());
        assert!(si.parameter_importance.contains_key("param1"));
    }

    #[tokio::test]
    async fn test_task_consolidation() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "Content 2".to_string(), MemoryType::LongTerm),
        ];

        let result = si.consolidate_task("task1", &memories).await;
        assert!(result.is_ok());
        assert!(si.get_task_parameters("task1").is_some());
        assert!(si.get_task_importance_weights("task1").is_some());
        assert_eq!(si.metrics.task_count, 1);
    }

    #[tokio::test]
    async fn test_regularization_penalty_calculation() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        // First consolidate a task
        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
        ];
        si.consolidate_task("task1", &memories).await.unwrap();

        // Now calculate penalty for parameter updates
        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("task_task1_key1".to_string(), 0.8);

        let penalty = si.calculate_regularization_penalty(&parameter_updates).await.unwrap();
        assert!(penalty >= 0.0);
    }

    #[tokio::test]
    async fn test_path_integral_accumulation() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.5);

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 0.1);

        // First update
        si.update_path_integrals(&parameter_updates, &gradients).await.unwrap();

        // Second update with different values
        parameter_updates.insert("param1".to_string(), 0.7);
        gradients.insert("param1".to_string(), 0.2);

        si.update_path_integrals(&parameter_updates, &gradients).await.unwrap();

        let path_integral = si.path_integrals.get("param1").unwrap();
        assert!(path_integral.update_count == 2);
        assert!(path_integral.gradient_accumulator > 0.0);
    }

    #[tokio::test]
    async fn test_importance_score_normalization() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        // Add path integrals with different magnitudes
        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 1.0);
        parameter_updates.insert("param2".to_string(), 0.1);

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 1.0);
        gradients.insert("param2".to_string(), 0.01);

        si.update_path_integrals(&parameter_updates, &gradients).await.unwrap();
        si.calculate_parameter_importance().await.unwrap();

        for importance in si.parameter_importance.values() {
            assert!(importance.importance_score >= 0.0);
            assert!(importance.importance_score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_multiple_task_consolidation() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        // Consolidate multiple tasks
        for i in 1..=3 {
            let memories = vec![
                MemoryEntry::new(format!("key{}", i), format!("Content {}", i), MemoryType::LongTerm),
            ];
            si.consolidate_task(&format!("task{}", i), &memories).await.unwrap();
        }

        assert_eq!(si.metrics.task_count, 3);
        assert!(si.get_task_parameters("task1").is_some());
        assert!(si.get_task_parameters("task2").is_some());
        assert!(si.get_task_parameters("task3").is_some());
    }

    #[tokio::test]
    async fn test_metrics_updates() {
        let config = ConsolidationConfig::default();
        let mut si = SynapticIntelligence::new(&config).unwrap();

        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.5);
        parameter_updates.insert("param2".to_string(), 0.3);

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 0.1);
        gradients.insert("param2".to_string(), 0.2);

        si.update_path_integrals(&parameter_updates, &gradients).await.unwrap();

        let metrics = si.get_metrics();
        assert_eq!(metrics.tracked_parameters, 2);
        assert!(metrics.avg_path_integral >= 0.0);
    }

    #[tokio::test]
    async fn test_damping_factor_effect() {
        let config = ConsolidationConfig::default();
        let mut si1 = SynapticIntelligence::new(&config).unwrap();
        let mut si2 = SynapticIntelligence::new(&config).unwrap();

        // Set different damping factors
        si1.damping_factor = 0.01;
        si2.damping_factor = 1.0;

        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.5);

        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), 0.1);

        // Same updates for both
        si1.update_path_integrals(&parameter_updates, &gradients).await.unwrap();
        si2.update_path_integrals(&parameter_updates, &gradients).await.unwrap();

        si1.calculate_parameter_importance().await.unwrap();
        si2.calculate_parameter_importance().await.unwrap();

        let importance1 = si1.parameter_importance.get("param1").unwrap();
        let importance2 = si2.parameter_importance.get("param1").unwrap();

        // Lower damping factor should result in higher importance scores
        assert!(importance1.importance_score >= importance2.importance_score);
    }
}
