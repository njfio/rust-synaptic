//! Elastic Weight Consolidation (EWC) Implementation
//! 
//! Implements the EWC algorithm for preventing catastrophic forgetting by
//! protecting important parameters using Fisher Information Matrix.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Fisher Information Matrix entry
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Parameter identifier
    pub parameter_id: String,
    /// Fisher information value
    pub fisher_value: f64,
    /// Parameter importance weight
    pub importance_weight: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// EWC parameter protection entry
#[derive(Debug, Clone)]
pub struct ParameterProtection {
    /// Memory key this parameter relates to
    pub memory_key: String,
    /// Parameter value at consolidation
    pub consolidated_value: f64,
    /// Fisher information for this parameter
    pub fisher_info: f64,
    /// Protection strength (0.0 to 1.0)
    pub protection_strength: f64,
    /// Consolidation timestamp
    pub consolidated_at: DateTime<Utc>,
}

/// EWC regularization term
#[derive(Debug, Clone)]
pub struct RegularizationTerm {
    /// Parameter identifier
    pub parameter_id: String,
    /// Regularization penalty
    pub penalty: f64,
    /// Lambda (regularization strength)
    pub lambda: f64,
    /// Parameter deviation from consolidated value
    pub deviation: f64,
}

/// EWC performance metrics
#[derive(Debug, Clone)]
pub struct EWCMetrics {
    /// Total parameters protected
    pub protected_parameters: usize,
    /// Average Fisher information
    pub avg_fisher_info: f64,
    /// Total regularization penalty
    pub total_penalty: f64,
    /// Catastrophic forgetting prevention rate
    pub prevention_rate: f64,
    /// Last metrics update
    pub last_updated: DateTime<Utc>,
}

/// Elastic Weight Consolidation implementation
#[derive(Debug)]
pub struct ElasticWeightConsolidation {
    /// Configuration
    config: ConsolidationConfig,
    /// Fisher Information Matrix
    fisher_matrix: HashMap<String, FisherInformation>,
    /// Protected parameters
    protected_parameters: HashMap<String, ParameterProtection>,
    /// Current regularization terms
    regularization_terms: Vec<RegularizationTerm>,
    /// EWC performance metrics
    metrics: EWCMetrics,
    /// Task-specific Fisher information
    task_fisher_info: HashMap<String, HashMap<String, f64>>,
    /// Consolidated parameter values per task
    task_parameters: HashMap<String, HashMap<String, f64>>,
}

impl ElasticWeightConsolidation {
    /// Create new EWC instance
    pub fn new(config: &ConsolidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            fisher_matrix: HashMap::new(),
            protected_parameters: HashMap::new(),
            regularization_terms: Vec::new(),
            metrics: EWCMetrics {
                protected_parameters: 0,
                avg_fisher_info: 0.0,
                total_penalty: 0.0,
                prevention_rate: 0.0,
                last_updated: Utc::now(),
            },
            task_fisher_info: HashMap::new(),
            task_parameters: HashMap::new(),
        })
    }

    /// Update Fisher Information Matrix based on memory importance scores
    pub async fn update_fisher_information(&mut self, importance_scores: &[MemoryImportance]) -> Result<()> {
        tracing::info!("Updating Fisher Information Matrix for {} memories", importance_scores.len());

        for importance in importance_scores {
            if let Some(fisher_info) = &importance.fisher_information {
                self.process_fisher_information(&importance.memory_key, fisher_info).await?;
            }
        }

        // Update metrics
        self.update_metrics().await?;

        tracing::debug!("Fisher Information Matrix updated with {} entries", self.fisher_matrix.len());
        Ok(())
    }

    /// Process Fisher information for a specific memory
    async fn process_fisher_information(&mut self, memory_key: &str, fisher_info: &[f64]) -> Result<()> {
        for (i, &fisher_value) in fisher_info.iter().enumerate() {
            let parameter_id = format!("{}_{}", memory_key, i);
            
            // Calculate importance weight based on Fisher information
            let importance_weight = self.calculate_importance_weight(fisher_value).await?;
            
            // Update or create Fisher information entry
            let fisher_entry = FisherInformation {
                parameter_id: parameter_id.clone(),
                fisher_value,
                importance_weight,
                last_updated: Utc::now(),
            };

            self.fisher_matrix.insert(parameter_id.clone(), fisher_entry);

            // Create parameter protection if Fisher information is significant
            if fisher_value > 0.1 {
                self.create_parameter_protection(memory_key, &parameter_id, fisher_value).await?;
            }
        }

        Ok(())
    }

    /// Calculate importance weight from Fisher information
    async fn calculate_importance_weight(&self, fisher_value: f64) -> Result<f64> {
        // Normalize Fisher information using sigmoid function
        let normalized_fisher = 1.0 / (1.0 + (-fisher_value * 2.0).exp());
        
        // Apply EWC lambda parameter
        let importance_weight = normalized_fisher * self.config.ewc_lambda;
        
        Ok(importance_weight.min(1.0).max(0.0))
    }

    /// Create parameter protection entry
    async fn create_parameter_protection(
        &mut self,
        memory_key: &str,
        parameter_id: &str,
        fisher_value: f64,
    ) -> Result<()> {
        // Simulate parameter value (in real implementation, this would be actual neural network weights)
        let consolidated_value = self.get_current_parameter_value(parameter_id).await?;
        
        let protection_strength = self.calculate_protection_strength(fisher_value).await?;

        let protection = ParameterProtection {
            memory_key: memory_key.to_string(),
            consolidated_value,
            fisher_info: fisher_value,
            protection_strength,
            consolidated_at: Utc::now(),
        };

        self.protected_parameters.insert(parameter_id.to_string(), protection);
        Ok(())
    }

    /// Calculate protection strength based on Fisher information
    async fn calculate_protection_strength(&self, fisher_value: f64) -> Result<f64> {
        // Protection strength increases with Fisher information
        let base_strength = 0.5;
        let fisher_contribution = fisher_value.min(1.0) * 0.5;
        let protection_strength = base_strength + fisher_contribution;
        
        Ok(protection_strength.min(1.0).max(0.0))
    }

    /// Get current parameter value (simulated)
    async fn get_current_parameter_value(&self, _parameter_id: &str) -> Result<f64> {
        // In a real implementation, this would retrieve actual parameter values
        // from the neural network or memory system
        use rand::Rng;
        Ok(rand::thread_rng().gen::<f64>() * 2.0 - 1.0) // Random value between -1 and 1
    }

    /// Calculate EWC regularization penalty for parameter updates
    pub async fn calculate_regularization_penalty(
        &mut self,
        parameter_updates: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut total_penalty = 0.0;
        self.regularization_terms.clear();

        for (parameter_id, new_value) in parameter_updates {
            if let Some(protection) = self.protected_parameters.get(parameter_id) {
                // Calculate penalty: λ/2 * F_i * (θ_i - θ*_i)^2
                let deviation = new_value - protection.consolidated_value;
                let penalty = 0.5 * self.config.ewc_lambda * protection.fisher_info * deviation.powi(2);
                
                total_penalty += penalty;

                // Record regularization term
                let reg_term = RegularizationTerm {
                    parameter_id: parameter_id.clone(),
                    penalty,
                    lambda: self.config.ewc_lambda,
                    deviation,
                };
                self.regularization_terms.push(reg_term);
            }
        }

        self.metrics.total_penalty = total_penalty;
        Ok(total_penalty)
    }

    /// Apply EWC constraints to parameter updates
    pub async fn apply_ewc_constraints(
        &self,
        parameter_updates: &mut HashMap<String, f64>,
    ) -> Result<()> {
        for (parameter_id, update_value) in parameter_updates.iter_mut() {
            if let Some(protection) = self.protected_parameters.get(parameter_id) {
                // Constrain update based on protection strength
                let max_deviation = 0.1 * (1.0 - protection.protection_strength);
                let current_deviation = *update_value - protection.consolidated_value;
                
                if current_deviation.abs() > max_deviation {
                    // Clamp the update to stay within allowed deviation
                    let sign = if current_deviation > 0.0 { 1.0 } else { -1.0 };
                    *update_value = protection.consolidated_value + sign * max_deviation;
                }
            }
        }

        Ok(())
    }

    /// Consolidate parameters for a new task
    pub async fn consolidate_task(&mut self, task_id: &str, memories: &[MemoryEntry]) -> Result<()> {
        tracing::info!("Consolidating EWC parameters for task: {}", task_id);

        let mut task_fisher = HashMap::new();
        let mut task_params = HashMap::new();

        // Calculate Fisher information for this task
        for memory in memories {
            let parameter_id = format!("task_{}_{}", task_id, memory.key);
            
            // Simulate Fisher information calculation
            let fisher_value = self.simulate_fisher_calculation(memory).await?;
            task_fisher.insert(parameter_id.clone(), fisher_value);
            
            // Store consolidated parameter value
            let param_value = self.get_current_parameter_value(&parameter_id).await?;
            task_params.insert(parameter_id, param_value);
        }

        // Store task-specific information
        self.task_fisher_info.insert(task_id.to_string(), task_fisher);
        self.task_parameters.insert(task_id.to_string(), task_params);

        tracing::debug!("Task {} consolidated with {} parameters", task_id, memories.len());
        Ok(())
    }

    /// Simulate Fisher information calculation for a memory
    async fn simulate_fisher_calculation(&self, memory: &MemoryEntry) -> Result<f64> {
        // Simplified Fisher information calculation based on memory characteristics
        let content_complexity = memory.value.len() as f64 / 1000.0; // Normalize by content length
        let access_importance = memory.access_count() as f64 / 100.0; // Normalize by access count
        let metadata_importance = memory.metadata.importance;
        
        let fisher_value = (content_complexity * 0.3 + access_importance * 0.4 + metadata_importance * 0.3)
            .min(1.0)
            .max(0.01);
        
        Ok(fisher_value)
    }

    /// Get EWC regularization terms for current state
    pub fn get_regularization_terms(&self) -> &[RegularizationTerm] {
        &self.regularization_terms
    }

    /// Get protected parameters
    pub fn get_protected_parameters(&self) -> &HashMap<String, ParameterProtection> {
        &self.protected_parameters
    }

    /// Get Fisher Information Matrix
    pub fn get_fisher_matrix(&self) -> &HashMap<String, FisherInformation> {
        &self.fisher_matrix
    }

    /// Get EWC performance metrics
    pub fn get_metrics(&self) -> &EWCMetrics {
        &self.metrics
    }

    /// Update EWC performance metrics
    async fn update_metrics(&mut self) -> Result<()> {
        self.metrics.protected_parameters = self.protected_parameters.len();
        
        if !self.fisher_matrix.is_empty() {
            self.metrics.avg_fisher_info = self.fisher_matrix.values()
                .map(|f| f.fisher_value)
                .sum::<f64>() / self.fisher_matrix.len() as f64;
        }

        // Calculate prevention rate based on protected parameters vs total parameters
        let total_params = self.fisher_matrix.len();
        self.metrics.prevention_rate = if total_params > 0 {
            self.metrics.protected_parameters as f64 / total_params as f64
        } else {
            0.0
        };

        self.metrics.last_updated = Utc::now();
        Ok(())
    }

    /// Reset EWC state for new learning phase
    pub async fn reset_for_new_task(&mut self) -> Result<()> {
        tracing::info!("Resetting EWC for new task");
        
        // Keep Fisher information but reset regularization terms
        self.regularization_terms.clear();
        
        // Reset metrics
        self.metrics.total_penalty = 0.0;
        self.metrics.last_updated = Utc::now();
        
        Ok(())
    }

    /// Get task-specific Fisher information
    pub fn get_task_fisher_info(&self, task_id: &str) -> Option<&HashMap<String, f64>> {
        self.task_fisher_info.get(task_id)
    }

    /// Get task-specific parameters
    pub fn get_task_parameters(&self, task_id: &str) -> Option<&HashMap<String, f64>> {
        self.task_parameters.get(task_id)
    }

    /// Calculate memory retention score using EWC
    pub async fn calculate_retention_score(&self, memory_key: &str) -> Result<f64> {
        let mut retention_factors = Vec::new();

        // Check if memory has protected parameters
        let protected_count = self.protected_parameters.values()
            .filter(|p| p.memory_key == memory_key)
            .count();

        if protected_count > 0 {
            // Calculate average protection strength for this memory
            let avg_protection = self.protected_parameters.values()
                .filter(|p| p.memory_key == memory_key)
                .map(|p| p.protection_strength)
                .sum::<f64>() / protected_count as f64;
            
            retention_factors.push(avg_protection);
        }

        // Check Fisher information strength
        let fisher_strength = self.fisher_matrix.values()
            .filter(|f| f.parameter_id.starts_with(memory_key))
            .map(|f| f.fisher_value)
            .sum::<f64>();

        if fisher_strength > 0.0 {
            retention_factors.push((fisher_strength / 10.0).min(1.0));
        }

        // Calculate overall retention score
        let retention_score = if retention_factors.is_empty() {
            0.5 // Default retention for unprotected memories
        } else {
            retention_factors.iter().sum::<f64>() / retention_factors.len() as f64
        };

        Ok(retention_score.min(1.0).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_ewc_creation() {
        let config = ConsolidationConfig::default();
        let ewc = ElasticWeightConsolidation::new(&config);
        assert!(ewc.is_ok());
    }

    #[tokio::test]
    async fn test_fisher_information_update() {
        let config = ConsolidationConfig::default();
        let mut ewc = ElasticWeightConsolidation::new(&config).unwrap();

        let importance_scores = vec![
            MemoryImportance {
                memory_key: "test_key".to_string(),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: Some(vec![0.5, 0.7, 0.3]),
            },
        ];

        let result = ewc.update_fisher_information(&importance_scores).await;
        assert!(result.is_ok());
        assert!(!ewc.get_fisher_matrix().is_empty());
    }

    #[tokio::test]
    async fn test_regularization_penalty() {
        let config = ConsolidationConfig::default();
        let mut ewc = ElasticWeightConsolidation::new(&config).unwrap();

        // Add some protected parameters
        ewc.protected_parameters.insert("param1".to_string(), ParameterProtection {
            memory_key: "test_key".to_string(),
            consolidated_value: 0.5,
            fisher_info: 0.8,
            protection_strength: 0.7,
            consolidated_at: Utc::now(),
        });

        let mut parameter_updates = HashMap::new();
        parameter_updates.insert("param1".to_string(), 0.7); // Deviation of 0.2

        let penalty = ewc.calculate_regularization_penalty(&parameter_updates).await.unwrap();
        assert!(penalty > 0.0);
    }

    #[tokio::test]
    async fn test_task_consolidation() {
        let config = ConsolidationConfig::default();
        let mut ewc = ElasticWeightConsolidation::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "Content 2".to_string(), MemoryType::LongTerm),
        ];

        let result = ewc.consolidate_task("task1", &memories).await;
        assert!(result.is_ok());
        assert!(ewc.get_task_fisher_info("task1").is_some());
        assert!(ewc.get_task_parameters("task1").is_some());
    }
}
