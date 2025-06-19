//! Adaptive Algorithm Selection System
//!
//! This module implements an intelligent algorithm selection system that adapts to workload patterns
//! and performance characteristics. It includes algorithm performance tracking, automatic switching,
//! and optimization feedback loops.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for the adaptive algorithm selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectorConfig {
    /// Default selection strategy
    pub default_strategy: SelectionStrategy,
    /// Maximum number of algorithms to track
    pub max_algorithms: usize,
    /// History window size
    pub history_window: usize,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Performance evaluation interval
    pub evaluation_interval_ms: u64,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            default_strategy: SelectionStrategy::PerformanceBased {
                weight_execution_time: 0.4,
                weight_accuracy: 0.4,
                weight_resource_usage: 0.2,
            },
            max_algorithms: 50,
            history_window: 1000,
            adaptation_threshold: 0.05,
            evaluation_interval_ms: 60000, // 1 minute
        }
    }
}

/// Algorithm performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    /// Algorithm identifier
    pub algorithm_id: String,
    /// Total executions
    pub execution_count: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average accuracy/quality score
    pub avg_quality_score: f64,
    /// Memory usage in MB
    pub avg_memory_usage_mb: f64,
    /// CPU utilization percentage
    pub avg_cpu_usage_percent: f64,
    /// Last execution timestamp
    pub last_execution: DateTime<Utc>,
    /// Performance trend (positive = improving, negative = degrading)
    pub performance_trend: f64,
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            algorithm_id: String::new(),
            execution_count: 0,
            avg_execution_time_ms: 0.0,
            success_rate: 1.0,
            avg_quality_score: 0.5,
            avg_memory_usage_mb: 0.0,
            avg_cpu_usage_percent: 0.0,
            last_execution: Utc::now(),
            performance_trend: 0.0,
        }
    }
}

/// Workload characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Data size category
    pub data_size: DataSizeCategory,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Latency requirements
    pub latency_requirement: LatencyRequirement,
    /// Accuracy requirements
    pub accuracy_requirement: AccuracyRequirement,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Temporal characteristics
    pub temporal_pattern: TemporalPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSizeCategory {
    Small,      // < 1MB
    Medium,     // 1MB - 100MB
    Large,      // 100MB - 1GB
    VeryLarge,  // > 1GB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LatencyRequirement {
    RealTime,      // < 10ms
    Interactive,   // < 100ms
    Responsive,    // < 1s
    Batch,         // > 1s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyRequirement {
    Approximate,   // 70-80%
    Good,         // 80-90%
    High,         // 90-95%
    Critical,     // > 95%
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub max_memory_mb: Option<f64>,
    pub max_cpu_percent: Option<f64>,
    pub max_execution_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPattern {
    Burst,         // High frequency, short duration
    Steady,        // Consistent load
    Periodic,      // Regular intervals
    Irregular,     // Random patterns
}

/// Algorithm selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Performance-based selection
    PerformanceBased {
        weight_execution_time: f64,
        weight_accuracy: f64,
        weight_resource_usage: f64,
    },
    /// Multi-armed bandit approach
    MultiArmedBandit {
        exploration_rate: f64,
        decay_factor: f64,
    },
    /// Contextual bandit with workload awareness
    ContextualBandit {
        context_features: Vec<String>,
        learning_rate: f64,
    },
    /// Reinforcement learning based
    ReinforcementLearning {
        q_learning_rate: f64,
        discount_factor: f64,
        exploration_strategy: ExplorationStrategy,
    },
    /// Ensemble approach
    Ensemble {
        base_strategies: Vec<SelectionStrategy>,
        voting_method: VotingMethod,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    EpsilonGreedy { epsilon: f64 },
    UCB { confidence_level: f64 },
    ThompsonSampling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingMethod {
    Majority,
    Weighted { weights: Vec<f64> },
    Ranked,
}

/// Algorithm execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub algorithm_id: String,
    pub execution_time_ms: u64,
    pub success: bool,
    pub quality_score: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub timestamp: DateTime<Utc>,
    pub workload_pattern: WorkloadPattern,
    pub error_message: Option<String>,
}

/// Adaptive algorithm selector
pub struct AdaptiveAlgorithmSelector {
    /// Available algorithms with their metrics
    algorithm_metrics: Arc<RwLock<HashMap<String, AlgorithmMetrics>>>,
    /// Workload pattern history
    workload_history: Arc<RwLock<Vec<WorkloadPattern>>>,
    /// Execution history
    execution_history: Arc<RwLock<Vec<ExecutionResult>>>,
    /// Current selection strategy
    selection_strategy: Arc<RwLock<SelectionStrategy>>,
    /// Algorithm registry
    algorithm_registry: Arc<RwLock<HashMap<String, AlgorithmInfo>>>,
    /// Performance feedback loop
    feedback_loop: Arc<RwLock<FeedbackLoop>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub supported_workloads: Vec<WorkloadPattern>,
    pub resource_requirements: ResourceConstraints,
    pub performance_characteristics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoop {
    pub adaptation_rate: f64,
    pub performance_window: usize,
    pub adaptation_threshold: f64,
    pub last_adaptation: DateTime<Utc>,
    pub adaptation_count: u64,
}

impl Default for FeedbackLoop {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            performance_window: 100,
            adaptation_threshold: 0.05,
            last_adaptation: Utc::now(),
            adaptation_count: 0,
        }
    }
}

impl AdaptiveAlgorithmSelector {
    /// Create a new adaptive algorithm selector
    pub fn new() -> Self {
        Self {
            algorithm_metrics: Arc::new(RwLock::new(HashMap::new())),
            workload_history: Arc::new(RwLock::new(Vec::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            selection_strategy: Arc::new(RwLock::new(SelectionStrategy::PerformanceBased {
                weight_execution_time: 0.4,
                weight_accuracy: 0.4,
                weight_resource_usage: 0.2,
            })),
            algorithm_registry: Arc::new(RwLock::new(HashMap::new())),
            feedback_loop: Arc::new(RwLock::new(FeedbackLoop::default())),
        }
    }

    /// Register a new algorithm
    pub async fn register_algorithm(&self, algorithm_info: AlgorithmInfo) -> Result<()> {
        let mut registry = self.algorithm_registry.write().await;
        let mut metrics = self.algorithm_metrics.write().await;
        
        registry.insert(algorithm_info.id.clone(), algorithm_info.clone());
        metrics.insert(algorithm_info.id.clone(), AlgorithmMetrics {
            algorithm_id: algorithm_info.id,
            ..Default::default()
        });
        
        tracing::info!("Registered algorithm: {}", algorithm_info.name);
        Ok(())
    }

    /// Select the best algorithm for a given workload pattern
    pub async fn select_algorithm(&self, workload: &WorkloadPattern) -> Result<String> {
        let strategy = self.selection_strategy.read().await.clone();
        
        match strategy {
            SelectionStrategy::PerformanceBased { weight_execution_time, weight_accuracy, weight_resource_usage } => {
                self.performance_based_selection(workload, weight_execution_time, weight_accuracy, weight_resource_usage).await
            },
            SelectionStrategy::MultiArmedBandit { exploration_rate, decay_factor } => {
                self.multi_armed_bandit_selection(workload, exploration_rate, decay_factor).await
            },
            SelectionStrategy::ContextualBandit { context_features, learning_rate } => {
                self.contextual_bandit_selection(workload, &context_features, learning_rate).await
            },
            SelectionStrategy::ReinforcementLearning { q_learning_rate, discount_factor, exploration_strategy } => {
                self.reinforcement_learning_selection(workload, q_learning_rate, discount_factor, &exploration_strategy).await
            },
            SelectionStrategy::Ensemble { base_strategies, voting_method } => {
                self.ensemble_selection(workload, &base_strategies, &voting_method).await
            },
        }
    }

    /// Record execution result and update metrics
    pub async fn record_execution(&self, result: ExecutionResult) -> Result<()> {
        // Update algorithm metrics
        {
            let mut metrics = self.algorithm_metrics.write().await;
            if let Some(algorithm_metrics) = metrics.get_mut(&result.algorithm_id) {
                self.update_algorithm_metrics(algorithm_metrics, &result).await;
            }
        }

        // Store execution history
        {
            let mut history = self.execution_history.write().await;
            history.push(result.clone());

            // Keep only recent history (last 10000 executions)
            if history.len() > 10000 {
                history.drain(0..1000);
            }
        }

        // Store workload pattern
        {
            let mut workload_history = self.workload_history.write().await;
            workload_history.push(result.workload_pattern);

            // Keep only recent patterns
            if workload_history.len() > 5000 {
                workload_history.drain(0..500);
            }
        }

        // Trigger adaptation if needed
        self.check_and_adapt().await?;

        Ok(())
    }

    /// Update algorithm metrics with new execution result
    async fn update_algorithm_metrics(&self, metrics: &mut AlgorithmMetrics, result: &ExecutionResult) {
        let old_count = metrics.execution_count as f64;
        let _new_count = old_count + 1.0;

        // Update execution count
        metrics.execution_count += 1;

        // Update averages using exponential moving average
        let alpha = 0.1; // Learning rate

        metrics.avg_execution_time_ms = if old_count == 0.0 {
            result.execution_time_ms as f64
        } else {
            metrics.avg_execution_time_ms * (1.0 - alpha) + (result.execution_time_ms as f64) * alpha
        };

        metrics.avg_quality_score = if old_count == 0.0 {
            result.quality_score
        } else {
            metrics.avg_quality_score * (1.0 - alpha) + result.quality_score * alpha
        };

        metrics.avg_memory_usage_mb = if old_count == 0.0 {
            result.memory_usage_mb
        } else {
            metrics.avg_memory_usage_mb * (1.0 - alpha) + result.memory_usage_mb * alpha
        };

        metrics.avg_cpu_usage_percent = if old_count == 0.0 {
            result.cpu_usage_percent
        } else {
            metrics.avg_cpu_usage_percent * (1.0 - alpha) + result.cpu_usage_percent * alpha
        };

        // Update success rate
        let success_value = if result.success { 1.0 } else { 0.0 };
        metrics.success_rate = if old_count == 0.0 {
            success_value
        } else {
            metrics.success_rate * (1.0 - alpha) + success_value * alpha
        };

        // Calculate performance trend
        if metrics.execution_count > 10 {
            let recent_quality = result.quality_score;
            let historical_quality = metrics.avg_quality_score;
            metrics.performance_trend = recent_quality - historical_quality;
        }

        metrics.last_execution = result.timestamp;
    }

    /// Performance-based algorithm selection
    async fn performance_based_selection(
        &self,
        workload: &WorkloadPattern,
        weight_execution_time: f64,
        weight_accuracy: f64,
        weight_resource_usage: f64,
    ) -> Result<String> {
        let metrics = self.algorithm_metrics.read().await;
        let registry = self.algorithm_registry.read().await;

        let mut best_algorithm = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for (algorithm_id, algorithm_metrics) in metrics.iter() {
            if let Some(algorithm_info) = registry.get(algorithm_id) {
                // Check if algorithm supports this workload
                if !self.is_algorithm_suitable(algorithm_info, workload) {
                    continue;
                }

                // Calculate composite score
                let time_score = 1.0 / (1.0 + algorithm_metrics.avg_execution_time_ms / 1000.0);
                let accuracy_score = algorithm_metrics.avg_quality_score;
                let resource_score = 1.0 / (1.0 + algorithm_metrics.avg_memory_usage_mb / 100.0 +
                                           algorithm_metrics.avg_cpu_usage_percent / 100.0);

                let composite_score = weight_execution_time * time_score +
                                    weight_accuracy * accuracy_score +
                                    weight_resource_usage * resource_score;

                if composite_score > best_score {
                    best_score = composite_score;
                    best_algorithm = algorithm_id.clone();
                }
            }
        }

        if best_algorithm.is_empty() {
            // Fallback to first available algorithm
            if let Some((algorithm_id, _)) = metrics.iter().next() {
                best_algorithm = algorithm_id.clone();
            }
        }

        Ok(best_algorithm)
    }

    /// Multi-armed bandit algorithm selection
    async fn multi_armed_bandit_selection(
        &self,
        _workload: &WorkloadPattern,
        exploration_rate: f64,
        _decay_factor: f64,
    ) -> Result<String> {
        let metrics = self.algorithm_metrics.read().await;

        // Epsilon-greedy strategy
        if fastrand::f64() < exploration_rate {
            // Exploration: random selection
            let algorithms: Vec<String> = metrics.keys().cloned().collect();
            if algorithms.is_empty() {
                return Err(crate::error::MemoryError::InvalidConfiguration {
                    message: "No algorithms registered".to_string(),
                });
            }
            let random_index = fastrand::usize(0..algorithms.len());
            Ok(algorithms[random_index].clone())
        } else {
            // Exploitation: select best performing algorithm
            let mut best_algorithm = String::new();
            let mut best_reward = f64::NEG_INFINITY;

            for (algorithm_id, algorithm_metrics) in metrics.iter() {
                let reward = algorithm_metrics.avg_quality_score * algorithm_metrics.success_rate;
                if reward > best_reward {
                    best_reward = reward;
                    best_algorithm = algorithm_id.clone();
                }
            }

            Ok(best_algorithm)
        }
    }

    /// Contextual bandit algorithm selection
    async fn contextual_bandit_selection(
        &self,
        workload: &WorkloadPattern,
        _context_features: &[String],
        _learning_rate: f64,
    ) -> Result<String> {
        // Simplified contextual bandit - in practice would use more sophisticated ML
        let metrics = self.algorithm_metrics.read().await;
        let registry = self.algorithm_registry.read().await;

        let mut best_algorithm = String::new();
        let mut best_contextual_score = f64::NEG_INFINITY;

        for (algorithm_id, algorithm_metrics) in metrics.iter() {
            if let Some(algorithm_info) = registry.get(algorithm_id) {
                // Calculate contextual score based on workload characteristics
                let context_score = self.calculate_context_score(algorithm_info, workload);
                let performance_score = algorithm_metrics.avg_quality_score * algorithm_metrics.success_rate;
                let contextual_score = context_score * performance_score;

                if contextual_score > best_contextual_score {
                    best_contextual_score = contextual_score;
                    best_algorithm = algorithm_id.clone();
                }
            }
        }

        Ok(best_algorithm)
    }

    /// Reinforcement learning algorithm selection
    async fn reinforcement_learning_selection(
        &self,
        workload: &WorkloadPattern,
        _q_learning_rate: f64,
        _discount_factor: f64,
        exploration_strategy: &ExplorationStrategy,
    ) -> Result<String> {
        // Simplified Q-learning approach
        let metrics = self.algorithm_metrics.read().await;

        match exploration_strategy {
            ExplorationStrategy::EpsilonGreedy { epsilon } => {
                if fastrand::f64() < *epsilon {
                    // Exploration
                    let algorithms: Vec<String> = metrics.keys().cloned().collect();
                    if algorithms.is_empty() {
                        return Err(crate::error::MemoryError::InvalidConfiguration {
                            message: "No algorithms registered".to_string(),
                        });
                    }
                    let random_index = fastrand::usize(0..algorithms.len());
                    Ok(algorithms[random_index].clone())
                } else {
                    // Exploitation
                    self.select_best_algorithm_by_q_value(workload).await
                }
            },
            ExplorationStrategy::UCB { confidence_level } => {
                self.ucb_selection(workload, *confidence_level).await
            },
            ExplorationStrategy::ThompsonSampling => {
                self.thompson_sampling_selection(workload).await
            },
        }
    }

    /// Ensemble algorithm selection
    async fn ensemble_selection(
        &self,
        workload: &WorkloadPattern,
        base_strategies: &[SelectionStrategy],
        voting_method: &VotingMethod,
    ) -> Result<String> {
        let mut strategy_votes: HashMap<String, f64> = HashMap::new();

        // Get votes from each base strategy
        for strategy in base_strategies {
            let selected_algorithm = match strategy {
                SelectionStrategy::PerformanceBased { weight_execution_time, weight_accuracy, weight_resource_usage } => {
                    self.performance_based_selection(workload, *weight_execution_time, *weight_accuracy, *weight_resource_usage).await?
                },
                SelectionStrategy::MultiArmedBandit { exploration_rate, decay_factor } => {
                    self.multi_armed_bandit_selection(workload, *exploration_rate, *decay_factor).await?
                },
                SelectionStrategy::ContextualBandit { context_features, learning_rate } => {
                    self.contextual_bandit_selection(workload, context_features, *learning_rate).await?
                },
                SelectionStrategy::ReinforcementLearning { q_learning_rate, discount_factor, exploration_strategy } => {
                    self.reinforcement_learning_selection(workload, *q_learning_rate, *discount_factor, exploration_strategy).await?
                },
                SelectionStrategy::Ensemble { .. } => {
                    // Avoid infinite recursion by using a simple fallback for nested ensembles
                    self.performance_based_selection(workload, 0.4, 0.4, 0.2).await?
                },
            };

            // Record vote
            *strategy_votes.entry(selected_algorithm).or_insert(0.0) += 1.0;
        }

        // Apply voting method
        match voting_method {
            VotingMethod::Majority => {
                strategy_votes.into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(algorithm, _)| algorithm)
                    .ok_or_else(|| crate::error::MemoryError::InvalidConfiguration {
                        message: "No votes received".to_string(),
                    })
            },
            VotingMethod::Weighted { weights } => {
                // Apply weights to votes
                let mut weighted_votes: HashMap<String, f64> = HashMap::new();
                for (i, (algorithm, votes)) in strategy_votes.iter().enumerate() {
                    let weight = weights.get(i).unwrap_or(&1.0);
                    *weighted_votes.entry(algorithm.clone()).or_insert(0.0) += votes * weight;
                }

                weighted_votes.into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(algorithm, _)| algorithm)
                    .ok_or_else(|| crate::error::MemoryError::InvalidConfiguration {
                        message: "No weighted votes received".to_string(),
                    })
            },
            VotingMethod::Ranked => {
                // Simple ranked voting (could be more sophisticated)
                strategy_votes.into_iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(algorithm, _)| algorithm)
                    .ok_or_else(|| crate::error::MemoryError::InvalidConfiguration {
                        message: "No ranked votes received".to_string(),
                    })
            },
        }
    }

    /// Check if algorithm is suitable for workload
    fn is_algorithm_suitable(&self, algorithm_info: &AlgorithmInfo, workload: &WorkloadPattern) -> bool {
        // Check resource constraints
        if let Some(max_memory) = workload.resource_constraints.max_memory_mb {
            if let Some(required_memory) = algorithm_info.resource_requirements.max_memory_mb {
                if required_memory > max_memory {
                    return false;
                }
            }
        }

        if let Some(max_cpu) = workload.resource_constraints.max_cpu_percent {
            if let Some(required_cpu) = algorithm_info.resource_requirements.max_cpu_percent {
                if required_cpu > max_cpu {
                    return false;
                }
            }
        }

        // Check if algorithm supports this type of workload
        // This is a simplified check - in practice would be more sophisticated
        true
    }

    /// Calculate context score for algorithm given workload
    fn calculate_context_score(&self, algorithm_info: &AlgorithmInfo, workload: &WorkloadPattern) -> f64 {
        let mut score: f64 = 0.5; // Base score

        // Adjust score based on workload characteristics
        match workload.data_size {
            DataSizeCategory::Small => {
                if algorithm_info.performance_characteristics.get("small_data_efficiency").unwrap_or(&0.5) > &0.7 {
                    score += 0.2;
                }
            },
            DataSizeCategory::Large | DataSizeCategory::VeryLarge => {
                if algorithm_info.performance_characteristics.get("large_data_efficiency").unwrap_or(&0.5) > &0.7 {
                    score += 0.2;
                }
            },
            _ => {},
        }

        match workload.latency_requirement {
            LatencyRequirement::RealTime | LatencyRequirement::Interactive => {
                if algorithm_info.performance_characteristics.get("low_latency").unwrap_or(&0.5) > &0.7 {
                    score += 0.2;
                }
            },
            _ => {},
        }

        match workload.accuracy_requirement {
            AccuracyRequirement::High | AccuracyRequirement::Critical => {
                if algorithm_info.performance_characteristics.get("high_accuracy").unwrap_or(&0.5) > &0.7 {
                    score += 0.2;
                }
            },
            _ => {},
        }

        score.clamp(0.0, 1.0)
    }

    /// Select best algorithm by Q-value (simplified)
    async fn select_best_algorithm_by_q_value(&self, _workload: &WorkloadPattern) -> Result<String> {
        let metrics = self.algorithm_metrics.read().await;

        let mut best_algorithm = String::new();
        let mut best_q_value = f64::NEG_INFINITY;

        for (algorithm_id, algorithm_metrics) in metrics.iter() {
            // Simplified Q-value calculation
            let q_value = algorithm_metrics.avg_quality_score * algorithm_metrics.success_rate -
                         algorithm_metrics.avg_execution_time_ms / 10000.0;

            if q_value > best_q_value {
                best_q_value = q_value;
                best_algorithm = algorithm_id.clone();
            }
        }

        Ok(best_algorithm)
    }

    /// UCB (Upper Confidence Bound) selection
    async fn ucb_selection(&self, _workload: &WorkloadPattern, confidence_level: f64) -> Result<String> {
        let metrics = self.algorithm_metrics.read().await;
        let total_executions: u64 = metrics.values().map(|m| m.execution_count).sum();

        let mut best_algorithm = String::new();
        let mut best_ucb_value = f64::NEG_INFINITY;

        for (algorithm_id, algorithm_metrics) in metrics.iter() {
            if algorithm_metrics.execution_count == 0 {
                // Unplayed arm gets infinite UCB value
                return Ok(algorithm_id.clone());
            }

            let mean_reward = algorithm_metrics.avg_quality_score * algorithm_metrics.success_rate;
            let confidence_interval = confidence_level *
                ((total_executions as f64).ln() / algorithm_metrics.execution_count as f64).sqrt();
            let ucb_value = mean_reward + confidence_interval;

            if ucb_value > best_ucb_value {
                best_ucb_value = ucb_value;
                best_algorithm = algorithm_id.clone();
            }
        }

        Ok(best_algorithm)
    }

    /// Thompson Sampling selection
    async fn thompson_sampling_selection(&self, _workload: &WorkloadPattern) -> Result<String> {
        let metrics = self.algorithm_metrics.read().await;

        let mut best_algorithm = String::new();
        let mut best_sample = f64::NEG_INFINITY;

        for (algorithm_id, algorithm_metrics) in metrics.iter() {
            // Sample from Beta distribution (simplified)
            let alpha = algorithm_metrics.execution_count as f64 * algorithm_metrics.success_rate + 1.0;
            let beta = algorithm_metrics.execution_count as f64 * (1.0 - algorithm_metrics.success_rate) + 1.0;

            // Simplified beta sampling using uniform random
            let sample = self.sample_beta(alpha, beta);

            if sample > best_sample {
                best_sample = sample;
                best_algorithm = algorithm_id.clone();
            }
        }

        Ok(best_algorithm)
    }

    /// Simplified beta distribution sampling
    fn sample_beta(&self, alpha: f64, beta: f64) -> f64 {
        // Very simplified beta sampling - in practice would use proper statistical library
        let x = fastrand::f64();
        let y = fastrand::f64();

        let gamma_alpha = self.gamma_sample(alpha);
        let gamma_beta = self.gamma_sample(beta);

        gamma_alpha / (gamma_alpha + gamma_beta)
    }

    /// Simplified gamma distribution sampling
    fn gamma_sample(&self, shape: f64) -> f64 {
        // Very simplified - in practice would use proper gamma sampling
        if shape < 1.0 {
            fastrand::f64().powf(1.0 / shape)
        } else {
            // Approximation for shape >= 1
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();

            loop {
                let x = self.normal_sample();
                let v = (1.0 + c * x).powi(3);
                if v > 0.0 {
                    let u = fastrand::f64();
                    if u < 1.0 - 0.0331 * x.powi(4) {
                        return d * v;
                    }
                    if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
                        return d * v;
                    }
                }
            }
        }
    }

    /// Simplified normal distribution sampling (Box-Muller)
    fn normal_sample(&self) -> f64 {
        let u1 = fastrand::f64();
        let u2 = fastrand::f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Check if adaptation is needed and perform it
    async fn check_and_adapt(&self) -> Result<()> {
        let feedback_loop = self.feedback_loop.read().await.clone();
        let execution_history = self.execution_history.read().await;

        // Check if we have enough data for adaptation
        if execution_history.len() < feedback_loop.performance_window {
            return Ok(());
        }

        // Calculate recent performance
        let recent_results = &execution_history[execution_history.len() - feedback_loop.performance_window..];
        let recent_performance: f64 = recent_results.iter()
            .map(|r| if r.success { r.quality_score } else { 0.0 })
            .sum::<f64>() / recent_results.len() as f64;

        // Calculate historical performance
        let historical_results = if execution_history.len() > feedback_loop.performance_window * 2 {
            &execution_history[execution_history.len() - feedback_loop.performance_window * 2..
                              execution_history.len() - feedback_loop.performance_window]
        } else {
            return Ok(()); // Not enough historical data
        };

        let historical_performance: f64 = historical_results.iter()
            .map(|r| if r.success { r.quality_score } else { 0.0 })
            .sum::<f64>() / historical_results.len() as f64;

        // Check if adaptation is needed
        let performance_change = recent_performance - historical_performance;
        if performance_change.abs() > feedback_loop.adaptation_threshold {
            self.adapt_selection_strategy(performance_change).await?;
        }

        Ok(())
    }

    /// Adapt the selection strategy based on performance feedback
    async fn adapt_selection_strategy(&self, performance_change: f64) -> Result<()> {
        let mut feedback_loop = self.feedback_loop.write().await;
        let current_strategy = self.selection_strategy.read().await.clone();

        let new_strategy = match current_strategy {
            SelectionStrategy::PerformanceBased { weight_execution_time, weight_accuracy, weight_resource_usage } => {
                if performance_change < 0.0 {
                    // Performance degraded, increase exploration
                    SelectionStrategy::MultiArmedBandit {
                        exploration_rate: 0.2,
                        decay_factor: 0.95,
                    }
                } else {
                    // Performance improved, fine-tune weights
                    SelectionStrategy::PerformanceBased {
                        weight_execution_time: weight_execution_time * 0.9 + 0.1 * 0.3,
                        weight_accuracy: weight_accuracy * 0.9 + 0.1 * 0.5,
                        weight_resource_usage: weight_resource_usage * 0.9 + 0.1 * 0.2,
                    }
                }
            },
            SelectionStrategy::MultiArmedBandit { exploration_rate, decay_factor } => {
                if performance_change > 0.0 {
                    // Performance improved, reduce exploration
                    SelectionStrategy::MultiArmedBandit {
                        exploration_rate: (exploration_rate * 0.9).max(0.05),
                        decay_factor,
                    }
                } else {
                    // Performance degraded, try contextual approach
                    SelectionStrategy::ContextualBandit {
                        context_features: vec!["data_size".to_string(), "complexity".to_string()],
                        learning_rate: 0.1,
                    }
                }
            },
            _ => current_strategy, // Keep current strategy for other types
        };

        *self.selection_strategy.write().await = new_strategy;
        feedback_loop.adaptation_count += 1;
        feedback_loop.last_adaptation = Utc::now();

        tracing::info!("Adapted selection strategy due to performance change: {:.4}", performance_change);

        Ok(())
    }

    /// Get current algorithm metrics
    pub async fn get_algorithm_metrics(&self) -> HashMap<String, AlgorithmMetrics> {
        self.algorithm_metrics.read().await.clone()
    }

    /// Get execution history
    pub async fn get_execution_history(&self) -> Vec<ExecutionResult> {
        self.execution_history.read().await.clone()
    }

    /// Get current selection strategy
    pub async fn get_selection_strategy(&self) -> SelectionStrategy {
        self.selection_strategy.read().await.clone()
    }

    /// Set selection strategy
    pub async fn set_selection_strategy(&self, strategy: SelectionStrategy) -> Result<()> {
        *self.selection_strategy.write().await = strategy;
        Ok(())
    }
}
