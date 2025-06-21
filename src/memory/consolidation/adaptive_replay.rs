//! Adaptive Replay Mechanisms Implementation
//! 
//! Implements sophisticated adaptive replay mechanisms that dynamically adjust
//! replay strategies, scheduling, and selection based on performance feedback,
//! memory characteristics, and learning objectives.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance};
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

/// Adaptive replay strategy that evolves based on performance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AdaptiveReplayStrategy {
    /// Performance-driven adaptation
    PerformanceDriven {
        success_threshold: f64,
        adaptation_rate: f64,
    },
    /// Context-aware adaptation
    ContextAware {
        context_weights: HashMap<String, f64>,
        adaptation_window: u64,
    },
    /// Multi-objective optimization
    MultiObjective {
        objectives: Vec<ReplayObjective>,
        weights: Vec<f64>,
    },
    /// Reinforcement learning based
    ReinforcementLearning {
        exploration_rate: f64,
        learning_rate: f64,
        discount_factor: f64,
    },
    /// Hybrid adaptive approach
    Hybrid {
        strategies: Vec<AdaptiveReplayStrategy>,
        selection_policy: SelectionPolicy,
    },
}

/// Replay objectives for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReplayObjective {
    /// Maximize retention rate
    MaximizeRetention,
    /// Minimize interference
    MinimizeInterference,
    /// Optimize learning efficiency
    OptimizeLearningEfficiency,
    /// Balance coverage and depth
    BalanceCoverageDepth,
    /// Minimize computational cost
    MinimizeComputationalCost,
}

/// Strategy selection policy for hybrid approaches
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SelectionPolicy {
    /// Round-robin selection
    RoundRobin,
    /// Performance-based selection
    PerformanceBased,
    /// Contextual bandit
    ContextualBandit,
    /// Weighted random selection
    WeightedRandom,
}

/// Adaptive replay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveReplayConfig {
    /// Base replay strategy
    pub base_strategy: AdaptiveReplayStrategy,
    /// Adaptation frequency in hours
    pub adaptation_frequency_hours: u64,
    /// Performance evaluation window
    pub evaluation_window_size: usize,
    /// Minimum adaptation threshold
    pub min_adaptation_threshold: f64,
    /// Maximum adaptation rate
    pub max_adaptation_rate: f64,
    /// Enable dynamic scheduling
    pub enable_dynamic_scheduling: bool,
    /// Enable context awareness
    pub enable_context_awareness: bool,
    /// Enable multi-objective optimization
    pub enable_multi_objective: bool,
}

impl Default for AdaptiveReplayConfig {
    fn default() -> Self {
        Self {
            base_strategy: AdaptiveReplayStrategy::PerformanceDriven {
                success_threshold: 0.7,
                adaptation_rate: 0.1,
            },
            adaptation_frequency_hours: 6,
            evaluation_window_size: 100,
            min_adaptation_threshold: 0.05,
            max_adaptation_rate: 0.5,
            enable_dynamic_scheduling: true,
            enable_context_awareness: true,
            enable_multi_objective: false,
        }
    }
}

/// Performance metrics for adaptive replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveReplayMetrics {
    /// Total adaptive replays performed
    pub total_adaptive_replays: usize,
    /// Current adaptation rate
    pub current_adaptation_rate: f64,
    /// Strategy effectiveness scores
    pub strategy_effectiveness: HashMap<String, f64>,
    /// Adaptation history
    pub adaptation_count: usize,
    /// Average performance improvement
    pub avg_performance_improvement: f64,
    /// Context adaptation accuracy
    pub context_adaptation_accuracy: f64,
    /// Multi-objective optimization score
    pub multi_objective_score: f64,
    /// Last adaptation timestamp
    pub last_adaptation: DateTime<Utc>,
}

/// Replay performance feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayPerformanceFeedback {
    /// Memory key
    pub memory_key: String,
    /// Replay success rate
    pub success_rate: f64,
    /// Retention improvement
    pub retention_improvement: f64,
    /// Learning efficiency
    pub learning_efficiency: f64,
    /// Interference level
    pub interference_level: f64,
    /// Context relevance
    pub context_relevance: f64,
    /// Computational cost
    pub computational_cost: f64,
    /// Feedback timestamp
    pub timestamp: DateTime<Utc>,
}

/// Context information for adaptive replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayContext {
    /// Current learning phase
    pub learning_phase: String,
    /// Memory domain
    pub memory_domain: String,
    /// User activity level
    pub activity_level: f64,
    /// System load
    pub system_load: f64,
    /// Time of day factor
    pub time_of_day_factor: f64,
    /// Recent performance trends
    pub performance_trends: Vec<f64>,
    /// Context timestamp
    pub timestamp: DateTime<Utc>,
}

/// Adaptive replay decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveReplayDecision {
    /// Memory key
    pub memory_key: String,
    /// Selected strategy
    pub selected_strategy: AdaptiveReplayStrategy,
    /// Replay priority
    pub replay_priority: f64,
    /// Scheduled time
    pub scheduled_time: DateTime<Utc>,
    /// Expected effectiveness
    pub expected_effectiveness: f64,
    /// Adaptation confidence
    pub adaptation_confidence: f64,
    /// Context factors
    pub context_factors: HashMap<String, f64>,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
}

/// Main Adaptive Replay Mechanisms implementation
#[derive(Debug)]
pub struct AdaptiveReplayMechanisms {
    /// Configuration
    config: AdaptiveReplayConfig,
    /// Consolidation configuration
    #[allow(dead_code)]
    consolidation_config: ConsolidationConfig,
    /// Current active strategy
    current_strategy: AdaptiveReplayStrategy,
    /// Performance feedback history
    performance_feedback: VecDeque<ReplayPerformanceFeedback>,
    /// Context history
    context_history: VecDeque<ReplayContext>,
    /// Adaptation decisions history
    decision_history: VecDeque<AdaptiveReplayDecision>,
    /// Strategy performance tracking
    strategy_performance: HashMap<String, Vec<f64>>,
    /// Performance metrics
    metrics: AdaptiveReplayMetrics,
    /// Last adaptation timestamp
    last_adaptation: Option<DateTime<Utc>>,
}

impl AdaptiveReplayMechanisms {
    /// Create new adaptive replay mechanisms
    pub fn new(
        config: AdaptiveReplayConfig,
        consolidation_config: ConsolidationConfig,
    ) -> Result<Self> {
        Ok(Self {
            current_strategy: config.base_strategy.clone(),
            config,
            consolidation_config,
            performance_feedback: VecDeque::new(),
            context_history: VecDeque::new(),
            decision_history: VecDeque::new(),
            strategy_performance: HashMap::new(),
            metrics: AdaptiveReplayMetrics {
                total_adaptive_replays: 0,
                current_adaptation_rate: 0.1,
                strategy_effectiveness: HashMap::new(),
                adaptation_count: 0,
                avg_performance_improvement: 0.0,
                context_adaptation_accuracy: 0.0,
                multi_objective_score: 0.0,
                last_adaptation: Utc::now(),
            },
            last_adaptation: None,
        })
    }

    /// Make adaptive replay decision for a memory
    pub async fn make_adaptive_decision(
        &mut self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
    ) -> Result<AdaptiveReplayDecision> {
        tracing::debug!("Making adaptive replay decision for memory: {}", memory.key);

        // Check if adaptation is needed
        if self.should_adapt().await? {
            self.perform_adaptation(context).await?;
        }

        // Select optimal strategy based on current context
        let selected_strategy = self.select_optimal_strategy(memory, importance, context).await?;

        // Calculate adaptive replay priority
        let replay_priority = self.calculate_adaptive_priority(memory, importance, context, &selected_strategy).await?;

        // Calculate scheduled time using adaptive scheduling
        let scheduled_time = self.calculate_adaptive_schedule(memory, importance, context, &selected_strategy).await?;

        // Estimate effectiveness
        let expected_effectiveness = self.estimate_effectiveness(&selected_strategy, memory, importance, context).await?;

        // Calculate adaptation confidence
        let adaptation_confidence = self.calculate_adaptation_confidence(&selected_strategy).await?;

        // Extract context factors
        let context_factors = self.extract_context_factors(context).await?;

        let decision = AdaptiveReplayDecision {
            memory_key: memory.key.clone(),
            selected_strategy,
            replay_priority,
            scheduled_time,
            expected_effectiveness,
            adaptation_confidence,
            context_factors,
            decided_at: Utc::now(),
        };

        // Store decision in history
        self.decision_history.push_back(decision.clone());
        if self.decision_history.len() > 1000 {
            self.decision_history.pop_front();
        }

        self.metrics.total_adaptive_replays += 1;

        Ok(decision)
    }

    /// Provide performance feedback for adaptation
    pub async fn provide_feedback(&mut self, feedback: ReplayPerformanceFeedback) -> Result<()> {
        tracing::debug!("Receiving performance feedback for memory: {}", feedback.memory_key);

        // Store feedback
        self.performance_feedback.push_back(feedback.clone());
        if self.performance_feedback.len() > self.config.evaluation_window_size {
            self.performance_feedback.pop_front();
        }

        // Update strategy performance tracking
        let strategy_key = self.get_strategy_key(&self.current_strategy);
        self.strategy_performance
            .entry(strategy_key.clone())
            .or_insert_with(Vec::new)
            .push(feedback.success_rate);

        // Update metrics
        self.update_performance_metrics().await?;

        tracing::debug!("Performance feedback processed for strategy: {}", strategy_key);

        Ok(())
    }

    /// Update context information
    pub async fn update_context(&mut self, context: ReplayContext) -> Result<()> {
        self.context_history.push_back(context);
        if self.context_history.len() > 100 {
            self.context_history.pop_front();
        }
        Ok(())
    }

    /// Check if adaptation should be performed
    async fn should_adapt(&self) -> Result<bool> {
        if let Some(last_adaptation) = self.last_adaptation {
            let hours_since = (Utc::now() - last_adaptation).num_hours();
            if hours_since < self.config.adaptation_frequency_hours as i64 {
                return Ok(false);
            }
        }

        // Check if performance indicates need for adaptation
        if self.performance_feedback.len() < 10 {
            return Ok(false);
        }

        let recent_performance: f64 = self.performance_feedback
            .iter()
            .rev()
            .take(10)
            .map(|f| f.success_rate)
            .sum::<f64>() / 10.0;

        let overall_performance: f64 = self.performance_feedback
            .iter()
            .map(|f| f.success_rate)
            .sum::<f64>() / self.performance_feedback.len() as f64;

        let performance_decline = overall_performance - recent_performance;
        Ok(performance_decline > self.config.min_adaptation_threshold)
    }

    /// Perform strategy adaptation
    async fn perform_adaptation(&mut self, context: &ReplayContext) -> Result<()> {
        tracing::info!("Performing adaptive replay strategy adaptation");

        // Clone the current strategy to avoid borrowing issues
        let current_strategy = self.current_strategy.clone();

        match current_strategy {
            AdaptiveReplayStrategy::PerformanceDriven { success_threshold, adaptation_rate } => {
                self.adapt_performance_driven(success_threshold, adaptation_rate).await?;
            },
            AdaptiveReplayStrategy::ContextAware { context_weights, adaptation_window } => {
                self.adapt_context_aware(&context_weights, adaptation_window, context).await?;
            },
            AdaptiveReplayStrategy::MultiObjective { objectives, weights } => {
                self.adapt_multi_objective(&objectives, &weights).await?;
            },
            AdaptiveReplayStrategy::ReinforcementLearning { exploration_rate, learning_rate, discount_factor } => {
                self.adapt_reinforcement_learning(exploration_rate, learning_rate, discount_factor).await?;
            },
            AdaptiveReplayStrategy::Hybrid { strategies, selection_policy } => {
                self.adapt_hybrid_strategy(&strategies, &selection_policy).await?;
            },
        }

        self.metrics.adaptation_count += 1;
        self.metrics.last_adaptation = Utc::now();
        self.last_adaptation = Some(Utc::now());

        tracing::info!("Adaptive replay strategy adaptation completed");

        Ok(())
    }

    /// Select optimal strategy based on context using sophisticated selection algorithms
    async fn select_optimal_strategy(
        &self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
    ) -> Result<AdaptiveReplayStrategy> {
        // Multi-criteria decision analysis for strategy selection
        let mut strategy_scores = HashMap::new();

        // Evaluate each available strategy type
        let candidate_strategies = vec![
            AdaptiveReplayStrategy::PerformanceDriven {
                success_threshold: 0.7,
                adaptation_rate: 0.1,
            },
            AdaptiveReplayStrategy::ContextAware {
                context_weights: self.create_default_context_weights(),
                adaptation_window: 24,
            },
            AdaptiveReplayStrategy::MultiObjective {
                objectives: vec![
                    ReplayObjective::MaximizeRetention,
                    ReplayObjective::OptimizeLearningEfficiency,
                    ReplayObjective::MinimizeInterference,
                ],
                weights: vec![0.4, 0.35, 0.25],
            },
            AdaptiveReplayStrategy::ReinforcementLearning {
                exploration_rate: 0.1,
                learning_rate: 0.05,
                discount_factor: 0.95,
            },
        ];

        for strategy in candidate_strategies {
            let score = self.evaluate_strategy_fitness(&strategy, memory, importance, context).await?;
            strategy_scores.insert(self.get_strategy_key(&strategy), (strategy, score));
        }

        // Select strategy with highest fitness score
        let best_strategy = strategy_scores
            .values()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy, _)| strategy.clone())
            .unwrap_or_else(|| self.current_strategy.clone());

        // Consider hybrid approach if multiple strategies score similarly
        let top_strategies: Vec<_> = strategy_scores
            .values()
            .filter(|(_, score)| *score > 0.7)
            .collect();

        if top_strategies.len() >= 2 {
            let hybrid_strategies: Vec<_> = top_strategies
                .into_iter()
                .map(|(strategy, _)| strategy.clone())
                .collect();

            Ok(AdaptiveReplayStrategy::Hybrid {
                strategies: hybrid_strategies,
                selection_policy: SelectionPolicy::PerformanceBased,
            })
        } else {
            Ok(best_strategy)
        }
    }

    /// Calculate adaptive replay priority
    async fn calculate_adaptive_priority(
        &self,
        _memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
        strategy: &AdaptiveReplayStrategy,
    ) -> Result<f64> {
        let mut priority_factors = Vec::new();

        // Base importance
        priority_factors.push(importance.importance_score);

        // Context-based adjustments
        priority_factors.push(context.activity_level);
        priority_factors.push(1.0 - context.system_load); // Lower load = higher priority
        priority_factors.push(context.time_of_day_factor);

        // Strategy-specific adjustments
        let strategy_factor = match strategy {
            AdaptiveReplayStrategy::PerformanceDriven { success_threshold, .. } => {
                if importance.importance_score > *success_threshold { 1.2 } else { 0.8 }
            },
            AdaptiveReplayStrategy::ContextAware { .. } => {
                context.performance_trends.last().copied().unwrap_or(0.5)
            },
            _ => 1.0,
        };
        priority_factors.push(strategy_factor);

        // Calculate weighted priority
        let weights = [0.3, 0.2, 0.2, 0.15, 0.15];
        let priority: f64 = priority_factors.iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        Ok(priority.min(1.0).max(0.0))
    }

    /// Calculate adaptive scheduling time
    async fn calculate_adaptive_schedule(
        &self,
        _memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
        strategy: &AdaptiveReplayStrategy,
    ) -> Result<DateTime<Utc>> {
        let base_delay_hours = 24.0; // Base 24-hour delay

        let adaptive_factor = match strategy {
            AdaptiveReplayStrategy::PerformanceDriven { .. } => {
                // Higher importance = shorter delay
                2.0 - importance.importance_score
            },
            AdaptiveReplayStrategy::ContextAware { .. } => {
                // Adjust based on context
                if context.activity_level > 0.7 { 0.5 } else { 1.5 }
            },
            _ => 1.0,
        };

        let delay_hours = base_delay_hours * adaptive_factor;
        Ok(Utc::now() + Duration::hours(delay_hours as i64))
    }

    /// Estimate strategy effectiveness
    async fn estimate_effectiveness(
        &self,
        strategy: &AdaptiveReplayStrategy,
        _memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
    ) -> Result<f64> {
        let strategy_key = self.get_strategy_key(strategy);
        
        // Get historical performance for this strategy
        let historical_performance = self.strategy_performance
            .get(&strategy_key)
            .map(|performances| {
                performances.iter().sum::<f64>() / performances.len() as f64
            })
            .unwrap_or(0.5);

        // Adjust based on current context and importance
        let context_adjustment = (context.activity_level + importance.importance_score) / 2.0;
        let estimated_effectiveness = historical_performance * 0.7 + context_adjustment * 0.3;

        Ok(estimated_effectiveness.min(1.0).max(0.0))
    }

    /// Calculate adaptation confidence
    async fn calculate_adaptation_confidence(&self, strategy: &AdaptiveReplayStrategy) -> Result<f64> {
        let strategy_key = self.get_strategy_key(strategy);
        
        // Confidence based on amount of historical data
        let data_points = self.strategy_performance
            .get(&strategy_key)
            .map(|performances| performances.len())
            .unwrap_or(0);

        let confidence = (data_points as f64 / 100.0).min(1.0);
        Ok(confidence)
    }

    /// Extract context factors for decision
    async fn extract_context_factors(&self, context: &ReplayContext) -> Result<HashMap<String, f64>> {
        let mut factors = HashMap::new();
        factors.insert("activity_level".to_string(), context.activity_level);
        factors.insert("system_load".to_string(), context.system_load);
        factors.insert("time_of_day_factor".to_string(), context.time_of_day_factor);
        
        if let Some(trend) = context.performance_trends.last() {
            factors.insert("performance_trend".to_string(), *trend);
        }

        Ok(factors)
    }

    /// Get strategy identifier key
    fn get_strategy_key(&self, strategy: &AdaptiveReplayStrategy) -> String {
        match strategy {
            AdaptiveReplayStrategy::PerformanceDriven { .. } => "performance_driven".to_string(),
            AdaptiveReplayStrategy::ContextAware { .. } => "context_aware".to_string(),
            AdaptiveReplayStrategy::MultiObjective { .. } => "multi_objective".to_string(),
            AdaptiveReplayStrategy::ReinforcementLearning { .. } => "reinforcement_learning".to_string(),
            AdaptiveReplayStrategy::Hybrid { .. } => "hybrid".to_string(),
        }
    }

    /// Update performance metrics
    async fn update_performance_metrics(&mut self) -> Result<()> {
        if self.performance_feedback.is_empty() {
            return Ok(());
        }

        // Calculate average performance improvement
        let recent_feedback: Vec<_> = self.performance_feedback.iter().rev().take(20).collect();
        if recent_feedback.len() >= 20 {
            let recent_avg = recent_feedback.iter().take(10).map(|f| f.success_rate).sum::<f64>() / 10.0;
            let older_count = recent_feedback.len().saturating_sub(10);
            if older_count > 0 {
                let older_avg = recent_feedback.iter().skip(10).map(|f| f.success_rate).sum::<f64>() / older_count as f64;
                self.metrics.avg_performance_improvement = recent_avg - older_avg;
            }
        }

        // Update strategy effectiveness
        for (strategy_key, performances) in &self.strategy_performance {
            if !performances.is_empty() {
                let effectiveness = performances.iter().sum::<f64>() / performances.len() as f64;
                self.metrics.strategy_effectiveness.insert(strategy_key.clone(), effectiveness);
            }
        }

        // Calculate context adaptation accuracy based on prediction vs actual performance
        if self.context_history.len() > 5 && self.performance_feedback.len() > 5 {
            let mut prediction_errors = Vec::new();

            // Compare predicted vs actual effectiveness for recent decisions
            for decision in self.decision_history.iter().rev().take(10) {
                if let Some(feedback) = self.performance_feedback.iter()
                    .find(|f| f.memory_key == decision.memory_key) {
                    let prediction_error = (decision.expected_effectiveness - feedback.success_rate).abs();
                    prediction_errors.push(prediction_error);
                }
            }

            if !prediction_errors.is_empty() {
                let mean_error = prediction_errors.iter().sum::<f64>() / prediction_errors.len() as f64;
                self.metrics.context_adaptation_accuracy = (1.0 - mean_error).max(0.0);
            }
        }

        // Calculate multi-objective optimization score
        if let AdaptiveReplayStrategy::MultiObjective { objectives, weights } = &self.current_strategy {
            if self.performance_feedback.len() >= 5 {
                let recent_feedback: Vec<_> = self.performance_feedback.iter().rev().take(5).collect();
                let mut objective_scores = Vec::new();

                for objective in objectives {
                    let score = match objective {
                        ReplayObjective::MaximizeRetention => {
                            recent_feedback.iter().map(|f| f.retention_improvement).sum::<f64>() / recent_feedback.len() as f64
                        },
                        ReplayObjective::MinimizeInterference => {
                            1.0 - (recent_feedback.iter().map(|f| f.interference_level).sum::<f64>() / recent_feedback.len() as f64)
                        },
                        ReplayObjective::OptimizeLearningEfficiency => {
                            recent_feedback.iter().map(|f| f.learning_efficiency).sum::<f64>() / recent_feedback.len() as f64
                        },
                        ReplayObjective::BalanceCoverageDepth => {
                            // Calculate balance between coverage and depth
                            let coverage_score = recent_feedback.len() as f64 / 10.0; // Normalize to expected feedback count
                            let depth_score = recent_feedback.iter().map(|f| f.context_relevance).sum::<f64>() / recent_feedback.len() as f64;
                            (coverage_score + depth_score) / 2.0
                        },
                        ReplayObjective::MinimizeComputationalCost => {
                            1.0 - (recent_feedback.iter().map(|f| f.computational_cost).sum::<f64>() / recent_feedback.len() as f64)
                        },
                    };
                    objective_scores.push(score);
                }

                // Calculate weighted multi-objective score
                if objective_scores.len() == weights.len() {
                    self.metrics.multi_objective_score = objective_scores.iter()
                        .zip(weights.iter())
                        .map(|(score, weight)| score * weight)
                        .sum();
                }
            }
        }

        Ok(())
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &AdaptiveReplayMetrics {
        &self.metrics
    }

    /// Get decision history
    pub fn get_decision_history(&self, limit: usize) -> Vec<&AdaptiveReplayDecision> {
        self.decision_history.iter().rev().take(limit).collect()
    }

    /// Get performance feedback history
    pub fn get_feedback_history(&self, limit: usize) -> Vec<&ReplayPerformanceFeedback> {
        self.performance_feedback.iter().rev().take(limit).collect()
    }

    /// Adapt performance-driven strategy
    async fn adapt_performance_driven(&mut self, success_threshold: f64, adaptation_rate: f64) -> Result<()> {
        if self.performance_feedback.len() < 10 {
            return Ok(());
        }

        let recent_success_rate: f64 = self.performance_feedback
            .iter()
            .rev()
            .take(10)
            .map(|f| f.success_rate)
            .sum::<f64>() / 10.0;

        // Adjust threshold based on recent performance
        let new_threshold = if recent_success_rate < success_threshold {
            (success_threshold - adaptation_rate * 0.1).max(0.3)
        } else {
            (success_threshold + adaptation_rate * 0.05).min(0.9)
        };

        // Update current strategy
        self.current_strategy = AdaptiveReplayStrategy::PerformanceDriven {
            success_threshold: new_threshold,
            adaptation_rate,
        };

        self.metrics.current_adaptation_rate = adaptation_rate;

        tracing::debug!("Adapted performance-driven strategy: threshold {:.3} -> {:.3}",
                       success_threshold, new_threshold);

        Ok(())
    }

    /// Adapt context-aware strategy
    async fn adapt_context_aware(
        &mut self,
        context_weights: &HashMap<String, f64>,
        adaptation_window: u64,
        _current_context: &ReplayContext,
    ) -> Result<()> {
        let mut new_weights = context_weights.clone();

        // Analyze recent context performance
        if self.context_history.len() >= 5 {
            let recent_contexts: Vec<_> = self.context_history.iter().rev().take(5).collect();

            // Adjust weights based on context effectiveness
            for context in recent_contexts {
                if context.activity_level > 0.7 {
                    *new_weights.entry("activity_level".to_string()).or_insert(0.5) += 0.1;
                }
                if context.system_load < 0.3 {
                    *new_weights.entry("system_load".to_string()).or_insert(0.5) += 0.1;
                }
            }

            // Normalize weights
            let total_weight: f64 = new_weights.values().sum();
            if total_weight > 0.0 {
                for weight in new_weights.values_mut() {
                    *weight /= total_weight;
                }
            }
        }

        self.current_strategy = AdaptiveReplayStrategy::ContextAware {
            context_weights: new_weights,
            adaptation_window,
        };

        tracing::debug!("Adapted context-aware strategy with updated weights");

        Ok(())
    }

    /// Adapt multi-objective strategy
    async fn adapt_multi_objective(
        &mut self,
        objectives: &[ReplayObjective],
        weights: &[f64],
    ) -> Result<()> {
        let mut new_weights = weights.to_vec();

        // Analyze objective performance
        if self.performance_feedback.len() >= 10 {
            let recent_feedback: Vec<_> = self.performance_feedback.iter().rev().take(10).collect();

            // Adjust weights based on objective achievement
            for (i, objective) in objectives.iter().enumerate() {
                if i < new_weights.len() {
                    let objective_score = match objective {
                        ReplayObjective::MaximizeRetention => {
                            recent_feedback.iter().map(|f| f.retention_improvement).sum::<f64>() / recent_feedback.len() as f64
                        },
                        ReplayObjective::MinimizeInterference => {
                            1.0 - (recent_feedback.iter().map(|f| f.interference_level).sum::<f64>() / recent_feedback.len() as f64)
                        },
                        ReplayObjective::OptimizeLearningEfficiency => {
                            recent_feedback.iter().map(|f| f.learning_efficiency).sum::<f64>() / recent_feedback.len() as f64
                        },
                        _ => 0.5,
                    };

                    // Increase weight for well-performing objectives
                    if objective_score > 0.7 {
                        new_weights[i] *= 1.1;
                    } else if objective_score < 0.3 {
                        new_weights[i] *= 0.9;
                    }
                }
            }

            // Normalize weights
            let total_weight: f64 = new_weights.iter().sum();
            if total_weight > 0.0 {
                for weight in &mut new_weights {
                    *weight /= total_weight;
                }
            }
        }

        self.current_strategy = AdaptiveReplayStrategy::MultiObjective {
            objectives: objectives.to_vec(),
            weights: new_weights,
        };

        tracing::debug!("Adapted multi-objective strategy with updated weights");

        Ok(())
    }

    /// Adapt reinforcement learning strategy
    async fn adapt_reinforcement_learning(
        &mut self,
        exploration_rate: f64,
        learning_rate: f64,
        discount_factor: f64,
    ) -> Result<()> {
        // Decay exploration rate over time
        let new_exploration_rate = (exploration_rate * 0.99).max(0.01);

        // Adjust learning rate based on performance stability
        let new_learning_rate = if self.performance_feedback.len() >= 10 {
            let recent_variance = self.calculate_performance_variance().await?;
            if recent_variance > 0.1 {
                (learning_rate * 1.1).min(0.5) // Increase learning rate for high variance
            } else {
                (learning_rate * 0.95).max(0.01) // Decrease for stable performance
            }
        } else {
            learning_rate
        };

        self.current_strategy = AdaptiveReplayStrategy::ReinforcementLearning {
            exploration_rate: new_exploration_rate,
            learning_rate: new_learning_rate,
            discount_factor,
        };

        tracing::debug!("Adapted RL strategy: exploration {:.3} -> {:.3}, learning {:.3} -> {:.3}",
                       exploration_rate, new_exploration_rate, learning_rate, new_learning_rate);

        Ok(())
    }

    /// Adapt hybrid strategy
    async fn adapt_hybrid_strategy(
        &mut self,
        strategies: &[AdaptiveReplayStrategy],
        selection_policy: &SelectionPolicy,
    ) -> Result<()> {
        // For hybrid strategies, we could adapt the selection policy
        // or the weights of individual strategies
        let new_policy = match selection_policy {
            SelectionPolicy::PerformanceBased => {
                // Could switch to contextual bandit if performance is inconsistent
                if self.calculate_performance_variance().await? > 0.15 {
                    SelectionPolicy::ContextualBandit
                } else {
                    selection_policy.clone()
                }
            },
            _ => selection_policy.clone(),
        };

        self.current_strategy = AdaptiveReplayStrategy::Hybrid {
            strategies: strategies.to_vec(),
            selection_policy: new_policy,
        };

        tracing::debug!("Adapted hybrid strategy with updated selection policy");

        Ok(())
    }

    /// Calculate performance variance for adaptation decisions
    async fn calculate_performance_variance(&self) -> Result<f64> {
        if self.performance_feedback.len() < 5 {
            return Ok(0.0);
        }

        let recent_scores: Vec<f64> = self.performance_feedback
            .iter()
            .rev()
            .take(10)
            .map(|f| f.success_rate)
            .collect();

        let mean = recent_scores.iter().sum::<f64>() / recent_scores.len() as f64;
        let variance = recent_scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / recent_scores.len() as f64;

        Ok(variance)
    }

    /// Evaluate strategy fitness for given context
    async fn evaluate_strategy_fitness(
        &self,
        strategy: &AdaptiveReplayStrategy,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
        context: &ReplayContext,
    ) -> Result<f64> {
        let mut fitness_factors = Vec::new();

        // Historical performance factor
        let strategy_key = self.get_strategy_key(strategy);
        let historical_performance = self.strategy_performance
            .get(&strategy_key)
            .map(|performances| {
                if performances.is_empty() {
                    0.5 // Neutral score for new strategies
                } else {
                    performances.iter().sum::<f64>() / performances.len() as f64
                }
            })
            .unwrap_or(0.5);
        fitness_factors.push(historical_performance);

        // Context compatibility factor
        let context_compatibility = match strategy {
            AdaptiveReplayStrategy::PerformanceDriven { success_threshold, .. } => {
                // Good for high-importance memories with clear success metrics
                if importance.importance_score > *success_threshold && context.performance_trends.len() >= 3 {
                    0.9
                } else {
                    0.6
                }
            },
            AdaptiveReplayStrategy::ContextAware { .. } => {
                // Good for dynamic environments with varying context
                let context_variance = self.calculate_context_variance(context).await?;
                if context_variance > 0.1 { 0.8 } else { 0.5 }
            },
            AdaptiveReplayStrategy::MultiObjective { .. } => {
                // Good for complex scenarios requiring balanced optimization
                if importance.importance_score > 0.6 && context.activity_level > 0.5 {
                    0.85
                } else {
                    0.7
                }
            },
            AdaptiveReplayStrategy::ReinforcementLearning { .. } => {
                // Good for learning scenarios with feedback
                if self.performance_feedback.len() >= 10 { 0.8 } else { 0.4 }
            },
            AdaptiveReplayStrategy::Hybrid { .. } => {
                // Generally good but with overhead
                0.75
            },
        };
        fitness_factors.push(context_compatibility);

        // Memory characteristics factor
        let memory_factor = self.calculate_memory_fitness_factor(memory, importance).await?;
        fitness_factors.push(memory_factor);

        // System resource factor
        let resource_factor = 1.0 - context.system_load; // Lower load = better fitness
        fitness_factors.push(resource_factor);

        // Time-based factor
        let time_factor = context.time_of_day_factor;
        fitness_factors.push(time_factor);

        // Calculate weighted fitness score
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1];
        let fitness_score: f64 = fitness_factors.iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        Ok(fitness_score.min(1.0).max(0.0))
    }

    /// Calculate context variance for strategy selection
    async fn calculate_context_variance(&self, _current_context: &ReplayContext) -> Result<f64> {
        if self.context_history.len() < 3 {
            return Ok(0.0);
        }

        let recent_contexts: Vec<_> = self.context_history.iter().rev().take(5).collect();

        // Calculate variance in activity level
        let activity_levels: Vec<f64> = recent_contexts.iter().map(|c| c.activity_level).collect();
        let activity_mean = activity_levels.iter().sum::<f64>() / activity_levels.len() as f64;
        let activity_variance = activity_levels.iter()
            .map(|level| (level - activity_mean).powi(2))
            .sum::<f64>() / activity_levels.len() as f64;

        // Calculate variance in system load
        let load_levels: Vec<f64> = recent_contexts.iter().map(|c| c.system_load).collect();
        let load_mean = load_levels.iter().sum::<f64>() / load_levels.len() as f64;
        let load_variance = load_levels.iter()
            .map(|load| (load - load_mean).powi(2))
            .sum::<f64>() / load_levels.len() as f64;

        // Combined variance
        let combined_variance = (activity_variance + load_variance) / 2.0;
        Ok(combined_variance)
    }

    /// Calculate memory-specific fitness factor
    async fn calculate_memory_fitness_factor(
        &self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
    ) -> Result<f64> {
        let mut factors = Vec::new();

        // Importance factor
        factors.push(importance.importance_score);

        // Access pattern factor
        let access_frequency = memory.access_count() as f64 /
            (Utc::now() - memory.created_at()).num_days().max(1) as f64;
        let access_factor = (access_frequency / 5.0).min(1.0); // Normalize to daily access
        factors.push(access_factor);

        // Recency factor
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours() as f64;
        let recency_factor = if hours_since_access < 24.0 {
            1.0 - (hours_since_access / 24.0)
        } else {
            0.1 // Old memories get low recency score
        };
        factors.push(recency_factor);

        // Content complexity factor
        let content_length = memory.value.len();
        let complexity_factor = if content_length > 1000 {
            0.9 // Complex content benefits from sophisticated replay
        } else if content_length > 100 {
            0.7
        } else {
            0.5 // Simple content needs less sophisticated replay
        };
        factors.push(complexity_factor);

        // Calculate weighted average
        let weights = [0.4, 0.25, 0.2, 0.15];
        let fitness_factor: f64 = factors.iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        Ok(fitness_factor.min(1.0).max(0.0))
    }

    /// Create default context weights for context-aware strategy
    fn create_default_context_weights(&self) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("activity_level".to_string(), 0.3);
        weights.insert("system_load".to_string(), 0.25);
        weights.insert("time_of_day_factor".to_string(), 0.2);
        weights.insert("performance_trend".to_string(), 0.25);
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    fn create_test_context() -> ReplayContext {
        ReplayContext {
            learning_phase: "active".to_string(),
            memory_domain: "general".to_string(),
            activity_level: 0.7,
            system_load: 0.3,
            time_of_day_factor: 0.8,
            performance_trends: vec![0.6, 0.7, 0.8],
            timestamp: Utc::now(),
        }
    }

    fn create_test_feedback(memory_key: &str, success_rate: f64) -> ReplayPerformanceFeedback {
        ReplayPerformanceFeedback {
            memory_key: memory_key.to_string(),
            success_rate,
            retention_improvement: 0.1,
            learning_efficiency: 0.8,
            interference_level: 0.2,
            context_relevance: 0.9,
            computational_cost: 0.5,
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_adaptive_replay_creation() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config);
        assert!(mechanisms.is_ok());
    }

    #[tokio::test]
    async fn test_performance_driven_adaptation() {
        let config = AdaptiveReplayConfig {
            base_strategy: AdaptiveReplayStrategy::PerformanceDriven {
                success_threshold: 0.7,
                adaptation_rate: 0.1,
            },
            ..Default::default()
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add performance feedback indicating poor performance
        for i in 0..15 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.4); // Low success rate
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let context = create_test_context();

        // Force adaptation
        mechanisms.perform_adaptation(&context).await.unwrap();

        // Check that adaptation occurred
        assert!(mechanisms.metrics.adaptation_count > 0);
        assert!(mechanisms.last_adaptation.is_some());
    }

    #[tokio::test]
    async fn test_context_aware_adaptation() {
        let mut context_weights = HashMap::new();
        context_weights.insert("activity_level".to_string(), 0.5);
        context_weights.insert("system_load".to_string(), 0.3);

        let config = AdaptiveReplayConfig {
            base_strategy: AdaptiveReplayStrategy::ContextAware {
                context_weights,
                adaptation_window: 24,
            },
            ..Default::default()
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add context history
        for _ in 0..10 {
            let context = create_test_context();
            mechanisms.update_context(context).await.unwrap();
        }

        let context = create_test_context();
        mechanisms.perform_adaptation(&context).await.unwrap();

        assert!(mechanisms.metrics.adaptation_count > 0);
    }

    #[tokio::test]
    async fn test_multi_objective_adaptation() {
        let objectives = vec![
            ReplayObjective::MaximizeRetention,
            ReplayObjective::MinimizeInterference,
            ReplayObjective::OptimizeLearningEfficiency,
        ];
        let weights = vec![0.4, 0.3, 0.3];

        let config = AdaptiveReplayConfig {
            base_strategy: AdaptiveReplayStrategy::MultiObjective {
                objectives,
                weights,
            },
            ..Default::default()
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add performance feedback
        for i in 0..15 {
            let mut feedback = create_test_feedback(&format!("key_{}", i), 0.8);
            feedback.retention_improvement = 0.9; // High retention
            feedback.interference_level = 0.1; // Low interference
            feedback.learning_efficiency = 0.8; // Good efficiency
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let context = create_test_context();
        mechanisms.perform_adaptation(&context).await.unwrap();

        assert!(mechanisms.metrics.adaptation_count > 0);
    }

    #[tokio::test]
    async fn test_reinforcement_learning_adaptation() {
        let config = AdaptiveReplayConfig {
            base_strategy: AdaptiveReplayStrategy::ReinforcementLearning {
                exploration_rate: 0.3,
                learning_rate: 0.1,
                discount_factor: 0.9,
            },
            ..Default::default()
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add performance feedback with high variance
        let success_rates = [0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.8, 0.2, 0.9];
        for (i, &rate) in success_rates.iter().enumerate() {
            let feedback = create_test_feedback(&format!("key_{}", i), rate);
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let context = create_test_context();
        mechanisms.perform_adaptation(&context).await.unwrap();

        assert!(mechanisms.metrics.adaptation_count > 0);
    }

    #[tokio::test]
    async fn test_adaptive_decision_making() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.8,
            access_frequency: 0.7,
            recency_score: 0.6,
            centrality_score: 0.5,
            uniqueness_score: 0.4,
            temporal_consistency: 0.3,
            calculated_at: Utc::now(),
            fisher_information: None,
        };
        let context = create_test_context();

        let decision = mechanisms.make_adaptive_decision(&memory, &importance, &context).await.unwrap();

        assert_eq!(decision.memory_key, "test_key");
        assert!(decision.replay_priority >= 0.0);
        assert!(decision.replay_priority <= 1.0);
        assert!(decision.expected_effectiveness >= 0.0);
        assert!(decision.expected_effectiveness <= 1.0);
        assert!(decision.adaptation_confidence >= 0.0);
        assert!(decision.adaptation_confidence <= 1.0);
        assert!(!decision.context_factors.is_empty());
    }

    #[tokio::test]
    async fn test_performance_feedback_processing() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add multiple feedback entries
        for i in 0..20 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.7 + (i as f64 * 0.01));
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let metrics = mechanisms.get_metrics();
        assert_eq!(metrics.total_adaptive_replays, 0); // No decisions made yet
        assert!(!metrics.strategy_effectiveness.is_empty());

        let feedback_history = mechanisms.get_feedback_history(10);
        assert_eq!(feedback_history.len(), 10);
    }

    #[tokio::test]
    async fn test_context_factor_extraction() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        let context = create_test_context();
        let factors = mechanisms.extract_context_factors(&context).await.unwrap();

        assert!(factors.contains_key("activity_level"));
        assert!(factors.contains_key("system_load"));
        assert!(factors.contains_key("time_of_day_factor"));
        assert!(factors.contains_key("performance_trend"));

        assert_eq!(factors["activity_level"], 0.7);
        assert_eq!(factors["system_load"], 0.3);
        assert_eq!(factors["time_of_day_factor"], 0.8);
    }

    #[tokio::test]
    async fn test_adaptation_threshold_detection() {
        let config = AdaptiveReplayConfig {
            min_adaptation_threshold: 0.1,
            ..Default::default()
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add feedback showing performance decline
        for i in 0..10 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.8); // Good performance
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        for i in 10..20 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.5); // Poor performance
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let should_adapt = mechanisms.should_adapt().await.unwrap();
        assert!(should_adapt);
    }

    #[tokio::test]
    async fn test_strategy_effectiveness_tracking() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add feedback for current strategy
        for i in 0..10 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.8);
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let strategy_key = mechanisms.get_strategy_key(&mechanisms.current_strategy);
        assert!(mechanisms.strategy_performance.contains_key(&strategy_key));

        let performances = &mechanisms.strategy_performance[&strategy_key];
        assert_eq!(performances.len(), 10);
        assert!(performances.iter().all(|&p| p == 0.8));
    }

    #[tokio::test]
    async fn test_performance_variance_calculation() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add feedback with known variance
        let success_rates = [0.5, 0.6, 0.7, 0.8, 0.9]; // Low variance
        for (i, &rate) in success_rates.iter().enumerate() {
            let feedback = create_test_feedback(&format!("key_{}", i), rate);
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let variance = mechanisms.calculate_performance_variance().await.unwrap();
        assert!(variance >= 0.0);
        assert!(variance < 0.1); // Should be low variance
    }

    #[tokio::test]
    async fn test_strategy_fitness_evaluation() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add some performance history
        for i in 0..10 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.8);
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let memory = MemoryEntry::new(
            "fitness_test".to_string(),
            "Test content for fitness evaluation".to_string(),
            MemoryType::LongTerm,
        );

        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.8,
            access_frequency: 0.7,
            recency_score: 0.9,
            centrality_score: 0.6,
            uniqueness_score: 0.5,
            temporal_consistency: 0.4,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let context = create_test_context();

        let performance_strategy = AdaptiveReplayStrategy::PerformanceDriven {
            success_threshold: 0.7,
            adaptation_rate: 0.1,
        };

        let context_strategy = AdaptiveReplayStrategy::ContextAware {
            context_weights: mechanisms.create_default_context_weights(),
            adaptation_window: 24,
        };

        let performance_fitness = mechanisms.evaluate_strategy_fitness(&performance_strategy, &memory, &importance, &context).await.unwrap();
        let context_fitness = mechanisms.evaluate_strategy_fitness(&context_strategy, &memory, &importance, &context).await.unwrap();

        assert!(performance_fitness >= 0.0 && performance_fitness <= 1.0);
        assert!(context_fitness >= 0.0 && context_fitness <= 1.0);

        // Both should have reasonable fitness scores
        assert!(performance_fitness > 0.3);
        assert!(context_fitness > 0.3);
    }

    #[tokio::test]
    async fn test_memory_fitness_factor_calculation() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Create memory with different characteristics
        let mut recent_memory = MemoryEntry::new(
            "recent_key".to_string(),
            "Recent memory content".to_string(),
            MemoryType::LongTerm,
        );
        recent_memory.mark_accessed(); // Make it recently accessed

        let old_memory = MemoryEntry::new(
            "old_key".to_string(),
            "Old memory content".to_string(),
            MemoryType::LongTerm,
        );

        let high_importance = MemoryImportance {
            memory_key: "recent_key".to_string(),
            importance_score: 0.9,
            access_frequency: 0.8,
            recency_score: 0.95,
            centrality_score: 0.85,
            uniqueness_score: 0.8,
            temporal_consistency: 0.7,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let low_importance = MemoryImportance {
            memory_key: "old_key".to_string(),
            importance_score: 0.2,
            access_frequency: 0.1,
            recency_score: 0.3,
            centrality_score: 0.2,
            uniqueness_score: 0.1,
            temporal_consistency: 0.1,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let recent_high_fitness = mechanisms.calculate_memory_fitness_factor(&recent_memory, &high_importance).await.unwrap();
        let old_low_fitness = mechanisms.calculate_memory_fitness_factor(&old_memory, &low_importance).await.unwrap();

        assert!(recent_high_fitness > old_low_fitness);
        assert!(recent_high_fitness >= 0.0 && recent_high_fitness <= 1.0);
        assert!(old_low_fitness >= 0.0 && old_low_fitness <= 1.0);
    }

    #[tokio::test]
    async fn test_context_variance_with_stable_context() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add stable contexts (low variance)
        for _ in 0..10 {
            let mut context = create_test_context();
            context.activity_level = 0.7; // Stable activity
            context.system_load = 0.3; // Stable load
            mechanisms.update_context(context).await.unwrap();
        }

        let current_context = create_test_context();
        let variance = mechanisms.calculate_context_variance(&current_context).await.unwrap();

        assert!(variance < 0.01); // Should be very low variance
    }

    #[tokio::test]
    async fn test_hybrid_strategy_selection() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Add performance history that makes multiple strategies viable
        for i in 0..20 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.75); // Good performance
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let memory = MemoryEntry::new(
            "hybrid_test".to_string(),
            "Test content for hybrid strategy selection".to_string(),
            MemoryType::LongTerm,
        );

        let importance = MemoryImportance {
            memory_key: "hybrid_test".to_string(),
            importance_score: 0.8,
            access_frequency: 0.7,
            recency_score: 0.9,
            centrality_score: 0.8,
            uniqueness_score: 0.7,
            temporal_consistency: 0.6,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let context = create_test_context();

        let selected_strategy = mechanisms.select_optimal_strategy(&memory, &importance, &context).await.unwrap();

        // With good performance across strategies, might select hybrid
        match selected_strategy {
            AdaptiveReplayStrategy::Hybrid { strategies, .. } => {
                assert!(strategies.len() >= 2);
            },
            _ => {
                // Single strategy is also valid if it has clearly superior fitness
            }
        }
    }

    #[tokio::test]
    async fn test_adaptation_confidence_calculation() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        let strategy = AdaptiveReplayStrategy::PerformanceDriven {
            success_threshold: 0.7,
            adaptation_rate: 0.1,
        };

        // Test with no historical data
        let confidence_no_data = mechanisms.calculate_adaptation_confidence(&strategy).await.unwrap();
        assert_eq!(confidence_no_data, 0.0);

        // Add some performance data
        for i in 0..50 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.8);
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        let confidence_with_data = mechanisms.calculate_adaptation_confidence(&strategy).await.unwrap();
        assert!(confidence_with_data > confidence_no_data);
        assert!(confidence_with_data <= 1.0);
    }

    #[tokio::test]
    async fn test_comprehensive_adaptive_decision_flow() {
        let config = AdaptiveReplayConfig::default();
        let consolidation_config = ConsolidationConfig::default();
        let mut mechanisms = AdaptiveReplayMechanisms::new(config, consolidation_config).unwrap();

        // Build up comprehensive history
        for i in 0..30 {
            let feedback = create_test_feedback(&format!("key_{}", i), 0.7 + (i as f64 * 0.01));
            mechanisms.provide_feedback(feedback).await.unwrap();
        }

        for i in 0..10 {
            let mut context = create_test_context();
            context.activity_level = 0.5 + (i as f64 * 0.05);
            mechanisms.update_context(context).await.unwrap();
        }

        let memory = MemoryEntry::new(
            "comprehensive_test".to_string(),
            "Comprehensive test content for full adaptive decision flow".to_string(),
            MemoryType::LongTerm,
        );

        let importance = MemoryImportance {
            memory_key: "comprehensive_test".to_string(),
            importance_score: 0.85,
            access_frequency: 0.8,
            recency_score: 0.9,
            centrality_score: 0.75,
            uniqueness_score: 0.7,
            temporal_consistency: 0.6,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let context = create_test_context();

        // Make adaptive decision
        let decision = mechanisms.make_adaptive_decision(&memory, &importance, &context).await.unwrap();

        // Verify decision quality
        assert!(decision.replay_priority > 0.5); // Should be high priority
        assert!(decision.expected_effectiveness > 0.5); // Should be effective
        assert!(decision.adaptation_confidence > 0.0); // Should have some confidence
        assert!(decision.scheduled_time > Utc::now()); // Should be scheduled in future
        assert!(!decision.context_factors.is_empty()); // Should have context factors

        // Verify metrics were updated
        assert_eq!(mechanisms.metrics.total_adaptive_replays, 1);
        assert!(!mechanisms.decision_history.is_empty());
    }
}
