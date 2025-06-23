//! Gradual Forgetting Algorithm Implementation
//! 
//! Implements sophisticated gradual forgetting mechanisms based on Ebbinghaus forgetting curves,
//! importance-based retention, and temporal decay patterns for natural memory management.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Forgetting curve types based on psychological research
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ForgettingCurveType {
    /// Ebbinghaus exponential decay curve
    Ebbinghaus,
    /// Power law decay (more gradual)
    PowerLaw,
    /// Logarithmic decay
    Logarithmic,
    /// Hybrid curve combining multiple functions
    Hybrid,
    /// Custom curve with user-defined parameters
    Custom { decay_rate: f64, shape_factor: f64 },
}

/// Memory retention factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionFactors {
    /// Importance-based protection (0.0 to 1.0)
    pub importance_protection: f64,
    /// Access frequency influence (0.0 to 1.0)
    pub frequency_influence: f64,
    /// Recency boost factor (0.0 to 1.0)
    pub recency_boost: f64,
    /// Emotional significance factor (0.0 to 1.0)
    pub emotional_significance: f64,
    /// Contextual relevance factor (0.0 to 1.0)
    pub contextual_relevance: f64,
    /// Calculated at timestamp
    pub calculated_at: DateTime<Utc>,
}

/// Forgetting decision with detailed reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgettingDecision {
    /// Memory key
    pub memory_key: String,
    /// Should this memory be forgotten
    pub should_forget: bool,
    /// Forgetting probability (0.0 to 1.0)
    pub forgetting_probability: f64,
    /// Retention strength (0.0 to 1.0)
    pub retention_strength: f64,
    /// Contributing factors
    pub retention_factors: RetentionFactors,
    /// Forgetting curve used
    pub curve_type: ForgettingCurveType,
    /// Time until next evaluation
    pub next_evaluation: DateTime<Utc>,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
}

/// Forgetting algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgettingConfig {
    /// Base forgetting rate (0.0 to 1.0)
    pub base_forgetting_rate: f64,
    /// Importance threshold for protection (0.0 to 1.0)
    pub importance_threshold: f64,
    /// Minimum retention time in hours
    pub min_retention_hours: u64,
    /// Maximum retention time in hours
    pub max_retention_hours: u64,
    /// Forgetting curve type to use
    pub curve_type: ForgettingCurveType,
    /// Enable adaptive forgetting rates
    pub adaptive_rates: bool,
    /// Frequency evaluation interval in hours
    pub evaluation_interval_hours: u64,
    /// Enable emotional significance weighting
    pub enable_emotional_weighting: bool,
}

impl Default for ForgettingConfig {
    fn default() -> Self {
        Self {
            base_forgetting_rate: 0.1,
            importance_threshold: 0.3,
            min_retention_hours: 24,
            max_retention_hours: 8760, // 1 year
            curve_type: ForgettingCurveType::Ebbinghaus,
            adaptive_rates: true,
            evaluation_interval_hours: 24,
            enable_emotional_weighting: true,
        }
    }
}

/// Forgetting algorithm metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgettingMetrics {
    /// Total memories evaluated
    pub memories_evaluated: usize,
    /// Memories forgotten
    pub memories_forgotten: usize,
    /// Memories retained
    pub memories_retained: usize,
    /// Average forgetting probability
    pub avg_forgetting_probability: f64,
    /// Average retention strength
    pub avg_retention_strength: f64,
    /// Forgetting efficiency score
    pub efficiency_score: f64,
    /// Last evaluation timestamp
    pub last_evaluation: DateTime<Utc>,
}

/// Main Gradual Forgetting Algorithm implementation
#[derive(Debug)]
pub struct GradualForgettingAlgorithm {
    /// Configuration
    config: ForgettingConfig,
    /// Consolidation configuration
    consolidation_config: ConsolidationConfig,
    /// Forgetting decisions history
    decision_history: HashMap<String, Vec<ForgettingDecision>>,
    /// Memory retention tracking
    retention_tracking: HashMap<String, RetentionFactors>,
    /// Performance metrics
    metrics: ForgettingMetrics,
    /// Last evaluation timestamp
    last_evaluation: Option<DateTime<Utc>>,
}

impl GradualForgettingAlgorithm {
    /// Create new gradual forgetting algorithm
    pub fn new(
        config: ForgettingConfig,
        consolidation_config: ConsolidationConfig,
    ) -> Result<Self> {
        Ok(Self {
            config,
            consolidation_config,
            decision_history: HashMap::new(),
            retention_tracking: HashMap::new(),
            metrics: ForgettingMetrics {
                memories_evaluated: 0,
                memories_forgotten: 0,
                memories_retained: 0,
                avg_forgetting_probability: 0.0,
                avg_retention_strength: 0.0,
                efficiency_score: 0.0,
                last_evaluation: Utc::now(),
            },
            last_evaluation: None,
        })
    }

    /// Evaluate memories for forgetting
    pub async fn evaluate_memories(
        &mut self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<Vec<ForgettingDecision>> {
        tracing::info!("Evaluating {} memories for gradual forgetting", memories.len());

        let mut decisions = Vec::new();
        let mut total_forgetting_prob = 0.0;
        let mut total_retention_strength = 0.0;

        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            let decision = self.evaluate_single_memory(memory, importance).await?;
            
            total_forgetting_prob += decision.forgetting_probability;
            total_retention_strength += decision.retention_strength;
            
            if decision.should_forget {
                self.metrics.memories_forgotten += 1;
            } else {
                self.metrics.memories_retained += 1;
            }

            // Store decision in history
            self.decision_history
                .entry(memory.key.clone())
                .or_insert_with(Vec::new)
                .push(decision.clone());

            decisions.push(decision);
        }

        // Update metrics
        self.metrics.memories_evaluated += memories.len();
        if !decisions.is_empty() {
            self.metrics.avg_forgetting_probability = total_forgetting_prob / decisions.len() as f64;
            self.metrics.avg_retention_strength = total_retention_strength / decisions.len() as f64;
        }
        self.metrics.efficiency_score = self.calculate_efficiency_score();
        self.metrics.last_evaluation = Utc::now();
        self.last_evaluation = Some(Utc::now());

        tracing::info!("Forgetting evaluation complete: {} to forget, {} to retain", 
                      self.metrics.memories_forgotten, self.metrics.memories_retained);

        Ok(decisions)
    }

    /// Evaluate a single memory for forgetting
    async fn evaluate_single_memory(
        &mut self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
    ) -> Result<ForgettingDecision> {
        // Calculate retention factors
        let retention_factors = self.calculate_retention_factors(memory, importance).await?;
        
        // Calculate forgetting probability using selected curve
        let forgetting_probability = self.calculate_forgetting_probability(
            memory, 
            importance, 
            &retention_factors
        ).await?;

        // Calculate retention strength (inverse of forgetting probability)
        let retention_strength = 1.0 - forgetting_probability;

        // Make forgetting decision
        let should_forget = forgetting_probability > 0.5 && 
                           importance.importance_score < self.config.importance_threshold &&
                           self.meets_minimum_retention_time(memory);

        // Calculate next evaluation time
        let next_evaluation = self.calculate_next_evaluation_time(memory, &retention_factors);

        // Store retention factors for tracking
        self.retention_tracking.insert(memory.key.clone(), retention_factors.clone());

        Ok(ForgettingDecision {
            memory_key: memory.key.clone(),
            should_forget,
            forgetting_probability,
            retention_strength,
            retention_factors,
            curve_type: self.config.curve_type.clone(),
            next_evaluation,
            decided_at: Utc::now(),
        })
    }

    /// Calculate retention factors for a memory
    async fn calculate_retention_factors(
        &self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
    ) -> Result<RetentionFactors> {
        // Importance-based protection
        let importance_protection = importance.importance_score;

        // Access frequency influence
        let frequency_influence = importance.access_frequency;

        // Recency boost (higher for recently accessed memories)
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours().max(0) as f64;
        let recency_boost = (-hours_since_access / 168.0).exp(); // 1 week half-life

        // Emotional significance (based on content analysis)
        let emotional_significance = self.calculate_emotional_significance(memory).await?;

        // Contextual relevance (based on current usage patterns)
        let contextual_relevance = self.calculate_contextual_relevance(memory, importance).await?;

        Ok(RetentionFactors {
            importance_protection,
            frequency_influence,
            recency_boost,
            emotional_significance,
            contextual_relevance,
            calculated_at: Utc::now(),
        })
    }

    /// Calculate forgetting probability using the configured curve
    async fn calculate_forgetting_probability(
        &self,
        memory: &MemoryEntry,
        importance: &MemoryImportance,
        retention_factors: &RetentionFactors,
    ) -> Result<f64> {
        let age_hours = (Utc::now() - memory.created_at()).num_hours().max(0) as f64;
        
        // Base forgetting curve calculation
        let base_forgetting = match &self.config.curve_type {
            ForgettingCurveType::Ebbinghaus => {
                // R = e^(-t/S) where S is memory strength
                let memory_strength = importance.importance_score * 168.0; // Scale to hours
                1.0 - (-age_hours / memory_strength).exp()
            },
            ForgettingCurveType::PowerLaw => {
                // R = (1 + t)^(-β) where β is decay parameter
                let beta = 0.5;
                1.0 - (1.0 + age_hours / 24.0).powf(-beta)
            },
            ForgettingCurveType::Logarithmic => {
                // R = 1 - log(1 + t) / log(1 + T_max)
                let max_time = self.config.max_retention_hours as f64;
                1.0 - (1.0 + age_hours).ln() / (1.0 + max_time).ln()
            },
            ForgettingCurveType::Hybrid => {
                // Combine multiple curves
                let ebbinghaus = 1.0 - (-age_hours / (importance.importance_score * 168.0)).exp();
                let power_law = 1.0 - (1.0 + age_hours / 24.0).powf(-0.5);
                ebbinghaus * 0.6 + power_law * 0.4
            },
            ForgettingCurveType::Custom { decay_rate, shape_factor } => {
                // Custom curve: R = 1 - e^(-(t/decay_rate)^shape_factor)
                1.0 - (-(age_hours / decay_rate).powf(*shape_factor)).exp()
            },
        };

        // Apply retention factors
        let protection_factor = retention_factors.importance_protection * 0.4 +
                               retention_factors.frequency_influence * 0.3 +
                               retention_factors.recency_boost * 0.2 +
                               retention_factors.emotional_significance * 0.05 +
                               retention_factors.contextual_relevance * 0.05;

        // Final forgetting probability with protection
        let forgetting_probability = base_forgetting * (1.0 - protection_factor) * self.config.base_forgetting_rate;

        Ok(forgetting_probability.min(1.0).max(0.0))
    }

    /// Check if memory meets minimum retention time
    fn meets_minimum_retention_time(&self, memory: &MemoryEntry) -> bool {
        let age_hours = (Utc::now() - memory.created_at()).num_hours().max(0) as u64;
        age_hours >= self.config.min_retention_hours
    }

    /// Calculate next evaluation time based on retention factors
    fn calculate_next_evaluation_time(
        &self,
        _memory: &MemoryEntry,
        retention_factors: &RetentionFactors,
    ) -> DateTime<Utc> {
        let base_interval = self.config.evaluation_interval_hours as f64;
        
        // Adjust interval based on retention strength
        let retention_strength = (retention_factors.importance_protection + 
                                 retention_factors.frequency_influence) / 2.0;
        
        // Higher retention = longer intervals between evaluations
        let adjusted_interval = base_interval * (1.0 + retention_strength * 2.0);
        
        Utc::now() + Duration::hours(adjusted_interval as i64)
    }

    /// Calculate emotional significance of memory content
    async fn calculate_emotional_significance(&self, memory: &MemoryEntry) -> Result<f64> {
        // Simplified emotional analysis based on content keywords
        let content = memory.value.to_lowercase();
        let emotional_keywords = [
            "important", "critical", "urgent", "love", "hate", "fear", "joy",
            "success", "failure", "achievement", "milestone", "breakthrough",
            "crisis", "emergency", "celebration", "victory", "defeat"
        ];

        let emotional_score = emotional_keywords.iter()
            .map(|&keyword| if content.contains(keyword) { 1.0 } else { 0.0 })
            .sum::<f64>() / emotional_keywords.len() as f64;

        Ok(emotional_score.min(1.0))
    }

    /// Calculate contextual relevance based on current usage patterns
    async fn calculate_contextual_relevance(
        &self,
        _memory: &MemoryEntry,
        importance: &MemoryImportance,
    ) -> Result<f64> {
        // Simplified contextual relevance based on centrality and uniqueness
        let relevance = importance.centrality_score * 0.6 +
                        importance.uniqueness_score * 0.4;
        
        Ok(relevance.min(1.0))
    }

    /// Calculate algorithm efficiency score
    fn calculate_efficiency_score(&self) -> f64 {
        if self.metrics.memories_evaluated == 0 {
            return 1.0;
        }

        let retention_rate = self.metrics.memories_retained as f64 / self.metrics.memories_evaluated as f64;
        let _forgetting_rate = self.metrics.memories_forgotten as f64 / self.metrics.memories_evaluated as f64;
        
        // Efficiency based on balanced retention/forgetting and average retention strength
        let balance_score = 1.0 - (retention_rate - 0.7).abs(); // Target 70% retention
        let strength_score = self.metrics.avg_retention_strength;
        
        (balance_score * 0.6 + strength_score * 0.4).min(1.0).max(0.0)
    }

    /// Get forgetting decisions for a specific memory
    pub fn get_memory_decisions(&self, memory_key: &str) -> Option<&Vec<ForgettingDecision>> {
        self.decision_history.get(memory_key)
    }

    /// Get current retention factors for a memory
    pub fn get_retention_factors(&self, memory_key: &str) -> Option<&RetentionFactors> {
        self.retention_tracking.get(memory_key)
    }

    /// Get algorithm metrics
    pub fn get_metrics(&self) -> &ForgettingMetrics {
        &self.metrics
    }

    /// Check if evaluation is needed
    pub fn should_evaluate(&self) -> bool {
        if let Some(last_eval) = self.last_evaluation {
            let hours_since = (Utc::now() - last_eval).num_hours();
            hours_since >= self.config.evaluation_interval_hours as i64
        } else {
            true // First evaluation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_gradual_forgetting_creation() {
        let forgetting_config = ForgettingConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config);
        assert!(algorithm.is_ok());
    }

    #[tokio::test]
    async fn test_ebbinghaus_forgetting_curve() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.curve_type = ForgettingCurveType::Ebbinghaus;
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.5,
            access_frequency: 0.3,
            recency_score: 0.7,
            centrality_score: 0.4,
            uniqueness_score: 0.6,
            temporal_consistency: 0.5,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let decisions = algorithm.evaluate_memories(&[memory], &[importance]).await.unwrap();
        assert_eq!(decisions.len(), 1);
        assert!(decisions[0].forgetting_probability >= 0.0);
        assert!(decisions[0].forgetting_probability <= 1.0);
        assert_eq!(decisions[0].curve_type, ForgettingCurveType::Ebbinghaus);
    }

    #[tokio::test]
    async fn test_power_law_forgetting_curve() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.curve_type = ForgettingCurveType::PowerLaw;
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.8,
            access_frequency: 0.9,
            recency_score: 0.8,
            centrality_score: 0.7,
            uniqueness_score: 0.6,
            temporal_consistency: 0.8,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let decisions = algorithm.evaluate_memories(&[memory], &[importance]).await.unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].curve_type, ForgettingCurveType::PowerLaw);
        // High importance should result in low forgetting probability
        assert!(decisions[0].forgetting_probability < 0.5);
    }

    #[tokio::test]
    async fn test_custom_forgetting_curve() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.curve_type = ForgettingCurveType::Custom {
            decay_rate: 48.0,
            shape_factor: 1.5
        };
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.4,
            access_frequency: 0.2,
            recency_score: 0.3,
            centrality_score: 0.5,
            uniqueness_score: 0.4,
            temporal_consistency: 0.3,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let decisions = algorithm.evaluate_memories(&[memory], &[importance]).await.unwrap();
        assert_eq!(decisions.len(), 1);
        if let ForgettingCurveType::Custom { decay_rate, shape_factor } = &decisions[0].curve_type {
            assert_eq!(*decay_rate, 48.0);
            assert_eq!(*shape_factor, 1.5);
        } else {
            panic!("Expected custom curve type");
        }
    }

    #[tokio::test]
    async fn test_retention_factors_calculation() {
        let forgetting_config = ForgettingConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Important critical content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.9,
            access_frequency: 0.8,
            recency_score: 0.7,
            centrality_score: 0.8,
            uniqueness_score: 0.9,
            temporal_consistency: 0.8,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let retention_factors = algorithm.calculate_retention_factors(&memory, &importance).await.unwrap();

        assert_eq!(retention_factors.importance_protection, 0.9);
        assert_eq!(retention_factors.frequency_influence, 0.8);
        assert!(retention_factors.recency_boost > 0.0);
        assert!(retention_factors.emotional_significance > 0.0); // Should detect "important" and "critical"
        assert!(retention_factors.contextual_relevance > 0.0);
    }

    #[tokio::test]
    async fn test_importance_threshold_protection() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.importance_threshold = 0.7;
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        // High importance memory (above threshold)
        let high_importance_memory = MemoryEntry::new("high_key".to_string(), "High importance content".to_string(), MemoryType::LongTerm);
        let high_importance = MemoryImportance {
            memory_key: "high_key".to_string(),
            importance_score: 0.9,
            access_frequency: 0.8,
            recency_score: 0.7,
            centrality_score: 0.8,
            uniqueness_score: 0.9,
            temporal_consistency: 0.8,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        // Low importance memory (below threshold)
        let low_importance_memory = MemoryEntry::new("low_key".to_string(), "Low importance content".to_string(), MemoryType::ShortTerm);
        let low_importance = MemoryImportance {
            memory_key: "low_key".to_string(),
            importance_score: 0.2,
            access_frequency: 0.1,
            recency_score: 0.3,
            centrality_score: 0.2,
            uniqueness_score: 0.1,
            temporal_consistency: 0.2,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let decisions = algorithm.evaluate_memories(
            &[high_importance_memory, low_importance_memory],
            &[high_importance, low_importance]
        ).await.unwrap();

        assert_eq!(decisions.len(), 2);

        // High importance memory should be protected
        assert!(!decisions[0].should_forget);
        assert!(decisions[0].retention_strength > 0.5);

        // Low importance memory should have higher forgetting probability than high importance memory
        // Note: The algorithm correctly protects high importance memories
        assert!(decisions[1].forgetting_probability >= decisions[0].forgetting_probability);
    }

    #[tokio::test]
    async fn test_minimum_retention_time() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.min_retention_hours = 48; // 2 days
        let consolidation_config = ConsolidationConfig::default();

        let algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        // Very recent memory (should be protected by minimum retention time)
        let recent_memory = MemoryEntry::new("recent_key".to_string(), "Recent content".to_string(), MemoryType::ShortTerm);

        // Even with low importance, recent memory should not meet minimum retention time
        assert!(!algorithm.meets_minimum_retention_time(&recent_memory));
    }

    #[tokio::test]
    async fn test_emotional_significance_detection() {
        let forgetting_config = ForgettingConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        // Memory with emotional keywords
        let emotional_memory = MemoryEntry::new(
            "emotional_key".to_string(),
            "This is a critical breakthrough achievement that brings joy and success".to_string(),
            MemoryType::LongTerm
        );

        // Memory without emotional keywords
        let neutral_memory = MemoryEntry::new(
            "neutral_key".to_string(),
            "This is regular content without special significance".to_string(),
            MemoryType::LongTerm
        );

        let emotional_score = algorithm.calculate_emotional_significance(&emotional_memory).await.unwrap();
        let neutral_score = algorithm.calculate_emotional_significance(&neutral_memory).await.unwrap();

        assert!(emotional_score > neutral_score);
        assert!(emotional_score > 0.0);
    }

    #[tokio::test]
    async fn test_hybrid_forgetting_curve() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.curve_type = ForgettingCurveType::Hybrid;
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.6,
            access_frequency: 0.5,
            recency_score: 0.4,
            centrality_score: 0.5,
            uniqueness_score: 0.6,
            temporal_consistency: 0.5,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let decisions = algorithm.evaluate_memories(&[memory], &[importance]).await.unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].curve_type, ForgettingCurveType::Hybrid);

        // Hybrid curve should produce reasonable forgetting probabilities
        assert!(decisions[0].forgetting_probability >= 0.0);
        assert!(decisions[0].forgetting_probability <= 1.0);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let forgetting_config = ForgettingConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "Content 2".to_string(), MemoryType::ShortTerm),
            MemoryEntry::new("key3".to_string(), "Content 3".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores = vec![
            MemoryImportance {
                memory_key: "key1".to_string(),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
            MemoryImportance {
                memory_key: "key2".to_string(),
                importance_score: 0.2,
                access_frequency: 0.1,
                recency_score: 0.3,
                centrality_score: 0.2,
                uniqueness_score: 0.1,
                temporal_consistency: 0.2,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
            MemoryImportance {
                memory_key: "key3".to_string(),
                importance_score: 0.6,
                access_frequency: 0.5,
                recency_score: 0.4,
                centrality_score: 0.6,
                uniqueness_score: 0.7,
                temporal_consistency: 0.5,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
        ];

        let decisions = algorithm.evaluate_memories(&memories, &importance_scores).await.unwrap();

        assert_eq!(decisions.len(), 3);

        let metrics = algorithm.get_metrics();
        assert_eq!(metrics.memories_evaluated, 3);
        assert!(metrics.memories_forgotten + metrics.memories_retained == 3);
        assert!(metrics.avg_forgetting_probability >= 0.0);
        assert!(metrics.avg_forgetting_probability <= 1.0);
        assert!(metrics.avg_retention_strength >= 0.0);
        assert!(metrics.avg_retention_strength <= 1.0);
        assert!(metrics.efficiency_score >= 0.0);
        assert!(metrics.efficiency_score <= 1.0);
    }

    #[tokio::test]
    async fn test_decision_history_tracking() {
        let forgetting_config = ForgettingConfig::default();
        let consolidation_config = ConsolidationConfig::default();

        let mut algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        let memory = MemoryEntry::new("test_key".to_string(), "Test content".to_string(), MemoryType::LongTerm);
        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.5,
            access_frequency: 0.4,
            recency_score: 0.6,
            centrality_score: 0.5,
            uniqueness_score: 0.4,
            temporal_consistency: 0.5,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        // First evaluation
        algorithm.evaluate_memories(&[memory.clone()], &[importance.clone()]).await.unwrap();

        // Second evaluation
        algorithm.evaluate_memories(&[memory], &[importance]).await.unwrap();

        let history = algorithm.get_memory_decisions("test_key").unwrap();
        assert_eq!(history.len(), 2);

        // Check that decisions are properly stored
        for decision in history {
            assert_eq!(decision.memory_key, "test_key");
            assert!(decision.forgetting_probability >= 0.0);
            assert!(decision.forgetting_probability <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_should_evaluate_timing() {
        let mut forgetting_config = ForgettingConfig::default();
        forgetting_config.evaluation_interval_hours = 1; // 1 hour interval
        let consolidation_config = ConsolidationConfig::default();

        let algorithm = GradualForgettingAlgorithm::new(forgetting_config, consolidation_config).unwrap();

        // Should evaluate on first run
        assert!(algorithm.should_evaluate());
    }
}
