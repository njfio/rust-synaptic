//! Temporal Decay Models
//! 
//! Comprehensive implementation of various temporal decay models for memory systems,
//! including Ebbinghaus forgetting curves, power law decay, logarithmic decay,
//! and adaptive decay parameters based on psychological and neuroscience research.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of temporal decay models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecayModelType {
    /// Ebbinghaus exponential forgetting curve
    Ebbinghaus,
    /// Power law decay (Zipf-like distribution)
    PowerLaw,
    /// Logarithmic decay
    Logarithmic,
    /// Gaussian decay
    Gaussian,
    /// Hyperbolic decay
    Hyperbolic,
    /// Weibull decay (flexible shape parameter)
    Weibull,
    /// Hybrid model combining multiple decay functions
    Hybrid(Vec<DecayModelType>),
    /// Custom decay with user-defined parameters
    Custom { 
        name: String,
        parameters: HashMap<String, f64> 
    },
}

/// Configuration for temporal decay models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Default decay model type
    pub default_model: DecayModelType,
    /// Base half-life in hours
    pub base_half_life_hours: f64,
    /// Minimum decay rate (prevents complete forgetting)
    pub min_decay_rate: f64,
    /// Maximum decay rate (prevents instant forgetting)
    pub max_decay_rate: f64,
    /// Adaptive parameters enabled
    pub adaptive_enabled: bool,
    /// Context-aware decay adjustments
    pub context_aware: bool,
    /// Importance-based decay modulation
    pub importance_modulation: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_model: DecayModelType::Ebbinghaus,
            base_half_life_hours: 24.0,
            min_decay_rate: 0.001,
            max_decay_rate: 0.999,
            adaptive_enabled: true,
            context_aware: true,
            importance_modulation: true,
        }
    }
}

/// Parameters for specific decay models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayParameters {
    /// Decay rate constant
    pub decay_rate: f64,
    /// Shape parameter (for Weibull, power law, etc.)
    pub shape_parameter: f64,
    /// Scale parameter
    pub scale_parameter: f64,
    /// Offset parameter
    pub offset: f64,
    /// Adaptive adjustment factor
    pub adaptive_factor: f64,
}

impl Default for DecayParameters {
    fn default() -> Self {
        Self {
            decay_rate: 0.693, // ln(2) for half-life calculations
            shape_parameter: 1.0,
            scale_parameter: 1.0,
            offset: 0.0,
            adaptive_factor: 1.0,
        }
    }
}

/// Context information for adaptive decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayContext {
    /// Memory importance score (0.0 to 1.0)
    pub importance: f64,
    /// Access frequency (accesses per day)
    pub access_frequency: f64,
    /// Recency of last access (hours ago)
    pub hours_since_access: f64,
    /// Content complexity score
    pub complexity: f64,
    /// Emotional significance
    pub emotional_weight: f64,
    /// Contextual relevance
    pub contextual_relevance: f64,
    /// User engagement level
    pub engagement_level: f64,
}

/// Result of decay calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayResult {
    /// Retention probability (0.0 to 1.0)
    pub retention_probability: f64,
    /// Forgetting probability (1.0 - retention)
    pub forgetting_probability: f64,
    /// Decay model used
    pub model_used: DecayModelType,
    /// Effective half-life in hours
    pub effective_half_life: f64,
    /// Confidence in the calculation
    pub confidence: f64,
    /// Adaptive adjustments applied
    pub adaptive_adjustments: HashMap<String, f64>,
}

/// Comprehensive temporal decay models manager
#[derive(Debug, Clone)]
pub struct TemporalDecayModels {
    /// Configuration
    config: DecayConfig,
    /// Model-specific parameters
    model_parameters: HashMap<String, DecayParameters>,
    /// Performance tracking for adaptive models
    model_performance: HashMap<String, Vec<f64>>,
    /// Context history for learning
    context_history: Vec<(DecayContext, DecayResult)>,
}

impl TemporalDecayModels {
    /// Create new temporal decay models manager
    pub fn new(config: DecayConfig) -> Result<Self> {
        let mut model_parameters = HashMap::new();
        
        // Initialize default parameters for each model type
        model_parameters.insert("ebbinghaus".to_string(), DecayParameters::default());
        model_parameters.insert("power_law".to_string(), DecayParameters {
            shape_parameter: 0.5, // Typical power law exponent
            ..DecayParameters::default()
        });
        model_parameters.insert("logarithmic".to_string(), DecayParameters {
            scale_parameter: 24.0, // Scale to hours
            ..DecayParameters::default()
        });
        model_parameters.insert("gaussian".to_string(), DecayParameters {
            scale_parameter: 48.0, // Standard deviation in hours
            ..DecayParameters::default()
        });
        model_parameters.insert("hyperbolic".to_string(), DecayParameters {
            shape_parameter: 2.0, // Hyperbolic exponent
            ..DecayParameters::default()
        });
        model_parameters.insert("weibull".to_string(), DecayParameters {
            shape_parameter: 1.5, // Weibull shape parameter
            scale_parameter: 72.0, // Weibull scale parameter
            ..DecayParameters::default()
        });

        Ok(Self {
            config,
            model_parameters,
            model_performance: HashMap::new(),
            context_history: Vec::new(),
        })
    }

    /// Calculate temporal decay using specified model
    pub fn calculate_decay<'a>(
        &'a mut self,
        model_type: &'a DecayModelType,
        time_elapsed_hours: f64,
        context: &'a DecayContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DecayResult>> + 'a>> {
        Box::pin(async move {
        let base_result = match model_type {
            DecayModelType::Ebbinghaus => {
                self.calculate_ebbinghaus_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::PowerLaw => {
                self.calculate_power_law_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::Logarithmic => {
                self.calculate_logarithmic_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::Gaussian => {
                self.calculate_gaussian_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::Hyperbolic => {
                self.calculate_hyperbolic_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::Weibull => {
                self.calculate_weibull_decay(time_elapsed_hours, context).await?
            },
            DecayModelType::Hybrid(models) => {
                self.calculate_hybrid_decay(models, time_elapsed_hours, context).await?
            },
            DecayModelType::Custom { name, parameters } => {
                self.calculate_custom_decay(name, parameters, time_elapsed_hours, context).await?
            },
        };

        // Apply adaptive adjustments if enabled
        let final_result = if self.config.adaptive_enabled {
            self.apply_adaptive_adjustments(base_result, context).await?
        } else {
            base_result
        };

        // Store context and result for learning
        if self.context_history.len() >= 1000 {
            self.context_history.remove(0); // Keep history bounded
        }
        self.context_history.push((context.clone(), final_result.clone()));

        Ok(final_result)
        })
    }

    /// Calculate decay for a memory entry
    pub async fn calculate_memory_decay(
        &mut self,
        memory: &MemoryEntry,
        model_type: Option<&DecayModelType>,
    ) -> Result<DecayResult> {
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours().max(0) as f64;
        
        let context = DecayContext {
            importance: memory.metadata.importance,
            access_frequency: self.calculate_access_frequency(memory),
            hours_since_access,
            complexity: self.estimate_content_complexity(memory),
            emotional_weight: self.estimate_emotional_weight(memory),
            contextual_relevance: self.estimate_contextual_relevance(memory),
            engagement_level: self.estimate_engagement_level(memory),
        };

        let model = model_type.cloned().unwrap_or_else(|| self.config.default_model.clone());
        self.calculate_decay(&model, hours_since_access, &context).await
    }

    /// Get optimal decay model for given context
    pub async fn get_optimal_model(&self, context: &DecayContext) -> Result<DecayModelType> {
        if !self.config.adaptive_enabled {
            return Ok(self.config.default_model.clone());
        }

        // Analyze context to determine best model
        if context.importance > 0.8 && context.access_frequency > 1.0 {
            // High importance, frequent access: use power law (slower decay)
            Ok(DecayModelType::PowerLaw)
        } else if context.hours_since_access < 24.0 {
            // Recent access: use Ebbinghaus (classic forgetting curve)
            Ok(DecayModelType::Ebbinghaus)
        } else if context.emotional_weight > 0.7 {
            // High emotional content: use Weibull (flexible shape)
            Ok(DecayModelType::Weibull)
        } else if context.complexity > 0.6 {
            // Complex content: use logarithmic (gradual decay)
            Ok(DecayModelType::Logarithmic)
        } else {
            // Default case: use hybrid approach
            Ok(DecayModelType::Hybrid(vec![
                DecayModelType::Ebbinghaus,
                DecayModelType::PowerLaw,
            ]))
        }
    }

    /// Calculate Ebbinghaus exponential decay
    async fn calculate_ebbinghaus_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let _params = self.model_parameters.get("ebbinghaus").unwrap();

        // Adaptive half-life based on context
        let base_half_life = self.config.base_half_life_hours;
        let adaptive_half_life = base_half_life *
            (1.0 + context.importance * 2.0) * // Importance extends half-life
            (1.0 + context.access_frequency * 0.5) * // Frequency extends half-life
            (1.0 + context.emotional_weight * 1.5); // Emotion extends half-life

        // Ebbinghaus formula: R(t) = e^(-t/S) where S is memory strength
        let decay_constant = (2.0_f64).ln() / adaptive_half_life;
        let retention_probability = (-decay_constant * time_elapsed_hours).exp();

        // Apply bounds
        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("half_life_adjustment".to_string(), adaptive_half_life / base_half_life);
        adaptive_adjustments.insert("importance_factor".to_string(), context.importance);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Ebbinghaus,
            effective_half_life: adaptive_half_life,
            confidence: 0.9, // High confidence for well-established model
            adaptive_adjustments,
        })
    }

    /// Calculate power law decay
    async fn calculate_power_law_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let params = self.model_parameters.get("power_law").unwrap();

        // Power law formula: R(t) = (1 + t/τ)^(-β)
        let time_scale = self.config.base_half_life_hours;
        let beta = params.shape_parameter * (1.0 + context.importance * 0.5); // Importance reduces decay rate

        let retention_probability = (1.0 + time_elapsed_hours / time_scale).powf(-beta);

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("beta_adjustment".to_string(), beta / params.shape_parameter);
        adaptive_adjustments.insert("time_scale".to_string(), time_scale);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::PowerLaw,
            effective_half_life: time_scale * (2.0_f64.powf(1.0 / beta) - 1.0),
            confidence: 0.85,
            adaptive_adjustments,
        })
    }

    /// Calculate logarithmic decay
    async fn calculate_logarithmic_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let params = self.model_parameters.get("logarithmic").unwrap();

        // Logarithmic formula: R(t) = 1 - log(1 + t/τ) / log(1 + T_max/τ)
        let time_scale = params.scale_parameter * (1.0 + context.complexity * 0.5);
        let max_time = time_scale * 100.0; // Maximum time for normalization

        let retention_probability = if time_elapsed_hours <= 0.0 {
            1.0
        } else {
            1.0 - (1.0 + time_elapsed_hours / time_scale).ln() / (1.0 + max_time / time_scale).ln()
        };

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("time_scale_adjustment".to_string(), time_scale / params.scale_parameter);
        adaptive_adjustments.insert("complexity_factor".to_string(), context.complexity);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Logarithmic,
            effective_half_life: time_scale * ((2.0_f64).exp() - 1.0),
            confidence: 0.8,
            adaptive_adjustments,
        })
    }

    /// Calculate Gaussian decay
    async fn calculate_gaussian_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let params = self.model_parameters.get("gaussian").unwrap();

        // Gaussian formula: R(t) = e^(-(t/σ)²/2)
        let sigma = params.scale_parameter * (1.0 + context.engagement_level * 0.8);

        let retention_probability = (-(time_elapsed_hours / sigma).powi(2) / 2.0).exp();

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("sigma_adjustment".to_string(), sigma / params.scale_parameter);
        adaptive_adjustments.insert("engagement_factor".to_string(), context.engagement_level);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Gaussian,
            effective_half_life: sigma * (2.0 * (2.0_f64).ln()).sqrt(),
            confidence: 0.75,
            adaptive_adjustments,
        })
    }

    /// Calculate hyperbolic decay
    async fn calculate_hyperbolic_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let params = self.model_parameters.get("hyperbolic").unwrap();

        // Hyperbolic formula: R(t) = 1 / (1 + (t/τ)^α)
        let time_scale = self.config.base_half_life_hours * (1.0 + context.contextual_relevance);
        let alpha = params.shape_parameter;

        let retention_probability = 1.0 / (1.0 + (time_elapsed_hours / time_scale).powf(alpha));

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("time_scale_adjustment".to_string(), time_scale / self.config.base_half_life_hours);
        adaptive_adjustments.insert("relevance_factor".to_string(), context.contextual_relevance);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Hyperbolic,
            effective_half_life: time_scale * (2.0_f64.powf(1.0 / alpha) - 1.0),
            confidence: 0.7,
            adaptive_adjustments,
        })
    }

    /// Calculate Weibull decay
    async fn calculate_weibull_decay(
        &self,
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        let params = self.model_parameters.get("weibull").unwrap();

        // Weibull formula: R(t) = e^(-(t/λ)^k)
        let lambda = params.scale_parameter * (1.0 + context.importance * 2.0);
        let k = params.shape_parameter * (1.0 + context.emotional_weight * 0.5);

        let retention_probability = (-(time_elapsed_hours / lambda).powf(k)).exp();

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("lambda_adjustment".to_string(), lambda / params.scale_parameter);
        adaptive_adjustments.insert("k_adjustment".to_string(), k / params.shape_parameter);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Weibull,
            effective_half_life: lambda * ((2.0_f64).ln()).powf(1.0 / k),
            confidence: 0.8,
            adaptive_adjustments,
        })
    }

    /// Calculate hybrid decay combining multiple models
    async fn calculate_hybrid_decay(
        &self,
        models: &[DecayModelType],
        time_elapsed_hours: f64,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        if models.is_empty() {
            return self.calculate_ebbinghaus_decay(time_elapsed_hours, context).await;
        }

        let mut results = Vec::new();
        let mut total_confidence = 0.0;

        // Calculate decay for each model
        for model in models {
            if let DecayModelType::Hybrid(_) = model {
                continue; // Avoid infinite recursion
            }

            let result = match model {
                DecayModelType::Ebbinghaus => {
                    self.calculate_ebbinghaus_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::PowerLaw => {
                    self.calculate_power_law_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::Logarithmic => {
                    self.calculate_logarithmic_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::Gaussian => {
                    self.calculate_gaussian_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::Hyperbolic => {
                    self.calculate_hyperbolic_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::Weibull => {
                    self.calculate_weibull_decay(time_elapsed_hours, context).await?
                },
                DecayModelType::Custom { name, parameters } => {
                    self.calculate_custom_decay(name, parameters, time_elapsed_hours, context).await?
                },
                DecayModelType::Hybrid(_) => continue, // Already handled above
            };
            total_confidence += result.confidence;
            results.push(result);
        }

        if results.is_empty() {
            return self.calculate_ebbinghaus_decay(time_elapsed_hours, context).await;
        }

        // Weighted combination based on confidence
        let mut weighted_retention = 0.0;
        let mut combined_adjustments = HashMap::new();
        let mut effective_half_life = 0.0;

        for result in &results {
            let weight = result.confidence / total_confidence;
            weighted_retention += result.retention_probability * weight;
            effective_half_life += result.effective_half_life * weight;

            // Combine adjustments
            for (key, value) in &result.adaptive_adjustments {
                let entry = combined_adjustments.entry(key.clone()).or_insert(0.0);
                *entry += value * weight;
            }
        }

        Ok(DecayResult {
            retention_probability: weighted_retention,
            forgetting_probability: 1.0 - weighted_retention,
            model_used: DecayModelType::Hybrid(models.to_vec()),
            effective_half_life,
            confidence: total_confidence / results.len() as f64,
            adaptive_adjustments: combined_adjustments,
        })
    }

    /// Calculate custom decay model
    async fn calculate_custom_decay(
        &self,
        name: &str,
        parameters: &HashMap<String, f64>,
        time_elapsed_hours: f64,
        _context: &DecayContext,
    ) -> Result<DecayResult> {
        // Extract parameters with defaults
        let decay_rate = parameters.get("decay_rate").copied().unwrap_or(0.693);
        let shape = parameters.get("shape").copied().unwrap_or(1.0);
        let scale = parameters.get("scale").copied().unwrap_or(24.0);
        let offset = parameters.get("offset").copied().unwrap_or(0.0);

        // Generic custom formula: R(t) = e^(-((t + offset)/scale)^shape * decay_rate)
        let normalized_time = (time_elapsed_hours + offset) / scale;
        let retention_probability = (-(normalized_time.powf(shape) * decay_rate)).exp();

        let bounded_retention = retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        let mut adaptive_adjustments = HashMap::new();
        adaptive_adjustments.insert("custom_decay_rate".to_string(), decay_rate);
        adaptive_adjustments.insert("custom_shape".to_string(), shape);
        adaptive_adjustments.insert("custom_scale".to_string(), scale);

        Ok(DecayResult {
            retention_probability: bounded_retention,
            forgetting_probability: 1.0 - bounded_retention,
            model_used: DecayModelType::Custom {
                name: name.to_string(),
                parameters: parameters.clone()
            },
            effective_half_life: scale * ((2.0_f64).ln() / decay_rate).powf(1.0 / shape),
            confidence: 0.6, // Lower confidence for custom models
            adaptive_adjustments,
        })
    }

    /// Apply adaptive adjustments to decay result
    async fn apply_adaptive_adjustments(
        &mut self,
        mut result: DecayResult,
        context: &DecayContext,
    ) -> Result<DecayResult> {
        if !self.config.adaptive_enabled {
            return Ok(result);
        }

        // Context-aware adjustments
        if self.config.context_aware {
            // High engagement reduces forgetting
            if context.engagement_level > 0.7 {
                result.retention_probability *= 1.0 + (context.engagement_level - 0.7) * 0.5;
                result.adaptive_adjustments.insert("engagement_boost".to_string(), context.engagement_level);
            }

            // High access frequency reduces forgetting
            if context.access_frequency > 1.0 {
                let frequency_factor = 1.0 + (context.access_frequency - 1.0).ln() * 0.2;
                result.retention_probability *= frequency_factor;
                result.adaptive_adjustments.insert("frequency_boost".to_string(), frequency_factor);
            }

            // Recent access provides protection
            if context.hours_since_access < 6.0 {
                let recency_factor = 1.0 + (6.0 - context.hours_since_access) / 6.0 * 0.3;
                result.retention_probability *= recency_factor;
                result.adaptive_adjustments.insert("recency_protection".to_string(), recency_factor);
            }
        }

        // Importance-based modulation
        if self.config.importance_modulation && context.importance > 0.5 {
            let importance_factor = 1.0 + (context.importance - 0.5) * 1.0;
            result.retention_probability *= importance_factor;
            result.adaptive_adjustments.insert("importance_protection".to_string(), importance_factor);
        }

        // Emotional weight protection
        if context.emotional_weight > 0.6 {
            let emotional_factor = 1.0 + (context.emotional_weight - 0.6) * 0.8;
            result.retention_probability *= emotional_factor;
            result.adaptive_adjustments.insert("emotional_protection".to_string(), emotional_factor);
        }

        // Apply bounds after adjustments
        result.retention_probability = result.retention_probability
            .max(self.config.min_decay_rate)
            .min(1.0 - self.config.min_decay_rate);

        result.forgetting_probability = 1.0 - result.retention_probability;

        Ok(result)
    }

    /// Calculate access frequency for a memory
    fn calculate_access_frequency(&self, memory: &MemoryEntry) -> f64 {
        let days_since_creation = (Utc::now() - memory.created_at()).num_days().max(1) as f64;
        memory.access_count() as f64 / days_since_creation
    }

    /// Estimate content complexity
    fn estimate_content_complexity(&self, memory: &MemoryEntry) -> f64 {
        let content = &memory.value;
        let word_count = content.split_whitespace().count();
        let unique_words = content.split_whitespace()
            .collect::<std::collections::HashSet<_>>()
            .len();

        let vocabulary_richness = if word_count > 0 {
            unique_words as f64 / word_count as f64
        } else {
            0.0
        };

        // Combine length and vocabulary richness
        let length_factor = (content.len() as f64 / 1000.0).min(1.0);
        let complexity = (vocabulary_richness + length_factor) / 2.0;

        complexity.min(1.0)
    }

    /// Estimate emotional weight
    fn estimate_emotional_weight(&self, memory: &MemoryEntry) -> f64 {
        let content = &memory.value.to_lowercase();

        // Simple emotional keyword detection
        let emotional_keywords = [
            "love", "hate", "fear", "joy", "anger", "sad", "happy", "excited",
            "worried", "anxious", "proud", "disappointed", "grateful", "frustrated",
            "amazing", "terrible", "wonderful", "awful", "brilliant", "disaster"
        ];

        let emotional_count = emotional_keywords.iter()
            .filter(|&&keyword| content.contains(keyword))
            .count();

        let word_count = content.split_whitespace().count().max(1);
        let emotional_density = emotional_count as f64 / word_count as f64;

        (emotional_density * 10.0).min(1.0)
    }

    /// Estimate contextual relevance
    fn estimate_contextual_relevance(&self, memory: &MemoryEntry) -> f64 {
        // Simple heuristic based on tags and metadata
        let tag_relevance = if memory.metadata.tags.is_empty() {
            0.3 // Default relevance for untagged content
        } else {
            0.7 // Higher relevance for tagged content
        };

        let importance_relevance = memory.metadata.importance;

        (tag_relevance + importance_relevance) / 2.0
    }

    /// Estimate engagement level
    fn estimate_engagement_level(&self, memory: &MemoryEntry) -> f64 {
        let access_frequency = self.calculate_access_frequency(memory);
        let recency_factor = {
            let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours() as f64;
            if hours_since_access < 24.0 {
                1.0 - hours_since_access / 24.0
            } else {
                0.0
            }
        };

        let frequency_factor = (access_frequency / 5.0).min(1.0); // Normalize to daily access

        (recency_factor * 0.6 + frequency_factor * 0.4).min(1.0)
    }

    /// Update model parameters based on performance feedback
    pub async fn update_model_performance(
        &mut self,
        model_type: &DecayModelType,
        performance_score: f64,
    ) -> Result<()> {
        let model_key = self.get_model_key(model_type);

        let performances = self.model_performance.entry(model_key).or_insert_with(Vec::new);
        performances.push(performance_score);

        // Keep only recent performance data
        if performances.len() > 100 {
            performances.remove(0);
        }

        Ok(())
    }

    /// Get model key for performance tracking
    fn get_model_key(&self, model_type: &DecayModelType) -> String {
        match model_type {
            DecayModelType::Ebbinghaus => "ebbinghaus".to_string(),
            DecayModelType::PowerLaw => "power_law".to_string(),
            DecayModelType::Logarithmic => "logarithmic".to_string(),
            DecayModelType::Gaussian => "gaussian".to_string(),
            DecayModelType::Hyperbolic => "hyperbolic".to_string(),
            DecayModelType::Weibull => "weibull".to_string(),
            DecayModelType::Hybrid(models) => {
                format!("hybrid_{}", models.len())
            },
            DecayModelType::Custom { name, .. } => {
                format!("custom_{}", name)
            },
        }
    }

    /// Get performance statistics for a model
    pub fn get_model_performance(&self, model_type: &DecayModelType) -> Option<(f64, f64)> {
        let model_key = self.get_model_key(model_type);

        if let Some(performances) = self.model_performance.get(&model_key) {
            if !performances.is_empty() {
                let mean = performances.iter().sum::<f64>() / performances.len() as f64;
                let variance = performances.iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f64>() / performances.len() as f64;
                let std_dev = variance.sqrt();

                Some((mean, std_dev))
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    fn create_test_context() -> DecayContext {
        DecayContext {
            importance: 0.7,
            access_frequency: 2.0,
            hours_since_access: 12.0,
            complexity: 0.6,
            emotional_weight: 0.5,
            contextual_relevance: 0.8,
            engagement_level: 0.7,
        }
    }

    fn create_test_memory() -> MemoryEntry {
        let mut memory = MemoryEntry::new(
            "test_key".to_string(),
            "This is a test memory with emotional content like love and excitement".to_string(),
            MemoryType::LongTerm,
        );
        memory.metadata.importance = 0.8;
        memory.metadata.tags = vec!["important".to_string(), "test".to_string()];
        memory
    }

    #[tokio::test]
    async fn test_ebbinghaus_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_ebbinghaus_decay(24.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert!(result.forgetting_probability >= 0.0);
        assert!(result.forgetting_probability <= 1.0);
        assert!((result.retention_probability + result.forgetting_probability - 1.0).abs() < 0.001);
        assert_eq!(result.model_used, DecayModelType::Ebbinghaus);
        assert!(result.effective_half_life > 0.0);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_power_law_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_power_law_decay(48.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert_eq!(result.model_used, DecayModelType::PowerLaw);
        assert!(result.effective_half_life > 0.0);
    }

    #[tokio::test]
    async fn test_logarithmic_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_logarithmic_decay(72.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert_eq!(result.model_used, DecayModelType::Logarithmic);
    }

    #[tokio::test]
    async fn test_gaussian_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_gaussian_decay(36.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert_eq!(result.model_used, DecayModelType::Gaussian);
    }

    #[tokio::test]
    async fn test_hyperbolic_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_hyperbolic_decay(60.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert_eq!(result.model_used, DecayModelType::Hyperbolic);
    }

    #[tokio::test]
    async fn test_weibull_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let result = models.calculate_weibull_decay(84.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert_eq!(result.model_used, DecayModelType::Weibull);
    }

    #[tokio::test]
    async fn test_hybrid_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let hybrid_models = vec![DecayModelType::Ebbinghaus, DecayModelType::PowerLaw];
        let result = models.calculate_hybrid_decay(&hybrid_models, 48.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);

        if let DecayModelType::Hybrid(models) = result.model_used {
            assert_eq!(models.len(), 2);
        } else {
            panic!("Expected hybrid model type");
        }
    }

    #[tokio::test]
    async fn test_custom_decay() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        let mut parameters = HashMap::new();
        parameters.insert("decay_rate".to_string(), 0.5);
        parameters.insert("shape".to_string(), 1.2);
        parameters.insert("scale".to_string(), 48.0);

        let result = models.calculate_custom_decay("test_custom", &parameters, 24.0, &context).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);

        if let DecayModelType::Custom { name, .. } = result.model_used {
            assert_eq!(name, "test_custom");
        } else {
            panic!("Expected custom model type");
        }
    }

    #[tokio::test]
    async fn test_memory_decay_calculation() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();
        let memory = create_test_memory();

        let result = models.calculate_memory_decay(&memory, None).await.unwrap();

        assert!(result.retention_probability >= 0.0);
        assert!(result.retention_probability <= 1.0);
        assert!(result.effective_half_life > 0.0);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_optimal_model_selection() {
        let config = DecayConfig::default();
        let models = TemporalDecayModels::new(config).unwrap();

        // High importance, frequent access should suggest PowerLaw
        let high_importance_context = DecayContext {
            importance: 0.9,
            access_frequency: 2.0,
            hours_since_access: 48.0,
            complexity: 0.5,
            emotional_weight: 0.3,
            contextual_relevance: 0.6,
            engagement_level: 0.7,
        };

        let optimal_model = models.get_optimal_model(&high_importance_context).await.unwrap();
        assert_eq!(optimal_model, DecayModelType::PowerLaw);

        // Recent access should suggest Ebbinghaus
        let recent_context = DecayContext {
            importance: 0.5,
            access_frequency: 1.0,
            hours_since_access: 12.0,
            complexity: 0.5,
            emotional_weight: 0.3,
            contextual_relevance: 0.6,
            engagement_level: 0.5,
        };

        let optimal_model = models.get_optimal_model(&recent_context).await.unwrap();
        assert_eq!(optimal_model, DecayModelType::Ebbinghaus);
    }

    #[tokio::test]
    async fn test_adaptive_adjustments() {
        let mut config = DecayConfig::default();
        config.adaptive_enabled = true;
        config.context_aware = true;
        config.importance_modulation = true;

        let mut models = TemporalDecayModels::new(config).unwrap();

        let high_engagement_context = DecayContext {
            importance: 0.8,
            access_frequency: 3.0,
            hours_since_access: 2.0, // Very recent
            complexity: 0.7,
            emotional_weight: 0.8,
            contextual_relevance: 0.9,
            engagement_level: 0.9,
        };

        let base_result = DecayResult {
            retention_probability: 0.5,
            forgetting_probability: 0.5,
            model_used: DecayModelType::Ebbinghaus,
            effective_half_life: 24.0,
            confidence: 0.8,
            adaptive_adjustments: HashMap::new(),
        };

        let adjusted_result = models.apply_adaptive_adjustments(base_result, &high_engagement_context).await.unwrap();

        // Should have higher retention due to adjustments
        assert!(adjusted_result.retention_probability > 0.5);
        assert!(!adjusted_result.adaptive_adjustments.is_empty());
    }

    #[tokio::test]
    async fn test_content_complexity_estimation() {
        let config = DecayConfig::default();
        let models = TemporalDecayModels::new(config).unwrap();

        let simple_memory = MemoryEntry::new(
            "simple".to_string(),
            "short text".to_string(),
            MemoryType::ShortTerm,
        );

        let complex_memory = MemoryEntry::new(
            "complex".to_string(),
            "This is a sophisticated piece of content with diverse vocabulary, complex concepts, and high semantic density that demonstrates various linguistic patterns and technical terminology".to_string(),
            MemoryType::LongTerm,
        );

        let simple_complexity = models.estimate_content_complexity(&simple_memory);
        let complex_complexity = models.estimate_content_complexity(&complex_memory);

        assert!(complex_complexity > simple_complexity);
        assert!(simple_complexity >= 0.0 && simple_complexity <= 1.0);
        assert!(complex_complexity >= 0.0 && complex_complexity <= 1.0);
    }

    #[tokio::test]
    async fn test_emotional_weight_estimation() {
        let config = DecayConfig::default();
        let models = TemporalDecayModels::new(config).unwrap();

        let neutral_memory = MemoryEntry::new(
            "neutral".to_string(),
            "This is a neutral piece of information about technical specifications".to_string(),
            MemoryType::LongTerm,
        );

        let emotional_memory = MemoryEntry::new(
            "emotional".to_string(),
            "I love this amazing experience and feel so happy and excited about the wonderful results".to_string(),
            MemoryType::LongTerm,
        );

        let neutral_weight = models.estimate_emotional_weight(&neutral_memory);
        let emotional_weight = models.estimate_emotional_weight(&emotional_memory);

        assert!(emotional_weight > neutral_weight);
        assert!(neutral_weight >= 0.0 && neutral_weight <= 1.0);
        assert!(emotional_weight >= 0.0 && emotional_weight <= 1.0);
    }

    #[tokio::test]
    async fn test_model_performance_tracking() {
        let config = DecayConfig::default();
        let mut models = TemporalDecayModels::new(config).unwrap();

        let model_type = DecayModelType::Ebbinghaus;

        // Add some performance scores
        models.update_model_performance(&model_type, 0.8).await.unwrap();
        models.update_model_performance(&model_type, 0.7).await.unwrap();
        models.update_model_performance(&model_type, 0.9).await.unwrap();

        let (mean, std_dev) = models.get_model_performance(&model_type).unwrap();

        assert!((mean - 0.8).abs() < 0.1); // Should be around 0.8
        assert!(std_dev >= 0.0);
    }

    #[tokio::test]
    async fn test_decay_bounds() {
        let mut config = DecayConfig::default();
        config.min_decay_rate = 0.01;
        config.max_decay_rate = 0.99;

        let mut models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        // Test very long time (should hit minimum retention)
        let result = models.calculate_ebbinghaus_decay(10000.0, &context).await.unwrap();
        assert!(result.retention_probability >= 0.01);

        // Test zero time (should be close to maximum retention)
        let result = models.calculate_ebbinghaus_decay(0.0, &context).await.unwrap();
        assert!(result.retention_probability <= 0.99);
    }

    #[tokio::test]
    async fn test_time_progression_decay() {
        let config = DecayConfig::default();
        let models = TemporalDecayModels::new(config).unwrap();
        let context = create_test_context();

        // Test that retention decreases over time
        let result_1h = models.calculate_ebbinghaus_decay(1.0, &context).await.unwrap();
        let result_24h = models.calculate_ebbinghaus_decay(24.0, &context).await.unwrap();
        let result_168h = models.calculate_ebbinghaus_decay(168.0, &context).await.unwrap();

        assert!(result_1h.retention_probability > result_24h.retention_probability);
        assert!(result_24h.retention_probability > result_168h.retention_probability);
    }
}
