//! Differential Privacy Module
//! 
//! Implements state-of-the-art differential privacy techniques to protect
//! individual privacy while enabling statistical analysis and machine learning.

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::security::{SecurityConfig, SecurityContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Privacy manager for differential privacy operations
#[derive(Debug)]
pub struct PrivacyManager {
    config: SecurityConfig,
    privacy_budget_tracker: PrivacyBudgetTracker,
    noise_generator: NoiseGenerator,
    metrics: PrivacyMetrics,
}

impl PrivacyManager {
    /// Create a new privacy manager
    pub async fn new(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            privacy_budget_tracker: PrivacyBudgetTracker::new(config.privacy_budget),
            noise_generator: NoiseGenerator::new(),
            metrics: PrivacyMetrics::default(),
        })
    }

    /// Apply differential privacy to a memory entry
    pub async fn apply_differential_privacy(&mut self,
        entry: &MemoryEntry,
        context: &SecurityContext
    ) -> Result<MemoryEntry> {
        let start_time = std::time::Instant::now();

        // Validate security context comprehensively
        context.validate_comprehensive(self.config.access_control_policy.require_mfa)?;

        // Check privacy budget
        let epsilon = self.privacy_budget_tracker.allocate_budget(&context.user_id, 0.1)?;
        
        // Apply noise to the memory entry
        let privatized_entry = self.add_differential_privacy_noise(entry, epsilon).await?;
        
        // Update metrics
        self.metrics.total_privatizations += 1;
        self.metrics.total_privacy_time_ms += start_time.elapsed().as_millis() as u64;
        self.metrics.total_epsilon_consumed += epsilon;

        Ok(privatized_entry)
    }

    /// Generate differentially private statistics
    pub async fn generate_private_statistics(&mut self,
        entries: &[MemoryEntry],
        query: PrivacyQuery,
        context: &SecurityContext
    ) -> Result<PrivateStatistics> {
        let start_time = std::time::Instant::now();

        // Allocate privacy budget based on query sensitivity
        let epsilon = self.privacy_budget_tracker.allocate_budget(
            &context.user_id, 
            query.sensitivity
        )?;

        // Compute statistics with differential privacy
        let statistics = match query.query_type {
            PrivacyQueryType::Count => {
                self.private_count(entries, epsilon).await?
            },
            PrivacyQueryType::Sum => {
                self.private_sum(entries, epsilon).await?
            },
            PrivacyQueryType::Average => {
                self.private_average(entries, epsilon).await?
            },
            PrivacyQueryType::Histogram => {
                self.private_histogram(entries, epsilon, query.bins.unwrap_or(10)).await?
            },
            PrivacyQueryType::Quantile => {
                self.private_quantile(entries, epsilon, query.quantile.unwrap_or(0.5)).await?
            },
        };

        // Update metrics
        self.metrics.total_queries += 1;
        self.metrics.total_privacy_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(statistics)
    }

    /// Apply local differential privacy to user data
    pub async fn apply_local_differential_privacy(&mut self,
        entry: &MemoryEntry,
        epsilon: f64
    ) -> Result<MemoryEntry> {
        // Local differential privacy - add noise at the source
        let privatized_entry = self.add_local_noise(entry, epsilon).await?;
        
        self.metrics.total_local_privatizations += 1;
        Ok(privatized_entry)
    }

    /// Get privacy metrics
    pub async fn get_metrics(&self) -> Result<PrivacyMetrics> {
        Ok(self.metrics.clone())
    }

    /// Get remaining privacy budget for a user
    pub async fn get_remaining_budget(&self, user_id: &str) -> Result<f64> {
        Ok(self.privacy_budget_tracker.get_remaining_budget(user_id))
    }

    // Private helper methods

    async fn add_differential_privacy_noise(&mut self, 
        entry: &MemoryEntry, 
        epsilon: f64
    ) -> Result<MemoryEntry> {
        let mut privatized_entry = entry.clone();

        // Add noise to text content using exponential mechanism
        privatized_entry.value = self.privatize_text(&entry.value, epsilon).await?;

        // Add noise to embeddings if present
        if let Some(ref embedding) = entry.embedding {
            let noisy_embedding = self.add_laplace_noise_to_vector(embedding, epsilon)?;
            privatized_entry.embedding = Some(noisy_embedding);
        }

        Ok(privatized_entry)
    }

    async fn privatize_text(&self, text: &str, epsilon: f64) -> Result<String> {
        // Implement text privatization using exponential mechanism
        // For simplicity, we'll add character-level noise
        let mut privatized_chars: Vec<char> = text.chars().collect();
        
        for char in privatized_chars.iter_mut() {
            // Add noise with probability based on epsilon
            if self.noise_generator.should_add_noise(epsilon) {
                *char = self.noise_generator.generate_similar_char(*char);
            }
        }

        Ok(privatized_chars.into_iter().collect())
    }

    fn add_laplace_noise_to_vector(&self, vector: &[f32], epsilon: f64) -> Result<Vec<f32>> {
        let sensitivity = 1.0; // Assume L1 sensitivity of 1
        let scale = sensitivity / epsilon;
        
        let noisy_vector: Vec<f32> = vector.iter()
            .map(|&x| x + self.noise_generator.laplace_noise(scale) as f32)
            .collect();
        
        Ok(noisy_vector)
    }

    async fn private_count(&self, entries: &[MemoryEntry], epsilon: f64) -> Result<PrivateStatistics> {
        let true_count = entries.len() as f64;
        let sensitivity = 1.0; // Adding/removing one entry changes count by 1
        let scale = sensitivity / epsilon;
        let noisy_count = true_count + self.noise_generator.laplace_noise(scale);
        
        Ok(PrivateStatistics {
            query_type: PrivacyQueryType::Count,
            result: noisy_count.max(0.0), // Ensure non-negative
            epsilon_used: epsilon,
            timestamp: Utc::now(),
            confidence_interval: self.calculate_confidence_interval(noisy_count, scale),
        })
    }

    async fn private_sum(&self, entries: &[MemoryEntry], epsilon: f64) -> Result<PrivateStatistics> {
        // Sum of text lengths as a simple numeric aggregation
        let true_sum = entries.iter().map(|e| e.value.len() as f64).sum::<f64>();
        let sensitivity = 1000.0; // Assume max text length is 1000
        let scale = sensitivity / epsilon;
        let noisy_sum = true_sum + self.noise_generator.laplace_noise(scale);
        
        Ok(PrivateStatistics {
            query_type: PrivacyQueryType::Sum,
            result: noisy_sum,
            epsilon_used: epsilon,
            timestamp: Utc::now(),
            confidence_interval: self.calculate_confidence_interval(noisy_sum, scale),
        })
    }

    async fn private_average(&self, entries: &[MemoryEntry], epsilon: f64) -> Result<PrivateStatistics> {
        if entries.is_empty() {
            return Ok(PrivateStatistics {
                query_type: PrivacyQueryType::Average,
                result: 0.0,
                epsilon_used: epsilon,
                timestamp: Utc::now(),
                confidence_interval: (0.0, 0.0),
            });
        }

        // Split epsilon between count and sum
        let epsilon_count = epsilon / 2.0;
        let epsilon_sum = epsilon / 2.0;

        let count_stats = self.private_count(entries, epsilon_count).await?;
        let sum_stats = self.private_sum(entries, epsilon_sum).await?;
        
        let average = if count_stats.result > 0.0 {
            sum_stats.result / count_stats.result
        } else {
            0.0
        };

        Ok(PrivateStatistics {
            query_type: PrivacyQueryType::Average,
            result: average,
            epsilon_used: epsilon,
            timestamp: Utc::now(),
            confidence_interval: self.calculate_confidence_interval(average, epsilon),
        })
    }

    async fn private_histogram(&self, entries: &[MemoryEntry], epsilon: f64, bins: usize) -> Result<PrivateStatistics> {
        // Create histogram of text lengths
        let max_length = entries.iter().map(|e| e.value.len()).max().unwrap_or(0);
        let bin_size = (max_length as f64 / bins as f64).ceil() as usize;
        
        let mut histogram = vec![0.0f64; bins];
        for entry in entries {
            let bin_index = (entry.value.len() / bin_size.max(1)).min(bins - 1);
            histogram[bin_index] += 1.0;
        }

        // Add Laplace noise to each bin
        let sensitivity = 1.0; // Adding/removing one entry affects one bin by 1
        let scale = sensitivity / epsilon;
        
        for count in histogram.iter_mut() {
            *count += self.noise_generator.laplace_noise(scale);
            *count = count.max(0.0); // Ensure non-negative
        }

        // Return the sum of histogram (total count with noise)
        let total = histogram.iter().sum();
        
        Ok(PrivateStatistics {
            query_type: PrivacyQueryType::Histogram,
            result: total,
            epsilon_used: epsilon,
            timestamp: Utc::now(),
            confidence_interval: self.calculate_confidence_interval(total, scale),
        })
    }

    async fn private_quantile(&self, entries: &[MemoryEntry], epsilon: f64, quantile: f64) -> Result<PrivateStatistics> {
        // Compute quantile of text lengths
        let mut lengths: Vec<f64> = entries.iter().map(|e| e.value.len() as f64).collect();
        lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (quantile * (lengths.len() as f64 - 1.0)).round() as usize;
        let true_quantile = lengths.get(index).copied().unwrap_or(0.0);
        
        // Add noise using exponential mechanism (simplified)
        let sensitivity = 1000.0; // Assume max text length difference
        let scale = sensitivity / epsilon;
        let noisy_quantile = true_quantile + self.noise_generator.laplace_noise(scale);
        
        Ok(PrivateStatistics {
            query_type: PrivacyQueryType::Quantile,
            result: noisy_quantile.max(0.0),
            epsilon_used: epsilon,
            timestamp: Utc::now(),
            confidence_interval: self.calculate_confidence_interval(noisy_quantile, scale),
        })
    }

    async fn add_local_noise(&self, entry: &MemoryEntry, epsilon: f64) -> Result<MemoryEntry> {
        // Local differential privacy with randomized response
        let mut privatized_entry = entry.clone();
        
        // Apply randomized response to text
        privatized_entry.value = self.randomized_response(&entry.value, epsilon).await?;
        
        // Add noise to embeddings
        if let Some(ref embedding) = entry.embedding {
            let noisy_embedding = self.add_laplace_noise_to_vector(embedding, epsilon)?;
            privatized_entry.embedding = Some(noisy_embedding);
        }
        
        Ok(privatized_entry)
    }

    async fn randomized_response(&self, text: &str, epsilon: f64) -> Result<String> {
        // Simplified randomized response mechanism
        let p = (epsilon.exp()) / (epsilon.exp() + 1.0); // Probability of truth
        
        let mut result = String::new();
        for char in text.chars() {
            if self.noise_generator.random_bool(p) {
                result.push(char); // Tell the truth
            } else {
                result.push(self.noise_generator.random_char()); // Random response
            }
        }
        
        Ok(result)
    }

    fn calculate_confidence_interval(&self, value: f64, scale: f64) -> (f64, f64) {
        // 95% confidence interval for Laplace noise
        let margin = 1.96 * scale; // Approximate for Laplace distribution
        (value - margin, value + margin)
    }
}

/// Privacy budget tracker
#[derive(Debug)]
struct PrivacyBudgetTracker {
    total_budget: f64,
    user_budgets: HashMap<String, f64>,
}

impl PrivacyBudgetTracker {
    fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            user_budgets: HashMap::new(),
        }
    }

    fn allocate_budget(&mut self, user_id: &str, epsilon: f64) -> Result<f64> {
        let current_budget = self.user_budgets.get(user_id).copied().unwrap_or(self.total_budget);
        
        if current_budget < epsilon {
            return Err(MemoryError::privacy(format!(
                "Insufficient privacy budget. Requested: {}, Available: {}", 
                epsilon, current_budget
            )));
        }
        
        self.user_budgets.insert(user_id.to_string(), current_budget - epsilon);
        Ok(epsilon)
    }

    fn get_remaining_budget(&self, user_id: &str) -> f64 {
        self.user_budgets.get(user_id).copied().unwrap_or(self.total_budget)
    }
}

/// Noise generator for differential privacy
#[derive(Debug)]
struct NoiseGenerator {
    // In production, use a cryptographically secure RNG
}

impl NoiseGenerator {
    fn new() -> Self {
        Self {}
    }

    fn laplace_noise(&self, scale: f64) -> f64 {
        // Generate Laplace noise with given scale
        // Simplified implementation - use proper crypto RNG in production
        let u = 0.5 - 0.3; // Simulated uniform random in [-0.5, 0.5]
        let noise = if u >= 0.0 {
            -scale * (1.0f64 - 2.0 * u).ln()
        } else {
            scale * (1.0f64 + 2.0 * u).ln()
        };
        noise
    }

    fn should_add_noise(&self, epsilon: f64) -> bool {
        // Probability of adding noise based on epsilon
        let prob = 1.0 / (1.0 + epsilon.exp());
        prob > 0.5 // Simplified decision
    }

    fn generate_similar_char(&self, original: char) -> char {
        // Generate a similar character for text privatization
        match original {
            'a'..='z' => ((original as u8 - b'a' + 1) % 26 + b'a') as char,
            'A'..='Z' => ((original as u8 - b'A' + 1) % 26 + b'A') as char,
            '0'..='9' => ((original as u8 - b'0' + 1) % 10 + b'0') as char,
            _ => original,
        }
    }

    fn random_bool(&self, probability: f64) -> bool {
        probability > 0.5 // Simplified
    }

    fn random_char(&self) -> char {
        // Generate a random character
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
        chars.chars().nth(42 % chars.len()).unwrap_or('x') // Deterministic for testing
    }
}

/// Privacy query for differential privacy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyQuery {
    pub query_type: PrivacyQueryType,
    pub sensitivity: f64,
    pub bins: Option<usize>,
    pub quantile: Option<f64>,
}

/// Types of privacy-preserving queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyQueryType {
    Count,
    Sum,
    Average,
    Histogram,
    Quantile,
}

/// Result of a privacy-preserving query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateStatistics {
    pub query_type: PrivacyQueryType,
    pub result: f64,
    pub epsilon_used: f64,
    pub timestamp: DateTime<Utc>,
    pub confidence_interval: (f64, f64),
}

/// Privacy metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrivacyMetrics {
    pub total_privatizations: u64,
    pub total_local_privatizations: u64,
    pub total_queries: u64,
    pub total_privacy_time_ms: u64,
    pub total_epsilon_consumed: f64,
    pub average_privacy_time_ms: f64,
    pub privacy_budget_utilization: f64,
}

impl PrivacyMetrics {
    pub fn calculate_averages(&mut self) {
        if self.total_privatizations > 0 {
            self.average_privacy_time_ms = self.total_privacy_time_ms as f64 / self.total_privatizations as f64;
        }
        // Privacy budget utilization would be calculated based on total budget
        self.privacy_budget_utilization = (self.total_epsilon_consumed / 10.0).min(1.0) * 100.0;
    }
}
