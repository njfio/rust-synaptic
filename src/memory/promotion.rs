//! Memory promotion policies for hierarchical memory management.
//!
//! This module provides traits and implementations for automatically promoting
//! memories from short-term to long-term storage based on various criteria such as
//! access frequency, age, importance scores, and hybrid combinations.

use crate::memory::types::{MemoryEntry, MemoryType};
use crate::error::Result;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use serde::{Deserialize, Serialize};

/// Configuration for memory promotion behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionConfig {
    /// Type of promotion policy to use
    pub policy_type: PromotionPolicyType,
    /// Access frequency threshold (for AccessFrequency policy)
    pub access_threshold: usize,
    /// Minimum age in days (for TimeBased policy)
    pub min_age_days: i64,
    /// Importance threshold (for Importance policy)
    pub importance_threshold: f64,
    /// Hybrid policy promotion threshold
    pub hybrid_promotion_threshold: f64,
    /// Weight for access frequency in hybrid policy
    pub hybrid_access_weight: f64,
    /// Weight for time-based in hybrid policy
    pub hybrid_time_weight: f64,
    /// Weight for importance in hybrid policy
    pub hybrid_importance_weight: f64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            policy_type: PromotionPolicyType::Hybrid,
            access_threshold: 5,
            min_age_days: 7,
            importance_threshold: 0.8,
            hybrid_promotion_threshold: 0.6,
            hybrid_access_weight: 0.4,
            hybrid_time_weight: 0.3,
            hybrid_importance_weight: 0.3,
        }
    }
}

/// Types of promotion policies available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionPolicyType {
    /// Promote based on access frequency
    AccessFrequency,
    /// Promote based on age
    TimeBased,
    /// Promote based on importance score
    Importance,
    /// Promote based on weighted combination of multiple factors
    Hybrid,
}

impl PromotionConfig {
    /// Create a promotion manager from this configuration.
    pub fn create_manager(&self) -> MemoryPromotionManager {
        let policy: Box<dyn MemoryPromotionPolicy> = match self.policy_type {
            PromotionPolicyType::AccessFrequency => {
                Box::new(AccessFrequencyPolicy::new(self.access_threshold))
            }
            PromotionPolicyType::TimeBased => {
                Box::new(TimeBasedPolicy::new(ChronoDuration::days(self.min_age_days)))
            }
            PromotionPolicyType::Importance => {
                Box::new(ImportancePolicy::new(self.importance_threshold))
            }
            PromotionPolicyType::Hybrid => {
                let mut hybrid = HybridPolicy::new(self.hybrid_promotion_threshold);
                hybrid.add_policy(
                    Box::new(AccessFrequencyPolicy::new(self.access_threshold)),
                    self.hybrid_access_weight,
                );
                hybrid.add_policy(
                    Box::new(TimeBasedPolicy::new(ChronoDuration::days(self.min_age_days))),
                    self.hybrid_time_weight,
                );
                hybrid.add_policy(
                    Box::new(ImportancePolicy::new(self.importance_threshold)),
                    self.hybrid_importance_weight,
                );
                Box::new(hybrid)
            }
        };

        MemoryPromotionManager::new(policy)
    }
}

/// Trait defining a policy for promoting memories from short-term to long-term.
///
/// Promotion policies evaluate memory entries and determine whether they should
/// be promoted to long-term storage based on various criteria.
///
/// # Examples
///
/// ```rust,no_run
/// use synaptic::memory::promotion::{MemoryPromotionPolicy, AccessFrequencyPolicy};
/// use synaptic::memory::types::MemoryEntry;
///
/// let policy = AccessFrequencyPolicy::new(5); // Promote after 5 accesses
/// let memory = MemoryEntry::new(/* ... */);
///
/// if policy.should_promote(&memory) {
///     // Promote to long-term
/// }
/// ```
pub trait MemoryPromotionPolicy: Send + Sync {
    /// Evaluate whether a memory should be promoted to long-term storage.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory entry to evaluate
    ///
    /// # Returns
    ///
    /// `true` if the memory should be promoted, `false` otherwise
    fn should_promote(&self, memory: &MemoryEntry) -> bool;

    /// Get a human-readable description of this policy.
    fn description(&self) -> String;

    /// Get the name of this policy.
    fn name(&self) -> &'static str;

    /// Calculate a promotion score for the memory (0.0 to 1.0).
    ///
    /// Higher scores indicate stronger candidates for promotion.
    /// A score >= 0.5 typically indicates the memory should be promoted.
    fn promotion_score(&self, memory: &MemoryEntry) -> f64;
}

/// Promotes memories based on access frequency.
///
/// Memories that have been accessed a certain number of times are considered
/// important enough to promote to long-term storage.
///
/// # Examples
///
/// ```rust,no_run
/// use synaptic::memory::promotion::AccessFrequencyPolicy;
///
/// // Promote memories after 10 accesses
/// let policy = AccessFrequencyPolicy::new(10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessFrequencyPolicy {
    /// Minimum number of accesses required for promotion
    pub access_threshold: usize,
}

impl AccessFrequencyPolicy {
    /// Create a new access frequency policy.
    ///
    /// # Arguments
    ///
    /// * `access_threshold` - Minimum number of accesses for promotion
    pub fn new(access_threshold: usize) -> Self {
        Self { access_threshold }
    }
}

impl MemoryPromotionPolicy for AccessFrequencyPolicy {
    fn should_promote(&self, memory: &MemoryEntry) -> bool {
        memory.memory_type == MemoryType::ShortTerm
            && memory.access_count >= self.access_threshold
    }

    fn description(&self) -> String {
        format!(
            "Promotes memories accessed {} or more times",
            self.access_threshold
        )
    }

    fn name(&self) -> &'static str {
        "AccessFrequency"
    }

    fn promotion_score(&self, memory: &MemoryEntry) -> f64 {
        if memory.memory_type != MemoryType::ShortTerm {
            return 0.0;
        }

        let ratio = memory.access_count as f64 / self.access_threshold as f64;
        ratio.min(1.0)
    }
}

/// Promotes memories based on age.
///
/// Memories that have existed for a certain duration and are still being
/// accessed are considered valuable enough for long-term storage.
///
/// # Examples
///
/// ```rust,no_run
/// use synaptic::memory::promotion::TimeBasedPolicy;
/// use chrono::Duration;
///
/// // Promote memories older than 7 days
/// let policy = TimeBasedPolicy::new(Duration::days(7));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedPolicy {
    /// Minimum age for promotion (in seconds)
    #[serde(with = "chrono::serde::ts_seconds")]
    pub min_age: ChronoDuration,
}

impl TimeBasedPolicy {
    /// Create a new time-based policy.
    ///
    /// # Arguments
    ///
    /// * `min_age` - Minimum age duration for promotion
    pub fn new(min_age: ChronoDuration) -> Self {
        Self { min_age }
    }
}

impl MemoryPromotionPolicy for TimeBasedPolicy {
    fn should_promote(&self, memory: &MemoryEntry) -> bool {
        if memory.memory_type != MemoryType::ShortTerm {
            return false;
        }

        let age = Utc::now() - memory.created_at;
        age >= self.min_age
    }

    fn description(&self) -> String {
        format!(
            "Promotes memories older than {} days",
            self.min_age.num_days()
        )
    }

    fn name(&self) -> &'static str {
        "TimeBased"
    }

    fn promotion_score(&self, memory: &MemoryEntry) -> f64 {
        if memory.memory_type != MemoryType::ShortTerm {
            return 0.0;
        }

        let age = Utc::now() - memory.created_at;
        let ratio = age.num_seconds() as f64 / self.min_age.num_seconds() as f64;
        ratio.min(1.0)
    }
}

/// Promotes memories based on importance score.
///
/// Memories with high importance scores are promoted regardless of
/// access patterns or age.
///
/// # Examples
///
/// ```rust,no_run
/// use synaptic::memory::promotion::ImportancePolicy;
///
/// // Promote memories with importance >= 0.8
/// let policy = ImportancePolicy::new(0.8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportancePolicy {
    /// Minimum importance score for promotion (0.0 to 1.0)
    pub importance_threshold: f64,
}

impl ImportancePolicy {
    /// Create a new importance-based policy.
    ///
    /// # Arguments
    ///
    /// * `importance_threshold` - Minimum importance score (0.0 to 1.0)
    pub fn new(importance_threshold: f64) -> Self {
        Self {
            importance_threshold: importance_threshold.clamp(0.0, 1.0),
        }
    }
}

impl MemoryPromotionPolicy for ImportancePolicy {
    fn should_promote(&self, memory: &MemoryEntry) -> bool {
        memory.memory_type == MemoryType::ShortTerm
            && memory.importance >= self.importance_threshold
    }

    fn description(&self) -> String {
        format!(
            "Promotes memories with importance >= {:.2}",
            self.importance_threshold
        )
    }

    fn name(&self) -> &'static str {
        "Importance"
    }

    fn promotion_score(&self, memory: &MemoryEntry) -> f64 {
        if memory.memory_type != MemoryType::ShortTerm {
            return 0.0;
        }

        (memory.importance / self.importance_threshold).min(1.0)
    }
}

/// Combines multiple promotion policies with weighted scoring.
///
/// This policy evaluates memories using multiple criteria and uses a weighted
/// average to make promotion decisions. This provides more nuanced promotion
/// behavior than any single policy.
///
/// # Examples
///
/// ```rust,no_run
/// use synaptic::memory::promotion::{HybridPolicy, AccessFrequencyPolicy, ImportancePolicy};
/// use chrono::Duration;
///
/// let mut policy = HybridPolicy::new(0.6); // 60% threshold for promotion
/// policy.add_policy(Box::new(AccessFrequencyPolicy::new(5)), 0.4);
/// policy.add_policy(Box::new(ImportancePolicy::new(0.7)), 0.6);
/// ```
#[derive(Default)]
pub struct HybridPolicy {
    /// Weighted policies (policy, weight)
    policies: Vec<(Box<dyn MemoryPromotionPolicy>, f64)>,
    /// Minimum combined score for promotion (0.0 to 1.0)
    promotion_threshold: f64,
}

impl HybridPolicy {
    /// Create a new hybrid policy.
    ///
    /// # Arguments
    ///
    /// * `promotion_threshold` - Minimum weighted score for promotion (0.0 to 1.0)
    pub fn new(promotion_threshold: f64) -> Self {
        Self {
            policies: Vec::new(),
            promotion_threshold: promotion_threshold.clamp(0.0, 1.0),
        }
    }

    /// Add a policy with a weight.
    ///
    /// # Arguments
    ///
    /// * `policy` - The policy to add
    /// * `weight` - Weight for this policy (will be normalized)
    pub fn add_policy(&mut self, policy: Box<dyn MemoryPromotionPolicy>, weight: f64) {
        self.policies.push((policy, weight.max(0.0)));
    }

    /// Calculate the weighted promotion score across all policies.
    fn calculate_weighted_score(&self, memory: &MemoryEntry) -> f64 {
        if self.policies.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = self.policies.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = self
            .policies
            .iter()
            .map(|(policy, weight)| policy.promotion_score(memory) * weight)
            .sum();

        weighted_sum / total_weight
    }
}

impl MemoryPromotionPolicy for HybridPolicy {
    fn should_promote(&self, memory: &MemoryEntry) -> bool {
        if memory.memory_type != MemoryType::ShortTerm {
            return false;
        }

        let score = self.calculate_weighted_score(memory);
        score >= self.promotion_threshold
    }

    fn description(&self) -> String {
        let policy_names: Vec<_> = self
            .policies
            .iter()
            .map(|(p, w)| format!("{}(w={:.2})", p.name(), w))
            .collect();

        format!(
            "Hybrid policy combining: {} (threshold: {:.2})",
            policy_names.join(", "),
            self.promotion_threshold
        )
    }

    fn name(&self) -> &'static str {
        "Hybrid"
    }

    fn promotion_score(&self, memory: &MemoryEntry) -> f64 {
        self.calculate_weighted_score(memory)
    }
}

/// Manager for applying promotion policies to memories.
///
/// This manager holds a promotion policy and provides methods for evaluating
/// and promoting memories based on that policy.
pub struct MemoryPromotionManager {
    policy: Box<dyn MemoryPromotionPolicy>,
}

impl MemoryPromotionManager {
    /// Create a new promotion manager with the given policy.
    ///
    /// # Arguments
    ///
    /// * `policy` - The promotion policy to use
    pub fn new(policy: Box<dyn MemoryPromotionPolicy>) -> Self {
        Self { policy }
    }

    /// Check if a memory should be promoted.
    pub fn should_promote(&self, memory: &MemoryEntry) -> bool {
        self.policy.should_promote(memory)
    }

    /// Promote a memory to long-term storage.
    ///
    /// This changes the memory's type to LongTerm and returns the modified entry.
    ///
    /// # Arguments
    ///
    /// * `memory` - The memory to promote
    ///
    /// # Returns
    ///
    /// The promoted memory entry
    pub fn promote_memory(&self, mut memory: MemoryEntry) -> Result<MemoryEntry> {
        if memory.memory_type == MemoryType::LongTerm {
            tracing::debug!(
                memory_key = %memory.key,
                "Memory already in long-term storage"
            );
            return Ok(memory);
        }

        tracing::info!(
            memory_key = %memory.key,
            access_count = memory.access_count,
            importance = memory.importance,
            policy = %self.policy.name(),
            "Promoting memory to long-term storage"
        );

        memory.memory_type = MemoryType::LongTerm;
        Ok(memory)
    }

    /// Get the promotion score for a memory.
    pub fn promotion_score(&self, memory: &MemoryEntry) -> f64 {
        self.policy.promotion_score(memory)
    }

    /// Get information about the current policy.
    pub fn policy_info(&self) -> (String, String) {
        (self.policy.name().to_string(), self.policy.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memory(
        key: &str,
        access_count: usize,
        importance: f64,
        created_at: DateTime<Utc>,
    ) -> MemoryEntry {
        let mut entry = MemoryEntry::new(
            key.to_string(),
            "test content".to_string(),
            MemoryType::ShortTerm,
        );
        entry.access_count = access_count;
        entry.importance = importance;
        entry.created_at = created_at;
        entry
    }

    #[test]
    fn test_access_frequency_policy() {
        let policy = AccessFrequencyPolicy::new(5);

        let low_access = create_test_memory("low", 3, 0.5, Utc::now());
        let high_access = create_test_memory("high", 10, 0.5, Utc::now());

        assert!(!policy.should_promote(&low_access));
        assert!(policy.should_promote(&high_access));

        assert!(policy.promotion_score(&low_access) < 1.0);
        assert_eq!(policy.promotion_score(&high_access), 1.0);
    }

    #[test]
    fn test_time_based_policy() {
        let policy = TimeBasedPolicy::new(ChronoDuration::days(7));

        let recent = create_test_memory("recent", 0, 0.5, Utc::now());
        let old = create_test_memory("old", 0, 0.5, Utc::now() - ChronoDuration::days(10));

        assert!(!policy.should_promote(&recent));
        assert!(policy.should_promote(&old));

        assert!(policy.promotion_score(&recent) < 0.1);
        assert!(policy.promotion_score(&old) > 0.9);
    }

    #[test]
    fn test_importance_policy() {
        let policy = ImportancePolicy::new(0.8);

        let low_importance = create_test_memory("low", 0, 0.5, Utc::now());
        let high_importance = create_test_memory("high", 0, 0.9, Utc::now());

        assert!(!policy.should_promote(&low_importance));
        assert!(policy.should_promote(&high_importance));
    }

    #[test]
    fn test_hybrid_policy() {
        let mut policy = HybridPolicy::new(0.5);
        policy.add_policy(Box::new(AccessFrequencyPolicy::new(5)), 0.5);
        policy.add_policy(Box::new(ImportancePolicy::new(0.8)), 0.5);

        // High access, low importance
        let mem1 = create_test_memory("mem1", 10, 0.3, Utc::now());
        assert!(policy.should_promote(&mem1)); // Access score alone should push over threshold

        // Low access, high importance
        let mem2 = create_test_memory("mem2", 1, 0.9, Utc::now());
        assert!(policy.should_promote(&mem2)); // Importance score alone should push over threshold

        // Low both
        let mem3 = create_test_memory("mem3", 1, 0.3, Utc::now());
        assert!(!policy.should_promote(&mem3));
    }

    #[test]
    fn test_promotion_manager() {
        let policy = Box::new(AccessFrequencyPolicy::new(5));
        let manager = MemoryPromotionManager::new(policy);

        let mut memory = create_test_memory("test", 10, 0.5, Utc::now());
        assert_eq!(memory.memory_type, MemoryType::ShortTerm);

        assert!(manager.should_promote(&memory));

        memory = manager.promote_memory(memory).unwrap();
        assert_eq!(memory.memory_type, MemoryType::LongTerm);
    }

    #[test]
    fn test_long_term_not_promoted() {
        let policy = AccessFrequencyPolicy::new(1);

        let mut memory = create_test_memory("test", 100, 0.9, Utc::now());
        memory.memory_type = MemoryType::LongTerm;

        assert!(!policy.should_promote(&memory));
    }
}
