//! Principled forgetting: importance-weighted decay driving tiered eviction.
//!
//! A [`ForgettingPolicy`] scores each memory with a *retained strength*:
//!
//! ```text
//! strength = decay(age) * importance * recency_of_access_factor
//! ```
//!
//! where `decay(age)` is computed by the real temporal decay models
//! (Ebbinghaus by default, see [`crate::memory::temporal::decay_models`]),
//! `importance` is the entry's stored importance score, and the recency
//! factor rewards recent and frequent access. Memories whose strength falls
//! below the policy's `retention_floor` are demoted a tier (long-term →
//! short-term, through the promotion machinery) or evicted outright when
//! already at the lowest tier.

use crate::error::Result;
use crate::memory::temporal::decay_models::{
    DecayConfig, DecayContext, DecayModelType, TemporalDecayModels,
};
use crate::memory::types::MemoryEntry;
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Half-life (hours) of the recency-of-access factor: an access `h` hours ago
/// contributes `exp(-h / RECENCY_HALF_LIFE_HOURS)` of its maximum protection.
const RECENCY_HALF_LIFE_HOURS: f64 = 24.0;

/// Floor of the recency factor: even a never-accessed memory keeps half of
/// the protection, so recency modulates rather than dominates the strength.
const RECENCY_FLOOR: f64 = 0.5;

/// Cap on the access-count boost so heavily accessed memories cannot become
/// unforgettable regardless of age and importance.
const MAX_ACCESS_BOOST: f64 = 2.0;

/// Policy controlling which memories are forgotten.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgettingPolicy {
    /// Memories whose retained strength falls below this floor are demoted or
    /// evicted. Strength is in `[0, importance * 2]`, typically well below 1.
    pub retention_floor: f64,
    /// Decay model used for the `decay(age)` term.
    pub decay: DecayModelType,
}

impl Default for ForgettingPolicy {
    fn default() -> Self {
        Self {
            retention_floor: 0.05,
            decay: DecayModelType::Ebbinghaus,
        }
    }
}

impl ForgettingPolicy {
    /// Compute the retained strength of a memory entry.
    ///
    /// Deterministic: depends only on the entry's `created_at`,
    /// `last_accessed`, `access_count`, `importance`, and the current time.
    /// The decay term is evaluated with adaptivity disabled and a neutral
    /// context so that importance and recency enter the product exactly once,
    /// via the explicit factors below.
    pub async fn retained_strength(&self, entry: &MemoryEntry) -> Result<f64> {
        let now = Utc::now();
        let age_hours = ((now - entry.created_at()).num_seconds().max(0) as f64) / 3600.0;
        let hours_since_access =
            ((now - entry.last_accessed()).num_seconds().max(0) as f64) / 3600.0;

        // Pure decay(age): non-adaptive config + neutral context so the model
        // yields the raw retention curve without importance/recency mixing.
        let config = DecayConfig {
            default_model: self.decay.clone(),
            adaptive_enabled: false,
            context_aware: false,
            importance_modulation: false,
            ..DecayConfig::default()
        };
        let mut models = TemporalDecayModels::new(config)?;
        let neutral_context = DecayContext {
            importance: 0.0,
            access_frequency: 0.0,
            hours_since_access,
            complexity: 0.0,
            emotional_weight: 0.0,
            contextual_relevance: 0.0,
            engagement_level: 0.0,
        };
        let decay = models
            .calculate_decay(&self.decay, age_hours, &neutral_context)
            .await?
            .retention_probability;

        let recency = Self::recency_of_access_factor(hours_since_access, entry.access_count());

        Ok(decay * entry.metadata.importance * recency)
    }

    /// Recency-of-access factor in `[RECENCY_FLOOR, MAX_ACCESS_BOOST]`.
    ///
    /// Exponentially decaying protection from the last access, multiplied by
    /// a logarithmic (capped) boost for cumulative access count.
    fn recency_of_access_factor(hours_since_access: f64, access_count: u64) -> f64 {
        let recency = RECENCY_FLOOR
            + (1.0 - RECENCY_FLOOR) * (-hours_since_access / RECENCY_HALF_LIFE_HOURS).exp();
        let access_boost = (1.0 + 0.25 * (1.0 + access_count as f64).ln()).min(MAX_ACCESS_BOOST);
        (recency * access_boost).min(MAX_ACCESS_BOOST)
    }
}

/// Outcome of a [`crate::AgentMemory::forget`] pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForgetReport {
    /// Keys removed from storage and state (were already at the lowest tier).
    pub evicted: Vec<String>,
    /// Keys demoted a tier (long-term → short-term) through the promotion
    /// machinery; they remain retrievable and may be evicted by a later pass.
    pub demoted: Vec<String>,
    /// Total number of memories examined.
    pub examined: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;
    use chrono::Duration;

    fn entry_with(importance: f64, age_hours: i64, accessed_hours_ago: i64) -> MemoryEntry {
        let mut e = MemoryEntry::new("k".into(), "v".into(), MemoryType::ShortTerm);
        e.metadata.created_at = Utc::now() - Duration::hours(age_hours);
        e.metadata.last_accessed = Utc::now() - Duration::hours(accessed_hours_ago);
        e.metadata.importance = importance;
        e
    }

    #[tokio::test]
    async fn strength_scales_with_importance() {
        let policy = ForgettingPolicy::default();
        let high = policy
            .retained_strength(&entry_with(0.9, 48, 48))
            .await
            .expect("strength computation succeeds for a valid entry");
        let low = policy
            .retained_strength(&entry_with(0.1, 48, 48))
            .await
            .expect("strength computation succeeds for a valid entry");
        assert!(high > low);
    }

    #[tokio::test]
    async fn strength_rewards_recent_access() {
        let policy = ForgettingPolicy::default();
        let recent = policy
            .retained_strength(&entry_with(0.5, 72, 0))
            .await
            .expect("strength computation succeeds for a valid entry");
        let stale = policy
            .retained_strength(&entry_with(0.5, 72, 72))
            .await
            .expect("strength computation succeeds for a valid entry");
        assert!(recent > stale);
    }

    #[test]
    fn recency_factor_is_bounded() {
        for (h, n) in [(0.0, 0), (0.0, 1_000_000), (10_000.0, 0), (10_000.0, 9)] {
            let f = ForgettingPolicy::recency_of_access_factor(h, n);
            assert!(
                (RECENCY_FLOOR..=MAX_ACCESS_BOOST).contains(&f),
                "factor {f} out of bounds"
            );
        }
    }
}
