//! Integration tests for principled forgetting (Task 5.1).
//!
//! Retained strength = decay(age) * importance * recency-of-access factor;
//! memories below the retention floor are evicted (short-term) or demoted
//! (long-term) through the promotion-tier machinery.

use chrono::{Duration, Utc};
use synaptic::memory::forgetting::ForgettingPolicy;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::{AgentMemory, MemoryConfig};

/// Build a memory entry backdated by `age_hours`, never accessed since creation.
fn aged_entry(key: &str, importance: f64, age_hours: i64) -> MemoryEntry {
    let mut entry = MemoryEntry::new(
        key.to_string(),
        format!("content for {key}"),
        MemoryType::ShortTerm,
    );
    let then = Utc::now() - Duration::hours(age_hours);
    entry.metadata.created_at = then;
    entry.metadata.last_accessed = then;
    entry.metadata.access_count = 0;
    entry.metadata.importance = importance;
    entry
}

async fn memory_system() -> AgentMemory {
    AgentMemory::new(MemoryConfig::default())
        .await
        .expect("default AgentMemory construction must succeed")
}

#[tokio::test]
async fn forget_evicts_low_importance_and_keeps_high_importance_same_age() {
    let mut memory = memory_system().await;

    // Two same-age (72h) memories differing only in importance.
    let high = aged_entry("high_importance", 0.9, 72);
    let low = aged_entry("low_importance", 0.1, 72);
    memory
        .storage()
        .store(&high)
        .await
        .expect("storing high-importance entry must succeed");
    memory
        .storage()
        .store(&low)
        .await
        .expect("storing low-importance entry must succeed");

    let policy = ForgettingPolicy {
        retention_floor: 0.02,
        ..ForgettingPolicy::default()
    };
    let report = memory
        .forget(policy)
        .await
        .expect("forget must succeed on a healthy store");

    assert!(
        report.evicted.contains(&"low_importance".to_string()),
        "low-importance memory should be evicted; report: {report:?}"
    );
    assert!(
        !report.evicted.contains(&"high_importance".to_string()),
        "high-importance memory must survive; report: {report:?}"
    );
    assert!(
        !report.demoted.contains(&"high_importance".to_string()),
        "high-importance short-term memory is above the floor, not demoted"
    );

    // Eviction is real: the entry is gone from storage and state.
    let gone = memory
        .retrieve("low_importance")
        .await
        .expect("retrieve must not error");
    assert!(gone.is_none(), "evicted memory must not be retrievable");
    let kept = memory
        .retrieve("high_importance")
        .await
        .expect("retrieve must not error");
    assert!(kept.is_some(), "surviving memory must still be retrievable");
}

#[tokio::test]
async fn forget_spares_recently_accessed_over_never_accessed_same_importance() {
    let mut memory = memory_system().await;

    // Same age (72h), same low importance (0.3); one was accessed just now.
    let never = aged_entry("never_accessed", 0.3, 72);
    let mut recent = aged_entry("recently_accessed", 0.3, 72);
    recent.metadata.last_accessed = Utc::now();
    recent.metadata.access_count = 4;

    memory
        .storage()
        .store(&never)
        .await
        .expect("storing never-accessed entry must succeed");
    memory
        .storage()
        .store(&recent)
        .await
        .expect("storing recently-accessed entry must succeed");

    let policy = ForgettingPolicy {
        retention_floor: 0.03,
        ..ForgettingPolicy::default()
    };
    let report = memory
        .forget(policy)
        .await
        .expect("forget must succeed on a healthy store");

    assert!(
        report.evicted.contains(&"never_accessed".to_string()),
        "never-accessed memory should be evicted; report: {report:?}"
    );
    assert!(
        !report.evicted.contains(&"recently_accessed".to_string()),
        "recently-accessed memory must survive; report: {report:?}"
    );

    let kept = memory
        .retrieve("recently_accessed")
        .await
        .expect("retrieve must not error");
    assert!(kept.is_some(), "recently-accessed memory must survive");
}

#[tokio::test]
async fn forget_demotes_long_term_memories_instead_of_evicting() {
    let mut memory = memory_system().await;

    // A long-term memory below the floor is demoted a tier, not deleted.
    let mut long_term = aged_entry("faded_long_term", 0.1, 72);
    long_term.memory_type = MemoryType::LongTerm;
    memory
        .storage()
        .store(&long_term)
        .await
        .expect("storing long-term entry must succeed");

    let policy = ForgettingPolicy {
        retention_floor: 0.02,
        ..ForgettingPolicy::default()
    };
    let report = memory
        .forget(policy)
        .await
        .expect("forget must succeed on a healthy store");

    assert!(
        report.demoted.contains(&"faded_long_term".to_string()),
        "long-term memory below floor should be demoted; report: {report:?}"
    );
    assert!(
        !report.evicted.contains(&"faded_long_term".to_string()),
        "long-term memory is demoted, not evicted, on first pass"
    );

    let entry = memory
        .retrieve("faded_long_term")
        .await
        .expect("retrieve must not error")
        .expect("demoted memory must still exist");
    assert_eq!(
        entry.memory_type,
        MemoryType::ShortTerm,
        "demotion must move the memory down a tier"
    );
}

#[tokio::test]
async fn demotion_preserves_access_metadata() {
    let mut memory = memory_system().await;

    // Long-term, below floor, with a definite last_accessed and access_count.
    let mut long_term = aged_entry("faded_lt", 0.1, 72);
    long_term.memory_type = MemoryType::LongTerm;
    let accessed_at = Utc::now() - Duration::hours(50);
    long_term.metadata.last_accessed = accessed_at;
    long_term.metadata.access_count = 3;
    memory
        .storage()
        .store(&long_term)
        .await
        .expect("storing long-term entry must succeed");

    let policy = ForgettingPolicy {
        retention_floor: 0.02,
        ..ForgettingPolicy::default()
    };
    let report = memory
        .forget(policy)
        .await
        .expect("forget must succeed on a healthy store");
    assert!(report.demoted.contains(&"faded_lt".to_string()));

    // Read the RAW storage copy (Storage::retrieve does not mark access, unlike
    // AgentMemory::retrieve) so the assertion sees exactly what demotion wrote.
    let after = memory
        .storage()
        .retrieve("faded_lt")
        .await
        .expect("retrieve must not error")
        .expect("demoted memory must still exist");
    assert_eq!(
        after.metadata.access_count, 3,
        "demotion must not bump access_count"
    );
    assert_eq!(
        after.metadata.last_accessed, accessed_at,
        "demotion must not bump last_accessed"
    );
    assert_eq!(after.memory_type, MemoryType::ShortTerm);
}

#[tokio::test]
async fn retained_strength_is_deterministic_and_ordered() {
    let policy = ForgettingPolicy::default();
    let fresh = aged_entry("fresh", 0.5, 1);
    let stale = aged_entry("stale", 0.5, 500);

    let s_fresh_1 = policy
        .retained_strength(&fresh)
        .await
        .expect("strength computation must succeed");
    let s_fresh_2 = policy
        .retained_strength(&fresh)
        .await
        .expect("strength computation must succeed");
    let s_stale = policy
        .retained_strength(&stale)
        .await
        .expect("strength computation must succeed");

    assert!(
        (s_fresh_1 - s_fresh_2).abs() < 1e-9,
        "retained strength must be deterministic"
    );
    assert!(
        s_fresh_1 > s_stale,
        "older memories must have lower retained strength ({s_fresh_1} vs {s_stale})"
    );
}
