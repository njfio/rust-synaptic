//! Scaling tests for auto-summarization triggers in `AdvancedMemoryManager`.
//!
//! Guards against the O(n^2) store path: per-store trigger evaluation must
//! examine a BOUNDED candidate set (index-backed search + a capped recent-
//! creations window), never the full corpus, and full summarization must be
//! debounced rather than re-run on every store once a cluster exists.
#![cfg(feature = "test-utils")]
// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use synaptic::memory::{
    management::{AdvancedMemoryManager, MemoryManagementConfig},
    storage::{memory::MemoryStorage, Storage},
    types::{MemoryEntry, MemoryType},
};

fn manager() -> AdvancedMemoryManager {
    AdvancedMemoryManager::new(MemoryManagementConfig::default())
}

/// Per-store trigger evaluation must examine a bounded number of candidates,
/// independent of corpus size: with 500 stored memories the max examined per
/// store stays under a fixed cap (search limit + recent-window cap), where the
/// old full-scan implementation examined ~n.
#[tokio::test]
async fn trigger_evaluation_examines_bounded_candidates_per_store() {
    let storage = Arc::new(MemoryStorage::new());
    let mut mgr = manager();

    for i in 0..500usize {
        // Unique vocabulary per entry: no related cluster forms, this
        // isolates the candidate-examination cost of trigger evaluation.
        let entry = MemoryEntry::new(
            format!("scale_key_{i}"),
            format!("alpha{i} beta{i} gamma{i} delta{i}"),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
        mgr.add_memory(&*storage, entry, None).await.unwrap();
    }

    let max_examined = mgr.max_trigger_candidates_examined();
    assert!(
        max_examined <= 200,
        "trigger evaluation examined {max_examined} candidates in one store; \
         must be bounded (<= 200), not O(n) with n=500"
    );
}

/// The feature must still WORK: when more than `summarization_threshold`
/// related memories about one topic accumulate, a related-memory-threshold
/// summarization is actually triggered and executed.
#[tokio::test]
async fn summarization_still_fires_when_related_cluster_crosses_threshold() {
    let storage = Arc::new(MemoryStorage::new());
    let mut mgr = manager(); // default threshold = 10

    let mut saw_related_trigger = false;
    for i in 0..12usize {
        // Shared topical vocabulary -> lexical relatedness (>= 2 word overlap).
        let entry = MemoryEntry::new(
            format!("topic_key_{i:02}"),
            format!("rust memory graph engine design note note variant{i}"),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
        let result = mgr.add_memory(&*storage, entry, None).await.unwrap();
        if result
            .messages
            .iter()
            .any(|m| m.contains("Related memory count"))
        {
            saw_related_trigger = true;
        }
    }

    assert!(
        saw_related_trigger,
        "related-memory-threshold summarization must still fire once a \
         topical cluster crosses the threshold"
    );
    assert!(
        mgr.auto_summarization_run_count() >= 1,
        "automatic summarization must actually execute at least once"
    );
}

/// Debounce: once a cluster has been summarized, the full summarization pass
/// must NOT re-run on every subsequent store; it re-fires only after the
/// cluster has grown by another `summarization_threshold` memories.
#[tokio::test]
async fn summarization_is_debounced_not_per_store() {
    let storage = Arc::new(MemoryStorage::new());
    let mut mgr = manager(); // default threshold = 10

    let stores = 30usize;
    for i in 0..stores {
        let entry = MemoryEntry::new(
            format!("debounce_key_{i:02}"),
            format!("rust memory graph engine design note note variant{i}"),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
        mgr.add_memory(&*storage, entry, None).await.unwrap();
    }

    let runs = mgr.auto_summarization_run_count();
    assert!(runs >= 1, "summarization must still run at least once");
    assert!(
        runs <= 8,
        "summarization ran {runs} times over {stores} stores; it must be \
         debounced (sublinear), not re-run on every store"
    );
}
