//! Test-only op-count telemetry for the retrieval pipeline (feature
//! `test-utils`). Deterministic proxies for latency: tests assert bounded
//! counts instead of wall-clock time.

use std::sync::atomic::{AtomicUsize, Ordering};

static KG_READ_LOCKS: AtomicUsize = AtomicUsize::new(0);

/// Reset all retrieval telemetry counters to zero.
pub fn reset() {
    KG_READ_LOCKS.store(0, Ordering::SeqCst);
}

/// Number of knowledge-graph read-lock acquisitions performed by retrieval
/// stages since the last [`reset`].
pub fn kg_read_locks() -> usize {
    KG_READ_LOCKS.load(Ordering::SeqCst)
}

pub(crate) fn record_kg_read_lock() {
    KG_READ_LOCKS.fetch_add(1, Ordering::SeqCst);
}
