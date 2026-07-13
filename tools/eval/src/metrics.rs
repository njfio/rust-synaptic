//! Retrieval-quality and latency metrics for the LLM-free evaluation harness.
//!
//! Everything here is pure arithmetic over ids and measured durations —
//! nothing is estimated or fabricated. Percentiles use the nearest-rank
//! method on the sorted sample.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

/// `|retrieved[..k] ∩ relevant| / k`.
///
/// Divides by `k` (not by the number retrieved), so a system that returns
/// fewer than `k` results is penalized. Returns 0.0 when `k == 0`.
pub fn precision_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f64 / k as f64
}

/// `|retrieved[..k] ∩ relevant| / |relevant|`; 0.0 when `relevant` is empty.
pub fn recall_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Reciprocal rank of the first relevant id (1-indexed); 0.0 if none appears.
pub fn mrr(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    retrieved
        .iter()
        .position(|id| relevant.contains(id))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0)
}

/// Nearest-rank percentile: sorts a copy of the sample ascending and returns
/// the element at rank `ceil(p/100 * n)` (1-indexed). `None` for an empty
/// sample. `p` is clamped to `(0, 100]`.
pub fn percentile(samples: &[Duration], p: f64) -> Option<Duration> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort_unstable();
    let p = p.clamp(f64::MIN_POSITIVE, 100.0);
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.clamp(1, sorted.len()) - 1;
    Some(sorted[idx])
}

/// p50/p95/p99 summary of measured operation latencies, in microseconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencySummary {
    /// Number of measured operations.
    pub count: usize,
    /// 50th percentile (nearest-rank), microseconds.
    pub p50_micros: u128,
    /// 95th percentile (nearest-rank), microseconds.
    pub p95_micros: u128,
    /// 99th percentile (nearest-rank), microseconds.
    pub p99_micros: u128,
}

impl LatencySummary {
    /// Summarize a sample of measured durations. An empty sample yields an
    /// all-zero summary with `count == 0` (never a fabricated latency).
    pub fn from_durations(samples: &[Duration]) -> Self {
        let micros = |p: f64| percentile(samples, p).map(|d| d.as_micros()).unwrap_or(0);
        Self {
            count: samples.len(),
            p50_micros: micros(50.0),
            p95_micros: micros(95.0),
            p99_micros: micros(99.0),
        }
    }
}

/// Retrieval metrics for a single evaluated question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionMetrics {
    /// Dataset-native question id.
    pub question_id: String,
    /// Debug-formatted [`crate::dataset::QType`].
    pub qtype: String,
    /// precision@k against the question's `evidence_ids`.
    pub precision_at_k: f64,
    /// recall@k against the question's `evidence_ids`.
    pub recall_at_k: f64,
    /// Reciprocal rank of the first relevant retrieved id.
    pub reciprocal_rank: f64,
}

/// Aggregate metrics for one question category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QTypeBreakdown {
    /// Questions in this category.
    pub questions: usize,
    /// Mean precision@k over the category.
    pub mean_precision_at_k: f64,
    /// Mean recall@k over the category.
    pub mean_recall_at_k: f64,
    /// Mean reciprocal rank over the category.
    pub mrr: f64,
}

/// One memory-growth measurement point.
///
/// **Size proxy (honest):** `stored_memories` is the number of successful
/// `AgentMemory::store` calls and `stored_bytes_estimate` is the sum of
/// UTF-8 key+value byte lengths written. This is the payload volume handed
/// to the store, NOT process RSS — no OS-level memory measurement is taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthPoint {
    /// The requested number of memories for this point.
    pub target_memories: usize,
    /// Memories actually stored (equals target on success).
    pub stored_memories: usize,
    /// Sum of key+value UTF-8 bytes handed to `store` (payload proxy, not RSS).
    pub stored_bytes_estimate: usize,
    /// `MemoryStats::total_size` as reported by the library after the fill.
    pub library_reported_total_size: usize,
    /// Store-latency percentiles measured during the fill.
    pub store_latency: LatencySummary,
    /// Latency of a single probe search executed after the fill, microseconds.
    pub probe_search_micros: u128,
}

/// Full harness report: retrieval quality, latency, and growth. Every number
/// is computed from a real run; absent measurements stay at zero counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// The k used for precision@k / recall@k.
    pub k: usize,
    /// Total questions evaluated.
    pub questions_evaluated: usize,
    /// Mean precision@k across all questions.
    pub mean_precision_at_k: f64,
    /// Mean recall@k across all questions.
    pub mean_recall_at_k: f64,
    /// Mean reciprocal rank across all questions.
    pub mrr: f64,
    /// Per-category breakdown keyed by debug-formatted `QType`.
    pub by_qtype: BTreeMap<String, QTypeBreakdown>,
    /// Per-question detail rows.
    pub per_question: Vec<QuestionMetrics>,
    /// Store-operation latency percentiles.
    pub store_latency: LatencySummary,
    /// Recall (search) operation latency percentiles.
    pub recall_latency: LatencySummary,
    /// Growth measurement points (empty unless a growth run was performed).
    pub growth: Vec<GrowthPoint>,
}

/// Aggregate per-question metrics into overall means and a per-qtype table.
pub fn aggregate(
    per_question: &[QuestionMetrics],
) -> (f64, f64, f64, BTreeMap<String, QTypeBreakdown>) {
    let n = per_question.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, BTreeMap::new());
    }
    let mean = |f: fn(&QuestionMetrics) -> f64| per_question.iter().map(f).sum::<f64>() / n as f64;
    let mut by_qtype: BTreeMap<String, QTypeBreakdown> = BTreeMap::new();
    for q in per_question {
        let b = by_qtype.entry(q.qtype.clone()).or_default();
        b.questions += 1;
        b.mean_precision_at_k += q.precision_at_k;
        b.mean_recall_at_k += q.recall_at_k;
        b.mrr += q.reciprocal_rank;
    }
    for b in by_qtype.values_mut() {
        let c = b.questions as f64;
        b.mean_precision_at_k /= c;
        b.mean_recall_at_k /= c;
        b.mrr /= c;
    }
    (
        mean(|q| q.precision_at_k),
        mean(|q| q.recall_at_k),
        mean(|q| q.reciprocal_rank),
        by_qtype,
    )
}
