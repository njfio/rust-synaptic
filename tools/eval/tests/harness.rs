//! LLM-free harness tests: pinned metric math on hand-constructed inputs and
//! an end-to-end ingest/retrieve run on the committed fixture.

use std::collections::HashSet;
use std::time::Duration;
use synaptic::{AgentMemory, MemoryConfig};
use synaptic_eval::dataset::{load_locomo_str, load_longmemeval_str, QType};
use synaptic_eval::metrics::{mrr, percentile, precision_at_k, recall_at_k, LatencySummary};
use synaptic_eval::runner::{evaluate_conversations, ingest, measure_growth, run_question};

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn relset(v: &[&str]) -> HashSet<String> {
    v.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// (c) precision@k on a known set
// ---------------------------------------------------------------------------
#[test]
fn precision_at_k_pinned_values() {
    let retrieved = ids(&["a", "x", "b", "y"]);
    let relevant = relset(&["a", "b"]);
    // top-1: {a} ∩ {a,b} = 1 → 1/1
    assert_eq!(precision_at_k(&retrieved, &relevant, 1), 1.0);
    // top-2: {a,x} → 1/2
    assert_eq!(precision_at_k(&retrieved, &relevant, 2), 0.5);
    // top-4: {a,x,b,y} → 2/4
    assert_eq!(precision_at_k(&retrieved, &relevant, 4), 0.5);
    // k beyond list length: still divides by k → 2/8
    assert_eq!(precision_at_k(&retrieved, &relevant, 8), 0.25);
    // k == 0 guarded → 0.0
    assert_eq!(precision_at_k(&retrieved, &relevant, 0), 0.0);
}

// ---------------------------------------------------------------------------
// recall@k on known sets (guarding empty relevant set)
// ---------------------------------------------------------------------------
#[test]
fn recall_at_k_pinned_values() {
    let retrieved = ids(&["a", "x", "b", "y"]);
    let relevant = relset(&["a", "b", "c"]);
    // top-2 finds only a → 1/3
    assert!((recall_at_k(&retrieved, &relevant, 2) - 1.0 / 3.0).abs() < 1e-12);
    // top-4 finds a and b → 2/3
    assert!((recall_at_k(&retrieved, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
    // empty relevant set → 0.0, no division by zero
    assert_eq!(recall_at_k(&retrieved, &HashSet::new(), 4), 0.0);
    // all relevant retrieved → exactly 1.0
    let all = relset(&["a", "b"]);
    assert_eq!(recall_at_k(&retrieved, &all, 4), 1.0);
}

// ---------------------------------------------------------------------------
// (b) MRR on a KNOWN ranking: relevant at rank 2 → 0.5
// ---------------------------------------------------------------------------
#[test]
fn mrr_pinned_values() {
    let relevant = relset(&["b"]);
    // b is ranked 2nd (1-indexed) → 1/2
    assert_eq!(mrr(&ids(&["a", "b", "c"]), &relevant), 0.5);
    // first position → 1.0
    assert_eq!(mrr(&ids(&["b", "a"]), &relevant), 1.0);
    // rank 4 → 0.25
    assert_eq!(mrr(&ids(&["x", "y", "z", "b"]), &relevant), 0.25);
    // never retrieved → 0.0
    assert_eq!(mrr(&ids(&["x", "y"]), &relevant), 0.0);
}

// ---------------------------------------------------------------------------
// (d) nearest-rank percentile on a known vector
// ---------------------------------------------------------------------------
#[test]
fn percentile_nearest_rank_pinned_values() {
    let v: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
    // nearest-rank: rank = ceil(p/100 * 100)
    assert_eq!(percentile(&v, 50.0), Some(Duration::from_millis(50)));
    assert_eq!(percentile(&v, 95.0), Some(Duration::from_millis(95)));
    assert_eq!(percentile(&v, 99.0), Some(Duration::from_millis(99)));
    assert_eq!(percentile(&v, 100.0), Some(Duration::from_millis(100)));
    assert_eq!(percentile(&[], 50.0), None);
    // single element: every percentile is that element
    let one = [Duration::from_millis(7)];
    assert_eq!(percentile(&one, 1.0), Some(Duration::from_millis(7)));
    assert_eq!(percentile(&one, 99.0), Some(Duration::from_millis(7)));

    let summary = LatencySummary::from_durations(&v);
    assert_eq!(summary.p50_micros, 50_000);
    assert_eq!(summary.p95_micros, 95_000);
    assert_eq!(summary.p99_micros, 99_000);
    assert_eq!(summary.count, 100);
}

// ---------------------------------------------------------------------------
// (a) fixture end-to-end: evidence ingested + retrievable → recall@k == 1.0
// ---------------------------------------------------------------------------

fn fixture_section(key: &str) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/sample_conversations.json"
    );
    let raw = std::fs::read_to_string(path).expect("fixture must exist");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture must be valid JSON");
    serde_json::to_string(&value[key]).expect("fixture section must serialize")
}

#[tokio::test]
async fn fixture_longmemeval_recall_is_perfect_for_retrievable_evidence() {
    let convs = load_longmemeval_str(&fixture_section("longmemeval")).expect("fixture must parse");
    // fixture_lme_1: evidence session fixture_sess_b contains "Lisbon" which
    // also appears in the question's obvious retrieval surface.
    let conv = &convs[0];
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("in-memory AgentMemory must construct");
    ingest(conv, &mut memory)
        .await
        .expect("ingest must succeed");

    let q = &conv.questions[0];
    let outcome = run_question(q, &memory, 5).await.expect("search must run");
    let relevant: HashSet<String> = q.evidence_ids.iter().cloned().collect();
    assert!(
        !outcome.retrieved_ids.is_empty(),
        "search must return results"
    );
    assert_eq!(
        recall_at_k(&outcome.retrieved_ids, &relevant, 5),
        1.0,
        "evidence session must be recalled: retrieved={:?}",
        outcome.retrieved_ids
    );
}

#[tokio::test]
async fn fixture_locomo_retrieved_ids_use_dia_id_scheme() {
    let convs = load_locomo_str(&fixture_section("locomo")).expect("fixture must parse");
    let conv = &convs[0];
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("in-memory AgentMemory must construct");
    ingest(conv, &mut memory)
        .await
        .expect("ingest must succeed");

    let q = &conv.questions[0];
    let outcome = run_question(q, &memory, 5).await.expect("search must run");
    // LoCoMo evidence ids are dia_ids like "D1:1"; retrieved ids must be in
    // the same scheme so intersection is meaningful.
    assert!(
        outcome.retrieved_ids.iter().all(|id| id.starts_with('D')),
        "retrieved ids must be dia-style: {:?}",
        outcome.retrieved_ids
    );
    let relevant: HashSet<String> = q.evidence_ids.iter().cloned().collect();
    assert!(
        recall_at_k(&outcome.retrieved_ids, &relevant, 5) > 0.0,
        "at least one evidence turn must be recalled: retrieved={:?} relevant={:?}",
        outcome.retrieved_ids,
        relevant
    );
}

#[tokio::test]
async fn fixture_report_aggregates_and_breaks_down_by_qtype() {
    let mut convs =
        load_longmemeval_str(&fixture_section("longmemeval")).expect("fixture must parse");
    convs.extend(load_locomo_str(&fixture_section("locomo")).expect("fixture must parse"));
    let report = evaluate_conversations(&convs, 5)
        .await
        .expect("evaluation must run");

    assert!(report.questions_evaluated > 0);
    assert!(report.mean_precision_at_k >= 0.0 && report.mean_precision_at_k <= 1.0);
    assert!(report.mean_recall_at_k >= 0.0 && report.mean_recall_at_k <= 1.0);
    assert!(report.mrr >= 0.0 && report.mrr <= 1.0);
    assert_eq!(report.k, 5);
    // Both datasets contribute qtypes; the breakdown must cover every
    // evaluated question exactly once.
    let breakdown_total: usize = report.by_qtype.values().map(|b| b.questions).sum();
    assert_eq!(breakdown_total, report.questions_evaluated);
    assert!(report
        .by_qtype
        .contains_key(&format!("{:?}", QType::KnowledgeUpdate)));
    // Latency summaries come from real measured operations.
    assert!(report.store_latency.count > 0);
    assert!(report.recall_latency.count > 0);
    // Report serializes to JSON.
    let json = serde_json::to_string(&report).expect("report must serialize");
    assert!(json.contains("mean_recall_at_k"));
}

#[tokio::test]
async fn growth_measurement_reports_honest_size_proxy() {
    // Small targets keep the test fast; the proxy semantics are identical at
    // the documented 1k/10k/100k production targets.
    let points = measure_growth(&[10, 50, 100])
        .await
        .expect("growth run must succeed");
    assert_eq!(points.len(), 3);
    assert_eq!(points[0].target_memories, 10);
    assert_eq!(points[2].target_memories, 100);
    for p in &points {
        assert_eq!(p.stored_memories, p.target_memories);
        // Byte proxy is the sum of key+value UTF-8 lengths: strictly positive
        // and monotonically increasing with entry count.
        assert!(p.stored_bytes_estimate > 0);
    }
    assert!(points[0].stored_bytes_estimate < points[2].stored_bytes_estimate);
    assert!(points[0].store_latency.count == 10);
}
