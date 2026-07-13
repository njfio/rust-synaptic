//! Tests for the Reranker trait and the deterministic HeuristicReranker.
//!
//! Core property: a candidate over-ranked by ONE signal alone is demoted
//! below a candidate that MULTIPLE cross-features (term overlap, embedding
//! agreement, recency) agree on, after rerank.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use synaptic::memory::embeddings::TfIdfProvider;
use synaptic::memory::retrieval::{HeuristicReranker, Reranker, RetrievalSignal, ScoredMemory};
use synaptic::memory::types::{MemoryEntry, MemoryFragment, MemoryType};

fn candidate(key: &str, content: &str, score: f64, age_days: i64) -> ScoredMemory {
    let mut entry = MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::LongTerm);
    entry.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(age_days);
    ScoredMemory::new(
        MemoryFragment::new(entry, score),
        score,
        RetrievalSignal::Hybrid,
    )
}

/// A candidate that only ONE signal over-ranked (high incoming score, but the
/// query terms barely appear in it, its embedding disagrees with the query,
/// and it is stale) must be demoted below a candidate that MULTIPLE
/// cross-features agree on (strong term overlap, embedding agreement,
/// recency) even though its incoming score is lower.
#[tokio::test]
async fn multi_feature_agreement_beats_single_signal_overrank() {
    let reranker = HeuristicReranker::new(Some(Arc::new(TfIdfProvider::default())), None);

    let query = "database connection pooling configuration";

    // Over-ranked by one signal alone: top incoming score, but shares only
    // one incidental term with the query, off-topic content, 90 days old.
    let one_signal = candidate(
        "one_signal_overranked",
        "carpool pooling signup sheet for the summer picnic and volleyball",
        0.95,
        90,
    );

    // Multiple cross-features agree: strong term overlap with the query,
    // embedding agreement (same vocabulary), fresh — but lower incoming score.
    let consensus = candidate(
        "multi_feature_consensus",
        "database connection pooling configuration tuning for postgres connection limits",
        0.55,
        0,
    );

    // Order in: the over-ranked candidate is first.
    let reranked = reranker
        .rerank(query, vec![one_signal, consensus])
        .await
        .unwrap();

    assert_eq!(reranked.len(), 2, "reranker must not drop candidates");
    assert_eq!(
        reranked[0].memory.entry.key, "multi_feature_consensus",
        "candidate with multi-feature agreement must be promoted above the single-signal over-rank"
    );
    assert_eq!(reranked[1].memory.entry.key, "one_signal_overranked");
}

/// The reranker is deterministic: same input, same output ordering, with a
/// stable tie-break by key for identical rerank scores.
#[tokio::test]
async fn rerank_is_deterministic_with_stable_tiebreak() {
    let reranker = HeuristicReranker::new(Some(Arc::new(TfIdfProvider::default())), None);
    let query = "alpha beta";

    // Identical content and age -> identical cross-features -> tie broken by key.
    let make = || {
        vec![
            candidate("zeta", "alpha beta gamma", 0.5, 1),
            candidate("acme", "alpha beta gamma", 0.5, 1),
        ]
    };

    let first = reranker.rerank(query, make()).await.unwrap();
    let second = reranker.rerank(query, make()).await.unwrap();

    let keys: Vec<&str> = first.iter().map(|s| s.memory.entry.key.as_str()).collect();
    let keys2: Vec<&str> = second.iter().map(|s| s.memory.entry.key.as_str()).collect();
    assert_eq!(keys, keys2, "rerank must be deterministic");
    assert_eq!(
        keys,
        vec!["acme", "zeta"],
        "exact ties must break by ascending key"
    );
}

/// The reranker reorders; it does not fabricate scores: the original
/// pipeline scores travel through unchanged.
#[tokio::test]
async fn rerank_preserves_original_scores() {
    let reranker = HeuristicReranker::new(Some(Arc::new(TfIdfProvider::default())), None);
    let a = candidate("a", "alpha beta", 0.9, 1);
    let b = candidate("b", "gamma delta", 0.4, 1);

    let reranked = reranker.rerank("alpha", vec![a, b]).await.unwrap();
    let mut scores: Vec<f64> = reranked.iter().map(|s| s.score).collect();
    scores.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(scores, vec![0.4, 0.9], "original scores must be preserved");
}

/// Reranker exposes a name and handles the empty candidate set.
#[tokio::test]
async fn reranker_name_and_empty_input() {
    let reranker = HeuristicReranker::new(None, None);
    assert_eq!(reranker.name(), "HeuristicReranker");
    let out = reranker.rerank("anything", Vec::new()).await.unwrap();
    assert!(out.is_empty());
}
