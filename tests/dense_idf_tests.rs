//! Integration tests proving dense retrieval performs REAL corpus-IDF
//! weighting, not degenerate hashed-TF cosine under a uniform IDF = 1.0.
//!
//! Wiring under test: `AgentMemory` retains the TF-IDF scoring provider that
//! the dense retriever and reranker share, and every `store` feeds the stored
//! content into it via the document (`embed`) path. That keeps the provider's
//! vocabulary / document-frequency statistics live, so query-time
//! `embed_for_scoring` sees real IDF values.
//!
//! Under the old inert wiring (provider never fed, vocabulary empty, IDF
//! falls back to 1.0) the rare-term ranking test below CANNOT pass: with
//! uniform IDF, a short document matching only the COMMON query term
//! out-scores the document matching the RARE discriminating term on pure
//! hashed-TF cosine. Only real IDF (downweighting the common term, boosting
//! the rare one) flips that ordering.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

async fn agent() -> AgentMemory {
    AgentMemory::new(MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        ..Default::default()
    })
    .await
    .expect("agent memory constructs")
}

/// Corpus: "database" is COMMON (appears in most docs); "quokka" is RARE
/// (one doc). The common-term doc is a single-token document so its
/// hashed-TF weight on "database" is maximal — under uniform IDF = 1.0 it
/// beats the rare-term doc for the query "quokka database". Real corpus IDF
/// must invert that.
fn corpus() -> Vec<(&'static str, &'static str)> {
    vec![
        ("rare_doc", "quokka sighting report"),
        ("common_hit", "database"),
        ("filler_1", "database schema migration tooling overview"),
        ("filler_2", "database connection retry logic timeout"),
        ("filler_3", "database index maintenance vacuum planning"),
        ("filler_4", "database backup restore verification runbook"),
        ("filler_5", "database replication lag monitoring alerts"),
    ]
}

/// The scoring provider's vocabulary must be populated by stores: after N
/// stores it reflects the corpus (it is NOT empty, which is exactly the
/// inert-IDF failure mode where `idf()` always returns the 1.0 fallback).
#[cfg(feature = "test-utils")]
#[tokio::test]
async fn scoring_provider_vocabulary_populated_by_stores() {
    let mut memory = agent().await;

    let before = memory
        .retrieval_scoring_vocabulary_size()
        .expect("embeddings enabled: scoring provider present");
    assert_eq!(before, 0, "fresh provider starts with empty vocabulary");

    for (key, value) in corpus() {
        memory
            .store(key, value)
            .await
            .unwrap_or_else(|e| panic!("store {key}: {e}"));
    }

    let after = memory
        .retrieval_scoring_vocabulary_size()
        .expect("embeddings enabled: scoring provider present");
    assert!(
        after > 0,
        "storing {} memories must populate the scoring provider's vocabulary \
         (empty vocabulary means IDF is the inert 1.0 fallback)",
        corpus().len()
    );
}

/// A query containing one RARE discriminating term and one COMMON term must
/// rank the rare-term document first. Under inert IDF = 1.0 the pure
/// hashed-TF cosine ranks the single-token common-term doc higher, so this
/// ordering is only achievable with real corpus IDF weighting.
#[tokio::test]
async fn rare_term_doc_ranks_first_with_real_idf() {
    let mut memory = agent().await;

    for (key, value) in corpus() {
        memory
            .store(key, value)
            .await
            .unwrap_or_else(|e| panic!("store {key}: {e}"));
    }

    let results = memory
        .search("quokka database", 3)
        .await
        .expect("search ok");
    assert!(!results.is_empty(), "search must return candidates");

    let keys: Vec<&str> = results.iter().map(|r| r.entry.key.as_str()).collect();
    assert_eq!(
        keys[0], "rare_doc",
        "real IDF must rank the rare-term doc first; got ranking {keys:?} \
         (a common-term doc on top means IDF is still the inert 1.0 fallback)"
    );
}
