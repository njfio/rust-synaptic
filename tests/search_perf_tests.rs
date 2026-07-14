//! Deterministic op-count and correctness tests for the search hot path.
//!
//! These tests are op-count based (not wall-clock): they assert that a
//! search embeds each distinct candidate content at most once, that
//! query-time scoring embeds never mutate the provider vocabulary, and that
//! the top-k ranking for a fixed corpus + query is unchanged by the
//! optimizations.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

/// Fixed corpus shared by the tests: keys with distinct contents.
fn corpus() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "doc_pooling",
            "database connection pooling configuration tuning for postgres",
        ),
        (
            "doc_conn",
            "database connection retry logic and timeout handling",
        ),
        ("doc_db", "database schema migration tooling overview notes"),
        ("doc_rust", "rust programming language memory safety"),
        ("doc_python", "python scripting for data pipelines"),
        ("doc_cooking", "cooking recipes for italian pasta dishes"),
    ]
}

/// (c) Correctness pin: the top-k ordering for a fixed corpus + query is
/// identical to the ordering produced before the hot-path optimizations
/// (captured from the pre-change pipeline on this branch).
#[tokio::test]
async fn top_k_ranking_unchanged_for_fixed_corpus() {
    let mut memory = AgentMemory::new(MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        ..Default::default()
    })
    .await
    .expect("agent memory constructs");

    for (key, value) in corpus() {
        memory
            .store(key, value)
            .await
            .unwrap_or_else(|e| panic!("store {key}: {e}"));
    }

    let results = memory
        .search("database connection pooling", 3)
        .await
        .expect("search ok");
    let keys: Vec<&str> = results.iter().map(|r| r.entry.key.as_str()).collect();

    // Pinned pre-optimization ordering (captured on this branch before the
    // read-only embed / cache changes). Rankings must be preserved.
    assert_eq!(
        keys,
        vec!["doc_pooling", "doc_conn", "doc_db"],
        "top-k ranking must be identical before/after the perf changes"
    );
}

#[cfg(feature = "test-utils")]
mod op_counts {
    use super::corpus;
    use std::sync::Arc;
    use synaptic::memory::embeddings::TfIdfProvider;
    use synaptic::memory::retrieval::{
        DenseVectorRetriever, HeuristicReranker, HybridRetriever, PipelineConfig,
    };
    use synaptic::memory::storage::memory::MemoryStorage;
    use synaptic::memory::storage::Storage;
    use synaptic::memory::types::{MemoryEntry, MemoryType};

    /// (a) A single hybrid search (dense retriever + heuristic reranker
    /// sharing one provider) embeds each DISTINCT content at most once:
    /// the number of actually computed scoring embeddings is bounded by
    /// distinct candidate contents + 1 (the query), not ~2N from the dense
    /// pass and the rerank pass embedding candidates independently.
    #[tokio::test]
    async fn single_search_embeds_each_distinct_content_at_most_once() {
        let storage = Arc::new(MemoryStorage::new());
        for (key, value) in corpus() {
            let entry = MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::LongTerm);
            storage.store(&entry).await.unwrap();
        }

        let provider = Arc::new(TfIdfProvider::default());
        // Threshold 0.0 so every candidate is scored and the top-K (>1)
        // reaches the reranker.
        let dense =
            DenseVectorRetriever::new(storage.clone(), provider.clone()).with_threshold(0.0);
        let reranker = HeuristicReranker::new(Some(provider.clone()), None);
        let hybrid = HybridRetriever::new(PipelineConfig::semantic_focus())
            .add_pipeline(Arc::new(dense))
            .with_reranker(Arc::new(reranker));

        let results = hybrid
            .search("database connection pooling", 5)
            .await
            .unwrap();
        assert!(!results.is_empty(), "search should return candidates");

        let distinct_contents = corpus().len();
        let computed = provider.scoring_embeds_computed();
        assert!(
            computed <= distinct_contents + 1,
            "one search must compute at most {} scoring embeds \
             (distinct contents + query), got {} — candidates are being re-embedded",
            distinct_contents + 1,
            computed
        );
        assert!(
            provider.scoring_cache_hits() > 0,
            "the reranker must reuse the dense retriever's cached embeddings"
        );
    }

    /// (b) Query-time scoring embeds are read-only: they never mutate the
    /// provider vocabulary (no write lock, no IDF rebuild).
    #[tokio::test]
    async fn scoring_embeds_do_not_mutate_vocabulary() {
        use synaptic::memory::embeddings::provider::EmbeddingProvider;

        let provider = TfIdfProvider::default();
        // Seed the vocabulary through the document (store-time) path.
        for (_, value) in corpus() {
            provider.embed(value, None).await.unwrap();
        }
        let vocab_before = provider.vocabulary_size();
        assert!(vocab_before > 0, "document embeds must build vocabulary");

        // Run many scoring embeds over NEW, unseen texts.
        for i in 0..50 {
            let text = format!("unseen scoring query text number {i} zebra quokka");
            provider.embed_for_scoring(&text, None).await.unwrap();
        }

        assert_eq!(
            provider.vocabulary_size(),
            vocab_before,
            "scoring embeds must not mutate the vocabulary"
        );
    }
}
