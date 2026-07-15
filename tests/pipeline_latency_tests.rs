//! Op-count regression tests for the search-pipeline latency fixes.
//!
//! These are deterministic proxies for the profiled hot spots — they count
//! knowledge-graph BFS traversals and KG read-lock acquisitions instead of
//! measuring wall-clock time:
//! - a query must perform a small CONSTANT number of KG BFS traversals,
//!   not O(candidates);
//! - `GraphRetriever` must acquire the KG read lock O(1) times per query;
//! - the optimizations must be ranking-preserving: graph scores and rerank
//!   proximities must equal the values the per-candidate BFS path produced.

#![cfg(feature = "test-utils")]
// Test code: unwrap/panic on failure is the intended behaviour. Holding the
// counter-serialization guard across awaits is deliberate: it is exactly what
// keeps the global op-counters from interleaving between tests.
#![allow(clippy::unwrap_used, clippy::panic, clippy::await_holding_lock)]

use std::sync::Arc;
use synaptic::memory::knowledge_graph::graph::traversal_telemetry;
use synaptic::memory::knowledge_graph::{GraphConfig, MemoryKnowledgeGraph, RelationshipType};
use synaptic::memory::retrieval::telemetry as retrieval_telemetry;
use synaptic::memory::retrieval::{
    GraphRetriever, HeuristicReranker, KeywordRetriever, MultiHopConfig, MultiHopGraphRetriever,
    PipelineConfig, Reranker, RetrievalPipeline, RetrievalSignal, ScoredMemory,
};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryFragment, MemoryType};

/// Serialize the counter-based tests: the telemetry counters are
/// process-global, so tests that reset/read them must not interleave.
static COUNTER_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_counters() -> std::sync::MutexGuard<'static, ()> {
    COUNTER_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Fixture: `n` memories that all lexically match the query, chained in the
/// knowledge graph (mem_0 -> mem_1 -> ... -> mem_{n-1}), with a couple of
/// extra cross edges so related-sets differ per candidate.
async fn fixture(
    n: usize,
) -> (
    Arc<MemoryStorage>,
    Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>,
) {
    let storage = Arc::new(MemoryStorage::new());
    let mut kg = MemoryKnowledgeGraph::new(GraphConfig::default());

    let mut entries = Vec::new();
    for i in 0..n {
        let mut entry = MemoryEntry::new(
            format!("mem_{i:03}"),
            format!("orbital station logistics report number {i}"),
            MemoryType::LongTerm,
        );
        // Space creation times > 1 hour apart so the KG's automatic
        // TemporallyRelated detection adds no edges beyond the explicit ones.
        entry.metadata.created_at =
            chrono::Utc::now() - chrono::Duration::hours(2 * (i as i64 + 1));
        storage.store(&entry).await.unwrap();
        kg.add_memory_node(&entry).await.unwrap();
        entries.push(entry);
    }

    for i in 0..n.saturating_sub(1) {
        kg.create_relationship(
            &format!("mem_{i:03}"),
            &format!("mem_{:03}", i + 1),
            RelationshipType::RelatedTo,
            None,
        )
        .await
        .unwrap();
    }
    // A couple of cross edges so proximity differs between candidates.
    if n >= 6 {
        kg.create_relationship("mem_000", "mem_004", RelationshipType::References, None)
            .await
            .unwrap();
        kg.create_relationship("mem_001", "mem_005", RelationshipType::SimilarTo, None)
            .await
            .unwrap();
    }

    (storage, Arc::new(tokio::sync::RwLock::new(kg)))
}

const QUERY: &str = "orbital station logistics";

fn candidates_from(fragments: Vec<MemoryFragment>) -> Vec<ScoredMemory> {
    fragments
        .into_iter()
        .map(|f| {
            let score = f.relevance_score;
            ScoredMemory::new(f, score, RetrievalSignal::SparseKeyword)
        })
        .collect()
}

/// (a)+(b): a GraphRetriever query performs a bounded, candidate-count
/// independent number of KG BFS traversals and acquires the KG read lock
/// O(1) times per query.
#[tokio::test]
async fn graph_retriever_traversals_and_locks_are_constant_per_query() {
    let _guard = lock_counters();

    let mut counts = Vec::new();
    for n in [8usize, 24] {
        let (storage, kg) = fixture(n).await;
        let retriever = GraphRetriever::new(storage, Some(kg));

        traversal_telemetry::reset();
        retrieval_telemetry::reset();
        let results = retriever.search(QUERY, n, None).await.unwrap();
        assert!(!results.is_empty());

        let traversals = traversal_telemetry::traversals();
        let locks = retrieval_telemetry::kg_read_locks();
        assert!(
            traversals <= 2,
            "GraphRetriever must not run a BFS per candidate: {traversals} traversals for {n} candidates"
        );
        assert!(
            locks <= 1,
            "GraphRetriever must acquire the KG read lock O(1) per query, got {locks}"
        );
        counts.push(traversals);
    }
    assert_eq!(
        counts[0], counts[1],
        "KG BFS traversal count must be independent of candidate count: {counts:?}"
    );
}

/// GraphRetriever's scores must be IDENTICAL to the per-candidate
/// `find_related_memories` formula (base + len/10 capped at 0.5 + 0.3 * avg
/// strength), i.e. the batching is a pure perf change.
#[tokio::test]
async fn graph_retriever_scores_match_per_candidate_bfs() {
    let _guard = lock_counters();

    let n = 12;
    let (storage, kg) = fixture(n).await;

    // Expected scores via the original per-candidate traversal API.
    let fragments = storage.search(QUERY, n * 2).await.unwrap();
    let mut expected: Vec<(String, f64)> = Vec::new();
    {
        let kg_guard = kg.read().await;
        for fragment in &fragments {
            let base = fragment.relevance_score;
            let score = match kg_guard
                .find_related_memories(&fragment.entry.key, 2, None)
                .await
            {
                Ok(related) => {
                    let boost = (related.len() as f64 / 10.0).min(0.5);
                    let avg = if related.is_empty() {
                        0.0
                    } else {
                        related.iter().map(|r| r.relationship_strength).sum::<f64>()
                            / related.len() as f64
                    };
                    (base + boost + avg * 0.3).min(1.0)
                }
                Err(_) => base,
            };
            expected.push((fragment.entry.key.clone(), score));
        }
    }

    let retriever = GraphRetriever::new(storage, Some(kg));
    let results = retriever.search(QUERY, n * 2, None).await.unwrap();

    for (key, want) in &expected {
        let got = results
            .iter()
            .find(|s| s.memory.entry.key == *key)
            .unwrap_or_else(|| panic!("candidate {key} missing from results"))
            .score;
        assert!(
            (got - want).abs() < 1e-12,
            "graph score for {key} changed: got {got}, want {want}"
        );
    }
}

/// Determinism/ranking pin: the same corpus + query must yield the same
/// top-k ordering on repeated runs of the full retriever.
#[tokio::test]
async fn graph_retriever_topk_is_deterministic() {
    let _guard = lock_counters();

    let (storage, kg) = fixture(12).await;
    let retriever = GraphRetriever::new(storage, Some(kg));

    let first: Vec<(String, f64)> = retriever
        .search(QUERY, 8, None)
        .await
        .unwrap()
        .into_iter()
        .map(|s| (s.memory.entry.key.clone(), s.score))
        .collect();
    for _ in 0..3 {
        let again: Vec<(String, f64)> = retriever
            .search(QUERY, 8, None)
            .await
            .unwrap()
            .into_iter()
            .map(|s| (s.memory.entry.key.clone(), s.score))
            .collect();
        assert_eq!(first, again, "top-k ordering must be deterministic");
    }
}

/// The heuristic reranker must compute all candidate graph proximities with
/// a bounded number of traversals (one batch pass), not one BFS per
/// candidate — and produce the same proximity-driven ordering.
#[tokio::test]
async fn reranker_proximity_is_single_pass_and_value_preserving() {
    let _guard = lock_counters();

    let n = 16;
    let (storage, kg) = fixture(n).await;
    let fragments = storage.search(QUERY, n).await.unwrap();
    let candidates = candidates_from(fragments);
    assert!(candidates.len() >= 8);

    // Expected proximities via the original per-candidate traversal.
    let all_keys: std::collections::HashSet<String> = candidates
        .iter()
        .map(|c| c.memory.entry.key.clone())
        .collect();
    let mut expected_order: Vec<(String, f64)> = Vec::new();
    {
        let kg_guard = kg.read().await;
        for candidate in &candidates {
            let key = &candidate.memory.entry.key;
            let related = kg_guard.find_related_memories(key, 2, None).await.unwrap();
            let hits = related
                .iter()
                .filter(|r| r.memory_key != *key && all_keys.contains(&r.memory_key))
                .count();
            let proximity = hits as f64 / (all_keys.len() - 1) as f64;
            expected_order.push((key.clone(), proximity));
        }
    }
    // Proximity is the only available feature below (no embeddings), so the
    // reranker must order by proximity desc, key asc.
    expected_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));

    let reranker = HeuristicReranker::new(None, Some(kg)).with_weights(
        synaptic::memory::retrieval::HeuristicRerankWeights {
            term_overlap: 0.0,
            embedding_agreement: 0.0,
            graph_proximity: 1.0,
            recency: 0.0,
        },
    );

    traversal_telemetry::reset();
    let reranked = reranker.rerank(QUERY, candidates).await.unwrap();
    let traversals = traversal_telemetry::traversals();
    assert!(
        traversals <= 1,
        "reranker must compute proximities in one batch traversal, got {traversals}"
    );

    let got: Vec<String> = reranked.into_iter().map(|s| s.memory.entry.key).collect();
    let want: Vec<String> = expected_order.into_iter().map(|(k, _)| k).collect();
    assert_eq!(got, want, "proximity-driven ordering must be preserved");
}

/// Storage wrapper that counts `search` calls, to prove the multi-hop
/// retriever reuses the pipeline's already-computed seed results instead of
/// re-running the seed retrievers from scratch.
struct SearchCountingStorage {
    inner: Arc<MemoryStorage>,
    searches: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl Storage for SearchCountingStorage {
    async fn store(&self, entry: &MemoryEntry) -> synaptic::error::Result<()> {
        self.inner.store(entry).await
    }
    async fn retrieve(&self, key: &str) -> synaptic::error::Result<Option<MemoryEntry>> {
        self.inner.retrieve(key).await
    }
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> synaptic::error::Result<Vec<MemoryFragment>> {
        self.searches
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.search(query, limit).await
    }
    async fn update(&self, key: &str, entry: &MemoryEntry) -> synaptic::error::Result<()> {
        self.inner.update(key, entry).await
    }
    async fn delete(&self, key: &str) -> synaptic::error::Result<bool> {
        self.inner.delete(key).await
    }
    async fn list_keys(&self) -> synaptic::error::Result<Vec<String>> {
        self.inner.list_keys().await
    }
    async fn count(&self) -> synaptic::error::Result<usize> {
        self.inner.count().await
    }
    async fn clear(&self) -> synaptic::error::Result<()> {
        self.inner.clear().await
    }
    async fn exists(&self, key: &str) -> synaptic::error::Result<bool> {
        self.inner.exists(key).await
    }
    async fn stats(&self) -> synaptic::error::Result<synaptic::memory::storage::StorageStats> {
        self.inner.stats().await
    }
    async fn maintenance(&self) -> synaptic::error::Result<()> {
        self.inner.maintenance().await
    }
    async fn backup(&self, path: &str) -> synaptic::error::Result<()> {
        self.inner.backup(path).await
    }
    async fn restore(&self, path: &str) -> synaptic::error::Result<()> {
        self.inner.restore(path).await
    }
    async fn get_all_entries(&self) -> synaptic::error::Result<Vec<MemoryEntry>> {
        self.inner.get_all_entries().await
    }
}

/// The multi-hop retriever inside a hybrid pipeline must not re-run its seed
/// retrievers: the keyword retriever's storage search runs ONCE per query.
#[tokio::test]
async fn multihop_reuses_pipeline_seed_results() {
    let _guard = lock_counters();

    let (inner, kg) = fixture(10).await;
    let storage = Arc::new(SearchCountingStorage {
        inner,
        searches: std::sync::atomic::AtomicUsize::new(0),
    });

    let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(
        storage.clone() as Arc<dyn Storage + Send + Sync>
    ));
    let multihop = MultiHopGraphRetriever::new(
        storage.clone() as Arc<dyn Storage + Send + Sync>,
        Some(kg),
        vec![keyword.clone()],
        MultiHopConfig::default(),
    );
    // Every fixture memory contains every query term, so BM25 IDF (and thus
    // the keyword scores) are near zero — drop the fusion min_score so the
    // candidates flow through and the search-call count is meaningful.
    let config = PipelineConfig::default().with_min_score(0.0);
    let hybrid = synaptic::memory::retrieval::HybridRetriever::new(config)
        .add_pipeline(keyword)
        .add_pipeline(Arc::new(multihop));

    storage
        .searches
        .store(0, std::sync::atomic::Ordering::SeqCst);
    let results = hybrid.search(QUERY, 10).await.unwrap();
    assert!(!results.is_empty());

    let searches = storage.searches.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        searches, 1,
        "multi-hop must reuse the pipeline's keyword results instead of re-running the seed retriever (storage.search called {searches} times)"
    );
}
