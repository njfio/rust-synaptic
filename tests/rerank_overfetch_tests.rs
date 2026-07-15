//! Tests for the rerank over-fetch pool: `fuse_results` used to truncate the
//! candidate pool to `limit` BEFORE composite scoring and reranking, so the
//! reranker could only ever reorder the final `limit` candidates and could
//! never promote a gold result sitting just outside the top-`limit` (e.g.
//! rank 11-50). `PipelineConfig::rerank_pool_size` controls how wide a pool
//! survives fusion into composite scoring + reranking before the caller's
//! `limit` is applied as the final step.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use async_trait::async_trait;
use std::sync::Arc;
use synaptic::error::Result;
use synaptic::memory::retrieval::pipeline::{
    HybridRetriever, PipelineConfig, RetrievalPipeline, RetrievalSignal, ScoredMemory,
};
use synaptic::memory::retrieval::rerank::Reranker;
use synaptic::memory::types::{MemoryEntry, MemoryFragment, MemoryType};

/// A deterministic stub pipeline that returns a fixed, pre-ranked list of
/// `n` candidates with strictly descending scores `1.0, 0.99, 0.98, ...`.
/// Candidate at 0-based index `i` is `mem_i`; `gold` is placed at a chosen
/// index so tests can control exactly where fusion ranks it.
struct StubPipeline {
    n: usize,
    gold_index: usize,
}

#[async_trait]
impl RetrievalPipeline for StubPipeline {
    async fn search(
        &self,
        _query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        let mut results = Vec::new();
        for i in 0..self.n.min(limit) {
            let key = if i == self.gold_index {
                "mem_gold".to_string()
            } else {
                format!("mem_{i}")
            };
            let entry = MemoryEntry::new(
                key,
                format!("distractor content number {i}"),
                MemoryType::LongTerm,
            );
            let score = 1.0 - (i as f64) * 0.01;
            let fragment = MemoryFragment::new(entry, score);
            results.push(ScoredMemory::new(
                fragment,
                score,
                RetrievalSignal::DenseVector,
            ));
        }
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "stub"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::DenseVector
    }
}

/// A reranker stub that always promotes `mem_gold` to the very top of
/// whatever candidate set it is given, leaving the relative order of every
/// other candidate untouched. This isolates the test from the heuristic
/// reranker's actual scoring logic: it only needs to prove that WHEN the
/// reranker wants to promote a low-ranked candidate, it can only do so if
/// that candidate survived fusion into the pool it's handed.
struct PromoteGoldReranker;

#[async_trait]
impl Reranker for PromoteGoldReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut candidates: Vec<ScoredMemory>,
    ) -> Result<Vec<ScoredMemory>> {
        candidates.sort_by(|a, b| {
            let a_gold = a.memory.entry.key == "mem_gold";
            let b_gold = b.memory.entry.key == "mem_gold";
            b_gold
                .cmp(&a_gold)
                .then_with(|| a.memory.entry.key.cmp(&b.memory.entry.key))
        });
        Ok(candidates)
    }

    fn name(&self) -> &str {
        "promote-gold"
    }
}

/// With the default over-fetch pool (50) and `limit = 10`, a gold candidate
/// that fusion ranks at position 15 (i.e. outside the naive top-10) still
/// survives into the reranker's candidate set and gets promoted into the
/// final top-10.
#[tokio::test]
async fn rerank_promotes_below_limit_candidate_with_overfetch() {
    let gold_index = 14; // 0-based rank 14 => rank 15, outside top-10.
    let pipeline = Arc::new(StubPipeline { n: 50, gold_index });

    let config = PipelineConfig::default(); // rerank_pool_size = 50
    let retriever = HybridRetriever::new(config)
        .add_pipeline(pipeline)
        .with_reranker(Arc::new(PromoteGoldReranker));

    let results = retriever.search("distractor content", 10).await.unwrap();

    assert!(results.len() <= 10, "must respect caller limit");
    assert!(
        results.iter().any(|f| f.entry.key == "mem_gold"),
        "gold candidate at fusion rank 15 should be promoted into the final \
         top-10 by the reranker when the pool is wide enough to include it"
    );
}

/// Same fixture, but with `rerank_pool_size` shrunk to equal `limit` (10):
/// the effective pool is `max(rerank_pool_size, limit) = 10`, so the gold
/// candidate at fusion rank 15 never survives into the pool the reranker
/// sees, and cannot be promoted into the final result. This proves the
/// over-fetch, not the reranker's promotion logic, is what creates the
/// headroom.
#[tokio::test]
async fn rerank_cannot_promote_below_pool_candidate_without_overfetch() {
    let gold_index = 14;
    let pipeline = Arc::new(StubPipeline { n: 50, gold_index });

    let config = PipelineConfig::default().with_rerank_pool_size(10);
    let retriever = HybridRetriever::new(config)
        .add_pipeline(pipeline)
        .with_reranker(Arc::new(PromoteGoldReranker));

    let results = retriever.search("distractor content", 10).await.unwrap();

    assert!(results.len() <= 10);
    assert!(
        !results.iter().any(|f| f.entry.key == "mem_gold"),
        "gold candidate at fusion rank 15 must NOT appear when the pool is \
         truncated to the caller's limit before reranking"
    );
}

/// Baseline sanity check: `search` never returns more than `limit` results,
/// regardless of how wide the over-fetch pool is.
#[tokio::test]
async fn search_respects_caller_limit() {
    let pipeline = Arc::new(StubPipeline {
        n: 50,
        gold_index: 0,
    });
    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config).add_pipeline(pipeline);

    let results = retriever.search("distractor content", 5).await.unwrap();
    assert!(results.len() <= 5);
}
