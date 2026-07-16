//! Tests for pseudo-relevance feedback (PRF) query expansion.
//!
//! Multi-evidence questions often fail because their complete evidence set
//! is not in the retrieved candidate pool at all: a single query surfaces
//! the turn most similar to the query but misses complementary evidence
//! that shares little vocabulary with the query itself. PRF mines salient
//! terms from the top results of the FIRST retrieval, expands the query
//! with them, retrieves AGAIN, and unions the second retrieval into the
//! pool — this can pull in evidence the first query alone could not reach.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use async_trait::async_trait;
use std::sync::Arc;
use synaptic::error::Result;
use synaptic::memory::retrieval::pipeline::{
    HybridRetriever, PipelineConfig, RetrievalPipeline, RetrievalSignal, ScoredMemory,
};
use synaptic::memory::types::{MemoryEntry, MemoryFragment, MemoryType};

/// A deterministic stub pipeline whose per-query score for each candidate is
/// the fraction of the QUERY's tokens found in the candidate's content
/// (case-insensitive, alphanumeric tokenization). This lets a test control
/// exactly which candidates the pipeline "finds" for a given query string,
/// so PRF's second (expanded) query can surface a candidate the first query
/// could not.
struct QueryAwarePipeline {
    candidates: Vec<(&'static str, &'static str)>,
}

fn tokenize(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

#[async_trait]
impl RetrievalPipeline for QueryAwarePipeline {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        let query_terms = tokenize(query);
        let mut results = Vec::new();
        for (key, content) in &self.candidates {
            let content_terms = tokenize(content);
            let hits = query_terms
                .iter()
                .filter(|t| content_terms.contains(*t))
                .count();
            let score = if query_terms.is_empty() {
                0.0
            } else {
                hits as f64 / query_terms.len() as f64
            };
            let entry =
                MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::LongTerm);
            let fragment = MemoryFragment::new(entry, score);
            results.push(ScoredMemory::new(
                fragment,
                score,
                RetrievalSignal::DenseVector,
            ));
        }
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.memory.entry.key.cmp(&b.memory.entry.key))
        });
        results.truncate(limit);
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "query-aware-stub"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::DenseVector
    }
}

/// The query is "alpha beta shared". `mem_a` matches all three query terms
/// AND contains a distinctive term ("zzzdistinct", repeated so it is the
/// clear top PRF expansion term). `mem_gold_b` shares NO terms with the
/// original query at all — it shares only "zzzdistinct" with `mem_a` — so a
/// single-query retrieval cannot find it (its raw score is 0.0, below
/// `min_score`, and RRF fusion filters it out entirely). PRF should mine
/// "zzzdistinct" out of `mem_a` (the top seed result), expand the query with
/// it, retrieve again, and pull `mem_gold_b` into the pool.
fn build_retriever(prf_enabled: bool) -> HybridRetriever {
    let pipeline = Arc::new(QueryAwarePipeline {
        candidates: vec![
            ("mem_a", "alpha beta zzzdistinct zzzdistinct shared"),
            ("mem_gold_b", "zzzdistinct completely unrelated payload"),
            ("mem_noise_1", "totally unrelated filler content one"),
            ("mem_noise_2", "totally unrelated filler content two"),
        ],
    });

    let config = PipelineConfig::default()
        .with_prf_enabled(prf_enabled)
        .with_prf_top_m(1)
        .with_prf_terms(1);

    HybridRetriever::new(config).add_pipeline(pipeline)
}

#[tokio::test]
async fn prf_disabled_misses_complementary_evidence() {
    let retriever = build_retriever(false);
    let results = retriever.search("alpha beta shared", 10).await.unwrap();

    assert!(
        results.iter().any(|f| f.entry.key == "mem_a"),
        "the directly-matching turn must still be found"
    );
    assert!(
        !results.iter().any(|f| f.entry.key == "mem_gold_b"),
        "without PRF, evidence sharing no terms with the query must be absent"
    );
}

#[tokio::test]
async fn prf_enabled_pulls_in_complementary_evidence() {
    let retriever = build_retriever(true);
    let results = retriever.search("alpha beta shared", 10).await.unwrap();

    assert!(
        results.iter().any(|f| f.entry.key == "mem_a"),
        "the directly-matching turn must still be found"
    );
    assert!(
        results.iter().any(|f| f.entry.key == "mem_gold_b"),
        "PRF must expand the query with a term mined from the top seed \
         result and pull complementary evidence into the pool"
    );
}

/// With PRF disabled, behaviour must be identical to the pre-PRF pipeline:
/// running the same query/limit twice yields the exact same ordered result.
#[tokio::test]
async fn prf_disabled_is_a_no_op() {
    let retriever = build_retriever(false);
    let first = retriever.search("alpha beta shared", 10).await.unwrap();
    let second = retriever.search("alpha beta shared", 10).await.unwrap();

    let first_keys: Vec<_> = first.iter().map(|f| f.entry.key.clone()).collect();
    let second_keys: Vec<_> = second.iter().map(|f| f.entry.key.clone()).collect();
    assert_eq!(
        first_keys, second_keys,
        "no-op PRF must be deterministic/identical"
    );
    assert_eq!(
        first_keys,
        vec!["mem_a"],
        "PRF disabled must match the pre-PRF single-query result set \
         (noise/gold candidates score 0.0 against this query and are \
         filtered by min_score)"
    );
}

/// PRF is disabled by default (no env override), so `PipelineConfig::default()`
/// must be a byte-identical no-op relative to a pre-PRF baseline config.
#[test]
fn prf_default_is_disabled() {
    // SAFETY: no other test in this process reads/writes this variable
    // concurrently as part of its assertions; this only guards against a
    // developer's local shell exporting SYNAPTIC_RETRIEVAL_PRF.
    std::env::remove_var(PipelineConfig::ENV_PRF);
    let config = PipelineConfig::default();
    assert!(!config.prf_enabled);
    assert_eq!(config.prf_top_m, 5);
    assert_eq!(config.prf_terms, 10);
}
