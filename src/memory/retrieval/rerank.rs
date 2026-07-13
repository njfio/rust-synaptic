//! Reranking stage for the retrieval pipeline.
//!
//! A [`Reranker`] takes the top-K candidates produced by fusion + composite
//! scoring and re-orders them using cross-features computed between the query
//! and each candidate. Rerankers reorder; they do not fabricate scores — each
//! candidate's original pipeline score travels through unchanged.

use crate::error::Result;
use crate::memory::embeddings::provider::EmbeddingProvider;
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use crate::memory::retrieval::pipeline::ScoredMemory;
use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;

/// A reranking stage applied to the top-K candidates after composite scoring.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Re-order the candidates by relevance to the query.
    ///
    /// Implementations must return the same candidates (no additions, no
    /// drops) in a new order, preserving each candidate's original score.
    async fn rerank(&self, query: &str, candidates: Vec<ScoredMemory>)
        -> Result<Vec<ScoredMemory>>;

    /// Name of this reranker.
    fn name(&self) -> &str;
}

/// Weights for the heuristic cross-features. Weights over unavailable
/// features (no embedding provider, no graph) are dropped and the remainder
/// renormalized, so the rerank score is always a `[0, 1]` blend of the
/// features actually computed.
#[derive(Debug, Clone, Copy)]
pub struct HeuristicRerankWeights {
    /// Weight of query-term / candidate-content overlap
    pub term_overlap: f64,
    /// Weight of query↔candidate embedding cosine agreement
    pub embedding_agreement: f64,
    /// Weight of graph proximity to the other top-K candidates
    pub graph_proximity: f64,
    /// Weight of recency (exponential decay by age)
    pub recency: f64,
}

impl Default for HeuristicRerankWeights {
    fn default() -> Self {
        Self {
            term_overlap: 0.35,
            embedding_agreement: 0.35,
            graph_proximity: 0.15,
            recency: 0.15,
        }
    }
}

/// Recency half-life for the heuristic reranker: one week, in hours.
const RECENCY_HALF_LIFE_HOURS: f64 = 168.0;

/// Deterministic cross-feature reranker.
///
/// For each candidate it computes, against the query:
/// - **term overlap** — fraction of query terms present in the candidate's
///   content (case-insensitive, alphanumeric tokenization);
/// - **embedding agreement** — cosine similarity between the query embedding
///   and the candidate-content embedding (via the configured
///   [`EmbeddingProvider`]), clamped to `[0, 1]`;
/// - **graph proximity** — fraction of the *other* top-K candidates reachable
///   from this candidate within 2 hops of the knowledge graph (when a graph
///   handle is configured and the candidate is a graph node);
/// - **recency** — exponential decay of the memory's age with a one-week
///   half-life.
///
/// These are blended with [`HeuristicRerankWeights`] (renormalized over the
/// available features) and the candidates re-ordered by the blend, descending,
/// with exact ties broken by ascending memory key so the ordering is stable
/// and deterministic. Original scores are preserved.
pub struct HeuristicReranker {
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
    weights: HeuristicRerankWeights,
}

impl HeuristicReranker {
    /// Create a reranker with the given (optional) embedding provider and
    /// (optional) knowledge-graph handle, using default weights.
    pub fn new(
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
        knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
    ) -> Self {
        Self {
            embedding_provider,
            knowledge_graph,
            weights: HeuristicRerankWeights::default(),
        }
    }

    /// Override the cross-feature weights.
    pub fn with_weights(mut self, weights: HeuristicRerankWeights) -> Self {
        self.weights = weights;
        self
    }

    fn tokenize(text: &str) -> HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect()
    }

    /// Fraction of query terms found in the candidate content.
    fn term_overlap(query_terms: &HashSet<String>, content: &str) -> f64 {
        if query_terms.is_empty() {
            return 0.0;
        }
        let content_terms = Self::tokenize(content);
        let hits = query_terms
            .iter()
            .filter(|t| content_terms.contains(*t))
            .count();
        hits as f64 / query_terms.len() as f64
    }

    /// Graph proximity: fraction of the other candidates reachable from this
    /// candidate within 2 hops. Candidates absent from the graph score 0.
    async fn graph_proximity(&self, key: &str, other_keys: &HashSet<String>) -> f64 {
        let Some(ref kg) = self.knowledge_graph else {
            return 0.0;
        };
        if other_keys.is_empty() {
            return 0.0;
        }
        let related = match kg.read().await.find_related_memories(key, 2, None).await {
            Ok(related) => related,
            Err(e) => {
                // The candidate may simply not be a graph node yet.
                tracing::debug!(key = %key, error = %e, "graph proximity unavailable for candidate");
                return 0.0;
            }
        };
        let hits = related
            .iter()
            .filter(|r| other_keys.contains(&r.memory_key))
            .count();
        (hits as f64 / other_keys.len() as f64).clamp(0.0, 1.0)
    }

    /// Recency via exponential decay: `0.5 ^ (age_hours / half_life)`.
    fn recency(
        created_at: chrono::DateTime<chrono::Utc>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> f64 {
        let age_hours = (now - created_at).num_seconds().max(0) as f64 / 3600.0;
        0.5_f64.powf(age_hours / RECENCY_HALF_LIFE_HOURS)
    }
}

#[async_trait]
impl Reranker for HeuristicReranker {
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<ScoredMemory>,
    ) -> Result<Vec<ScoredMemory>> {
        if candidates.len() < 2 {
            return Ok(candidates);
        }

        let query_terms = Self::tokenize(query);
        let now = chrono::Utc::now();

        // Embed the query once; embedding agreement is only computed when a
        // provider is configured and available.
        let query_embedding = match self.embedding_provider {
            Some(ref provider) if provider.is_available() => {
                Some(provider.embed(query, None).await?)
            }
            _ => None,
        };

        // Renormalize weights over the available features.
        let w = self.weights;
        let embedding_weight = if query_embedding.is_some() {
            w.embedding_agreement
        } else {
            0.0
        };
        let graph_weight = if self.knowledge_graph.is_some() {
            w.graph_proximity
        } else {
            0.0
        };
        let total_weight = w.term_overlap + embedding_weight + graph_weight + w.recency;
        if total_weight <= f64::EPSILON {
            return Ok(candidates);
        }

        let all_keys: HashSet<String> = candidates
            .iter()
            .map(|c| c.memory.entry.key.clone())
            .collect();

        let mut ranked: Vec<(f64, ScoredMemory)> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let content = &candidate.memory.entry.value;
            let key = &candidate.memory.entry.key;

            let overlap = Self::term_overlap(&query_terms, content);

            let agreement = match query_embedding {
                Some(ref qe) => {
                    let provider = self
                        .embedding_provider
                        .as_ref()
                        .expect("query_embedding is Some only when a provider is configured");
                    let ce = provider.embed(content, None).await?;
                    qe.cosine_similarity(&ce).clamp(0.0, 1.0)
                }
                None => 0.0,
            };

            let mut other_keys = all_keys.clone();
            other_keys.remove(key);
            let proximity = self.graph_proximity(key, &other_keys).await;

            let recency = Self::recency(candidate.memory.entry.created_at(), now);

            let rerank_score = (w.term_overlap * overlap
                + embedding_weight * agreement
                + graph_weight * proximity
                + w.recency * recency)
                / total_weight;

            tracing::trace!(
                key = %key,
                overlap,
                agreement,
                proximity,
                recency,
                rerank_score,
                "heuristic rerank features"
            );
            ranked.push((rerank_score, candidate));
        }

        // Deterministic ordering: rerank score descending, ties broken by
        // ascending memory key. Original scores are left untouched.
        ranked.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.memory.entry.key.cmp(&b.1.memory.entry.key))
        });

        Ok(ranked.into_iter().map(|(_, c)| c).collect())
    }

    fn name(&self) -> &str {
        "HeuristicReranker"
    }
}
