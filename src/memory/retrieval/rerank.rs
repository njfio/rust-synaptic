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
    /// Weights chosen by a measured LoCoMo sweep (see docs/evaluation.md,
    /// "reranker weighting"): embedding-dominant, with `graph_proximity`
    /// dropped. `graph_proximity` rewards a candidate's closeness to the OTHER
    /// candidates (query-agnostic), which mildly demoted query-relevant gold;
    /// leaning on query↔candidate semantic agreement lifted full-set recall@10
    /// 0.5565→0.5691 (MultiHop +9.6%, OpenDomain +12%). The peak is genuine
    /// (embed 0.80 scored lower than 0.60). Tuned on one dataset — override per
    /// deployment via `SYNAPTIC_RERANK_W_*`. `recency` is retained for real
    /// deployments (it is inert in the eval, where all memories share an
    /// ingest time).
    fn default() -> Self {
        Self {
            term_overlap: 0.25,
            embedding_agreement: 0.60,
            graph_proximity: 0.0,
            recency: 0.15,
        }
    }
}

impl HeuristicRerankWeights {
    /// Environment variable overriding [`Self::term_overlap`].
    pub const ENV_W_TERM: &'static str = "SYNAPTIC_RERANK_W_TERM";
    /// Environment variable overriding [`Self::embedding_agreement`].
    pub const ENV_W_EMBED: &'static str = "SYNAPTIC_RERANK_W_EMBED";
    /// Environment variable overriding [`Self::graph_proximity`].
    pub const ENV_W_GRAPH: &'static str = "SYNAPTIC_RERANK_W_GRAPH";
    /// Environment variable overriding [`Self::recency`].
    pub const ENV_W_RECENCY: &'static str = "SYNAPTIC_RERANK_W_RECENCY";

    /// Build weights from environment variables, falling back to the
    /// default for any field whose variable is absent, empty, or fails to
    /// parse as an `f64`. Negative values are clamped to `0.0`. This lets an
    /// experiment override a subset of the weights without recompiling.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        let read = |var: &str, default: f64| {
            std::env::var(var)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .map(|v| v.max(0.0))
                .unwrap_or(default)
        };
        Self {
            term_overlap: read(Self::ENV_W_TERM, defaults.term_overlap),
            embedding_agreement: read(Self::ENV_W_EMBED, defaults.embedding_agreement),
            graph_proximity: read(Self::ENV_W_GRAPH, defaults.graph_proximity),
            recency: read(Self::ENV_W_RECENCY, defaults.recency),
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

    /// Graph proximity for ALL candidates in one pass: fraction of the other
    /// candidates reachable from each candidate within 2 hops. Candidates
    /// absent from the graph score 0. The knowledge-graph read lock is taken
    /// ONCE for the whole batch and released before this returns, so it is
    /// never held across the embedding awaits of the main rerank loop.
    async fn graph_proximities(
        &self,
        candidates: &[ScoredMemory],
        all_keys: &HashSet<String>,
    ) -> std::collections::HashMap<String, f64> {
        let mut proximities = std::collections::HashMap::with_capacity(candidates.len());
        let Some(ref kg) = self.knowledge_graph else {
            return proximities;
        };
        #[cfg(feature = "test-utils")]
        crate::memory::retrieval::telemetry::record_kg_read_lock();
        let kg_guard = kg.read().await;
        // ONE batched graph pass for every candidate's 2-hop related set,
        // instead of a full BFS per candidate.
        let keys: Vec<String> = candidates
            .iter()
            .map(|c| c.memory.entry.key.clone())
            .collect();
        let related_map = kg_guard.find_related_memories_batch(&keys, 2, None);
        drop(kg_guard);
        for candidate in candidates {
            let key = &candidate.memory.entry.key;
            let other_count = all_keys.len().saturating_sub(1);
            if other_count == 0 {
                continue;
            }
            let Some(related) = related_map.get(key) else {
                // The candidate may simply not be a graph node yet.
                tracing::debug!(key = %key, "graph proximity unavailable for candidate");
                continue;
            };
            let hits = related
                .iter()
                .filter(|r| r.memory_key != *key && all_keys.contains(&r.memory_key))
                .count();
            proximities.insert(
                key.clone(),
                (hits as f64 / other_count as f64).clamp(0.0, 1.0),
            );
        }
        proximities
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
                Some(provider.embed_for_scoring(query, None).await?)
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

        // Batch the knowledge-graph reads under a single, short-lived read
        // lock, released before any embedding awaits below.
        let proximities = self.graph_proximities(&candidates, &all_keys).await;

        // Embed ALL candidate contents in ONE batched call on the read-only,
        // content-hash-cached scoring path: cache hits reuse the embeddings
        // the dense retriever computed when the provider instance is shared,
        // and the misses go through a single batched forward pass instead of
        // one per candidate.
        let candidate_embeddings = match query_embedding {
            Some(_) => {
                let provider = self
                    .embedding_provider
                    .as_ref()
                    .expect("query_embedding is Some only when a provider is configured");
                let contents: Vec<&str> = candidates
                    .iter()
                    .map(|c| c.memory.entry.value.as_str())
                    .collect();
                Some(provider.embed_for_scoring_batch(&contents).await?)
            }
            None => None,
        };

        let mut ranked: Vec<(f64, ScoredMemory)> = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.into_iter().enumerate() {
            let content = &candidate.memory.entry.value;
            let key = &candidate.memory.entry.key;

            let overlap = Self::term_overlap(&query_terms, content);

            let agreement = match (&query_embedding, &candidate_embeddings) {
                (Some(qe), Some(embeddings)) => {
                    let ce = embeddings.get(index).ok_or_else(|| {
                        crate::error::MemoryError::processing_error(
                            "reranker batch returned fewer embeddings than candidates \
                             (invariant violation)",
                        )
                    })?;
                    qe.cosine_similarity(ce).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };

            let proximity = proximities.get(key).copied().unwrap_or(0.0);

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

#[cfg(feature = "reranker-model")]
pub use cross_encoder::CrossEncoderReranker;

/// Candle-backed BERT cross-encoder reranker (feature `reranker-model`).
#[cfg(feature = "reranker-model")]
mod cross_encoder {
    use super::Reranker;
    use crate::error::{MemoryError, Result};
    use crate::memory::retrieval::pipeline::ScoredMemory;
    use async_trait::async_trait;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{linear, Linear, Module, VarBuilder};
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use std::path::Path;
    use tokenizers::Tokenizer;

    /// A cross-encoder reranker: for each candidate it tokenizes the
    /// `(query, candidate content)` pair, runs a BERT forward pass, and reads
    /// a single relevance logit off a linear classification head over the
    /// `[CLS]` position. Candidates are re-ordered by that logit, descending,
    /// with exact ties broken by ascending memory key. Original pipeline
    /// scores travel through unchanged.
    ///
    /// Construction fails closed: if the model files (config, tokenizer,
    /// weights) cannot be loaded, [`CrossEncoderReranker::new`] returns an
    /// error — there is no fallback scoring path in this type.
    pub struct CrossEncoderReranker {
        model: BertModel,
        classifier: Linear,
        tokenizer: Tokenizer,
        device: Device,
        max_length: usize,
    }

    impl CrossEncoderReranker {
        /// Load a cross-encoder from a local model directory containing
        /// `config.json`, `tokenizer.json` and `model.safetensors` (the
        /// standard HuggingFace layout for BERT sequence-classification
        /// cross-encoders, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`).
        ///
        /// Fails closed with an error if any file is missing or malformed.
        /// Inference runs on CPU.
        pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
            let model_dir = model_dir.as_ref();
            let device = Device::Cpu;

            let config_path = model_dir.join("config.json");
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                MemoryError::configuration(format!(
                    "cross-encoder config not readable at {}: {}",
                    config_path.display(),
                    e
                ))
            })?;
            let config: BertConfig = serde_json::from_str(&config_str).map_err(|e| {
                MemoryError::configuration(format!(
                    "cross-encoder config at {} is not a valid BERT config: {}",
                    config_path.display(),
                    e
                ))
            })?;

            let tokenizer_path = model_dir.join("tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                MemoryError::configuration(format!(
                    "cross-encoder tokenizer not loadable from {}: {}",
                    tokenizer_path.display(),
                    e
                ))
            })?;

            let weights_path = model_dir.join("model.safetensors");
            let weights = std::fs::read(&weights_path).map_err(|e| {
                MemoryError::configuration(format!(
                    "cross-encoder weights not readable at {}: {}",
                    weights_path.display(),
                    e
                ))
            })?;
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;

            let max_length = config.max_position_embeddings;
            Self::from_var_builder(vb, &config, tokenizer, max_length)
        }

        /// Build a cross-encoder from an already-constructed [`VarBuilder`]
        /// and tokenizer. This is the shared trunk of [`Self::new`]; it also
        /// lets tests exercise the real inference path from in-memory
        /// weights without touching the network or filesystem.
        pub fn from_var_builder(
            vb: VarBuilder,
            config: &BertConfig,
            tokenizer: Tokenizer,
            max_length: usize,
        ) -> Result<Self> {
            if max_length == 0 || max_length > config.max_position_embeddings {
                return Err(MemoryError::configuration(format!(
                    "cross-encoder max_length {} must be in 1..={}",
                    max_length, config.max_position_embeddings
                )));
            }
            let device = vb.device().clone();
            // BertModel::load itself falls back to the `bert.` prefix used by
            // sequence-classification checkpoints.
            let model = BertModel::load(vb.clone(), config)?;
            let classifier = linear(config.hidden_size, 1, vb.pp("classifier"))?;
            Ok(Self {
                model,
                classifier,
                tokenizer,
                device,
                max_length,
            })
        }

        /// Relevance logit for a `(query, candidate)` pair: tokenize the
        /// pair, run the BERT forward pass, apply the classification head to
        /// the `[CLS]` position. Higher means more relevant.
        pub fn score(&self, query: &str, candidate: &str) -> Result<f32> {
            let encoding = self
                .tokenizer
                .encode((query, candidate), true)
                .map_err(|e| {
                    MemoryError::processing_error(format!(
                        "cross-encoder tokenization failed: {}",
                        e
                    ))
                })?;

            let take = encoding.get_ids().len().min(self.max_length);
            if take == 0 {
                return Err(MemoryError::processing_error(
                    "cross-encoder tokenization produced no tokens",
                ));
            }
            let ids = &encoding.get_ids()[..take];
            let type_ids = &encoding.get_type_ids()[..take];
            let mask = &encoding.get_attention_mask()[..take];

            let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
            let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;
            let attention_mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;

            let hidden = self
                .model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            // [CLS] position: (batch=1, hidden) -> classifier -> (1, 1).
            let cls = hidden.i((.., 0))?;
            let logit = self.classifier.forward(&cls)?;
            Ok(logit.flatten_all()?.to_vec1::<f32>()?[0])
        }
    }

    #[async_trait]
    impl Reranker for CrossEncoderReranker {
        async fn rerank(
            &self,
            query: &str,
            candidates: Vec<ScoredMemory>,
        ) -> Result<Vec<ScoredMemory>> {
            if candidates.len() < 2 {
                return Ok(candidates);
            }

            let mut ranked: Vec<(f32, ScoredMemory)> = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                let logit = self.score(query, &candidate.memory.entry.value)?;
                tracing::trace!(
                    key = %candidate.memory.entry.key,
                    logit,
                    "cross-encoder rerank logit"
                );
                ranked.push((logit, candidate));
            }

            // Logit descending; exact ties broken by ascending memory key so
            // the ordering is stable and deterministic.
            ranked.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.memory.entry.key.cmp(&b.1.memory.entry.key))
            });

            Ok(ranked.into_iter().map(|(_, c)| c).collect())
        }

        fn name(&self) -> &str {
            "CrossEncoderReranker"
        }
    }
}

#[cfg(test)]
mod weights_env_tests {
    use super::HeuristicRerankWeights;
    use std::sync::Mutex;

    /// Guards the process-global `SYNAPTIC_RERANK_W_*` env vars so tests
    /// that set/unset them don't race with each other under parallel test
    /// execution.
    static ENV_GUARD: Mutex<()> = Mutex::new(());

    fn clear_env() {
        std::env::remove_var(HeuristicRerankWeights::ENV_W_TERM);
        std::env::remove_var(HeuristicRerankWeights::ENV_W_EMBED);
        std::env::remove_var(HeuristicRerankWeights::ENV_W_GRAPH);
        std::env::remove_var(HeuristicRerankWeights::ENV_W_RECENCY);
    }

    #[test]
    fn from_env_with_no_vars_matches_default() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();

        let env_weights = HeuristicRerankWeights::from_env();
        let default_weights = HeuristicRerankWeights::default();

        assert_eq!(env_weights.term_overlap, default_weights.term_overlap);
        assert_eq!(
            env_weights.embedding_agreement,
            default_weights.embedding_agreement
        );
        assert_eq!(env_weights.graph_proximity, default_weights.graph_proximity);
        assert_eq!(env_weights.recency, default_weights.recency);
    }

    #[test]
    fn from_env_overrides_only_set_vars_and_clamps_and_falls_back() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();

        std::env::set_var(HeuristicRerankWeights::ENV_W_GRAPH, "0");
        std::env::set_var(HeuristicRerankWeights::ENV_W_EMBED, "0.6");
        std::env::set_var(HeuristicRerankWeights::ENV_W_TERM, "abc");
        std::env::set_var(HeuristicRerankWeights::ENV_W_RECENCY, "-0.5");

        let weights = HeuristicRerankWeights::from_env();
        let default_weights = HeuristicRerankWeights::default();

        // Overridden.
        assert_eq!(weights.graph_proximity, 0.0);
        assert_eq!(weights.embedding_agreement, 0.6);
        // Invalid value falls back to the field's default.
        assert_eq!(weights.term_overlap, default_weights.term_overlap);
        // Negative value is clamped to 0.0, not the default.
        assert_eq!(weights.recency, 0.0);

        clear_env();
    }
}
