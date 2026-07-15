//! Pluggable retrieval pipeline for hybrid semantic search
//!
//! This module provides a flexible retrieval pipeline that combines multiple
//! signals (embeddings, keywords, graph relationships, temporal relevance) to
//! deliver high-quality search results.

use crate::error::Result;
use crate::memory::retrieval::rerank::Reranker;
use crate::memory::temporal::decay_models::{
    DecayConfig, DecayContext, DecayModelType, TemporalDecayModels,
};
use crate::memory::types::MemoryFragment;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for pluggable retrieval strategies
///
/// Implementors provide different retrieval mechanisms that can be
/// composed in a hybrid pipeline for optimal search quality.
#[async_trait]
pub trait RetrievalPipeline: Send + Sync {
    /// Search for relevant memories
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `limit` - Maximum number of results to return
    /// * `config` - Optional configuration for this search
    ///
    /// # Returns
    /// * Vector of memory fragments with relevance scores
    async fn search(
        &self,
        query: &str,
        limit: usize,
        config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>>;

    /// Get the name of this retrieval strategy
    fn name(&self) -> &'static str;

    /// Get the signal type this pipeline produces
    fn signal_type(&self) -> RetrievalSignal;

    /// Check if this pipeline is available/initialized
    fn is_available(&self) -> bool {
        true
    }
}

/// Types of retrieval signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetrievalSignal {
    /// Dense vector embeddings (semantic similarity)
    DenseVector,
    /// Sparse keyword matching (BM25, TF-IDF)
    SparseKeyword,
    /// Graph-based relationships
    GraphRelationship,
    /// Temporal relevance (recency, access patterns)
    TemporalRelevance,
    /// Hybrid combination of multiple signals
    Hybrid,
}

/// Memory with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    /// The memory fragment
    pub memory: MemoryFragment,
    /// Relevance score (0.0 to 1.0)
    pub score: f64,
    /// Signal type that produced this score
    pub signal: RetrievalSignal,
    /// Optional explanation of why this memory was retrieved
    pub explanation: Option<String>,
}

impl ScoredMemory {
    pub fn new(memory: MemoryFragment, score: f64, signal: RetrievalSignal) -> Self {
        Self {
            memory,
            score,
            signal,
            explanation: None,
        }
    }

    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }
}

/// Weights for the post-fusion composite score.
///
/// After fusion, each result's final ordering score is
/// `composite = relevance_weight · norm(fused_score)
///            + recency_weight · recency_decay(age)
///            + importance_weight · importance`
/// where `norm` min-max normalizes the fused score to `[0, 1]` across the
/// result set, `recency_decay` is the Ebbinghaus retention curve from
/// `memory::temporal::decay_models` evaluated at the memory's age, and
/// `importance` is `MemoryEntry.metadata.importance` (already `[0, 1]`).
///
/// Defaults: relevance 0.6, recency 0.2, importance 0.2.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompositeWeights {
    /// Weight of the normalized fused relevance score
    pub relevance: f64,
    /// Weight of the recency decay term
    pub recency: f64,
    /// Weight of the memory's stored importance
    pub importance: f64,
}

impl Default for CompositeWeights {
    fn default() -> Self {
        Self {
            relevance: 0.6,
            recency: 0.2,
            importance: 0.2,
        }
    }
}

/// Configuration for the retrieval pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Maximum number of results per signal
    pub max_per_signal: usize,
    /// Minimum score threshold (0.0 to 1.0)
    pub min_score: f64,
    /// Enable result fusion
    pub enable_fusion: bool,
    /// Fusion strategy to use
    pub fusion_strategy: FusionStrategy,
    /// Signal weights for fusion
    pub signal_weights: HashMap<RetrievalSignal, f64>,
    /// Enable score caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Weights for the post-fusion composite (relevance × recency ×
    /// importance) score. Defaults to 0.6 / 0.2 / 0.2.
    #[serde(default)]
    pub composite_weights: CompositeWeights,
    /// Enable the deterministic query-understanding stage (temporal-
    /// constraint extraction + multi-part splitting). Default `true`; set
    /// to `false` for ablation baselines. Simple queries (no temporal
    /// expression, single part) are a no-op passthrough either way.
    #[serde(default = "default_enable_query_understanding")]
    pub enable_query_understanding: bool,
    /// Multiplicative boost applied to a memory's fused score when it
    /// matches an extracted temporal constraint:
    /// `score *= 1.0 + temporal_boost`. Default `0.3`.
    #[serde(default = "default_temporal_boost")]
    pub temporal_boost: f64,
}

fn default_enable_query_understanding() -> bool {
    true
}

fn default_temporal_boost() -> f64 {
    0.3
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut signal_weights = HashMap::new();
        signal_weights.insert(RetrievalSignal::DenseVector, 0.4);
        signal_weights.insert(RetrievalSignal::SparseKeyword, 0.3);
        signal_weights.insert(RetrievalSignal::GraphRelationship, 0.2);
        signal_weights.insert(RetrievalSignal::TemporalRelevance, 0.1);

        Self {
            max_per_signal: 50,
            min_score: 0.1,
            enable_fusion: true,
            fusion_strategy: FusionStrategy::ReciprocRankFusion,
            signal_weights,
            enable_caching: true,
            cache_ttl_seconds: 300,
            composite_weights: CompositeWeights::default(),
            enable_query_understanding: default_enable_query_understanding(),
            temporal_boost: default_temporal_boost(),
        }
    }
}

impl PipelineConfig {
    /// Create a configuration optimized for semantic search
    pub fn semantic_focus() -> Self {
        let mut config = Self::default();
        config
            .signal_weights
            .insert(RetrievalSignal::DenseVector, 0.6);
        config
            .signal_weights
            .insert(RetrievalSignal::SparseKeyword, 0.2);
        config
            .signal_weights
            .insert(RetrievalSignal::GraphRelationship, 0.15);
        config
            .signal_weights
            .insert(RetrievalSignal::TemporalRelevance, 0.05);
        config
    }

    /// Create a configuration optimized for keyword search
    pub fn keyword_focus() -> Self {
        let mut config = Self::default();
        config
            .signal_weights
            .insert(RetrievalSignal::DenseVector, 0.2);
        config
            .signal_weights
            .insert(RetrievalSignal::SparseKeyword, 0.6);
        config
            .signal_weights
            .insert(RetrievalSignal::GraphRelationship, 0.15);
        config
            .signal_weights
            .insert(RetrievalSignal::TemporalRelevance, 0.05);
        config
    }

    /// Create a configuration optimized for graph-based discovery
    pub fn graph_focus() -> Self {
        let mut config = Self::default();
        config
            .signal_weights
            .insert(RetrievalSignal::DenseVector, 0.25);
        config
            .signal_weights
            .insert(RetrievalSignal::SparseKeyword, 0.25);
        config
            .signal_weights
            .insert(RetrievalSignal::GraphRelationship, 0.4);
        config
            .signal_weights
            .insert(RetrievalSignal::TemporalRelevance, 0.1);
        config
    }

    /// Create a configuration optimized for recent/relevant content
    pub fn temporal_focus() -> Self {
        let mut config = Self::default();
        config
            .signal_weights
            .insert(RetrievalSignal::DenseVector, 0.3);
        config
            .signal_weights
            .insert(RetrievalSignal::SparseKeyword, 0.2);
        config
            .signal_weights
            .insert(RetrievalSignal::GraphRelationship, 0.1);
        config
            .signal_weights
            .insert(RetrievalSignal::TemporalRelevance, 0.4);
        config
    }

    /// Builder method to set signal weight
    pub fn with_signal_weight(mut self, signal: RetrievalSignal, weight: f64) -> Self {
        self.signal_weights.insert(signal, weight);
        self
    }

    /// Builder method to set fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Builder method to set minimum score threshold
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Builder method to set composite scoring weights
    pub fn with_composite_weights(mut self, weights: CompositeWeights) -> Self {
        self.composite_weights = weights;
        self
    }
}

/// Strategies for fusing results from multiple signals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (RRF)
    /// Score = sum(1 / (rank + k)) where k=60 by default
    ReciprocRankFusion,
    /// Weighted average of normalized scores
    WeightedAverage,
    /// Maximum score across all signals
    MaxScore,
    /// Borda count voting
    BordaCount,
}

/// Hybrid retriever combining multiple signals
pub struct HybridRetriever {
    /// Individual retrieval pipelines
    pipelines: Vec<Arc<dyn RetrievalPipeline>>,
    /// Configuration
    config: PipelineConfig,
    /// Score cache
    cache: Option<Arc<tokio::sync::RwLock<ScoreCache>>>,
    /// Optional reranking stage applied to the top-K after composite scoring
    reranker: Option<Arc<dyn Reranker>>,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new(config: PipelineConfig) -> Self {
        let cache = if config.enable_caching {
            Some(Arc::new(tokio::sync::RwLock::new(ScoreCache::new(
                config.cache_ttl_seconds,
            ))))
        } else {
            None
        };

        Self {
            pipelines: Vec::new(),
            config,
            cache,
            reranker: None,
        }
    }

    /// Set an optional reranking stage, applied to the top-K results after
    /// composite scoring and before the final results are returned. Default:
    /// no reranker (pipeline behaviour unchanged).
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Add a retrieval pipeline to the hybrid retriever
    pub fn add_pipeline(mut self, pipeline: Arc<dyn RetrievalPipeline>) -> Self {
        self.pipelines.push(pipeline);
        self
    }

    /// Search using all available pipelines and fuse results
    ///
    /// A deterministic query-understanding stage
    /// (see [`crate::memory::retrieval::QueryPlan`]) runs first when
    /// `config.enable_query_understanding` is set (the default):
    /// multi-part questions are split into sub-queries whose fused results
    /// are unioned (dedup, best score wins), and an extracted temporal
    /// constraint multiplicatively boosts matching memories' fused scores
    /// before composite scoring. Simple queries are a no-op passthrough.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            num_pipelines = self.pipelines.len(),
            "Starting hybrid search"
        );

        // Check cache first
        if let Some(ref cache) = self.cache {
            let cache_key = format!("{}:{}", query, limit);
            if let Some(cached_results) = cache.read().await.get(&cache_key) {
                tracing::debug!("Cache hit for query");
                return Ok(cached_results.clone());
            }
        }

        // Query understanding: temporal-constraint extraction + multi-part
        // splitting. Disabled (or a simple query) degenerates to the
        // original single-query path.
        let plan = if self.config.enable_query_understanding {
            super::query_understanding::QueryPlan::analyze(query)
        } else {
            super::query_understanding::QueryPlan::passthrough(query)
        };

        let fused = if plan.is_passthrough() {
            self.collect_and_fuse(query, limit).await?
        } else {
            tracing::debug!(
                sub_queries = plan.sub_queries.len(),
                temporal = plan.temporal.is_some(),
                "Query understanding applied"
            );

            // Run each sub-query through the full retrieval + fusion path
            // and union the results, keeping each memory's best fused score.
            let mut best: HashMap<String, (MemoryFragment, f64)> = HashMap::new();
            for sub_query in &plan.sub_queries {
                for (memory, score) in self.collect_and_fuse(sub_query, limit).await? {
                    match best.entry(memory.entry.key.clone()) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if score > e.get().1 {
                                e.get_mut().1 = score;
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert((memory, score));
                        }
                    }
                }
            }
            let mut unioned: Vec<(MemoryFragment, f64)> = best.into_values().collect();

            // Temporal boost: memories matching the extracted constraint
            // (created within the range, or content mentioning a
            // constrained year) get a multiplicative bump on their fused
            // score BEFORE composite normalization.
            if let Some(ref temporal) = plan.temporal {
                let boost = 1.0 + self.config.temporal_boost.max(0.0);
                for (memory, score) in &mut unioned {
                    if temporal.matches(&memory.entry) {
                        *score *= boost;
                    }
                }
            }

            // Deterministic ordering: score descending, then key ascending.
            unioned.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.entry.key.cmp(&b.0.entry.key))
            });
            unioned.truncate(limit);
            unioned
        };

        // Apply the composite relevance × recency × importance score
        // (see `CompositeWeights`).
        let composite = self.apply_composite_scoring(fused).await?;

        // Optional reranking stage over the top-K: reorders by cross-features
        // computed against the query; preserves scores and candidates.
        let fused_results: Vec<MemoryFragment> = match self.reranker {
            Some(ref reranker) => {
                let reranked = reranker.rerank(query, composite).await?;
                tracing::debug!(reranker = reranker.name(), "Rerank stage applied");
                reranked.into_iter().map(|s| s.memory).collect()
            }
            None => composite.into_iter().map(|s| s.memory).collect(),
        };

        tracing::info!(final_count = fused_results.len(), "Hybrid search completed");

        // Cache results
        if let Some(ref cache) = self.cache {
            let cache_key = format!("{}:{}", query, limit);
            cache.write().await.insert(cache_key, fused_results.clone());
        }

        Ok(fused_results)
    }

    /// Run one query through every available pipeline and fuse the results.
    ///
    /// This is the single-query core of `search`, factored out so the
    /// query-understanding stage can run it once per sub-query and union
    /// the fused outputs.
    async fn collect_and_fuse(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryFragment, f64)>> {
        // Available pipelines run CONCURRENTLY. `join_all` yields results in
        // input order, so they are re-assembled in pipeline registration
        // order and fusion input is deterministic regardless of completion
        // order.
        let searches = self
            .pipelines
            .iter()
            .filter(|pipeline| {
                let available = pipeline.is_available();
                if !available {
                    tracing::debug!(pipeline = pipeline.name(), "Skipping unavailable pipeline");
                }
                available
            })
            .map(|pipeline| async move {
                let outcome = pipeline
                    .search(query, self.config.max_per_signal, Some(&self.config))
                    .await;
                (pipeline.name(), outcome)
            });

        let mut all_results: Vec<Vec<ScoredMemory>> = Vec::new();
        for (name, outcome) in futures::future::join_all(searches).await {
            match outcome {
                Ok(results) => {
                    tracing::debug!(
                        pipeline = name,
                        result_count = results.len(),
                        "Pipeline search completed"
                    );
                    all_results.push(results);
                }
                Err(e) => {
                    tracing::warn!(
                        pipeline = name,
                        error = %e,
                        "Pipeline search failed"
                    );
                    // Continue with other pipelines
                }
            }
        }

        self.fuse_results(all_results, limit)
    }

    /// Fuse results from multiple pipelines using the configured strategy.
    ///
    /// Returns each fused memory together with its fused score so the
    /// post-fusion composite scoring stage can normalize relevance across
    /// the result set.
    fn fuse_results(
        &self,
        all_results: Vec<Vec<ScoredMemory>>,
        limit: usize,
    ) -> Result<Vec<(MemoryFragment, f64)>> {
        if all_results.is_empty() {
            return Ok(Vec::new());
        }

        // Rank-based Reciprocal Rank Fusion needs each retriever's ordered
        // result list (not just per-item scores), so it is handled separately.
        if self.config.fusion_strategy == FusionStrategy::ReciprocRankFusion {
            return self.fuse_reciprocal_rank(all_results, limit);
        }

        // Collect all unique memories
        let mut memory_scores: HashMap<String, (MemoryFragment, Vec<(RetrievalSignal, f64)>)> =
            HashMap::new();

        for results in all_results {
            for scored in results {
                let entry = memory_scores
                    .entry(scored.memory.entry.key.clone())
                    .or_insert_with(|| (scored.memory.clone(), Vec::new()));

                entry.1.push((scored.signal, scored.score));
            }
        }

        // Apply fusion strategy
        let mut fused: Vec<(MemoryFragment, f64)> = memory_scores
            .into_iter()
            .map(|(_key, (memory, signal_scores))| {
                let combined_score = self.combine_scores(&signal_scores);
                (memory, combined_score)
            })
            .filter(|(_, score)| *score >= self.config.min_score)
            .collect();

        // Sort by score descending
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top limit results
        Ok(fused.into_iter().take(limit).collect())
    }

    /// Fuse results with rank-based Reciprocal Rank Fusion.
    ///
    /// For each retriever's result list (ordered by that retriever's score,
    /// descending), an item at 0-based rank `r` contributes `1 / (k + r + 1)`
    /// with the standard `k = 60`; contributions are summed per item across
    /// retrievers. RRF scores are inherently small (at most `n / (k + 1)` for
    /// `n` retrievers), so `min_score` is applied to each retriever's raw
    /// signal scores before ranking rather than to the fused score.
    ///
    /// Each contribution is multiplied by the configured `signal_weights`
    /// entry for the retriever's signal (default 1.0 when unset), so
    /// e.g. `semantic_focus` genuinely favors dense-vector votes over
    /// temporal votes instead of every signal voting with equal power.
    fn fuse_reciprocal_rank(
        &self,
        all_results: Vec<Vec<ScoredMemory>>,
        limit: usize,
    ) -> Result<Vec<(MemoryFragment, f64)>> {
        const RRF_K: f64 = 60.0;

        // Value: (memory, rrf_score, weighted_raw_score_sum). The weighted raw
        // score sum breaks RRF ties deterministically (e.g. two items that
        // swap ranks #1/#2 across two retrievers have identical RRF scores).
        let mut fused_scores: HashMap<String, (MemoryFragment, f64, f64)> = HashMap::new();

        for mut results in all_results {
            // Filter low-quality candidates on their raw signal scores, then
            // derive each item's rank by ordering the retriever's output.
            results.retain(|scored| scored.score >= self.config.min_score);
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (rank, scored) in results.into_iter().enumerate() {
                let weight = self
                    .config
                    .signal_weights
                    .get(&scored.signal)
                    .copied()
                    .unwrap_or(1.0);
                let contribution = weight / (RRF_K + rank as f64 + 1.0);
                let entry = fused_scores
                    .entry(scored.memory.entry.key.clone())
                    .or_insert_with(|| (scored.memory.clone(), 0.0, 0.0));
                entry.1 += contribution;
                entry.2 += scored.score * weight;
            }
        }

        let mut fused: Vec<(MemoryFragment, f64, f64)> = fused_scores.into_values().collect();
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });

        Ok(fused
            .into_iter()
            .take(limit)
            .map(|(memory, rrf, _raw)| (memory, rrf))
            .collect())
    }

    /// Apply the composite relevance × recency × importance score after
    /// fusion and re-sort the result set.
    ///
    /// `composite = wr · norm(fused) + wc · recency_decay(age) + wi · importance`
    ///
    /// - `norm` min-max normalizes fused scores across the result set (a
    ///   uniform set normalizes to 1.0 so relevance does not discriminate);
    /// - `recency_decay` is the Ebbinghaus retention curve from
    ///   `memory::temporal::decay_models`, evaluated at the memory's age
    ///   (hours since creation) with a neutral context so age alone drives
    ///   the term — importance is credited once, by its own weight;
    /// - `importance` is `MemoryEntry.metadata.importance` (`[0, 1]`).
    ///
    /// The sort is stable, so exact composite ties preserve fused order.
    ///
    /// Returns `ScoredMemory` (score = composite, signal = `Hybrid`) so the
    /// optional reranking stage can consume the top-K with scores intact.
    async fn apply_composite_scoring(
        &self,
        fused: Vec<(MemoryFragment, f64)>,
    ) -> Result<Vec<ScoredMemory>> {
        if fused.is_empty() {
            return Ok(Vec::new());
        }

        let weights = self.config.composite_weights;

        let min = fused.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
        let max = fused
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        // Deterministic Ebbinghaus decay: adaptivity disabled, neutral
        // context, so retention depends only on age and the base half-life.
        let mut decay_models = TemporalDecayModels::new(DecayConfig {
            adaptive_enabled: false,
            ..DecayConfig::default()
        })?;
        let neutral_context = DecayContext {
            importance: 0.0,
            access_frequency: 0.0,
            hours_since_access: 0.0,
            complexity: 0.0,
            emotional_weight: 0.0,
            contextual_relevance: 0.0,
            engagement_level: 0.0,
        };

        let now = chrono::Utc::now();
        let mut composite: Vec<(MemoryFragment, f64)> = Vec::with_capacity(fused.len());
        for (memory, fused_score) in fused {
            let relevance = if range > f64::EPSILON {
                (fused_score - min) / range
            } else {
                1.0
            };
            let age_hours = (now - memory.entry.created_at()).num_seconds().max(0) as f64 / 3600.0;
            let recency = decay_models
                .calculate_decay(&DecayModelType::Ebbinghaus, age_hours, &neutral_context)
                .await?
                .retention_probability;
            let importance = memory.entry.metadata.importance.clamp(0.0, 1.0);

            let score = weights.relevance * relevance
                + weights.recency * recency
                + weights.importance * importance;
            composite.push((memory, score));
        }

        // Stable sort: composite ties keep the fused ordering.
        composite.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(composite
            .into_iter()
            .map(|(memory, score)| ScoredMemory::new(memory, score, RetrievalSignal::Hybrid))
            .collect())
    }

    /// Combine scores from multiple signals using the configured strategy
    fn combine_scores(&self, signal_scores: &[(RetrievalSignal, f64)]) -> f64 {
        match self.config.fusion_strategy {
            FusionStrategy::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (signal, score) in signal_scores {
                    let weight = self
                        .config
                        .signal_weights
                        .get(signal)
                        .copied()
                        .unwrap_or(1.0);
                    weighted_sum += score * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                }
            }
            FusionStrategy::MaxScore => signal_scores
                .iter()
                .map(|(_, score)| score)
                .fold(0.0, |max, &score| max.max(score)),
            FusionStrategy::ReciprocRankFusion => {
                // Rank-based RRF requires each retriever's ordered result list,
                // which is not visible here; `fuse_results` routes this strategy
                // through `fuse_reciprocal_rank` instead. As a per-item fallback
                // (e.g. direct calls), treat each contributing signal as a
                // rank-1 hit: `1 / (k + 1)` per signal, k = 60.
                let k = 60.0;
                signal_scores.iter().map(|_| 1.0 / (k + 1.0)).sum()
            }
            FusionStrategy::BordaCount => {
                // Borda count: assign points based on ranking
                // Higher ranked items get more points
                let max_score = signal_scores.len() as f64;
                signal_scores
                    .iter()
                    .map(|(_, score)| score * max_score)
                    .sum()
            }
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: PipelineConfig) {
        self.config = config;
    }

    /// Clear the cache
    pub async fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.write().await.clear();
        }
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> Option<CacheStats> {
        if let Some(ref cache) = self.cache {
            Some(cache.read().await.stats())
        } else {
            None
        }
    }
}

/// Simple score cache with TTL
struct ScoreCache {
    entries: HashMap<String, CachedResult>,
    ttl_seconds: u64,
}

struct CachedResult {
    results: Vec<MemoryFragment>,
    cached_at: std::time::Instant,
}

impl ScoreCache {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_seconds,
        }
    }

    fn get(&self, key: &str) -> Option<Vec<MemoryFragment>> {
        if let Some(cached) = self.entries.get(key) {
            let age = cached.cached_at.elapsed().as_secs();
            if age < self.ttl_seconds {
                return Some(cached.results.clone());
            }
        }
        None
    }

    fn insert(&mut self, key: String, results: Vec<MemoryFragment>) {
        self.entries.insert(
            key,
            CachedResult {
                results,
                cached_at: std::time::Instant::now(),
            },
        );

        // Simple cleanup: remove expired entries
        self.entries
            .retain(|_, cached| cached.cached_at.elapsed().as_secs() < self.ttl_seconds);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.entries.len(),
            ttl_seconds: self.ttl_seconds,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entry_count: usize,
    pub ttl_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_per_signal, 50);
        assert_eq!(config.min_score, 0.1);
        assert!(config.enable_fusion);
        assert_eq!(config.fusion_strategy, FusionStrategy::ReciprocRankFusion);
    }

    #[test]
    fn test_pipeline_config_semantic_focus() {
        let config = PipelineConfig::semantic_focus();
        let dense_weight = config
            .signal_weights
            .get(&RetrievalSignal::DenseVector)
            .expect("value should be available");
        assert_eq!(*dense_weight, 0.6);
    }

    #[test]
    fn test_scored_memory_creation() {
        let memory = MemoryFragment::new(
            crate::memory::types::MemoryEntry::new(
                "test".to_string(),
                "test content".to_string(),
                crate::memory::types::MemoryType::ShortTerm,
            ),
            0.8,
        );

        let scored = ScoredMemory::new(memory.clone(), 0.9, RetrievalSignal::DenseVector);
        assert_eq!(scored.score, 0.9);
        assert_eq!(scored.signal, RetrievalSignal::DenseVector);
        assert!(scored.explanation.is_none());

        let scored_with_exp = scored.with_explanation("High semantic similarity".to_string());
        assert!(scored_with_exp.explanation.is_some());
    }

    #[test]
    fn test_fusion_strategies() {
        let signal_scores = vec![
            (RetrievalSignal::DenseVector, 0.8),
            (RetrievalSignal::SparseKeyword, 0.6),
        ];

        // Test WeightedAverage
        let mut config = PipelineConfig::default();
        config.fusion_strategy = FusionStrategy::WeightedAverage;
        let hybrid = HybridRetriever::new(config);
        let score = hybrid.combine_scores(&signal_scores);
        assert!(score > 0.0 && score <= 1.0);

        // Test MaxScore
        let mut config = PipelineConfig::default();
        config.fusion_strategy = FusionStrategy::MaxScore;
        let hybrid = HybridRetriever::new(config);
        let score = hybrid.combine_scores(&signal_scores);
        assert_eq!(score, 0.8);
    }

    #[test]
    fn test_rrf_rank_based_fusion_prefers_multi_retriever_consensus() {
        fn fragment(key: &str) -> MemoryFragment {
            MemoryFragment::new(
                crate::memory::types::MemoryEntry::new(
                    key.to_string(),
                    format!("{key} content"),
                    crate::memory::types::MemoryType::ShortTerm,
                ),
                0.8,
            )
        }

        let hybrid = HybridRetriever::new(PipelineConfig::default());

        // Item "a" is ranked #1 by two retrievers; item "b" is ranked #1 by
        // only one retriever. Rank-based RRF must place "a" above "b".
        let all_results = vec![
            vec![
                ScoredMemory::new(fragment("a"), 0.9, RetrievalSignal::DenseVector),
                ScoredMemory::new(fragment("c"), 0.5, RetrievalSignal::DenseVector),
            ],
            vec![
                ScoredMemory::new(fragment("a"), 0.7, RetrievalSignal::SparseKeyword),
                ScoredMemory::new(fragment("c"), 0.4, RetrievalSignal::SparseKeyword),
            ],
            vec![ScoredMemory::new(
                fragment("b"),
                0.95,
                RetrievalSignal::TemporalRelevance,
            )],
        ];

        let fused = hybrid
            .fuse_results(all_results, 10)
            .expect("fusion should succeed");

        assert!(
            !fused.is_empty(),
            "RRF fusion must not filter out all results"
        );
        assert_eq!(
            fused[0].0.entry.key, "a",
            "item ranked #1 by two retrievers must outrank item ranked #1 by one"
        );
        let pos_b = fused
            .iter()
            .position(|(m, _)| m.entry.key == "b")
            .expect("item b should be present");
        assert!(pos_b >= 1);
    }

    #[tokio::test]
    async fn test_hybrid_retriever_creation() {
        let config = PipelineConfig::default();
        let retriever = HybridRetriever::new(config);
        assert_eq!(retriever.pipelines.len(), 0);
        assert!(retriever.cache.is_some());
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let mut cache = ScoreCache::new(300);

        let memory = MemoryFragment::new(
            crate::memory::types::MemoryEntry::new(
                "test".to_string(),
                "test content".to_string(),
                crate::memory::types::MemoryType::ShortTerm,
            ),
            0.8,
        );

        // Insert
        cache.insert("test_query".to_string(), vec![memory.clone()]);

        // Get
        let cached = cache.get("test_query");
        assert!(cached.is_some());
        assert_eq!(cached.expect("cached should be valid").len(), 1);

        // Clear
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
    }
}
