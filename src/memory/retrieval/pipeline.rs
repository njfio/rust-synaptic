//! Pluggable retrieval pipeline for hybrid semantic search
//!
//! This module provides a flexible retrieval pipeline that combines multiple
//! signals (embeddings, keywords, graph relationships, temporal relevance) to
//! deliver high-quality search results.

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryFragment};
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
        }
    }
}

impl PipelineConfig {
    /// Create a configuration optimized for semantic search
    pub fn semantic_focus() -> Self {
        let mut config = Self::default();
        config.signal_weights.insert(RetrievalSignal::DenseVector, 0.6);
        config.signal_weights.insert(RetrievalSignal::SparseKeyword, 0.2);
        config.signal_weights.insert(RetrievalSignal::GraphRelationship, 0.15);
        config.signal_weights.insert(RetrievalSignal::TemporalRelevance, 0.05);
        config
    }

    /// Create a configuration optimized for keyword search
    pub fn keyword_focus() -> Self {
        let mut config = Self::default();
        config.signal_weights.insert(RetrievalSignal::DenseVector, 0.2);
        config.signal_weights.insert(RetrievalSignal::SparseKeyword, 0.6);
        config.signal_weights.insert(RetrievalSignal::GraphRelationship, 0.15);
        config.signal_weights.insert(RetrievalSignal::TemporalRelevance, 0.05);
        config
    }

    /// Create a configuration optimized for graph-based discovery
    pub fn graph_focus() -> Self {
        let mut config = Self::default();
        config.signal_weights.insert(RetrievalSignal::DenseVector, 0.25);
        config.signal_weights.insert(RetrievalSignal::SparseKeyword, 0.25);
        config.signal_weights.insert(RetrievalSignal::GraphRelationship, 0.4);
        config.signal_weights.insert(RetrievalSignal::TemporalRelevance, 0.1);
        config
    }

    /// Create a configuration optimized for recent/relevant content
    pub fn temporal_focus() -> Self {
        let mut config = Self::default();
        config.signal_weights.insert(RetrievalSignal::DenseVector, 0.3);
        config.signal_weights.insert(RetrievalSignal::SparseKeyword, 0.2);
        config.signal_weights.insert(RetrievalSignal::GraphRelationship, 0.1);
        config.signal_weights.insert(RetrievalSignal::TemporalRelevance, 0.4);
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
        }
    }

    /// Add a retrieval pipeline to the hybrid retriever
    pub fn add_pipeline(mut self, pipeline: Arc<dyn RetrievalPipeline>) -> Self {
        self.pipelines.push(pipeline);
        self
    }

    /// Search using all available pipelines and fuse results
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

        // Collect results from all pipelines
        let mut all_results: Vec<Vec<ScoredMemory>> = Vec::new();

        for pipeline in &self.pipelines {
            if !pipeline.is_available() {
                tracing::debug!(
                    pipeline = pipeline.name(),
                    "Skipping unavailable pipeline"
                );
                continue;
            }

            match pipeline
                .search(query, self.config.max_per_signal, Some(&self.config))
                .await
            {
                Ok(results) => {
                    tracing::debug!(
                        pipeline = pipeline.name(),
                        result_count = results.len(),
                        "Pipeline search completed"
                    );
                    all_results.push(results);
                }
                Err(e) => {
                    tracing::warn!(
                        pipeline = pipeline.name(),
                        error = %e,
                        "Pipeline search failed"
                    );
                    // Continue with other pipelines
                }
            }
        }

        // Fuse results
        let fused_results = self.fuse_results(all_results, limit)?;

        tracing::info!(
            final_count = fused_results.len(),
            "Hybrid search completed"
        );

        // Cache results
        if let Some(ref cache) = self.cache {
            let cache_key = format!("{}:{}", query, limit);
            cache.write().await.insert(cache_key, fused_results.clone());
        }

        Ok(fused_results)
    }

    /// Fuse results from multiple pipelines using the configured strategy
    fn fuse_results(
        &self,
        all_results: Vec<Vec<ScoredMemory>>,
        limit: usize,
    ) -> Result<Vec<MemoryFragment>> {
        if all_results.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all unique memories
        let mut memory_scores: HashMap<String, (MemoryFragment, Vec<(RetrievalSignal, f64)>)> =
            HashMap::new();

        for results in all_results {
            for scored in results {
                let entry = memory_scores
                    .entry(scored.memory.key.clone())
                    .or_insert_with(|| (scored.memory.clone(), Vec::new()));

                entry.1.push((scored.signal, scored.score));
            }
        }

        // Apply fusion strategy
        let mut fused: Vec<(MemoryFragment, f64)> = memory_scores
            .into_iter()
            .map(|(key, (memory, signal_scores))| {
                let combined_score = self.combine_scores(&signal_scores);
                (memory, combined_score)
            })
            .filter(|(_, score)| *score >= self.config.min_score)
            .collect();

        // Sort by score descending
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top limit results
        Ok(fused
            .into_iter()
            .take(limit)
            .map(|(memory, _score)| memory)
            .collect())
    }

    /// Combine scores from multiple signals using the configured strategy
    fn combine_scores(&self, signal_scores: &[(RetrievalSignal, f64)]) -> f64 {
        match self.config.fusion_strategy {
            FusionStrategy::WeightedAverage => {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (signal, score) in signal_scores {
                    let weight = self.config.signal_weights.get(signal).copied().unwrap_or(1.0);
                    weighted_sum += score * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                }
            }
            FusionStrategy::MaxScore => {
                signal_scores
                    .iter()
                    .map(|(_, score)| score)
                    .fold(0.0, |max, &score| max.max(score))
            }
            FusionStrategy::ReciprocRankFusion => {
                // RRF: score = sum(1 / (rank + k)) where k=60
                // For simplicity, we'll use normalized scores as approximation
                let k = 60.0;
                signal_scores
                    .iter()
                    .map(|(_, score)| 1.0 / (score.recip() + k))
                    .sum()
            }
            FusionStrategy::BordaCount => {
                // Borda count: assign points based on ranking
                // Higher ranked items get more points
                let max_score = signal_scores.len() as f64;
                signal_scores.iter().map(|(_, score)| score * max_score).sum()
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
        self.entries.retain(|_, cached| {
            cached.cached_at.elapsed().as_secs() < self.ttl_seconds
        });
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
        let dense_weight = config.signal_weights.get(&RetrievalSignal::DenseVector).unwrap();
        assert_eq!(*dense_weight, 0.6);
    }

    #[test]
    fn test_scored_memory_creation() {
        let memory = MemoryFragment {
            key: "test".to_string(),
            content: "test content".to_string(),
            memory_type: crate::memory::types::MemoryType::ShortTerm,
            relevance_score: 0.8,
            timestamp: chrono::Utc::now(),
        };

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

        let memory = MemoryFragment {
            key: "test".to_string(),
            content: "test content".to_string(),
            memory_type: crate::memory::types::MemoryType::ShortTerm,
            relevance_score: 0.8,
            timestamp: chrono::Utc::now(),
        };

        // Insert
        cache.insert("test_query".to_string(), vec![memory.clone()]);

        // Get
        let cached = cache.get("test_query");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);

        // Clear
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
    }
}
