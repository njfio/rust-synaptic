//! Concrete implementations of retrieval pipelines
//!
//! This module provides specific retrieval strategies that can be composed
//! in a hybrid pipeline for optimal search quality.

use super::pipeline::{RetrievalPipeline, RetrievalSignal, ScoredMemory, PipelineConfig};
use crate::error::{MemoryError, Result};
use crate::memory::storage::Storage;
use crate::memory::types::{MemoryEntry, MemoryFragment, MemoryType};
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use chrono::Utc;

/// Keyword-based retriever using BM25-style scoring
///
/// Implements sparse keyword matching with TF-IDF weighting and BM25 scoring.
/// Good for exact keyword matches and traditional search queries.
pub struct KeywordRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
}

impl KeywordRetriever {
    pub fn new(storage: Arc<dyn Storage + Send + Sync>) -> Self {
        Self { storage }
    }

    /// Compute BM25-style score for a memory against a query
    fn compute_bm25_score(&self, memory_content: &str, query: &str) -> f64 {
        let query_terms: Vec<&str> = query.to_lowercase().split_whitespace().collect();
        let content_lower = memory_content.to_lowercase();
        let content_terms: Vec<&str> = content_lower.split_whitespace().collect();

        if query_terms.is_empty() || content_terms.is_empty() {
            return 0.0;
        }

        // Count term frequencies
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for term in &content_terms {
            *term_freq.entry(term.to_string()).or_insert(0) += 1;
        }

        // BM25 parameters
        let k1 = 1.2; // Term frequency saturation parameter
        let b = 0.75; // Length normalization parameter
        let avg_doc_length = 100.0; // Approximate average document length
        let doc_length = content_terms.len() as f64;

        // Calculate score
        let mut score = 0.0;
        for query_term in query_terms {
            if let Some(&freq) = term_freq.get(query_term) {
                let tf = freq as f64;
                let idf = 1.0; // Simplified IDF (would need document collection for real IDF)

                // BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                let numerator = tf * (k1 + 1.0);
                let denominator = tf + k1 * (1.0 - b + b * doc_length / avg_doc_length);
                score += idf * (numerator / denominator);
            }
        }

        // Normalize to 0-1 range (approximation)
        (score / query_terms.len() as f64).min(1.0)
    }
}

#[async_trait]
impl RetrievalPipeline for KeywordRetriever {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            "KeywordRetriever: starting search"
        );

        // Get all memories from storage (in real implementation, this would use an index)
        let fragments = self.storage.search(query, limit * 2).await?;

        // Score each result with BM25
        let mut scored_results: Vec<ScoredMemory> = fragments
            .into_iter()
            .map(|fragment| {
                let score = self.compute_bm25_score(&fragment.content, query);
                ScoredMemory::new(fragment, score, RetrievalSignal::SparseKeyword)
            })
            .collect();

        // Sort by score descending
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top limit
        scored_results.truncate(limit);

        tracing::debug!(
            result_count = scored_results.len(),
            "KeywordRetriever: search completed"
        );

        Ok(scored_results)
    }

    fn name(&self) -> &'static str {
        "KeywordRetriever"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::SparseKeyword
    }
}

/// Temporal-based retriever emphasizing recency and access patterns
///
/// Scores memories based on:
/// - Recency (recently created/modified memories score higher)
/// - Access frequency (frequently accessed memories score higher)
/// - Access recency (recently accessed memories score higher)
pub struct TemporalRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    recency_weight: f64,
    frequency_weight: f64,
}

impl TemporalRetriever {
    pub fn new(storage: Arc<dyn Storage + Send + Sync>) -> Self {
        Self {
            storage,
            recency_weight: 0.6,
            frequency_weight: 0.4,
        }
    }

    pub fn with_weights(mut self, recency_weight: f64, frequency_weight: f64) -> Self {
        self.recency_weight = recency_weight;
        self.frequency_weight = frequency_weight;
        self
    }

    /// Compute temporal relevance score
    fn compute_temporal_score(&self, memory: &MemoryFragment) -> f64 {
        let now = Utc::now();

        // Recency score (exponential decay)
        let age_days = (now - memory.timestamp).num_days() as f64;
        let recency_score = (-age_days / 30.0).exp(); // 30-day half-life

        // For frequency, we'd need access count from full MemoryEntry
        // For now, use a placeholder based on relevance_score
        let frequency_score = memory.relevance_score;

        // Combine scores
        let combined = self.recency_weight * recency_score + self.frequency_weight * frequency_score;

        combined.min(1.0).max(0.0)
    }
}

#[async_trait]
impl RetrievalPipeline for TemporalRetriever {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            "TemporalRetriever: starting search"
        );

        // Get memories from storage
        let fragments = self.storage.search(query, limit * 2).await?;

        // Score with temporal relevance
        let mut scored_results: Vec<ScoredMemory> = fragments
            .into_iter()
            .map(|fragment| {
                let score = self.compute_temporal_score(&fragment);
                ScoredMemory::new(fragment, score, RetrievalSignal::TemporalRelevance)
                    .with_explanation(format!(
                        "Temporal score based on recency and access patterns"
                    ))
            })
            .collect();

        // Sort by score descending
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top limit
        scored_results.truncate(limit);

        tracing::debug!(
            result_count = scored_results.len(),
            "TemporalRetriever: search completed"
        );

        Ok(scored_results)
    }

    fn name(&self) -> &'static str {
        "TemporalRetriever"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::TemporalRelevance
    }
}

/// Graph-based retriever using relationship strength
///
/// Scores memories based on their position in the knowledge graph:
/// - Highly connected memories score higher (centrality)
/// - Memories related to query matches score higher (propagation)
/// - Relationship strength influences score
pub struct GraphRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
}

impl GraphRetriever {
    pub fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
    ) -> Self {
        Self {
            storage,
            knowledge_graph,
        }
    }

    /// Compute graph-based score
    async fn compute_graph_score(
        &self,
        memory_key: &str,
        base_score: f64,
    ) -> Result<f64> {
        if let Some(ref kg) = self.knowledge_graph {
            let kg_guard = kg.read().await;

            // Find related memories
            match kg_guard.find_related_memories(memory_key, 2, None).await {
                Ok(related) => {
                    // More relationships = higher score
                    let relationship_boost = (related.len() as f64 / 10.0).min(0.5);

                    // Average relationship strength
                    let avg_strength = if !related.is_empty() {
                        related.iter().map(|r| r.relationship_strength).sum::<f64>()
                            / related.len() as f64
                    } else {
                        0.0
                    };

                    let graph_score = base_score + relationship_boost + (avg_strength * 0.3);
                    Ok(graph_score.min(1.0))
                }
                Err(_) => Ok(base_score),
            }
        } else {
            Ok(base_score)
        }
    }
}

#[async_trait]
impl RetrievalPipeline for GraphRetriever {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            graph_available = self.knowledge_graph.is_some(),
            "GraphRetriever: starting search"
        );

        // Get base results from storage
        let fragments = self.storage.search(query, limit * 2).await?;

        let mut scored_results = Vec::new();

        for fragment in fragments {
            // Compute graph-enhanced score
            let base_score = fragment.relevance_score;
            let graph_score = self
                .compute_graph_score(&fragment.key, base_score)
                .await
                .unwrap_or(base_score);

            scored_results.push(
                ScoredMemory::new(fragment, graph_score, RetrievalSignal::GraphRelationship)
                    .with_explanation("Score enhanced by graph relationships".to_string()),
            );
        }

        // Sort by score descending
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top limit
        scored_results.truncate(limit);

        tracing::debug!(
            result_count = scored_results.len(),
            "GraphRetriever: search completed"
        );

        Ok(scored_results)
    }

    fn name(&self) -> &'static str {
        "GraphRetriever"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::GraphRelationship
    }

    fn is_available(&self) -> bool {
        self.knowledge_graph.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_score_computation() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let retriever = KeywordRetriever::new(storage);

        let content = "rust programming language systems";
        let query = "rust systems";

        let score = retriever.compute_bm25_score(content, query);
        assert!(score > 0.0);
        assert!(score <= 1.0);

        // Exact match should score higher
        let score_exact = retriever.compute_bm25_score("rust systems", "rust systems");
        assert!(score_exact >= score);
    }

    #[test]
    fn test_temporal_score_recent() {
        use chrono::Duration;

        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let retriever = TemporalRetriever::new(storage);

        // Recent memory
        let recent_fragment = MemoryFragment {
            key: "recent".to_string(),
            content: "recent content".to_string(),
            memory_type: MemoryType::ShortTerm,
            relevance_score: 0.5,
            timestamp: Utc::now() - Duration::days(1),
        };

        // Old memory
        let old_fragment = MemoryFragment {
            key: "old".to_string(),
            content: "old content".to_string(),
            memory_type: MemoryType::LongTerm,
            relevance_score: 0.5,
            timestamp: Utc::now() - Duration::days(365),
        };

        let recent_score = retriever.compute_temporal_score(&recent_fragment);
        let old_score = retriever.compute_temporal_score(&old_fragment);

        // Recent should score higher
        assert!(recent_score > old_score);
    }
}
