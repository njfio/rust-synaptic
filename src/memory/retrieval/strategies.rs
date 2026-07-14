//! Concrete implementations of retrieval pipelines
//!
//! This module provides specific retrieval strategies that can be composed
//! in a hybrid pipeline for optimal search quality.

use super::pipeline::{PipelineConfig, RetrievalPipeline, RetrievalSignal, ScoredMemory};
use crate::error::Result;
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use crate::memory::storage::Storage;
use crate::memory::types::MemoryFragment;
use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;

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

    /// Corpus statistics over the candidate set, used for real BM25 IDF.
    fn corpus_stats(query_terms: &[String], candidate_contents: &[String]) -> CorpusStats {
        let n_docs = candidate_contents.len();
        let mut total_len = 0usize;
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for content in candidate_contents {
            let content_lower = content.to_lowercase();
            let terms: Vec<&str> = content_lower.split_whitespace().collect();
            total_len += terms.len();
            for query_term in query_terms {
                if terms.iter().any(|t| t == query_term) {
                    *doc_freq.entry(query_term.clone()).or_insert(0) += 1;
                }
            }
        }

        let avg_doc_length = if n_docs > 0 {
            total_len as f64 / n_docs as f64
        } else {
            0.0
        };

        CorpusStats {
            n_docs,
            avg_doc_length,
            doc_freq,
        }
    }

    /// Compute BM25 score for a memory against a query using corpus-derived IDF.
    ///
    /// IDF follows the standard BM25 form:
    /// `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`
    /// where N is the number of candidate documents and df is the number of
    /// candidates containing the term.
    fn compute_bm25_score(
        &self,
        memory_content: &str,
        query_terms: &[String],
        stats: &CorpusStats,
    ) -> f64 {
        let content_lower = memory_content.to_lowercase();
        let content_terms: Vec<&str> = content_lower.split_whitespace().collect();

        if query_terms.is_empty() || content_terms.is_empty() || stats.n_docs == 0 {
            return 0.0;
        }

        // Count term frequencies
        let mut term_freq: HashMap<&str, usize> = HashMap::new();
        for term in &content_terms {
            *term_freq.entry(term).or_insert(0) += 1;
        }

        // BM25 parameters
        let k1 = 1.2; // Term frequency saturation parameter
        let b = 0.75; // Length normalization parameter
        let n = stats.n_docs as f64;
        let avg_doc_length = stats.avg_doc_length.max(1.0);
        let doc_length = content_terms.len() as f64;

        // Maximum possible IDF (df = 0 hypothetical lower bound is df >= 1 for
        // any matched term), used to normalize the final score into 0-1.
        let max_idf = (((n - 1.0 + 0.5) / 1.5) + 1.0).ln().max(f64::MIN_POSITIVE);

        // Calculate score
        let mut score = 0.0;
        for query_term in query_terms {
            if let Some(&freq) = term_freq.get(query_term.as_str()) {
                let tf = freq as f64;
                let df = *stats.doc_freq.get(query_term).unwrap_or(&0) as f64;
                let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln();

                // BM25: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                let numerator = tf * (k1 + 1.0);
                let denominator = tf + k1 * (1.0 - b + b * doc_length / avg_doc_length);
                score += idf * (numerator / denominator);
            }
        }

        // Normalize to 0-1: divide by the best case of every query term
        // matching at maximum IDF with saturated tf (tf-component < k1 + 1).
        let max_score = query_terms.len() as f64 * max_idf * (k1 + 1.0);
        (score / max_score).clamp(0.0, 1.0)
    }
}

/// Per-query corpus statistics for BM25 IDF computation.
struct CorpusStats {
    /// Number of candidate documents (N)
    n_docs: usize,
    /// Average candidate document length in terms
    avg_doc_length: f64,
    /// Document frequency per query term (df)
    doc_freq: HashMap<String, usize>,
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

        // Fetch candidate memories via the storage backend's search, over-fetching
        // 2x so BM25 re-ranking below has a wider pool to choose from.
        let fragments = self.storage.search(query, limit * 2).await?;

        // Build per-query corpus statistics over the candidate set so IDF
        // reflects actual term rarity among candidates.
        let query_lower = query.to_lowercase();
        let query_terms: Vec<String> = query_lower
            .split_whitespace()
            .map(|t| t.to_string())
            .collect();
        let candidate_contents: Vec<String> =
            fragments.iter().map(|f| f.entry.value.clone()).collect();
        let stats = Self::corpus_stats(&query_terms, &candidate_contents);

        // Score each result with BM25
        let mut scored_results: Vec<ScoredMemory> = fragments
            .into_iter()
            .map(|fragment| {
                let score = self.compute_bm25_score(&fragment.entry.value, &query_terms, &stats);
                ScoredMemory::new(fragment, score, RetrievalSignal::SparseKeyword)
            })
            .collect();

        // Sort by score descending
        scored_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
        let age_days = (now - memory.entry.created_at()).num_days() as f64;
        let recency_score = (-age_days / 30.0).exp(); // 30-day half-life

        // Access-frequency score from the entry's real access count,
        // saturating so heavily-accessed memories approach 1.0.
        // count / (count + k) gives 0.0 for never-accessed, ~0.67 at 2k accesses.
        let access_count = memory.entry.access_count() as f64;
        let frequency_score = access_count / (access_count + 10.0);

        // Combine scores
        let combined =
            self.recency_weight * recency_score + self.frequency_weight * frequency_score;

        combined.clamp(0.0, 1.0)
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
                    .with_explanation(
                        "Temporal score based on recency and access patterns".to_string(),
                    )
            })
            .collect();

        // Sort by score descending
        scored_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
    async fn compute_graph_score(&self, memory_key: &str, base_score: f64) -> Result<f64> {
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
                .compute_graph_score(&fragment.entry.key, base_score)
                .await
                .unwrap_or(base_score);

            scored_results.push(
                ScoredMemory::new(fragment, graph_score, RetrievalSignal::GraphRelationship)
                    .with_explanation("Score enhanced by graph relationships".to_string()),
            );
        }

        // Expand candidates through the knowledge graph: memories connected
        // to the strongest direct matches are surfaced even when they share
        // no terms with the query, scored by propagating the seed's score
        // through the relationship strength.
        if let Some(ref kg) = self.knowledge_graph {
            const MAX_EXPANSION_SEEDS: usize = 5;

            let mut seeds: Vec<(String, f64)> = scored_results
                .iter()
                .map(|s| (s.memory.entry.key.clone(), s.score))
                .collect();
            seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            seeds.truncate(MAX_EXPANSION_SEEDS);

            let mut seen: std::collections::HashSet<String> = scored_results
                .iter()
                .map(|s| s.memory.entry.key.clone())
                .collect();

            // Phase 1: collect related keys under the graph read lock, then
            // DROP the guard before any storage awaits — the lock must never
            // be held across `storage.retrieve`.
            let mut expansions: Vec<(String, f64, Vec<_>)> = Vec::with_capacity(seeds.len());
            {
                let kg_guard = kg.read().await;
                for (seed_key, seed_score) in seeds {
                    let related = match kg_guard.find_related_memories(&seed_key, 2, None).await {
                        Ok(related) => related,
                        Err(e) => {
                            // Seed may simply not be a graph node yet.
                            tracing::debug!(seed = %seed_key, error = %e, "graph expansion skipped for seed");
                            continue;
                        }
                    };
                    expansions.push((seed_key, seed_score, related));
                }
            }

            // Phase 2: fetch expansion candidates from storage, lock-free.
            for (seed_key, seed_score, related) in expansions {
                for rel in related {
                    if !seen.insert(rel.memory_key.clone()) {
                        continue;
                    }
                    let entry = match self.storage.retrieve(&rel.memory_key).await {
                        Ok(Some(entry)) => entry,
                        Ok(None) => continue,
                        Err(e) => {
                            tracing::warn!(key = %rel.memory_key, error = %e, "graph expansion candidate fetch failed");
                            continue;
                        }
                    };
                    let score =
                        (seed_score * rel.relationship_strength.clamp(0.0, 1.0)).clamp(0.0, 1.0);
                    let fragment = MemoryFragment::new(entry, score);
                    scored_results.push(
                        ScoredMemory::new(fragment, score, RetrievalSignal::GraphRelationship)
                            .with_explanation(format!(
                                "Surfaced via graph relationship to '{seed_key}'"
                            )),
                    );
                }
            }
        }

        // Sort by score descending
        scored_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
    use crate::memory::types::{MemoryEntry, MemoryType};

    #[test]
    fn test_bm25_score_computation() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let retriever = KeywordRetriever::new(storage);

        let query_terms = vec!["rust".to_string(), "systems".to_string()];
        let corpus = vec![
            "rust programming language systems".to_string(),
            "rust systems".to_string(),
            "python scripting language".to_string(),
        ];
        let stats = KeywordRetriever::corpus_stats(&query_terms, &corpus);

        let score = retriever.compute_bm25_score(&corpus[0], &query_terms, &stats);
        assert!(score > 0.0);
        assert!(score <= 1.0);

        // Exact (shorter) match should score at least as high
        let score_exact = retriever.compute_bm25_score(&corpus[1], &query_terms, &stats);
        assert!(score_exact >= score);
    }

    #[test]
    fn test_bm25_idf_rare_term_weighted_higher() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let retriever = KeywordRetriever::new(storage);

        let query_terms = vec!["quokka".to_string(), "database".to_string()];
        let corpus = vec![
            "quokka index entry".to_string(),
            "database index entry".to_string(),
            "database schema design overview".to_string(),
            "database migration tooling notes".to_string(),
        ];
        let stats = KeywordRetriever::corpus_stats(&query_terms, &corpus);

        assert_eq!(stats.n_docs, 4);
        assert_eq!(stats.doc_freq.get("quokka"), Some(&1));
        assert_eq!(stats.doc_freq.get("database"), Some(&3));

        let rare = retriever.compute_bm25_score(&corpus[0], &query_terms, &stats);
        let common = retriever.compute_bm25_score(&corpus[1], &query_terms, &stats);
        assert!(
            rare > common,
            "rare-term doc must outscore common-term doc: {rare} vs {common}"
        );
    }

    #[test]
    fn test_temporal_score_recent() {
        use chrono::Duration;

        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let retriever = TemporalRetriever::new(storage);

        // Recent memory
        let mut recent_entry = MemoryEntry::new(
            "recent".to_string(),
            "recent content".to_string(),
            MemoryType::ShortTerm,
        );
        recent_entry.metadata.created_at = Utc::now() - Duration::days(1);
        let recent_fragment = MemoryFragment::new(recent_entry, 0.5);

        // Old memory
        let mut old_entry = MemoryEntry::new(
            "old".to_string(),
            "old content".to_string(),
            MemoryType::LongTerm,
        );
        old_entry.metadata.created_at = Utc::now() - Duration::days(365);
        let old_fragment = MemoryFragment::new(old_entry, 0.5);

        let recent_score = retriever.compute_temporal_score(&recent_fragment);
        let old_score = retriever.compute_temporal_score(&old_fragment);

        // Recent should score higher
        assert!(recent_score > old_score);
    }
}
