//! Advanced search engine for memory retrieval

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use strsim::{levenshtein, jaro_winkler, normalized_damerau_levenshtein, sorensen_dice};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Advanced search query with multiple criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Text query for content search
    pub text_query: Option<String>,
    /// Filters to apply
    pub filters: Vec<SearchFilter>,
    /// Ranking strategy
    pub ranking_strategy: RankingStrategy,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
    /// Include related memories
    pub include_related: bool,
    /// Minimum relevance score
    pub min_relevance_score: Option<f64>,
    /// Search scope
    pub scope: SearchScope,
}

/// Search filters for refining results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchFilter {
    /// Filter by memory type
    MemoryType(MemoryType),
    /// Filter by tags (any of these tags)
    Tags(Vec<String>),
    /// Filter by importance range
    ImportanceRange { min: f64, max: f64 },
    /// Filter by confidence range
    ConfidenceRange { min: f64, max: f64 },
    /// Filter by creation date range
    CreatedDateRange { start: DateTime<Utc>, end: DateTime<Utc> },
    /// Filter by last access date range
    AccessedDateRange { start: DateTime<Utc>, end: DateTime<Utc> },
    /// Filter by access frequency
    AccessFrequency { min_count: u64 },
    /// Filter by content length
    ContentLength { min_length: usize, max_length: Option<usize> },
    /// Filter by custom field
    CustomField { key: String, value: String },
    /// Filter by similarity to a reference memory
    SimilarTo { memory_key: String, threshold: f64 },
    /// Filter by relationship in knowledge graph
    RelatedTo { memory_key: String, max_distance: usize },
}

/// Ranking strategies for search results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RankingStrategy {
    /// Rank by relevance score (default)
    Relevance,
    /// Rank by importance
    Importance,
    /// Rank by recency (newest first)
    Recency,
    /// Rank by access frequency
    Frequency,
    /// Rank by confidence
    Confidence,
    /// Rank by content length
    ContentLength,
    /// Combined ranking using multiple factors
    Combined(Vec<RankingFactor>),
    /// Custom ranking algorithm
    Custom(String),
}

/// Factors for combined ranking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingFactor {
    pub factor_type: RankingFactorType,
    pub weight: f64,
}

/// Types of ranking factors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingFactorType {
    Relevance,
    Importance,
    Recency,
    Frequency,
    Confidence,
    GraphCentrality,
    UserPreference,
}

/// Search scope definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search all memories
    All,
    /// Search only active memories
    Active,
    /// Search only archived memories
    Archived,
    /// Search specific memory keys
    Specific(Vec<String>),
    /// Search within a knowledge graph cluster
    GraphCluster(String),
}

/// Search result with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The memory entry
    pub memory: MemoryEntry,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f64,
    /// Ranking score based on strategy
    pub ranking_score: f64,
    /// Highlighted text snippets
    pub highlights: Vec<TextHighlight>,
    /// Explanation of why this result was returned
    pub explanation: String,
    /// Related memories (if requested)
    pub related_memories: Vec<RelatedMemoryRef>,
    /// Search metadata
    pub search_metadata: SearchResultMetadata,
}

/// Text highlight in search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextHighlight {
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Highlighted text
    pub text: String,
    /// Context around the highlight
    pub context: String,
    /// Type of highlight
    pub highlight_type: HighlightType,
}

/// Types of text highlights
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighlightType {
    /// Exact query match
    ExactMatch,
    /// Partial query match
    PartialMatch,
    /// Semantic match
    SemanticMatch,
    /// Entity match
    EntityMatch,
    /// Concept match
    ConceptMatch,
}

/// Reference to a related memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedMemoryRef {
    /// Memory key
    pub memory_key: String,
    /// Relationship type
    pub relationship_type: String,
    /// Relationship strength
    pub strength: f64,
}

/// Metadata for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultMetadata {
    /// When this result was generated
    pub generated_at: DateTime<Utc>,
    /// Search algorithm used
    pub algorithm: String,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Additional debug information
    pub debug_info: HashMap<String, String>,
}

/// Advanced search engine implementation
pub struct AdvancedSearchEngine {
    /// Search index for fast text retrieval
    search_index: SearchIndex,
    /// Configuration
    config: SearchConfig,
    /// Search history for analytics
    search_history: Vec<SearchQuery>,
    /// Performance metrics
    performance_metrics: SearchPerformanceMetrics,
}

/// Search index for efficient text search
struct SearchIndex {
    /// Inverted index: word -> list of memory keys
    word_index: HashMap<String, HashSet<String>>,
    /// Memory metadata index
    metadata_index: HashMap<String, MemoryIndexEntry>,
    /// Last update timestamp
    last_updated: DateTime<Utc>,
}

/// Index entry for a memory
#[derive(Debug, Clone)]
struct MemoryIndexEntry {
    pub memory_key: String,
    pub content_words: HashSet<String>,
    pub tags: HashSet<String>,
    pub importance: f64,
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub content_length: usize,
}

/// Configuration for the search engine
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Enable fuzzy matching
    pub enable_fuzzy_matching: bool,
    /// Fuzzy matching threshold
    pub fuzzy_threshold: f64,
    /// Enable semantic search
    pub enable_semantic_search: bool,
    /// Maximum results per query
    pub max_results: usize,
    /// Enable search result caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable search analytics
    pub enable_analytics: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy_matching: true,
            fuzzy_threshold: 0.8,
            enable_semantic_search: false, // Requires embeddings
            max_results: 100,
            enable_caching: true,
            cache_ttl_seconds: 300, // 5 minutes
            enable_analytics: true,
        }
    }
}

/// Performance metrics for search operations
#[derive(Debug, Clone, Default)]
pub struct SearchPerformanceMetrics {
    pub total_searches: usize,
    pub avg_search_time_ms: f64,
    pub cache_hit_rate: f64,
    pub index_size: usize,
    pub last_index_update: Option<DateTime<Utc>>,
}

impl AdvancedSearchEngine {
    /// Create a new advanced search engine
    pub fn new() -> Self {
        Self {
            search_index: SearchIndex::new(),
            config: SearchConfig::default(),
            search_history: Vec::new(),
            performance_metrics: SearchPerformanceMetrics::default(),
        }
    }

    /// Perform a search query
    pub async fn search(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        query: SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        // Note: Search history update removed for immutable method

        // Execute the search
        let mut results = self.execute_search(storage, &query).await?;

        // Apply ranking
        self.rank_results(&mut results, &query.ranking_strategy).await?;

        // Apply limit and offset
        if let Some(offset) = query.offset {
            if offset < results.len() {
                results.drain(0..offset);
            } else {
                results.clear();
            }
        }

        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        // Note: Performance metrics update removed for immutable method

        Ok(results)
    }

    /// Execute the core search logic
    async fn execute_search(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Text-based search
        if let Some(text_query) = &query.text_query {
            let text_results = self.search_by_text(storage, text_query).await?;
            results.extend(text_results);
        }

        // Apply filters
        results = self.apply_filters(results, &query.filters).await?;

        // Apply minimum relevance score filter
        if let Some(min_score) = query.min_relevance_score {
            results.retain(|r| r.relevance_score >= min_score);
        }

        // Add related memories if requested
        if query.include_related {
            results = self.add_related_memories(results).await?;
        }

        Ok(results)
    }

    /// Search by text content
    async fn search_by_text(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        text_query: &str,
    ) -> Result<Vec<SearchResult>> {
        let query_lower = text_query.to_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .collect();

        let mut candidate_memories: HashMap<String, f64> = HashMap::new();

        // Find memories containing query words
        for word in &query_words {
            if let Some(memory_keys) = self.search_index.word_index.get(*word) {
                for memory_key in memory_keys {
                    *candidate_memories.entry(memory_key.clone()).or_insert(0.0) += 1.0;
                }
            }

            // Fuzzy matching if enabled
            if self.config.enable_fuzzy_matching {
                let fuzzy_matches = self.find_fuzzy_matches(word).await?;
                for (fuzzy_word, similarity) in fuzzy_matches {
                    if let Some(memory_keys) = self.search_index.word_index.get(&fuzzy_word) {
                        for memory_key in memory_keys {
                            *candidate_memories.entry(memory_key.clone()).or_insert(0.0) += similarity;
                        }
                    }
                }
            }
        }

        // Convert to search results by retrieving actual memories from storage
        let mut results = Vec::new();
        for (memory_key, score) in candidate_memories {
            if let Some(_index_entry) = self.search_index.metadata_index.get(&memory_key) {
                // Retrieve the actual memory from storage
                if let Some(memory) = storage.retrieve(&memory_key).await? {
                    let relevance_score = score / query_words.len() as f64;
                    let highlights = self.generate_highlights(&memory.value, text_query).await?;

                    results.push(SearchResult {
                        memory,
                        relevance_score,
                        ranking_score: relevance_score,
                        highlights,
                        explanation: format!("Matched {} query terms with score {:.2}", score as usize, relevance_score),
                        related_memories: Vec::new(),
                        search_metadata: SearchResultMetadata {
                            generated_at: Utc::now(),
                            algorithm: "text_search".to_string(),
                            processing_time_ms: 0,
                            debug_info: HashMap::new(),
                        },
                    });
                } else {
                    // Memory not found in storage, but exists in index - log warning
                    tracing::warn!("Memory '{}' found in search index but not in storage", memory_key);
                }
            }
        }

        Ok(results)
    }

    /// Apply search filters to results
    async fn apply_filters(&self, mut results: Vec<SearchResult>, filters: &[SearchFilter]) -> Result<Vec<SearchResult>> {
        for filter in filters {
            results = self.apply_single_filter(results, filter).await?;
        }
        Ok(results)
    }

    /// Apply a single filter
    async fn apply_single_filter(&self, results: Vec<SearchResult>, filter: &SearchFilter) -> Result<Vec<SearchResult>> {
        match filter {
            SearchFilter::MemoryType(memory_type) => {
                Ok(results.into_iter()
                    .filter(|r| r.memory.memory_type == *memory_type)
                    .collect())
            }
            SearchFilter::Tags(tags) => {
                Ok(results.into_iter()
                    .filter(|r| tags.iter().any(|tag| r.memory.has_tag(tag)))
                    .collect())
            }
            SearchFilter::ImportanceRange { min, max } => {
                Ok(results.into_iter()
                    .filter(|r| r.memory.metadata.importance >= *min && r.memory.metadata.importance <= *max)
                    .collect())
            }
            SearchFilter::ConfidenceRange { min, max } => {
                Ok(results.into_iter()
                    .filter(|r| r.memory.metadata.confidence >= *min && r.memory.metadata.confidence <= *max)
                    .collect())
            }
            SearchFilter::CreatedDateRange { start, end } => {
                Ok(results.into_iter()
                    .filter(|r| {
                        let created = r.memory.created_at();
                        created >= *start && created <= *end
                    })
                    .collect())
            }
            SearchFilter::AccessedDateRange { start, end } => {
                Ok(results.into_iter()
                    .filter(|r| {
                        let accessed = r.memory.last_accessed();
                        accessed >= *start && accessed <= *end
                    })
                    .collect())
            }
            SearchFilter::AccessFrequency { min_count } => {
                Ok(results.into_iter()
                    .filter(|r| r.memory.access_count() >= *min_count)
                    .collect())
            }
            SearchFilter::ContentLength { min_length, max_length } => {
                Ok(results.into_iter()
                    .filter(|r| {
                        let len = r.memory.value.len();
                        len >= *min_length && max_length.map_or(true, |max| len <= max)
                    })
                    .collect())
            }
            SearchFilter::CustomField { key, value } => {
                Ok(results.into_iter()
                    .filter(|r| r.memory.metadata.get_custom_field(key) == Some(value))
                    .collect())
            }
            SearchFilter::SimilarTo { memory_key, threshold } => {
                self.apply_similarity_filter(results, memory_key, *threshold).await
            }
            SearchFilter::RelatedTo { memory_key, max_distance } => {
                self.apply_graph_distance_filter(results, memory_key, *max_distance).await
            }
        }
    }

    /// Rank search results based on strategy
    async fn rank_results(&self, results: &mut [SearchResult], strategy: &RankingStrategy) -> Result<()> {
        match strategy {
            RankingStrategy::Relevance => {
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
            }
            RankingStrategy::Importance => {
                results.sort_by(|a, b| b.memory.metadata.importance.partial_cmp(&a.memory.metadata.importance).unwrap());
            }
            RankingStrategy::Recency => {
                results.sort_by(|a, b| b.memory.created_at().cmp(&a.memory.created_at()));
            }
            RankingStrategy::Frequency => {
                results.sort_by(|a, b| b.memory.access_count().cmp(&a.memory.access_count()));
            }
            RankingStrategy::Confidence => {
                results.sort_by(|a, b| b.memory.metadata.confidence.partial_cmp(&a.memory.metadata.confidence).unwrap());
            }
            RankingStrategy::ContentLength => {
                results.sort_by(|a, b| b.memory.value.len().cmp(&a.memory.value.len()));
            }
            RankingStrategy::Combined(factors) => {
                self.apply_combined_ranking(results, factors).await?;
            }
            RankingStrategy::Custom(name) => {
                self.apply_custom_ranking(results, name).await?;
            }
        }

        // Update ranking scores
        let results_len = results.len();
        for (i, result) in results.iter_mut().enumerate() {
            result.ranking_score = 1.0 - (i as f64 / results_len as f64);
        }

        Ok(())
    }

    /// Apply combined ranking using multiple factors
    async fn apply_combined_ranking(&self, results: &mut [SearchResult], factors: &[RankingFactor]) -> Result<()> {
        for result in results.iter_mut() {
            let mut combined_score = 0.0;
            let mut total_weight = 0.0;

            for factor in factors {
                let factor_score = match factor.factor_type {
                    RankingFactorType::Relevance => result.relevance_score,
                    RankingFactorType::Importance => result.memory.metadata.importance,
                    RankingFactorType::Recency => {
                        let age_hours = (Utc::now() - result.memory.created_at()).num_hours() as f64;
                        (1.0 / (1.0 + age_hours / 24.0)).max(0.0)
                    }
                    RankingFactorType::Frequency => {
                        (result.memory.access_count() as f64 / 100.0).min(1.0)
                    }
                    RankingFactorType::Confidence => result.memory.metadata.confidence,
                    RankingFactorType::GraphCentrality => {
                        self.calculate_graph_centrality(&result.memory.key).await.unwrap_or(0.5)
                    }
                    RankingFactorType::UserPreference => {
                        self.calculate_user_preference_score(&result.memory).await.unwrap_or(0.5)
                    }
                };

                combined_score += factor_score * factor.weight;
                total_weight += factor.weight;
            }

            result.ranking_score = if total_weight > 0.0 {
                combined_score / total_weight
            } else {
                result.relevance_score
            };
        }

        results.sort_by(|a, b| b.ranking_score.partial_cmp(&a.ranking_score).unwrap());
        Ok(())
    }

    /// Apply custom ranking strategy
    async fn apply_custom_ranking(&self, results: &mut [SearchResult], strategy_name: &str) -> Result<()> {
        match strategy_name {
            "recent_and_important" => {
                // Combine recency and importance with equal weight
                for result in results.iter_mut() {
                    let age_hours = (Utc::now() - result.memory.created_at()).num_hours() as f64;
                    let recency_score = (1.0 / (1.0 + age_hours / 24.0)).max(0.0);
                    let importance_score = result.memory.metadata.importance;
                    result.ranking_score = (recency_score + importance_score) / 2.0;
                }
            }
            "user_engagement" => {
                // Rank by access frequency and confidence
                for result in results.iter_mut() {
                    let frequency_score = (result.memory.access_count() as f64 / 100.0).min(1.0);
                    let confidence_score = result.memory.metadata.confidence;
                    result.ranking_score = (frequency_score * 0.6 + confidence_score * 0.4);
                }
            }
            "content_richness" => {
                // Rank by content length and tag diversity
                for result in results.iter_mut() {
                    let content_score = (result.memory.value.len() as f64 / 1000.0).min(1.0);
                    let tag_score = (result.memory.metadata.tags.len() as f64 / 10.0).min(1.0);
                    result.ranking_score = (content_score * 0.7 + tag_score * 0.3);
                }
            }
            "balanced" => {
                // Balanced ranking considering multiple factors
                for result in results.iter_mut() {
                    let relevance = result.relevance_score;
                    let importance = result.memory.metadata.importance;
                    let age_hours = (Utc::now() - result.memory.created_at()).num_hours() as f64;
                    let recency = (1.0 / (1.0 + age_hours / 24.0)).max(0.0);
                    let frequency = (result.memory.access_count() as f64 / 100.0).min(1.0);

                    result.ranking_score = relevance * 0.3 + importance * 0.25 + recency * 0.25 + frequency * 0.2;
                }
            }
            _ => {
                // Default to relevance-based ranking for unknown strategies
                tracing::warn!("Unknown custom ranking strategy '{}', falling back to relevance", strategy_name);
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
                return Ok(());
            }
        }

        results.sort_by(|a, b| b.ranking_score.partial_cmp(&a.ranking_score).unwrap());
        Ok(())
    }

    /// Find fuzzy matches for a word
    async fn find_fuzzy_matches(&self, word: &str) -> Result<Vec<(String, f64)>> {
        let mut matches = Vec::new();
        
        for indexed_word in self.search_index.word_index.keys() {
            let similarity = self.calculate_string_similarity(word, indexed_word);
            if similarity >= self.config.fuzzy_threshold {
                matches.push((indexed_word.clone(), similarity));
            }
        }
        
        Ok(matches)
    }

    /// Calculate advanced string similarity using multiple algorithms
    pub fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }

        let len1 = s1.len();
        let len2 = s2.len();

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        // Use multiple similarity algorithms and combine them
        let similarities = self.calculate_multi_dimensional_similarity(s1, s2);

        // Weighted combination of different similarity measures
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // Jaro-Winkler, Levenshtein, Dice, N-gram, Semantic

        similarities.iter()
            .zip(weights.iter())
            .map(|(sim, weight)| sim * weight)
            .sum()
    }

    /// Calculate multi-dimensional similarity using various algorithms
    pub fn calculate_multi_dimensional_similarity(&self, s1: &str, s2: &str) -> Vec<f64> {
        let mut similarities = Vec::new();

        // 1. Jaro-Winkler similarity (good for names and short strings)
        let jaro_winkler_sim = jaro_winkler(s1, s2);
        similarities.push(jaro_winkler_sim);

        // 2. Normalized Damerau-Levenshtein distance
        let damerau_lev_sim = 1.0 - normalized_damerau_levenshtein(s1, s2);
        similarities.push(damerau_lev_sim);

        // 3. Sørensen-Dice coefficient (good for longer texts)
        let dice_sim = sorensen_dice(s1, s2);
        similarities.push(dice_sim);

        // 4. N-gram similarity (character-level)
        let ngram_sim = self.calculate_ngram_similarity(s1, s2, 3);
        similarities.push(ngram_sim);

        // 5. Semantic similarity (word-level overlap with weights)
        let semantic_sim = self.calculate_semantic_similarity_words(s1, s2);
        similarities.push(semantic_sim);

        similarities
    }

    /// Calculate N-gram similarity
    pub fn calculate_ngram_similarity(&self, s1: &str, s2: &str, n: usize) -> f64 {
        if s1.len() < n || s2.len() < n {
            return if s1 == s2 { 1.0 } else { 0.0 };
        }

        let ngrams1: HashSet<String> = s1.chars()
            .collect::<Vec<_>>()
            .windows(n)
            .map(|window| window.iter().collect())
            .collect();

        let ngrams2: HashSet<String> = s2.chars()
            .collect::<Vec<_>>()
            .windows(n)
            .map(|window| window.iter().collect())
            .collect();

        let intersection = ngrams1.intersection(&ngrams2).count();
        let union = ngrams1.union(&ngrams2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic similarity based on word overlap and importance
    fn calculate_semantic_similarity_words(&self, s1: &str, s2: &str) -> f64 {
        let words1: HashSet<String> = s1.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2) // Filter out very short words
            .map(|w| w.to_string())
            .collect();

        let words2: HashSet<String> = s2.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_string())
            .collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Calculate weighted word overlap
        let mut similarity_score = 0.0;
        let mut total_weight = 0.0;

        for word1 in &words1 {
            let mut best_match = 0.0;
            for word2 in &words2 {
                let word_sim = jaro_winkler(word1, word2);
                if word_sim > best_match {
                    best_match = word_sim;
                }
            }

            // Weight longer words more heavily
            let word_weight = (word1.len() as f64).sqrt();
            similarity_score += best_match * word_weight;
            total_weight += word_weight;
        }

        if total_weight > 0.0 {
            similarity_score / total_weight
        } else {
            0.0
        }
    }

    /// Generate text highlights for search results
    async fn generate_highlights(&self, content: &str, query: &str) -> Result<Vec<TextHighlight>> {
        let mut highlights = Vec::new();
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();
        
        // Find exact matches
        let mut start = 0;
        while let Some(pos) = content_lower[start..].find(&query_lower) {
            let actual_pos = start + pos;
            highlights.push(TextHighlight {
                start: actual_pos,
                end: actual_pos + query.len(),
                text: content[actual_pos..actual_pos + query.len()].to_string(),
                context: self.extract_context(content, actual_pos, query.len()),
                highlight_type: HighlightType::ExactMatch,
            });
            start = actual_pos + query.len();
        }
        
        Ok(highlights)
    }

    /// Extract context around a highlight
    fn extract_context(&self, content: &str, start: usize, length: usize) -> String {
        let context_size = 50;
        let context_start = start.saturating_sub(context_size);
        let context_end = (start + length + context_size).min(content.len());
        
        content[context_start..context_end].to_string()
    }

    /// Apply similarity filter to search results
    async fn apply_similarity_filter(
        &self,
        results: Vec<SearchResult>,
        reference_memory_key: &str,
        threshold: f64,
    ) -> Result<Vec<SearchResult>> {
        // Get reference memory from index
        let reference_entry = match self.search_index.metadata_index.get(reference_memory_key) {
            Some(entry) => entry,
            None => return Ok(results), // Reference memory not found, return all results
        };

        let filtered_results = results
            .into_iter()
            .filter(|result| {
                if let Some(result_entry) = self.search_index.metadata_index.get(&result.memory.key) {
                    let similarity = self.calculate_memory_similarity(reference_entry, result_entry);
                    similarity >= threshold
                } else {
                    false
                }
            })
            .collect();

        Ok(filtered_results)
    }

    /// Apply graph distance filter to search results
    async fn apply_graph_distance_filter(
        &self,
        results: Vec<SearchResult>,
        reference_memory_key: &str,
        max_distance: usize,
    ) -> Result<Vec<SearchResult>> {
        // In a full implementation, this would:
        // 1. Use the knowledge graph to find all memories within max_distance
        // 2. Filter results to only include those memories
        // 3. Calculate actual graph distances

        // For now, implement a simplified version based on tag similarity
        let reference_entry = match self.search_index.metadata_index.get(reference_memory_key) {
            Some(entry) => entry,
            None => return Ok(results),
        };

        let filtered_results = results
            .into_iter()
            .filter(|result| {
                if let Some(result_entry) = self.search_index.metadata_index.get(&result.memory.key) {
                    // Simulate graph distance using tag overlap
                    let tag_overlap = reference_entry.tags
                        .intersection(&result_entry.tags)
                        .count();

                    // More tag overlap = closer in graph (simplified)
                    let simulated_distance = if tag_overlap > 0 { 1 } else { 3 };
                    simulated_distance <= max_distance
                } else {
                    false
                }
            })
            .collect();

        Ok(filtered_results)
    }

    /// Calculate similarity between two memory index entries
    fn calculate_memory_similarity(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> f64 {
        let mut similarity_score = 0.0;
        let mut weight_sum = 0.0;

        // Content word similarity (weight: 0.4)
        let word_intersection = entry1.content_words.intersection(&entry2.content_words).count();
        let word_union = entry1.content_words.union(&entry2.content_words).count();
        let word_similarity = if word_union > 0 {
            word_intersection as f64 / word_union as f64
        } else {
            0.0
        };
        similarity_score += word_similarity * 0.4;
        weight_sum += 0.4;

        // Tag similarity (weight: 0.3)
        let tag_intersection = entry1.tags.intersection(&entry2.tags).count();
        let tag_union = entry1.tags.union(&entry2.tags).count();
        let tag_similarity = if tag_union > 0 {
            tag_intersection as f64 / tag_union as f64
        } else {
            0.0
        };
        similarity_score += tag_similarity * 0.3;
        weight_sum += 0.3;

        // Importance similarity (weight: 0.1)
        let importance_diff = (entry1.importance - entry2.importance).abs();
        let importance_similarity = (1.0 - importance_diff).max(0.0);
        similarity_score += importance_similarity * 0.1;
        weight_sum += 0.1;

        // Temporal proximity (weight: 0.2)
        let time_diff = (entry1.created_at - entry2.created_at).num_hours().abs() as f64;
        let temporal_similarity = (1.0 / (1.0 + time_diff / 24.0)).max(0.0); // Decay over days
        similarity_score += temporal_similarity * 0.2;
        weight_sum += 0.2;

        if weight_sum > 0.0 {
            similarity_score / weight_sum
        } else {
            0.0
        }
    }

    /// Add related memories to search results using comprehensive relationship analysis
    async fn add_related_memories(&self, mut results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        for result in &mut results {
            let related_memories = self.find_related_memories(&result.memory.key).await?;
            result.related_memories = related_memories;
        }
        Ok(results)
    }

    /// Find related memories for a given memory key
    async fn find_related_memories(&self, memory_key: &str) -> Result<Vec<RelatedMemoryRef>> {
        let mut related = Vec::new();

        if let Some(source_entry) = self.search_index.metadata_index.get(memory_key) {
            // Find memories with similar content, tags, or temporal proximity
            for (other_key, other_entry) in &self.search_index.metadata_index {
                if other_key == memory_key {
                    continue; // Skip self
                }

                let similarity = self.calculate_memory_similarity(source_entry, other_entry);

                if similarity > 0.3 { // Threshold for related memories
                    let relationship_type = self.determine_relationship_type(source_entry, other_entry, similarity);

                    related.push(RelatedMemoryRef {
                        memory_key: other_key.clone(),
                        relationship_type,
                        strength: similarity,
                    });
                }
            }
        }

        // Sort by strength and limit to top 10
        related.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        related.truncate(10);

        Ok(related)
    }

    /// Determine the type of relationship between two memories
    fn determine_relationship_type(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry, similarity: f64) -> String {
        // Tag-based relationships
        let tag_overlap = entry1.tags.intersection(&entry2.tags).count();
        if tag_overlap > 0 {
            return "tag_similarity".to_string();
        }

        // Content-based relationships
        let content_overlap = entry1.content_words.intersection(&entry2.content_words).count();
        if content_overlap > 5 {
            return "content_similarity".to_string();
        }

        // Temporal relationships
        let time_diff = (entry1.created_at - entry2.created_at).num_hours().abs();
        if time_diff < 24 {
            return "temporal_proximity".to_string();
        }

        // Importance-based relationships
        let importance_diff = (entry1.importance - entry2.importance).abs();
        if importance_diff < 0.1 {
            return "importance_similarity".to_string();
        }

        // Default to general similarity
        if similarity > 0.7 {
            "high_similarity".to_string()
        } else if similarity > 0.5 {
            "medium_similarity".to_string()
        } else {
            "low_similarity".to_string()
        }
    }

    /// Calculate graph centrality score for a memory (measures how connected it is)
    async fn calculate_graph_centrality(&self, memory_key: &str) -> Result<f64> {
        // In a full implementation, this would:
        // 1. Use the knowledge graph to calculate centrality metrics (betweenness, closeness, PageRank)
        // 2. Consider the memory's position in the graph structure
        // 3. Weight by relationship strengths and types

        // For now, implement a simplified version based on relationship count and strength
        let related_memories = self.find_related_memories(memory_key).await?;

        if related_memories.is_empty() {
            return Ok(0.1); // Low centrality for isolated memories
        }

        // Calculate centrality based on:
        // 1. Number of relationships (breadth)
        // 2. Average relationship strength (quality)
        // 3. Diversity of relationship types

        let relationship_count = related_memories.len() as f64;
        let avg_strength: f64 = related_memories.iter().map(|r| r.strength).sum::<f64>() / relationship_count;

        let unique_relationship_types: std::collections::HashSet<_> =
            related_memories.iter().map(|r| &r.relationship_type).collect();
        let relationship_diversity = unique_relationship_types.len() as f64;

        // Combine factors with weights
        let centrality_score = (
            (relationship_count / 20.0).min(1.0) * 0.4 +  // Normalize to max 20 relationships
            avg_strength * 0.4 +
            (relationship_diversity / 5.0).min(1.0) * 0.2  // Normalize to max 5 types
        ).min(1.0);

        Ok(centrality_score)
    }

    /// Calculate user preference score for a memory
    async fn calculate_user_preference_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // In a full implementation, this would:
        // 1. Analyze user interaction patterns (clicks, time spent, bookmarks)
        // 2. Consider user-defined preferences and tags
        // 3. Use machine learning to predict user interest
        // 4. Factor in collaborative filtering from similar users

        // For now, implement a simplified version based on:
        // 1. Access frequency (more accessed = higher preference)
        // 2. Recency of access (recently accessed = higher preference)
        // 3. Memory importance (user-assigned importance)
        // 4. Tag preferences (simulate based on common tags)

        let access_count = memory.access_count() as f64;
        let access_frequency_score = (access_count / 50.0).min(1.0); // Normalize to max 50 accesses

        let time_since_access = Utc::now() - memory.last_accessed();
        let recency_score = if time_since_access.num_days() == 0 {
            1.0
        } else {
            (1.0 / (1.0 + time_since_access.num_days() as f64 / 7.0)).max(0.1) // Decay over weeks
        };

        let importance_score = memory.metadata.importance;

        // Simulate tag preferences (in reality, this would be learned from user behavior)
        let preferred_tags = ["important", "work", "project", "research", "personal"];
        let tag_preference_score = memory.metadata.tags.iter()
            .filter(|tag| preferred_tags.contains(&tag.as_str()))
            .count() as f64 / preferred_tags.len() as f64;

        // Combine factors with weights
        let preference_score = (
            access_frequency_score * 0.3 +
            recency_score * 0.3 +
            importance_score * 0.2 +
            tag_preference_score * 0.2
        ).min(1.0);

        Ok(preference_score)
    }

    /// Perform semantic search using embeddings (if available)
    pub async fn semantic_search(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        query_embedding: &[f32],
        limit: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        if !self.config.enable_semantic_search {
            return Err(MemoryError::configuration("Semantic search not enabled"));
        }

        let mut results = Vec::new();

        // Find memories with embeddings and calculate cosine similarity
        for (memory_key, index_entry) in &self.search_index.metadata_index {
            // In a full implementation, embeddings would be stored in the index
            // For now, simulate semantic similarity based on content overlap
            let semantic_score = self.simulate_semantic_similarity(query_embedding, index_entry);

            if semantic_score > 0.3 { // Threshold for semantic relevance
                // Retrieve the actual memory from storage
                if let Some(memory) = storage.retrieve(memory_key).await? {
                    results.push(SearchResult {
                        memory,
                        relevance_score: semantic_score,
                        ranking_score: semantic_score,
                        highlights: Vec::new(), // Semantic search doesn't have text highlights
                        explanation: format!("Semantic similarity: {:.2}", semantic_score),
                        related_memories: Vec::new(),
                        search_metadata: SearchResultMetadata {
                            generated_at: Utc::now(),
                            algorithm: "semantic_search".to_string(),
                            processing_time_ms: 0,
                            debug_info: HashMap::new(),
                        },
                    });
                } else {
                    tracing::warn!("Memory '{}' found in search index but not in storage", memory_key);
                }
            }
        }

        // Sort by semantic similarity
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        // Apply limit
        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Simulate semantic similarity (placeholder for actual embedding comparison)
    fn simulate_semantic_similarity(&self, _query_embedding: &[f32], index_entry: &MemoryIndexEntry) -> f64 {
        // In a real implementation, this would calculate cosine similarity between embeddings
        // For now, simulate based on content characteristics

        let content_richness = (index_entry.content_length as f64 / 1000.0).min(1.0);
        let importance_factor = index_entry.importance;
        let tag_diversity = (index_entry.tags.len() as f64 / 10.0).min(1.0);

        // Combine factors to simulate semantic relevance
        (content_richness * 0.4 + importance_factor * 0.4 + tag_diversity * 0.2).min(1.0)
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, search_time_ms: u64) {
        self.performance_metrics.total_searches += 1;
        
        let total_time = self.performance_metrics.avg_search_time_ms * (self.performance_metrics.total_searches - 1) as f64;
        self.performance_metrics.avg_search_time_ms = (total_time + search_time_ms as f64) / self.performance_metrics.total_searches as f64;
    }

    /// Update the search index with new memories
    pub async fn update_index(&mut self, memories: &[MemoryEntry]) -> Result<()> {
        for memory in memories {
            self.search_index.add_memory(memory).await?;
        }
        Ok(())
    }

    /// Get search performance metrics
    pub fn get_performance_metrics(&self) -> &SearchPerformanceMetrics {
        &self.performance_metrics
    }
}

impl SearchIndex {
    fn new() -> Self {
        Self {
            word_index: HashMap::new(),
            metadata_index: HashMap::new(),
            last_updated: Utc::now(),
        }
    }

    async fn add_memory(&mut self, memory: &MemoryEntry) -> Result<()> {
        // Extract words from content
        let words: HashSet<String> = memory.value
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        // Update word index
        for word in &words {
            self.word_index
                .entry(word.clone())
                .or_insert_with(HashSet::new)
                .insert(memory.key.clone());
        }

        // Update metadata index
        let index_entry = MemoryIndexEntry {
            memory_key: memory.key.clone(),
            content_words: words,
            tags: memory.metadata.tags.iter().cloned().collect(),
            importance: memory.metadata.importance,
            confidence: memory.metadata.confidence,
            created_at: memory.created_at(),
            last_accessed: memory.last_accessed(),
            access_count: memory.access_count(),
            content_length: memory.value.len(),
        };

        self.metadata_index.insert(memory.key.clone(), index_entry);
        self.last_updated = Utc::now();

        Ok(())
    }
}

impl Default for AdvancedSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}
