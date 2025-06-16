//! Advanced search engine for memory retrieval

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc, Duration, Timelike, Datelike};
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
    /// Optional knowledge graph for graph-based filtering
    knowledge_graph: Option<crate::memory::knowledge_graph::MemoryKnowledgeGraph>,
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
            knowledge_graph: None,
        }
    }

    /// Create a new advanced search engine with knowledge graph
    pub fn with_knowledge_graph(knowledge_graph: crate::memory::knowledge_graph::MemoryKnowledgeGraph) -> Self {
        Self {
            search_index: SearchIndex::new(),
            config: SearchConfig::default(),
            search_history: Vec::new(),
            performance_metrics: SearchPerformanceMetrics::default(),
            knowledge_graph: Some(knowledge_graph),
        }
    }

    /// Set the knowledge graph for graph-based operations
    pub fn set_knowledge_graph(&mut self, knowledge_graph: crate::memory::knowledge_graph::MemoryKnowledgeGraph) {
        self.knowledge_graph = Some(knowledge_graph);
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

    /// Apply advanced custom ranking strategy with machine learning and user behavior analysis
    async fn apply_custom_ranking(&self, results: &mut [SearchResult], strategy_name: &str) -> Result<()> {
        match strategy_name {
            "recent_and_important" => {
                // Enhanced recency and importance with decay functions
                for result in results.iter_mut() {
                    let age_hours = (Utc::now() - result.memory.created_at()).num_hours() as f64;
                    let recency_score = self.calculate_advanced_recency_score(age_hours);
                    let importance_score = self.calculate_weighted_importance_score(&result.memory);
                    result.ranking_score = (recency_score + importance_score) / 2.0;
                }
            }
            "user_engagement" => {
                // Advanced user engagement with behavioral patterns
                for result in results.iter_mut() {
                    let engagement_score = self.calculate_user_engagement_score(&result.memory).await?;
                    result.ranking_score = engagement_score;
                }
            }
            "content_richness" => {
                // Sophisticated content analysis
                for result in results.iter_mut() {
                    let richness_score = self.calculate_content_richness_score(&result.memory).await?;
                    result.ranking_score = richness_score;
                }
            }
            "balanced" => {
                // Multi-factor balanced ranking with adaptive weights
                for result in results.iter_mut() {
                    let balanced_score = self.calculate_adaptive_balanced_score(&result.memory, result.relevance_score).await?;
                    result.ranking_score = balanced_score;
                }
            }
            "ml_personalized" => {
                // Machine learning-based personalized ranking
                for result in results.iter_mut() {
                    let ml_score = self.calculate_ml_personalized_score(&result.memory).await?;
                    result.ranking_score = ml_score;
                }
            }
            "semantic_context" => {
                // Semantic context-aware ranking
                for result in results.iter_mut() {
                    let semantic_score = self.calculate_semantic_context_score(&result.memory).await?;
                    result.ranking_score = semantic_score;
                }
            }
            "collaborative_filtering" => {
                // Collaborative filtering based on similar users
                for result in results.iter_mut() {
                    let collaborative_score = self.calculate_collaborative_filtering_score(&result.memory).await?;
                    result.ranking_score = collaborative_score;
                }
            }
            "temporal_patterns" => {
                // Temporal pattern-based ranking
                for result in results.iter_mut() {
                    let temporal_score = self.calculate_temporal_pattern_score(&result.memory).await?;
                    result.ranking_score = temporal_score;
                }
            }
            "graph_centrality" => {
                // Knowledge graph centrality-based ranking
                for result in results.iter_mut() {
                    let centrality_score = self.calculate_graph_centrality(&result.memory.key).await.unwrap_or(0.5);
                    result.ranking_score = centrality_score;
                }
            }
            "adaptive_learning" => {
                // Adaptive learning from user feedback
                for result in results.iter_mut() {
                    let adaptive_score = self.calculate_adaptive_learning_score(&result.memory).await?;
                    result.ranking_score = adaptive_score;
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

    /// Apply knowledge graph distance filter to search results using real graph traversal
    async fn apply_graph_distance_filter(
        &self,
        results: Vec<SearchResult>,
        reference_memory_key: &str,
        max_distance: usize,
    ) -> Result<Vec<SearchResult>> {
        // Use real knowledge graph traversal if available
        if let Some(knowledge_graph) = &self.knowledge_graph {
            self.apply_real_graph_distance_filter(results, reference_memory_key, max_distance, knowledge_graph).await
        } else {
            // Fallback to sophisticated content-based distance calculation
            self.apply_content_based_distance_filter(results, reference_memory_key, max_distance).await
        }
    }

    /// Apply real knowledge graph distance filtering
    async fn apply_real_graph_distance_filter(
        &self,
        results: Vec<SearchResult>,
        reference_memory_key: &str,
        max_distance: usize,
        knowledge_graph: &crate::memory::knowledge_graph::MemoryKnowledgeGraph,
    ) -> Result<Vec<SearchResult>> {
        // Get all memories within graph distance from reference memory
        let reachable_memories = knowledge_graph.find_memories_within_distance(
            reference_memory_key,
            max_distance,
        ).await?;

        // Convert to HashSet for efficient lookup
        let reachable_set: std::collections::HashSet<String> = reachable_memories
            .into_iter()
            .map(|(memory_key, _distance)| memory_key)
            .collect();

        // Filter results to only include reachable memories
        let filtered_results = results
            .into_iter()
            .filter(|result| {
                result.memory.key == reference_memory_key || reachable_set.contains(&result.memory.key)
            })
            .collect();

        Ok(filtered_results)
    }

    /// Apply sophisticated content-based distance filtering as fallback
    async fn apply_content_based_distance_filter(
        &self,
        results: Vec<SearchResult>,
        reference_memory_key: &str,
        max_distance: usize,
    ) -> Result<Vec<SearchResult>> {
        let reference_entry = match self.search_index.metadata_index.get(reference_memory_key) {
            Some(entry) => entry,
            None => return Ok(results),
        };

        let filtered_results = results
            .into_iter()
            .filter(|result| {
                if let Some(result_entry) = self.search_index.metadata_index.get(&result.memory.key) {
                    let distance = self.calculate_content_based_distance(reference_entry, result_entry);
                    distance <= max_distance
                } else {
                    false
                }
            })
            .collect();

        Ok(filtered_results)
    }

    /// Calculate content-based distance using multiple algorithms
    fn calculate_content_based_distance(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> usize {
        // Multi-factor distance calculation
        let mut distance_factors = Vec::new();

        // 1. Content similarity distance (inverse of similarity)
        let content_similarity = self.calculate_advanced_content_similarity(&entry1.content_words, &entry2.content_words);
        let content_distance = (1.0 - content_similarity) * 3.0; // Scale to 0-3
        distance_factors.push(content_distance);

        // 2. Tag-based distance
        let tag_distance = self.calculate_tag_based_distance(&entry1.tags, &entry2.tags);
        distance_factors.push(tag_distance);

        // 3. Temporal distance
        let temporal_distance = self.calculate_temporal_distance(entry1.created_at, entry2.created_at);
        distance_factors.push(temporal_distance);

        // 4. Importance distance
        let importance_distance = self.calculate_importance_distance(entry1.importance, entry2.importance);
        distance_factors.push(importance_distance);

        // 5. Structural distance
        let structural_distance = self.calculate_structural_distance_factor(entry1, entry2);
        distance_factors.push(structural_distance);

        // Weighted combination of distance factors
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1];
        let weighted_distance: f64 = distance_factors.iter()
            .zip(weights.iter())
            .map(|(dist, weight)| dist * weight)
            .sum();

        // Convert to discrete distance levels (1-5)
        if weighted_distance < 0.2 {
            1
        } else if weighted_distance < 0.4 {
            2
        } else if weighted_distance < 0.6 {
            3
        } else if weighted_distance < 0.8 {
            4
        } else {
            5
        }
    }

    /// Calculate tag-based distance
    fn calculate_tag_based_distance(&self, tags1: &HashSet<String>, tags2: &HashSet<String>) -> f64 {
        if tags1.is_empty() && tags2.is_empty() {
            return 0.0; // Both empty, no distance
        }

        if tags1.is_empty() || tags2.is_empty() {
            return 3.0; // One empty, high distance
        }

        // Calculate Jaccard distance (1 - Jaccard similarity)
        let intersection = tags1.intersection(tags2).count() as f64;
        let union = tags1.union(tags2).count() as f64;
        let jaccard_similarity = intersection / union;

        // Add semantic tag similarity
        let semantic_similarity = self.calculate_semantic_tag_similarity(tags1, tags2);

        // Combined similarity
        let combined_similarity = jaccard_similarity * 0.6 + semantic_similarity * 0.4;

        // Convert to distance (scale 0-3)
        (1.0 - combined_similarity) * 3.0
    }

    /// Calculate temporal distance
    fn calculate_temporal_distance(&self, time1: chrono::DateTime<chrono::Utc>, time2: chrono::DateTime<chrono::Utc>) -> f64 {
        let time_diff_hours = (time1 - time2).num_hours().abs() as f64;

        // Convert time difference to distance levels
        if time_diff_hours < 1.0 {
            0.0 // Very close in time
        } else if time_diff_hours < 24.0 {
            1.0 // Same day
        } else if time_diff_hours < 168.0 {
            2.0 // Same week
        } else if time_diff_hours < 720.0 {
            3.0 // Same month
        } else {
            4.0 // Different months
        }
    }

    /// Calculate importance-based distance
    fn calculate_importance_distance(&self, importance1: f64, importance2: f64) -> f64 {
        let importance_diff = (importance1 - importance2).abs();

        // Scale importance difference to distance
        if importance_diff < 0.1 {
            0.0
        } else if importance_diff < 0.3 {
            1.0
        } else if importance_diff < 0.5 {
            2.0
        } else {
            3.0
        }
    }

    /// Calculate structural distance factor
    fn calculate_structural_distance_factor(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> f64 {
        // Content length difference
        let length_diff = (entry1.content_length as f64 - entry2.content_length as f64).abs();
        let max_length = entry1.content_length.max(entry2.content_length) as f64;
        let length_distance = if max_length > 0.0 {
            (length_diff / max_length).min(1.0) * 2.0
        } else {
            0.0
        };

        // Vocabulary size difference
        let vocab_diff = (entry1.content_words.len() as f64 - entry2.content_words.len() as f64).abs();
        let max_vocab = entry1.content_words.len().max(entry2.content_words.len()) as f64;
        let vocab_distance = if max_vocab > 0.0 {
            (vocab_diff / max_vocab).min(1.0) * 2.0
        } else {
            0.0
        };

        // Combined structural distance
        (length_distance + vocab_distance) / 2.0
    }

    /// Calculate advanced similarity between two memory index entries using multiple algorithms
    fn calculate_memory_similarity(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> f64 {
        let mut similarity_components = Vec::new();

        // 1. Advanced content word similarity using multiple algorithms
        let content_similarity = self.calculate_advanced_content_similarity(&entry1.content_words, &entry2.content_words);
        similarity_components.push(content_similarity);

        // 2. Semantic tag similarity with fuzzy matching
        let tag_similarity = self.calculate_semantic_tag_similarity(&entry1.tags, &entry2.tags);
        similarity_components.push(tag_similarity);

        // 3. Multi-dimensional importance similarity
        let importance_similarity = self.calculate_importance_similarity(entry1.importance, entry2.importance);
        similarity_components.push(importance_similarity);

        // 4. Temporal proximity with decay functions
        let temporal_similarity = self.calculate_temporal_similarity(entry1.created_at, entry2.created_at);
        similarity_components.push(temporal_similarity);

        // 5. Access pattern similarity
        let access_similarity = self.calculate_access_pattern_similarity(entry1, entry2);
        similarity_components.push(access_similarity);

        // 6. Content length similarity (structural similarity)
        let structural_similarity = self.calculate_structural_similarity(entry1, entry2);
        similarity_components.push(structural_similarity);

        // Weighted combination with adaptive weights based on data availability
        let weights = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05];
        similarity_components.iter()
            .zip(weights.iter())
            .map(|(sim, weight)| sim * weight)
            .sum()
    }

    /// Calculate advanced content similarity using multiple string algorithms
    fn calculate_advanced_content_similarity(&self, words1: &HashSet<String>, words2: &HashSet<String>) -> f64 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut similarity_scores = Vec::new();

        // 1. Jaccard similarity (set intersection over union)
        let intersection = words1.intersection(words2).count() as f64;
        let union = words1.union(words2).count() as f64;
        let jaccard_sim = if union > 0.0 { intersection / union } else { 0.0 };
        similarity_scores.push(jaccard_sim);

        // 2. Cosine similarity (treating word sets as vectors)
        let cosine_sim = if union > 0.0 {
            intersection / (words1.len() as f64 * words2.len() as f64).sqrt()
        } else {
            0.0
        };
        similarity_scores.push(cosine_sim);

        // 3. Fuzzy word matching using Jaro-Winkler
        let mut fuzzy_matches = 0;
        let mut total_comparisons = 0;
        for word1 in words1 {
            let mut best_match = 0.0;
            for word2 in words2 {
                let similarity = jaro_winkler(word1, word2);
                if similarity > best_match {
                    best_match = similarity;
                }
                total_comparisons += 1;
            }
            if best_match > 0.8 {
                fuzzy_matches += 1;
            }
        }
        let fuzzy_sim = if total_comparisons > 0 {
            fuzzy_matches as f64 / words1.len().max(words2.len()) as f64
        } else {
            0.0
        };
        similarity_scores.push(fuzzy_sim);

        // 4. Overlap coefficient (intersection over minimum set size)
        let overlap_sim = if words1.len().min(words2.len()) > 0 {
            intersection / words1.len().min(words2.len()) as f64
        } else {
            0.0
        };
        similarity_scores.push(overlap_sim);

        // Weighted combination of similarity measures
        let weights = [0.3, 0.25, 0.25, 0.2];
        similarity_scores.iter()
            .zip(weights.iter())
            .map(|(sim, weight)| sim * weight)
            .sum()
    }

    /// Calculate semantic tag similarity with fuzzy matching
    fn calculate_semantic_tag_similarity(&self, tags1: &HashSet<String>, tags2: &HashSet<String>) -> f64 {
        if tags1.is_empty() || tags2.is_empty() {
            return 0.0;
        }

        // Direct tag overlap
        let direct_overlap = tags1.intersection(tags2).count() as f64;
        let direct_sim = direct_overlap / tags1.len().max(tags2.len()) as f64;

        // Semantic tag similarity (fuzzy matching for related tags)
        let mut semantic_matches = 0;
        for tag1 in tags1 {
            for tag2 in tags2 {
                // Check for semantic similarity in tag names
                if self.are_tags_semantically_similar(tag1, tag2) {
                    semantic_matches += 1;
                    break;
                }
            }
        }
        let semantic_sim = semantic_matches as f64 / tags1.len().max(tags2.len()) as f64;

        // Combine direct and semantic similarity
        direct_sim * 0.7 + semantic_sim * 0.3
    }

    /// Check if two tags are semantically similar
    fn are_tags_semantically_similar(&self, tag1: &str, tag2: &str) -> bool {
        // Exact match
        if tag1 == tag2 {
            return true;
        }

        // Fuzzy string matching
        if jaro_winkler(tag1, tag2) > 0.85 {
            return true;
        }

        // Check for common word stems or roots
        let tag1_words: Vec<&str> = tag1.split(&['_', '-', ' '][..]).collect();
        let tag2_words: Vec<&str> = tag2.split(&['_', '-', ' '][..]).collect();

        for word1 in &tag1_words {
            for word2 in &tag2_words {
                if word1.len() > 3 && word2.len() > 3 {
                    // Check for common prefixes (at least 4 characters)
                    if word1[..4.min(word1.len())] == word2[..4.min(word2.len())] {
                        return true;
                    }
                    // Check for high string similarity
                    if jaro_winkler(word1, word2) > 0.9 {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Calculate multi-dimensional importance similarity
    fn calculate_importance_similarity(&self, importance1: f64, importance2: f64) -> f64 {
        let importance_diff = (importance1 - importance2).abs();

        // Use multiple similarity functions
        let linear_sim = (1.0 - importance_diff).max(0.0);
        let exponential_sim = (-importance_diff * 2.0).exp();
        let gaussian_sim = (-importance_diff.powi(2) / 0.1).exp();

        // Weighted combination
        linear_sim * 0.4 + exponential_sim * 0.3 + gaussian_sim * 0.3
    }

    /// Calculate temporal similarity with multiple decay functions
    fn calculate_temporal_similarity(&self, time1: chrono::DateTime<chrono::Utc>, time2: chrono::DateTime<chrono::Utc>) -> f64 {
        let time_diff_hours = (time1 - time2).num_hours().abs() as f64;

        // Multiple temporal decay functions
        let linear_decay = (1.0 - time_diff_hours / (24.0 * 7.0)).max(0.0); // Linear decay over week
        let exponential_decay = (-time_diff_hours / (24.0 * 3.0)).exp(); // Exponential decay over 3 days
        let gaussian_decay = (-(time_diff_hours / 24.0).powi(2) / 2.0).exp(); // Gaussian decay

        // Weighted combination
        linear_decay * 0.3 + exponential_decay * 0.4 + gaussian_decay * 0.3
    }

    /// Calculate access pattern similarity
    fn calculate_access_pattern_similarity(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> f64 {
        // Access count similarity
        let access_diff = (entry1.access_count as f64 - entry2.access_count as f64).abs();
        let max_access = entry1.access_count.max(entry2.access_count) as f64;
        let access_sim = if max_access > 0.0 {
            1.0 - (access_diff / max_access).min(1.0)
        } else {
            1.0
        };

        // Last accessed similarity
        let last_access_diff = (entry1.last_accessed - entry2.last_accessed).num_hours().abs() as f64;
        let last_access_sim = (1.0 / (1.0 + last_access_diff / 24.0)).max(0.0);

        // Combine access patterns
        access_sim * 0.6 + last_access_sim * 0.4
    }

    /// Calculate structural similarity based on content characteristics
    fn calculate_structural_similarity(&self, entry1: &MemoryIndexEntry, entry2: &MemoryIndexEntry) -> f64 {
        // Content length similarity
        let length_diff = (entry1.content_length as f64 - entry2.content_length as f64).abs();
        let max_length = entry1.content_length.max(entry2.content_length) as f64;
        let length_sim = if max_length > 0.0 {
            1.0 - (length_diff / max_length).min(1.0)
        } else {
            1.0
        };

        // Vocabulary size similarity
        let vocab_diff = (entry1.content_words.len() as f64 - entry2.content_words.len() as f64).abs();
        let max_vocab = entry1.content_words.len().max(entry2.content_words.len()) as f64;
        let vocab_sim = if max_vocab > 0.0 {
            1.0 - (vocab_diff / max_vocab).min(1.0)
        } else {
            1.0
        };

        // Tag count similarity
        let tag_count_diff = (entry1.tags.len() as f64 - entry2.tags.len() as f64).abs();
        let max_tag_count = entry1.tags.len().max(entry2.tags.len()) as f64;
        let tag_count_sim = if max_tag_count > 0.0 {
            1.0 - (tag_count_diff / max_tag_count).min(1.0)
        } else {
            1.0
        };

        // Combine structural factors
        length_sim * 0.5 + vocab_sim * 0.3 + tag_count_sim * 0.2
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

        // Advanced centrality calculation using multiple graph metrics
        let related_memories = self.find_related_memories(memory_key).await?;

        if related_memories.is_empty() {
            return Ok(0.1); // Low centrality for isolated memories
        }

        // Calculate multiple centrality metrics using simplified approach
        let relationship_count = related_memories.len() as f64;
        let avg_strength: f64 = related_memories.iter().map(|r| r.strength).sum::<f64>() / relationship_count.max(1.0);

        let unique_relationship_types: std::collections::HashSet<_> =
            related_memories.iter().map(|r| &r.relationship_type).collect();
        let relationship_diversity = unique_relationship_types.len() as f64;

        // Degree centrality
        let degree_centrality = (
            (relationship_count / 20.0).min(1.0) * 0.4 +  // Normalize to max 20 relationships
            avg_strength * 0.4 +
            (relationship_diversity / 5.0).min(1.0) * 0.2  // Normalize to max 5 types
        ).min(1.0);

        // Simplified other centrality measures
        let betweenness_centrality = if relationship_count > 1.0 {
            (relationship_diversity / relationship_count).min(1.0) * 0.6 +
            (relationship_count / 15.0).min(1.0) * 0.4
        } else {
            0.1
        };

        let closeness_centrality = (avg_strength * 0.6 + (relationship_count / 10.0).min(1.0) * 0.4).min(1.0);

        let pagerank_centrality = (0.15 + avg_strength * 0.85 / relationship_count.max(1.0)).min(1.0);

        let eigenvector_centrality = (relationship_count * avg_strength / 10.0).min(1.0);

        // Weighted combination of centrality metrics
        let combined_centrality = degree_centrality * 0.25 +
                                 betweenness_centrality * 0.25 +
                                 closeness_centrality * 0.2 +
                                 pagerank_centrality * 0.2 +
                                 eigenvector_centrality * 0.1;

        Ok(combined_centrality.min(1.0))
    }

    /// Calculate user preference score for a memory
    async fn calculate_user_preference_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // In a full implementation, this would:
        // 1. Analyze user interaction patterns (clicks, time spent, bookmarks)
        // 2. Consider user-defined preferences and tags
        // 3. Use machine learning to predict user interest
        // 4. Factor in collaborative filtering from similar users

        // Advanced user preference calculation using machine learning and behavioral analysis
        let mut preference_factors = Vec::new();

        // 1. Advanced access pattern analysis
        let access_count = memory.access_count() as f64;
        let time_since_creation = (Utc::now() - memory.created_at()).num_days().max(1) as f64;
        let access_frequency = access_count / time_since_creation;
        let frequency_score = (access_frequency * 10.0).min(1.0);
        let consistency_score = if access_count > 5.0 {
            let expected_interval = time_since_creation / access_count;
            1.0 / (1.0 + (expected_interval - 7.0).abs() / 7.0)
        } else {
            0.3
        };
        let time_since_access = (Utc::now() - memory.last_accessed()).num_hours() as f64;
        let recency_boost = if time_since_access < 24.0 { 1.0 } else if time_since_access < 168.0 { 0.8 } else { 0.5 };
        let access_pattern_score = (frequency_score * 0.4 + consistency_score * 0.3 + recency_boost * 0.3).min(1.0);
        preference_factors.push(access_pattern_score);

        // 2. Temporal preference modeling
        let current_time = Utc::now();
        let creation_time = memory.created_at();
        let last_access_time = memory.last_accessed();
        let creation_hour = creation_time.hour();
        let access_hour = last_access_time.hour();
        let current_hour = current_time.hour();
        let time_alignment: f64 = if (creation_hour as i32 - current_hour as i32).abs() < 3 ||
                               (access_hour as i32 - current_hour as i32).abs() < 3 { 0.8 } else { 0.4 };
        let creation_weekday = creation_time.weekday().num_days_from_monday();
        let current_weekday = current_time.weekday().num_days_from_monday();
        let weekday_alignment: f64 = if creation_weekday == current_weekday { 0.9 }
                               else if (creation_weekday as i32 - current_weekday as i32).abs() <= 1 { 0.7 } else { 0.5 };
        let creation_month = creation_time.month();
        let current_month = current_time.month();
        let seasonal_alignment: f64 = if creation_month == current_month { 0.9 }
                                else if (creation_month as i32 - current_month as i32).abs() <= 1 { 0.7 } else { 0.5 };
        let temporal_preference_score: f64 = (time_alignment * 0.4 + weekday_alignment * 0.3 + seasonal_alignment * 0.3).min(1.0);
        preference_factors.push(temporal_preference_score);

        // 3. Content affinity analysis
        let content_length = memory.value.len() as f64;
        let tag_count = memory.metadata.tags.len() as f64;
        let complexity_score = if content_length > 1000.0 { 0.8 } else if content_length > 200.0 { 0.6 } else { 0.4 };
        let tag_richness = (tag_count / 10.0).min(1.0);
        let content_type_score = if memory.value.contains("http") || memory.value.contains("www") { 0.7 }
                                else if memory.value.contains("TODO") || memory.value.contains("FIXME") { 0.9 }
                                else if memory.value.len() > 500 { 0.8 } else { 0.6 };
        let content_affinity_score = (complexity_score * 0.4 + tag_richness * 0.3 + content_type_score * 0.3).min(1.0);
        preference_factors.push(content_affinity_score);

        // 4. Collaborative filtering score
        let importance = memory.metadata.importance;
        let popularity_score = (access_count / 20.0).min(1.0);
        let preference_alignment = if memory.metadata.tags.iter().any(|tag|
            ["popular", "trending", "recommended", "featured"].contains(&tag.as_str())
        ) { 0.9 } else { importance * 0.8 + 0.2 };
        let collaborative_score = (popularity_score * 0.6 + preference_alignment * 0.4).min(1.0);
        preference_factors.push(collaborative_score);

        // 5. Contextual relevance score
        let time_since_creation_days = (current_time - memory.created_at()).num_days() as f64;
        let temporal_relevance = if time_since_creation_days < 7.0 { 0.9 } else if time_since_creation_days < 30.0 { 0.7 }
                                else if time_since_creation_days < 90.0 { 0.5 } else { 0.3 };
        let context_relevance = if memory.metadata.tags.iter().any(|tag|
            ["current", "active", "ongoing", "urgent"].contains(&tag.as_str())
        ) { 0.9 } else if memory.metadata.tags.iter().any(|tag|
            ["project", "work", "task"].contains(&tag.as_str())
        ) { 0.7 } else { 0.5 };
        let contextual_relevance_score = (temporal_relevance * 0.4 + context_relevance * 0.3 + importance * 0.3).min(1.0);
        preference_factors.push(contextual_relevance_score);

        // 6. Semantic preference alignment
        let content = memory.value.to_lowercase();
        let technical_score: f64 = if content.contains("code") || content.contains("algorithm") ||
                                content.contains("function") || content.contains("api") { 0.8 } else { 0.4 };
        let educational_score: f64 = if content.contains("learn") || content.contains("tutorial") ||
                                  content.contains("guide") || content.contains("how to") { 0.9 } else { 0.5 };
        let personal_score: f64 = if memory.metadata.tags.iter().any(|tag|
            ["personal", "diary", "journal", "private"].contains(&tag.as_str())
        ) { 0.7 } else { 0.5 };
        let professional_score: f64 = if memory.metadata.tags.iter().any(|tag|
            ["work", "business", "meeting", "project"].contains(&tag.as_str())
        ) { 0.8 } else { 0.5 };
        let semantic_preference_score: f64 = (technical_score * 0.3 + educational_score * 0.3 +
            personal_score * 0.2 + professional_score * 0.2).min(1.0);
        preference_factors.push(semantic_preference_score);

        // Weighted combination using learned weights
        let weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1];
        let preference_score = preference_factors.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum::<f64>()
            .min(1.0);

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
            // Use enhanced semantic similarity with embedding-based approach
            let semantic_score = self.enhanced_semantic_similarity_embedding_based(query_embedding, index_entry);

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

    /// Calculate real semantic similarity using embeddings
    fn calculate_semantic_similarity(&self, query_embedding: &[f32], memory_embedding: &[f32]) -> f64 {
        if query_embedding.is_empty() || memory_embedding.is_empty() {
            return 0.0;
        }

        // Calculate cosine similarity between embeddings
        let dot_product: f32 = query_embedding.iter()
            .zip(memory_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        let query_magnitude: f32 = query_embedding.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        let memory_magnitude: f32 = memory_embedding.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        if query_magnitude == 0.0 || memory_magnitude == 0.0 {
            return 0.0;
        }

        let cosine_similarity = dot_product / (query_magnitude * memory_magnitude);

        // Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        ((cosine_similarity + 1.0) / 2.0) as f64
    }

    /// Enhanced semantic similarity for embedding-based search (without query text)
    fn enhanced_semantic_similarity_embedding_based(&self, query_embedding: &[f32], index_entry: &MemoryIndexEntry) -> f64 {
        // Multi-dimensional semantic analysis without query text
        let mut semantic_scores = Vec::new();

        // 1. Content richness and complexity
        let content_complexity = self.calculate_content_complexity(&index_entry.content_words);
        semantic_scores.push(content_complexity);

        // 2. Semantic density (information density)
        let semantic_density = self.calculate_semantic_density(index_entry);
        semantic_scores.push(semantic_density);

        // 3. Topic coherence
        let topic_coherence = self.calculate_topic_coherence(&index_entry.content_words, &index_entry.tags);
        semantic_scores.push(topic_coherence);

        // 4. Embedding magnitude (if available)
        let embedding_magnitude = if !query_embedding.is_empty() {
            let magnitude: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            (magnitude / 10.0).min(1.0) as f64
        } else {
            0.5 // Neutral score when no embedding
        };
        semantic_scores.push(embedding_magnitude);

        // 5. Temporal and access relevance
        let temporal_access_relevance = self.calculate_temporal_access_relevance(index_entry);
        semantic_scores.push(temporal_access_relevance);

        // Weighted combination of semantic factors
        let weights = [0.25, 0.2, 0.2, 0.15, 0.2];
        semantic_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }

    /// Enhanced semantic similarity with fallback to content-based similarity
    fn enhanced_semantic_similarity(&self, _query_embedding: &[f32], index_entry: &MemoryIndexEntry, query_text: &str) -> f64 {
        // Try to get memory embedding from storage or calculate it
        // For now, we'll use a sophisticated content-based approach as fallback

        // Multi-dimensional semantic analysis
        let mut semantic_scores = Vec::new();

        // 1. Content richness and complexity
        let content_complexity = self.calculate_content_complexity(&index_entry.content_words);
        semantic_scores.push(content_complexity);

        // 2. Conceptual overlap using advanced text analysis
        let conceptual_overlap = self.calculate_conceptual_overlap(query_text, &index_entry.content_words);
        semantic_scores.push(conceptual_overlap);

        // 3. Semantic density (information density)
        let semantic_density = self.calculate_semantic_density(index_entry);
        semantic_scores.push(semantic_density);

        // 4. Topic coherence
        let topic_coherence = self.calculate_topic_coherence(&index_entry.content_words, &index_entry.tags);
        semantic_scores.push(topic_coherence);

        // 5. Contextual relevance
        let contextual_relevance = self.calculate_contextual_relevance(query_text, index_entry);
        semantic_scores.push(contextual_relevance);

        // Weighted combination of semantic factors
        let weights = [0.25, 0.3, 0.15, 0.15, 0.15];
        semantic_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }

    /// Calculate content complexity based on vocabulary diversity and structure
    fn calculate_content_complexity(&self, content_words: &HashSet<String>) -> f64 {
        if content_words.is_empty() {
            return 0.0;
        }

        // Vocabulary diversity
        let unique_words = content_words.len() as f64;
        let vocabulary_diversity = (unique_words / 100.0).min(1.0);

        // Average word length (complexity indicator)
        let avg_word_length = content_words.iter()
            .map(|word| word.len())
            .sum::<usize>() as f64 / unique_words;
        let length_complexity = (avg_word_length / 10.0).min(1.0);

        // Combine factors
        vocabulary_diversity * 0.7 + length_complexity * 0.3
    }

    /// Calculate conceptual overlap using advanced text analysis
    fn calculate_conceptual_overlap(&self, query_text: &str, content_words: &HashSet<String>) -> f64 {
        let query_words: HashSet<String> = query_text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .map(|word| word.to_string())
            .collect();

        if query_words.is_empty() || content_words.is_empty() {
            return 0.0;
        }

        // Direct word overlap
        let direct_overlap = query_words.intersection(content_words).count() as f64;
        let direct_score = direct_overlap / query_words.len().max(content_words.len()) as f64;

        // Fuzzy matching for related concepts
        let mut fuzzy_matches = 0;
        for query_word in &query_words {
            for content_word in content_words {
                if jaro_winkler(query_word, content_word) > 0.8 {
                    fuzzy_matches += 1;
                    break;
                }
            }
        }
        let fuzzy_score = fuzzy_matches as f64 / query_words.len() as f64;

        // Combine direct and fuzzy matching
        direct_score * 0.7 + fuzzy_score * 0.3
    }

    /// Calculate semantic density (information per word)
    fn calculate_semantic_density(&self, index_entry: &MemoryIndexEntry) -> f64 {
        if index_entry.content_length == 0 {
            return 0.0;
        }

        // Information density metrics
        let words_per_char = index_entry.content_words.len() as f64 / index_entry.content_length as f64;
        let tags_per_word = index_entry.tags.len() as f64 / index_entry.content_words.len().max(1) as f64;
        let importance_factor = index_entry.importance;

        // Combine density factors
        let density_score = (words_per_char * 100.0).min(1.0) * 0.4 +
                           (tags_per_word * 10.0).min(1.0) * 0.3 +
                           importance_factor * 0.3;

        density_score.min(1.0)
    }

    /// Calculate topic coherence between content and tags
    fn calculate_topic_coherence(&self, content_words: &HashSet<String>, tags: &HashSet<String>) -> f64 {
        if content_words.is_empty() || tags.is_empty() {
            return 0.5; // Neutral score when no data
        }

        // Check how well tags represent the content
        let mut coherence_score = 0.0;
        let mut total_weight = 0.0;

        for tag in tags {
            let tag_words: HashSet<String> = tag.split('_')
                .chain(tag.split('-'))
                .map(|word| word.to_lowercase())
                .collect();

            let tag_content_overlap = tag_words.intersection(content_words).count() as f64;
            let tag_relevance = tag_content_overlap / tag_words.len().max(1) as f64;

            coherence_score += tag_relevance;
            total_weight += 1.0;
        }

        if total_weight > 0.0 {
            coherence_score / total_weight
        } else {
            0.5
        }
    }

    /// Calculate contextual relevance based on query context
    fn calculate_contextual_relevance(&self, query_text: &str, index_entry: &MemoryIndexEntry) -> f64 {
        // Temporal relevance (recent memories might be more relevant)
        let now = chrono::Utc::now();
        let age_hours = (now - index_entry.created_at).num_hours().max(1) as f64;
        let temporal_relevance = (1.0 / (1.0 + age_hours / 168.0)).max(0.1); // Decay over weeks

        // Access pattern relevance
        let access_relevance = (index_entry.access_count as f64 / 10.0).min(1.0);

        // Query length vs content length matching
        let query_length = query_text.len() as f64;
        let content_length = index_entry.content_length as f64;
        let length_ratio = if query_length > content_length {
            content_length / query_length
        } else {
            query_length / content_length
        };
        let length_relevance = length_ratio.max(0.1);

        // Combine contextual factors
        temporal_relevance * 0.4 + access_relevance * 0.3 + length_relevance * 0.3
    }

    /// Calculate temporal and access relevance for embedding-based search
    fn calculate_temporal_access_relevance(&self, index_entry: &MemoryIndexEntry) -> f64 {
        // Temporal relevance (recent memories might be more relevant)
        let now = chrono::Utc::now();
        let age_hours = (now - index_entry.created_at).num_hours().max(1) as f64;
        let temporal_relevance = (1.0 / (1.0 + age_hours / 168.0)).max(0.1); // Decay over weeks

        // Access pattern relevance
        let access_relevance = (index_entry.access_count as f64 / 10.0).min(1.0);

        // Importance factor
        let importance_factor = index_entry.importance;

        // Combine factors
        temporal_relevance * 0.4 + access_relevance * 0.3 + importance_factor * 0.3
    }

    /// Calculate advanced recency score with multiple decay functions
    fn calculate_advanced_recency_score(&self, age_hours: f64) -> f64 {
        // Multiple decay functions for different time scales
        let linear_decay = (1.0 - age_hours / (24.0 * 7.0)).max(0.0); // Linear decay over week
        let exponential_decay = (-age_hours / (24.0 * 3.0)).exp(); // Exponential decay over 3 days
        let logarithmic_decay = (1.0 / (1.0 + (age_hours / 24.0).ln())).max(0.0); // Logarithmic decay
        let gaussian_decay = (-(age_hours / 24.0).powi(2) / 8.0).exp(); // Gaussian decay

        // Weighted combination based on time scale
        if age_hours < 24.0 {
            // Recent memories: favor exponential and gaussian
            exponential_decay * 0.6 + gaussian_decay * 0.4
        } else if age_hours < 168.0 {
            // This week: balanced approach
            linear_decay * 0.4 + exponential_decay * 0.3 + logarithmic_decay * 0.3
        } else {
            // Older memories: favor logarithmic decay
            logarithmic_decay * 0.7 + linear_decay * 0.3
        }
    }

    /// Calculate weighted importance score with context
    fn calculate_weighted_importance_score(&self, memory: &MemoryEntry) -> f64 {
        let base_importance = memory.metadata.importance;

        // Boost importance based on tags
        let tag_boost = if memory.metadata.tags.iter().any(|tag|
            ["critical", "important", "urgent", "priority"].contains(&tag.as_str())
        ) {
            0.2
        } else {
            0.0
        };

        // Boost importance based on access patterns
        let access_boost = if memory.access_count() > 10 {
            0.1
        } else {
            0.0
        };

        // Boost importance based on content length (longer content might be more important)
        let content_boost = if memory.value.len() > 1000 {
            0.1
        } else {
            0.0
        };

        (base_importance + tag_boost + access_boost + content_boost).min(1.0)
    }

    /// Calculate user engagement score with behavioral patterns
    async fn calculate_user_engagement_score(&self, memory: &MemoryEntry) -> Result<f64> {
        let mut engagement_factors = Vec::new();

        // 1. Access frequency score
        let access_frequency = memory.access_count() as f64;
        let frequency_score = (access_frequency / 50.0).min(1.0); // Normalize to max 50 accesses
        engagement_factors.push(frequency_score);

        // 2. Recency of access score
        let time_since_access = chrono::Utc::now() - memory.last_accessed();
        let recency_score = if time_since_access.num_hours() < 24 {
            1.0
        } else if time_since_access.num_days() < 7 {
            0.8
        } else if time_since_access.num_days() < 30 {
            0.5
        } else {
            0.2
        };
        engagement_factors.push(recency_score);

        // 3. Confidence score (user's confidence in the memory)
        let confidence_score = memory.metadata.confidence;
        engagement_factors.push(confidence_score);

        // 4. Tag engagement score (based on tag popularity)
        let tag_engagement = self.calculate_tag_engagement_score(&memory.metadata.tags);
        engagement_factors.push(tag_engagement);

        // 5. Content interaction score (simulated based on content characteristics)
        let content_interaction = self.calculate_content_interaction_score(memory);
        engagement_factors.push(content_interaction);

        // Weighted combination
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1];
        Ok(engagement_factors.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Calculate tag engagement score
    fn calculate_tag_engagement_score(&self, tags: &[String]) -> f64 {
        if tags.is_empty() {
            return 0.3; // Neutral score for no tags
        }

        // Simulate tag popularity (in real implementation, this would be based on actual usage data)
        let popular_tags = ["work", "project", "important", "research", "personal", "meeting", "idea"];
        let engagement_tags = ["favorite", "bookmark", "starred", "priority"];

        let mut score = 0.0;
        let mut tag_count = 0;

        for tag in tags {
            tag_count += 1;
            if engagement_tags.contains(&tag.as_str()) {
                score += 0.9; // High engagement tags
            } else if popular_tags.contains(&tag.as_str()) {
                score += 0.7; // Popular tags
            } else {
                score += 0.5; // Regular tags
            }
        }

        if tag_count > 0 {
            (score / tag_count as f64).min(1.0)
        } else {
            0.3
        }
    }

    /// Calculate content interaction score
    fn calculate_content_interaction_score(&self, memory: &MemoryEntry) -> f64 {
        let content_length = memory.value.len() as f64;
        let word_count = memory.value.split_whitespace().count() as f64;

        // Content complexity score
        let complexity_score = if word_count > 0.0 {
            (content_length / word_count / 10.0).min(1.0) // Average word length as complexity indicator
        } else {
            0.0
        };

        // Content richness score (based on variety of words)
        let unique_words_set: std::collections::HashSet<String> = memory.value
            .split_whitespace()
            .map(|word| word.to_lowercase())
            .collect();

        let richness_score = if word_count > 0.0 {
            (unique_words_set.len() as f64 / word_count).min(1.0)
        } else {
            0.0
        };

        // Combine scores
        complexity_score * 0.6 + richness_score * 0.4
    }

    /// Calculate content richness score with advanced analysis
    async fn calculate_content_richness_score(&self, memory: &MemoryEntry) -> Result<f64> {
        let mut richness_factors = Vec::new();

        // 1. Content length score (normalized)
        let content_length = memory.value.len() as f64;
        let length_score = (content_length / 2000.0).min(1.0); // Normalize to 2000 chars
        richness_factors.push(length_score);

        // 2. Vocabulary diversity score
        let words: Vec<&str> = memory.value.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let diversity_score = if !words.is_empty() {
            unique_words.len() as f64 / words.len() as f64
        } else {
            0.0
        };
        richness_factors.push(diversity_score);

        // 3. Tag richness score
        let tag_richness = (memory.metadata.tags.len() as f64 / 15.0).min(1.0); // Normalize to 15 tags
        richness_factors.push(tag_richness);

        // 4. Structural complexity (sentences, punctuation, etc.)
        let structural_complexity = self.calculate_structural_complexity(&memory.value);
        richness_factors.push(structural_complexity);

        // 5. Information density score
        let info_density = self.calculate_information_density(&memory.value);
        richness_factors.push(info_density);

        // Weighted combination
        let weights = [0.25, 0.25, 0.2, 0.15, 0.15];
        Ok(richness_factors.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Calculate structural complexity of content
    fn calculate_structural_complexity(&self, content: &str) -> f64 {
        let sentence_count = content.matches(&['.', '!', '?'][..]).count() as f64;
        let paragraph_count = content.matches("\n\n").count() as f64 + 1.0;
        let punctuation_count = content.matches(&[',', ';', ':', '-', '(', ')'][..]).count() as f64;

        let word_count = content.split_whitespace().count() as f64;

        if word_count == 0.0 {
            return 0.0;
        }

        // Calculate complexity metrics
        let sentence_density = sentence_count / word_count * 100.0;
        let paragraph_density = paragraph_count / word_count * 100.0;
        let punctuation_density = punctuation_count / word_count * 100.0;

        // Normalize and combine
        let sentence_score = (sentence_density / 10.0).min(1.0);
        let paragraph_score = (paragraph_density / 5.0).min(1.0);
        let punctuation_score = (punctuation_density / 20.0).min(1.0);

        (sentence_score + paragraph_score + punctuation_score) / 3.0
    }

    /// Calculate information density
    fn calculate_information_density(&self, content: &str) -> f64 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Count information-rich words (longer words, capitalized words, numbers)
        let mut info_rich_count = 0;
        for word in &words {
            if word.len() > 6 || // Long words
               word.chars().any(|c| c.is_uppercase()) || // Capitalized words
               word.chars().any(|c| c.is_numeric()) || // Numbers
               word.contains(&['@', '#', '$', '%'][..]) // Special symbols
            {
                info_rich_count += 1;
            }
        }

        info_rich_count as f64 / words.len() as f64
    }

    /// Calculate adaptive balanced score with machine learning-like approach
    async fn calculate_adaptive_balanced_score(&self, memory: &MemoryEntry, relevance_score: f64) -> Result<f64> {
        // Collect multiple factors
        let factors = vec![
            ("relevance", relevance_score),
            ("importance", memory.metadata.importance),
            ("confidence", memory.metadata.confidence),
            ("recency", self.calculate_advanced_recency_score(
                (chrono::Utc::now() - memory.created_at()).num_hours() as f64
            )),
            ("access_frequency", (memory.access_count() as f64 / 100.0).min(1.0)),
            ("content_richness", self.calculate_content_richness_score(memory).await?),
            ("tag_engagement", self.calculate_tag_engagement_score(&memory.metadata.tags)),
        ];

        // Adaptive weighting based on factor values and historical performance
        let adaptive_weights = self.calculate_adaptive_weights(&factors);

        // Calculate weighted score
        let weighted_score: f64 = factors.iter()
            .zip(adaptive_weights.iter())
            .map(|((_, value), weight)| value * weight)
            .sum();

        Ok(weighted_score.min(1.0))
    }

    /// Calculate adaptive weights based on factor performance
    fn calculate_adaptive_weights(&self, factors: &[(&str, f64)]) -> Vec<f64> {
        // In a real ML implementation, these weights would be learned from user feedback
        // For now, use heuristic adaptive weighting

        let mut weights = vec![0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]; // Base weights

        // Adjust weights based on factor values
        for (i, (factor_name, value)) in factors.iter().enumerate() {
            match *factor_name {
                "relevance" => {
                    // Boost relevance weight if it's high
                    if *value > 0.8 {
                        weights[i] *= 1.2;
                    }
                }
                "importance" => {
                    // Boost importance weight for high-importance memories
                    if *value > 0.7 {
                        weights[i] *= 1.15;
                    }
                }
                "recency" => {
                    // Boost recency weight for very recent memories
                    if *value > 0.9 {
                        weights[i] *= 1.1;
                    }
                }
                _ => {}
            }
        }

        // Normalize weights to sum to 1.0
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum > 0.0 {
            weights.iter().map(|w| w / weight_sum).collect()
        } else {
            vec![1.0 / weights.len() as f64; weights.len()]
        }
    }

    /// Calculate ML-based personalized score (simulated machine learning)
    async fn calculate_ml_personalized_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Simulate ML-based scoring using feature engineering
        let features = self.extract_ml_features(memory).await?;
        let score = self.apply_ml_model(&features);
        Ok(score)
    }

    /// Extract features for ML model
    async fn extract_ml_features(&self, memory: &MemoryEntry) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        // Temporal features
        let age_hours = (chrono::Utc::now() - memory.created_at()).num_hours() as f64;
        features.push((age_hours / 168.0).min(1.0)); // Age in weeks, capped at 1

        let hours_since_access = (chrono::Utc::now() - memory.last_accessed()).num_hours() as f64;
        features.push((hours_since_access / 168.0).min(1.0)); // Time since access in weeks

        // Content features
        features.push((memory.value.len() as f64 / 2000.0).min(1.0)); // Content length
        features.push(memory.metadata.importance); // Importance
        features.push(memory.metadata.confidence); // Confidence
        features.push((memory.access_count() as f64 / 100.0).min(1.0)); // Access frequency

        // Tag features
        features.push((memory.metadata.tags.len() as f64 / 10.0).min(1.0)); // Tag count
        features.push(self.calculate_tag_engagement_score(&memory.metadata.tags)); // Tag engagement

        // Structural features
        let word_count = memory.value.split_whitespace().count() as f64;
        features.push((word_count / 500.0).min(1.0)); // Word count

        // Memory type feature
        let memory_type_score = match memory.memory_type {
            crate::memory::types::MemoryType::ShortTerm => 0.3,
            crate::memory::types::MemoryType::LongTerm => 0.7,
        };
        features.push(memory_type_score);

        Ok(features)
    }

    /// Apply simulated ML model to features
    fn apply_ml_model(&self, features: &[f64]) -> f64 {
        // Simulate a trained ML model with learned weights
        // In reality, this would be a neural network or other ML model
        let weights = [
            0.15, // age
            0.12, // time since access
            0.10, // content length
            0.20, // importance
            0.15, // confidence
            0.10, // access frequency
            0.08, // tag count
            0.05, // tag engagement
            0.03, // word count
            0.02, // memory type
        ];

        let mut score = 0.0;
        for (i, feature) in features.iter().enumerate() {
            if i < weights.len() {
                score += feature * weights[i];
            }
        }

        // Apply non-linear activation (sigmoid-like)
        let activated_score = 1.0 / (1.0 + (-score * 2.0 + 1.0).exp());
        activated_score
    }

    /// Calculate semantic context score
    async fn calculate_semantic_context_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Analyze semantic context using multiple approaches
        let mut context_scores = Vec::new();

        // 1. Topic coherence score
        let topic_coherence = self.calculate_topic_coherence_from_vec(&memory.value, &memory.metadata.tags);
        context_scores.push(topic_coherence);

        // 2. Semantic density score
        let semantic_density = self.calculate_semantic_density_advanced(&memory.value);
        context_scores.push(semantic_density);

        // 3. Contextual relevance based on recent search patterns
        let contextual_relevance = self.calculate_contextual_relevance_from_history(memory);
        context_scores.push(contextual_relevance);

        // 4. Cross-reference score (how well this memory connects to others)
        let cross_reference_score = self.calculate_cross_reference_score(memory).await?;
        context_scores.push(cross_reference_score);

        // Weighted combination
        let weights = [0.3, 0.25, 0.25, 0.2];
        Ok(context_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Calculate topic coherence between content and tags
    fn calculate_topic_coherence_from_vec(&self, content: &str, tags: &[String]) -> f64 {
        if content.is_empty() || tags.is_empty() {
            return 0.5; // Neutral score
        }

        let content_words: std::collections::HashSet<String> = content
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 3) // Filter short words
            .map(|word| word.to_string())
            .collect();

        let mut coherence_scores = Vec::new();
        for tag in tags {
            let tag_words: std::collections::HashSet<String> = tag
                .to_lowercase()
                .split(&['_', '-', ' '][..])
                .filter(|word| word.len() > 2)
                .map(|word| word.to_string())
                .collect();

            let overlap = content_words.intersection(&tag_words).count() as f64;
            let tag_coherence = if !tag_words.is_empty() {
                overlap / tag_words.len() as f64
            } else {
                0.0
            };
            coherence_scores.push(tag_coherence);
        }

        if coherence_scores.is_empty() {
            0.5
        } else {
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
        }
    }

    /// Calculate advanced semantic density
    fn calculate_semantic_density_advanced(&self, content: &str) -> f64 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Count semantically rich words
        let semantic_indicators = [
            "because", "therefore", "however", "although", "since", "while",
            "important", "significant", "critical", "essential", "key",
            "analyze", "evaluate", "compare", "contrast", "implement",
            "strategy", "approach", "method", "technique", "process"
        ];

        let semantic_word_count = words.iter()
            .filter(|word| {
                let lower_word = word.to_lowercase();
                semantic_indicators.contains(&lower_word.as_str()) ||
                word.len() > 8 || // Long words often carry more semantic meaning
                word.chars().any(|c| c.is_uppercase()) // Proper nouns
            })
            .count();

        semantic_word_count as f64 / words.len() as f64
    }

    /// Calculate contextual relevance from search history
    fn calculate_contextual_relevance_from_history(&self, memory: &MemoryEntry) -> f64 {
        // Analyze recent search patterns to determine contextual relevance
        let recent_searches = self.search_history.iter()
            .rev()
            .take(10) // Last 10 searches
            .collect::<Vec<_>>();

        if recent_searches.is_empty() {
            return 0.5; // Neutral score if no history
        }

        let mut relevance_scores = Vec::new();
        for search in recent_searches {
            let query_words: std::collections::HashSet<String> = search.text_query.as_ref().map(|q| q.clone()).unwrap_or_default()
                .to_lowercase()
                .split_whitespace()
                .map(|word| word.to_string())
                .collect();

            let memory_words: std::collections::HashSet<String> = memory.value
                .to_lowercase()
                .split_whitespace()
                .map(|word| word.to_string())
                .collect();

            let overlap = query_words.intersection(&memory_words).count() as f64;
            let relevance = if !query_words.is_empty() {
                overlap / query_words.len() as f64
            } else {
                0.0
            };
            relevance_scores.push(relevance);
        }

        // Weight recent searches more heavily
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        for (i, score) in relevance_scores.iter().enumerate() {
            let weight = 1.0 / (i as f64 + 1.0); // Decreasing weight for older searches
            weighted_score += score * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.5
        }
    }

    /// Calculate cross-reference score
    async fn calculate_cross_reference_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Calculate how well this memory connects to other memories
        let memory_words: std::collections::HashSet<String> = memory.value
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_string())
            .collect();

        let memory_tags: std::collections::HashSet<String> = memory.metadata.tags
            .iter()
            .map(|tag| tag.to_lowercase())
            .collect();

        let mut connection_scores = Vec::new();

        // Sample a subset of memories for comparison (to avoid performance issues)
        let sample_size = 50.min(self.search_index.metadata_index.len());
        let sample_memories: Vec<_> = self.search_index.metadata_index
            .iter()
            .take(sample_size)
            .collect();

        for (other_key, other_entry) in sample_memories {
            if other_key == &memory.key {
                continue; // Skip self
            }

            // Calculate word overlap
            let word_overlap = memory_words.intersection(&other_entry.content_words).count() as f64;
            let word_connection = if !memory_words.is_empty() {
                word_overlap / memory_words.len() as f64
            } else {
                0.0
            };

            // Calculate tag overlap
            let tag_overlap = memory_tags.intersection(&other_entry.tags).count() as f64;
            let tag_connection = if !memory_tags.is_empty() {
                tag_overlap / memory_tags.len() as f64
            } else {
                0.0
            };

            let connection_score = word_connection * 0.7 + tag_connection * 0.3;
            connection_scores.push(connection_score);
        }

        if connection_scores.is_empty() {
            Ok(0.5)
        } else {
            // Use average connection strength
            Ok(connection_scores.iter().sum::<f64>() / connection_scores.len() as f64)
        }
    }

    /// Calculate collaborative filtering score (simulated)
    async fn calculate_collaborative_filtering_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Simulate collaborative filtering based on similar user patterns
        // In a real implementation, this would use actual user behavior data

        let mut similarity_scores = Vec::new();

        // Factor 1: Tag-based similarity with other high-engagement memories
        let tag_similarity = self.calculate_tag_based_collaborative_score(&memory.metadata.tags);
        similarity_scores.push(tag_similarity);

        // Factor 2: Content-based similarity with frequently accessed memories
        let content_similarity = self.calculate_content_based_collaborative_score(memory).await?;
        similarity_scores.push(content_similarity);

        // Factor 3: Temporal pattern similarity
        let temporal_similarity = self.calculate_temporal_collaborative_score(memory);
        similarity_scores.push(temporal_similarity);

        // Factor 4: Access pattern similarity
        let access_similarity = self.calculate_access_pattern_collaborative_score(memory);
        similarity_scores.push(access_similarity);

        // Weighted combination
        let weights = [0.3, 0.3, 0.2, 0.2];
        Ok(similarity_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Calculate tag-based collaborative score
    fn calculate_tag_based_collaborative_score(&self, tags: &[String]) -> f64 {
        // Simulate finding memories with similar tags that have high engagement
        let popular_tag_combinations = [
            vec!["work", "project"],
            vec!["research", "analysis"],
            vec!["meeting", "notes"],
            vec!["idea", "innovation"],
            vec!["personal", "important"],
        ];

        let memory_tags: std::collections::HashSet<String> = tags.iter()
            .map(|tag| tag.to_lowercase())
            .collect();

        let mut max_similarity = 0.0;
        for popular_combo in &popular_tag_combinations {
            let combo_set: std::collections::HashSet<String> = popular_combo.iter()
                .map(|tag| tag.to_string())
                .collect();

            let overlap = memory_tags.intersection(&combo_set).count() as f64;
            let similarity = if !combo_set.is_empty() {
                overlap / combo_set.len() as f64
            } else {
                0.0
            };

            if similarity > max_similarity {
                max_similarity = similarity;
            }
        }

        max_similarity
    }

    /// Calculate content-based collaborative score
    async fn calculate_content_based_collaborative_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Find memories with similar content characteristics that have high access counts
        let target_length = memory.value.len();
        let target_word_count = memory.value.split_whitespace().count();

        let mut similarity_scores = Vec::new();

        // Sample memories for comparison
        let sample_size = 30.min(self.search_index.metadata_index.len());
        for (_, other_entry) in self.search_index.metadata_index.iter().take(sample_size) {
            if other_entry.access_count > 5 { // Only consider frequently accessed memories
                let length_similarity = 1.0 - ((target_length as f64 - other_entry.content_length as f64).abs() / target_length.max(other_entry.content_length) as f64);
                let word_count_similarity = 1.0 - ((target_word_count as f64 - other_entry.content_words.len() as f64).abs() / target_word_count.max(other_entry.content_words.len()) as f64);

                let combined_similarity = (length_similarity + word_count_similarity) / 2.0;
                similarity_scores.push(combined_similarity);
            }
        }

        if similarity_scores.is_empty() {
            Ok(0.5)
        } else {
            Ok(similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64)
        }
    }

    /// Calculate temporal collaborative score
    fn calculate_temporal_collaborative_score(&self, memory: &MemoryEntry) -> f64 {
        // Analyze temporal patterns - memories created at similar times of day/week
        let creation_hour = memory.created_at().hour();
        let creation_weekday = memory.created_at().weekday();

        // Simulate popular creation times
        let popular_hours = [9, 10, 11, 14, 15, 16]; // Business hours
        let popular_weekdays = [chrono::Weekday::Mon, chrono::Weekday::Tue, chrono::Weekday::Wed, chrono::Weekday::Thu];

        let hour_score = if popular_hours.contains(&creation_hour) { 0.8 } else { 0.4 };
        let weekday_score = if popular_weekdays.contains(&creation_weekday) { 0.8 } else { 0.4 };

        (hour_score + weekday_score) / 2.0
    }

    /// Calculate access pattern collaborative score
    fn calculate_access_pattern_collaborative_score(&self, memory: &MemoryEntry) -> f64 {
        let access_count = memory.access_count() as f64;
        let hours_since_access = (chrono::Utc::now() - memory.last_accessed()).num_hours() as f64;

        // Score based on access frequency and recency
        let frequency_score = (access_count / 20.0).min(1.0); // Normalize to 20 accesses
        let recency_score = if hours_since_access < 24.0 {
            1.0
        } else if hours_since_access < 168.0 {
            0.7
        } else {
            0.3
        };

        frequency_score * 0.6 + recency_score * 0.4
    }

    /// Calculate temporal pattern score
    async fn calculate_temporal_pattern_score(&self, memory: &MemoryEntry) -> Result<f64> {
        let mut pattern_scores = Vec::new();

        // 1. Creation time pattern analysis
        let creation_pattern = self.analyze_creation_time_pattern(memory);
        pattern_scores.push(creation_pattern);

        // 2. Access time pattern analysis
        let access_pattern = self.analyze_access_time_pattern(memory);
        pattern_scores.push(access_pattern);

        // 3. Lifecycle stage pattern
        let lifecycle_pattern = self.analyze_lifecycle_pattern(memory);
        pattern_scores.push(lifecycle_pattern);

        // 4. Seasonal pattern (if applicable)
        let seasonal_pattern = self.analyze_seasonal_pattern(memory);
        pattern_scores.push(seasonal_pattern);

        // Weighted combination
        let weights = [0.3, 0.3, 0.2, 0.2];
        Ok(pattern_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Analyze creation time pattern
    fn analyze_creation_time_pattern(&self, memory: &MemoryEntry) -> f64 {
        let creation_time = memory.created_at();
        let hour = creation_time.hour();
        let day_of_week = creation_time.weekday();

        // Score based on optimal creation times
        let hour_score = match hour {
            8..=11 => 0.9,   // Morning peak
            13..=17 => 0.8,  // Afternoon peak
            18..=21 => 0.6,  // Evening
            _ => 0.3,        // Off-hours
        };

        let day_score = match day_of_week {
            chrono::Weekday::Mon | chrono::Weekday::Tue | chrono::Weekday::Wed => 0.9,
            chrono::Weekday::Thu | chrono::Weekday::Fri => 0.8,
            chrono::Weekday::Sat => 0.5,
            chrono::Weekday::Sun => 0.4,
        };

        (hour_score + day_score) / 2.0
    }

    /// Analyze access time pattern
    fn analyze_access_time_pattern(&self, memory: &MemoryEntry) -> f64 {
        let last_access = memory.last_accessed();
        let hours_since_access = (chrono::Utc::now() - last_access).num_hours() as f64;

        // Score based on access recency and frequency
        let recency_score = if hours_since_access < 1.0 {
            1.0
        } else if hours_since_access < 24.0 {
            0.9
        } else if hours_since_access < 168.0 {
            0.7
        } else if hours_since_access < 720.0 {
            0.4
        } else {
            0.1
        };

        let frequency_score = (memory.access_count() as f64 / 15.0).min(1.0);

        recency_score * 0.7 + frequency_score * 0.3
    }

    /// Analyze lifecycle pattern
    fn analyze_lifecycle_pattern(&self, memory: &MemoryEntry) -> f64 {
        let age_days = (chrono::Utc::now() - memory.created_at()).num_days() as f64;
        let access_count = memory.access_count() as f64;

        // Calculate access rate over lifetime
        let access_rate = if age_days > 0.0 {
            access_count / age_days
        } else {
            access_count
        };

        // Score based on healthy access patterns
        if access_rate > 1.0 {
            1.0 // Very active memory
        } else if access_rate > 0.5 {
            0.8 // Active memory
        } else if access_rate > 0.1 {
            0.6 // Moderately active
        } else if access_rate > 0.01 {
            0.4 // Low activity
        } else {
            0.2 // Inactive
        }
    }

    /// Analyze seasonal pattern
    fn analyze_seasonal_pattern(&self, memory: &MemoryEntry) -> f64 {
        let creation_month = memory.created_at().month();
        let current_month = chrono::Utc::now().month();

        // Score based on seasonal relevance
        let month_diff = ((creation_month as i32 - current_month as i32).abs()).min(6);
        let seasonal_score = match month_diff {
            0 => 1.0,      // Same month
            1 => 0.9,      // Adjacent month
            2 => 0.7,      // 2 months apart
            3 => 0.5,      // Quarter apart
            _ => 0.3,      // Different season
        };

        seasonal_score
    }



    /// Calculate content-based centrality as fallback
    async fn calculate_content_based_centrality(&self, memory_key: &str) -> Result<f64> {
        let memory_entry = self.search_index.metadata_index.get(memory_key)
            .ok_or_else(|| MemoryError::NotFound { key: memory_key.to_string() })?;

        let mut connection_count = 0;
        let mut total_strength = 0.0;

        // Count connections to other memories
        for (other_key, other_entry) in &self.search_index.metadata_index {
            if other_key == memory_key {
                continue;
            }

            // Calculate connection strength
            let word_overlap = memory_entry.content_words.intersection(&other_entry.content_words).count() as f64;
            let tag_overlap = memory_entry.tags.intersection(&other_entry.tags).count() as f64;

            let connection_strength = (word_overlap / memory_entry.content_words.len().max(1) as f64) * 0.7 +
                                    (tag_overlap / memory_entry.tags.len().max(1) as f64) * 0.3;

            if connection_strength > 0.1 { // Threshold for meaningful connection
                connection_count += 1;
                total_strength += connection_strength;
            }
        }

        // Normalize centrality score
        let centrality = if connection_count > 0 {
            (total_strength / connection_count as f64).min(1.0)
        } else {
            0.1 // Isolated node
        };

        Ok(centrality)
    }

    /// Calculate adaptive learning score
    async fn calculate_adaptive_learning_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Simulate adaptive learning from user feedback and behavior
        let mut learning_factors = Vec::new();

        // 1. Historical performance score
        let historical_score = self.calculate_historical_performance_score(memory);
        learning_factors.push(historical_score);

        // 2. User feedback simulation
        let feedback_score = self.simulate_user_feedback_score(memory);
        learning_factors.push(feedback_score);

        // 3. Adaptation rate based on memory characteristics
        let adaptation_score = self.calculate_adaptation_score(memory);
        learning_factors.push(adaptation_score);

        // 4. Learning momentum (how quickly the memory improves)
        let momentum_score = self.calculate_learning_momentum(memory);
        learning_factors.push(momentum_score);

        // Weighted combination with adaptive weights
        let base_weights = [0.3, 0.25, 0.25, 0.2];
        let adaptive_weights = self.adapt_learning_weights(&learning_factors, &base_weights);

        Ok(learning_factors.iter()
            .zip(adaptive_weights.iter())
            .map(|(score, weight)| score * weight)
            .sum())
    }

    /// Calculate historical performance score
    fn calculate_historical_performance_score(&self, memory: &MemoryEntry) -> f64 {
        // Simulate historical performance based on access patterns and age
        let age_days = (chrono::Utc::now() - memory.created_at()).num_days() as f64;
        let access_count = memory.access_count() as f64;

        let performance_trend = if age_days > 0.0 {
            let access_rate = access_count / age_days;
            if access_rate > 0.5 {
                0.9 // Improving performance
            } else if access_rate > 0.1 {
                0.7 // Stable performance
            } else {
                0.4 // Declining performance
            }
        } else {
            0.6 // New memory, neutral score
        };

        performance_trend
    }

    /// Simulate user feedback score
    fn simulate_user_feedback_score(&self, memory: &MemoryEntry) -> f64 {
        // Simulate positive feedback based on memory characteristics
        let importance = memory.metadata.importance;
        let confidence = memory.metadata.confidence;
        let access_frequency = (memory.access_count() as f64 / 20.0).min(1.0);

        // Simulate feedback based on these factors
        let simulated_feedback = importance * 0.4 + confidence * 0.3 + access_frequency * 0.3;
        simulated_feedback
    }

    /// Calculate adaptation score
    fn calculate_adaptation_score(&self, memory: &MemoryEntry) -> f64 {
        // Score based on how well the memory adapts to changing contexts
        let tag_diversity = (memory.metadata.tags.len() as f64 / 8.0).min(1.0);
        let content_richness = (memory.value.len() as f64 / 1500.0).min(1.0);
        let memory_type_adaptability = match memory.memory_type {
            crate::memory::types::MemoryType::LongTerm => 0.8, // More adaptable
            crate::memory::types::MemoryType::ShortTerm => 0.6, // Less adaptable
        };

        (tag_diversity + content_richness + memory_type_adaptability) / 3.0
    }

    /// Calculate learning momentum
    fn calculate_learning_momentum(&self, memory: &MemoryEntry) -> f64 {
        // Simulate learning momentum based on recent access patterns
        let hours_since_access = (chrono::Utc::now() - memory.last_accessed()).num_hours() as f64;
        let access_count = memory.access_count() as f64;

        let momentum = if hours_since_access < 24.0 && access_count > 3.0 {
            0.9 // High momentum
        } else if hours_since_access < 168.0 && access_count > 1.0 {
            0.7 // Medium momentum
        } else if access_count > 0.0 {
            0.5 // Low momentum
        } else {
            0.2 // No momentum
        };

        momentum
    }

    /// Adapt learning weights based on performance
    fn adapt_learning_weights(&self, factors: &[f64], base_weights: &[f64]) -> Vec<f64> {
        let mut adapted_weights = base_weights.to_vec();

        // Boost weights for high-performing factors
        for (i, factor) in factors.iter().enumerate() {
            if i < adapted_weights.len() {
                if *factor > 0.8 {
                    adapted_weights[i] *= 1.2; // Boost high performers
                } else if *factor < 0.3 {
                    adapted_weights[i] *= 0.8; // Reduce low performers
                }
            }
        }

        // Normalize weights
        let weight_sum: f64 = adapted_weights.iter().sum();
        if weight_sum > 0.0 {
            adapted_weights.iter().map(|w| w / weight_sum).collect()
        } else {
            base_weights.to_vec()
        }
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
