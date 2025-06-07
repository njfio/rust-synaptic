//! Advanced search engine for memory retrieval

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

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
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let start_time = std::time::Instant::now();
        
        // Note: Search history update removed for immutable method

        // Execute the search
        let mut results = self.execute_search(&query).await?;

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
    async fn execute_search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Text-based search
        if let Some(text_query) = &query.text_query {
            let text_results = self.search_by_text(text_query).await?;
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
    async fn search_by_text(&self, text_query: &str) -> Result<Vec<SearchResult>> {
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

        // Convert to search results
        let mut results = Vec::new();
        for (memory_key, score) in candidate_memories {
            if let Some(index_entry) = self.search_index.metadata_index.get(&memory_key) {
                // Create a placeholder memory entry
                let memory = MemoryEntry::new(
                    memory_key.clone(),
                    format!("Content for {}", memory_key), // Placeholder
                    MemoryType::ShortTerm, // Placeholder
                );

                let relevance_score = score / query_words.len() as f64;
                let highlights = self.generate_highlights(&memory.value, text_query).await?;

                results.push(SearchResult {
                    memory,
                    relevance_score,
                    ranking_score: relevance_score,
                    highlights,
                    explanation: format!("Matched {} query terms", score as usize),
                    related_memories: Vec::new(),
                    search_metadata: SearchResultMetadata {
                        generated_at: Utc::now(),
                        algorithm: "text_search".to_string(),
                        processing_time_ms: 0,
                        debug_info: HashMap::new(),
                    },
                });
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
            SearchFilter::SimilarTo { memory_key: _, threshold: _ } => {
                // TODO: Implement similarity filtering
                Ok(results)
            }
            SearchFilter::RelatedTo { memory_key: _, max_distance: _ } => {
                // TODO: Implement graph-based filtering
                Ok(results)
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
            RankingStrategy::Custom(_name) => {
                // TODO: Implement custom ranking strategies
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
                    RankingFactorType::GraphCentrality => 0.5, // TODO: Implement graph centrality
                    RankingFactorType::UserPreference => 0.5, // TODO: Implement user preferences
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

    /// Calculate string similarity (simple Levenshtein-based)
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }
        
        let len1 = s1.len();
        let len2 = s2.len();
        
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }
        
        // Simple character overlap similarity
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();
        
        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
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

    /// Add related memories to search results
    async fn add_related_memories(&self, results: Vec<SearchResult>) -> Result<Vec<SearchResult>> {
        // TODO: Implement related memory lookup using knowledge graph
        Ok(results)
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
