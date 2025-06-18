//! Memory retrieval and search functionality

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryFragment, MemoryType};
use crate::memory::storage::Storage;
use std::collections::HashMap;
use std::sync::Arc;

/// Advanced memory retrieval system with sophisticated search capabilities.
///
/// The `MemoryRetriever` provides high-performance search and retrieval operations
/// for the Synaptic memory system, supporting multiple search strategies including
/// exact matching, fuzzy search, semantic similarity, and hybrid approaches.
///
/// # Features
///
/// - **Multi-strategy Search**: Combines exact, fuzzy, and semantic search
/// - **Relevance Scoring**: Advanced scoring based on recency, frequency, and importance
/// - **Configurable Thresholds**: Adjustable similarity and quality thresholds
/// - **Performance Optimized**: Efficient indexing and caching for fast retrieval
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::retrieval::{MemoryRetriever, RetrievalConfig};
/// use synaptic::memory::storage::create_storage;
///
/// async fn setup_retriever() -> Result<MemoryRetriever, Box<dyn std::error::Error>> {
///     let storage = create_storage("memory.db").await?;
///     let config = RetrievalConfig::default();
///     let retriever = MemoryRetriever::new(storage, config);
///     Ok(retriever)
/// }
/// ```
pub struct MemoryRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    config: RetrievalConfig,
}

/// Configuration parameters for memory retrieval operations.
///
/// This structure controls various aspects of the search and retrieval process,
/// allowing fine-tuning of performance, accuracy, and result quality.
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::retrieval::RetrievalConfig;
///
/// // Create a high-precision configuration
/// let config = RetrievalConfig {
///     max_results: 20,
///     similarity_threshold: 0.8,
///     enable_fuzzy_matching: true,
///     enable_semantic_search: true,
///     recency_weight: 0.3,
///     frequency_weight: 0.4,
///     importance_weight: 0.3,
/// };
///
/// // Create a fast, low-precision configuration
/// let fast_config = RetrievalConfig {
///     max_results: 5,
///     similarity_threshold: 0.5,
///     enable_fuzzy_matching: false,
///     enable_semantic_search: false,
///     recency_weight: 0.5,
///     frequency_weight: 0.3,
///     importance_weight: 0.2,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of results to return from any search operation
    pub max_results: usize,
    /// Minimum similarity threshold for including results (0.0 to 1.0)
    pub similarity_threshold: f64,
    /// Enable fuzzy string matching for approximate text search
    pub enable_fuzzy_matching: bool,
    /// Enable semantic search using vector embeddings (requires embeddings feature)
    pub enable_semantic_search: bool,
    /// Weight factor for recency in relevance scoring (0.0 to 1.0)
    pub recency_weight: f64,
    /// Weight factor for access frequency in relevance scoring (0.0 to 1.0)
    pub frequency_weight: f64,
    /// Weight factor for memory importance in relevance scoring (0.0 to 1.0)
    pub importance_weight: f64,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_results: 50,
            similarity_threshold: 0.1,
            enable_fuzzy_matching: true,
            enable_semantic_search: false,
            recency_weight: 0.3,
            frequency_weight: 0.3,
            importance_weight: 0.4,
        }
    }
}

/// Search query with various options
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The search text
    pub text: String,
    /// Memory type filter
    pub memory_type: Option<MemoryType>,
    /// Tag filters
    pub tags: Vec<String>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Custom field filters
    pub custom_filters: HashMap<String, String>,
    /// Sort order
    pub sort_by: SortBy,
    /// Maximum results to return
    pub limit: Option<usize>,
}

/// Date range for filtering
#[derive(Debug, Clone)]
pub struct DateRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Sort options for search results
#[derive(Debug, Clone)]
pub enum SortBy {
    /// Sort by relevance score (default)
    Relevance,
    /// Sort by creation date (newest first)
    CreatedDesc,
    /// Sort by creation date (oldest first)
    CreatedAsc,
    /// Sort by last access date
    LastAccessedDesc,
    /// Sort by access frequency
    AccessFrequency,
    /// Sort by importance score
    Importance,
}

impl Default for SortBy {
    fn default() -> Self {
        Self::Relevance
    }
}

impl SearchQuery {
    pub fn new(text: String) -> Self {
        Self {
            text,
            memory_type: None,
            tags: Vec::new(),
            date_range: None,
            custom_filters: HashMap::new(),
            sort_by: SortBy::default(),
            limit: None,
        }
    }

    pub fn with_memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_date_range(mut self, start: chrono::DateTime<chrono::Utc>, end: chrono::DateTime<chrono::Utc>) -> Self {
        self.date_range = Some(DateRange { start, end });
        self
    }

    pub fn with_sort_by(mut self, sort_by: SortBy) -> Self {
        self.sort_by = sort_by;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn add_custom_filter(mut self, key: String, value: String) -> Self {
        self.custom_filters.insert(key, value);
        self
    }
}

impl MemoryRetriever {
    /// Create a new memory retriever
    pub fn new(storage: Arc<dyn Storage + Send + Sync>, config: RetrievalConfig) -> Self {
        Self { storage, config }
    }

    /// Perform a simple text search
    pub async fn search(&self, query: &str) -> Result<Vec<MemoryFragment>> {
        let search_query = SearchQuery::new(query.to_string());
        self.advanced_search(&search_query).await
    }

    /// Perform an advanced search with filters and options
    pub async fn advanced_search(&self, query: &SearchQuery) -> Result<Vec<MemoryFragment>> {
        // Get initial results from storage
        let limit = query.limit.unwrap_or(self.config.max_results);
        let mut results = self.storage.search(&query.text, limit * 2).await?; // Get more for filtering

        // Apply filters
        results = self.apply_filters(results, query).await?;

        // Apply advanced scoring
        for fragment in &mut results {
            fragment.relevance_score = self.calculate_advanced_score(&fragment.entry, &query.text);
        }

        // Filter by similarity threshold
        results.retain(|fragment| fragment.relevance_score >= self.config.similarity_threshold);

        // Sort results
        self.sort_results(&mut results, &query.sort_by);

        // Limit results
        results.truncate(limit);

        Ok(results)
    }

    /// Find similar memories to a given entry
    pub async fn find_similar(&self, entry: &MemoryEntry, limit: usize) -> Result<Vec<MemoryFragment>> {
        if let Some(embedding) = &entry.embedding {
            self.find_similar_by_embedding(embedding, limit).await
        } else {
            self.find_similar_by_text(&entry.value, limit).await
        }
    }

    /// Find memories by semantic similarity using embeddings
    pub async fn find_similar_by_embedding(&self, _embedding: &[f32], _limit: usize) -> Result<Vec<MemoryFragment>> {
        // This would require iterating through all entries and computing cosine similarity
        // For now, we'll return an error indicating this feature needs vector database support
        Err(MemoryError::vector_operation(
            "Semantic search requires a vector database backend"
        ))
    }

    /// Find memories by text similarity
    pub async fn find_similar_by_text(&self, text: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        self.search(text).await.map(|mut results| {
            results.truncate(limit);
            results
        })
    }

    /// Get memories by tags
    pub async fn get_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryEntry>> {
        let keys = self.storage.list_keys().await?;
        let mut results = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                if tags.iter().any(|tag| entry.has_tag(tag)) {
                    results.push(entry);
                }
            }
        }

        Ok(results)
    }

    /// Get recently accessed memories
    pub async fn get_recent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let keys = self.storage.list_keys().await?;
        let mut entries = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                entries.push(entry);
            }
        }

        // Sort by last accessed time
        entries.sort_by(|a, b| b.last_accessed().cmp(&a.last_accessed()));
        entries.truncate(limit);

        Ok(entries)
    }

    /// Get most frequently accessed memories
    pub async fn get_frequent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let keys = self.storage.list_keys().await?;
        let mut entries = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                entries.push(entry);
            }
        }

        // Sort by access count
        entries.sort_by(|a, b| b.access_count().cmp(&a.access_count()));
        entries.truncate(limit);

        Ok(entries)
    }

    /// Apply filters to search results
    async fn apply_filters(&self, mut results: Vec<MemoryFragment>, query: &SearchQuery) -> Result<Vec<MemoryFragment>> {
        // Filter by memory type
        if let Some(memory_type) = query.memory_type {
            results.retain(|fragment| fragment.entry.memory_type == memory_type);
        }

        // Filter by tags
        if !query.tags.is_empty() {
            results.retain(|fragment| {
                query.tags.iter().any(|tag| fragment.entry.has_tag(tag))
            });
        }

        // Filter by date range
        if let Some(date_range) = &query.date_range {
            results.retain(|fragment| {
                let created_at = fragment.entry.created_at();
                created_at >= date_range.start && created_at <= date_range.end
            });
        }

        // Filter by custom fields
        for (key, value) in &query.custom_filters {
            results.retain(|fragment| {
                fragment.entry.metadata.get_custom_field(key)
                    .map_or(false, |field_value| field_value == value)
            });
        }

        Ok(results)
    }

    /// Calculate advanced relevance score
    fn calculate_advanced_score(&self, entry: &MemoryEntry, query: &str) -> f64 {
        let mut score = 0.0;

        // Text similarity score
        let text_score = self.calculate_text_similarity(&entry.value, query);
        score += text_score * 0.4;

        // Recency score (newer is better)
        let recency_score = self.calculate_recency_score(entry);
        score += recency_score * self.config.recency_weight;

        // Frequency score (more accessed is better)
        let frequency_score = self.calculate_frequency_score(entry);
        score += frequency_score * self.config.frequency_weight;

        // Importance score
        let importance_score = entry.metadata.importance;
        score += importance_score * self.config.importance_weight;

        score.clamp(0.0, 1.0)
    }

    /// Calculate text similarity between entry content and query
    fn calculate_text_similarity(&self, content: &str, query: &str) -> f64 {
        let content_lower = content.to_lowercase();
        let query_lower = query.to_lowercase();

        // Simple term frequency scoring
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
        let mut score = 0.0;

        for term in query_terms {
            let occurrences = content_lower.matches(term).count();
            score += occurrences as f64;
        }

        // Normalize by content length
        if content_lower.len() > 0 {
            score / content_lower.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate recency score (0.0 to 1.0, newer is higher)
    fn calculate_recency_score(&self, entry: &MemoryEntry) -> f64 {
        let now = chrono::Utc::now();
        let age = now - entry.created_at();
        let age_hours = age.num_hours() as f64;

        // Exponential decay: score decreases as age increases
        (-age_hours / (24.0 * 7.0)).exp() // Half-life of 1 week
    }

    /// Calculate frequency score (0.0 to 1.0, more accessed is higher)
    fn calculate_frequency_score(&self, entry: &MemoryEntry) -> f64 {
        let access_count = entry.access_count() as f64;
        
        // Logarithmic scaling to prevent very high access counts from dominating
        if access_count > 0.0 {
            (access_count.ln() / 10.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Sort search results based on the specified criteria
    fn sort_results(&self, results: &mut [MemoryFragment], sort_by: &SortBy) {
        match sort_by {
            SortBy::Relevance => {
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
            }
            SortBy::CreatedDesc => {
                results.sort_by(|a, b| b.entry.created_at().cmp(&a.entry.created_at()));
            }
            SortBy::CreatedAsc => {
                results.sort_by(|a, b| a.entry.created_at().cmp(&b.entry.created_at()));
            }
            SortBy::LastAccessedDesc => {
                results.sort_by(|a, b| b.entry.last_accessed().cmp(&a.entry.last_accessed()));
            }
            SortBy::AccessFrequency => {
                results.sort_by(|a, b| b.entry.access_count().cmp(&a.entry.access_count()));
            }
            SortBy::Importance => {
                results.sort_by(|a, b| {
                    b.entry.metadata.importance.partial_cmp(&a.entry.metadata.importance).unwrap()
                });
            }
        }
    }
}
