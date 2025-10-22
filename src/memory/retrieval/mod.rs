//! Memory retrieval module with optimized implementations

use crate::error::Result;
use crate::memory::types::{MemoryEntry, MemoryFragment, MemoryType};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod indexed;
pub mod pipeline;
pub mod strategies;

// Re-export pipeline types
pub use pipeline::{
    RetrievalPipeline, RetrievalSignal, ScoredMemory, PipelineConfig,
    FusionStrategy, HybridRetriever, CacheStats,
};

// Re-export concrete strategies
pub use strategies::{
    KeywordRetriever, TemporalRetriever, GraphRetriever,
};

/// Configuration for memory retrieval operations
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of results to return
    pub max_results: usize,
    /// Minimum relevance score threshold
    pub min_relevance_score: f64,
    /// Enable caching of retrieval results
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_results: 100,
            min_relevance_score: 0.1,
            enable_caching: true,
            cache_ttl_seconds: 300,
            enable_parallel_processing: true,
        }
    }
}

/// Search query for memory retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Text query for content search
    pub text: String,
    /// Memory type filter
    pub memory_type: Option<MemoryType>,
    /// Date range filter
    pub date_range: Option<DateRange>,
    /// Sort order
    pub sort_by: Option<SortBy>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Tags to filter by
    pub tags: Vec<String>,
    /// Minimum importance score
    pub min_importance: Option<f64>,
}

impl SearchQuery {
    pub fn new(text: String) -> Self {
        Self {
            text,
            memory_type: None,
            date_range: None,
            sort_by: None,
            limit: None,
            tags: Vec::new(),
            min_importance: None,
        }
    }

    pub fn with_memory_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    pub fn with_sort_by(mut self, sort_by: SortBy) -> Self {
        self.sort_by = Some(sort_by);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Date range for filtering memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Sort order for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortBy {
    /// Sort by relevance score (default)
    Relevance,
    /// Sort by creation date (newest first)
    CreatedAt,
    /// Sort by last modified date
    LastModified,
    /// Sort by importance score
    Importance,
    /// Sort by access frequency
    AccessFrequency,
}

/// Memory retriever trait for different retrieval strategies
#[async_trait]
pub trait MemoryRetriever: Send + Sync {
    /// Search for memories matching the query
    async fn search(&self, query: &SearchQuery) -> Result<Vec<MemoryFragment>>;

    /// Retrieve a specific memory by key
    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>>;

    /// Get similar memories to a given entry
    async fn find_similar(&self, entry: &MemoryEntry, limit: usize) -> Result<Vec<MemoryFragment>>;

    /// Update retrieval statistics for a memory
    async fn update_access_stats(&self, key: &str) -> Result<()>;
}

// Re-export the optimized implementations
pub use indexed::{
    IndexedMemoryRetriever, IndexingConfig, IndexStats,
    AccessTimeIndex, AccessFrequencyIndex, TagIndex,
    HotDataCache, QueryResultCache,
};
