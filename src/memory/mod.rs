//! Memory system modules for AI agents
//!
//! This module contains all the core components of the memory system:
//! - Types and data structures
//! - State management
//! - Storage backends
//! - Retrieval mechanisms
//! - Checkpointing system

pub mod types;
pub mod state;
pub mod storage;
pub mod retrieval;
pub mod checkpoint;
pub mod knowledge_graph;
pub mod temporal;
pub mod management;
pub mod consolidation;
pub mod meta_learning;
pub mod operations;
pub mod promotion;

#[cfg(feature = "embeddings")]
pub mod embeddings;

// Re-export commonly used types
pub use types::{MemoryEntry, MemoryFragment, MemoryType, MemoryMetadata};
pub use state::AgentState;
pub use storage::{Storage, create_storage};
pub use retrieval::MemoryRetriever;
pub use checkpoint::CheckpointManager;
pub use knowledge_graph::{
    MemoryKnowledgeGraph, Node, Edge, RelationshipType, NodeType,
    GraphQuery, GraphQueryBuilder, KnowledgeGraph, GraphConfig,
};
pub use meta_learning::{
    MetaLearningSystem, MetaLearningConfig, MetaTask, TaskType, MetaAlgorithm,
    AdaptationResult, MetaLearningMetrics, MAMLLearner, ReptileLearner, PrototypicalLearner,
};

use crate::error::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Core memory operations trait providing fundamental memory management capabilities.
///
/// This trait defines the essential operations for storing, retrieving, and managing
/// memory entries in the Synaptic AI agent memory system. All implementations must
/// provide these core functionalities with proper error handling and async support.
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::{MemoryOperations, MemoryEntry, MemoryType};
///
/// async fn example_usage<T: MemoryOperations>(memory: &mut T) -> Result<(), Box<dyn std::error::Error>> {
///     // Create a new memory entry
///     let entry = MemoryEntry::new(
///         "user_preference".to_string(),
///         "Dark mode enabled".to_string(),
///         MemoryType::ShortTerm,
///     );
///
///     // Store the memory
///     memory.store_memory(entry).await?;
///
///     // Retrieve it back
///     if let Some(retrieved) = memory.get_memory("user_preference").await? {
///         println!("Retrieved: {}", retrieved.value);
///     }
///
///     // Search for related memories
///     let results = memory.search_memories("dark mode", 10).await?;
///     println!("Found {} related memories", results.len());
///
///     Ok(())
/// }
/// ```
#[async_trait::async_trait]
pub trait MemoryOperations {
    /// Store a memory entry in the system.
    ///
    /// # Arguments
    ///
    /// * `entry` - The memory entry to store, containing key, value, metadata, and type
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the memory was successfully stored
    /// * `Err(MemoryError)` - If storage failed due to validation, duplicate key, or storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use synaptic::memory::{MemoryEntry, MemoryType};
    /// let entry = MemoryEntry::new(
    ///     "task_123".to_string(),
    ///     "Complete project documentation".to_string(),
    ///     MemoryType::LongTerm,
    /// );
    /// memory.store_memory(entry).await?;
    /// ```
    async fn store_memory(&mut self, entry: MemoryEntry) -> Result<()>;

    /// Retrieve a memory entry by its unique key.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique identifier for the memory entry
    ///
    /// # Returns
    ///
    /// * `Ok(Some(MemoryEntry))` - If the memory exists
    /// * `Ok(None)` - If no memory with the given key exists
    /// * `Err(MemoryError)` - If retrieval failed due to storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// if let Some(memory) = memory.get_memory("task_123").await? {
    ///     println!("Task: {}", memory.value);
    /// } else {
    ///     println!("Memory not found");
    /// }
    /// ```
    async fn get_memory(&self, key: &str) -> Result<Option<MemoryEntry>>;

    /// Search for memories matching the given query string.
    ///
    /// Performs content-based search across all stored memories, returning
    /// the most relevant results ranked by similarity score.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<MemoryFragment>)` - List of matching memory fragments with relevance scores
    /// * `Err(MemoryError)` - If search failed due to indexing or storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// let results = memory.search_memories("project documentation", 5).await?;
    /// for fragment in results {
    ///     println!("Score: {:.2}, Content: {}", fragment.relevance_score, fragment.entry.value);
    /// }
    /// ```
    async fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>>;

    /// Update the content of an existing memory entry.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique identifier of the memory to update
    /// * `value` - The new content value for the memory
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the memory was successfully updated
    /// * `Err(MemoryError)` - If the key doesn't exist or update failed
    ///
    /// # Examples
    ///
    /// ```rust
    /// memory.update_memory("task_123", "Complete project documentation - IN PROGRESS").await?;
    /// ```
    async fn update_memory(&mut self, key: &str, value: &str) -> Result<()>;

    /// Delete a memory entry from the system.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique identifier of the memory to delete
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the memory was found and deleted
    /// * `Ok(false)` - If no memory with the given key existed
    /// * `Err(MemoryError)` - If deletion failed due to storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// if memory.delete_memory("task_123").await? {
    ///     println!("Memory deleted successfully");
    /// } else {
    ///     println!("Memory not found");
    /// }
    /// ```
    async fn delete_memory(&mut self, key: &str) -> Result<bool>;

    /// List all memory keys currently stored in the system.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - List of all memory keys
    /// * `Err(MemoryError)` - If listing failed due to storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// let keys = memory.list_keys().await?;
    /// println!("Total memories: {}", keys.len());
    /// for key in keys {
    ///     println!("Key: {}", key);
    /// }
    /// ```
    async fn list_keys(&self) -> Result<Vec<String>>;

    /// Get comprehensive statistics about the memory system.
    ///
    /// # Returns
    ///
    /// A `MemoryStats` struct containing information about:
    /// - Total number of entries
    /// - Memory type distribution
    /// - Storage size metrics
    /// - Age and access patterns
    ///
    /// # Examples
    ///
    /// ```rust
    /// let stats = memory.get_stats();
    /// println!("Total entries: {}", stats.total_entries);
    /// println!("Average size: {:.2} bytes", stats.average_entry_size);
    /// ```
    fn get_stats(&self) -> CoreMemoryStats;

    /// Clear all memories from the system.
    ///
    /// **Warning**: This operation is irreversible and will permanently delete
    /// all stored memories. Use with caution.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all memories were successfully cleared
    /// * `Err(MemoryError)` - If clearing failed due to storage issues
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Clear all memories (use with caution!)
    /// memory.clear_all().await?;
    /// println!("All memories cleared");
    /// ```
    async fn clear_all(&mut self) -> Result<()>;
}

/// Comprehensive memory system statistics.
///
/// This structure provides detailed metrics about the current state of the memory system,
/// including entry counts, size metrics, temporal information, and session tracking.
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::MemoryStats;
///
/// let stats = memory.get_stats();
///
/// // Check memory usage
/// if stats.total_size_bytes > 1_000_000 {
///     println!("Memory usage is high: {} bytes", stats.total_size_bytes);
/// }
///
/// // Analyze memory distribution
/// let short_term_ratio = stats.short_term_entries as f64 / stats.total_entries as f64;
/// println!("Short-term memory ratio: {:.2}%", short_term_ratio * 100.0);
///
/// // Check system age
/// if let Some(oldest) = stats.oldest_entry {
///     let age_days = (chrono::Utc::now() - oldest).num_days();
///     println!("System has been running for {} days", age_days);
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMemoryStats {
    /// Total number of memory entries in the system
    pub total_entries: usize,
    /// Number of short-term memory entries
    pub short_term_entries: usize,
    /// Number of long-term memory entries
    pub long_term_entries: usize,
    /// Total storage size in bytes across all entries
    pub total_size_bytes: usize,
    /// Average size per memory entry in bytes
    pub average_entry_size: f64,
    /// Timestamp of the oldest memory entry, if any exists
    pub oldest_entry: Option<DateTime<Utc>>,
    /// Timestamp of the newest memory entry, if any exists
    pub newest_entry: Option<DateTime<Utc>>,
    /// Unique identifier for the current memory session
    pub session_id: Uuid,
}

impl CoreMemoryStats {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            total_entries: 0,
            short_term_entries: 0,
            long_term_entries: 0,
            total_size_bytes: 0,
            average_entry_size: 0.0,
            oldest_entry: None,
            newest_entry: None,
            session_id,
        }
    }

    pub fn update_with_entry(&mut self, entry: &MemoryEntry) {
        self.total_entries += 1;
        
        match entry.memory_type {
            MemoryType::ShortTerm => self.short_term_entries += 1,
            MemoryType::LongTerm => self.long_term_entries += 1,
        }
        
        let entry_size = entry.estimated_size();
        self.total_size_bytes += entry_size;
        self.average_entry_size = self.total_size_bytes as f64 / self.total_entries as f64;
        
        let entry_time = entry.created_at();
        if self.oldest_entry.is_none() || Some(entry_time) < self.oldest_entry {
            self.oldest_entry = Some(entry_time);
        }
        if self.newest_entry.is_none() || Some(entry_time) > self.newest_entry {
            self.newest_entry = Some(entry_time);
        }
    }

    pub fn remove_entry(&mut self, entry: &MemoryEntry) {
        if self.total_entries > 0 {
            self.total_entries -= 1;
            
            match entry.memory_type {
                MemoryType::ShortTerm => {
                    if self.short_term_entries > 0 {
                        self.short_term_entries -= 1;
                    }
                }
                MemoryType::LongTerm => {
                    if self.long_term_entries > 0 {
                        self.long_term_entries -= 1;
                    }
                }
            }
            
            let entry_size = entry.estimated_size();
            if self.total_size_bytes >= entry_size {
                self.total_size_bytes -= entry_size;
            }
            
            if self.total_entries > 0 {
                self.average_entry_size = self.total_size_bytes as f64 / self.total_entries as f64;
            } else {
                self.average_entry_size = 0.0;
                self.oldest_entry = None;
                self.newest_entry = None;
            }
        }
    }
}

/// Basic memory configuration options for core memory operations
#[derive(Debug, Clone)]
pub struct BasicMemoryConfig {
    pub max_short_term_entries: usize,
    pub max_long_term_entries: usize,
    pub checkpoint_interval: usize,
    pub similarity_threshold: f64,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub storage_path: Option<String>,
}

impl Default for BasicMemoryConfig {
    fn default() -> Self {
        Self {
            max_short_term_entries: 1000,
            max_long_term_entries: 10000,
            checkpoint_interval: 100,
            similarity_threshold: 0.7,
            enable_compression: false,
            enable_encryption: false,
            storage_path: None,
        }
    }
}

/// Memory cleanup policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupPolicy {
    /// Remove oldest entries when limit is reached
    LeastRecentlyUsed,
    /// Remove entries based on access frequency
    LeastFrequentlyUsed,
    /// Remove entries older than specified duration
    TimeBasedExpiry { max_age_hours: u64 },
    /// Custom cleanup logic
    Custom,
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self::LeastRecentlyUsed
    }
}
