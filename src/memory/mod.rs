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

use crate::error::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Core memory operations trait
#[async_trait::async_trait]
pub trait MemoryOperations {
    /// Store a memory entry
    async fn store_memory(&mut self, entry: MemoryEntry) -> Result<()>;
    
    /// Retrieve a memory by key
    async fn get_memory(&self, key: &str) -> Result<Option<MemoryEntry>>;
    
    /// Search memories by content
    async fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>>;
    
    /// Update an existing memory
    async fn update_memory(&mut self, key: &str, value: &str) -> Result<()>;
    
    /// Delete a memory
    async fn delete_memory(&mut self, key: &str) -> Result<bool>;
    
    /// List all memory keys
    async fn list_keys(&self) -> Result<Vec<String>>;
    
    /// Get memory statistics
    fn get_stats(&self) -> MemoryStats;
    
    /// Clear all memories
    async fn clear_all(&mut self) -> Result<()>;
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub short_term_entries: usize,
    pub long_term_entries: usize,
    pub total_size_bytes: usize,
    pub average_entry_size: f64,
    pub oldest_entry: Option<DateTime<Utc>>,
    pub newest_entry: Option<DateTime<Utc>>,
    pub session_id: Uuid,
}

impl MemoryStats {
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

/// Memory configuration options
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub max_short_term_entries: usize,
    pub max_long_term_entries: usize,
    pub checkpoint_interval: usize,
    pub similarity_threshold: f64,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub storage_path: Option<String>,
}

impl Default for MemoryConfig {
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
