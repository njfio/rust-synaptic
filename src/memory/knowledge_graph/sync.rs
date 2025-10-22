

//! Graph synchronization trait and implementation
//!
//! This module provides automatic synchronization between memory storage
//! and the knowledge graph, ensuring consistency across the system.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use async_trait::async_trait;
use uuid::Uuid;

/// Trait for automatic memory-graph synchronization
///
/// Implementors of this trait provide automatic bidirectional synchronization
/// between memory storage operations and knowledge graph state, ensuring that
/// graph nodes and relationships stay consistent with memory lifecycle events.
#[async_trait]
pub trait MemoryGraphSync: Send + Sync {
    /// Synchronize a memory entry after creation
    ///
    /// Creates a new node in the knowledge graph and auto-detects relationships
    /// with existing memories based on content, metadata, and temporal proximity.
    ///
    /// # Arguments
    /// * `memory` - The memory entry that was created
    ///
    /// # Returns
    /// * `Ok(node_id)` - ID of the created node
    /// * `Err` - If synchronization failed
    async fn sync_created(&mut self, memory: &MemoryEntry) -> Result<Uuid>;

    /// Synchronize a memory entry after update
    ///
    /// Updates the corresponding node in the knowledge graph and refreshes
    /// relationships that may have changed due to content modifications.
    ///
    /// # Arguments
    /// * `old_memory` - The memory entry before update
    /// * `new_memory` - The memory entry after update
    ///
    /// # Returns
    /// * `Ok(node_id)` - ID of the updated node
    /// * `Err` - If synchronization failed
    async fn sync_updated(&mut self, old_memory: &MemoryEntry, new_memory: &MemoryEntry) -> Result<Uuid>;

    /// Synchronize a memory entry after deletion
    ///
    /// Removes the corresponding node from the knowledge graph and cleans up
    /// all associated relationships, ensuring no dangling edges remain.
    ///
    /// # Arguments
    /// * `memory` - The memory entry that was deleted
    ///
    /// # Returns
    /// * `Ok(())` - If deletion succeeded
    /// * `Err` - If synchronization failed
    async fn sync_deleted(&mut self, memory: &MemoryEntry) -> Result<()>;

    /// Synchronize a memory entry after access
    ///
    /// Updates access tracking in the graph, potentially strengthening relationship
    /// weights for frequently co-accessed memories.
    ///
    /// # Arguments
    /// * `memory` - The memory entry that was accessed
    ///
    /// # Returns
    /// * `Ok(())` - If access tracking succeeded
    /// * `Err` - If synchronization failed
    async fn sync_accessed(&mut self, memory: &MemoryEntry) -> Result<()>;

    /// Synchronize relationships after temporal event
    ///
    /// Updates temporal relationships when significant time-based events occur,
    /// such as memory consolidation or time-based clustering.
    ///
    /// # Arguments
    /// * `memory_key` - Key of the memory involved in temporal event
    /// * `event_type` - Type of temporal event (Created, Updated, Accessed)
    ///
    /// # Returns
    /// * `Ok(())` - If temporal sync succeeded
    /// * `Err` - If synchronization failed
    async fn sync_temporal_event(&mut self, memory_key: &str, event_type: TemporalEventType) -> Result<()>;

    /// Check if a memory has a corresponding graph node
    ///
    /// # Arguments
    /// * `memory_key` - Key of the memory to check
    ///
    /// # Returns
    /// * `Ok(Some(node_id))` - If node exists
    /// * `Ok(None)` - If no node exists
    /// * `Err` - If check failed
    async fn has_node(&self, memory_key: &str) -> Result<Option<Uuid>>;

    /// Batch synchronize multiple memories
    ///
    /// Efficiently synchronizes multiple memories in a single operation,
    /// useful for bulk imports or checkpoint restoration.
    ///
    /// # Arguments
    /// * `memories` - Slice of memory entries to synchronize
    ///
    /// # Returns
    /// * `Ok(node_ids)` - IDs of created/updated nodes
    /// * `Err` - If batch sync failed
    async fn sync_batch(&mut self, memories: &[MemoryEntry]) -> Result<Vec<Uuid>>;
}

/// Types of temporal events that can trigger synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalEventType {
    /// Memory was created
    Created,
    /// Memory was updated
    Updated,
    /// Memory was accessed
    Accessed,
    /// Memory was consolidated
    Consolidated,
    /// Memory was archived
    Archived,
}

/// Configuration for graph synchronization behavior
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphSyncConfig {
    /// Enable automatic synchronization (default: true)
    pub enabled: bool,

    /// Auto-detect relationships on creation (default: true)
    pub auto_detect_relationships: bool,

    /// Update relationships on memory modification (default: true)
    pub update_relationships_on_change: bool,

    /// Clean up orphaned nodes on deletion (default: true)
    pub cleanup_orphaned_nodes: bool,

    /// Track access patterns in graph weights (default: true)
    pub track_access_patterns: bool,

    /// Maximum depth for relationship detection (default: 2)
    pub max_relationship_depth: usize,

    /// Minimum similarity threshold for auto-relationships (default: 0.7)
    pub similarity_threshold: f64,

    /// Batch size for bulk operations (default: 100)
    pub batch_size: usize,
}

impl Default for GraphSyncConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_detect_relationships: true,
            update_relationships_on_change: true,
            cleanup_orphaned_nodes: true,
            track_access_patterns: true,
            max_relationship_depth: 2,
            similarity_threshold: 0.7,
            batch_size: 100,
        }
    }
}

impl GraphSyncConfig {
    /// Create a new configuration with all features enabled
    pub fn enabled() -> Self {
        Self::default()
    }

    /// Create a configuration with synchronization disabled
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Create a lightweight configuration with minimal features
    pub fn lightweight() -> Self {
        Self {
            enabled: true,
            auto_detect_relationships: false,
            update_relationships_on_change: false,
            cleanup_orphaned_nodes: true,
            track_access_patterns: false,
            max_relationship_depth: 1,
            similarity_threshold: 0.8,
            batch_size: 50,
        }
    }

    /// Builder method to enable/disable sync
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder method to configure relationship auto-detection
    pub fn with_auto_detect(mut self, auto_detect: bool) -> Self {
        self.auto_detect_relationships = auto_detect;
        self
    }

    /// Builder method to configure relationship depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_relationship_depth = depth;
        self
    }

    /// Builder method to configure similarity threshold
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}
