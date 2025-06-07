//! # Synaptic 🧠
//!
//! An intelligent AI agent memory system built in Rust that creates and manages
//! dynamic knowledge graphs with smart content updates. Unlike traditional memory
//! systems that create duplicate entries, Synaptic intelligently merges similar
//! content and evolves relationships over time.
//!
//! ## Key Features
//!
//! - **🧠 Intelligent Memory Updates**: Smart node merging and content evolution tracking
//! - **🕸️ Advanced Knowledge Graph**: Dynamic relationship detection and reasoning engine
//! - **⏰ Temporal Intelligence**: Version history and pattern detection
//! - **🔍 Advanced Search & Retrieval**: Multi-criteria search with relevance ranking
//! - **🎯 Memory Management**: Intelligent summarization and lifecycle policies
//!
//! ## Quick Start
//!
//! ```rust
//! use synaptic::{AgentMemory, MemoryConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = MemoryConfig {
//!         enable_knowledge_graph: true,
//!         enable_temporal_tracking: true,
//!         enable_advanced_management: true,
//!         ..Default::default()
//!     };
//!     let mut memory = AgentMemory::new(config).await?;
//!
//!     // Store memories - similar content will be intelligently merged
//!     memory.store("project_alpha", "A web application using React").await?;
//!     memory.store("project_alpha", "A web application using React and Node.js").await?;
//!
//!     // Find related memories
//!     let related = memory.find_related_memories("project_alpha", 5).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod memory;

// Re-export main types for convenience
pub use error::{MemoryError, Result};
pub use memory::{
    MemoryEntry, MemoryFragment, MemoryType, CheckpointManager,
};

use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Main memory system for AI agents
pub struct AgentMemory {
    config: MemoryConfig,
    state: memory::state::AgentState,
    storage: std::sync::Arc<dyn memory::storage::Storage + Send + Sync>,
    checkpoint_manager: memory::checkpoint::CheckpointManager,
    knowledge_graph: Option<memory::knowledge_graph::MemoryKnowledgeGraph>,
    temporal_manager: Option<memory::temporal::TemporalMemoryManager>,
    advanced_manager: Option<memory::management::AdvancedMemoryManager>,
}

impl AgentMemory {
    /// Create a new agent memory system with the given configuration
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        let storage = memory::storage::create_storage(&config.storage_backend).await?;
        let checkpoint_manager = memory::checkpoint::CheckpointManager::new(
            config.checkpoint_interval,
            storage.clone(),
        );

        let state = memory::state::AgentState::new(config.session_id.unwrap_or_else(Uuid::new_v4));

        // Initialize knowledge graph if enabled
        let knowledge_graph = if config.enable_knowledge_graph {
            let graph_config = memory::knowledge_graph::GraphConfig::default();
            Some(memory::knowledge_graph::MemoryKnowledgeGraph::new(graph_config))
        } else {
            None
        };

        // Initialize temporal manager if enabled
        let temporal_manager = if config.enable_temporal_tracking {
            let temporal_config = memory::temporal::TemporalConfig::default();
            Some(memory::temporal::TemporalMemoryManager::new(temporal_config))
        } else {
            None
        };

        // Initialize advanced memory manager if enabled
        let advanced_manager = if config.enable_advanced_management {
            let mgmt_config = memory::management::MemoryManagementConfig::default();
            Some(memory::management::AdvancedMemoryManager::new(mgmt_config))
        } else {
            None
        };

        Ok(Self {
            config,
            state,
            storage,
            checkpoint_manager,
            knowledge_graph,
            temporal_manager,
            advanced_manager,
        })
    }

    /// Store a memory entry with intelligent updating
    pub async fn store(&mut self, key: &str, value: &str) -> Result<()> {
        let entry = MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::ShortTerm);

        // Check if this is an update to existing memory
        let is_update = self.state.has_memory(key);
        let change_type = if is_update {
            memory::temporal::ChangeType::Updated
        } else {
            memory::temporal::ChangeType::Created
        };

        self.state.add_memory(entry.clone());
        self.storage.store(&entry).await?;

        // Track temporal changes if enabled
        if let Some(ref mut tm) = self.temporal_manager {
            let _ = tm.track_memory_change(&entry, change_type).await;
        }

        // Add or update in knowledge graph if enabled (intelligent merging)
        if let Some(ref mut kg) = self.knowledge_graph {
            let _ = kg.add_or_update_memory_node(&entry).await;
        }

        // Use advanced management if enabled
        if let Some(ref mut am) = self.advanced_manager {
            let _ = am.add_memory(entry.clone(), self.knowledge_graph.as_mut()).await;
        }

        // Check if we need to create a checkpoint
        if self.checkpoint_manager.should_checkpoint(&self.state) {
            self.checkpoint_manager.create_checkpoint(&self.state).await?;
        }

        Ok(())
    }

    /// Retrieve a memory by key
    pub async fn retrieve(&mut self, key: &str) -> Result<Option<MemoryEntry>> {
        // First check short-term memory
        if let Some(entry) = self.state.get_memory(key) {
            return Ok(Some(entry.clone()));
        }

        // Then check storage
        self.storage.retrieve(key).await
    }

    /// Search memories by content similarity
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        self.storage.search(query, limit).await
    }

    /// Create a checkpoint of the current state
    pub async fn checkpoint(&self) -> Result<Uuid> {
        self.checkpoint_manager.create_checkpoint(&self.state).await
    }

    /// Restore from a checkpoint
    pub async fn restore_checkpoint(&mut self, checkpoint_id: Uuid) -> Result<()> {
        let state = self.checkpoint_manager.restore_checkpoint(checkpoint_id).await?;
        self.state = state;

        // Also need to sync the storage with the restored state
        // For now, we'll clear storage and re-populate it from the state
        self.storage.clear().await?;

        // Re-populate storage with memories from restored state
        for (_, entry) in self.state.get_short_term_memories() {
            self.storage.store(entry).await?;
        }
        for (_, entry) in self.state.get_long_term_memories() {
            self.storage.store(entry).await?;
        }

        Ok(())
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            short_term_count: self.state.short_term_memory_count(),
            long_term_count: self.state.long_term_memory_count(),
            total_size: self.state.total_memory_size(),
            session_id: self.state.session_id(),
            created_at: self.state.created_at(),
        }
    }

    /// Clear all memories (use with caution)
    pub async fn clear(&mut self) -> Result<()> {
        self.state.clear();
        self.storage.clear().await?;

        // Clear knowledge graph if enabled
        if let Some(ref mut kg) = self.knowledge_graph {
            *kg = memory::knowledge_graph::MemoryKnowledgeGraph::new(
                memory::knowledge_graph::GraphConfig::default()
            );
        }

        Ok(())
    }

    /// Create a relationship between two memories in the knowledge graph
    pub async fn create_memory_relationship(
        &mut self,
        from_memory: &str,
        to_memory: &str,
        relationship_type: memory::knowledge_graph::RelationshipType,
    ) -> Result<Option<Uuid>> {
        if let Some(ref mut kg) = self.knowledge_graph {
            let relationship_id = kg.create_relationship(
                from_memory,
                to_memory,
                relationship_type,
                None,
            ).await?;
            Ok(Some(relationship_id))
        } else {
            Ok(None)
        }
    }

    /// Find related memories using the knowledge graph
    pub async fn find_related_memories(
        &self,
        memory_key: &str,
        max_depth: usize,
    ) -> Result<Vec<memory::knowledge_graph::RelatedMemory>> {
        if let Some(ref kg) = self.knowledge_graph {
            kg.find_related_memories(memory_key, max_depth, None).await
        } else {
            Ok(Vec::new())
        }
    }

    /// Get knowledge graph statistics
    pub fn knowledge_graph_stats(&self) -> Option<memory::knowledge_graph::GraphStats> {
        self.knowledge_graph.as_ref().map(|kg| kg.get_stats())
    }

    /// Perform inference to discover new relationships
    pub async fn infer_relationships(&mut self) -> Result<Vec<memory::knowledge_graph::reasoning::InferenceResult>> {
        if let Some(ref mut kg) = self.knowledge_graph {
            kg.infer_relationships().await
        } else {
            Ok(Vec::new())
        }
    }

    /// Check if a memory exists
    pub fn has_memory(&self, key: &str) -> bool {
        self.state.has_memory(key)
    }
}

/// Memory system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub short_term_count: usize,
    pub long_term_count: usize,
    pub total_size: usize,
    pub session_id: Uuid,
    pub created_at: DateTime<Utc>,
}

/// Configuration for the memory system
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub storage_backend: StorageBackend,
    pub session_id: Option<Uuid>,
    pub checkpoint_interval: usize,
    pub max_short_term_memories: usize,
    pub max_long_term_memories: usize,
    pub similarity_threshold: f64,
    pub enable_knowledge_graph: bool,
    pub enable_temporal_tracking: bool,
    pub enable_advanced_management: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            storage_backend: StorageBackend::Memory,
            session_id: None,
            checkpoint_interval: 100,
            max_short_term_memories: 1000,
            max_long_term_memories: 10000,
            similarity_threshold: 0.7,
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
        }
    }
}

/// Storage backend options
#[derive(Debug, Clone)]
pub enum StorageBackend {
    Memory,
    File { path: String },
    #[cfg(feature = "sql-storage")]
    Sql { connection_string: String },
}
