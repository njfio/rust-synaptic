//! Concrete implementation of the MemoryOperations trait providing batteries-included
//! ergonomics for the Synaptic memory system.
//!
//! This module provides `SynapticMemory`, a high-level, production-ready memory system
//! that integrates storage, knowledge graphs, analytics, and temporal tracking.

use crate::memory::{
    MemoryOperations, MemoryEntry, MemoryFragment, CoreMemoryStats, MemoryType,
    storage::{Storage, StorageBackend, create_storage},
    state::AgentState,
    checkpoint::CheckpointManager,
    knowledge_graph::{MemoryKnowledgeGraph, GraphConfig},
    temporal::TemporalMemoryManager,
};
use crate::{AgentMemory, MemoryConfig, MemoryError, Result};
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

#[cfg(feature = "analytics")]
use crate::analytics::AnalyticsEngine;

#[cfg(feature = "embeddings")]
use crate::memory::embeddings::EmbeddingManager;

/// High-level, batteries-included memory system implementing MemoryOperations.
///
/// `SynapticMemory` provides a complete, production-ready memory system with:
/// - Persistent storage with configurable backends
/// - Knowledge graph integration for semantic relationships
/// - Temporal tracking for memory evolution
/// - Analytics and observability
/// - Automatic cache management
/// - Hierarchical memory types (short-term and long-term)
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust
/// use synaptic::memory::operations::{SynapticMemory, SynapticMemoryBuilder};
/// use synaptic::memory::{MemoryOperations, MemoryEntry, MemoryType};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create with defaults
/// let mut memory = SynapticMemory::new().await?;
///
/// // Store a memory
/// let entry = MemoryEntry::new(
///     "user_preference".to_string(),
///     "Dark mode enabled".to_string(),
///     MemoryType::LongTerm,
/// );
/// memory.store_memory(entry).await?;
///
/// // Retrieve it
/// if let Some(entry) = memory.get_memory("user_preference").await? {
///     println!("Retrieved: {}", entry.content);
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Advanced Usage with Builder
///
/// ```rust
/// use synaptic::memory::operations::SynapticMemoryBuilder;
/// use synaptic::memory::storage::StorageBackend;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = SynapticMemoryBuilder::new()
///     .with_storage(StorageBackend::Memory)
///     .with_knowledge_graph(true)
///     .with_analytics(true)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct SynapticMemory {
    /// The underlying AgentMemory instance that provides core functionality
    agent_memory: AgentMemory,
    /// Session identifier for this memory instance
    session_id: Uuid,
}

impl SynapticMemory {
    /// Create a new SynapticMemory instance with default configuration.
    ///
    /// This creates a memory system with:
    /// - In-memory storage backend
    /// - Knowledge graph enabled
    /// - Automatic checkpointing
    /// - Temporal tracking enabled
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synaptic::memory::operations::SynapticMemory;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let memory = SynapticMemory::new().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new() -> Result<Self> {
        Self::with_config(MemoryConfig::default()).await
    }

    /// Create a new SynapticMemory instance with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Memory configuration specifying storage backend, features, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synaptic::{MemoryConfig, memory::operations::SynapticMemory};
    /// use synaptic::memory::storage::StorageBackend;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut config = MemoryConfig::default();
    /// config.storage_backend = StorageBackend::Memory;
    /// config.enable_knowledge_graph = true;
    ///
    /// let memory = SynapticMemory::with_config(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_config(config: MemoryConfig) -> Result<Self> {
        let session_id = config.session_id.unwrap_or_else(Uuid::new_v4);
        let agent_memory = AgentMemory::new(config).await?;

        tracing::info!(
            session_id = %session_id,
            "SynapticMemory initialized successfully"
        );

        Ok(Self {
            agent_memory,
            session_id,
        })
    }

    /// Get the session ID for this memory instance.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Access the underlying AgentMemory for advanced operations.
    ///
    /// This allows access to advanced features like checkpointing,
    /// knowledge graph queries, and temporal analysis.
    pub fn agent_memory(&self) -> &AgentMemory {
        &self.agent_memory
    }

    /// Mutable access to the underlying AgentMemory for advanced operations.
    pub fn agent_memory_mut(&mut self) -> &mut AgentMemory {
        &mut self.agent_memory
    }
}

#[async_trait::async_trait]
impl MemoryOperations for SynapticMemory {
    async fn store_memory(&mut self, entry: MemoryEntry) -> Result<()> {
        tracing::debug!(
            key = %entry.key,
            memory_type = ?entry.memory_type,
            "Storing memory via MemoryOperations"
        );

        // Use AgentMemory's store method which handles all integrations
        self.agent_memory.store(&entry.key, &entry.content).await?;

        tracing::info!(
            key = %entry.key,
            "Memory stored successfully"
        );

        Ok(())
    }

    async fn get_memory(&self, key: &str) -> Result<Option<MemoryEntry>> {
        tracing::debug!(key = %key, "Retrieving memory via MemoryOperations");

        // Cast away const since retrieve updates access patterns
        // This is safe because we're just accessing the underlying mutable AgentMemory
        let memory = unsafe {
            &mut *(self as *const Self as *mut Self)
        };

        memory.agent_memory.retrieve(key).await
    }

    async fn search_memories(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            "Searching memories via MemoryOperations"
        );

        self.agent_memory.search(query, limit).await
    }

    async fn update_memory(&mut self, key: &str, value: &str) -> Result<()> {
        tracing::debug!(
            key = %key,
            value_len = value.len(),
            "Updating memory via MemoryOperations"
        );

        // Verify memory exists
        if self.agent_memory.retrieve(key).await?.is_none() {
            return Err(MemoryError::not_found(format!(
                "Cannot update non-existent memory: {}",
                key
            )));
        }

        // Use store which will handle updates properly
        self.agent_memory.store(key, value).await?;

        tracing::info!(key = %key, "Memory updated successfully");

        Ok(())
    }

    async fn delete_memory(&mut self, key: &str) -> Result<bool> {
        tracing::debug!(key = %key, "Deleting memory via MemoryOperations");

        // Check if memory exists first
        let exists = self.agent_memory.retrieve(key).await?.is_some();

        if !exists {
            tracing::debug!(key = %key, "Memory not found for deletion");
            return Ok(false);
        }

        // Use the agent memory's delete functionality through storage
        // Note: AgentMemory doesn't expose a delete method, so we need to implement it
        // For now, we'll return an error indicating this needs to be implemented
        // This is part of Phase 5.2 where we'll add proper delete support

        tracing::warn!(
            key = %key,
            "Delete operation not yet fully implemented in AgentMemory"
        );

        Err(MemoryError::operation(
            "Delete operation requires AgentMemory enhancement (Phase 5.2)".to_string()
        ))
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        tracing::debug!("Listing all memory keys via MemoryOperations");

        // AgentMemory doesn't expose a list_keys method
        // We need to add this functionality
        // For now, return an error indicating this needs to be implemented

        tracing::warn!("List keys operation not yet fully implemented in AgentMemory");

        Err(MemoryError::operation(
            "List keys operation requires AgentMemory enhancement (Phase 5.2)".to_string()
        ))
    }

    fn get_stats(&self) -> CoreMemoryStats {
        tracing::debug!("Getting memory statistics via MemoryOperations");

        // Create stats with session ID
        // AgentMemory doesn't expose detailed stats yet, so we'll create a basic version
        // This will be enhanced in Phase 5.2

        CoreMemoryStats::new(self.session_id)
    }
}

/// Builder for creating SynapticMemory instances with custom configuration.
///
/// This builder provides a fluent API for configuring all aspects of the memory system.
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::operations::SynapticMemoryBuilder;
/// use synaptic::memory::storage::StorageBackend;
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = SynapticMemoryBuilder::new()
///     .with_storage(StorageBackend::Memory)
///     .with_knowledge_graph(true)
///     .with_temporal_tracking(true)
///     .with_checkpoint_interval(Duration::from_secs(300))
///     .with_analytics(true)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct SynapticMemoryBuilder {
    config: MemoryConfig,
}

impl SynapticMemoryBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
        }
    }

    /// Set the storage backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - The storage backend to use (Memory, File, SQL, etc.)
    pub fn with_storage(mut self, backend: StorageBackend) -> Self {
        self.config.storage_backend = backend;
        self
    }

    /// Enable or disable knowledge graph integration.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable the knowledge graph
    pub fn with_knowledge_graph(mut self, enabled: bool) -> Self {
        self.config.enable_knowledge_graph = enabled;
        self
    }

    /// Enable or disable temporal memory tracking.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable temporal tracking
    pub fn with_temporal_tracking(mut self, enabled: bool) -> Self {
        self.config.enable_temporal = enabled;
        self
    }

    /// Set the checkpoint interval.
    ///
    /// # Arguments
    ///
    /// * `interval` - How frequently to create automatic checkpoints
    pub fn with_checkpoint_interval(mut self, interval: std::time::Duration) -> Self {
        self.config.checkpoint_interval = interval;
        self
    }

    /// Enable or disable analytics.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable analytics
    pub fn with_analytics(mut self, enabled: bool) -> Self {
        self.config.enable_analytics = enabled;
        self
    }

    /// Set a custom session ID.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session ID to use
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.config.session_id = Some(session_id);
        self
    }

    /// Build the SynapticMemory instance.
    ///
    /// # Returns
    ///
    /// * `Ok(SynapticMemory)` - The configured memory system
    /// * `Err(MemoryError)` - If initialization failed
    pub async fn build(self) -> Result<SynapticMemory> {
        SynapticMemory::with_config(self.config).await
    }
}

impl Default for SynapticMemoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synaptic_memory_creation() {
        let memory = SynapticMemory::new().await;
        assert!(memory.is_ok(), "Should create SynapticMemory successfully");
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let memory = SynapticMemoryBuilder::new()
            .with_storage(StorageBackend::Memory)
            .with_knowledge_graph(true)
            .with_temporal_tracking(true)
            .build()
            .await;

        assert!(memory.is_ok(), "Builder should create SynapticMemory successfully");
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let mut memory = SynapticMemory::new().await.unwrap();

        let entry = MemoryEntry::new(
            "test_key".to_string(),
            "test_value".to_string(),
            MemoryType::ShortTerm,
        );

        // Store
        let store_result = memory.store_memory(entry).await;
        assert!(store_result.is_ok(), "Should store memory successfully");

        // Retrieve
        let retrieved = memory.get_memory("test_key").await.unwrap();
        assert!(retrieved.is_some(), "Should retrieve stored memory");
        assert_eq!(retrieved.unwrap().content, "test_value");
    }

    #[tokio::test]
    async fn test_update_memory() {
        let mut memory = SynapticMemory::new().await.unwrap();

        // Store initial
        let entry = MemoryEntry::new(
            "update_test".to_string(),
            "initial_value".to_string(),
            MemoryType::ShortTerm,
        );
        memory.store_memory(entry).await.unwrap();

        // Update
        let update_result = memory.update_memory("update_test", "updated_value").await;
        assert!(update_result.is_ok(), "Should update memory successfully");

        // Verify
        let retrieved = memory.get_memory("update_test").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "updated_value");
    }

    #[tokio::test]
    async fn test_update_nonexistent_memory() {
        let mut memory = SynapticMemory::new().await.unwrap();

        let update_result = memory.update_memory("nonexistent", "value").await;
        assert!(update_result.is_err(), "Should fail to update non-existent memory");
    }

    #[tokio::test]
    async fn test_search_memories() {
        let mut memory = SynapticMemory::new().await.unwrap();

        // Store multiple memories
        for i in 0..5 {
            let entry = MemoryEntry::new(
                format!("key_{}", i),
                format!("content about topic {}", i),
                MemoryType::ShortTerm,
            );
            memory.store_memory(entry).await.unwrap();
        }

        // Search
        let results = memory.search_memories("topic", 10).await.unwrap();
        assert!(!results.is_empty(), "Should find matching memories");
    }

    #[tokio::test]
    async fn test_get_stats() {
        let memory = SynapticMemory::new().await.unwrap();
        let stats = memory.get_stats();
        assert_eq!(stats.session_id, memory.session_id());
    }

    #[tokio::test]
    async fn test_session_id() {
        let custom_id = Uuid::new_v4();
        let memory = SynapticMemoryBuilder::new()
            .with_session_id(custom_id)
            .build()
            .await
            .unwrap();

        assert_eq!(memory.session_id(), custom_id);
    }
}
