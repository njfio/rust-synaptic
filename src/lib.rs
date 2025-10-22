//! # Synaptic
//!
//! An intelligent AI agent memory system built in Rust that creates and manages
//! dynamic knowledge graphs with smart content updates. Unlike traditional memory
//! systems that create duplicate entries, Synaptic intelligently merges similar
//! content and evolves relationships over time.
//!
//! ## Key Features
//!
//! - **Intelligent Memory Updates**: Smart node merging and content evolution tracking
//! - **Advanced Knowledge Graph**: Dynamic relationship detection and reasoning engine
//! - **Temporal Intelligence**: Version history and pattern detection
//! - **Advanced Search & Retrieval**: Multi-criteria search with relevance ranking
//! - **Memory Management**: Intelligent summarization and lifecycle policies
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
pub mod error_handling;
pub mod memory;
pub mod logging;
pub mod cli;
pub mod observability;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "analytics")]
pub mod analytics;

pub mod integrations;
pub mod performance;

#[cfg(feature = "security")]
pub mod security;

#[cfg(feature = "multimodal")]
pub mod multimodal;

#[cfg(feature = "cross-platform")]
pub mod cross_platform;

// Basic Phase 5 implementation (always available)
pub mod phase5_basic;

// Phase 5B: Advanced Document Processing (Basic implementation always available)
pub mod phase5b_basic;

// Re-export main types for convenience
pub use error::{MemoryError, Result};
pub use memory::{
    MemoryEntry, MemoryFragment, MemoryType, CheckpointManager,
};

use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Main memory system for AI agents
pub struct AgentMemory {
    _config: MemoryConfig,
    state: memory::state::AgentState,
    storage: std::sync::Arc<dyn memory::storage::Storage + Send + Sync>,
    checkpoint_manager: memory::checkpoint::CheckpointManager,
    knowledge_graph: Option<memory::knowledge_graph::MemoryKnowledgeGraph>,
    temporal_manager: Option<memory::temporal::TemporalMemoryManager>,
    advanced_manager: Option<memory::management::AdvancedMemoryManager>,
    #[cfg(feature = "embeddings")]
    embedding_manager: Option<memory::embeddings::EmbeddingManager>,
    #[cfg(feature = "distributed")]
    distributed_coordinator: Option<std::sync::Arc<distributed::coordination::DistributedCoordinator>>,
    #[cfg(feature = "analytics")]
    analytics_engine: Option<analytics::AnalyticsEngine>,
    _integration_manager: Option<integrations::IntegrationManager>,
    #[cfg(feature = "security")]
    _security_manager: Option<security::SecurityManager>,
    #[cfg(not(feature = "security"))]
    _security_manager: Option<()>,
    #[cfg(feature = "multimodal")]
    multimodal_memory: Option<std::sync::Arc<tokio::sync::RwLock<multimodal::unified::UnifiedMultiModalMemory>>>,
    #[cfg(feature = "cross-platform")]
    cross_platform_manager: Option<cross_platform::CrossPlatformMemoryManager>,
    /// Memory promotion manager for hierarchical memory management
    promotion_manager: Option<memory::promotion::MemoryPromotionManager>,
}

impl AgentMemory {
    /// Create a new agent memory system with the given configuration
    #[tracing::instrument(skip(config), fields(session_id = %config.session_id.unwrap_or_else(Uuid::new_v4)))]
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        // Initialize logging system first
        if let Some(ref logging_config) = config.logging_config {
            let logging_manager = logging::LoggingManager::new(logging_config.clone());
            if let Err(e) = logging_manager.initialize() {
                // Use tracing warn since this is a non-critical initialization failure
                tracing::warn!(
                    error = %e,
                    "Failed to initialize logging system"
                );
            }
        }

        tracing::info!("Initializing AgentMemory with configuration");

        let storage = memory::storage::create_storage(&config.storage_backend).await?;
        tracing::debug!("Storage backend initialized: {:?}", config.storage_backend);

        let checkpoint_manager = memory::checkpoint::CheckpointManager::new(
            config.checkpoint_interval,
            Arc::clone(&storage),
        );
        tracing::debug!("Checkpoint manager initialized with interval: {:?}", config.checkpoint_interval);

        let state = memory::state::AgentState::new(config.session_id.unwrap_or_else(Uuid::new_v4));
        tracing::debug!("Agent state initialized");

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

        // Initialize embedding manager if enabled
        #[cfg(feature = "embeddings")]
        let embedding_manager = if config.enable_embeddings {
            let embedding_config = memory::embeddings::EmbeddingConfig::default();
            Some(memory::embeddings::EmbeddingManager::new(embedding_config))
        } else {
            None
        };

        // Initialize distributed coordinator if enabled
        #[cfg(feature = "distributed")]
        let distributed_coordinator = if config.enable_distributed {
            if let Some(dist_config) = config.distributed_config.clone() {
                let coordinator = distributed::coordination::DistributedCoordinator::new(dist_config).await?;
                Some(std::sync::Arc::new(coordinator))
            } else {
                None
            }
        } else {
            None
        };

        // Initialize analytics engine if enabled
        #[cfg(feature = "analytics")]
        let analytics_engine = if config.enable_analytics {
            let analytics_config = config.analytics_config.clone().unwrap_or_default();
            Some(analytics::AnalyticsEngine::new(analytics_config)?)
        } else {
            None
        };

        // Initialize integration manager if enabled
        let integration_manager = if config.enable_integrations {
            let integrations_config = config.integrations_config.clone().unwrap_or_default();
            Some(integrations::IntegrationManager::new(integrations_config).await?)
        } else {
            None
        };

        // Initialize security manager if enabled
        #[cfg(feature = "security")]
        let security_manager = if config.enable_security {
            let security_config = config.security_config.clone().unwrap_or_default();
            Some(security::SecurityManager::new(security_config).await?)
        } else {
            None
        };

        #[cfg(not(feature = "security"))]
        let security_manager: Option<()> = None;

        // Initialize cross-platform manager if enabled
        #[cfg(feature = "cross-platform")]
        let cross_platform_manager = if config.enable_cross_platform {
            let cross_platform_config = config.cross_platform_config.clone().unwrap_or_default();
            Some(cross_platform::CrossPlatformMemoryManager::new(cross_platform_config)?)
        } else {
            None
        };

        // Initialize memory promotion manager if enabled
        let promotion_manager = if config.enable_memory_promotion {
            let promo_config = config.promotion_config.clone().unwrap_or_default();
            tracing::debug!(
                policy_type = ?promo_config.policy_type,
                "Initializing memory promotion manager"
            );
            Some(promo_config.create_manager())
        } else {
            None
        };

        // Build base agent without multimodal memory initialized
        let agent = Self {
            _config: config.clone(),
            state,
            storage,
            checkpoint_manager,
            knowledge_graph,
            temporal_manager,
            advanced_manager,
            #[cfg(feature = "embeddings")]
            embedding_manager,
            #[cfg(feature = "distributed")]
            distributed_coordinator,
            #[cfg(feature = "analytics")]
            analytics_engine,
            _integration_manager: integration_manager,
            _security_manager: security_manager,
            #[cfg(feature = "multimodal")]
            multimodal_memory: None,
            #[cfg(feature = "cross-platform")]
            cross_platform_manager,
            promotion_manager,
        };

        // Initialize multimodal memory after creating base agent to avoid circular dependency
        #[cfg(feature = "multimodal")]
        if config.enable_multimodal {
            let multimodal_config = config.multimodal_config.clone().unwrap_or_default();
            let agent_arc = std::sync::Arc::new(tokio::sync::RwLock::new(agent));
            let mm = multimodal::unified::UnifiedMultiModalMemory::new(agent_arc.clone(), multimodal_config).await?;
            {
                let mut guard = agent_arc.write().await;
                guard.multimodal_memory = Some(std::sync::Arc::new(tokio::sync::RwLock::new(mm)));
            }
            agent = std::sync::Arc::try_unwrap(agent_arc)
                .map_err(|_| MemoryError::concurrency("Failed to unwrap Arc during initialization"))?
                .into_inner();
        }

        Ok(agent)
    }

    /// Store a memory entry with intelligent updating
    #[tracing::instrument(skip(self, value), fields(key = %key, value_len = value.len()))]
    pub async fn store(&mut self, key: &str, value: &str) -> Result<()> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};

        tracing::debug!("Storing memory entry");

        // Validate inputs
        validate_non_empty_string(key, "memory key")?;
        validate_non_empty_string(value, "memory value")?;
        validate_range(value.len(), 1, 10_000_000, "memory value length")?; // Max 10MB
        validate_range(key.len(), 1, 1000, "memory key length")?; // Max 1000 chars

        let entry = MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::ShortTerm);

        // Check if this is an update to existing memory
        let is_update = self.state.has_memory(key);
        let change_type = if is_update {
            memory::temporal::ChangeType::Updated
        } else {
            memory::temporal::ChangeType::Created
        };

        tracing::debug!("Memory operation type: {:?}", change_type);

        self.state.add_memory(entry.clone());
        self.storage.store(&entry).await?;
        tracing::debug!("Memory stored in state and storage");

        // Track temporal changes if enabled
        if let Some(ref mut tm) = self.temporal_manager {
            let _ = tm.track_memory_change(&entry, change_type).await;
        }

        #[cfg(feature = "analytics")]
        if let Some(ref mut analytics) = self.analytics_engine {
            use crate::analytics::{AnalyticsEvent, ModificationType};
            let event = AnalyticsEvent::MemoryModification {
                memory_key: key.to_string(),
                modification_type: ModificationType::ContentUpdate,
                timestamp: Utc::now(),
                change_magnitude: 1.0,
            };
            let _ = analytics.record_event(event).await;
        }

        // Sync with knowledge graph if enabled (uses MemoryGraphSync trait)
        if let Some(ref mut kg) = self.knowledge_graph {
            use memory::knowledge_graph::MemoryGraphSync;
            let _ = kg.sync_created(&entry).await;
        }

        // Use advanced management if enabled
        if let Some(ref mut am) = self.advanced_manager {
            let _ = am.add_memory(&*self.storage, entry.clone(), self.knowledge_graph.as_mut()).await;
        }

        // Generate embeddings if enabled
        #[cfg(feature = "embeddings")]
        if let Some(ref mut em) = self.embedding_manager {
            let _ = em.add_memory(entry.clone()).map_err(|e| {
                eprintln!("Warning: Failed to generate embedding: {}", e);
            });
        }

        // Check if we need to create a checkpoint
        if self.checkpoint_manager.should_checkpoint(&self.state) {
            self.checkpoint_manager.create_checkpoint(&self.state).await?;
        }

        Ok(())
    }

    /// Retrieve a memory by key
    #[tracing::instrument(skip(self), fields(key = %key))]
    pub async fn retrieve(&mut self, key: &str) -> Result<Option<MemoryEntry>> {
        use crate::error_handling::utils::validate_non_empty_string;

        tracing::debug!("Retrieving memory entry");

        // Validate input
        validate_non_empty_string(key, "memory key")?;

        // First check short-term memory
        if let Some(mut entry) = self.state.get_memory(key) {
            tracing::debug!("Memory found in short-term memory");

            // Check if memory should be promoted to long-term
            if let Some(ref pm) = self.promotion_manager {
                if pm.should_promote(&entry) {
                    tracing::info!(
                        memory_key = %entry.key,
                        access_count = entry.access_count,
                        importance = entry.importance,
                        "Automatically promoting memory to long-term storage"
                    );
                    entry = pm.promote_memory(entry)?;
                    // Update in state and storage
                    self.state.add_memory(entry.clone());
                    self.storage.store(&entry).await?;
                }
            }

            #[cfg(feature = "analytics")]
            if let Some(ref mut analytics) = self.analytics_engine {
                use crate::analytics::{AnalyticsEvent, AccessType};
                let event = AnalyticsEvent::MemoryAccess {
                    memory_key: key.to_string(),
                    access_type: AccessType::Read,
                    timestamp: Utc::now(),
                    user_context: None,
                };
                let _ = analytics.record_event(event).await;
            }
            return Ok(Some(entry.clone()));
        }

        // Then check storage
        tracing::debug!("Memory not found in short-term, checking storage");
        let result = self.storage.retrieve(key).await?;

        if let Some(mut entry) = result {
            tracing::debug!("Memory found in storage (cache miss), rehydrating state");

            // CRITICAL: Inject cache miss back into state for future fast access
            // This fixes the cache synchronization issue where repeated access
            // to the same memory would hit storage every time.

            // Update access patterns
            entry.mark_accessed();

            // Check if memory should be promoted to long-term
            if let Some(ref pm) = self.promotion_manager {
                if pm.should_promote(&entry) {
                    tracing::info!(
                        memory_key = %entry.key,
                        access_count = entry.access_count,
                        importance = entry.importance,
                        "Automatically promoting memory to long-term storage"
                    );
                    entry = pm.promote_memory(entry)?;
                    // Store the promoted memory back
                    self.storage.store(&entry).await?;
                }
            }

            // Add to state for future fast access
            self.state.add_memory(entry.clone());

            // Refresh knowledge graph if enabled
            if let Some(ref mut kg) = self.knowledge_graph {
                tracing::debug!("Refreshing knowledge graph node for cache miss");
                let _ = kg.add_or_update_memory_node(&entry).await;
            }

            #[cfg(feature = "analytics")]
            if let Some(ref mut analytics) = self.analytics_engine {
                use crate::analytics::{AnalyticsEvent, AccessType};
                let event = AnalyticsEvent::MemoryAccess {
                    memory_key: key.to_string(),
                    access_type: AccessType::Read,
                    timestamp: Utc::now(),
                    user_context: None,
                };
                let _ = analytics.record_event(event).await;
            }

            Ok(Some(entry))
        } else {
            tracing::debug!("Memory not found in storage");
            Ok(None)
        }
    }

    /// Update an existing memory entry
    ///
    /// This method updates the value of an existing memory while preserving its metadata
    /// and automatically synchronizing changes with the knowledge graph.
    #[tracing::instrument(skip(self, new_value), fields(key = %key, new_value_len = new_value.len()))]
    pub async fn update(&mut self, key: &str, new_value: &str) -> Result<()> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};

        tracing::debug!("Updating memory entry");

        // Validate inputs
        validate_non_empty_string(key, "memory key")?;
        validate_non_empty_string(new_value, "memory value")?;
        validate_range(new_value.len(), 1, 10_000_000, "memory value length")?; // Max 10MB

        // Retrieve existing memory
        let mut existing = self.retrieve(key).await?
            .ok_or_else(|| MemoryError::not_found(key))?;

        // Keep a copy of the old memory for graph sync
        let old_memory = existing.clone();

        // Update the memory value
        existing.update_value(new_value.to_string());

        // Update in state and storage
        self.state.add_memory(existing.clone());
        self.storage.store(&existing).await?;

        tracing::debug!("Memory updated in state and storage");

        // Track temporal changes if enabled
        if let Some(ref mut tm) = self.temporal_manager {
            let _ = tm.track_memory_change(&existing, memory::temporal::ChangeType::Updated).await;
        }

        #[cfg(feature = "analytics")]
        if let Some(ref mut analytics) = self.analytics_engine {
            use crate::analytics::{AnalyticsEvent, ModificationType};
            let event = AnalyticsEvent::MemoryModification {
                memory_key: key.to_string(),
                modification_type: ModificationType::ContentUpdate,
                timestamp: Utc::now(),
                change_magnitude: 1.0,
            };
            let _ = analytics.record_event(event).await;
        }

        // Sync with knowledge graph if enabled
        if let Some(ref mut kg) = self.knowledge_graph {
            use memory::knowledge_graph::MemoryGraphSync;
            let _ = kg.sync_updated(&old_memory, &existing).await;
        }

        tracing::info!(
            memory_key = %key,
            "Successfully updated memory"
        );

        Ok(())
    }

    /// Delete a memory entry
    ///
    /// This method removes a memory from both state and storage, and automatically
    /// cleans up the corresponding node in the knowledge graph.
    #[tracing::instrument(skip(self), fields(key = %key))]
    pub async fn delete(&mut self, key: &str) -> Result<()> {
        use crate::error_handling::utils::validate_non_empty_string;

        tracing::debug!("Deleting memory entry");

        // Validate input
        validate_non_empty_string(key, "memory key")?;

        // Retrieve the memory before deletion (for graph sync)
        let existing = self.retrieve(key).await?
            .ok_or_else(|| MemoryError::not_found(key))?;

        // Delete from state
        self.state.remove_memory(key);

        // Delete from storage
        self.storage.delete(key).await?;

        tracing::debug!("Memory deleted from state and storage");

        // Track deletion in temporal manager
        if let Some(ref mut tm) = self.temporal_manager {
            let _ = tm.track_memory_change(&existing, memory::temporal::ChangeType::Deleted).await;
        }

        #[cfg(feature = "analytics")]
        if let Some(ref mut analytics) = self.analytics_engine {
            use crate::analytics::{AnalyticsEvent, ModificationType};
            let event = AnalyticsEvent::MemoryModification {
                memory_key: key.to_string(),
                modification_type: ModificationType::Deleted,
                timestamp: Utc::now(),
                change_magnitude: 0.0,
            };
            let _ = analytics.record_event(event).await;
        }

        // Sync deletion with knowledge graph if enabled
        if let Some(ref mut kg) = self.knowledge_graph {
            use memory::knowledge_graph::MemoryGraphSync;
            let _ = kg.sync_deleted(&existing).await;
        }

        tracing::info!(
            memory_key = %key,
            "Successfully deleted memory"
        );

        Ok(())
    }

    /// Search memories by content similarity
    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), limit = limit))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};

        tracing::debug!("Searching memories by content similarity");

        // Validate inputs
        validate_non_empty_string(query, "search query")?;
        validate_range(limit, 1, 10000, "search limit")?; // Max 10k results

        let results = self.storage.search(query, limit).await?;
        tracing::debug!("Search completed, found {} results", results.len());
        Ok(results)
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
        for entry in self.state.get_short_term_memories().values() {
            self.storage.store(entry).await?;
        }
        for entry in self.state.get_long_term_memories().values() {
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
        use crate::error_handling::utils::validate_non_empty_string;

        // Validate inputs
        validate_non_empty_string(from_memory, "from_memory key")?;
        validate_non_empty_string(to_memory, "to_memory key")?;

        if let Some(ref mut kg) = self.knowledge_graph {
            let relationship_id = kg.create_relationship(
                from_memory,
                to_memory,
                relationship_type,
                None,
            ).await?;

            #[cfg(feature = "analytics")]
            if let Some(ref mut analytics) = self.analytics_engine {
                use crate::analytics::AnalyticsEvent;
                let event = AnalyticsEvent::RelationshipDiscovery {
                    source_key: from_memory.to_string(),
                    target_key: to_memory.to_string(),
                    relationship_strength: 1.0,
                    timestamp: Utc::now(),
                };
                let _ = analytics.record_event(event).await;
            }

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

    /// Find shortest path between two memories in the knowledge graph
    pub async fn find_path_between_memories(
        &self,
        from_memory: &str,
        to_memory: &str,
        max_depth: Option<usize>,
    ) -> Result<Option<memory::knowledge_graph::GraphPath>> {
        if let Some(ref kg) = self.knowledge_graph {
            kg.find_path_between_memories(from_memory, to_memory, max_depth).await
        } else {
            Ok(None)
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

    /// Query memories with full graph context
    ///
    /// This unified API retrieves memories that match the query and enriches them with
    /// knowledge graph context including related memories, relationships, and graph metrics.
    #[tracing::instrument(skip(self), fields(query = %query, max_depth = options.max_depth))]
    pub async fn query_with_graph_context(
        &mut self,
        query: &str,
        options: QueryContextOptions,
    ) -> Result<Vec<MemoryWithGraphContext>> {
        tracing::debug!("Querying memories with graph context");

        // Step 1: Search for relevant memories
        let search_results = self.search(query, options.search_limit).await?;

        let mut contextualized_memories = Vec::new();

        for fragment in search_results {
            // Retrieve full memory entry
            let memory = match self.retrieve(&fragment.key).await? {
                Some(m) => m,
                None => continue, // Skip if memory was deleted
            };

            // Step 2: Get graph context if enabled
            let graph_context = if options.include_graph_context && self.knowledge_graph.is_some() {
                let related = self.find_related_memories(
                    &fragment.key,
                    options.max_depth,
                    None,
                ).await.unwrap_or_default();

                Some(GraphContext {
                    related_memories: related,
                    total_relationships: related.len(),
                })
            } else {
                None
            };

            contextualized_memories.push(MemoryWithGraphContext {
                memory,
                relevance_score: fragment.relevance_score,
                graph_context,
            });
        }

        tracing::info!(
            found_count = contextualized_memories.len(),
            "Query with graph context completed"
        );

        Ok(contextualized_memories)
    }

    /// Check if a memory exists
    pub fn has_memory(&self, key: &str) -> bool {
        self.state.has_memory(key)
    }

    /// Semantic search using embeddings (if enabled)
    #[cfg(feature = "embeddings")]
    pub fn semantic_search(&mut self, query: &str, limit: Option<usize>) -> Result<Vec<memory::embeddings::SimilarMemory>> {
        if let Some(ref mut embedding_manager) = self.embedding_manager {
            embedding_manager.find_similar_to_query(query, limit)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get embedding statistics (if enabled)
    #[cfg(feature = "embeddings")]
    pub fn embedding_stats(&self) -> Option<memory::embeddings::EmbeddingStats> {
        self.embedding_manager.as_ref().map(|em| em.get_stats())
    }

    /// Get analytics metrics (if enabled)
    #[cfg(feature = "analytics")]
    pub fn get_analytics_metrics(&self) -> Option<analytics::AnalyticsMetrics> {
        self.analytics_engine.as_ref().map(|eng| eng.get_usage_stats())
    }

    /// Get temporal usage statistics (if enabled)
    pub async fn get_temporal_usage_stats(&self) -> Option<memory::temporal::TemporalUsageStats> {
        if let Some(ref tm) = self.temporal_manager {
            tm.get_usage_stats().await.ok()
        } else {
            None
        }
    }

    /// Get differential metrics from the temporal manager
    pub fn get_temporal_diff_metrics(&self) -> Option<memory::temporal::DiffMetrics> {
        self.temporal_manager.as_ref().map(|tm| tm.get_diff_metrics())
    }

    /// Get global evolution metrics from the temporal manager
    pub async fn get_global_evolution_metrics(&self) -> Option<memory::temporal::GlobalEvolutionMetrics> {
        if let Some(ref tm) = self.temporal_manager {
            tm.get_global_evolution_metrics().await.ok()
        } else {
            None
        }
    }

    /// Check if the multi-modal subsystem is initialized
    #[cfg(feature = "multimodal")]
    pub fn multimodal_enabled(&self) -> bool {
        self.multimodal_memory.is_some()
    }

    /// Get access to the storage backend
    pub fn storage(&self) -> &std::sync::Arc<dyn memory::storage::Storage + Send + Sync> {
        &self.storage
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

/// Options for querying memories with graph context
#[derive(Debug, Clone)]
pub struct QueryContextOptions {
    /// Maximum number of memories to return from search
    pub search_limit: usize,
    /// Include graph context (related memories, relationships)
    pub include_graph_context: bool,
    /// Maximum depth for relationship traversal
    pub max_depth: usize,
}

impl Default for QueryContextOptions {
    fn default() -> Self {
        Self {
            search_limit: 10,
            include_graph_context: true,
            max_depth: 2,
        }
    }
}

impl QueryContextOptions {
    /// Create options with custom search limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.search_limit = limit;
        self
    }

    /// Create options with custom max depth
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Create options without graph context
    pub fn without_graph_context(mut self) -> Self {
        self.include_graph_context = false;
        self
    }
}

/// A memory enriched with knowledge graph context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWithGraphContext {
    /// The core memory entry
    pub memory: MemoryEntry,
    /// Relevance score from search
    pub relevance_score: f64,
    /// Optional graph context with relationships
    pub graph_context: Option<GraphContext>,
}

/// Graph context for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContext {
    /// Related memories discovered through graph traversal
    pub related_memories: Vec<memory::knowledge_graph::RelatedMemory>,
    /// Total number of relationships
    pub total_relationships: usize,
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
    /// Knowledge graph synchronization configuration
    pub graph_sync_config: Option<memory::knowledge_graph::GraphSyncConfig>,
    pub enable_temporal_tracking: bool,
    pub enable_advanced_management: bool,
    #[cfg(feature = "embeddings")]
    pub enable_embeddings: bool,
    #[cfg(feature = "distributed")]
    pub enable_distributed: bool,
    #[cfg(feature = "distributed")]
    pub distributed_config: Option<distributed::DistributedConfig>,
    #[cfg(feature = "analytics")]
    pub enable_analytics: bool,
    #[cfg(feature = "analytics")]
    pub analytics_config: Option<analytics::AnalyticsConfig>,
    pub enable_integrations: bool,
    pub integrations_config: Option<integrations::IntegrationConfig>,
    #[cfg(feature = "security")]
    pub enable_security: bool,
    #[cfg(feature = "security")]
    pub security_config: Option<security::SecurityConfig>,
    #[cfg(feature = "multimodal")]
    pub enable_multimodal: bool,
    #[cfg(feature = "multimodal")]
    pub multimodal_config: Option<multimodal::unified::UnifiedMultiModalConfig>,
    #[cfg(feature = "cross-platform")]
    pub enable_cross_platform: bool,
    #[cfg(feature = "cross-platform")]
    pub cross_platform_config: Option<cross_platform::CrossPlatformConfig>,
    /// Logging and monitoring configuration
    pub logging_config: Option<logging::LoggingConfig>,
    /// Enable automatic memory promotion from short-term to long-term
    pub enable_memory_promotion: bool,
    /// Memory promotion configuration
    pub promotion_config: Option<memory::promotion::PromotionConfig>,
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
            graph_sync_config: Some(memory::knowledge_graph::GraphSyncConfig::default()),
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            #[cfg(feature = "embeddings")]
            enable_embeddings: true,
            #[cfg(feature = "distributed")]
            enable_distributed: false,
            #[cfg(feature = "distributed")]
            distributed_config: None,
            #[cfg(feature = "analytics")]
            enable_analytics: false,
            #[cfg(feature = "analytics")]
            analytics_config: None,
            enable_integrations: false,
            integrations_config: None,
            #[cfg(feature = "security")]
            enable_security: false,
            #[cfg(feature = "security")]
            security_config: None,
            #[cfg(feature = "multimodal")]
            enable_multimodal: false,
            #[cfg(feature = "multimodal")]
            multimodal_config: None,
            #[cfg(feature = "cross-platform")]
            enable_cross_platform: false,
            #[cfg(feature = "cross-platform")]
            cross_platform_config: None,
            logging_config: Some(logging::LoggingConfig::default()),
            enable_memory_promotion: true,
            promotion_config: Some(memory::promotion::PromotionConfig::default()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();

        assert!(matches!(config.storage_backend, StorageBackend::Memory));
        assert!(config.session_id.is_none());
        assert_eq!(config.checkpoint_interval, 100);
        assert_eq!(config.max_short_term_memories, 1000);
        assert_eq!(config.max_long_term_memories, 10000);
        assert_eq!(config.similarity_threshold, 0.7);
        assert!(config.enable_knowledge_graph);
        assert!(config.enable_temporal_tracking);
        assert!(config.enable_advanced_management);
        assert!(!config.enable_integrations);
        assert!(config.logging_config.is_some());
    }

    #[test]
    fn test_memory_config_clone() {
        let config1 = MemoryConfig::default();
        let config2 = config1.clone();

        assert_eq!(config1.checkpoint_interval, config2.checkpoint_interval);
        assert_eq!(config1.max_short_term_memories, config2.max_short_term_memories);
        assert_eq!(config1.max_long_term_memories, config2.max_long_term_memories);
        assert_eq!(config1.similarity_threshold, config2.similarity_threshold);
    }

    #[test]
    fn test_memory_config_custom_values() {
        let session_id = Uuid::new_v4();
        let mut config = MemoryConfig::default();
        config.session_id = Some(session_id);
        config.checkpoint_interval = 50;
        config.max_short_term_memories = 500;
        config.max_long_term_memories = 5000;
        config.similarity_threshold = 0.8;
        config.enable_knowledge_graph = false;

        assert_eq!(config.session_id, Some(session_id));
        assert_eq!(config.checkpoint_interval, 50);
        assert_eq!(config.max_short_term_memories, 500);
        assert_eq!(config.max_long_term_memories, 5000);
        assert_eq!(config.similarity_threshold, 0.8);
        assert!(!config.enable_knowledge_graph);
    }

    #[test]
    fn test_storage_backend_memory() {
        let backend = StorageBackend::Memory;
        assert!(matches!(backend, StorageBackend::Memory));
    }

    #[test]
    fn test_storage_backend_file() {
        let backend = StorageBackend::File {
            path: "/tmp/test.db".to_string(),
        };

        match backend {
            StorageBackend::File { path } => {
                assert_eq!(path, "/tmp/test.db");
            }
            _ => panic!("Expected File backend"),
        }
    }

    #[test]
    fn test_storage_backend_clone() {
        let backend1 = StorageBackend::File {
            path: "/tmp/test.db".to_string(),
        };
        let backend2 = backend1.clone();

        match (backend1, backend2) {
            (StorageBackend::File { path: p1 }, StorageBackend::File { path: p2 }) => {
                assert_eq!(p1, p2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[cfg(feature = "sql-storage")]
    #[test]
    fn test_storage_backend_sql() {
        let backend = StorageBackend::Sql {
            connection_string: "postgres://localhost/test".to_string(),
        };

        match backend {
            StorageBackend::Sql { connection_string } => {
                assert_eq!(connection_string, "postgres://localhost/test");
            }
            _ => panic!("Expected Sql backend"),
        }
    }

    #[test]
    fn test_memory_config_with_session_id() {
        let session_id = Uuid::new_v4();
        let mut config = MemoryConfig::default();
        config.session_id = Some(session_id);

        assert_eq!(config.session_id, Some(session_id));
    }

    #[test]
    fn test_memory_config_knowledge_graph_disabled() {
        let mut config = MemoryConfig::default();
        config.enable_knowledge_graph = false;
        config.enable_temporal_tracking = false;
        config.enable_advanced_management = false;

        assert!(!config.enable_knowledge_graph);
        assert!(!config.enable_temporal_tracking);
        assert!(!config.enable_advanced_management);
    }

    #[test]
    fn test_memory_config_similarity_threshold() {
        let mut config = MemoryConfig::default();

        // Test default
        assert_eq!(config.similarity_threshold, 0.7);

        // Test custom values
        config.similarity_threshold = 0.9;
        assert_eq!(config.similarity_threshold, 0.9);

        config.similarity_threshold = 0.5;
        assert_eq!(config.similarity_threshold, 0.5);
    }

    #[test]
    fn test_memory_config_checkpoint_interval() {
        let mut config = MemoryConfig::default();

        // Test default
        assert_eq!(config.checkpoint_interval, 100);

        // Test custom values
        config.checkpoint_interval = 10;
        assert_eq!(config.checkpoint_interval, 10);

        config.checkpoint_interval = 1000;
        assert_eq!(config.checkpoint_interval, 1000);
    }

    #[test]
    fn test_memory_config_memory_limits() {
        let mut config = MemoryConfig::default();

        // Test defaults
        assert_eq!(config.max_short_term_memories, 1000);
        assert_eq!(config.max_long_term_memories, 10000);

        // Test custom values
        config.max_short_term_memories = 100;
        config.max_long_term_memories = 1000;

        assert_eq!(config.max_short_term_memories, 100);
        assert_eq!(config.max_long_term_memories, 1000);
    }

    #[test]
    fn test_memory_config_file_storage_backend() {
        let mut config = MemoryConfig::default();
        config.storage_backend = StorageBackend::File {
            path: "/tmp/memory.db".to_string(),
        };

        match config.storage_backend {
            StorageBackend::File { path } => {
                assert_eq!(path, "/tmp/memory.db");
            }
            _ => panic!("Expected File backend"),
        }
    }

    #[test]
    fn test_memory_config_integrations_disabled() {
        let config = MemoryConfig::default();
        assert!(!config.enable_integrations);
        assert!(config.integrations_config.is_none());
    }

    #[test]
    fn test_memory_config_logging_enabled_by_default() {
        let config = MemoryConfig::default();
        assert!(config.logging_config.is_some());
    }

    #[cfg(feature = "embeddings")]
    #[test]
    fn test_memory_config_embeddings_feature() {
        let config = MemoryConfig::default();
        assert!(config.enable_embeddings);
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_memory_config_distributed_feature() {
        let config = MemoryConfig::default();
        assert!(!config.enable_distributed);
        assert!(config.distributed_config.is_none());
    }

    #[cfg(feature = "analytics")]
    #[test]
    fn test_memory_config_analytics_feature() {
        let config = MemoryConfig::default();
        assert!(!config.enable_analytics);
        assert!(config.analytics_config.is_none());
    }

    #[cfg(feature = "security")]
    #[test]
    fn test_memory_config_security_feature() {
        let config = MemoryConfig::default();
        assert!(!config.enable_security);
        assert!(config.security_config.is_none());
    }

    #[cfg(feature = "multimodal")]
    #[test]
    fn test_memory_config_multimodal_feature() {
        let config = MemoryConfig::default();
        assert!(!config.enable_multimodal);
        assert!(config.multimodal_config.is_none());
    }

    #[cfg(feature = "cross-platform")]
    #[test]
    fn test_memory_config_cross_platform_feature() {
        let config = MemoryConfig::default();
        assert!(!config.enable_cross_platform);
        assert!(config.cross_platform_config.is_none());
    }
}
