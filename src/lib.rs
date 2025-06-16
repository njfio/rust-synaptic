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
pub mod memory;
pub mod logging;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "analytics")]
pub mod analytics;

pub mod integrations;
pub mod performance;
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

/// Main memory system for AI agents
pub struct AgentMemory {
    config: MemoryConfig,
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
    integration_manager: Option<integrations::IntegrationManager>,
    security_manager: Option<security::SecurityManager>,
    #[cfg(feature = "multimodal")]
    multimodal_memory: Option<std::sync::Arc<tokio::sync::RwLock<multimodal::unified::UnifiedMultiModalMemory>>>,
    #[cfg(feature = "cross-platform")]
    cross_platform_manager: Option<cross_platform::CrossPlatformMemoryManager>,
}

impl AgentMemory {
    /// Create a new agent memory system with the given configuration
    #[tracing::instrument(skip(config), fields(session_id = %config.session_id.unwrap_or_else(Uuid::new_v4)))]
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        tracing::info!("Initializing AgentMemory with configuration");

        let storage = memory::storage::create_storage(&config.storage_backend).await?;
        tracing::debug!("Storage backend initialized: {:?}", config.storage_backend);

        let checkpoint_manager = memory::checkpoint::CheckpointManager::new(
            config.checkpoint_interval,
            storage.clone(),
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
        let security_manager = if config.enable_security {
            let security_config = config.security_config.clone().unwrap_or_default();
            Some(security::SecurityManager::new(security_config).await?)
        } else {
            None
        };

        // Initialize cross-platform manager if enabled
        #[cfg(feature = "cross-platform")]
        let cross_platform_manager = if config.enable_cross_platform {
            let cross_platform_config = config.cross_platform_config.clone().unwrap_or_default();
            Some(cross_platform::CrossPlatformMemoryManager::new(cross_platform_config)?)
        } else {
            None
        };

        // Build base agent without multimodal memory initialized
        let agent = Self {
            config: config.clone(),
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
            integration_manager,
            security_manager,
            #[cfg(feature = "multimodal")]
            multimodal_memory: None,
            #[cfg(feature = "cross-platform")]
            cross_platform_manager,
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
        tracing::debug!("Storing memory entry");

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

        // Add or update in knowledge graph if enabled (intelligent merging)
        if let Some(ref mut kg) = self.knowledge_graph {
            let _ = kg.add_or_update_memory_node(&entry).await;
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
        tracing::debug!("Retrieving memory entry");

        // First check short-term memory
        if let Some(entry) = self.state.get_memory(key) {
            tracing::debug!("Memory found in short-term memory");
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

        if result.is_some() {
            tracing::debug!("Memory found in storage");
        } else {
            tracing::debug!("Memory not found");
        }
        #[cfg(feature = "analytics")]
        if let Some(ref mut analytics) = self.analytics_engine {
            if result.is_some() {
                use crate::analytics::{AnalyticsEvent, AccessType};
                let event = AnalyticsEvent::MemoryAccess {
                    memory_key: key.to_string(),
                    access_type: AccessType::Read,
                    timestamp: Utc::now(),
                    user_context: None,
                };
                let _ = analytics.record_event(event).await;
            }
        }
        Ok(result)
    }

    /// Search memories by content similarity
    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), limit = limit))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        tracing::debug!("Searching memories by content similarity");
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
    pub enable_security: bool,
    pub security_config: Option<security::SecurityConfig>,
    #[cfg(feature = "multimodal")]
    pub enable_multimodal: bool,
    #[cfg(feature = "multimodal")]
    pub multimodal_config: Option<multimodal::unified::UnifiedMultiModalConfig>,
    #[cfg(feature = "cross-platform")]
    pub enable_cross_platform: bool,
    #[cfg(feature = "cross-platform")]
    pub cross_platform_config: Option<cross_platform::CrossPlatformConfig>,
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
            enable_security: false,
            security_config: None,
            #[cfg(feature = "multimodal")]
            enable_multimodal: false,
            #[cfg(feature = "multimodal")]
            multimodal_config: None,
            #[cfg(feature = "cross-platform")]
            enable_cross_platform: false,
            #[cfg(feature = "cross-platform")]
            cross_platform_config: None,
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
