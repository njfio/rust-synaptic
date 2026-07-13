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
//!     // Default config: in-memory storage, knowledge graph enabled
//!     let mut memory = AgentMemory::new(MemoryConfig::default()).await?;
//!
//!     // Store memories by key
//!     memory.store("user_name", "Alice").await?;
//!     memory.store("user_preference", "prefers dark mode").await?;
//!
//!     // Retrieve by key
//!     let entry = memory.retrieve("user_name").await?;
//!     assert_eq!(entry.map(|e| e.value), Some("Alice".to_string()));
//!
//!     // Search (tokenized keyword + semantic hybrid retrieval)
//!     let results = memory.search("dark mode", 10).await?;
//!     assert!(!results.is_empty());
//!
//!     Ok(())
//! }
//! ```

pub mod cli;
pub mod error;
pub mod error_handling;
pub mod logging;
pub mod memory;
#[cfg(feature = "observability")]
pub mod observability;

#[cfg(feature = "distributed-experimental")]
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
    store_result::StoreDegradations, CheckpointManager, MemoryEntry, MemoryFragment, MemoryType,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Main memory system for AI agents
pub struct AgentMemory {
    _config: MemoryConfig,
    state: memory::state::AgentState,
    storage: std::sync::Arc<dyn memory::storage::Storage + Send + Sync>,
    checkpoint_manager: memory::checkpoint::CheckpointManager,
    /// Knowledge graph shared with the retrieval pipeline's `GraphRetriever`
    /// (hence the `Arc<RwLock<..>>`).
    knowledge_graph:
        Option<Arc<tokio::sync::RwLock<memory::knowledge_graph::MemoryKnowledgeGraph>>>,
    temporal_manager: Option<memory::temporal::TemporalMemoryManager>,
    advanced_manager: Option<memory::management::AdvancedMemoryManager>,
    #[cfg(feature = "embeddings")]
    embedding_manager: Option<memory::embeddings::EmbeddingManager>,
    /// Reasoner driving the intelligent write path (extraction + conflict
    /// resolution); defaults to a `HeuristicReasoner` over the active embedder.
    #[cfg(feature = "embeddings")]
    reasoner: Arc<dyn memory::reasoning::MemoryReasoner>,
    /// Reflection engine that clusters accumulated memories and synthesizes
    /// provenance-linked insights via the reasoner.
    #[cfg(feature = "embeddings")]
    reflection_engine: memory::reflection::ReflectionEngine,
    /// Keys of memories stored since the last reflection (insertion order).
    #[cfg(feature = "embeddings")]
    reflection_pending: Vec<String>,
    /// Importance accumulated since the last reflection; `reflect` fires only
    /// once this reaches the configured threshold.
    #[cfg(feature = "embeddings")]
    reflection_accumulated_importance: f64,
    #[cfg(feature = "distributed-experimental")]
    distributed_coordinator:
        Option<std::sync::Arc<distributed::coordination::DistributedCoordinator>>,
    #[cfg(feature = "analytics")]
    analytics_engine: Option<analytics::AnalyticsEngine>,
    _integration_manager: Option<integrations::IntegrationManager>,
    #[cfg(feature = "security")]
    _security_manager: Option<security::SecurityManager>,
    #[cfg(not(feature = "security"))]
    _security_manager: Option<()>,
    #[cfg(feature = "multimodal")]
    multimodal_memory:
        Option<std::sync::Arc<tokio::sync::RwLock<multimodal::unified::UnifiedMultiModalMemory>>>,
    #[cfg(feature = "cross-platform")]
    cross_platform_manager: Option<cross_platform::CrossPlatformMemoryManager>,
    /// Memory promotion manager for hierarchical memory management
    promotion_manager: Option<memory::promotion::MemoryPromotionManager>,
    /// Hybrid retrieval pipeline used by `search` when embeddings are enabled;
    /// combines dense-vector, keyword, graph, and temporal signals for ranked
    /// results instead of naive substring matching, then applies composite
    /// relevance × recency × importance scoring. `None` falls back to
    /// `storage.search`.
    retrieval_pipeline: Option<Arc<memory::retrieval::HybridRetriever>>,
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
        tracing::debug!(
            "Checkpoint manager initialized with interval: {:?}",
            config.checkpoint_interval
        );

        let state = memory::state::AgentState::new(config.session_id.unwrap_or_else(Uuid::new_v4));
        tracing::debug!("Agent state initialized");

        // Initialize knowledge graph if enabled
        let knowledge_graph = if config.enable_knowledge_graph {
            let graph_config = memory::knowledge_graph::GraphConfig::default();
            Some(Arc::new(tokio::sync::RwLock::new(
                memory::knowledge_graph::MemoryKnowledgeGraph::new(graph_config),
            )))
        } else {
            None
        };

        // Initialize temporal manager if enabled
        let temporal_manager = if config.enable_temporal_tracking {
            let temporal_config = memory::temporal::TemporalConfig::default();
            Some(memory::temporal::TemporalMemoryManager::new(
                temporal_config,
            ))
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

        // Default reasoner for the intelligent write path: the deterministic
        // heuristic reasoner over the active (TF-IDF) embedding provider.
        #[cfg(feature = "embeddings")]
        let reasoner: Arc<dyn memory::reasoning::MemoryReasoner> =
            Arc::new(memory::reasoning::HeuristicReasoner::new(Arc::new(
                memory::embeddings::TfIdfProvider::default(),
            )));

        // Reflection engine: clusters memories accumulated since the last
        // reflection by TF-IDF embedding similarity and synthesizes insights
        // through the same reasoner as the write path.
        #[cfg(feature = "embeddings")]
        let reflection_engine = memory::reflection::ReflectionEngine::new(
            memory::reflection::ReflectionConfig::default(),
            Arc::new(memory::embeddings::TfIdfProvider::default()),
        );

        // Initialize the hybrid retrieval pipeline if embeddings are enabled. This
        // routes `search` through ranked dense-vector + keyword signals instead of
        // naive substring matching. It intentionally builds its own lightweight
        // TF-IDF embedding provider rather than sharing `embedding_manager`
        // (which owns a different, stateful embedding representation used for
        // `find_similar_memories`); both read from the same underlying storage,
        // so results stay consistent without a large ownership refactor.
        //
        // Candidate generation for each signal retriever goes through
        // `storage.search`, which performs tokenized OR-matching (any query
        // term matches), so multi-word queries yield candidates directly and
        // no widening wrapper is needed.
        let retrieval_pipeline = if config.enable_embeddings {
            let provider = Arc::new(memory::embeddings::TfIdfProvider::default());
            let dense_vector =
                memory::retrieval::DenseVectorRetriever::new(Arc::clone(&storage), provider);
            let keyword = memory::retrieval::KeywordRetriever::new(Arc::clone(&storage));
            // Graph and temporal signals share the same storage; the graph
            // retriever additionally shares the live knowledge-graph handle
            // (it reports itself unavailable when the graph is disabled).
            let graph = memory::retrieval::GraphRetriever::new(
                Arc::clone(&storage),
                knowledge_graph.clone(),
            );
            let temporal = memory::retrieval::TemporalRetriever::new(Arc::clone(&storage));
            let pipeline_config = memory::retrieval::PipelineConfig::semantic_focus();
            // Deterministic heuristic reranker over the top-K: cross-features
            // (term overlap, embedding agreement, graph proximity, recency)
            // reorder the fused + composite-scored results.
            let reranker = memory::retrieval::HeuristicReranker::new(
                Some(Arc::new(memory::embeddings::TfIdfProvider::default())),
                knowledge_graph.clone(),
            );
            let hybrid = memory::retrieval::HybridRetriever::new(pipeline_config)
                .add_pipeline(Arc::new(dense_vector))
                .add_pipeline(Arc::new(keyword))
                .add_pipeline(Arc::new(graph))
                .add_pipeline(Arc::new(temporal))
                .with_reranker(Arc::new(reranker));
            Some(Arc::new(hybrid))
        } else {
            None
        };
        tracing::debug!(
            enabled = retrieval_pipeline.is_some(),
            "Retrieval pipeline initialized"
        );

        // Initialize distributed coordinator if enabled
        #[cfg(feature = "distributed-experimental")]
        let distributed_coordinator = if config.enable_distributed {
            if let Some(dist_config) = config.distributed_config.clone() {
                let coordinator =
                    distributed::coordination::DistributedCoordinator::new(dist_config).await?;
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
            Some(cross_platform::CrossPlatformMemoryManager::new(
                cross_platform_config,
            )?)
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
            #[cfg(feature = "embeddings")]
            reasoner,
            #[cfg(feature = "embeddings")]
            reflection_engine,
            #[cfg(feature = "embeddings")]
            reflection_pending: Vec::new(),
            #[cfg(feature = "embeddings")]
            reflection_accumulated_importance: 0.0,
            #[cfg(feature = "distributed-experimental")]
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
            retrieval_pipeline,
        };

        // Initialize multimodal memory after creating base agent to avoid circular dependency
        #[cfg(feature = "multimodal")]
        if config.enable_multimodal {
            let multimodal_config = config.multimodal_config.clone().unwrap_or_default();
            let agent_arc = std::sync::Arc::new(tokio::sync::RwLock::new(agent));
            let mm = multimodal::unified::UnifiedMultiModalMemory::new(
                agent_arc.clone(),
                multimodal_config,
            )
            .await?;
            {
                let mut guard = agent_arc.write().await;
                guard.multimodal_memory = Some(std::sync::Arc::new(tokio::sync::RwLock::new(mm)));
            }
            agent = std::sync::Arc::try_unwrap(agent_arc)
                .map_err(|_| {
                    MemoryError::concurrency("Failed to unwrap Arc during initialization")
                })?
                .into_inner();
        }

        Ok(agent)
    }

    /// Test-only hook to swap the storage backend after construction, used to
    /// inject failing/faulty storage doubles for testing storage-failure
    /// behavior (e.g. that a storage write failure cannot leave the state
    /// cache polluted with an entry that was never durably persisted).
    #[cfg(feature = "test-utils")]
    pub fn set_storage_for_test(
        &mut self,
        storage: std::sync::Arc<dyn memory::storage::Storage + Send + Sync>,
    ) {
        self.storage = storage;
    }

    /// Store a memory entry with intelligent updating
    #[tracing::instrument(skip(self, value), fields(key = %key, value_len = value.len()))]
    pub async fn store(&mut self, key: &str, value: &str) -> Result<()> {
        // `store_with_report` already logs each degradation at `warn` as it
        // occurs; this wrapper preserves the original `store()` contract by
        // discarding the report and always returning `Ok(())` once the core
        // storage write succeeds.
        let _degradations = self.store_with_report(key, value).await?;

        Ok(())
    }

    /// Store a memory entry with intelligent updating, reporting any subsystem
    /// degradations instead of silently swallowing them. Returns an error only
    /// if the core storage write itself fails.
    #[tracing::instrument(skip(self, value), fields(key = %key, value_len = value.len()))]
    pub async fn store_with_report(
        &mut self,
        key: &str,
        value: &str,
    ) -> Result<memory::store_result::StoreDegradations> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};
        use memory::store_result::StoreDegradations;

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

        self.storage.store(&entry).await?;
        self.state.add_memory(entry.clone());
        tracing::debug!("Memory stored in state and storage");

        // Track accumulation for triggered reflection.
        #[cfg(feature = "embeddings")]
        {
            if !self.reflection_pending.iter().any(|k| k == key) {
                self.reflection_pending.push(key.to_string());
            }
            self.reflection_accumulated_importance += entry.metadata.importance;
        }

        let mut degradations = StoreDegradations::default();

        // Track temporal changes if enabled
        if let Some(ref mut tm) = self.temporal_manager {
            if let Err(e) = tm.track_memory_change(&entry, change_type).await {
                tracing::warn!(error = %e, "temporal tracking degraded during store");
                degradations.temporal = Some(e.to_string());
            }
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
            if let Err(e) = analytics.record_event(event).await {
                tracing::warn!(error = %e, "analytics recording degraded during store");
                degradations.analytics = Some(e.to_string());
            }
        }

        // Add or update in knowledge graph if enabled (intelligent merging)
        if let Some(ref kg) = self.knowledge_graph {
            if let Err(e) = kg.write().await.add_or_update_memory_node(&entry).await {
                tracing::warn!(error = %e, "knowledge graph update degraded during store");
                degradations.knowledge_graph = Some(e.to_string());
            }
        }

        // Use advanced management if enabled
        if let Some(ref mut am) = self.advanced_manager {
            let mut kg_guard = match self.knowledge_graph {
                Some(ref kg) => Some(kg.write().await),
                None => None,
            };
            if let Err(e) = am
                .add_memory(&*self.storage, entry.clone(), kg_guard.as_deref_mut())
                .await
            {
                tracing::warn!(error = %e, "advanced management update degraded during store");
                degradations.advanced_management = Some(e.to_string());
            }
        }

        // Generate embeddings if enabled
        #[cfg(feature = "embeddings")]
        if let Some(ref mut em) = self.embedding_manager {
            if let Err(e) = em.add_memory(entry.clone()) {
                tracing::warn!(error = %e, "embedding generation degraded during store");
                degradations.embeddings = Some(e.to_string());
            }
        }

        // Intelligent write path: extract facts from the new value, resolve
        // each against similar existing memories, and apply the outcome to
        // the knowledge graph. Best-effort like the other subsystems.
        #[cfg(feature = "embeddings")]
        if self.knowledge_graph.is_some() {
            if let Err(e) = self.reason_over_store(&entry).await {
                tracing::warn!(error = %e, "reasoning degraded during store");
                degradations.reasoning = Some(e.to_string());
            }
        }

        // Check if we need to create a checkpoint
        if self.checkpoint_manager.should_checkpoint(&self.state) {
            self.checkpoint_manager
                .create_checkpoint(&self.state)
                .await?;
        }

        Ok(degradations)
    }

    /// Run the reasoner over a freshly stored entry: extract facts, resolve
    /// each against the most similar existing memories, and apply the
    /// resolution to the knowledge graph.
    #[cfg(feature = "embeddings")]
    async fn reason_over_store(&mut self, entry: &MemoryEntry) -> Result<()> {
        let ctx = memory::reasoning::ExtractionContext {
            source_key: entry.key.clone(),
            timestamp: Utc::now(),
        };
        let reasoner = Arc::clone(&self.reasoner);
        let extraction = reasoner.extract(&entry.value, &ctx).await?;
        // Empty extraction: default behavior stays identical (no KG change
        // beyond what the earlier subsystems already applied).
        if extraction.facts.is_empty() {
            return Ok(());
        }
        for fact in &extraction.facts {
            let neighbors = self.neighbor_facts(&entry.key, &fact.text, 5);
            let resolution = reasoner.resolve(fact, &neighbors).await?;
            // The reasoner picks its target among the neighbors we supplied;
            // for id-less outcomes (UpdateInPlace/Append) the target is the
            // best neighbor by the same deterministic ordering.
            let best_neighbor_id = neighbors.first().map(|n| n.id.clone());
            self.apply_resolution(entry, fact, resolution, best_neighbor_id)
                .await?;
        }
        Ok(())
    }

    /// Build the top-`k` neighbor list for a candidate fact from existing
    /// memories (short- and long-term state), excluding the entry being
    /// stored. Similarity is a deterministic token-set cosine; ties break by
    /// key so resolution is reproducible.
    #[cfg(feature = "embeddings")]
    fn neighbor_facts(
        &self,
        current_key: &str,
        fact_text: &str,
        k: usize,
    ) -> Vec<memory::reasoning::NeighborFact> {
        fn tokens(text: &str) -> std::collections::HashSet<String> {
            text.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect()
        }
        let candidate_tokens = tokens(fact_text);
        if candidate_tokens.is_empty() {
            return Vec::new();
        }
        let mut neighbors: Vec<memory::reasoning::NeighborFact> = self
            .state
            .get_short_term_memories()
            .iter()
            .chain(self.state.get_long_term_memories().iter())
            .filter(|(key, _)| key.as_str() != current_key)
            .filter_map(|(key, mem)| {
                let mem_tokens = tokens(&mem.value);
                if mem_tokens.is_empty() {
                    return None;
                }
                let intersection = candidate_tokens.intersection(&mem_tokens).count() as f64;
                let similarity = intersection
                    / ((candidate_tokens.len() as f64).sqrt() * (mem_tokens.len() as f64).sqrt());
                if similarity > 0.0 {
                    Some(memory::reasoning::NeighborFact {
                        id: key.clone(),
                        similarity,
                        text: mem.value.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        neighbors.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.id.cmp(&b.id))
        });
        neighbors.truncate(k);
        neighbors
    }

    /// Apply a conflict resolution outcome to the knowledge graph.
    #[cfg(feature = "embeddings")]
    async fn apply_resolution(
        &mut self,
        entry: &MemoryEntry,
        fact: &memory::reasoning::Fact,
        resolution: memory::reasoning::ConflictResolution,
        best_neighbor_id: Option<String>,
    ) -> Result<()> {
        use memory::reasoning::ConflictResolution;
        let Some(ref kg) = self.knowledge_graph else {
            return Ok(());
        };
        let mut kg = kg.write().await;
        match resolution {
            ConflictResolution::Insert => {
                kg.add_extracted_fact(&entry.key, fact).await?;
            }
            ConflictResolution::Supersede { old_id, reason } => {
                tracing::debug!(old_id = %old_id, reason = %reason, "superseding fact");
                // Bi-temporally invalidate only the contradicted edges of
                // the old memory (same subject + predicate as the new fact).
                kg.supersede_matching_relations(&old_id, fact).await?;
                kg.add_extracted_fact(&entry.key, fact).await?;
            }
            ConflictResolution::UpdateInPlace { reason } => {
                tracing::debug!(reason = %reason, "updating fact in place");
                if let Some(target) = best_neighbor_id {
                    kg.update_memory_node_content(&target, &fact.text).await?;
                }
            }
            ConflictResolution::Append { reason } => {
                tracing::debug!(reason = %reason, "appending to existing fact");
                if let Some(target) = best_neighbor_id {
                    kg.append_memory_content(&target, &fact.text).await?;
                }
            }
            ConflictResolution::NoOp { reason } => {
                tracing::debug!(reason = %reason, "reasoner chose no-op");
            }
        }
        Ok(())
    }

    /// Query the knowledge graph for all extracted relations whose subject is
    /// the given entity (case-insensitive). Returns an empty list when the
    /// knowledge graph is disabled or the entity is unknown.
    pub async fn entity_relations(
        &self,
        entity_name: &str,
    ) -> Result<Vec<memory::knowledge_graph::ExtractedRelation>> {
        match self.knowledge_graph {
            Some(ref kg) => kg.read().await.relations_for_entity(entity_name).await,
            None => Ok(Vec::new()),
        }
    }

    /// Replace the active reflection configuration (threshold, cluster
    /// parameters). Does not reset the importance already accumulated.
    #[cfg(feature = "embeddings")]
    pub fn set_reflection_config(&mut self, config: memory::reflection::ReflectionConfig) {
        self.reflection_engine.set_config(config);
    }

    /// Reflect over the memories accumulated since the last reflection.
    ///
    /// Triggers only once the accumulated importance reaches the configured
    /// `importance_threshold` (returns an empty vec otherwise). When it
    /// fires, the accumulated memories are clustered by embedding similarity
    /// (connected components over pairwise cosine); every cluster of at least
    /// `min_cluster` members is passed to the reasoner's `synthesize`. Each
    /// resulting [`memory::reasoning::Insight`] is written as a new
    /// long-term memory (tagged `insight`, custom field
    /// `memory_source = "reflection"`) and linked in the knowledge graph
    /// with a `derives` edge from the insight to each source memory.
    /// Firing consumes the accumulated importance and pending set.
    #[cfg(feature = "embeddings")]
    pub async fn reflect(&mut self) -> Result<Vec<memory::reasoning::Insight>> {
        let threshold = self.reflection_engine.config().importance_threshold;
        if self.reflection_accumulated_importance < threshold {
            return Ok(Vec::new());
        }

        let pending: Vec<String> = self.reflection_pending.clone();
        let mut entries: Vec<MemoryEntry> = Vec::with_capacity(pending.len());
        for key in &pending {
            // A pending memory may have been deleted since it was stored;
            // reflection simply skips it.
            if let Some(entry) = self.state.get_memory(key) {
                entries.push(entry);
            }
        }

        let reasoner = Arc::clone(&self.reasoner);
        let insights = self
            .reflection_engine
            .reflect(&entries, reasoner.as_ref())
            .await?;

        // `Insight::derived_from` carries memory ids; provenance edges are
        // keyed by memory key, so map between the two.
        let id_to_key: std::collections::HashMap<String, String> = entries
            .iter()
            .map(|e| (e.id().to_string(), e.key.clone()))
            .collect();
        for insight in &insights {
            self.write_insight(insight, &id_to_key).await?;
        }

        // The trigger is consumed whether or not any cluster produced an
        // insight: the same accumulation is never reflected over twice.
        self.reflection_pending.clear();
        self.reflection_accumulated_importance = 0.0;

        Ok(insights)
    }

    /// Persist a synthesized insight as a new long-term memory and record
    /// `derives` provenance edges from it to each source memory in the
    /// knowledge graph.
    #[cfg(feature = "embeddings")]
    async fn write_insight(
        &mut self,
        insight: &memory::reasoning::Insight,
        id_to_key: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let key = format!("insight_{}", Uuid::new_v4());
        let mut metadata = memory::types::MemoryMetadata::new()
            .with_tags(vec!["insight".to_string()])
            .with_importance(insight.confidence);
        metadata.set_custom_field("memory_source".to_string(), "reflection".to_string());
        metadata.set_custom_field("derived_from".to_string(), insight.derived_from.join(","));
        let entry = MemoryEntry::new(key.clone(), insight.text.clone(), MemoryType::LongTerm)
            .with_metadata(metadata);

        self.storage.store(&entry).await?;
        self.state.add_memory(entry.clone());

        if let Some(ref kg) = self.knowledge_graph {
            let mut kg = kg.write().await;
            kg.add_or_update_memory_node(&entry).await?;
            for source_id in &insight.derived_from {
                if let Some(source_key) = id_to_key.get(source_id) {
                    let properties = std::collections::HashMap::from([
                        ("source".to_string(), "reflection".to_string()),
                        ("confidence".to_string(), insight.confidence.to_string()),
                    ]);
                    kg.create_relationship(
                        &key,
                        source_key,
                        memory::knowledge_graph::RelationshipType::Custom("derives".to_string()),
                        Some(properties),
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }

    /// All insight memories produced by reflection (custom field
    /// `memory_source == "reflection"`), sorted by key for determinism.
    #[cfg(feature = "embeddings")]
    pub fn insight_memories(&self) -> Vec<MemoryEntry> {
        let mut insights: Vec<MemoryEntry> = self
            .state
            .get_short_term_memories()
            .values()
            .chain(self.state.get_long_term_memories().values())
            .filter(|e| {
                e.metadata
                    .get_custom_field("memory_source")
                    .map(String::as_str)
                    == Some("reflection")
            })
            .cloned()
            .collect();
        insights.sort_by(|a, b| a.key.cmp(&b.key));
        insights
    }

    /// Memory keys reachable from `memory_key` over `derives` provenance
    /// edges (depth 1). For an insight memory this is exactly its source
    /// memories. Returns an empty list when the knowledge graph is disabled.
    pub async fn derived_sources(&self, memory_key: &str) -> Result<Vec<String>> {
        let Some(ref kg) = self.knowledge_graph else {
            return Ok(Vec::new());
        };
        let related = kg
            .read()
            .await
            .find_related_memories(
                memory_key,
                1,
                Some(vec![memory::knowledge_graph::RelationshipType::Custom(
                    "derives".to_string(),
                )]),
            )
            .await?;
        Ok(related
            .into_iter()
            .map(|r| r.memory_key)
            .filter(|k| k != memory_key)
            .collect())
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
                        access_count = entry.access_count(),
                        importance = entry.metadata.importance,
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
                use crate::analytics::{AccessType, AnalyticsEvent};
                let event = AnalyticsEvent::MemoryAccess {
                    memory_key: key.to_string(),
                    access_type: AccessType::Read,
                    timestamp: Utc::now(),
                    user_context: None,
                };
                if let Err(e) = analytics.record_event(event).await {
                    tracing::warn!(error = %e, memory_key = %key, "failed to record analytics event for short-term retrieve");
                }
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
                        access_count = entry.access_count(),
                        importance = entry.metadata.importance,
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
            if let Some(ref kg) = self.knowledge_graph {
                tracing::debug!("Refreshing knowledge graph node for cache miss");
                if let Err(e) = kg.write().await.add_or_update_memory_node(&entry).await {
                    tracing::warn!(error = %e, memory_key = %entry.key, "failed to refresh knowledge graph node for cache miss");
                }
            }

            #[cfg(feature = "analytics")]
            if let Some(ref mut analytics) = self.analytics_engine {
                use crate::analytics::{AccessType, AnalyticsEvent};
                let event = AnalyticsEvent::MemoryAccess {
                    memory_key: key.to_string(),
                    access_type: AccessType::Read,
                    timestamp: Utc::now(),
                    user_context: None,
                };
                if let Err(e) = analytics.record_event(event).await {
                    tracing::warn!(error = %e, memory_key = %key, "failed to record analytics event for storage retrieve");
                }
            }

            Ok(Some(entry))
        } else {
            tracing::debug!("Memory not found in storage");
            Ok(None)
        }
    }

    /// Search memories by content similarity
    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), limit = limit))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};

        tracing::debug!("Searching memories by content similarity");

        // Validate inputs
        validate_non_empty_string(query, "search query")?;
        validate_range(limit, 1, 10000, "search limit")?; // Max 10k results

        let results = if let Some(ref pipeline) = self.retrieval_pipeline {
            pipeline.search(query, limit).await?
        } else {
            self.storage.search(query, limit).await?
        };
        tracing::debug!("Search completed, found {} results", results.len());
        Ok(results)
    }

    /// Create a checkpoint of the current state
    pub async fn checkpoint(&self) -> Result<Uuid> {
        self.checkpoint_manager.create_checkpoint(&self.state).await
    }

    /// Restore from a checkpoint
    ///
    /// This is a non-destructive, upsert-then-prune restore: the checkpoint's
    /// entries are written to storage first (idempotent upserts), and only
    /// once every write has succeeded are the storage keys that are absent
    /// from the checkpoint deleted. `self.state` is swapped last, after
    /// storage reconciliation has fully succeeded. If any step fails, the
    /// error is returned and both storage and `self.state` are left exactly
    /// as they were before the call — no `clear()` is ever used, so data
    /// already durably persisted before a failure is never lost.
    ///
    /// # Recovery contract on prune failure
    ///
    /// If a delete during the prune phase fails (after all upserts have
    /// succeeded), no data is lost, but storage may be left holding the
    /// checkpoint's entries *plus* post-checkpoint keys that were not yet
    /// deleted, while `self.state` remains the pre-restore state — storage
    /// and the in-process state cache temporarily diverge. This is safe and
    /// non-destructive; the caller may simply retry `restore_checkpoint`
    /// (every phase is idempotent) to converge.
    pub async fn restore_checkpoint(&mut self, checkpoint_id: Uuid) -> Result<()> {
        let restored_state = self
            .checkpoint_manager
            .restore_checkpoint(checkpoint_id)
            .await?;

        // Step 1: upsert every checkpoint entry into storage first. These
        // writes are idempotent, so a failure partway through leaves
        // storage with a superset of the checkpoint's data plus whatever
        // was already there beforehand — nothing is lost.
        for entry in restored_state.get_short_term_memories().values() {
            self.storage.store(entry).await?;
        }
        for entry in restored_state.get_long_term_memories().values() {
            self.storage.store(entry).await?;
        }

        // Step 2: only after all upserts succeeded, compute keys present in
        // storage but absent from the checkpoint, and delete only those.
        let checkpoint_keys: std::collections::HashSet<&str> = restored_state
            .get_short_term_memories()
            .keys()
            .chain(restored_state.get_long_term_memories().keys())
            .map(|k| k.as_str())
            .collect();

        let storage_keys = self.storage.list_keys().await?;
        for key in storage_keys {
            if !checkpoint_keys.contains(key.as_str()) {
                self.storage.delete(&key).await?;
            }
        }

        // Step 3: only now that storage matches the checkpoint exactly,
        // swap the in-process state cache.
        self.state = restored_state;

        Ok(())
    }

    /// Get current memory statistics
    /// Get the session ID for this agent memory instance
    pub fn session_id(&self) -> Uuid {
        self.state.session_id()
    }

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
        if let Some(ref kg) = self.knowledge_graph {
            *kg.write().await = memory::knowledge_graph::MemoryKnowledgeGraph::new(
                memory::knowledge_graph::GraphConfig::default(),
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

        if let Some(ref kg) = self.knowledge_graph {
            let relationship_id = kg
                .write()
                .await
                .create_relationship(from_memory, to_memory, relationship_type, None)
                .await?;

            #[cfg(feature = "analytics")]
            if let Some(ref mut analytics) = self.analytics_engine {
                use crate::analytics::AnalyticsEvent;
                let event = AnalyticsEvent::RelationshipDiscovery {
                    source_key: from_memory.to_string(),
                    target_key: to_memory.to_string(),
                    relationship_strength: 1.0,
                    timestamp: Utc::now(),
                };
                if let Err(e) = analytics.record_event(event).await {
                    tracing::warn!(error = %e, from_memory = %from_memory, to_memory = %to_memory, "failed to record analytics event for relationship discovery");
                }
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
            kg.read()
                .await
                .find_related_memories(memory_key, max_depth, None)
                .await
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
            kg.read()
                .await
                .find_path_between_memories(from_memory, to_memory, max_depth)
                .await
        } else {
            Ok(None)
        }
    }

    /// Get knowledge graph statistics
    ///
    /// Async because the graph is shared with the retrieval pipeline behind
    /// an async `RwLock`.
    pub async fn knowledge_graph_stats(&self) -> Option<memory::knowledge_graph::GraphStats> {
        match self.knowledge_graph {
            Some(ref kg) => Some(kg.read().await.get_stats()),
            None => None,
        }
    }

    /// Perform inference to discover new relationships
    pub async fn infer_relationships(
        &mut self,
    ) -> Result<Vec<memory::knowledge_graph::reasoning::InferenceResult>> {
        if let Some(ref kg) = self.knowledge_graph {
            kg.write().await.infer_relationships().await
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
    pub fn semantic_search(
        &mut self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<memory::embeddings::SimilarMemory>> {
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
        self.analytics_engine
            .as_ref()
            .map(|eng| eng.get_usage_stats())
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
        self.temporal_manager
            .as_ref()
            .map(|tm| tm.get_diff_metrics())
    }

    /// Get global evolution metrics from the temporal manager
    pub async fn get_global_evolution_metrics(
        &self,
    ) -> Option<memory::temporal::GlobalEvolutionMetrics> {
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
    #[cfg(feature = "distributed-experimental")]
    pub enable_distributed: bool,
    #[cfg(feature = "distributed-experimental")]
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
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            #[cfg(feature = "embeddings")]
            enable_embeddings: true,
            #[cfg(feature = "distributed-experimental")]
            enable_distributed: false,
            #[cfg(feature = "distributed-experimental")]
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
    File {
        path: String,
    },
    #[cfg(feature = "sql-storage")]
    Sql {
        connection_string: String,
    },
}

#[cfg(test)]
// Test code: panic on unexpected variants is the intended behaviour.
#[allow(clippy::panic, clippy::unwrap_used)]
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
        assert_eq!(
            config1.max_short_term_memories,
            config2.max_short_term_memories
        );
        assert_eq!(
            config1.max_long_term_memories,
            config2.max_long_term_memories
        );
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

    #[cfg(feature = "distributed-experimental")]
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
