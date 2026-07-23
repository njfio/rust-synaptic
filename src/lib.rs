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
#[cfg(feature = "mcp")]
pub mod mcp;
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
    /// Immutable session identifier, cached so `session_id()` stays sync
    /// (the value equals the id passed to `AgentState::new`).
    session_id: Uuid,
    state: std::sync::Arc<tokio::sync::RwLock<memory::state::AgentState>>,
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
    /// Re-entrancy guard for the intelligent write path: set while storing an
    /// extracted fact-memory so the nested `store_with_report` does not itself
    /// re-run extraction (which would recurse forever).
    #[cfg(feature = "embeddings")]
    storing_facts: bool,
    /// Whether write-time fact distillation is active for this instance
    /// (resolved once at construction time from `MemoryConfig::distillation` /
    /// the deprecated `store_extracted_facts` alias, the active reasoner kind,
    /// and prerequisite availability).
    #[cfg(feature = "embeddings")]
    distillation_live: bool,
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
    /// The embedding provider shared (same `Arc`) by the dense retriever and
    /// the reranker inside `retrieval_pipeline`, selected via
    /// `MemoryConfig::retrieval_embedding_provider` (default: TF-IDF;
    /// optional semantic providers such as Ollama `nomic-embed-text` with a
    /// TF-IDF fallback). Retained here so the store path can feed each stored
    /// memory's content into it via the document (`embed`) path: for TF-IDF
    /// this keeps the corpus IDF statistics live (without the feed the
    /// vocabulary stays empty, `idf()` returns its 1.0 fallback, and dense
    /// retrieval degenerates to hashed-TF cosine); for semantic providers the
    /// same feed warms the TF-IDF fallback corpus inside the wrapper.
    #[cfg(feature = "embeddings")]
    retrieval_scoring_provider: Option<Arc<dyn memory::embeddings::provider::EmbeddingProvider>>,
    /// The concrete TF-IDF instance backing `retrieval_scoring_provider`
    /// (either as the provider itself or as the semantic provider's
    /// fallback). Test-only corpus-statistics introspection.
    #[cfg(all(feature = "embeddings", feature = "test-utils"))]
    retrieval_tfidf_stats: Option<Arc<memory::embeddings::TfIdfProvider>>,
    /// Test-only op-count proxy: max number of candidate memories examined by
    /// a single `neighbor_facts` call. Bounded write path keeps this <= the
    /// candidate cap regardless of store size.
    #[cfg(all(feature = "embeddings", feature = "test-utils"))]
    neighbor_examined_max: std::sync::atomic::AtomicUsize,
    /// Entries awaiting enrichment when `MemoryConfig::enrichment_mode` is
    /// `Deferred` or `Background` (see `memory::enrichment`).
    #[cfg(feature = "embeddings")]
    enrichment_queue: std::sync::Arc<
        tokio::sync::Mutex<std::collections::VecDeque<memory::enrichment::PendingEnrichment>>,
    >,
}

impl AgentMemory {
    /// Metadata field name used to tag a memory as a raw source that
    /// write-time distillation has summarized into fact-memories. Consulted
    /// by retrieval when `MemoryConfig::retrieval_excludes_raw_sources` is set.
    #[cfg(feature = "embeddings")]
    const RAW_SOURCE_FIELD: &'static str = "raw_source";

    #[cfg(all(feature = "embeddings", test))]
    fn set_reasoner_for_test(&mut self, r: Arc<dyn memory::reasoning::MemoryReasoner>) {
        self.reasoner = r;
    }

    #[cfg(all(feature = "embeddings", test))]
    async fn get_entry_for_test(&self, key: &str) -> Option<MemoryEntry> {
        self.storage.retrieve(key).await.ok().flatten()
    }

    #[cfg(all(feature = "embeddings", test))]
    async fn enrichment_queue_len(&self) -> usize {
        self.enrichment_queue.lock().await.len()
    }

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

        let session_id = config.session_id.unwrap_or_else(Uuid::new_v4);
        let state = memory::state::AgentState::new(session_id);
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
        // With the `llm-reasoning` feature enabled AND an endpoint configured
        // via SYNAPTIC_LLM_URL/SYNAPTIC_LLM_MODEL (or the SYNAPTIC_EVAL_LLM_*
        // fallbacks), an LlmReasoner is used instead; it fails open to its own
        // inner HeuristicReasoner on any LLM error, so the write path never
        // hard-fails or fabricates. Feature off or unconfigured: heuristic.
        #[cfg(feature = "embeddings")]
        let (reasoner, resolved_reasoner_kind): (
            Arc<dyn memory::reasoning::MemoryReasoner>,
            memory::reasoning::ResolvedReasonerKind,
        ) = {
            let heuristic: Arc<dyn memory::reasoning::MemoryReasoner> =
                Arc::new(memory::reasoning::HeuristicReasoner::new(Arc::new(
                    memory::embeddings::TfIdfProvider::default(),
                )));
            #[cfg(feature = "llm-reasoning")]
            {
                match memory::reasoning::LlmReasoner::from_env() {
                    Some(llm) => (
                        Arc::new(llm) as Arc<dyn memory::reasoning::MemoryReasoner>,
                        memory::reasoning::ResolvedReasonerKind::Llm,
                    ),
                    None => (
                        heuristic,
                        memory::reasoning::ResolvedReasonerKind::Heuristic,
                    ),
                }
            }
            #[cfg(not(feature = "llm-reasoning"))]
            {
                (
                    heuristic,
                    memory::reasoning::ResolvedReasonerKind::Heuristic,
                )
            }
        };

        // Resolve write-time distillation activation from config (and the
        // deprecated `store_extracted_facts` alias, which forces `On`).
        #[cfg(feature = "embeddings")]
        let distillation_live = {
            let mode = if config.store_extracted_facts {
                memory::reasoning::DistillationMode::On
            } else {
                config.distillation
            };
            let prerequisites_met = config.enable_knowledge_graph;
            match memory::reasoning::resolve_distillation(
                mode,
                resolved_reasoner_kind,
                prerequisites_met,
            ) {
                Ok(decision) => {
                    if decision.live
                        && resolved_reasoner_kind
                            == memory::reasoning::ResolvedReasonerKind::Heuristic
                    {
                        tracing::warn!(
                            "write-time distillation is live with the heuristic reasoner; \
                             the measured LoCoMo lift requires an LLM endpoint \
                             (SYNAPTIC_LLM_URL / SYNAPTIC_LLM_MODEL)"
                        );
                    }
                    decision.live
                }
                Err(msg) => {
                    return Err(crate::error::MemoryError::Configuration { message: msg });
                }
            }
        };

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
        #[cfg(all(feature = "embeddings", feature = "test-utils"))]
        let mut retrieval_tfidf_stats: Option<Arc<memory::embeddings::TfIdfProvider>> = None;
        let (retrieval_pipeline, retrieval_scoring_provider) = if config.enable_embeddings {
            // ONE shared provider instance for the dense retriever AND the
            // reranker: its content-hash scoring cache means each candidate
            // content is embedded at most once per query across both stages.
            // The dyn Arc is ALSO retained on AgentMemory so the store path
            // can feed corpus content into it (real IDF weighting for the
            // TF-IDF default; fallback-corpus warming for semantic
            // providers). The provider is selected via
            // `config.retrieval_embedding_provider` (TF-IDF default; Ollama
            // or a custom semantic provider opt-in, with a TF-IDF fallback
            // that never lets the pipeline hard-fail).
            let (provider, tfidf_stats) =
                memory::embeddings::build_retrieval_provider(&config.retrieval_embedding_provider);
            #[cfg(feature = "test-utils")]
            {
                retrieval_tfidf_stats = Some(tfidf_stats);
            }
            #[cfg(not(feature = "test-utils"))]
            let _ = tfidf_stats;
            let dense_vector: Arc<dyn memory::retrieval::RetrievalPipeline> =
                Arc::new(memory::retrieval::DenseVectorRetriever::new(
                    Arc::clone(&storage),
                    Arc::clone(&provider),
                ));
            let keyword: Arc<dyn memory::retrieval::RetrievalPipeline> = Arc::new(
                memory::retrieval::KeywordRetriever::new(Arc::clone(&storage)),
            );
            // Graph and temporal signals share the same storage; the graph
            // retriever additionally shares the live knowledge-graph handle
            // (it reports itself unavailable when the graph is disabled).
            let graph = memory::retrieval::GraphRetriever::new(
                Arc::clone(&storage),
                knowledge_graph.clone(),
            );
            let temporal = memory::retrieval::TemporalRetriever::new(Arc::clone(&storage));
            let mut pipeline_config = memory::retrieval::PipelineConfig::semantic_focus();
            // Query understanding (temporal-constraint extraction +
            // multi-part splitting) is toggleable for ablation baselines.
            pipeline_config.enable_query_understanding = config.enable_query_understanding;
            // Deterministic heuristic reranker over the top-K: cross-features
            // (term overlap, embedding agreement, graph proximity, recency)
            // reorder the fused + composite-scored results.
            // Reranker selection: `SYNAPTIC_RERANKER=cross-encoder` opts into
            // the candle BERT cross-encoder (feature `reranker-model`);
            // anything else (including unset) keeps today's default
            // heuristic reranker. The cross-encoder attempt fails closed —
            // a missing/unloadable model or a binary built without the
            // feature falls back to the heuristic reranker with a warning,
            // so the pipeline is never left without a reranker.
            let heuristic_reranker: Arc<dyn memory::retrieval::Reranker> = Arc::new(
                memory::retrieval::HeuristicReranker::new(
                    Some(Arc::clone(&provider)),
                    knowledge_graph.clone(),
                )
                .with_weights(memory::retrieval::HeuristicRerankWeights::from_env()),
            );
            let reranker: Arc<dyn memory::retrieval::Reranker> =
                memory::retrieval::select_cross_encoder_reranker().unwrap_or(heuristic_reranker);
            let mut hybrid = memory::retrieval::HybridRetriever::new(pipeline_config)
                .add_pipeline(Arc::clone(&dense_vector))
                .add_pipeline(Arc::clone(&keyword))
                .add_pipeline(Arc::new(graph))
                .add_pipeline(Arc::new(temporal))
                .with_reranker(reranker);
            // Multi-hop graph expansion (HippoRAG-style): seeds with the
            // dense + keyword retrievers' top hits and expands along KG
            // relation edges so 2-hop-reachable answers surface. Toggleable
            // via `enable_multihop_retrieval` for single-hop ablations;
            // inert without a knowledge graph.
            if config.enable_multihop_retrieval && knowledge_graph.is_some() {
                let multihop = memory::retrieval::MultiHopGraphRetriever::new(
                    Arc::clone(&storage),
                    knowledge_graph.clone(),
                    vec![Arc::clone(&dense_vector), Arc::clone(&keyword)],
                    memory::retrieval::MultiHopConfig::default(),
                );
                hybrid = hybrid.add_pipeline(Arc::new(multihop));
            }
            (Some(Arc::new(hybrid)), Some(provider))
        } else {
            (None, None)
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
            session_id,
            state: std::sync::Arc::new(tokio::sync::RwLock::new(state)),
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
            #[cfg(feature = "embeddings")]
            storing_facts: false,
            #[cfg(feature = "embeddings")]
            distillation_live,
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
            #[cfg(feature = "embeddings")]
            retrieval_scoring_provider,
            #[cfg(all(feature = "embeddings", feature = "test-utils"))]
            retrieval_tfidf_stats,
            #[cfg(all(feature = "embeddings", feature = "test-utils"))]
            neighbor_examined_max: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "embeddings")]
            enrichment_queue: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::VecDeque::new(),
            )),
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

    /// Vocabulary size of the retrieval scoring provider (test-utils).
    /// `None` when embeddings/retrieval are disabled. Non-zero after stores
    /// proves the corpus is being fed into the dense-scoring IDF statistics.
    #[cfg(all(feature = "embeddings", feature = "test-utils"))]
    pub fn retrieval_scoring_vocabulary_size(&self) -> Option<usize> {
        self.retrieval_tfidf_stats
            .as_ref()
            .map(|p| p.vocabulary_size())
    }

    /// Shared handle to the knowledge graph, if enabled. Used by the MCP
    /// `recall` tool to apply bi-temporal `as_of` filtering via the graph's
    /// node validity (`Node::is_valid_at`).
    #[cfg(feature = "mcp")]
    pub(crate) fn knowledge_graph_handle(
        &self,
    ) -> Option<Arc<tokio::sync::RwLock<memory::knowledge_graph::MemoryKnowledgeGraph>>> {
        self.knowledge_graph.clone()
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
        let is_update = self.state.read().await.has_memory(key);
        let change_type = if is_update {
            memory::temporal::ChangeType::Updated
        } else {
            memory::temporal::ChangeType::Created
        };

        tracing::debug!("Memory operation type: {:?}", change_type);

        self.storage.store(&entry).await?;
        self.state.write().await.add_memory(entry.clone());
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

        // Add or update in knowledge graph if enabled (intelligent merging).
        // The O(n) similarity/relationship computation runs under READ locks;
        // the write lock is held only for the actual node/edge mutations so
        // concurrent readers are not blocked behind full-graph scans.
        if let Some(ref kg) = self.knowledge_graph {
            let kg_result: Result<()> = async {
                let plan = kg.read().await.plan_memory_node_update(&entry).await?;
                let (node_id, needs_detection) = kg
                    .write()
                    .await
                    .apply_memory_node_plan(plan, &entry)
                    .await?;
                if needs_detection {
                    let edges = kg
                        .read()
                        .await
                        .detect_relationship_edges(node_id, &entry)
                        .await?;
                    if !edges.is_empty() {
                        kg.write().await.add_detected_edges(edges).await?;
                    }
                }
                Ok(())
            }
            .await;
            if let Err(e) = kg_result {
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

        // Feed the stored content into the retrieval scoring provider via the
        // document (`embed`) path so the corpus IDF statistics the dense
        // retriever and reranker score against stay live. O(tokens) per
        // store; the provider's scoring cache self-invalidates on this
        // ingest, so query-time vectors never reflect stale IDF. Best-effort
        // like the other subsystems: a failure degrades ranking quality only,
        // never the write.
        #[cfg(feature = "embeddings")]
        if let Some(ref provider) = self.retrieval_scoring_provider {
            if let Err(e) = provider.embed(&entry.value, None).await {
                tracing::warn!(error = %e, "retrieval IDF corpus ingest degraded during store");
            }
        }

        // Intelligent write path: extract facts from the new value, resolve
        // each against similar existing memories, and apply the outcome to
        // the knowledge graph. Best-effort like the other subsystems. Skipped
        // while storing an already-extracted fact-memory (`storing_facts`) so
        // extraction does not recurse.
        #[cfg(feature = "embeddings")]
        if self.knowledge_graph.is_some() && !self.storing_facts {
            match self._config.enrichment_mode {
                memory::enrichment::EnrichmentMode::Inline => {
                    if let Err(e) = self.reason_over_store(&entry).await {
                        tracing::warn!(error = %e, "reasoning degraded during store");
                        degradations.reasoning = Some(e.to_string());
                    }
                }
                memory::enrichment::EnrichmentMode::Deferred
                | memory::enrichment::EnrichmentMode::Background => {
                    // Never enqueue a fact-memory (re-entrancy guard already
                    // ensures fact stores set storing_facts, so we never reach
                    // here for them; the key check is belt-and-braces).
                    if !key.contains("::fact") {
                        self.enrichment_queue.lock().await.push_back(
                            memory::enrichment::PendingEnrichment {
                                key: key.to_string(),
                                value: value.to_string(),
                            },
                        );
                        // Background notification wired in Task 5.
                    }
                }
            }
        }

        // Check if we need to create a checkpoint
        let state_guard = self.state.read().await;
        if self.checkpoint_manager.should_checkpoint(&state_guard) {
            self.checkpoint_manager
                .create_checkpoint(&state_guard)
                .await?;
        }
        drop(state_guard);

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
        let mut stored_any_fact = false;
        for (i, fact) in extraction.facts.iter().enumerate() {
            let neighbors = self.neighbor_facts(&entry.key, fact, 5).await?;
            let resolution = reasoner.resolve(fact, &neighbors).await?;
            // The reasoner picks its target among the neighbors we supplied;
            // for id-less outcomes (UpdateInPlace/Append) the target is the
            // best neighbor by the same deterministic ordering.
            let best_neighbor_id = neighbors.first().map(|n| n.id.clone());
            let superseded = self
                .apply_resolution(entry, fact, resolution, best_neighbor_id)
                .await?;
            // Mark the superseded memory itself (not just its KG edges) so
            // bi-temporal validity is visible to retrieval (see `mark_superseded`
            // and `MemoryConfig::retrieval_excludes_superseded`).
            if let Some(old_key) = superseded {
                self.mark_superseded(&old_key).await?;
            }

            // Persist the extracted fact as its own retrievable memory so
            // search surfaces distilled facts, not just raw turns.
            if self.distillation_live && !fact.text.trim().is_empty() {
                let fact_key = format!("{}::fact{}", entry.key, i);
                self.storing_facts = true;
                // Box::pin breaks the recursive-future size cycle
                // (store_with_report → reason_over_store → store_with_report).
                let result = Box::pin(self.store_with_report(&fact_key, &fact.text)).await;
                self.storing_facts = false;
                result?;
                stored_any_fact = true;
            }
        }
        // Failure-ordering invariant: tag the raw turn as a summarized source
        // ONLY after ≥1 fact has been stored. If extraction or fact storage
        // failed above (propagated via `?`), we never reach here, so the raw
        // turn stays untagged and searchable — a distillation outage degrades
        // to raw-turn retrieval rather than hiding a turn with no replacement.
        if stored_any_fact {
            self.mark_raw_source(&entry.key).await?;
        }
        Ok(())
    }

    /// Tag a raw source turn as summarized-by-distillation (facts-primary
    /// retrieval). Mirrors [`Self::mark_superseded`]: records `raw_source` in the
    /// entry's custom fields and persists it, without re-running reasoning.
    #[cfg(feature = "embeddings")]
    async fn mark_raw_source(&mut self, key: &str) -> Result<()> {
        if let Some(mut entry) = self.storage.retrieve(key).await? {
            entry
                .metadata
                .custom_fields
                .insert(Self::RAW_SOURCE_FIELD.to_string(), Utc::now().to_rfc3339());
            self.storage.store(&entry).await?;
            if self.state.read().await.has_memory(key) {
                self.state.write().await.add_memory(entry);
            }
        }
        Ok(())
    }

    /// Mark a memory as superseded (bi-temporal invalidation at the memory
    /// level): records `superseded_at` in the entry's custom fields and persists
    /// it, without re-running the write-path reasoning. With
    /// [`MemoryConfig::retrieval_excludes_superseded`] set, such memories are
    /// dropped from `search` results so retrieval surfaces the *current* fact.
    #[cfg(feature = "embeddings")]
    async fn mark_superseded(&mut self, key: &str) -> Result<()> {
        if let Some(mut entry) = self.storage.retrieve(key).await? {
            entry
                .metadata
                .custom_fields
                .insert(Self::SUPERSEDED_FIELD.to_string(), Utc::now().to_rfc3339());
            self.storage.store(&entry).await?;
            // Keep in-memory state consistent (best-effort; state is the search
            // corpus for the non-pipeline path).
            if self.state.read().await.has_memory(key) {
                self.state.write().await.add_memory(entry);
            }
        }
        Ok(())
    }

    /// Build the top-`k` neighbor list for a candidate `fact`, excluding the
    /// entry being stored.
    ///
    /// Candidate source (changed from the previous full state scan): the bulk
    /// of candidates come from the storage's tokenized search over the fact
    /// text (top [`Self::NEIGHBOR_CANDIDATE_LIMIT`] by term overlap), so the
    /// per-fact cost is bounded (O(K)) instead of tokenizing every stored
    /// short/long-term memory (O(n)).
    ///
    /// Bounded-recall tradeoff and its mitigation: a pure top-K keyword search
    /// can miss a genuinely contradicting memory when the two share only their
    /// subject (e.g. "Alice lives in Berlin" vs. "Alice lives in Munich" share
    /// just "alice"/"lives"/"in") while >=K unrelated memories share more
    /// terms — which would wrongly downgrade a Supersede to an Insert. To make
    /// same-subject contradictions robust we (a) drop trivial (<3 char) tokens
    /// from the search query so distinctive terms dominate ranking, and (b)
    /// UNION the keyword hits with an exact per-subject lookup: for every
    /// relation subject in the fact we run a targeted search so a memory about
    /// the same subject is always in the candidate set. The final similarity
    /// ranking over the union is unchanged (deterministic token-set cosine,
    /// ties broken by key).
    #[cfg(feature = "embeddings")]
    async fn neighbor_facts(
        &self,
        current_key: &str,
        fact: &memory::reasoning::Fact,
        k: usize,
    ) -> Result<Vec<memory::reasoning::NeighborFact>> {
        fn tokens(text: &str) -> std::collections::HashSet<String> {
            text.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect()
        }
        let candidate_tokens = tokens(&fact.text);
        if candidate_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // (a) Build a keyword query from distinctive (>=3 char) tokens so a
        // handful of long, discriminating terms outrank many short stopword
        // matches. Fall back to the raw text if nothing survives the filter.
        let distinctive: Vec<&str> = fact
            .text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 3)
            .collect();
        let query = if distinctive.is_empty() {
            fact.text.clone()
        } else {
            distinctive.join(" ")
        };

        // Collect a bounded, deduplicated candidate set: the keyword hits ...
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut candidates: Vec<memory::types::MemoryFragment> = Vec::new();
        // +1 head-room: the search may return the entry being stored itself.
        for fragment in self
            .storage
            .search(&query, Self::NEIGHBOR_CANDIDATE_LIMIT + 1)
            .await?
        {
            if seen.insert(fragment.entry.key.clone()) {
                candidates.push(fragment);
            }
        }
        // (b) ... unioned with a targeted lookup per relation subject so a
        // same-subject memory can never be squeezed out of the candidate set.
        for relation in &fact.relations {
            if relation.subject.trim().is_empty() {
                continue;
            }
            for fragment in self
                .storage
                .search(&relation.subject, Self::NEIGHBOR_CANDIDATE_LIMIT + 1)
                .await?
            {
                if seen.insert(fragment.entry.key.clone()) {
                    candidates.push(fragment);
                }
            }
        }

        #[cfg(feature = "test-utils")]
        let examined = std::sync::atomic::AtomicUsize::new(0);
        let mut neighbors: Vec<memory::reasoning::NeighborFact> = candidates
            .iter()
            .filter(|fragment| fragment.entry.key != current_key)
            .filter_map(|fragment| {
                #[cfg(feature = "test-utils")]
                examined.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mem_tokens = tokens(&fragment.entry.value);
                if mem_tokens.is_empty() {
                    return None;
                }
                let intersection = candidate_tokens.intersection(&mem_tokens).count() as f64;
                let similarity = intersection
                    / ((candidate_tokens.len() as f64).sqrt() * (mem_tokens.len() as f64).sqrt());
                if similarity > 0.0 {
                    Some(memory::reasoning::NeighborFact {
                        id: fragment.entry.key.clone(),
                        similarity,
                        text: fragment.entry.value.clone(),
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
        #[cfg(feature = "test-utils")]
        self.neighbor_examined_max.fetch_max(
            examined.load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(neighbors)
    }

    /// Upper bound on the keyword-hit candidates `neighbor_facts` pulls per
    /// search. Keeps the intelligent write path O(K) per fact instead of O(n)
    /// over the whole store; the storage's tokenized search ranks candidates
    /// by term overlap so the strongest conflict candidates are retained. The
    /// per-subject union searches add at most a small constant multiple.
    #[cfg(feature = "embeddings")]
    const NEIGHBOR_CANDIDATE_LIMIT: usize = 20;

    /// Custom-field key recording when a memory was bi-temporally superseded.
    #[cfg(feature = "embeddings")]
    const SUPERSEDED_FIELD: &'static str = "superseded_at";

    /// Test-only op-count proxy: the maximum number of candidate memories a
    /// single `neighbor_facts` call has examined so far.
    #[cfg(all(feature = "embeddings", feature = "test-utils"))]
    pub fn max_neighbor_candidates_examined(&self) -> usize {
        self.neighbor_examined_max
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Apply a conflict resolution outcome to the knowledge graph.
    #[cfg(feature = "embeddings")]
    async fn apply_resolution(
        &mut self,
        entry: &MemoryEntry,
        fact: &memory::reasoning::Fact,
        resolution: memory::reasoning::ConflictResolution,
        best_neighbor_id: Option<String>,
    ) -> Result<Option<String>> {
        use memory::reasoning::ConflictResolution;
        let Some(ref kg) = self.knowledge_graph else {
            return Ok(None);
        };
        let mut superseded_key: Option<String> = None;
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
                // `old_id` is the superseded memory's key (see `neighbor_facts`):
                // report it so the write path can mark the memory itself
                // superseded, making bi-temporal validity visible to retrieval.
                superseded_key = Some(old_id);
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
        Ok(superseded_key)
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
            // reflection simply skips it. Bind by value first so the write
            // guard drops before the body (avoids holding it across the loop
            // body / any future state access).
            let maybe_entry = self.state.write().await.get_memory(key);
            if let Some(entry) = maybe_entry {
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
        self.state.write().await.add_memory(entry.clone());

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
    pub async fn insight_memories(&self) -> Vec<MemoryEntry> {
        let state_guard = self.state.read().await;
        let mut insights: Vec<MemoryEntry> = state_guard
            .get_short_term_memories()
            .values()
            .chain(state_guard.get_long_term_memories().values())
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

        // First check short-term memory. Bind by value BEFORE the `if let`:
        // an `if let ... = scrutinee` extends the scrutinee temporary (the
        // write guard) across the ENTIRE body, and the body reacquires
        // `self.state.write()` on the promotion path — holding both would
        // deadlock (tokio RwLock is not reentrant).
        let maybe_entry = self.state.write().await.get_memory(key);
        if let Some(mut entry) = maybe_entry {
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
                    self.state.write().await.add_memory(entry.clone());
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
            self.state.write().await.add_memory(entry.clone());

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

    /// Apply a forgetting pass: demote or evict memories whose retained
    /// strength (`decay(age) * importance * recency_of_access_factor`, see
    /// [`memory::forgetting::ForgettingPolicy::retained_strength`]) has fallen
    /// below the policy's retention floor.
    ///
    /// Tier transitions route through the same [`memory::promotion::MemoryPromotionManager`]
    /// that drives promotion: a long-term memory below the floor is demoted to
    /// short-term (and survives until a later pass); a short-term memory below
    /// the floor is evicted from both storage and session state. Requires
    /// memory promotion to be enabled (`MemoryConfig::enable_memory_promotion`,
    /// the default) so eviction is not a parallel delete path.
    #[tracing::instrument(skip(self, policy), fields(retention_floor = policy.retention_floor))]
    pub async fn forget(
        &mut self,
        policy: memory::forgetting::ForgettingPolicy,
    ) -> Result<memory::forgetting::ForgetReport> {
        let promotion_manager = self.promotion_manager.as_ref().ok_or_else(|| {
            MemoryError::configuration(
                "forget() requires memory promotion to be enabled so tier transitions \
                 (demotion/eviction) route through the promotion machinery",
            )
        })?;

        let mut entries = self.storage.get_all_entries().await?;
        // Deterministic processing order regardless of backend iteration order.
        entries.sort_by(|a, b| a.key.cmp(&b.key));

        let mut report = memory::forgetting::ForgetReport {
            examined: entries.len(),
            ..Default::default()
        };

        for stored in entries {
            // Prefer the in-session copy: state carries fresher access
            // metadata (retrievals bump the state entry, not storage).
            let entry = self
                .state
                .read()
                .await
                .peek_memory(&stored.key)
                .cloned()
                .unwrap_or(stored);

            let strength = policy.retained_strength(&entry).await?;
            if strength >= policy.retention_floor {
                continue;
            }

            match entry.memory_type {
                memory::types::MemoryType::LongTerm => {
                    let demoted = promotion_manager.demote_memory(entry)?;
                    tracing::info!(
                        memory_key = %demoted.key,
                        retained_strength = strength,
                        "Forgetting pass demoting memory to short-term tier"
                    );
                    self.storage.store(&demoted).await?;
                    if self.state.read().await.peek_memory(&demoted.key).is_some() {
                        // Preserve real access metadata: relocating between
                        // tiers is not an access, so do not bump
                        // last_accessed/access_count (which would grant
                        // artificial recency protection on later passes).
                        self.state.write().await.insert_preserving_access(demoted.clone());
                    }
                    report.demoted.push(demoted.key);
                }
                memory::types::MemoryType::ShortTerm => {
                    tracing::info!(
                        memory_key = %entry.key,
                        retained_strength = strength,
                        "Forgetting pass evicting memory at lowest tier"
                    );
                    self.state.write().await.remove_memory(&entry.key);
                    self.storage.delete(&entry.key).await?;
                    report.evicted.push(entry.key);
                }
            }
        }

        tracing::debug!(
            examined = report.examined,
            evicted = report.evicted.len(),
            demoted = report.demoted.len(),
            "Forgetting pass complete"
        );
        Ok(report)
    }

    /// Search memories by content similarity
    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), limit = limit))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        use crate::error_handling::utils::{validate_non_empty_string, validate_range};

        tracing::debug!("Searching memories by content similarity");

        // Validate inputs
        validate_non_empty_string(query, "search query")?;
        validate_range(limit, 1, 10000, "search limit")?; // Max 10k results

        let mut results = if let Some(ref pipeline) = self.retrieval_pipeline {
            pipeline.search(query, limit).await?
        } else {
            self.storage.search(query, limit).await?
        };
        // Bi-temporal validity: when enabled, drop memories that have been
        // superseded so retrieval surfaces the *current* fact after an update.
        #[cfg(feature = "embeddings")]
        if self._config.retrieval_excludes_superseded {
            results.retain(|f| {
                !f.entry
                    .metadata
                    .custom_fields
                    .contains_key(Self::SUPERSEDED_FIELD)
            });
        }
        // Facts-primary retrieval: when enabled, drop raw source turns that
        // distillation has summarized, so search surfaces distilled facts.
        #[cfg(feature = "embeddings")]
        if self._config.retrieval_excludes_raw_sources {
            results.retain(|f| {
                !f.entry
                    .metadata
                    .custom_fields
                    .contains_key(Self::RAW_SOURCE_FIELD)
            });
        }
        tracing::debug!("Search completed, found {} results", results.len());
        Ok(results)
    }

    /// Create a checkpoint of the current state
    pub async fn checkpoint(&self) -> Result<Uuid> {
        let state_guard = self.state.read().await;
        self.checkpoint_manager.create_checkpoint(&state_guard).await
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
        *self.state.write().await = restored_state;

        Ok(())
    }

    /// Get current memory statistics
    /// Get the session ID for this agent memory instance
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub async fn stats(&self) -> MemoryStats {
        let state_guard = self.state.read().await;
        MemoryStats {
            short_term_count: state_guard.short_term_memory_count(),
            long_term_count: state_guard.long_term_memory_count(),
            total_size: state_guard.total_memory_size(),
            session_id: state_guard.session_id(),
            created_at: state_guard.created_at(),
        }
    }

    /// Clear all memories (use with caution)
    pub async fn clear(&mut self) -> Result<()> {
        self.state.write().await.clear();
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
    pub async fn has_memory(&self, key: &str) -> bool {
        self.state.read().await.has_memory(key)
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
    /// Enable the multi-hop graph-expansion retrieval stage (HippoRAG-style
    /// seed → expand → score over the knowledge graph). Default: `true`.
    /// Requires `enable_knowledge_graph` (it is inert without a graph); set
    /// to `false` for single-hop ablation baselines.
    pub enable_multihop_retrieval: bool,
    /// Enable the deterministic query-understanding stage in the retrieval
    /// pipeline (temporal-constraint extraction + multi-part question
    /// splitting with result union). Default: `true`. Simple queries are a
    /// no-op passthrough; set to `false` for ablation baselines.
    pub enable_query_understanding: bool,
    pub enable_temporal_tracking: bool,
    pub enable_advanced_management: bool,
    #[cfg(feature = "embeddings")]
    pub enable_embeddings: bool,
    /// Embedding provider for the retrieval pipeline (dense retriever,
    /// reranker, store-time corpus feed). Default selection (see
    /// [`memory::embeddings::RetrievalEmbeddingConfig::auto`]): an explicit
    /// `SYNAPTIC_RETRIEVAL_EMBEDDER` env selection (`tfidf`/`ollama`/`candle`)
    /// wins; else, with the `ml-models` feature built AND the bundled MiniLM
    /// model fetched locally (`scripts/fetch_embedding_model.sh`, default dir
    /// `models/all-MiniLM-L6-v2`, override via `SYNAPTIC_EMBED_MODEL_DIR`),
    /// the offline candle model is used; else TF-IDF (offline, no network,
    /// no model download). All semantic providers fall back to TF-IDF with a
    /// warn on any failure.
    #[cfg(feature = "embeddings")]
    pub retrieval_embedding_provider: memory::embeddings::RetrievalEmbeddingConfig,
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
    /// DEPRECATED — use [`MemoryConfig::distillation`]. When `true`, forces
    /// `DistillationMode::On` (facts stored as `<key>::fact<N>` memories). Kept
    /// for backwards compatibility; slated for removal in a future major.
    /// Default `false`. When this alias is live, distilled facts are added
    /// alongside raw turns (augment mode), since `retrieval_excludes_raw_sources`
    /// defaults to `false`.
    pub store_extracted_facts: bool,
    /// Bi-temporal validity in retrieval: when `true`, `search` drops memories
    /// that the intelligent write path has marked superseded (a later fact
    /// contradicted them), so retrieval returns the *current* fact after an
    /// update instead of the stale one. Default: `false` (superseded memories
    /// remain retrievable, matching prior behavior).
    pub retrieval_excludes_superseded: bool,
    /// Write-time fact distillation activation (see
    /// [`memory::reasoning::DistillationMode`]). Default `Off` (opt-in): a
    /// matched LoCoMo A/B measured facts-primary distillation to hurt QA (see
    /// `docs/evaluation.md`), so the default write path never distills. Set to
    /// `On` to enable; quality scales with the active reasoner (an `LlmReasoner`
    /// under `llm-reasoning`, else the heuristic).
    pub distillation: memory::reasoning::DistillationMode,
    /// Facts-primary retrieval: when `true`, `search` drops memories tagged as
    /// raw sources (the original turns distillation summarized), so retrieval
    /// surfaces distilled facts *instead of* raw turns. Default `false`
    /// (**augment** mode: distilled facts are added alongside raw turns rather
    /// than replacing them). The A/B showed that *excluding* raw turns is what
    /// hurt QA — it removed the verbatim fallback (dates, exact details) — while
    /// augment recovered baseline accuracy (0.505 → 0.258 facts-primary vs 0.500
    /// augment). Only has an effect when distillation is live.
    pub retrieval_excludes_raw_sources: bool,
    /// How write-path LLM enrichment is scheduled (see
    /// [`memory::enrichment::EnrichmentMode`]). Default `Inline` (enrich
    /// synchronously inside `store`), byte-for-byte today's behavior.
    pub enrichment_mode: memory::enrichment::EnrichmentMode,
    /// Bounded concurrency for `enrich_pending`/background enrichment. Default 8.
    pub enrichment_concurrency: usize,
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
            enable_multihop_retrieval: true,
            enable_query_understanding: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            #[cfg(feature = "embeddings")]
            enable_embeddings: true,
            #[cfg(feature = "embeddings")]
            retrieval_embedding_provider: memory::embeddings::RetrievalEmbeddingConfig::auto(),
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
            store_extracted_facts: false,
            retrieval_excludes_superseded: false,
            // Default OFF (opt-in), not Auto: a full-LoCoMo-subset A/B measured
            // facts-primary distillation to HURT QA (raw baseline 0.505 ->
            // distill 0.258 on 198 matched questions with a local 7B extractor;
            // abstention doubled). Lossy facts drop answer detail and excluding
            // raw turns removes the verbatim fallback (incl. the session-date
            // prefix temporal questions need). A frontier extractor helps but is
            // impractically slow for the write path. See docs/evaluation.md
            // "Write-time distillation A/B". Enable explicitly to opt in.
            distillation: memory::reasoning::DistillationMode::Off,
            // Augment, not facts-primary: distillation (when opted in) adds
            // facts alongside raw turns rather than excluding raw from search.
            // The A/B showed exclusion is what hurt QA; augment ties baseline.
            retrieval_excludes_raw_sources: false,
            enrichment_mode: memory::enrichment::EnrichmentMode::Inline,
            enrichment_concurrency: 8,
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

    // A stub reasoner that returns a fixed set of facts (or an error) so the write
    // path is testable without an LLM.
    #[cfg(feature = "embeddings")]
    #[derive(Clone)]
    struct StubReasoner {
        facts: Vec<String>,
        fail: bool,
    }

    #[cfg(feature = "embeddings")]
    #[async_trait::async_trait]
    impl memory::reasoning::MemoryReasoner for StubReasoner {
        async fn extract(
            &self,
            _text: &str,
            _ctx: &memory::reasoning::ExtractionContext,
        ) -> Result<memory::reasoning::Extraction> {
            if self.fail {
                return Err(crate::error::MemoryError::ProcessingError(
                    "stub extract failure".to_string(),
                ));
            }
            Ok(memory::reasoning::Extraction {
                facts: self
                    .facts
                    .iter()
                    .map(|t| memory::reasoning::Fact {
                        text: t.clone(),
                        entities: Vec::new(),
                        relations: Vec::new(),
                    })
                    .collect(),
            })
        }
        async fn resolve(
            &self,
            _candidate: &memory::reasoning::Fact,
            _neighbors: &[memory::reasoning::NeighborFact],
        ) -> Result<memory::reasoning::ConflictResolution> {
            Ok(memory::reasoning::ConflictResolution::Insert)
        }
        async fn synthesize(
            &self,
            _cluster: &[MemoryEntry],
        ) -> Result<Option<memory::reasoning::Insight>> {
            Ok(None)
        }
        fn name(&self) -> &str {
            "StubReasoner"
        }
    }

    #[tokio::test]
    async fn enrichment_defaults_to_inline() {
        let config = MemoryConfig::default();
        assert_eq!(
            config.enrichment_mode,
            memory::enrichment::EnrichmentMode::Inline
        );
        assert_eq!(config.enrichment_concurrency, 8);
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn deferred_mode_queues_without_inline_enrich() {
        let config = MemoryConfig {
            enrichment_mode: memory::enrichment::EnrichmentMode::Deferred,
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin").await.unwrap();
        // Deferred: no fact-memory yet (enrichment not run), raw is present.
        assert!(mem.get_entry_for_test("turn1").await.is_some());
        assert!(mem.get_entry_for_test("turn1::fact0").await.is_none());
        assert_eq!(mem.enrichment_queue_len().await, 1);
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn distillation_tags_raw_source_after_facts_land() {
        let config = MemoryConfig {
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin last year")
            .await
            .unwrap();

        // Raw turn is tagged; the fact memory exists and is not tagged.
        let raw = mem.get_entry_for_test("turn1").await.unwrap();
        assert!(raw.metadata.custom_fields.contains_key("raw_source"));
        let fact = mem.get_entry_for_test("turn1::fact0").await.unwrap();
        assert!(!fact.metadata.custom_fields.contains_key("raw_source"));
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn distillation_failure_leaves_raw_untagged() {
        let config = MemoryConfig {
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec![],
            fail: true,
        }));
        // Write still succeeds (best-effort); raw turn stays searchable/untagged.
        mem.store("turn1", "Alice: I moved to Berlin")
            .await
            .unwrap();
        let raw = mem.get_entry_for_test("turn1").await.unwrap();
        assert!(!raw.metadata.custom_fields.contains_key("raw_source"));
    }

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

    #[tokio::test]
    async fn store_extracted_facts_off_by_default_keeps_raw_only() {
        let mem = AgentMemory::new(MemoryConfig::default()).await.unwrap();
        assert!(
            !mem._config.store_extracted_facts,
            "intelligent-write fact storage must be opt-in"
        );
        let mut mem = mem;
        mem.store("note", "Alice lives in Berlin. Bob works at Acme.")
            .await
            .unwrap();
        // No `::fact` memories are created when the flag is off.
        assert!(
            mem.retrieve("note::fact0").await.unwrap().is_none(),
            "no fact-memories should exist with the default (off) config"
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn distillation_default_is_off() {
        // Default is DistillationMode::Off (opt-in) after the A/B measured
        // facts-primary distillation to hurt QA — so the default write path is
        // never live, even if an LLM endpoint happens to be configured.
        let mem = AgentMemory::new(MemoryConfig::default()).await.unwrap();
        assert!(
            !mem.distillation_live,
            "default distillation must be off (opt-in)"
        );
        assert_eq!(
            mem._config.distillation,
            memory::reasoning::DistillationMode::Off
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn store_extracted_facts_alias_forces_on() {
        // Deprecated alias: store_extracted_facts=true behaves as DistillationMode::On.
        // With KG+embeddings present and no llm feature endpoint, On is live (heuristic).
        let config = MemoryConfig {
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mem = AgentMemory::new(config).await.unwrap();
        assert!(
            mem.distillation_live,
            "store_extracted_facts=true must make distillation live"
        );
    }

    #[tokio::test]
    async fn retrieval_excludes_raw_sources_defaults_false_augment() {
        // Default is augment (raw retained), not facts-primary: the A/B showed
        // excluding raw turns hurts QA.
        let config = MemoryConfig::default();
        assert!(!config.retrieval_excludes_raw_sources);
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn superseded_memory_excluded_from_retrieval_when_enabled() {
        // A later contradicting fact supersedes the earlier one; with
        // retrieval_excludes_superseded the stale memory is dropped from search.
        let config = MemoryConfig {
            retrieval_excludes_superseded: true,
            ..MemoryConfig::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.store("r1", "Alice lives in Berlin").await.unwrap();
        mem.store("r2", "Alice lives in Munich").await.unwrap();

        let results = mem.search("Where does Alice live", 10).await.unwrap();
        let values: Vec<String> = results.iter().map(|f| f.entry.value.clone()).collect();
        // The current fact (Munich) is retrievable; the superseded one (Berlin)
        // is filtered out.
        assert!(
            values.iter().any(|v| v.contains("Munich")),
            "current fact should be retrieved, got {values:?}"
        );
        assert!(
            !values.iter().any(|v| v.contains("Berlin")),
            "superseded fact should be excluded, got {values:?}"
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn superseded_memory_retained_by_default() {
        // Same sequence, but with the default (flag off): the superseded memory
        // stays retrievable (no behavior change).
        let mut mem = AgentMemory::new(MemoryConfig::default()).await.unwrap();
        mem.store("r1", "Alice lives in Berlin").await.unwrap();
        mem.store("r2", "Alice lives in Munich").await.unwrap();
        let results = mem.search("Where does Alice live", 10).await.unwrap();
        let values: Vec<String> = results.iter().map(|f| f.entry.value.clone()).collect();
        assert!(
            values.iter().any(|v| v.contains("Berlin")),
            "with the flag off the superseded memory must remain, got {values:?}"
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn store_extracted_facts_makes_facts_retrievable() {
        let config = MemoryConfig {
            store_extracted_facts: true,
            ..MemoryConfig::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.store("note", "Alice lives in Berlin. Bob works at Acme.")
            .await
            .unwrap();
        // The deterministic heuristic reasoner extracts at least one fact, which
        // is stored under a `::fact<N>` key and is retrievable.
        let fact0 = mem.retrieve("note::fact0").await.unwrap();
        assert!(
            fact0.is_some(),
            "an extracted fact-memory should be stored and retrievable"
        );
        assert!(!fact0.unwrap().value.trim().is_empty());
        // Storing a fact-memory must NOT recurse into further extraction.
        assert!(
            mem.retrieve("note::fact0::fact0").await.unwrap().is_none(),
            "fact-memories must not themselves be re-extracted (recursion guard)"
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn search_excludes_raw_sources_when_enabled() {
        // Facts-primary is now opt-in (default is augment), so set the flag
        // explicitly to exercise the exclusion filter.
        let config = MemoryConfig {
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            retrieval_excludes_raw_sources: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin last year")
            .await
            .unwrap();

        let hits = mem.search("Berlin", 10).await.unwrap();
        let keys: Vec<&str> = hits.iter().map(|f| f.entry.key.as_str()).collect();
        assert!(
            keys.iter().any(|k| k.contains("::fact")),
            "fact must be retrievable"
        );
        assert!(
            !keys.contains(&"turn1"),
            "raw source must be excluded by default"
        );
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn search_includes_raw_sources_when_disabled() {
        let config = MemoryConfig {
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            retrieval_excludes_raw_sources: false,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin last year")
            .await
            .unwrap();

        let hits = mem.search("Berlin", 10).await.unwrap();
        let keys: Vec<&str> = hits.iter().map(|f| f.entry.key.as_str()).collect();
        assert!(
            keys.contains(&"turn1"),
            "raw source must reappear when filter disabled"
        );
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

    /// Regression test for the `retrieve()` promotion-on-retrieve deadlock.
    ///
    /// `retrieve()` previously took `self.state.write().await.get_memory(key)`
    /// as an `if let` scrutinee, which keeps the write guard alive across the
    /// whole body. When promotion fired (`should_promote` true), the body then
    /// took a SECOND `self.state.write().await` to re-insert the promoted
    /// entry — `tokio::sync::RwLock` is not reentrant, so this deadlocked.
    ///
    /// This config uses the Importance policy with threshold 0.0 so promotion
    /// fires for any freshly stored short-term entry. On the buggy code this
    /// test HANGS; with the guard bound-and-dropped before the `if let`, it
    /// returns. A wall-clock timeout guards against a regression re-hanging the
    /// suite instead of failing it.
    #[tokio::test]
    async fn retrieve_with_promotion_does_not_deadlock() {
        let config = MemoryConfig {
            enable_memory_promotion: true,
            promotion_config: Some(memory::promotion::PromotionConfig {
                policy_type: memory::promotion::PromotionPolicyType::Importance,
                importance_threshold: 0.0,
                ..memory::promotion::PromotionConfig::default()
            }),
            ..MemoryConfig::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.store("promo_k", "value to promote").await.unwrap();

        let got = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            mem.retrieve("promo_k"),
        )
        .await
        .expect("retrieve() must not deadlock on the promotion path")
        .unwrap();
        assert!(got.is_some(), "retrieved promoted memory must be present");
    }
}
