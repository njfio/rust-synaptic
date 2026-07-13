//! Tests for composite relevance × recency × importance scoring and for the
//! graph/temporal signals being wired into the retrieval pipeline.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use synaptic::memory::knowledge_graph::{GraphConfig, MemoryKnowledgeGraph, RelationshipType};
use synaptic::memory::retrieval::{
    CompositeWeights, FusionStrategy, GraphRetriever, HybridRetriever, KeywordRetriever,
    PipelineConfig,
};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

#[test]
fn composite_weights_defaults() {
    let weights = CompositeWeights::default();
    assert_eq!(weights.relevance, 0.6);
    assert_eq!(weights.recency, 0.2);
    assert_eq!(weights.importance, 0.2);
}

/// (a) Two candidates with identical content (equal fused relevance): the more
/// recent AND more important one must rank first after composite scoring.
#[tokio::test]
async fn composite_prefers_recent_and_important_on_relevance_tie() {
    let storage = Arc::new(MemoryStorage::new());

    let mut stale = MemoryEntry::new(
        "stale_low_importance".to_string(),
        "shared topic content".to_string(),
        MemoryType::LongTerm,
    )
    .with_importance(0.1);
    stale.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(30);

    let fresh = MemoryEntry::new(
        "fresh_high_importance".to_string(),
        "shared topic content".to_string(),
        MemoryType::LongTerm,
    )
    .with_importance(0.9);

    storage.store(&stale).await.unwrap();
    storage.store(&fresh).await.unwrap();

    // WeightedAverage over a single keyword signal yields identical fused
    // relevance for the two identical documents, so only the composite
    // recency/importance terms can order them.
    let config = PipelineConfig::default().with_fusion_strategy(FusionStrategy::WeightedAverage);
    let retriever =
        HybridRetriever::new(config).add_pipeline(Arc::new(KeywordRetriever::new(storage.clone())));

    let results = retriever.search("shared topic", 10).await.unwrap();
    assert_eq!(results.len(), 2, "both candidates must be retrieved");
    assert_eq!(
        results[0].entry.key, "fresh_high_importance",
        "more recent + more important candidate must rank first on a relevance tie: {results:?}"
    );
}

/// (b) A graph-connected candidate that shares no terms with the query (so
/// pure dense/keyword retrieval can never return it) is surfaced when the
/// graph signal is wired into the hybrid pipeline.
#[tokio::test]
async fn graph_connected_candidate_surfaces_via_graph_signal() {
    let storage = Arc::new(MemoryStorage::new());

    let hub = MemoryEntry::new(
        "hub_report".to_string(),
        "quarterly report with financial summary".to_string(),
        MemoryType::LongTerm,
    );
    let target = MemoryEntry::new(
        "linked_projections".to_string(),
        "cash flow projections and fiscal planning notes".to_string(),
        MemoryType::LongTerm,
    );
    storage.store(&hub).await.unwrap();
    storage.store(&target).await.unwrap();

    let mut kg = MemoryKnowledgeGraph::new(GraphConfig::default());
    kg.add_memory_node(&hub).await.unwrap();
    kg.add_memory_node(&target).await.unwrap();
    kg.create_relationship(
        "hub_report",
        "linked_projections",
        RelationshipType::RelatedTo,
        None,
    )
    .await
    .unwrap();
    let kg = Arc::new(tokio::sync::RwLock::new(kg));

    // Control: keyword-only pipeline can never surface the target (zero
    // query-term overlap).
    let keyword_only = HybridRetriever::new(PipelineConfig::default())
        .add_pipeline(Arc::new(KeywordRetriever::new(storage.clone())));
    let control = keyword_only.search("quarterly report", 10).await.unwrap();
    assert!(
        !control.iter().any(|m| m.entry.key == "linked_projections"),
        "control: keyword-only search must miss the graph-connected candidate"
    );

    // With the graph signal wired, the graph-connected candidate surfaces.
    let with_graph = HybridRetriever::new(PipelineConfig::default())
        .add_pipeline(Arc::new(KeywordRetriever::new(storage.clone())))
        .add_pipeline(Arc::new(GraphRetriever::new(storage.clone(), Some(kg))));
    let results = with_graph.search("quarterly report", 10).await.unwrap();
    assert!(
        results.iter().any(|m| m.entry.key == "linked_projections"),
        "graph-connected candidate must be surfaced by the graph signal: {results:?}"
    );
}

/// The default AgentMemory pipeline wires graph + temporal retrievers: a
/// memory related only through the knowledge graph is surfaced end-to-end.
#[tokio::test]
async fn default_pipeline_wires_graph_signal_end_to_end() {
    let mut memory = AgentMemory::new(MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        enable_knowledge_graph: true,
        ..Default::default()
    })
    .await
    .unwrap();

    memory
        .store("hub_report", "quarterly report with financial summary")
        .await
        .unwrap();
    memory
        .store(
            "linked_projections",
            "cash flow projections and fiscal planning notes",
        )
        .await
        .unwrap();
    memory
        .create_memory_relationship(
            "hub_report",
            "linked_projections",
            RelationshipType::RelatedTo,
        )
        .await
        .unwrap();

    let results = memory.search("quarterly report", 10).await.unwrap();
    assert!(
        results.iter().any(|m| m.entry.key == "linked_projections"),
        "default pipeline must surface graph-connected memories: {results:?}"
    );
}
