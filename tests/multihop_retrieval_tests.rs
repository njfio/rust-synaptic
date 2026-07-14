//! Multi-hop graph-expansion retrieval tests (HippoRAG-style).
//!
//! Constructs memories A -> B -> C in the knowledge graph where the query
//! matches A lexically but the answer lives at C (which shares no terms with
//! the query). Single-hop dense/keyword retrieval cannot surface C; the
//! multi-hop expansion stage must.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use std::sync::Arc;
use synaptic::memory::knowledge_graph::{GraphConfig, MemoryKnowledgeGraph, RelationshipType};
use synaptic::memory::retrieval::{
    HybridRetriever, KeywordRetriever, MultiHopConfig, MultiHopGraphRetriever, PipelineConfig,
    RetrievalPipeline,
};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

/// A -> B -> C chain where only A shares terms with the query and the answer
/// is at C. Returns (storage, kg).
async fn chain_fixture() -> (
    Arc<MemoryStorage>,
    Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>,
) {
    let storage = Arc::new(MemoryStorage::new());

    // Creation times are spaced > 1 hour apart so the knowledge graph's
    // automatic TemporallyRelated edge detection (co-created within an hour)
    // does not add edges: ONLY the explicit A -> B -> C relations exist.
    let mut a = MemoryEntry::new(
        "mem_a".to_string(),
        "zanzibar shipping manifest logged at the harbor office".to_string(),
        MemoryType::LongTerm,
    );
    a.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(4);
    // B: bridge memory, shares no query terms.
    let mut b = MemoryEntry::new(
        "mem_b".to_string(),
        "the cargo was transferred to warehouse seventeen".to_string(),
        MemoryType::LongTerm,
    );
    b.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(3);
    // C: the answer, shares no terms with the query at all.
    let mut c = MemoryEntry::new(
        "mem_c".to_string(),
        "warehouse seventeen inventory lists forty crates of cloves".to_string(),
        MemoryType::LongTerm,
    );
    c.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(2);
    // Distractor with no graph connection.
    let mut d = MemoryEntry::new(
        "mem_d".to_string(),
        "unrelated note about garden maintenance".to_string(),
        MemoryType::LongTerm,
    );
    d.metadata.created_at = chrono::Utc::now() - chrono::Duration::days(1);

    for entry in [&a, &b, &c, &d] {
        storage.store(entry).await.unwrap();
    }

    let mut kg = MemoryKnowledgeGraph::new(GraphConfig::default());
    kg.add_memory_node(&a).await.unwrap();
    kg.add_memory_node(&b).await.unwrap();
    kg.add_memory_node(&c).await.unwrap();
    kg.add_memory_node(&d).await.unwrap();
    kg.create_relationship("mem_a", "mem_b", RelationshipType::RelatedTo, None)
        .await
        .unwrap();
    kg.create_relationship("mem_b", "mem_c", RelationshipType::RelatedTo, None)
        .await
        .unwrap();

    (storage, Arc::new(tokio::sync::RwLock::new(kg)))
}

const QUERY: &str = "zanzibar shipping manifest";

/// Control: a single-hop pipeline (keyword only, no graph expansion) must NOT
/// surface C — it shares zero terms with the query.
#[tokio::test]
async fn single_hop_pipeline_misses_two_hop_answer() {
    let (storage, _kg) = chain_fixture().await;

    let single_hop = HybridRetriever::new(PipelineConfig::default())
        .add_pipeline(Arc::new(KeywordRetriever::new(storage.clone())));
    let results = single_hop.search(QUERY, 10).await.unwrap();

    assert!(
        results.iter().any(|m| m.entry.key == "mem_a"),
        "single-hop must find the lexical match A: {results:?}"
    );
    assert!(
        !results.iter().any(|m| m.entry.key == "mem_c"),
        "single-hop keyword retrieval must NOT surface the 2-hop answer C: {results:?}"
    );
}

/// The multi-hop retriever, seeded by the keyword retriever, surfaces C via
/// the A -> B -> C chain when fused into the hybrid pipeline.
#[tokio::test]
async fn multihop_pipeline_surfaces_two_hop_answer() {
    let (storage, kg) = chain_fixture().await;

    let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(storage.clone()));
    let multihop = MultiHopGraphRetriever::new(
        storage.clone(),
        Some(kg),
        vec![keyword.clone()],
        MultiHopConfig::default(),
    );

    let hybrid = HybridRetriever::new(PipelineConfig::default())
        .add_pipeline(keyword)
        .add_pipeline(Arc::new(multihop));
    let results = hybrid.search(QUERY, 10).await.unwrap();

    assert!(
        results.iter().any(|m| m.entry.key == "mem_c"),
        "multi-hop expansion must surface the 2-hop answer C: {results:?}"
    );
    assert!(
        results.iter().any(|m| m.entry.key == "mem_b"),
        "the 1-hop bridge B should surface too: {results:?}"
    );
    assert!(
        !results.iter().any(|m| m.entry.key == "mem_d"),
        "the unconnected distractor must not surface: {results:?}"
    );
}

/// Hop cap is respected: with max_hops = 1 the retriever reaches B but not C.
#[tokio::test]
async fn hop_cap_bounds_expansion() {
    let (storage, kg) = chain_fixture().await;

    let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(storage.clone()));
    let multihop = MultiHopGraphRetriever::new(
        storage.clone(),
        Some(kg),
        vec![keyword],
        MultiHopConfig {
            max_hops: 1,
            ..MultiHopConfig::default()
        },
    );

    let results = multihop.search(QUERY, 10, None).await.unwrap();
    assert!(
        results.iter().any(|s| s.memory.entry.key == "mem_b"),
        "1-hop expansion must reach B: {results:?}"
    );
    assert!(
        !results.iter().any(|s| s.memory.entry.key == "mem_c"),
        "1-hop expansion must NOT reach the 2-hop node C: {results:?}"
    );
}

/// Scores decay per hop: the 1-hop bridge B outscores the 2-hop answer C.
#[tokio::test]
async fn scores_decay_per_hop() {
    let (storage, kg) = chain_fixture().await;

    let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(storage.clone()));
    let multihop = MultiHopGraphRetriever::new(
        storage.clone(),
        Some(kg),
        vec![keyword],
        MultiHopConfig::default(),
    );

    let results = multihop.search(QUERY, 10, None).await.unwrap();
    let score_of = |key: &str| {
        results
            .iter()
            .find(|s| s.memory.entry.key == key)
            .unwrap_or_else(|| panic!("{key} missing from {results:?}"))
            .score
    };
    assert!(
        score_of("mem_b") > score_of("mem_c"),
        "1-hop score must exceed 2-hop score: b={} c={}",
        score_of("mem_b"),
        score_of("mem_c")
    );
}

/// Determinism: identical corpus and query produce the identical ranked key
/// sequence across repeated runs.
#[tokio::test]
async fn multihop_results_are_deterministic() {
    let mut baseline: Option<Vec<String>> = None;
    for _ in 0..3 {
        let (storage, kg) = chain_fixture().await;
        let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(storage.clone()));
        let multihop = MultiHopGraphRetriever::new(
            storage.clone(),
            Some(kg),
            vec![keyword.clone()],
            MultiHopConfig::default(),
        );
        let hybrid = HybridRetriever::new(PipelineConfig::default())
            .add_pipeline(keyword)
            .add_pipeline(Arc::new(multihop));
        let keys: Vec<String> = hybrid
            .search(QUERY, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.entry.key)
            .collect();
        match &baseline {
            None => baseline = Some(keys),
            Some(expected) => assert_eq!(&keys, expected, "ranking must be deterministic"),
        }
    }
}

/// Without a knowledge graph the retriever reports itself unavailable and the
/// pipeline skips it (no error, no results fabricated).
#[tokio::test]
async fn unavailable_without_knowledge_graph() {
    let (storage, _kg) = chain_fixture().await;
    let keyword: Arc<dyn RetrievalPipeline> = Arc::new(KeywordRetriever::new(storage.clone()));
    let multihop =
        MultiHopGraphRetriever::new(storage, None, vec![keyword], MultiHopConfig::default());
    assert!(!multihop.is_available());
}

/// End-to-end: the default AgentMemory pipeline (multi-hop ON by default)
/// surfaces the 2-hop answer.
#[tokio::test]
async fn agent_memory_default_pipeline_surfaces_two_hop_answer() {
    let mut memory = AgentMemory::new(MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        enable_knowledge_graph: true,
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(
        MemoryConfig::default().enable_multihop_retrieval,
        "multi-hop must default ON"
    );

    memory
        .store(
            "mem_a",
            "zanzibar shipping manifest logged at the harbor office",
        )
        .await
        .unwrap();
    memory
        .store("mem_b", "the cargo was transferred to warehouse seventeen")
        .await
        .unwrap();
    memory
        .store(
            "mem_c",
            "warehouse seventeen inventory lists forty crates of cloves",
        )
        .await
        .unwrap();
    memory
        .create_memory_relationship("mem_a", "mem_b", RelationshipType::RelatedTo)
        .await
        .unwrap();
    memory
        .create_memory_relationship("mem_b", "mem_c", RelationshipType::RelatedTo)
        .await
        .unwrap();

    let results = memory.search(QUERY, 10).await.unwrap();
    assert!(
        results.iter().any(|m| m.entry.key == "mem_c"),
        "default pipeline must surface the 2-hop answer C: {results:?}"
    );
}

/// Ablation toggle: `enable_multihop_retrieval = false` builds a pipeline
/// without the multi-hop stage (measurable single-hop baseline).
#[tokio::test]
async fn multihop_toggle_can_be_disabled_for_ablation() {
    let mut memory = AgentMemory::new(MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        // Disable the KG so both the legacy graph signal and multi-hop are
        // out of the picture: this is the single-hop dense+keyword baseline.
        enable_knowledge_graph: false,
        enable_multihop_retrieval: false,
        ..Default::default()
    })
    .await
    .unwrap();

    memory
        .store(
            "mem_a",
            "zanzibar shipping manifest logged at the harbor office",
        )
        .await
        .unwrap();
    memory
        .store("mem_b", "the cargo was transferred to warehouse seventeen")
        .await
        .unwrap();
    memory
        .store(
            "mem_c",
            "warehouse seventeen inventory lists forty crates of cloves",
        )
        .await
        .unwrap();

    let results = memory.search(QUERY, 10).await.unwrap();
    assert!(
        results.iter().any(|m| m.entry.key == "mem_a"),
        "baseline must still find the lexical match A: {results:?}"
    );
    assert!(
        !results.iter().any(|m| m.entry.key == "mem_c"),
        "single-hop baseline must NOT surface the 2-hop answer C: {results:?}"
    );
}
