//! Tests for the intelligent write path (Task 1.3, agent-memory-v2).
//!
//! `store_with_report` runs the configured `MemoryReasoner` after the core
//! storage write: facts are extracted, resolved against similar existing
//! memories, and the outcome is applied to the knowledge graph.

// Test code: panic on wrong variant is the intended behaviour.
#![allow(clippy::panic)]

use synaptic::{AgentMemory, MemoryConfig};

#[tokio::test]
async fn contradicting_fact_supersedes_old_residence_in_kg() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("create memory");

    let first = memory
        .store_with_report("residence_2021", "Alice lives in Berlin.")
        .await
        .expect("store first");
    assert!(
        first.reasoning.is_none(),
        "first store must not degrade reasoning: {:?}",
        first.reasoning
    );

    // The first fact must land in the KG as an active relation.
    let relations = memory
        .entity_relations("Alice")
        .await
        .expect("query relations after first store");
    let berlin = relations
        .iter()
        .find(|r| r.predicate == "lives_in" && r.object == "Berlin")
        .expect("Alice lives_in Berlin relation after first store");
    assert!(!berlin.superseded, "fresh fact must not be superseded");
    assert_eq!(berlin.source_memory.as_deref(), Some("residence_2021"));

    // Second store contradicts the first: the resolution path must run,
    // supersede the Berlin fact, and add Munich as the current residence.
    let second = memory
        .store_with_report("residence_2024", "Alice lives in Munich.")
        .await
        .expect("store second");
    assert!(
        second.reasoning.is_none(),
        "second store must not degrade reasoning: {:?}",
        second.reasoning
    );

    let relations = memory
        .entity_relations("Alice")
        .await
        .expect("query relations after second store");

    let munich = relations
        .iter()
        .find(|r| r.predicate == "lives_in" && r.object == "Munich")
        .expect("Alice lives_in Munich relation after second store");
    assert!(
        !munich.superseded,
        "current residence (Munich) must not be superseded"
    );
    assert_eq!(munich.source_memory.as_deref(), Some("residence_2024"));

    let berlin = relations
        .iter()
        .find(|r| r.predicate == "lives_in" && r.object == "Berlin")
        .expect("Berlin relation must still exist (flagged, not deleted)");
    assert!(
        berlin.superseded,
        "Berlin residence must be flagged superseded after the contradicting store"
    );
}

#[tokio::test]
async fn store_without_extractable_facts_reports_clean() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("create memory");

    let report = memory
        .store_with_report("note", "just some lowercase words without entities")
        .await
        .expect("store");
    assert!(report.reasoning.is_none(), "{:?}", report.reasoning);
    assert!(report.is_clean(), "{report:?}");

    let relations = memory.entity_relations("Alice").await.expect("query");
    assert!(
        relations.is_empty(),
        "no entities were stored: {relations:?}"
    );
}

#[tokio::test]
async fn exact_duplicate_store_does_not_duplicate_relations() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("create memory");

    memory
        .store_with_report("fact_a", "Alice lives in Berlin.")
        .await
        .expect("store first");
    memory
        .store_with_report("fact_b", "Alice lives in Berlin.")
        .await
        .expect("store duplicate");

    let relations = memory
        .entity_relations("Alice")
        .await
        .expect("query relations");
    let berlin_count = relations
        .iter()
        .filter(|r| r.predicate == "lives_in" && r.object == "Berlin")
        .count();
    assert_eq!(
        berlin_count, 1,
        "duplicate store must resolve to NoOp, not duplicate the relation: {relations:?}"
    );
}
