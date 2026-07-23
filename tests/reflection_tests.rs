//! Tests for triggered reflection producing provenance-linked insights
//! (Task 4.1, agent-memory-v2).
//!
//! `AgentMemory::reflect` clusters the memories accumulated since the last
//! reflection by embedding similarity, synthesizes an `Insight` per cluster,
//! writes each insight as a distinguishable memory, and records provenance
//! `derives` edges in the knowledge graph from the insight to each source.

// Test code: panic on wrong variant is the intended behaviour.
#![allow(clippy::panic)]

use std::collections::HashSet;
use synaptic::memory::reflection::ReflectionConfig;
use synaptic::{AgentMemory, MemoryConfig};

const TOPIC_MEMORIES: [(&str, &str); 3] = [
    (
        "rust_borrow_1",
        "The Rust borrow checker prevents data races by enforcing ownership rules.",
    ),
    (
        "rust_borrow_2",
        "Rust ownership rules and the borrow checker catch data races at compile time.",
    ),
    (
        "rust_borrow_3",
        "Compile time enforcement of ownership by the Rust borrow checker eliminates data races.",
    ),
];

const UNRELATED_MEMORY: (&str, &str) = (
    "banana_bread",
    "Grandma's banana bread recipe calls for ripe fruit, cinnamon and a slow oven.",
);

async fn populated_memory() -> AgentMemory {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("create memory");
    for (key, value) in TOPIC_MEMORIES {
        memory.store(key, value).await.expect("store topic memory");
    }
    memory
        .store(UNRELATED_MEMORY.0, UNRELATED_MEMORY.1)
        .await
        .expect("store unrelated memory");
    memory
}

#[tokio::test]
async fn reflect_does_not_trigger_below_importance_threshold() {
    let mut memory = populated_memory().await;
    // Four default-importance (0.5) memories accumulate 2.0, below the
    // default threshold of 3.0: reflection must not fire.
    let insights = memory.reflect().await.expect("reflect");
    assert!(
        insights.is_empty(),
        "reflection must not trigger below the importance threshold"
    );
}

#[tokio::test]
async fn reflect_synthesizes_one_insight_with_provenance_edges() {
    let mut memory = populated_memory().await;

    // Collect the ids the stored memories were assigned.
    let mut topic_ids = HashSet::new();
    for (key, _) in TOPIC_MEMORIES {
        let entry = memory
            .retrieve(key)
            .await
            .expect("retrieve topic memory")
            .expect("topic memory exists");
        topic_ids.insert(entry.id().to_string());
    }
    let unrelated_id = memory
        .retrieve(UNRELATED_MEMORY.0)
        .await
        .expect("retrieve unrelated memory")
        .expect("unrelated memory exists")
        .id()
        .to_string();

    // Lower the trigger so the accumulated importance (4 x 0.5) fires it.
    memory.set_reflection_config(ReflectionConfig {
        importance_threshold: 1.0,
        ..ReflectionConfig::default()
    });

    let insights = memory.reflect().await.expect("reflect");
    assert_eq!(insights.len(), 1, "exactly one cluster reaches min_cluster");

    let insight = &insights[0];
    let derived: HashSet<String> = insight.derived_from.iter().cloned().collect();
    assert_eq!(
        derived, topic_ids,
        "insight must derive from exactly the three topic memories"
    );
    assert!(
        !derived.contains(&unrelated_id),
        "unrelated memory must not be a source"
    );
    assert!(!insight.text.is_empty(), "insight text must be non-empty");

    // The insight was written as a distinguishable memory.
    let insight_entries = memory.insight_memories().await;
    assert_eq!(insight_entries.len(), 1, "one insight memory written");
    let insight_entry = &insight_entries[0];
    assert_eq!(insight_entry.value, insight.text);
    assert!(insight_entry.has_tag("insight"), "tagged as insight");
    assert_eq!(
        insight_entry
            .metadata
            .get_custom_field("memory_source")
            .map(String::as_str),
        Some("reflection")
    );

    // Provenance: a `derives` edge exists from the insight to each of the
    // three source memories, and NOT to the unrelated one.
    let sources = memory
        .derived_sources(&insight_entry.key)
        .await
        .expect("query derives edges");
    let source_set: HashSet<&str> = sources.iter().map(String::as_str).collect();
    let expected: HashSet<&str> = TOPIC_MEMORIES.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        source_set, expected,
        "derives edges must point at exactly the three topic memories"
    );
    assert!(
        !source_set.contains(UNRELATED_MEMORY.0),
        "no derives edge to the unrelated memory"
    );

    // Reflection consumed the accumulated importance: an immediate second
    // reflect() has nothing new to work with.
    let again = memory.reflect().await.expect("second reflect");
    assert!(again.is_empty(), "trigger resets after a reflection");
}
