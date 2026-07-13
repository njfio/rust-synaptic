//! Tests for the deterministic `HeuristicReasoner` (Task 1.2, agent-memory-v2).

// Test code: panic on wrong variant is the intended behaviour.
#![allow(clippy::panic)]

use std::sync::Arc;
use synaptic::memory::embeddings::TfIdfProvider;
use synaptic::memory::reasoning::{
    ConflictResolution, EntityKind, ExtractionContext, HeuristicReasoner, MemoryReasoner,
    NeighborFact,
};
use synaptic::memory::types::{MemoryEntry, MemoryType};

fn reasoner() -> HeuristicReasoner {
    HeuristicReasoner::new(Arc::new(TfIdfProvider::default()))
}

fn ctx(key: &str) -> ExtractionContext {
    ExtractionContext {
        source_key: key.to_string(),
        timestamp: chrono::Utc::now(),
    }
}

fn neighbor(id: &str, similarity: f64, text: &str) -> NeighborFact {
    NeighborFact {
        id: id.to_string(),
        similarity,
        text: text.to_string(),
    }
}

#[tokio::test]
async fn extract_finds_entities_dates_and_relation() {
    let r = reasoner();
    let text = "Alice moved to Berlin in 2021.";
    let extraction = r.extract(text, &ctx("m-alice")).await.expect("extract");

    assert_eq!(extraction.facts.len(), 1, "one sentence, one fact");
    let fact = &extraction.facts[0];
    assert_eq!(fact.text, "Alice moved to Berlin in 2021.");

    let alice = fact
        .entities
        .iter()
        .find(|e| e.name == "Alice")
        .expect("Alice entity");
    assert_eq!(alice.kind, EntityKind::Person);
    assert_eq!(alice.span, (0, 5));
    assert_eq!(&text[alice.span.0..alice.span.1], "Alice");

    let berlin = fact
        .entities
        .iter()
        .find(|e| e.name == "Berlin")
        .expect("Berlin entity");
    assert_eq!(berlin.kind, EntityKind::Place);
    assert_eq!(&text[berlin.span.0..berlin.span.1], "Berlin");

    let year = fact
        .entities
        .iter()
        .find(|e| e.name == "2021")
        .expect("2021 date entity");
    assert_eq!(year.kind, EntityKind::Date);
    assert_eq!(&text[year.span.0..year.span.1], "2021");

    let rel = fact
        .relations
        .iter()
        .find(|r| r.subject == "Alice" && r.object == "Berlin")
        .expect("Alice-Berlin relation");
    assert_eq!(rel.predicate, "lives_in");
}

#[tokio::test]
async fn extract_classifies_org_date_number_and_quote() {
    let r = reasoner();
    let text = "Bob joined Acme Corp on 2023-05-01 and said \"great team\" after 42 days.";
    let extraction = r.extract(text, &ctx("m-bob")).await.expect("extract");
    let fact = &extraction.facts[0];

    let org = fact
        .entities
        .iter()
        .find(|e| e.name == "Acme Corp")
        .expect("org entity");
    assert_eq!(org.kind, EntityKind::Org);
    assert_eq!(&text[org.span.0..org.span.1], "Acme Corp");

    let date = fact
        .entities
        .iter()
        .find(|e| e.name == "2023-05-01")
        .expect("ISO date");
    assert_eq!(date.kind, EntityKind::Date);

    let num = fact
        .entities
        .iter()
        .find(|e| e.name == "42")
        .expect("number");
    assert_eq!(num.kind, EntityKind::Number);

    let quote = fact
        .entities
        .iter()
        .find(|e| e.kind == EntityKind::Quoted)
        .expect("quoted span");
    assert_eq!(quote.name, "great team");

    let rel = fact
        .relations
        .iter()
        .find(|r| r.subject == "Bob" && r.object == "Acme Corp")
        .expect("works_at relation");
    assert_eq!(rel.predicate, "works_at");
}

#[tokio::test]
async fn extract_is_deterministic() {
    let r = reasoner();
    let text = "Alice moved to Berlin in 2021. Bob joined Acme Corp.";
    let a = r.extract(text, &ctx("k")).await.expect("extract");
    let b = r.extract(text, &ctx("k")).await.expect("extract");
    assert_eq!(a, b);
}

#[tokio::test]
async fn resolve_supersedes_contradicting_residence() {
    let r = reasoner();
    // Candidate contradicts: same subject+predicate, different object.
    let candidate = &r
        .extract("Alice moved to Munich.", &ctx("mem-munich"))
        .await
        .expect("extract new")
        .facts[0];

    // Supersede is detected from the neighbor's CONTENT, not any cache.
    let resolution = r
        .resolve(
            candidate,
            &[neighbor(
                "mem-berlin",
                0.9,
                "Alice moved to Berlin in 2021.",
            )],
        )
        .await
        .expect("resolve");
    match resolution {
        ConflictResolution::Supersede { old_id, .. } => assert_eq!(old_id, "mem-berlin"),
        other => panic!("expected Supersede, got {other:?}"),
    }
}

#[tokio::test]
async fn resolve_noop_on_duplicate_and_insert_on_unrelated() {
    let r = reasoner();
    let dup = &r
        .extract("Alice moved to Berlin in 2021.", &ctx("mem-dup"))
        .await
        .expect("extract dup")
        .facts[0];
    match r
        .resolve(
            dup,
            &[neighbor(
                "mem-berlin",
                0.99,
                "Alice moved to Berlin in 2021.",
            )],
        )
        .await
        .expect("resolve dup")
    {
        ConflictResolution::NoOp { .. } => {}
        other => panic!("expected NoOp, got {other:?}"),
    }

    let unrelated = &r
        .extract("Carol likes tea.", &ctx("mem-carol"))
        .await
        .expect("extract unrelated")
        .facts[0];
    match r
        .resolve(
            unrelated,
            &[neighbor(
                "mem-berlin",
                0.2,
                "Alice moved to Berlin in 2021.",
            )],
        )
        .await
        .expect("resolve unrelated")
    {
        ConflictResolution::Insert => {}
        other => panic!("expected Insert, got {other:?}"),
    }
}

#[tokio::test]
async fn resolve_updates_in_place_when_same_fact_gains_detail() {
    let r = reasoner();
    let detailed = &r
        .extract(
            "Alice moved to Berlin in 2021 after finishing school.",
            &ctx("mem-detail"),
        )
        .await
        .expect("extract detailed")
        .facts[0];
    match r
        .resolve(
            detailed,
            &[neighbor("mem-berlin", 0.9, "Alice moved to Berlin.")],
        )
        .await
        .expect("resolve detailed")
    {
        ConflictResolution::UpdateInPlace { .. } => {}
        other => panic!("expected UpdateInPlace, got {other:?}"),
    }
}

#[tokio::test]
async fn resolve_inserts_on_high_similarity_without_relation_evidence() {
    let r = reasoner();
    let candidate = &r
        .extract("Alice moved to Munich.", &ctx("mem-munich"))
        .await
        .expect("extract candidate")
        .facts[0];
    // Neighbor text has no comparable extracted relation: must NOT be a blind
    // UpdateInPlace/Supersede — falls back to Insert.
    match r
        .resolve(
            candidate,
            &[neighbor(
                "mem-vague",
                0.9,
                "Some unrelated note about weather.",
            )],
        )
        .await
        .expect("resolve")
    {
        ConflictResolution::Insert => {}
        other => panic!("expected Insert, got {other:?}"),
    }
}

#[tokio::test]
async fn extract_gap_guard_rejects_intervening_entity_relation() {
    let r = reasoner();
    let fact = &r
        .extract("Alice met Bob who moved to Berlin.", &ctx("k"))
        .await
        .expect("extract")
        .facts[0];
    // Must not spuriously link Alice to Berlin across the intervening Bob.
    assert!(
        !fact
            .relations
            .iter()
            .any(|rel| rel.subject == "Alice" && rel.object == "Berlin"),
        "gap-guard should reject Alice-Berlin across intervening entity: {:?}",
        fact.relations
    );
}

#[tokio::test]
async fn synthesize_returns_insight_with_cluster_ids() {
    let r = reasoner();
    let entries: Vec<MemoryEntry> = [
        "Alice works on the memory system project every day.",
        "Alice improved the memory system retrieval quality.",
        "The memory system project shipped a new release.",
    ]
    .iter()
    .enumerate()
    .map(|(i, text)| {
        MemoryEntry::new(
            format!("cluster-mem-{i}"),
            (*text).to_string(),
            MemoryType::LongTerm,
        )
    })
    .collect();
    let ids: Vec<String> = entries.iter().map(|e| e.id().to_string()).collect();

    let insight = r
        .synthesize(&entries)
        .await
        .expect("synthesize")
        .expect("insight for a coherent cluster");
    assert_eq!(insight.derived_from, ids);
    assert!(insight.text.contains("Across 3 related memories:"));
    assert!(insight.confidence > 0.0 && insight.confidence <= 1.0);
}

#[tokio::test]
async fn synthesize_empty_cluster_is_none() {
    let r = reasoner();
    assert!(r.synthesize(&[]).await.expect("synthesize").is_none());
    assert_eq!(r.name(), "heuristic");
}
