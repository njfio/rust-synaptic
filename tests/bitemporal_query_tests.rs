//! Tests for edge invalidation and point-in-time queries (Task 2.2,
//! agent-memory-v2).
//!
//! `KnowledgeGraph::invalidate_edge` ends an edge's event-time validity and
//! expires it in system time without deleting it; `neighbors_as_of` returns
//! only the edges incident to a node that are valid at the given instant.

// Test code: panic on wrong variant is the intended behaviour.
#![allow(clippy::panic)]

use chrono::{Duration, Utc};
use synaptic::memory::knowledge_graph::{
    Edge, GraphConfig, KnowledgeGraph, MemoryKnowledgeGraph, Node, NodeType, RelationshipType,
};
use synaptic::memory::reasoning::{Entity, EntityKind, Fact, Relation};

fn residence_fact(city: &str, extra_relations: Vec<Relation>) -> Fact {
    let mut relations = vec![Relation {
        subject: "Alice".to_string(),
        predicate: "lives_in".to_string(),
        object: city.to_string(),
    }];
    relations.extend(extra_relations);
    Fact {
        text: format!("Alice lives in {city}."),
        entities: vec![
            Entity {
                name: "Alice".to_string(),
                kind: EntityKind::Person,
                span: (0, 5),
            },
            Entity {
                name: city.to_string(),
                kind: EntityKind::Place,
                span: (0, 0),
            },
        ],
        relations,
    }
}

#[tokio::test]
async fn neighbors_as_of_respects_invalidation_without_deleting_edges() {
    let graph = KnowledgeGraph::new(GraphConfig::default());
    let alice = graph
        .add_node(Node::new(NodeType::Person, "Alice".to_string()))
        .await
        .expect("add alice");
    let berlin = graph
        .add_node(Node::new(NodeType::Location, "Berlin".to_string()))
        .await
        .expect("add berlin");
    let munich = graph
        .add_node(Node::new(NodeType::Location, "Munich".to_string()))
        .await
        .expect("add munich");

    let t0 = Utc::now();
    let t1 = t0 + Duration::seconds(60);

    // Berlin residence known since before t0.
    let mut berlin_edge = Edge::new(
        alice,
        berlin,
        RelationshipType::Custom("lives_in".to_string()),
        None,
    );
    berlin_edge.valid_from = t0 - Duration::seconds(60);
    let berlin_edge_id = graph.add_edge(berlin_edge).await.expect("add berlin edge");

    // At t1 the residence is superseded by Munich.
    let found = graph
        .invalidate_edge(&berlin_edge_id.to_string(), t1)
        .expect("invalidate berlin edge");
    assert!(found, "the berlin edge must be found and invalidated");

    let mut munich_edge = Edge::new(
        alice,
        munich,
        RelationshipType::Custom("lives_in".to_string()),
        None,
    );
    munich_edge.valid_from = t1;
    graph.add_edge(munich_edge).await.expect("add munich edge");

    // As of t0: Berlin only.
    let at_t0 = graph.neighbors_as_of(&alice.to_string(), t0);
    assert_eq!(at_t0.len(), 1, "exactly one edge valid at t0: {at_t0:?}");
    assert_eq!(at_t0[0].to_node, berlin, "the t0 residence is Berlin");

    // Just after t1: Munich only.
    let after_t1 = graph.neighbors_as_of(&alice.to_string(), t1 + Duration::seconds(1));
    assert_eq!(
        after_t1.len(),
        1,
        "exactly one edge valid after t1: {after_t1:?}"
    );
    assert_eq!(
        after_t1[0].to_node, munich,
        "the residence after t1 is Munich"
    );

    // The Berlin edge still exists in the graph — invalid, not deleted.
    let stored = graph
        .get_edge(berlin_edge_id)
        .await
        .expect("get edge")
        .expect("berlin edge must still be stored");
    assert_eq!(stored.valid_to, Some(t1));
    assert_eq!(stored.expired_at, Some(t1));
    assert!(!stored.is_valid_at(t1 + Duration::seconds(1)));
}

#[tokio::test]
async fn invalidate_edge_returns_false_for_unknown_edge() {
    let graph = KnowledgeGraph::new(GraphConfig::default());
    let found = graph
        .invalidate_edge(&uuid::Uuid::new_v4().to_string(), Utc::now())
        .expect("invalidate unknown edge");
    assert!(!found, "an unknown edge id must report not-found");
}

#[tokio::test]
async fn superseding_invalidates_only_matching_predicate_edges() {
    let mut kg = MemoryKnowledgeGraph::new(GraphConfig::default());

    // One source memory contributes two facts: a residence and an employer.
    let old_fact = residence_fact(
        "Berlin",
        vec![Relation {
            subject: "Alice".to_string(),
            predicate: "works_at".to_string(),
            object: "Acme".to_string(),
        }],
    );
    kg.add_extracted_fact("residence_2021", &old_fact)
        .await
        .expect("add old fact");

    // Superseding the residence must only touch the lives_in edge.
    let new_fact = residence_fact("Munich", Vec::new());
    let invalidated = kg
        .supersede_matching_relations("residence_2021", &new_fact)
        .await
        .expect("supersede");
    assert_eq!(invalidated, 1, "exactly the lives_in edge is invalidated");
    kg.add_extracted_fact("residence_2024", &new_fact)
        .await
        .expect("add new fact");

    let relations = kg.relations_for_entity("Alice").await.expect("query");
    let berlin = relations
        .iter()
        .find(|r| r.predicate == "lives_in" && r.object == "Berlin")
        .expect("Berlin relation must still exist (invalid, not deleted)");
    assert!(
        berlin.superseded,
        "Berlin residence must read as superseded"
    );

    let munich = relations
        .iter()
        .find(|r| r.predicate == "lives_in" && r.object == "Munich")
        .expect("Munich relation");
    assert!(
        !munich.superseded,
        "current residence must not be superseded"
    );

    let acme = relations
        .iter()
        .find(|r| r.predicate == "works_at" && r.object == "Acme")
        .expect("works_at relation");
    assert!(
        !acme.superseded,
        "superseding lives_in must not invalidate the unrelated works_at \
         edge from the same source memory"
    );
}
