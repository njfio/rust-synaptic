//! Tests for bi-temporal validity fields on knowledge graph nodes and edges.

use chrono::{Duration, Utc};
use synaptic::memory::knowledge_graph::{Edge, Node, NodeType, RelationshipType};
use uuid::Uuid;

#[test]
fn fresh_edge_is_valid_now() {
    let edge = Edge::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        RelationshipType::RelatedTo,
        None,
    );
    let now = Utc::now();
    assert!(edge.is_valid_at(now));
    assert!(edge.valid_to.is_none());
    assert!(edge.expired_at.is_none());
    assert!(edge.valid_from <= now);
    assert!(edge.ingested_at <= now);
}

#[test]
fn expired_edge_is_invalid_after_expiry_and_valid_before() {
    let mut edge = Edge::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        RelationshipType::Causes,
        None,
    );
    let t = Utc::now() + Duration::seconds(60);
    edge.expire_at(t);
    assert!(!edge.is_valid_at(t + Duration::seconds(1)));
    assert!(edge.is_valid_at(t - Duration::seconds(1)));
    assert_eq!(edge.expired_at, Some(t));
}

#[test]
fn invalidated_edge_is_invalid_after_valid_to() {
    let mut edge = Edge::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        RelationshipType::PartOf,
        None,
    );
    let t = Utc::now() + Duration::seconds(60);
    edge.invalidate(t);
    assert!(!edge.is_valid_at(t + Duration::seconds(1)));
    assert!(edge.is_valid_at(t - Duration::seconds(1)));
    assert_eq!(edge.valid_to, Some(t));
}

#[test]
fn node_bitemporal_behaviour_matches_edge() {
    let mut node = Node::new(NodeType::Concept, "alice".to_string());
    let now = Utc::now();
    assert!(node.is_valid_at(now));

    let t = now + Duration::seconds(60);
    node.expire_at(t);
    assert!(!node.is_valid_at(t + Duration::seconds(1)));
    assert!(node.is_valid_at(t - Duration::seconds(1)));

    let mut node2 = Node::new(NodeType::Concept, "bob".to_string());
    node2.invalidate(t);
    assert!(!node2.is_valid_at(t + Duration::seconds(1)));
    assert!(node2.is_valid_at(t - Duration::seconds(1)));
}

#[test]
fn not_yet_valid_edge_is_invalid_before_valid_from() {
    let edge = Edge::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        RelationshipType::RelatedTo,
        None,
    );
    assert!(!edge.is_valid_at(edge.valid_from - Duration::seconds(1)));
}

#[test]
fn legacy_serialized_edge_without_bitemporal_fields_deserializes() {
    // Simulates a graph serialized before bi-temporal fields existed.
    let legacy = serde_json::json!({
        "id": Uuid::new_v4(),
        "from_node": Uuid::new_v4(),
        "to_node": Uuid::new_v4(),
        "relationship": {
            "relationship_type": "RelatedTo",
            "properties": {},
            "strength": 1.0,
            "confidence": 1.0,
            "created_at": Utc::now(),
            "last_modified": Utc::now()
        }
    });
    let edge: Edge = serde_json::from_value(legacy).expect("legacy edge must deserialize");
    // Legacy data defaults valid_from/ingested_at to MIN_UTC: always valid.
    assert!(edge.is_valid_at(Utc::now()));
}
