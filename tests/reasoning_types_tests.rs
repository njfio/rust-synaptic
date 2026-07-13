//! Tests for the memory reasoning types (Task 1.1, agent-memory-v2).

// Test code: panic on wrong variant is the intended behaviour.
#![allow(clippy::panic)]
use synaptic::memory::reasoning::{ConflictResolution, Entity, EntityKind, Fact, Relation};

#[test]
fn conflict_resolution_carries_reason() {
    let r = ConflictResolution::Supersede {
        old_id: "m1".into(),
        reason: "newer value".into(),
    };
    match r {
        ConflictResolution::Supersede { old_id, reason } => {
            assert_eq!(old_id, "m1");
            assert!(reason.contains("newer"));
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn fact_holds_entities_and_relations() {
    let f = Fact {
        text: "Alice lives in Berlin".into(),
        entities: vec![
            Entity {
                name: "Alice".into(),
                kind: EntityKind::Person,
                span: (0, 5),
            },
            Entity {
                name: "Berlin".into(),
                kind: EntityKind::Place,
                span: (15, 21),
            },
        ],
        relations: vec![Relation {
            subject: "Alice".into(),
            predicate: "lives_in".into(),
            object: "Berlin".into(),
        }],
    };
    assert_eq!(f.entities.len(), 2);
    assert_eq!(f.relations[0].predicate, "lives_in");
}
