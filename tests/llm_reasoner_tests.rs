//! Tests for the optional `LlmReasoner` (feature `llm-reasoning`).
//!
//! NO NETWORK: the reasoner's chat call is injectable, so these tests drive
//! it with canned JSON responses (the parse path) and with failing calls
//! (the fail-open-to-heuristic path).
#![cfg(feature = "llm-reasoning")]

use std::sync::Arc;

use synaptic::memory::embeddings::TfIdfProvider;
use synaptic::memory::reasoning::llm::{ChatCall, LlmReasoner};
use synaptic::memory::reasoning::{
    ConflictResolution, ExtractionContext, Fact, HeuristicReasoner, MemoryReasoner, NeighborFact,
};
use synaptic::memory::types::MemoryEntry;

fn canned(reply: &str) -> ChatCall {
    let reply = reply.to_string();
    Arc::new(move |_system, _user| {
        let reply = reply.clone();
        Box::pin(async move { Ok(reply) })
    })
}

fn failing() -> ChatCall {
    Arc::new(|_system, _user| Box::pin(async move { Err("connection refused".to_string()) }))
}

fn ctx() -> ExtractionContext {
    ExtractionContext {
        source_key: "k1".to_string(),
        timestamp: chrono::Utc::now(),
    }
}

fn heuristic() -> HeuristicReasoner {
    HeuristicReasoner::new(Arc::new(TfIdfProvider::default()))
}

#[tokio::test]
async fn extract_parses_structured_json_from_llm() {
    let reply = r#"{"facts":[{"text":"Alice moved to Berlin.","entities":[{"name":"Alice","kind":"Person"},{"name":"Berlin","kind":"Place"}],"relations":[{"subject":"Alice","predicate":"lives_in","object":"Berlin"}]}]}"#;
    let reasoner = LlmReasoner::with_call(canned(reply));
    let extraction = reasoner
        .extract("Alice moved to Berlin.", &ctx())
        .await
        .expect("extract must not fail");
    assert_eq!(extraction.facts.len(), 1);
    let fact = &extraction.facts[0];
    assert_eq!(fact.text, "Alice moved to Berlin.");
    assert_eq!(fact.entities.len(), 2);
    assert_eq!(fact.entities[0].name, "Alice");
    assert_eq!(fact.entities[1].name, "Berlin");
    // Spans are recovered by locating the entity in the fact text.
    assert_eq!(fact.entities[0].span, (0, 5));
    assert_eq!(fact.relations.len(), 1);
    assert_eq!(fact.relations[0].predicate, "lives_in");
}

#[tokio::test]
async fn extract_handles_code_fenced_json() {
    let reply = "```json\n{\"facts\":[{\"text\":\"Bob works at Acme Corp.\",\"entities\":[{\"name\":\"Bob\",\"kind\":\"Person\"}],\"relations\":[]}]}\n```";
    let reasoner = LlmReasoner::with_call(canned(reply));
    let extraction = reasoner
        .extract("Bob works at Acme Corp.", &ctx())
        .await
        .expect("extract must not fail");
    assert_eq!(extraction.facts.len(), 1);
    assert_eq!(extraction.facts[0].entities[0].name, "Bob");
}

#[tokio::test]
async fn resolve_parses_llm_verdict() {
    let reply = r#"{"action":"supersede","old_id":"n1","reason":"newer location"}"#;
    let reasoner = LlmReasoner::with_call(canned(reply));
    let candidate = Fact {
        text: "Alice moved to Munich.".to_string(),
        entities: vec![],
        relations: vec![],
    };
    let neighbors = vec![NeighborFact {
        id: "n1".to_string(),
        similarity: 0.8,
        text: "Alice lives in Berlin.".to_string(),
    }];
    let resolution = reasoner
        .resolve(&candidate, &neighbors)
        .await
        .expect("resolve must not fail");
    assert_eq!(
        resolution,
        ConflictResolution::Supersede {
            old_id: "n1".to_string(),
            reason: "newer location".to_string(),
        }
    );
}

#[tokio::test]
async fn synthesize_parses_llm_insight_with_cluster_ids() {
    let reply = r#"{"insight":"Alice has been relocating across Europe.","confidence":0.9}"#;
    let reasoner = LlmReasoner::with_call(canned(reply));
    let cluster = vec![
        MemoryEntry::new(
            "a".to_string(),
            "Alice moved to Berlin.".to_string(),
            synaptic::memory::types::MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "b".to_string(),
            "Alice moved to Munich.".to_string(),
            synaptic::memory::types::MemoryType::LongTerm,
        ),
    ];
    let insight = reasoner
        .synthesize(&cluster)
        .await
        .expect("synthesize must not fail")
        .expect("insight expected");
    assert_eq!(insight.text, "Alice has been relocating across Europe.");
    assert!((insight.confidence - 0.9).abs() < 1e-9);
    let ids: Vec<String> = cluster.iter().map(|e| e.id().to_string()).collect();
    assert_eq!(insight.derived_from, ids);
}

#[tokio::test]
async fn extract_falls_back_to_heuristic_on_call_failure() {
    let reasoner = LlmReasoner::with_call(failing());
    let text = "Alice moved to Berlin.";
    let got = reasoner
        .extract(text, &ctx())
        .await
        .expect("fail-open: must return heuristic result, not Err");
    let want = heuristic()
        .extract(text, &ctx())
        .await
        .expect("heuristic extract");
    assert_eq!(
        got, want,
        "fallback must equal the HeuristicReasoner result"
    );
    assert!(!got.facts.is_empty(), "heuristic extracts a real fact here");
}

#[tokio::test]
async fn extract_falls_back_to_heuristic_on_unparseable_reply() {
    let reasoner = LlmReasoner::with_call(canned("sorry, I cannot help with that"));
    let text = "Alice moved to Berlin.";
    let got = reasoner
        .extract(text, &ctx())
        .await
        .expect("fail-open: must return heuristic result, not Err");
    let want = heuristic()
        .extract(text, &ctx())
        .await
        .expect("heuristic extract");
    assert_eq!(got, want);
}

#[tokio::test]
async fn resolve_falls_back_to_heuristic_on_call_failure() {
    let reasoner = LlmReasoner::with_call(failing());
    let candidate = Fact {
        text: "Alice moved to Berlin.".to_string(),
        entities: vec![],
        relations: vec![],
    };
    let neighbors = vec![NeighborFact {
        id: "n1".to_string(),
        similarity: 0.99,
        text: "Alice moved to Berlin.".to_string(),
    }];
    let got = reasoner
        .resolve(&candidate, &neighbors)
        .await
        .expect("fail-open: must return heuristic result, not Err");
    let want = heuristic()
        .resolve(&candidate, &neighbors)
        .await
        .expect("heuristic resolve");
    assert_eq!(got, want);
}

#[tokio::test]
async fn resolve_falls_back_when_supersede_targets_unknown_neighbor() {
    // A verdict referencing a non-neighbor id is invalid; fail open.
    let reply = r#"{"action":"supersede","old_id":"not-a-neighbor","reason":"x"}"#;
    let reasoner = LlmReasoner::with_call(canned(reply));
    let candidate = Fact {
        text: "Alice moved to Munich.".to_string(),
        entities: vec![],
        relations: vec![],
    };
    let neighbors = vec![NeighborFact {
        id: "n1".to_string(),
        similarity: 0.8,
        text: "Alice lives in Berlin.".to_string(),
    }];
    let got = reasoner
        .resolve(&candidate, &neighbors)
        .await
        .expect("fail-open: must return heuristic result, not Err");
    let want = heuristic()
        .resolve(&candidate, &neighbors)
        .await
        .expect("heuristic resolve");
    assert_eq!(got, want);
}

#[tokio::test]
async fn synthesize_falls_back_to_heuristic_on_call_failure() {
    let reasoner = LlmReasoner::with_call(failing());
    let cluster = vec![
        MemoryEntry::new(
            "a".to_string(),
            "Alice moved to Berlin.".to_string(),
            synaptic::memory::types::MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "b".to_string(),
            "Alice moved to Munich.".to_string(),
            synaptic::memory::types::MemoryType::LongTerm,
        ),
    ];
    let got = reasoner
        .synthesize(&cluster)
        .await
        .expect("fail-open: must return heuristic result, not Err");
    let want = heuristic()
        .synthesize(&cluster)
        .await
        .expect("heuristic synthesize");
    assert_eq!(got, want);
}
