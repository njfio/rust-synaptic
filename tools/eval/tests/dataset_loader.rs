//! Loader tests against the committed hand-authored fixture, which mirrors
//! the real LongMemEval-S and LoCoMo JSON schemas.

use synaptic_eval::dataset::{load_locomo_str, load_longmemeval_str, QType};

fn fixture_section(key: &str) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/sample_conversations.json"
    );
    let raw = std::fs::read_to_string(path).expect("fixture must exist");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture must be valid JSON");
    serde_json::to_string(&value[key]).expect("fixture section must serialize")
}

#[test]
fn longmemeval_fixture_parses_into_typed_model() {
    let convs = load_longmemeval_str(&fixture_section("longmemeval")).expect("loader must parse");
    assert_eq!(convs.len(), 2);

    let c = &convs[0];
    assert_eq!(c.id, "fixture_lme_1");
    assert_eq!(c.sessions.len(), 2);
    assert_eq!(c.sessions[0].id, "fixture_sess_a");
    assert_eq!(
        c.sessions[0].timestamp.as_deref(),
        Some("2023/05/20 (Sat) 02:21")
    );
    assert_eq!(c.sessions[1].turns.len(), 2);
    assert_eq!(c.sessions[1].turns[0].speaker, "user");
    assert!(c.sessions[1].turns[0].text.contains("Lisbon"));

    assert_eq!(c.questions.len(), 1);
    let q = &c.questions[0];
    assert_eq!(q.id, "fixture_lme_1");
    assert_eq!(q.qtype, QType::KnowledgeUpdate);
    assert_eq!(q.evidence_ids, vec!["fixture_sess_b".to_string()]);
    assert_eq!(q.gold_answer, "Lisbon");
}

#[test]
fn longmemeval_abstention_suffix_maps_to_abstention_qtype() {
    let convs = load_longmemeval_str(&fixture_section("longmemeval")).expect("loader must parse");
    let q = &convs[1].questions[0];
    assert_eq!(q.id, "fixture_lme_2_abs");
    assert_eq!(q.qtype, QType::Abstention);
    assert!(q.evidence_ids.is_empty());
}

#[test]
fn locomo_fixture_parses_into_typed_model() {
    let convs = load_locomo_str(&fixture_section("locomo")).expect("loader must parse");
    assert_eq!(convs.len(), 1);

    let c = &convs[0];
    assert_eq!(c.id, "fixture_locomo_1");
    assert_eq!(c.sessions.len(), 2);
    assert_eq!(c.sessions[0].id, "session_1");
    assert_eq!(
        c.sessions[0].timestamp.as_deref(),
        Some("1:56 pm on 8 May, 2023")
    );
    assert_eq!(c.sessions[0].turns[0].speaker, "Maya");
    assert!(c.sessions[0].turns[0].text.contains("Waffles"));
    assert_eq!(c.sessions[1].turns.len(), 2);

    assert_eq!(c.questions.len(), 3);
    let multi = &c.questions[0];
    assert_eq!(multi.qtype, QType::MultiHop);
    assert_eq!(
        multi.evidence_ids,
        vec!["D1:1".to_string(), "D2:1".to_string(), "D2:2".to_string()]
    );
    assert!(multi.gold_answer.contains("Waffles"));

    let temporal = &c.questions[1];
    assert_eq!(temporal.qtype, QType::Temporal);
    assert_eq!(temporal.evidence_ids, vec!["D1:1".to_string()]);

    let adversarial = &c.questions[2];
    assert_eq!(adversarial.qtype, QType::Abstention);
    assert!(adversarial.gold_answer.contains("Not mentioned"));
}

#[test]
fn malformed_input_errors_instead_of_fabricating() {
    assert!(load_longmemeval_str("not json").is_err());
    assert!(load_locomo_str("not json").is_err());
    // Structurally valid JSON but wrong shape must also fail.
    assert!(load_longmemeval_str("[{\"foo\": 1}]").is_err());
    assert!(load_locomo_str("[{\"foo\": 1}]").is_err());
    // Unknown LongMemEval question_type must be rejected, not guessed.
    let bad_qtype = r#"[{
        "question_id": "x", "question_type": "made-up-type",
        "question": "q", "answer": "a",
        "haystack_dates": [], "haystack_session_ids": [],
        "haystack_sessions": [], "answer_session_ids": []
    }]"#;
    assert!(load_longmemeval_str(bad_qtype).is_err());
}
