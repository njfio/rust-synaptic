//! QA-pipeline tests: prove end-to-end accuracy is computed from real judge
//! outputs (mock judge, no network), and that the gated entry point reports
//! an explicit not-run marker — never a fabricated number — when no LLM
//! endpoint is available.

use std::collections::HashSet;
use synaptic_eval::dataset::{EvalConversation, EvalQuestion, QType, Session, Turn};
use synaptic_eval::qa::{run_qa, run_qa_gated, Judge, QaError, QaResult};

/// Deterministic, network-free judge for pipeline tests. `answer` echoes the
/// top recalled memory; `grade` returns a fixed verdict.
struct MockJudge {
    verdict: bool,
}

#[async_trait::async_trait]
impl Judge for MockJudge {
    async fn answer(&self, _question: &str, recalled: &[String]) -> Result<String, QaError> {
        Ok(recalled.first().cloned().unwrap_or_default())
    }

    async fn grade(&self, _question: &str, _gold: &str, _predicted: &str) -> Result<bool, QaError> {
        Ok(self.verdict)
    }
}

/// A judge whose verdict depends on the question, to pin per-QType
/// aggregation: correct iff the question mentions "city".
struct SelectiveJudge;

#[async_trait::async_trait]
impl Judge for SelectiveJudge {
    async fn answer(&self, question: &str, _recalled: &[String]) -> Result<String, QaError> {
        Ok(format!("answer to: {question}"))
    }

    async fn grade(&self, question: &str, _gold: &str, _predicted: &str) -> Result<bool, QaError> {
        Ok(question.contains("city"))
    }
}

/// A judge that always fails, to pin fail-closed behavior.
struct FailingJudge;

#[async_trait::async_trait]
impl Judge for FailingJudge {
    async fn answer(&self, _question: &str, _recalled: &[String]) -> Result<String, QaError> {
        Err(QaError::Judge("endpoint unreachable".to_string()))
    }

    async fn grade(&self, _question: &str, _gold: &str, _predicted: &str) -> Result<bool, QaError> {
        Err(QaError::Judge("endpoint unreachable".to_string()))
    }
}

fn fixture_conversation() -> EvalConversation {
    EvalConversation {
        id: "qa_fixture".to_string(),
        sessions: vec![Session {
            id: "qa_sess_1".to_string(),
            timestamp: Some("2023/05/28 (Sun) 14:05".to_string()),
            turns: vec![
                Turn {
                    speaker: "user".to_string(),
                    text: "I live in Lisbon now.".to_string(),
                    timestamp: None,
                },
                Turn {
                    speaker: "user".to_string(),
                    text: "My dog is named Bruno.".to_string(),
                    timestamp: None,
                },
            ],
        }],
        questions: vec![
            EvalQuestion {
                id: "qa_q1".to_string(),
                text: "What city does the user live in?".to_string(),
                qtype: QType::SingleHop,
                evidence_ids: vec!["qa_sess_1".to_string()],
                gold_answer: "Lisbon".to_string(),
            },
            EvalQuestion {
                id: "qa_q2".to_string(),
                text: "What is the dog's name?".to_string(),
                qtype: QType::MultiHop,
                evidence_ids: vec!["qa_sess_1".to_string()],
                gold_answer: "Bruno".to_string(),
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Pipeline computes real accuracy from judge outputs (mock, end-to-end).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn always_correct_judge_yields_accuracy_one() {
    let convs = vec![fixture_conversation()];
    let report = run_qa(&convs, &MockJudge { verdict: true }, 5)
        .await
        .expect("qa run");
    assert_eq!(report.questions, 2);
    assert_eq!(report.correct, 2);
    assert_eq!(report.accuracy, 1.0);
    // The predicted answer must come from the judge over real recalled text.
    assert!(report.per_question.iter().all(|q| !q.predicted.is_empty()));
}

#[tokio::test]
async fn always_wrong_judge_yields_accuracy_zero() {
    let convs = vec![fixture_conversation()];
    let report = run_qa(&convs, &MockJudge { verdict: false }, 5)
        .await
        .expect("qa run");
    assert_eq!(report.questions, 2);
    assert_eq!(report.correct, 0);
    assert_eq!(report.accuracy, 0.0);
}

#[tokio::test]
async fn per_qtype_accuracy_is_aggregated_from_verdicts() {
    let convs = vec![fixture_conversation()];
    let report = run_qa(&convs, &SelectiveJudge, 5).await.expect("qa run");
    // "city" question correct, dog question wrong → 0.5 overall.
    assert_eq!(report.correct, 1);
    assert!((report.accuracy - 0.5).abs() < 1e-12);
    let single = &report.by_qtype["SingleHop"];
    assert_eq!((single.questions, single.correct), (1, 1));
    assert_eq!(single.accuracy, 1.0);
    let multi = &report.by_qtype["MultiHop"];
    assert_eq!((multi.questions, multi.correct), (1, 0));
    assert_eq!(multi.accuracy, 0.0);
    let qtypes: HashSet<&str> = report.by_qtype.keys().map(String::as_str).collect();
    assert_eq!(qtypes, HashSet::from(["SingleHop", "MultiHop"]));
}

// ---------------------------------------------------------------------------
// Fail closed: a failing judge aborts the run with Err — no partial or
// fabricated accuracy.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn failing_judge_fails_closed() {
    let convs = vec![fixture_conversation()];
    let err = run_qa(&convs, &FailingJudge, 5)
        .await
        .expect_err("must fail");
    assert!(err.to_string().contains("endpoint unreachable"));
}

// ---------------------------------------------------------------------------
// Gated entry point: no endpoint (or feature off) → explicit NotRun marker,
// not a 0.0 masquerading as a measured accuracy.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn gated_run_without_endpoint_is_marked_not_run() {
    // Ensure a stray environment does not turn this into a live call.
    std::env::remove_var("SYNAPTIC_EVAL_LLM_URL");
    let convs = vec![fixture_conversation()];
    let result = run_qa_gated(&convs, 5).await.expect("gated run");
    match result {
        QaResult::NotRun { reason } => {
            assert!(
                reason.contains("endpoint") || reason.contains("llm-reasoning"),
                "reason must explain why QA did not run, got: {reason}"
            );
        }
        QaResult::Ran(report) => panic!(
            "expected NotRun marker without an endpoint, got a report with accuracy {}",
            report.accuracy
        ),
    }
}
