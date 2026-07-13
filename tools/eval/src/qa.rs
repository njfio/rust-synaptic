//! LLM-gated end-to-end QA accuracy with a pluggable [`Judge`].
//!
//! For each benchmark question the pipeline recalls memories via
//! `AgentMemory::search`, asks the judge to generate an answer from the
//! recalled text, then asks the judge to grade that answer against the gold
//! answer, and aggregates accuracy overall and per [`QType`].
//!
//! ## Honesty contract
//!
//! - Accuracy is only ever computed from real judge verdicts. There is no
//!   heuristic fallback, no simulated grading, and no default score.
//! - The gated entry point ([`run_qa_gated`]) returns [`QaResult::NotRun`]
//!   with a human-readable reason when the `llm-reasoning` feature is off or
//!   no endpoint is configured — an explicit marker, never a `0.0`
//!   masquerading as a measurement.
//! - A judge failure mid-run (endpoint unreachable, malformed reply) aborts
//!   the run with an error: fail closed, no partial numbers.
//!
//! ## Real judge configuration (`llm-reasoning` feature)
//!
//! [`LlmJudge`] talks to any OpenAI-compatible chat-completions endpoint
//! (including Ollama's `/v1` compatibility layer) via `reqwest`:
//!
//! - `SYNAPTIC_EVAL_LLM_URL` — endpoint base, e.g.
//!   `https://api.openai.com/v1` or `http://localhost:11434/v1`.
//!   `/chat/completions` is appended unless already present.
//! - `SYNAPTIC_EVAL_LLM_MODEL` — model name (required when URL is set).
//! - `SYNAPTIC_EVAL_LLM_KEY` — bearer token (optional; omit for Ollama).

use crate::dataset::{EvalConversation, QType};
use crate::runner::ingest;
use std::collections::BTreeMap;
use synaptic::{AgentMemory, MemoryConfig};

/// Errors from the QA pipeline or a judge.
#[derive(Debug, thiserror::Error)]
pub enum QaError {
    /// The underlying memory system returned an error.
    #[error("memory operation failed: {0}")]
    Memory(String),
    /// The judge failed (endpoint unreachable, bad reply, misconfiguration).
    #[error("judge failed: {0}")]
    Judge(String),
}

/// Answer generation and grading, pluggable so the pipeline can be tested
/// deterministically without a network.
#[async_trait::async_trait]
pub trait Judge: Send + Sync {
    /// Generate an answer to `question` from the `recalled` memory texts.
    async fn answer(&self, question: &str, recalled: &[String]) -> Result<String, QaError>;

    /// Judge whether `predicted` correctly answers `question` given `gold`.
    async fn grade(&self, question: &str, gold: &str, predicted: &str) -> Result<bool, QaError>;
}

/// Accuracy over one question category.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QTypeAccuracy {
    /// Questions of this category that were graded.
    pub questions: usize,
    /// Of those, how many the judge graded correct.
    pub correct: usize,
    /// `correct / questions`.
    pub accuracy: f64,
}

/// Per-question QA record (all values come from this run's judge).
#[derive(Debug, Clone, serde::Serialize)]
pub struct QaQuestionRecord {
    /// Dataset-native question id.
    pub question_id: String,
    /// Debug-formatted [`QType`].
    pub qtype: String,
    /// The judge-generated answer.
    pub predicted: String,
    /// The judge's verdict against the gold answer.
    pub correct: bool,
}

/// Measured end-to-end QA accuracy report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QaReport {
    /// Retrieval depth used for recall.
    pub k: usize,
    /// Total questions graded.
    pub questions: usize,
    /// Questions the judge graded correct.
    pub correct: usize,
    /// Overall end-to-end accuracy (`correct / questions`; 0.0 only when
    /// `questions > 0` and none were correct — an empty run is reported as
    /// [`QaResult::NotRun`] by the gated entry point, not as 0.0).
    pub accuracy: f64,
    /// Accuracy broken down by question category.
    pub by_qtype: BTreeMap<String, QTypeAccuracy>,
    /// Per-question records.
    pub per_question: Vec<QaQuestionRecord>,
}

/// Outcome of the gated QA evaluation: either a real measured report or an
/// explicit not-run marker. Never a fabricated number.
#[derive(Debug, Clone, serde::Serialize)]
pub enum QaResult {
    /// QA accuracy was not measured; `reason` says why (feature off, no
    /// endpoint configured).
    NotRun {
        /// Why QA did not run.
        reason: String,
    },
    /// QA ran against a real judge; the report contains measured numbers.
    Ran(QaReport),
}

fn mem_err(e: impl std::fmt::Display) -> QaError {
    QaError::Memory(e.to_string())
}

fn qtype_key(qtype: QType) -> String {
    format!("{qtype:?}")
}

/// Run end-to-end QA over `conversations` with `judge`: per question, recall
/// up to `k` memories, generate an answer, grade it against the gold answer,
/// and aggregate accuracy. Any judge failure aborts the run (fail closed).
pub async fn run_qa(
    conversations: &[EvalConversation],
    judge: &dyn Judge,
    k: usize,
) -> Result<QaReport, QaError> {
    let mut per_question: Vec<QaQuestionRecord> = Vec::new();
    let mut by_qtype: BTreeMap<String, QTypeAccuracy> = BTreeMap::new();

    for conversation in conversations {
        let mut memory = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(mem_err)?;
        ingest(conversation, &mut memory)
            .await
            .map_err(|e| QaError::Memory(e.to_string()))?;

        for question in &conversation.questions {
            let recalled: Vec<String> = memory
                .search(&question.text, k)
                .await
                .map_err(mem_err)?
                .into_iter()
                .map(|f| f.entry.value)
                .collect();
            let predicted = judge.answer(&question.text, &recalled).await?;
            let correct = judge
                .grade(&question.text, &question.gold_answer, &predicted)
                .await?;

            let bucket = by_qtype
                .entry(qtype_key(question.qtype))
                .or_insert(QTypeAccuracy {
                    questions: 0,
                    correct: 0,
                    accuracy: 0.0,
                });
            bucket.questions += 1;
            bucket.correct += usize::from(correct);

            per_question.push(QaQuestionRecord {
                question_id: question.id.clone(),
                qtype: qtype_key(question.qtype),
                predicted,
                correct,
            });
        }
    }

    for bucket in by_qtype.values_mut() {
        bucket.accuracy = if bucket.questions == 0 {
            0.0
        } else {
            bucket.correct as f64 / bucket.questions as f64
        };
    }
    let questions = per_question.len();
    let correct = per_question.iter().filter(|q| q.correct).count();
    Ok(QaReport {
        k,
        questions,
        correct,
        accuracy: if questions == 0 {
            0.0
        } else {
            correct as f64 / questions as f64
        },
        by_qtype,
        per_question,
    })
}

/// Gated QA evaluation. With the `llm-reasoning` feature enabled and an
/// endpoint configured (see module docs for the env vars), runs [`run_qa`]
/// with the real [`LlmJudge`]. Otherwise returns [`QaResult::NotRun`] with
/// the reason — QA accuracy is never fabricated.
#[cfg(feature = "llm-reasoning")]
pub async fn run_qa_gated(
    conversations: &[EvalConversation],
    k: usize,
) -> Result<QaResult, QaError> {
    match LlmJudge::from_env()? {
        Some(judge) => Ok(QaResult::Ran(run_qa(conversations, &judge, k).await?)),
        None => Ok(QaResult::NotRun {
            reason: "not run — requires endpoint: set SYNAPTIC_EVAL_LLM_URL (and \
                     SYNAPTIC_EVAL_LLM_MODEL) to an OpenAI-compatible endpoint"
                .to_string(),
        }),
    }
}

/// Gated QA evaluation (feature off): always [`QaResult::NotRun`].
#[cfg(not(feature = "llm-reasoning"))]
pub async fn run_qa_gated(
    _conversations: &[EvalConversation],
    _k: usize,
) -> Result<QaResult, QaError> {
    Ok(QaResult::NotRun {
        reason: "not run — requires endpoint: build with feature `llm-reasoning` and \
                 set SYNAPTIC_EVAL_LLM_URL / SYNAPTIC_EVAL_LLM_MODEL"
            .to_string(),
    })
}

/// Real LLM judge over an OpenAI-compatible chat-completions endpoint.
///
/// Fail-closed: construction requires explicit configuration, and any HTTP
/// or parse failure surfaces as [`QaError::Judge`] — nothing is guessed.
#[cfg(feature = "llm-reasoning")]
pub struct LlmJudge {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
}

#[cfg(feature = "llm-reasoning")]
impl LlmJudge {
    /// Environment variable holding the endpoint base URL.
    pub const ENV_URL: &'static str = "SYNAPTIC_EVAL_LLM_URL";
    /// Environment variable holding the model name.
    pub const ENV_MODEL: &'static str = "SYNAPTIC_EVAL_LLM_MODEL";
    /// Environment variable holding the optional bearer token.
    pub const ENV_KEY: &'static str = "SYNAPTIC_EVAL_LLM_KEY";

    /// Build from environment variables. `Ok(None)` when no endpoint is
    /// configured (the caller reports not-run); `Err` when the configuration
    /// is present but incomplete (URL without model) — fail closed rather
    /// than guessing a model.
    pub fn from_env() -> Result<Option<Self>, QaError> {
        let url = match std::env::var(Self::ENV_URL) {
            Ok(u) if !u.trim().is_empty() => u.trim().to_string(),
            _ => return Ok(None),
        };
        let model = std::env::var(Self::ENV_MODEL)
            .ok()
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| {
                QaError::Judge(format!(
                    "{} is set but {} is not — refusing to guess a model",
                    Self::ENV_URL,
                    Self::ENV_MODEL
                ))
            })?;
        let api_key = std::env::var(Self::ENV_KEY)
            .ok()
            .filter(|k| !k.trim().is_empty());
        Ok(Some(Self {
            client: reqwest::Client::new(),
            url,
            model: model.trim().to_string(),
            api_key,
        }))
    }

    fn chat_completions_url(&self) -> String {
        let base = self.url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }

    /// One chat call; returns the assistant message content or a
    /// [`QaError::Judge`] on any transport/shape failure.
    async fn chat(&self, system: &str, user: &str) -> Result<String, QaError> {
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        let mut req = self.client.post(self.chat_completions_url()).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| QaError::Judge(format!("request to LLM endpoint failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(QaError::Judge(format!(
                "LLM endpoint returned {status}: {text}"
            )));
        }
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| QaError::Judge(format!("LLM endpoint returned invalid JSON: {e}")))?;
        json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| {
                QaError::Judge("LLM reply missing choices[0].message.content".to_string())
            })
    }
}

#[cfg(feature = "llm-reasoning")]
#[async_trait::async_trait]
impl Judge for LlmJudge {
    async fn answer(&self, question: &str, recalled: &[String]) -> Result<String, QaError> {
        let context = if recalled.is_empty() {
            "(no memories recalled)".to_string()
        } else {
            recalled
                .iter()
                .enumerate()
                .map(|(i, m)| format!("[{}] {m}", i + 1))
                .collect::<Vec<_>>()
                .join("\n")
        };
        self.chat(
            "You answer questions using ONLY the recalled memories provided. \
             Answer concisely. If the memories do not contain the answer, say \
             you don't know.",
            &format!("Recalled memories:\n{context}\n\nQuestion: {question}\nAnswer:"),
        )
        .await
    }

    async fn grade(&self, question: &str, gold: &str, predicted: &str) -> Result<bool, QaError> {
        let verdict = self
            .chat(
                "You are grading a QA system. Given the question, the gold answer, \
                 and the predicted answer, reply with exactly one word: CORRECT if \
                 the prediction conveys the gold answer, or INCORRECT otherwise.",
                &format!(
                    "Question: {question}\nGold answer: {gold}\nPredicted answer: \
                     {predicted}\nVerdict (CORRECT or INCORRECT):"
                ),
            )
            .await?;
        let normalized = verdict.trim().trim_end_matches('.').to_ascii_uppercase();
        match normalized.as_str() {
            "CORRECT" => Ok(true),
            "INCORRECT" => Ok(false),
            other => Err(QaError::Judge(format!(
                "unparseable grading verdict from LLM: {other:?}"
            ))),
        }
    }
}
