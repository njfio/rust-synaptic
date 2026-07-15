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
//! ## Judge selection (`SYNAPTIC_EVAL_JUDGE`)
//!
//! [`run_qa_gated`] picks the judge based on `SYNAPTIC_EVAL_JUDGE`:
//!
//! - `codex` — use [`CodexCliJudge`], which shells out to a locally
//!   installed, logged-in `codex` CLI (`codex exec`). No feature flag or
//!   network crate required; works in the default build. Configure the
//!   binary with `SYNAPTIC_EVAL_CODEX_BIN` (default `codex`) and the
//!   per-call timeout with `SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS` (default
//!   120s). If the CLI is not runnable this returns [`QaResult::NotRun`]
//!   rather than fabricating a number.
//! - `llm` or unset — the existing behavior: with the `llm-reasoning`
//!   feature enabled and an endpoint configured, uses [`LlmJudge`];
//!   otherwise [`QaResult::NotRun`].
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

/// Environment variable selecting which judge [`run_qa_gated`] uses:
/// `codex` for [`CodexCliJudge`], `llm` (or unset) for [`LlmJudge`] /
/// existing behavior.
pub const ENV_JUDGE: &str = "SYNAPTIC_EVAL_JUDGE";

/// If `SYNAPTIC_EVAL_JUDGE=codex`, run QA with [`CodexCliJudge`] and return
/// `Some(result)` (a real report if the CLI is available, `NotRun`
/// otherwise). Returns `None` when a different judge should be selected
/// (env var unset or set to something else), so callers fall through to
/// their existing behavior.
async fn run_qa_with_codex_if_selected(
    conversations: &[EvalConversation],
    k: usize,
) -> Option<Result<QaResult, QaError>> {
    let selected = std::env::var(ENV_JUDGE).ok()?;
    if selected.trim() != "codex" {
        return None;
    }
    let judge = CodexCliJudge::from_env();
    if !judge.is_available() {
        return Some(Ok(QaResult::NotRun {
            reason: format!(
                "SYNAPTIC_EVAL_JUDGE=codex but codex CLI not runnable — \
                 install/login the codex CLI or set {}",
                CodexCliJudge::ENV_BIN
            ),
        }));
    }
    Some(run_qa(conversations, &judge, k).await.map(QaResult::Ran))
}

/// Gated QA evaluation. Honors `SYNAPTIC_EVAL_JUDGE=codex` (see module
/// docs) first; otherwise, with the `llm-reasoning` feature enabled and an
/// endpoint configured, runs [`run_qa`] with the real [`LlmJudge`].
/// Otherwise returns [`QaResult::NotRun`] with the reason — QA accuracy is
/// never fabricated.
#[cfg(feature = "llm-reasoning")]
pub async fn run_qa_gated(
    conversations: &[EvalConversation],
    k: usize,
) -> Result<QaResult, QaError> {
    if let Some(result) = run_qa_with_codex_if_selected(conversations, k).await {
        return result;
    }
    match LlmJudge::from_env()? {
        Some(judge) => Ok(QaResult::Ran(run_qa(conversations, &judge, k).await?)),
        None => Ok(QaResult::NotRun {
            reason: "not run — requires endpoint: set SYNAPTIC_EVAL_LLM_URL (and \
                     SYNAPTIC_EVAL_LLM_MODEL) to an OpenAI-compatible endpoint, \
                     or set SYNAPTIC_EVAL_JUDGE=codex with the codex CLI installed"
                .to_string(),
        }),
    }
}

/// Gated QA evaluation (feature off). Honors `SYNAPTIC_EVAL_JUDGE=codex`
/// (see module docs); otherwise always [`QaResult::NotRun`].
#[cfg(not(feature = "llm-reasoning"))]
pub async fn run_qa_gated(
    conversations: &[EvalConversation],
    k: usize,
) -> Result<QaResult, QaError> {
    if let Some(result) = run_qa_with_codex_if_selected(conversations, k).await {
        return result;
    }
    Ok(QaResult::NotRun {
        reason: "not run — requires endpoint: build with feature `llm-reasoning` and \
                 set SYNAPTIC_EVAL_LLM_URL / SYNAPTIC_EVAL_LLM_MODEL, or set \
                 SYNAPTIC_EVAL_JUDGE=codex with the codex CLI installed"
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

// ---------------------------------------------------------------------------
// Codex CLI judge
// ---------------------------------------------------------------------------

/// Extract the model's answer from `codex exec` stdout.
///
/// codex 0.144.0 prints just the final answer on stdout (banner/progress go
/// to stderr), but this parser is defensive against the alternate format
/// where stdout carries a banner, a `codex` marker line, the answer, and a
/// `tokens used` trailer. Returns `None` when no answer text remains —
/// callers must treat that as an error (fail closed).
pub fn parse_codex_stdout(stdout: &str) -> Option<String> {
    let lines: Vec<&str> = stdout.lines().collect();
    // If a `codex` marker line exists, the answer is what follows the last one.
    let start = lines
        .iter()
        .rposition(|l| {
            let t = l.trim();
            t == "codex" || t.ends_with("] codex")
        })
        .map(|i| i + 1)
        .unwrap_or(0);
    let is_banner = |t: &str| {
        t.starts_with("OpenAI Codex")
            || t.starts_with("--------")
            || t.starts_with("workdir:")
            || t.starts_with("model:")
            || t.starts_with("provider:")
            || t.starts_with("approval:")
            || t.starts_with("sandbox:")
            || t.starts_with("reasoning ")
            || t.starts_with("session id:")
            || t.contains("tokens used")
    };
    let answer = lines[start..]
        .iter()
        .filter(|l| !is_banner(l.trim()))
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();
    if answer.is_empty() {
        None
    } else {
        Some(answer)
    }
}

/// Parse a YES/NO grading verdict out of codex stdout. `None` when the reply
/// is not clearly YES or NO — callers must fail closed, never guess.
pub fn parse_grade_verdict(stdout: &str) -> Option<bool> {
    let text = parse_codex_stdout(stdout)?;
    let word = text
        .split_whitespace()
        .last()?
        .trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_ascii_uppercase();
    match word.as_str() {
        "YES" => Some(true),
        "NO" => Some(false),
        _ => None,
    }
}

/// Judge that shells out to the `codex` CLI (`codex exec`) for answer
/// generation and grading. No network crate dependency and no feature gate;
/// requires a locally installed, logged-in codex CLI. Fail-closed: any spawn
/// failure, non-zero exit, timeout, or unparseable output is a
/// [`QaError::Judge`].
pub struct CodexCliJudge {
    bin: String,
    timeout_secs: u64,
}

impl Default for CodexCliJudge {
    fn default() -> Self {
        Self::from_env()
    }
}

impl CodexCliJudge {
    /// Env var overriding the codex binary path (default `codex`).
    pub const ENV_BIN: &'static str = "SYNAPTIC_EVAL_CODEX_BIN";
    /// Env var overriding the per-call timeout in seconds (default 120).
    pub const ENV_TIMEOUT: &'static str = "SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS";

    /// Build from environment (`SYNAPTIC_EVAL_CODEX_BIN`,
    /// `SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS`), with defaults `codex` / 120 s.
    pub fn from_env() -> Self {
        let bin = std::env::var(Self::ENV_BIN)
            .ok()
            .filter(|b| !b.trim().is_empty())
            .unwrap_or_else(|| "codex".to_string());
        let timeout_secs = std::env::var(Self::ENV_TIMEOUT)
            .ok()
            .and_then(|t| t.trim().parse().ok())
            .unwrap_or(120);
        Self { bin, timeout_secs }
    }

    /// Check that the codex CLI is runnable (`codex --version` exits 0).
    pub fn is_available(&self) -> bool {
        std::process::Command::new(&self.bin)
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Run one `codex exec` call with `prompt` and return parsed stdout.
    /// Runs on a blocking thread; enforces the timeout via the coreutils
    /// `timeout` wrapper (exit 124 → timeout error).
    async fn run_codex(&self, prompt: String) -> Result<String, QaError> {
        let bin = self.bin.clone();
        let secs = self.timeout_secs;
        let output = tokio::task::spawn_blocking(move || {
            std::process::Command::new("timeout")
                .arg(secs.to_string())
                .arg(&bin)
                .args(["exec", "--skip-git-repo-check", &prompt])
                .stdin(std::process::Stdio::null())
                .output()
        })
        .await
        .map_err(|e| QaError::Judge(format!("codex task join failed: {e}")))?
        .map_err(|e| QaError::Judge(format!("failed to spawn codex CLI: {e}")))?;

        if output.status.code() == Some(124) {
            return Err(QaError::Judge(format!(
                "codex exec timed out after {secs}s"
            )));
        }
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: String = stderr.chars().rev().take(400).collect::<String>();
            let tail: String = tail.chars().rev().collect();
            return Err(QaError::Judge(format!(
                "codex exec exited with {}: {tail}",
                output.status
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

#[async_trait::async_trait]
impl Judge for CodexCliJudge {
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
        let prompt = format!(
            "Answer concisely using ONLY the provided memories; if not \
             answerable from them, say 'I don't know'. Do not use any other \
             knowledge or tools.\n\nMemories:\n{context}\n\nQuestion: \
             {question}\nAnswer:"
        );
        let stdout = self.run_codex(prompt).await?;
        parse_codex_stdout(&stdout)
            .ok_or_else(|| QaError::Judge("codex exec produced no parseable answer".to_string()))
    }

    async fn grade(&self, question: &str, gold: &str, predicted: &str) -> Result<bool, QaError> {
        let prompt = format!(
            "Question: {question}\nGold answer: {gold}\nPredicted answer: \
             {predicted}\n\nDoes the predicted answer match the gold answer \
             in meaning? Reply exactly YES or NO."
        );
        let stdout = self.run_codex(prompt).await?;
        parse_grade_verdict(&stdout).ok_or_else(|| {
            QaError::Judge(format!(
                "unparseable YES/NO grading verdict from codex: {:?}",
                stdout.trim()
            ))
        })
    }
}

#[cfg(test)]
mod run_qa_tests {
    use super::*;
    use crate::dataset::{EvalConversation, EvalQuestion, QType, Session};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Deterministic judge for testing aggregation without a network or CLI.
    /// Grades correct on odd calls, incorrect on even calls (0-indexed).
    struct FakeJudge {
        call: AtomicUsize,
    }

    impl FakeJudge {
        fn new() -> Self {
            Self {
                call: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl Judge for FakeJudge {
        async fn answer(&self, _question: &str, _recalled: &[String]) -> Result<String, QaError> {
            Ok("fake answer".to_string())
        }

        async fn grade(
            &self,
            _question: &str,
            _gold: &str,
            _predicted: &str,
        ) -> Result<bool, QaError> {
            let n = self.call.fetch_add(1, Ordering::SeqCst);
            Ok(n % 2 == 0)
        }
    }

    fn fixture() -> Vec<EvalConversation> {
        vec![EvalConversation {
            id: "conv1".to_string(),
            sessions: vec![Session {
                id: "s1".to_string(),
                timestamp: None,
                turns: vec![],
            }],
            questions: vec![
                EvalQuestion {
                    id: "q1".to_string(),
                    text: "What color is the sky?".to_string(),
                    qtype: QType::SingleHop,
                    evidence_ids: vec![],
                    gold_answer: "blue".to_string(),
                },
                EvalQuestion {
                    id: "q2".to_string(),
                    text: "What color is grass?".to_string(),
                    qtype: QType::SingleHop,
                    evidence_ids: vec![],
                    gold_answer: "green".to_string(),
                },
            ],
        }]
    }

    #[tokio::test]
    async fn run_qa_aggregates_accuracy_over_fixture() {
        let conversations = fixture();
        let judge = FakeJudge::new();
        let report = run_qa(&conversations, &judge, 5)
            .await
            .expect("run_qa should succeed with a fake judge");

        assert_eq!(report.questions, 2);
        assert_eq!(report.correct, 1);
        assert!((report.accuracy - 0.5).abs() < f64::EPSILON);

        let bucket = report
            .by_qtype
            .get(&qtype_key(QType::SingleHop))
            .expect("SingleHop bucket present");
        assert_eq!(bucket.questions, 2);
        assert_eq!(bucket.correct, 1);
        assert!((bucket.accuracy - 0.5).abs() < f64::EPSILON);
    }

    #[cfg(not(feature = "llm-reasoning"))]
    #[tokio::test]
    async fn run_qa_gated_not_run_when_judge_env_unset() {
        // Guard against test-order env races on this process-wide var.
        std::env::remove_var(ENV_JUDGE);
        let conversations = fixture();
        let result = run_qa_gated(&conversations, 5)
            .await
            .expect("run_qa_gated should not error when no judge is configured");
        match result {
            QaResult::NotRun { reason } => {
                assert!(reason.contains("SYNAPTIC_EVAL_JUDGE=codex"));
            }
            QaResult::Ran(_) => panic!("expected NotRun with no judge configured"),
        }
    }
}

#[cfg(test)]
mod codex_parse_tests {
    use super::{parse_codex_stdout, parse_grade_verdict};

    #[test]
    fn parses_plain_stdout_answer() {
        // Real captured behavior of codex 0.144.0: stdout is just the answer.
        assert_eq!(parse_codex_stdout("4\n").as_deref(), Some("4"));
    }

    #[test]
    fn parses_marker_and_banner_format() {
        // Defensive: older/alternate format with banner + `codex` marker +
        // tokens-used trailer on stdout.
        let sample = "\
OpenAI Codex v0.144.0
--------
workdir: /tmp/x
model: gpt-5.6-sol
provider: openai
approval: never
sandbox: read-only
--------
[2026-07-13T01:02:03] User instructions:
Some prompt text
[2026-07-13T01:02:10] thinking
Working it out
[2026-07-13T01:02:12] codex
7 May 2023
[2026-07-13T01:02:12] tokens used: 1234
";
        assert_eq!(parse_codex_stdout(sample).as_deref(), Some("7 May 2023"));
    }

    #[test]
    fn empty_stdout_is_none() {
        assert_eq!(parse_codex_stdout("  \n"), None);
        assert_eq!(parse_codex_stdout(""), None);
    }

    #[test]
    fn multiline_answer_after_marker_is_kept() {
        let sample = "[t] codex\nline one\nline two\n[t] tokens used: 9\n";
        assert_eq!(
            parse_codex_stdout(sample).as_deref(),
            Some("line one\nline two")
        );
    }

    #[test]
    fn grade_verdict_yes_no() {
        assert_eq!(parse_grade_verdict("YES\n"), Some(true));
        assert_eq!(parse_grade_verdict("yes."), Some(true));
        assert_eq!(parse_grade_verdict("NO"), Some(false));
        assert_eq!(
            parse_grade_verdict("[t] codex\nNO\n[t] tokens used: 3"),
            Some(false)
        );
        assert_eq!(parse_grade_verdict("maybe"), None);
        assert_eq!(parse_grade_verdict(""), None);
    }
}
