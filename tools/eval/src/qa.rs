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

use crate::dataset::{EvalConversation, EvalQuestion, QType};
use crate::runner::{ingest, normalize_retrieved};
use futures::stream::{self, StreamExt};
use std::collections::{BTreeMap, HashSet};
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

/// Phrases (matched case-insensitively as substrings of the predicted
/// answer) that indicate the judge abstained — i.e. said "I don't know"
/// rather than confabulating an answer. Kept as a documented, reviewable
/// list rather than a heuristic classifier.
///
/// Known limitation: substring matching is intentionally simple and can
/// false-positive on answers that happen to contain one of these phrases as
/// a sub-string of a longer, confident answer (e.g. an answer mentioning an
/// "unknown suspect" by name, or a movie titled "None the Wiser"). This is
/// an accepted tradeoff for a reviewable, dependency-free classifier; the
/// phrases were chosen to be as specific as practical while still catching
/// real refusals phrased in different ways.
pub const ABSTENTION_PHRASES: &[&str] = &[
    "i don't know",
    "i do not know",
    "no information",
    "not mentioned",
    "not available",
    "cannot determine",
    "can't determine",
    "no answer",
    "not enough information",
    "not specified",
    "no relevant",
    "isn't mentioned",
    "doesn't say",
    "does not say",
    "unknown",
    "none",
];

/// Classify `answer` as an abstention (a refusal / "I don't know" response)
/// vs. a confident answer, by case-insensitive substring match against
/// [`ABSTENTION_PHRASES`]. See that constant's docs for the known
/// false-positive limitation of substring matching.
pub fn is_abstention(answer: &str) -> bool {
    let lower = answer.to_lowercase();
    ABSTENTION_PHRASES
        .iter()
        .any(|phrase| lower.contains(phrase))
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
    /// Whether ANY of the question's `evidence_ids` appeared in the top-`k`
    /// recalled set (normalized into evidence-id space the same way the
    /// retrieval eval does, via [`normalize_retrieved`]). `false` for
    /// questions with empty `evidence_ids` (e.g. some Abstention questions
    /// have no labeled gold evidence) — those are excluded from the
    /// recall/judge cross-tab denominator and counted separately as
    /// `no_gold_labeled` (see [`QaRecallBreakdown`]).
    pub gold_retrieved: bool,
    /// Whether this question had any `evidence_ids` labeled at all. When
    /// `false`, `gold_retrieved` is definitionally `false` and this question
    /// is excluded from the A/B/C/D cross-tab.
    pub has_gold: bool,
    /// Whether `predicted` reads as a refusal / "I don't know" response
    /// (see [`is_abstention`]), rather than a confident answer.
    pub abstained: bool,
}

/// Cross-tab of gold-evidence recall vs. judge verdict, over the questions
/// that have at least one labeled `evidence_id` (`has_gold == true`).
/// Questions with no labeled evidence are counted in `no_gold_labeled` and
/// excluded from `a`/`b`/`c`/`d` so they don't distort the ratio.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct QaRecallBreakdown {
    /// gold_retrieved AND correct.
    pub a: usize,
    /// gold_retrieved AND wrong — JUDGE-bound loss.
    pub b: usize,
    /// NOT gold_retrieved AND correct.
    pub c: usize,
    /// NOT gold_retrieved AND wrong — RETRIEVAL-bound loss.
    pub d: usize,
    /// Questions excluded from a/b/c/d because they had no labeled
    /// `evidence_ids`.
    pub no_gold_labeled: usize,
}

impl QaRecallBreakdown {
    /// Compute the cross-tab from a set of per-question records.
    pub fn from_records(records: &[QaQuestionRecord]) -> Self {
        let mut b = QaRecallBreakdown::default();
        for r in records {
            if !r.has_gold {
                b.no_gold_labeled += 1;
                continue;
            }
            match (r.gold_retrieved, r.correct) {
                (true, true) => b.a += 1,
                (true, false) => b.b += 1,
                (false, true) => b.c += 1,
                (false, false) => b.d += 1,
            }
        }
        b
    }

    /// Total questions in the cross-tab (`a + b + c + d`), excluding
    /// `no_gold_labeled`.
    pub fn total(&self) -> usize {
        self.a + self.b + self.c + self.d
    }

    /// `(a + b) / total` — fraction of graded (has-gold) questions whose
    /// gold evidence was actually retrieved. `0.0` when `total() == 0`.
    pub fn gold_retrieved_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            (self.a + self.b) as f64 / total as f64
        }
    }

    /// Of the wrong answers (`b + d`), the fraction attributable to missed
    /// retrieval (`d`). `0.0` when there are no wrong answers.
    pub fn retrieval_bound_fraction(&self) -> f64 {
        let wrong = self.b + self.d;
        if wrong == 0 {
            0.0
        } else {
            self.d as f64 / wrong as f64
        }
    }

    /// Of the wrong answers (`b + d`), the fraction attributable to the
    /// judge failing despite having the gold evidence (`b`). `0.0` when
    /// there are no wrong answers.
    pub fn judge_bound_fraction(&self) -> f64 {
        let wrong = self.b + self.d;
        if wrong == 0 {
            0.0
        } else {
            self.b as f64 / wrong as f64
        }
    }
}

/// Faithfulness / confabulation-vs-abstention breakdown, computed from a set
/// of per-question records. Groundedness matters as much as recall: when the
/// system lacks the evidence, does it say "I don't know" (abstain) or
/// confabulate a confident wrong answer?
///
/// Three disjoint views over the records:
///
/// 1. Abstention-category questions (`qtype == "Abstention"`, LoCoMo's
///    unanswerable questions with gold answer "None") — the cleanest
///    confabulation test, since there is definitionally no evidence to find.
/// 2. Answerable (non-Abstention) questions where gold evidence was NOT
///    retrieved (`gold_retrieved == false`) — did the system honestly abstain
///    given missing evidence, or fabricate an answer?
/// 3. Answerable questions where gold evidence WAS retrieved
///    (`gold_retrieved == true`) but the system abstained anyway
///    (`over_abstention`) — a missed answerable question, the opposite
///    failure mode from confabulation.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct FaithfulnessBreakdown {
    /// Abstention-category questions, total.
    pub abstention_qtype_total: usize,
    /// Abstention-category questions the system correctly abstained on.
    pub abstention_qtype_abstained: usize,
    /// Abstention-category questions the system confabulated an answer for
    /// (unambiguous fabrication: there was no information to find).
    pub abstention_qtype_confabulated: usize,
    /// Answerable (non-Abstention) questions where gold evidence was not
    /// retrieved, total.
    pub evidence_missing_total: usize,
    /// Of those, how many the system honestly abstained on (faithful).
    pub evidence_missing_abstained: usize,
    /// Of those, how many the system answered wrong without evidence — a
    /// made-up wrong answer (confabulation).
    pub evidence_missing_confabulated: usize,
    /// Of those, how many the system answered correctly despite missing
    /// evidence (e.g. from world knowledge or context elsewhere) — not
    /// confabulation, tracked separately from the confabulated bucket.
    pub evidence_missing_answered_correct: usize,
    /// Answerable questions where gold evidence WAS retrieved but the system
    /// abstained anyway — a missed answerable question.
    pub over_abstention: usize,
}

impl FaithfulnessBreakdown {
    /// Compute the breakdown from a set of per-question records ([`QaQuestionRecord`]s).
    pub fn from_records(records: &[QaQuestionRecord]) -> Self {
        let mut b = FaithfulnessBreakdown::default();
        for r in records {
            if r.qtype == "Abstention" {
                b.abstention_qtype_total += 1;
                if r.abstained {
                    b.abstention_qtype_abstained += 1;
                } else {
                    b.abstention_qtype_confabulated += 1;
                }
                continue;
            }

            if r.gold_retrieved {
                if r.abstained {
                    b.over_abstention += 1;
                }
            } else {
                b.evidence_missing_total += 1;
                if r.abstained {
                    b.evidence_missing_abstained += 1;
                } else if r.correct {
                    b.evidence_missing_answered_correct += 1;
                } else {
                    b.evidence_missing_confabulated += 1;
                }
            }
        }
        b
    }
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
    /// Gold-retrieval-vs-judge-verdict cross-tab (see [`QaRecallBreakdown`]),
    /// separating retrieval-bound loss from judge-bound loss.
    pub recall_breakdown: QaRecallBreakdown,
    /// Confabulation-vs-abstention breakdown (see [`FaithfulnessBreakdown`]).
    pub faithfulness: FaithfulnessBreakdown,
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

/// Environment variable bounding how many judge (`answer`+`grade`) pipelines
/// run concurrently within a conversation. Parsed as `usize`, clamped to at
/// least 1; defaults to 4 when unset or unparseable.
pub const ENV_QA_CONCURRENCY: &str = "SYNAPTIC_EVAL_QA_CONCURRENCY";

/// Read [`ENV_QA_CONCURRENCY`], defaulting to 4 and clamping to at least 1.
fn qa_concurrency() -> usize {
    std::env::var(ENV_QA_CONCURRENCY)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(4)
        .max(1)
}

/// One question with its recalled memories, ready for the judge pipeline.
struct PendingQuestion {
    question_id: String,
    qtype: String,
    question_text: String,
    gold_answer: String,
    recalled: Vec<String>,
    gold_retrieved: bool,
    has_gold: bool,
}

/// Run `judge.answer` then `judge.grade` for one pending question.
async fn judge_one(judge: &dyn Judge, item: PendingQuestion) -> Result<QaQuestionRecord, QaError> {
    let predicted = judge.answer(&item.question_text, &item.recalled).await?;
    let correct = judge
        .grade(&item.question_text, &item.gold_answer, &predicted)
        .await?;
    let abstained = is_abstention(&predicted);
    Ok(QaQuestionRecord {
        question_id: item.question_id,
        qtype: item.qtype,
        predicted,
        correct,
        gold_retrieved: item.gold_retrieved,
        has_gold: item.has_gold,
        abstained,
    })
}

/// Run the `judge.answer` + `judge.grade` pipeline for `pending` with bounded
/// concurrency (limit `concurrency`, clamped to at least 1). Fail closed: as
/// soon as any task returns `Err`, that error is returned; the underlying
/// stream is not driven any further, so no partial report is produced.
async fn judge_questions(
    judge: &dyn Judge,
    pending: Vec<PendingQuestion>,
    concurrency: usize,
) -> Result<Vec<QaQuestionRecord>, QaError> {
    let concurrency = concurrency.max(1);
    let mut records = Vec::with_capacity(pending.len());
    let mut results = stream::iter(pending)
        .map(|item| judge_one(judge, item))
        .buffer_unordered(concurrency);
    while let Some(result) = results.next().await {
        records.push(result?);
    }
    Ok(records)
}

/// Run end-to-end QA over `conversations` with `judge`: per question, recall
/// up to `k` memories, generate an answer, grade it against the gold answer,
/// and aggregate accuracy. Any judge failure aborts the run (fail closed).
///
/// The `judge.answer` + `judge.grade` calls for a conversation's questions
/// run with bounded concurrency (see [`ENV_QA_CONCURRENCY`]); recall
/// (`memory.search`) stays sequential since it needs `&memory` and is fast.
/// Results are independent of the concurrency level: `per_question` is
/// sorted by `question_id` before returning, and the aggregate counts are
/// order-independent, so two runs over the same inputs produce identical
/// reports.
pub async fn run_qa(
    conversations: &[EvalConversation],
    judge: &dyn Judge,
    k: usize,
) -> Result<QaReport, QaError> {
    let concurrency = qa_concurrency();
    let mut per_question: Vec<QaQuestionRecord> = Vec::new();
    let mut by_qtype: BTreeMap<String, QTypeAccuracy> = BTreeMap::new();

    for conversation in conversations {
        let mut memory = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(mem_err)?;
        ingest(conversation, &mut memory)
            .await
            .map_err(|e| QaError::Memory(e.to_string()))?;

        let mut pending: Vec<PendingQuestion> = Vec::with_capacity(conversation.questions.len());
        for question in &conversation.questions {
            let fragments = memory.search(&question.text, k).await.map_err(mem_err)?;
            let keys: Vec<String> = fragments.iter().map(|f| f.entry.key.clone()).collect();
            let recalled: Vec<String> = fragments.into_iter().map(|f| f.entry.value).collect();

            // Same normalization the retrieval eval uses (`runner::run_question`),
            // so `gold_retrieved` is consistent with the recall@k numbers.
            let retrieved_ids = normalize_retrieved(&keys);
            let has_gold = !question.evidence_ids.is_empty();
            // Questions with no labeled evidence (e.g. some Abstention
            // questions) have no gold to retrieve; define gold_retrieved =
            // false for them and exclude them from the cross-tab denominator
            // via `has_gold` rather than letting them masquerade as misses.
            let gold_retrieved = has_gold
                && question
                    .evidence_ids
                    .iter()
                    .any(|id| retrieved_ids.contains(id));

            pending.push(PendingQuestion {
                question_id: question.id.clone(),
                qtype: qtype_key(question.qtype),
                question_text: question.text.clone(),
                gold_answer: question.gold_answer.clone(),
                recalled,
                gold_retrieved,
                has_gold,
            });
        }

        let records = judge_questions(judge, pending, concurrency).await?;
        for record in records {
            let bucket = by_qtype
                .entry(record.qtype.clone())
                .or_insert(QTypeAccuracy {
                    questions: 0,
                    correct: 0,
                    accuracy: 0.0,
                });
            bucket.questions += 1;
            bucket.correct += usize::from(record.correct);
            per_question.push(record);
        }
    }

    for bucket in by_qtype.values_mut() {
        bucket.accuracy = if bucket.questions == 0 {
            0.0
        } else {
            bucket.correct as f64 / bucket.questions as f64
        };
    }
    per_question.sort_by(|a, b| a.question_id.cmp(&b.question_id));
    let questions = per_question.len();
    let correct = per_question.iter().filter(|q| q.correct).count();
    let recall_breakdown = QaRecallBreakdown::from_records(&per_question);
    let faithfulness = FaithfulnessBreakdown::from_records(&per_question);
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
        recall_breakdown,
        faithfulness,
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

    /// Single free-form call to the codex CLI: run `prompt` and return the
    /// model's raw text reply (parsed the same way as `answer`/`grade`, but
    /// without imposing either shape). Used by the agentic answer-guided
    /// retrieval loop ([`run_agentic_qa`]), which needs unstructured
    /// `ANSWER:`/`SEARCH:` replies rather than the fixed answer/grade shapes
    /// of the [`Judge`] trait.
    pub async fn complete(&self, prompt: String) -> Result<String, QaError> {
        let stdout = self.run_codex(prompt).await?;
        parse_codex_stdout(&stdout)
            .ok_or_else(|| QaError::Judge("codex exec produced no parseable output".to_string()))
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

// ---------------------------------------------------------------------------
// Agentic answer-guided retrieval QA
// ---------------------------------------------------------------------------
//
// Instead of a single retrieve-then-answer pass ([`run_qa`]), the judge
// DRIVES retrieval: it reads whatever memories have been recalled so far and
// either answers, or names one specific missing fact to search for next.
// This targets multi-evidence questions where blind top-k retrieval alone
// misses supporting evidence that a reasoning model would know to go look
// for.
//
// Concurrency: conversations are processed sequentially (ingest dominates
// there), but the per-question agentic pipelines within each conversation run
// with bounded concurrency (`SYNAPTIC_EVAL_QA_CONCURRENCY`, default 4) via
// `buffer_unordered`, matching `run_qa`. Each question can issue several
// `codex exec` calls (one per round plus a grade call), so this bounding keeps
// a bounded sample tractable. `--agentic-qa` is still meant for small, bounded
// samples (see `--max-conversations` / `--max-questions`), not full sweeps.

/// Env var overriding the max rounds of the agentic retrieval loop (default
/// 3, i.e. up to 2 follow-up searches before a forced final answer).
pub const ENV_AGENTIC_ROUNDS: &str = "SYNAPTIC_EVAL_AGENTIC_ROUNDS";

/// Read [`ENV_AGENTIC_ROUNDS`], defaulting to 3 and clamping to at least 1.
pub fn agentic_max_rounds() -> usize {
    std::env::var(ENV_AGENTIC_ROUNDS)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

/// The judge's parsed intent for one round of the agentic loop.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AgenticStep {
    /// `ANSWER: <text>` — the judge considers the question answerable from
    /// the current context.
    Answer(String),
    /// `SEARCH: <text>` — the judge wants one more specific fact looked up.
    Search(String),
    /// Neither prefix was present. Fail-safe, not fail-closed: the raw text
    /// is used as a forced answer so the loop always terminates rather than
    /// erroring on a slightly malformed reply.
    Forced(String),
}

/// Parse the first line of a judge reply into an [`AgenticStep`]. Only the
/// first line is inspected, per the prompt contract (`build_agentic_prompt`)
/// that instructs the judge to reply with exactly one such line; any
/// following lines (extra reasoning the model added anyway) are ignored for
/// classification but the raw text is preserved verbatim in the `Forced`
/// case.
fn parse_agentic_reply(text: &str) -> AgenticStep {
    let first_line = text.lines().next().unwrap_or("").trim();
    if let Some(rest) = first_line.strip_prefix("ANSWER:") {
        return AgenticStep::Answer(rest.trim().to_string());
    }
    if let Some(rest) = first_line.strip_prefix("SEARCH:") {
        return AgenticStep::Search(rest.trim().to_string());
    }
    AgenticStep::Forced(text.trim().to_string())
}

/// Build the per-round prompt: the question plus the numbered context
/// memories accumulated so far. `force_answer` is set on the last permitted
/// round, instructing the judge to answer no matter how incomplete the
/// context is (the loop's bound / fail-safe against running forever).
fn build_agentic_prompt(
    question: &str,
    context: &[(String, String)],
    force_answer: bool,
) -> String {
    let ctx = if context.is_empty() {
        "(no memories recalled yet)".to_string()
    } else {
        context
            .iter()
            .enumerate()
            .map(|(i, (_, v))| format!("[{}] {v}", i + 1))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if force_answer {
        format!(
            "You are answering a question using ONLY the memories below. This is \
             the FINAL round: you must answer now, even if the memories seem \
             incomplete — do the best you can with what is here. Reply with \
             EXACTLY one line in the form:\nANSWER: <your answer>\n\n\
             Memories:\n{ctx}\n\nQuestion: {question}"
        )
    } else {
        format!(
            "You are answering a question using ONLY the memories below. If the \
             memories fully answer the question, reply with EXACTLY one line:\n\
             ANSWER: <your answer>\n\
             If they do NOT contain enough information, reply with EXACTLY one \
             line naming ONE specific missing fact to look up:\n\
             SEARCH: <one specific missing fact>\n\
             Do not use any knowledge beyond the memories shown. Do not add any \
             other text.\n\nMemories:\n{ctx}\n\nQuestion: {question}"
        )
    }
}

/// Add `fragments` (key, value pairs) to `context`, skipping any key already
/// present (dedup by key, first write wins). Returns the keys that were
/// actually newly added, in order.
fn add_new_context(
    context: &mut Vec<(String, String)>,
    fragments: Vec<(String, String)>,
) -> Vec<String> {
    let mut seen: HashSet<String> = context.iter().map(|(k, _)| k.clone()).collect();
    let mut added_keys = Vec::new();
    for (key, value) in fragments {
        if seen.insert(key.clone()) {
            context.push((key.clone(), value));
            added_keys.push(key);
        }
    }
    added_keys
}

/// Did the follow-up searches surface gold evidence that the initial search
/// missed? `evidence_ids` is the question's labeled gold evidence,
/// `initial_ids` the normalized ids from the seed search, `followup_keys`
/// the raw (un-normalized) keys added by ALL follow-up searches across the
/// loop. `false` when the question has no labeled evidence.
fn gold_added_by_followup(
    evidence_ids: &[String],
    initial_ids: &HashSet<String>,
    followup_keys: &[String],
) -> bool {
    if evidence_ids.is_empty() {
        return false;
    }
    let followup_ids: HashSet<String> = normalize_retrieved(followup_keys).into_iter().collect();
    evidence_ids
        .iter()
        .any(|id| followup_ids.contains(id) && !initial_ids.contains(id))
}

/// Per-question agentic QA record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgenticQuestionRecord {
    /// Dataset-native question id.
    pub question_id: String,
    /// Debug-formatted [`QType`].
    pub qtype: String,
    /// The judge's final answer (either an `ANSWER:` reply or, in the rare
    /// fail-safe case, the raw unparseable reply text).
    pub predicted: String,
    /// The judge's verdict against the gold answer.
    pub correct: bool,
    /// How many rounds of the loop ran (1 = answered on the seed context;
    /// up to `max_rounds`).
    pub rounds_used: usize,
    /// How many `SEARCH:` follow-up queries were issued.
    pub follow_up_searches: usize,
    /// Whether ANY gold `evidence_id` appeared anywhere in the FINAL
    /// accumulated context (seed + all follow-ups), normalized the same way
    /// [`normalize_retrieved`] does for [`run_qa`]. `false` for questions
    /// with no labeled evidence.
    pub gold_in_context: bool,
    /// Whether a follow-up search added gold evidence that the seed search
    /// had missed (see [`gold_added_by_followup`]).
    pub gold_added_by_followup: bool,
    /// Whether this question had any `evidence_ids` labeled at all.
    pub has_gold: bool,
    /// Whether `predicted` reads as a refusal / "I don't know" response
    /// (see [`is_abstention`]), rather than a confident answer.
    pub abstained: bool,
}

/// Measured agentic QA report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgenticReport {
    /// Retrieval depth used for each search call (seed and follow-ups).
    pub k: usize,
    /// Max rounds permitted per question.
    pub max_rounds: usize,
    /// Total questions graded.
    pub questions: usize,
    /// Questions the judge graded correct.
    pub correct: usize,
    /// Overall accuracy (`correct / questions`).
    pub accuracy: f64,
    /// Accuracy broken down by question category.
    pub by_qtype: BTreeMap<String, QTypeAccuracy>,
    /// Mean `rounds_used` across all questions.
    pub mean_rounds_used: f64,
    /// Fraction of questions where a follow-up search added gold evidence
    /// the seed search had missed.
    pub gold_added_by_followup_fraction: f64,
    /// Per-question records.
    pub per_question: Vec<AgenticQuestionRecord>,
    /// Gold-in-final-context-vs-judge-verdict cross-tab, computed the same
    /// way as [`QaRecallBreakdown`] but over the final accumulated context
    /// rather than a single seed search.
    pub recall_breakdown: QaRecallBreakdown,
    /// Confabulation-vs-abstention breakdown (see [`FaithfulnessBreakdown`]),
    /// computed over the same records as `recall_breakdown`.
    pub faithfulness: FaithfulnessBreakdown,
}

/// The two records produced by [`run_one_agentic_question`] for a single
/// question: the report-facing [`AgenticQuestionRecord`] and the
/// [`QaQuestionRecord`] used to build the shared cross-tab, kept in lockstep
/// so callers never have to recompute overlapping fields.
struct AgenticQuestionOutcome {
    per_question: AgenticQuestionRecord,
    cross_tab: QaQuestionRecord,
}

/// Run the agentic answer-guided retrieval loop for a single `question`
/// against the already-ingested `memory`, then grade the result with
/// `judge`. This is the per-question unit of work that [`run_agentic_qa`]
/// fans out across questions with bounded concurrency.
///
/// Seeds the context from `memory.search(question, k)`, then loops up to
/// `max_rounds` times, each round asking the judge to either answer from the
/// current context or name a specific follow-up search; `SEARCH:` results
/// are merged into the context (deduped by key) and the loop continues. The
/// loop is bounded: on the last permitted round the judge is asked to answer
/// no matter what, and an unparseable reply at any point is treated as a
/// forced answer rather than retried — so the loop always terminates within
/// `max_rounds` calls to `judge.complete` plus one call to `judge.grade`.
async fn run_one_agentic_question(
    memory: &AgentMemory,
    judge: &CodexCliJudge,
    question: &EvalQuestion,
    k: usize,
    max_rounds: usize,
) -> Result<AgenticQuestionOutcome, QaError> {
    let seed_fragments = memory.search(&question.text, k).await.map_err(mem_err)?;
    let seed_keys: Vec<String> = seed_fragments.iter().map(|f| f.entry.key.clone()).collect();
    let initial_ids: HashSet<String> = normalize_retrieved(&seed_keys).into_iter().collect();

    let mut context: Vec<(String, String)> = Vec::new();
    add_new_context(
        &mut context,
        seed_fragments
            .into_iter()
            .map(|f| (f.entry.key, f.entry.value))
            .collect(),
    );

    let mut followup_keys: Vec<String> = Vec::new();
    let mut follow_up_searches = 0usize;
    let mut rounds_used = 0usize;
    let mut final_answer: Option<String> = None;

    for round in 1..=max_rounds {
        rounds_used = round;
        let force_answer = round == max_rounds;
        let prompt = build_agentic_prompt(&question.text, &context, force_answer);
        let reply = judge.complete(prompt).await?;

        if force_answer {
            final_answer = Some(match parse_agentic_reply(&reply) {
                AgenticStep::Answer(a) => a,
                AgenticStep::Search(_) | AgenticStep::Forced(_) => reply.trim().to_string(),
            });
            break;
        }

        match parse_agentic_reply(&reply) {
            AgenticStep::Answer(a) => {
                final_answer = Some(a);
                break;
            }
            AgenticStep::Search(follow_up_query) => {
                follow_up_searches += 1;
                let fragments = memory.search(&follow_up_query, k).await.map_err(mem_err)?;
                let added = add_new_context(
                    &mut context,
                    fragments
                        .into_iter()
                        .map(|f| (f.entry.key, f.entry.value))
                        .collect(),
                );
                followup_keys.extend(added);
            }
            AgenticStep::Forced(text) => {
                final_answer = Some(text);
                break;
            }
        }
    }

    // `final_answer` is always `Some` by this point: either an explicit
    // ANSWER, a Forced fail-safe, or the forced-round reply — every branch
    // above sets it before falling out of the loop.
    let predicted = final_answer.expect("agentic loop always sets a final answer before exiting");
    let correct = judge
        .grade(&question.text, &question.gold_answer, &predicted)
        .await?;

    let has_gold = !question.evidence_ids.is_empty();
    let all_context_keys: Vec<String> = context.iter().map(|(k, _)| k.clone()).collect();
    let all_ids: HashSet<String> = normalize_retrieved(&all_context_keys).into_iter().collect();
    let gold_in_context = has_gold && question.evidence_ids.iter().any(|id| all_ids.contains(id));
    let gold_added = gold_added_by_followup(&question.evidence_ids, &initial_ids, &followup_keys);

    let qtype = qtype_key(question.qtype);
    let abstained = is_abstention(&predicted);

    Ok(AgenticQuestionOutcome {
        per_question: AgenticQuestionRecord {
            question_id: question.id.clone(),
            qtype: qtype.clone(),
            predicted: predicted.clone(),
            correct,
            rounds_used,
            follow_up_searches,
            gold_in_context,
            gold_added_by_followup: gold_added,
            has_gold,
            abstained,
        },
        cross_tab: QaQuestionRecord {
            question_id: question.id.clone(),
            qtype,
            predicted,
            correct,
            gold_retrieved: gold_in_context,
            has_gold,
            abstained,
        },
    })
}

/// Run agentic answer-guided retrieval QA over `conversations` with `judge`.
///
/// Conversations run sequentially (ingest is the dominant cost there), but
/// within each conversation the per-question agentic pipelines — seed
/// search, the `judge.complete` round loop, and the final `judge.grade` —
/// run concurrently with bounded concurrency (see [`ENV_QA_CONCURRENCY`],
/// mirroring [`run_qa`]'s `judge_questions`), since each question makes
/// several sequential judge calls and is otherwise independent of the
/// others. `memory.search` and `judge`'s methods all take `&self`, so they
/// are safe to call concurrently from multiple in-flight questions.
///
/// Fail-closed on infrastructure failure: any `memory.search`,
/// `judge.complete`, or `judge.grade` error aborts the whole run (no partial
/// report), matching [`run_qa`]'s contract. Results are independent of the
/// concurrency level: `per_question` (and the internal cross-tab records)
/// are sorted by `question_id` before returning, and the aggregate counts
/// are order-independent, so two runs over the same inputs produce
/// identical reports.
pub async fn run_agentic_qa(
    conversations: &[EvalConversation],
    judge: &CodexCliJudge,
    k: usize,
    max_rounds: usize,
) -> Result<AgenticReport, QaError> {
    let max_rounds = max_rounds.max(1);
    let concurrency = qa_concurrency();
    let mut per_question: Vec<AgenticQuestionRecord> = Vec::new();
    let mut by_qtype: BTreeMap<String, QTypeAccuracy> = BTreeMap::new();
    let mut cross_tab_records: Vec<QaQuestionRecord> = Vec::new();

    for conversation in conversations {
        let mut memory = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(mem_err)?;
        ingest(conversation, &mut memory)
            .await
            .map_err(|e| QaError::Memory(e.to_string()))?;

        let memory_ref = &memory;
        let mut outcomes = stream::iter(conversation.questions.iter())
            .map(|question| run_one_agentic_question(memory_ref, judge, question, k, max_rounds))
            .buffer_unordered(concurrency);
        while let Some(outcome) = outcomes.next().await {
            let outcome = outcome?;

            let bucket = by_qtype
                .entry(outcome.per_question.qtype.clone())
                .or_insert(QTypeAccuracy {
                    questions: 0,
                    correct: 0,
                    accuracy: 0.0,
                });
            bucket.questions += 1;
            bucket.correct += usize::from(outcome.per_question.correct);

            cross_tab_records.push(outcome.cross_tab);
            per_question.push(outcome.per_question);
        }
    }

    for bucket in by_qtype.values_mut() {
        bucket.accuracy = if bucket.questions == 0 {
            0.0
        } else {
            bucket.correct as f64 / bucket.questions as f64
        };
    }
    per_question.sort_by(|a, b| a.question_id.cmp(&b.question_id));
    cross_tab_records.sort_by(|a, b| a.question_id.cmp(&b.question_id));

    let questions = per_question.len();
    let correct = per_question.iter().filter(|q| q.correct).count();
    let mean_rounds_used = if questions == 0 {
        0.0
    } else {
        per_question
            .iter()
            .map(|q| q.rounds_used as f64)
            .sum::<f64>()
            / questions as f64
    };
    let gold_added_by_followup_fraction = if questions == 0 {
        0.0
    } else {
        per_question
            .iter()
            .filter(|q| q.gold_added_by_followup)
            .count() as f64
            / questions as f64
    };
    let recall_breakdown = QaRecallBreakdown::from_records(&cross_tab_records);
    let faithfulness = FaithfulnessBreakdown::from_records(&cross_tab_records);

    Ok(AgenticReport {
        k,
        max_rounds,
        questions,
        correct,
        accuracy: if questions == 0 {
            0.0
        } else {
            correct as f64 / questions as f64
        },
        by_qtype,
        mean_rounds_used,
        gold_added_by_followup_fraction,
        per_question,
        recall_breakdown,
        faithfulness,
    })
}

#[cfg(test)]
mod agentic_qa_tests {
    use super::*;

    #[test]
    fn parses_answer_prefix() {
        assert_eq!(
            parse_agentic_reply("ANSWER: Paris\nignored trailing line"),
            AgenticStep::Answer("Paris".to_string())
        );
    }

    #[test]
    fn parses_search_prefix() {
        assert_eq!(
            parse_agentic_reply("SEARCH: Alice's birth year"),
            AgenticStep::Search("Alice's birth year".to_string())
        );
    }

    #[test]
    fn unparseable_reply_is_forced() {
        assert_eq!(
            parse_agentic_reply("I think it's Paris, probably."),
            AgenticStep::Forced("I think it's Paris, probably.".to_string())
        );
    }

    #[test]
    fn empty_reply_is_forced_empty() {
        assert_eq!(parse_agentic_reply(""), AgenticStep::Forced(String::new()));
    }

    #[test]
    fn add_new_context_dedups_by_key_and_reports_only_new_keys() {
        let mut context: Vec<(String, String)> = vec![("k1".to_string(), "v1".to_string())];
        let added = add_new_context(
            &mut context,
            vec![
                (
                    "k1".to_string(),
                    "v1-duplicate-should-be-ignored".to_string(),
                ),
                ("k2".to_string(), "v2".to_string()),
            ],
        );
        assert_eq!(added, vec!["k2".to_string()]);
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].1, "v1", "first write for a key wins");
    }

    #[test]
    fn add_new_context_dedups_within_a_single_batch() {
        let mut context: Vec<(String, String)> = Vec::new();
        let added = add_new_context(
            &mut context,
            vec![
                ("k1".to_string(), "v1".to_string()),
                ("k1".to_string(), "v1-again".to_string()),
            ],
        );
        assert_eq!(added, vec!["k1".to_string()]);
        assert_eq!(context.len(), 1);
    }

    #[test]
    fn gold_added_by_followup_true_when_followup_finds_gold_initial_missed() {
        let evidence = vec!["D1:1".to_string()];
        let initial_ids: HashSet<String> = HashSet::new();
        let followup_keys = vec!["D1:1".to_string()];
        assert!(gold_added_by_followup(
            &evidence,
            &initial_ids,
            &followup_keys
        ));
    }

    #[test]
    fn gold_added_by_followup_false_when_initial_already_had_gold() {
        let evidence = vec!["D1:1".to_string()];
        let mut initial_ids: HashSet<String> = HashSet::new();
        initial_ids.insert("D1:1".to_string());
        let followup_keys = vec!["D1:1".to_string()];
        assert!(!gold_added_by_followup(
            &evidence,
            &initial_ids,
            &followup_keys
        ));
    }

    #[test]
    fn gold_added_by_followup_false_when_followups_never_find_gold() {
        let evidence = vec!["D1:1".to_string()];
        let initial_ids: HashSet<String> = HashSet::new();
        let followup_keys = vec!["D1:2".to_string(), "D1:3".to_string()];
        assert!(!gold_added_by_followup(
            &evidence,
            &initial_ids,
            &followup_keys
        ));
    }

    #[test]
    fn gold_added_by_followup_false_when_question_has_no_gold() {
        let evidence: Vec<String> = vec![];
        let initial_ids: HashSet<String> = HashSet::new();
        let followup_keys = vec!["D1:1".to_string()];
        assert!(!gold_added_by_followup(
            &evidence,
            &initial_ids,
            &followup_keys
        ));
    }

    #[test]
    fn agentic_max_rounds_defaults_and_clamps() {
        // Guard against test-order env races on this process-wide var, same
        // pattern as the run_qa concurrency-env tests.
        std::env::remove_var(ENV_AGENTIC_ROUNDS);
        assert_eq!(agentic_max_rounds(), 3);

        std::env::set_var(ENV_AGENTIC_ROUNDS, "0");
        assert_eq!(agentic_max_rounds(), 1, "clamped to at least 1");

        std::env::set_var(ENV_AGENTIC_ROUNDS, "7");
        assert_eq!(agentic_max_rounds(), 7);

        std::env::remove_var(ENV_AGENTIC_ROUNDS);
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

    fn fixture_with_n_questions(n: usize) -> Vec<EvalConversation> {
        let questions = (0..n)
            .map(|i| EvalQuestion {
                id: format!("q{i:02}"),
                text: format!("Question number {i}?"),
                qtype: QType::SingleHop,
                evidence_ids: vec![],
                gold_answer: format!("answer {i}"),
            })
            .collect();
        vec![EvalConversation {
            id: "conv1".to_string(),
            sessions: vec![Session {
                id: "s1".to_string(),
                timestamp: None,
                turns: vec![],
            }],
            questions,
        }]
    }

    /// Guards process-wide env var races between tests that touch
    /// `ENV_QA_CONCURRENCY` — only one such test should run its critical
    /// section at a time.
    static CONCURRENCY_ENV_GUARD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn run_qa_with_bounded_concurrency_matches_sequential_semantics() {
        let _guard = CONCURRENCY_ENV_GUARD.lock().await;
        std::env::set_var(ENV_QA_CONCURRENCY, "3");

        let conversations = fixture_with_n_questions(6);

        let judge_a = FakeJudge::new();
        let report_a = run_qa(&conversations, &judge_a, 5)
            .await
            .expect("run_qa should succeed with concurrency > 1");

        let judge_b = FakeJudge::new();
        let report_b = run_qa(&conversations, &judge_b, 5)
            .await
            .expect("run_qa should succeed with concurrency > 1 (second run)");

        std::env::remove_var(ENV_QA_CONCURRENCY);
        drop(_guard);

        // Same aggregate accuracy/counts as the sequential expectation:
        // alternating correct/incorrect over 6 questions -> 3 correct.
        assert_eq!(report_a.questions, 6);
        assert_eq!(report_a.correct, 3);
        assert!((report_a.accuracy - 0.5).abs() < f64::EPSILON);
        let bucket = report_a
            .by_qtype
            .get(&qtype_key(QType::SingleHop))
            .expect("SingleHop bucket present");
        assert_eq!(bucket.questions, 6);
        assert_eq!(bucket.correct, 3);

        // Identical aggregates across two independent runs.
        assert_eq!(report_a.questions, report_b.questions);
        assert_eq!(report_a.correct, report_b.correct);
        assert_eq!(report_a.accuracy, report_b.accuracy);

        // per_question is sorted by question_id, deterministically, in both
        // runs — proving concurrency doesn't scramble output order.
        let ids_a: Vec<&str> = report_a
            .per_question
            .iter()
            .map(|q| q.question_id.as_str())
            .collect();
        let ids_b: Vec<&str> = report_b
            .per_question
            .iter()
            .map(|q| q.question_id.as_str())
            .collect();
        assert_eq!(ids_a, ids_b);
        let mut sorted_ids = ids_a.clone();
        sorted_ids.sort();
        assert_eq!(ids_a, sorted_ids, "per_question must be sorted by id");
    }

    /// Judge whose `grade` fails on one specific question, to prove
    /// concurrent `run_qa` still fails closed (returns `Err`, no partial
    /// report) rather than swallowing the error or reporting it as
    /// incorrect.
    struct FailingJudge;

    #[async_trait::async_trait]
    impl Judge for FailingJudge {
        async fn answer(&self, _question: &str, _recalled: &[String]) -> Result<String, QaError> {
            Ok("fake answer".to_string())
        }

        async fn grade(
            &self,
            question: &str,
            _gold: &str,
            _predicted: &str,
        ) -> Result<bool, QaError> {
            if question.contains("number 3") {
                Err(QaError::Judge("simulated grading failure".to_string()))
            } else {
                Ok(true)
            }
        }
    }

    #[tokio::test]
    async fn run_qa_fails_closed_on_judge_error_under_concurrency() {
        let _guard = CONCURRENCY_ENV_GUARD.lock().await;
        std::env::set_var(ENV_QA_CONCURRENCY, "4");

        let conversations = fixture_with_n_questions(6);
        let judge = FailingJudge;
        let result = run_qa(&conversations, &judge, 5).await;

        std::env::remove_var(ENV_QA_CONCURRENCY);
        drop(_guard);

        assert!(
            result.is_err(),
            "a single failing judge call must fail the whole run"
        );
    }

    fn recall_fixture() -> Vec<EvalConversation> {
        vec![EvalConversation {
            id: "conv1".to_string(),
            sessions: vec![Session {
                id: "session_1".to_string(),
                timestamp: None,
                turns: vec![crate::dataset::Turn {
                    speaker: "alice".to_string(),
                    text: "The sky is a bright shade of blue today.".to_string(),
                    timestamp: None,
                }],
            }],
            questions: vec![
                // Lexically close to the stored turn -> should be retrieved,
                // and its evidence_ids ("D1:1") matches turn_memory_key for
                // session_1's first (0-indexed) turn.
                EvalQuestion {
                    id: "q_hit".to_string(),
                    text: "What color is the sky today?".to_string(),
                    qtype: QType::SingleHop,
                    evidence_ids: vec!["D1:1".to_string()],
                    gold_answer: "blue".to_string(),
                },
                // evidence_ids reference an id that is never stored -> gold
                // can never be retrieved, regardless of what search returns.
                EvalQuestion {
                    id: "q_miss".to_string(),
                    text: "What color is the sky today?".to_string(),
                    qtype: QType::SingleHop,
                    evidence_ids: vec!["D1:99".to_string()],
                    gold_answer: "blue".to_string(),
                },
                // No labeled evidence at all -> excluded from the cross-tab,
                // counted in no_gold_labeled.
                EvalQuestion {
                    id: "q_nogold".to_string(),
                    text: "Is there a dragon in this conversation?".to_string(),
                    qtype: QType::Abstention,
                    evidence_ids: vec![],
                    gold_answer: "I don't know".to_string(),
                },
            ],
        }]
    }

    #[tokio::test]
    async fn run_qa_computes_gold_retrieved_and_cross_tab() {
        let conversations = recall_fixture();
        let judge = FakeJudge::new();
        let report = run_qa(&conversations, &judge, 5)
            .await
            .expect("run_qa should succeed with a fake judge");

        let by_id = |id: &str| {
            report
                .per_question
                .iter()
                .find(|q| q.question_id == id)
                .unwrap_or_else(|| panic!("question {id} present in report"))
        };

        let hit = by_id("q_hit");
        assert!(hit.has_gold);
        assert!(
            hit.gold_retrieved,
            "gold evidence D1:1 has a clear lexical match and must be retrieved"
        );

        let miss = by_id("q_miss");
        assert!(miss.has_gold);
        assert!(
            !miss.gold_retrieved,
            "evidence id D1:99 was never stored, so it can never be retrieved"
        );

        let nogold = by_id("q_nogold");
        assert!(!nogold.has_gold);
        assert!(
            !nogold.gold_retrieved,
            "questions with no labeled evidence are defined as gold_retrieved = false"
        );

        // Cross-tab must match a hand-tally over per_question:
        // - q_nogold is excluded (has_gold == false) -> no_gold_labeled == 1.
        // - q_hit and q_miss are the only two in the a/b/c/d denominator.
        let tab = report.recall_breakdown;
        assert_eq!(tab.no_gold_labeled, 1);
        assert_eq!(tab.total(), 2);
        assert_eq!(tab.a + tab.b, usize::from(hit.gold_retrieved));
        assert_eq!(tab.c + tab.d, usize::from(!miss.gold_retrieved));
        let expected = QaRecallBreakdown::from_records(&report.per_question);
        assert_eq!(tab.a, expected.a);
        assert_eq!(tab.b, expected.b);
        assert_eq!(tab.c, expected.c);
        assert_eq!(tab.d, expected.d);
        assert_eq!(tab.no_gold_labeled, expected.no_gold_labeled);
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

#[cfg(test)]
mod faithfulness_tests {
    use super::*;

    #[test]
    fn is_abstention_matches_each_documented_phrase() {
        for phrase in ABSTENTION_PHRASES {
            let answer = format!("Well, {phrase}, sorry.");
            assert!(
                is_abstention(&answer),
                "expected phrase {phrase:?} to be detected as abstention in {answer:?}"
            );
        }
    }

    #[test]
    fn is_abstention_true_on_real_i_dont_know_answer() {
        assert!(is_abstention("I don't know based on the given context."));
    }

    #[test]
    fn is_abstention_false_on_normal_factual_answer() {
        assert!(!is_abstention("Adoption agencies"));
    }

    fn record(
        qtype: &str,
        gold_retrieved: bool,
        correct: bool,
        abstained: bool,
    ) -> QaQuestionRecord {
        QaQuestionRecord {
            question_id: "q".to_string(),
            qtype: qtype.to_string(),
            predicted: if abstained {
                "I don't know".to_string()
            } else {
                "some confident answer".to_string()
            },
            correct,
            gold_retrieved,
            has_gold: qtype != "Abstention",
            abstained,
        }
    }

    #[test]
    fn faithfulness_breakdown_aggregates_all_buckets() {
        let records = vec![
            // Abstention-category question that correctly abstained.
            record("Abstention", false, false, true),
            // Abstention-category question that confabulated an answer.
            record("Abstention", false, false, false),
            // Answerable question, evidence missing, honestly abstained.
            record("SingleHop", false, false, true),
            // Answerable question, evidence missing, answered wrong (confabulated).
            record("SingleHop", false, false, false),
            // Answerable question, evidence missing, answered correct (bucket C).
            record("SingleHop", false, true, false),
            // Answerable question, evidence present, but abstained anyway.
            record("SingleHop", true, false, true),
            // Answerable question, evidence present, answered correctly — not
            // counted anywhere in the faithfulness breakdown.
            record("SingleHop", true, true, false),
        ];

        let b = FaithfulnessBreakdown::from_records(&records);

        assert_eq!(b.abstention_qtype_total, 2);
        assert_eq!(b.abstention_qtype_abstained, 1);
        assert_eq!(b.abstention_qtype_confabulated, 1);

        assert_eq!(b.evidence_missing_total, 3);
        assert_eq!(b.evidence_missing_abstained, 1);
        assert_eq!(b.evidence_missing_confabulated, 1);
        assert_eq!(b.evidence_missing_answered_correct, 1);

        assert_eq!(b.over_abstention, 1);
    }

    #[test]
    fn faithfulness_breakdown_empty_records_is_all_zero() {
        let b = FaithfulnessBreakdown::from_records(&[]);
        assert_eq!(b.abstention_qtype_total, 0);
        assert_eq!(b.evidence_missing_total, 0);
        assert_eq!(b.over_abstention, 0);
    }
}
