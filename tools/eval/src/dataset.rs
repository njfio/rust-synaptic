//! Loaders for the LongMemEval-S and LoCoMo benchmark datasets.
//!
//! Both real formats are parsed into a common typed model
//! ([`EvalConversation`]). The loaders are strict: malformed or unexpected
//! input yields a [`DatasetError`] — nothing is fabricated or defaulted.
//!
//! ## Native schemas mapped here
//!
//! **LongMemEval-S** (`longmemeval_s.json`, HuggingFace
//! `xiaowu0162/longmemeval`): a JSON array of question instances, each with
//! `question_id`, `question_type` (`single-session-user`,
//! `single-session-assistant`, `single-session-preference`, `multi-session`,
//! `temporal-reasoning`, `knowledge-update`), `question`, `answer`,
//! `haystack_dates`, `haystack_session_ids`, `haystack_sessions` (list of
//! sessions, each a list of `{role, content}` turns) and
//! `answer_session_ids`. Abstention instances are marked by a `question_id`
//! ending in `_abs`. Each instance becomes one [`EvalConversation`] holding
//! its haystack sessions and a single question whose `evidence_ids` are the
//! `answer_session_ids`.
//!
//! **LoCoMo** (`locomo10.json`, github `snap-research/locomo`): a JSON array
//! of samples, each with `sample_id`, a `conversation` object containing
//! `speaker_a`/`speaker_b` plus `session_N` turn arrays (`{speaker, dia_id,
//! text}`) and `session_N_date_time` strings, and a `qa` array of
//! `{question, answer | adversarial_answer, evidence, category}`. Categories
//! follow the LoCoMo paper: 1 = multi-hop, 2 = temporal, 3 = open-domain,
//! 4 = single-hop, 5 = adversarial (abstention). Question `evidence_ids` are
//! the `dia_id` references from `evidence`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Errors produced by the dataset loaders.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// The input was not valid JSON or did not match the expected schema.
    #[error("failed to parse dataset JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The JSON was well-formed but semantically invalid for this dataset.
    #[error("invalid dataset content: {0}")]
    Invalid(String),
}

/// Benchmark question category, unified across LongMemEval and LoCoMo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QType {
    /// Answerable from a single session / single evidence turn.
    SingleHop,
    /// Requires synthesizing information across sessions/turns.
    MultiHop,
    /// Requires temporal reasoning about when things happened.
    Temporal,
    /// The answer changed over time; the latest state is required.
    KnowledgeUpdate,
    /// Open-domain / world-knowledge-flavored question grounded in the chat.
    OpenDomain,
    /// Unanswerable; the correct behavior is to abstain.
    Abstention,
}

/// A single utterance in a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Who spoke: a LongMemEval role (`user`/`assistant`) or a LoCoMo name.
    pub speaker: String,
    /// The utterance text.
    pub text: String,
    /// Turn-level timestamp when the dataset provides one (kept verbatim).
    pub timestamp: Option<String>,
}

/// One conversational session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Dataset-native session id (LongMemEval haystack id, LoCoMo `session_N`).
    pub id: String,
    /// Session-level timestamp string, verbatim from the dataset.
    pub timestamp: Option<String>,
    /// The turns of the session, in order.
    pub turns: Vec<Turn>,
}

/// A benchmark question with its ground truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalQuestion {
    /// Dataset-native question id (synthesized `{sample}_q{i}` for LoCoMo).
    pub id: String,
    /// The question text.
    pub text: String,
    /// Unified question category.
    pub qtype: QType,
    /// Ground-truth evidence ids: LongMemEval `answer_session_ids` (session
    /// ids) or LoCoMo `evidence` (`dia_id` turn references).
    pub evidence_ids: Vec<String>,
    /// The gold answer string.
    pub gold_answer: String,
}

/// A full conversation with its sessions and evaluation questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalConversation {
    /// Dataset-native conversation id.
    pub id: String,
    /// Sessions in dataset order.
    pub sessions: Vec<Session>,
    /// Questions grounded in this conversation.
    pub questions: Vec<EvalQuestion>,
}

// ---------------------------------------------------------------------------
// LongMemEval-S
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LmeInstance {
    question_id: String,
    question_type: String,
    question: String,
    answer: Value,
    #[serde(default)]
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LmeTurn>>,
    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LmeTurn {
    role: String,
    content: String,
}

fn lme_qtype(question_type: &str, question_id: &str) -> Result<QType, DatasetError> {
    // Abstention variants keep their base question_type but are marked by
    // the `_abs` id suffix in LongMemEval.
    if question_id.ends_with("_abs") {
        return Ok(QType::Abstention);
    }
    match question_type {
        "single-session-user" | "single-session-assistant" | "single-session-preference" => {
            Ok(QType::SingleHop)
        }
        "multi-session" => Ok(QType::MultiHop),
        "temporal-reasoning" => Ok(QType::Temporal),
        "knowledge-update" => Ok(QType::KnowledgeUpdate),
        other => Err(DatasetError::Invalid(format!(
            "unknown LongMemEval question_type: {other:?}"
        ))),
    }
}

fn value_to_answer_string(v: &Value) -> Result<String, DatasetError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(DatasetError::Invalid(format!(
            "unsupported gold-answer JSON type: {other}"
        ))),
    }
}

/// Parse a LongMemEval-S JSON document (an array of question instances).
pub fn load_longmemeval_str(json: &str) -> Result<Vec<EvalConversation>, DatasetError> {
    let instances: Vec<LmeInstance> = serde_json::from_str(json)?;
    instances
        .into_iter()
        .map(|inst| {
            if inst.haystack_session_ids.len() != inst.haystack_sessions.len() {
                return Err(DatasetError::Invalid(format!(
                    "instance {}: {} haystack_session_ids but {} haystack_sessions",
                    inst.question_id,
                    inst.haystack_session_ids.len(),
                    inst.haystack_sessions.len()
                )));
            }
            let sessions = inst
                .haystack_session_ids
                .iter()
                .zip(inst.haystack_sessions.iter())
                .enumerate()
                .map(|(i, (sid, turns))| Session {
                    id: sid.clone(),
                    timestamp: inst.haystack_dates.get(i).cloned(),
                    turns: turns
                        .iter()
                        .map(|t| Turn {
                            speaker: t.role.clone(),
                            text: t.content.clone(),
                            timestamp: None,
                        })
                        .collect(),
                })
                .collect();
            let qtype = lme_qtype(&inst.question_type, &inst.question_id)?;
            let gold_answer = value_to_answer_string(&inst.answer)?;
            Ok(EvalConversation {
                id: inst.question_id.clone(),
                sessions,
                questions: vec![EvalQuestion {
                    id: inst.question_id,
                    text: inst.question,
                    qtype,
                    evidence_ids: inst.answer_session_ids,
                    gold_answer,
                }],
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LoCoMo
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LocomoSample {
    sample_id: String,
    conversation: BTreeMap<String, Value>,
    qa: Vec<LocomoQa>,
}

#[derive(Debug, Deserialize)]
struct LocomoQa {
    question: String,
    #[serde(default)]
    answer: Option<Value>,
    #[serde(default)]
    adversarial_answer: Option<Value>,
    #[serde(default)]
    evidence: Vec<String>,
    category: u32,
}

#[derive(Debug, Deserialize)]
struct LocomoTurn {
    speaker: String,
    text: String,
}

fn locomo_qtype(category: u32) -> Result<QType, DatasetError> {
    match category {
        1 => Ok(QType::MultiHop),
        2 => Ok(QType::Temporal),
        3 => Ok(QType::OpenDomain),
        4 => Ok(QType::SingleHop),
        5 => Ok(QType::Abstention),
        other => Err(DatasetError::Invalid(format!(
            "unknown LoCoMo qa category: {other}"
        ))),
    }
}

/// Parse a LoCoMo JSON document (`locomo10.json`, an array of samples).
pub fn load_locomo_str(json: &str) -> Result<Vec<EvalConversation>, DatasetError> {
    let samples: Vec<LocomoSample> = serde_json::from_str(json)?;
    samples
        .into_iter()
        .map(|sample| {
            // Collect session_N keys and sort numerically by N.
            let mut indices: Vec<u32> = sample
                .conversation
                .keys()
                .filter_map(|k| {
                    k.strip_prefix("session_")
                        .and_then(|rest| rest.parse::<u32>().ok())
                })
                .collect();
            indices.sort_unstable();
            if indices.is_empty() {
                return Err(DatasetError::Invalid(format!(
                    "sample {}: conversation has no session_N keys",
                    sample.sample_id
                )));
            }
            let sessions = indices
                .iter()
                .map(|n| {
                    let key = format!("session_{n}");
                    let raw = sample.conversation.get(&key).ok_or_else(|| {
                        DatasetError::Invalid(format!("sample {}: missing {key}", sample.sample_id))
                    })?;
                    let turns: Vec<LocomoTurn> = serde_json::from_value(raw.clone())?;
                    let timestamp = sample
                        .conversation
                        .get(&format!("{key}_date_time"))
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    Ok(Session {
                        id: key,
                        timestamp: timestamp.clone(),
                        turns: turns
                            .into_iter()
                            .map(|t| Turn {
                                speaker: t.speaker,
                                text: t.text,
                                timestamp: timestamp.clone(),
                            })
                            .collect(),
                    })
                })
                .collect::<Result<Vec<_>, DatasetError>>()?;
            let questions = sample
                .qa
                .iter()
                .enumerate()
                .map(|(i, qa)| {
                    let qtype = locomo_qtype(qa.category)?;
                    // Category-5 (adversarial) items carry adversarial_answer
                    // instead of answer.
                    let gold = qa
                        .answer
                        .as_ref()
                        .or(qa.adversarial_answer.as_ref())
                        .ok_or_else(|| {
                            DatasetError::Invalid(format!(
                                "sample {} qa[{i}]: no answer or adversarial_answer",
                                sample.sample_id
                            ))
                        })?;
                    Ok(EvalQuestion {
                        id: format!("{}_q{i}", sample.sample_id),
                        text: qa.question.clone(),
                        qtype,
                        evidence_ids: qa.evidence.clone(),
                        gold_answer: value_to_answer_string(gold)?,
                    })
                })
                .collect::<Result<Vec<_>, DatasetError>>()?;
            Ok(EvalConversation {
                id: sample.sample_id,
                sessions,
                questions,
            })
        })
        .collect()
}

/// Read and parse a LongMemEval-S file from disk.
pub fn load_longmemeval(path: &std::path::Path) -> Result<Vec<EvalConversation>, DatasetError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| DatasetError::Invalid(format!("cannot read {}: {e}", path.display())))?;
    load_longmemeval_str(&raw)
}

/// Read and parse a LoCoMo file from disk.
pub fn load_locomo(path: &std::path::Path) -> Result<Vec<EvalConversation>, DatasetError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| DatasetError::Invalid(format!("cannot read {}: {e}", path.display())))?;
    load_locomo_str(&raw)
}
