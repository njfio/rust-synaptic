//! LLM-free evaluation runner: ingest benchmark conversations into
//! [`AgentMemory`], run each question through `search`, and score the ranked
//! retrieved ids against the dataset's `evidence_ids`.
//!
//! ## Id scheme (how retrieved ids match `evidence_ids`)
//!
//! Each turn is stored under a key that reproduces the dataset's evidence id
//! scheme:
//!
//! - **LoCoMo**: sessions are named `session_N` and evidence ids are dia_ids
//!   `DN:t` (1-indexed turn within session N). A turn stored from
//!   `session_N` at index `i` gets key `DN:{i+1}` — identical to its dia_id.
//! - **LongMemEval**: evidence ids are *session* ids, so a turn in session
//!   `S` at index `i` gets key `S::t{i+1}`. When scoring, retrieved keys of
//!   this form are normalized back to `S` (deduplicated, keeping the best
//!   rank) so they intersect directly with `answer_session_ids`.
//!
//! No LLM is involved anywhere in this module.

use crate::dataset::{EvalConversation, EvalQuestion, Session};
use crate::metrics::{
    aggregate, mrr, precision_at_k, recall_at_k, GrowthPoint, LatencySummary, QuestionMetrics,
    Report,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use synaptic::{AgentMemory, MemoryConfig};

/// Errors from the evaluation runner.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// The underlying memory system returned an error.
    #[error("memory operation failed: {0}")]
    Memory(String),
}

fn mem_err(e: impl std::fmt::Display) -> RunnerError {
    RunnerError::Memory(e.to_string())
}

/// Separator between a session id and the turn ordinal for datasets whose
/// evidence ids are session-level (LongMemEval).
const TURN_SEP: &str = "::t";

/// The stable memory key for turn `index` (0-based) of `session`, matching
/// the dataset's evidence id scheme (see module docs).
pub fn turn_memory_key(session: &Session, index: usize) -> String {
    if let Some(n) = session.id.strip_prefix("session_") {
        if n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty() {
            // LoCoMo dia_id scheme: D<session>:<turn>, 1-indexed.
            return format!("D{n}:{}", index + 1);
        }
    }
    format!("{}{TURN_SEP}{}", session.id, index + 1)
}

/// Normalize a ranked list of retrieved memory keys into evidence-id space:
/// session-level keys (`S::tN`) collapse to `S` (deduplicated at first, i.e.
/// best, rank); dia-style keys pass through unchanged.
pub fn normalize_retrieved(keys: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let id = match key.split_once(TURN_SEP) {
            Some((session, _)) => session.to_string(),
            None => key.clone(),
        };
        if seen.insert(id.clone()) {
            out.push(id);
        }
    }
    out
}

/// Measured result of ingesting one conversation.
#[derive(Debug, Clone)]
pub struct IngestStats {
    /// Number of turn memories stored.
    pub turns_stored: usize,
    /// Per-`store` wall-clock durations, in call order.
    pub store_durations: Vec<Duration>,
}

/// Store every turn of every session of `conversation` into `memory`, one
/// memory per turn, keyed by [`turn_memory_key`]. The stored value is
/// `"{speaker}: {text}"` (plus the timestamp when the dataset provides one).
pub async fn ingest(
    conversation: &EvalConversation,
    memory: &mut AgentMemory,
) -> Result<IngestStats, RunnerError> {
    let mut store_durations = Vec::new();
    for session in &conversation.sessions {
        for (i, turn) in session.turns.iter().enumerate() {
            let key = turn_memory_key(session, i);
            let value = match turn.timestamp.as_ref().or(session.timestamp.as_ref()) {
                Some(ts) => format!("[{ts}] {}: {}", turn.speaker, turn.text),
                None => format!("{}: {}", turn.speaker, turn.text),
            };
            let start = Instant::now();
            memory.store(&key, &value).await.map_err(mem_err)?;
            store_durations.push(start.elapsed());
        }
    }
    Ok(IngestStats {
        turns_stored: store_durations.len(),
        store_durations,
    })
}

/// Ingest all turns with enrichment deferred, then run one bounded-concurrent
/// enrich_pending pass. Returns (raw_ingest_secs, enrich_secs).
pub async fn ingest_deferred(
    conversation: &EvalConversation,
    memory: &mut AgentMemory,
) -> Result<(f64, f64), RunnerError> {
    let t0 = Instant::now();
    for session in &conversation.sessions {
        for (i, turn) in session.turns.iter().enumerate() {
            let key = turn_memory_key(session, i);
            let value = match turn.timestamp.as_ref().or(session.timestamp.as_ref()) {
                Some(ts) => format!("[{ts}] {}: {}", turn.speaker, turn.text),
                None => format!("{}: {}", turn.speaker, turn.text),
            };
            memory.store(&key, &value).await.map_err(mem_err)?;
        }
    }
    let raw_secs = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let _report = memory.enrich_pending().await;
    let enrich_secs = t1.elapsed().as_secs_f64();
    Ok((raw_secs, enrich_secs))
}

/// Result of running one question through retrieval.
#[derive(Debug, Clone)]
pub struct QuestionOutcome {
    /// Ranked retrieved ids, normalized into evidence-id space.
    pub retrieved_ids: Vec<String>,
    /// Wall-clock duration of the `search` call.
    pub recall_duration: Duration,
}

/// Run `question.text` through `AgentMemory::search` with limit `k` and
/// return the ranked retrieved ids normalized into evidence-id space.
pub async fn run_question(
    question: &EvalQuestion,
    memory: &AgentMemory,
    k: usize,
) -> Result<QuestionOutcome, RunnerError> {
    let start = Instant::now();
    let fragments = memory.search(&question.text, k).await.map_err(mem_err)?;
    let recall_duration = start.elapsed();
    let keys: Vec<String> = fragments.into_iter().map(|f| f.entry.key).collect();
    Ok(QuestionOutcome {
        retrieved_ids: normalize_retrieved(&keys),
        recall_duration,
    })
}

/// Evaluate every question of every conversation. Each conversation gets a
/// fresh in-memory [`AgentMemory`] (its own haystack), mirroring how the
/// benchmarks are defined. Returns a [`Report`] whose numbers all come from
/// this run; `growth` is left empty (see [`measure_growth`]).
pub async fn evaluate_conversations(
    conversations: &[EvalConversation],
    k: usize,
) -> Result<Report, RunnerError> {
    let mut per_question: Vec<QuestionMetrics> = Vec::new();
    let mut store_durations: Vec<Duration> = Vec::new();
    let mut recall_durations: Vec<Duration> = Vec::new();

    for conversation in conversations {
        let mut memory = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(mem_err)?;
        let ingest_stats = ingest(conversation, &mut memory).await?;
        store_durations.extend(ingest_stats.store_durations);

        for question in &conversation.questions {
            let outcome = run_question(question, &memory, k).await?;
            recall_durations.push(outcome.recall_duration);
            let relevant: HashSet<String> = question.evidence_ids.iter().cloned().collect();
            per_question.push(QuestionMetrics {
                question_id: question.id.clone(),
                qtype: format!("{:?}", question.qtype),
                precision_at_k: precision_at_k(&outcome.retrieved_ids, &relevant, k),
                recall_at_k: recall_at_k(&outcome.retrieved_ids, &relevant, k),
                reciprocal_rank: mrr(&outcome.retrieved_ids, &relevant),
            });
        }
    }

    let (mean_p, mean_r, mean_rr, by_qtype) = aggregate(&per_question);
    Ok(Report {
        k,
        questions_evaluated: per_question.len(),
        mean_precision_at_k: mean_p,
        mean_recall_at_k: mean_r,
        mrr: mean_rr,
        by_qtype,
        per_question,
        store_latency: LatencySummary::from_durations(&store_durations),
        recall_latency: LatencySummary::from_durations(&recall_durations),
        growth: Vec::new(),
    })
}

/// Fill fresh `AgentMemory` instances with synthetic turn-like entries at
/// each target size and record honest growth measurements.
///
/// Size proxy (documented, not RSS): entries-stored count plus the sum of
/// key+value UTF-8 bytes handed to `store`, alongside the library's own
/// `MemoryStats::total_size`. Production targets are `[1_000, 10_000,
/// 100_000]`; tests may pass smaller targets — the semantics are identical.
pub async fn measure_growth(targets: &[usize]) -> Result<Vec<GrowthPoint>, RunnerError> {
    let mut points = Vec::with_capacity(targets.len());
    for &target in targets {
        let mut config = MemoryConfig::default();
        // Raise eviction caps so the fill is not silently truncated.
        config.max_short_term_memories = target.max(config.max_short_term_memories);
        config.max_long_term_memories = target.max(config.max_long_term_memories);
        let mut memory = AgentMemory::new(config).await.map_err(mem_err)?;

        let mut store_durations = Vec::with_capacity(target);
        let mut stored_bytes_estimate = 0usize;
        for i in 0..target {
            let key = format!("growth:{i}");
            let value = format!(
                "synthetic turn {i}: the user mentioned topic {} during session {}",
                i % 97,
                i / 100
            );
            stored_bytes_estimate += key.len() + value.len();
            let start = Instant::now();
            memory.store(&key, &value).await.map_err(mem_err)?;
            store_durations.push(start.elapsed());
        }

        let probe_start = Instant::now();
        let _ = memory
            .search("user mentioned topic", 10)
            .await
            .map_err(mem_err)?;
        let probe_search_micros = probe_start.elapsed().as_micros();

        points.push(GrowthPoint {
            target_memories: target,
            stored_memories: store_durations.len(),
            stored_bytes_estimate,
            library_reported_total_size: memory.stats().await.total_size,
            store_latency: LatencySummary::from_durations(&store_durations),
            probe_search_micros,
        });
    }
    Ok(points)
}
