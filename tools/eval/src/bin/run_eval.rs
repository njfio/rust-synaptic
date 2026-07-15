//! Run the full LLM-free evaluation harness against a real dataset file and
//! print a structured summary of measured numbers (Task 7.5).
//!
//! Usage:
//!   cargo run --release -p synaptic-eval --bin run_eval -- [DATASET_PATH] [--growth-100k]
//!
//! `DATASET_PATH` defaults to `tools/eval/data/locomo10.json` (LoCoMo). Files
//! whose name contains "longmemeval" are parsed with the LongMemEval loader;
//! everything else uses the LoCoMo loader.
//!
//! Conversations are evaluated concurrently (one task per conversation; each
//! conversation is its own independent haystack, so results are identical to
//! a sequential run). Per-operation latencies are wall-clock times measured
//! under that concurrency and are reported as such.
//!
//! Every printed number comes from this process's real run. QA accuracy is
//! gated: without a selected, available judge it is reported as not-run,
//! never fabricated. Select the judge via `SYNAPTIC_EVAL_JUDGE`:
//!
//! - `SYNAPTIC_EVAL_JUDGE=codex` — use a locally installed, logged-in
//!   `codex` CLI (works in the default build, no feature flag needed).
//!   `SYNAPTIC_EVAL_CODEX_BIN` overrides the binary (default `codex`);
//!   `SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS` overrides the per-call timeout
//!   (default 120s).
//! - `SYNAPTIC_EVAL_JUDGE=llm` or unset — requires the `llm-reasoning`
//!   feature and `SYNAPTIC_EVAL_LLM_URL` / `SYNAPTIC_EVAL_LLM_MODEL`.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use synaptic::{AgentMemory, MemoryConfig};
use synaptic_eval::ablation::{
    self, configs, run_config, AblationRow, AblationTable, ConfigReport, MetricDelta,
};
use synaptic_eval::dataset::{load_locomo, load_longmemeval, EvalConversation};
use synaptic_eval::metrics::{
    aggregate, mrr, precision_at_k, recall_at_k, LatencySummary, QuestionMetrics,
};
use synaptic_eval::{qa, runner};

const DEFAULT_DATASET: &str = "tools/eval/data/locomo10.json";
const K: usize = 10;

fn fmt_latency(name: &str, l: &LatencySummary) -> String {
    format!(
        "{name}: n={} p50={}us p95={}us p99={}us",
        l.count, l.p50_micros, l.p95_micros, l.p99_micros
    )
}

fn load(path: &Path) -> Result<Vec<EvalConversation>, String> {
    let is_lme = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.to_ascii_lowercase().contains("longmemeval"));
    if is_lme {
        load_longmemeval(path).map_err(|e| e.to_string())
    } else {
        load_locomo(path).map_err(|e| e.to_string())
    }
}

/// Per-conversation retrieval evaluation (default `AgentMemory` pipeline),
/// returning per-question metrics plus the raw measured durations so the
/// caller can aggregate percentiles over the whole run.
async fn eval_one_conversation(
    conversation: EvalConversation,
) -> Result<(Vec<QuestionMetrics>, Vec<Duration>, Vec<Duration>), String> {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .map_err(|e| e.to_string())?;
    let ingest_stats = runner::ingest(&conversation, &mut memory)
        .await
        .map_err(|e| e.to_string())?;
    let mut per_question = Vec::with_capacity(conversation.questions.len());
    let mut recall_durations = Vec::with_capacity(conversation.questions.len());
    for question in &conversation.questions {
        let outcome = runner::run_question(question, &memory, K)
            .await
            .map_err(|e| e.to_string())?;
        recall_durations.push(outcome.recall_duration);
        let relevant: HashSet<String> = question.evidence_ids.iter().cloned().collect();
        per_question.push(QuestionMetrics {
            question_id: question.id.clone(),
            qtype: format!("{:?}", question.qtype),
            precision_at_k: precision_at_k(&outcome.retrieved_ids, &relevant, K),
            recall_at_k: recall_at_k(&outcome.retrieved_ids, &relevant, K),
            reciprocal_rank: mrr(&outcome.retrieved_ids, &relevant),
        });
    }
    Ok((per_question, ingest_stats.store_durations, recall_durations))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), String> {
    // Quiet subscriber first: the library's own `try_init` then no-ops, so
    // the measurement loop is not dominated by INFO-level log formatting.
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::WARN)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let grow_100k = args.iter().any(|a| a == "--growth-100k");
    let growth_only = args.iter().any(|a| a == "--growth-only");
    // `--retrieval-only` skips the ablation ladder, growth, and QA (fast
    // retrieval-quality signal). `--max-conversations N` truncates to the first
    // N conversations (a labelled subset for a quick, cheaper measurement).
    let retrieval_only = args.iter().any(|a| a == "--retrieval-only");
    // `--qa-only` runs ONLY the LLM-gated end-to-end QA phase (no retrieval
    // metrics, no ablation ladder, no memory-growth phase). This isolates the
    // slow judge calls so a bounded QA sample fits a short wall-clock budget;
    // the growth phase in particular (10k sequential stores) would otherwise
    // dominate the run before QA starts.
    let qa_only = args.iter().any(|a| a == "--qa-only");
    let max_conversations: Option<usize> = args
        .iter()
        .position(|a| a == "--max-conversations")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    // `--max-questions N` caps EACH conversation to its first N questions (a
    // quick labelled subset; questions within a conversation run sequentially
    // so this bounds wall-clock directly).
    let max_questions: Option<usize> = args
        .iter()
        .position(|a| a == "--max-questions")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    let dataset_path = args
        .iter()
        .find(|a| !a.starts_with("--") && a.parse::<usize>().is_err())
        .cloned()
        .unwrap_or_else(|| DEFAULT_DATASET.to_string());
    let path = Path::new(&dataset_path);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut w = |s: String| {
        writeln!(out, "{s}")
            .and_then(|()| out.flush())
            .map_err(|e| e.to_string())
    };

    let mut conversations = load(path)?;
    if let Some(n) = max_conversations {
        conversations.truncate(n);
    }
    if let Some(q) = max_questions {
        for c in &mut conversations {
            c.questions.truncate(q);
        }
    }
    let n_questions: usize = conversations.iter().map(|c| c.questions.len()).sum();
    let n_turns: usize = conversations
        .iter()
        .map(|c| c.sessions.iter().map(|s| s.turns.len()).sum::<usize>())
        .sum();
    w(format!("== run_eval: dataset={dataset_path} =="))?;
    w(format!(
        "conversations={} sessions={} turns={} questions={} k={K}",
        conversations.len(),
        conversations
            .iter()
            .map(|c| c.sessions.len())
            .sum::<usize>(),
        n_turns,
        n_questions
    ))?;

    // 1. Retrieval quality + latency (default AgentMemory pipeline), one
    //    concurrent task per conversation (independent haystacks).
    if qa_only {
        return run_qa_only(&conversations, &mut w).await;
    }
    if growth_only {
        return run_growth_and_qa(&conversations, grow_100k, &mut w).await;
    }
    w("\n== retrieval (AgentMemory default pipeline) ==".to_string())?;
    let handles: Vec<_> = conversations
        .iter()
        .cloned()
        .map(|c| tokio::spawn(eval_one_conversation(c)))
        .collect();
    let mut per_question: Vec<QuestionMetrics> = Vec::new();
    let mut store_durations: Vec<Duration> = Vec::new();
    let mut recall_durations: Vec<Duration> = Vec::new();
    for handle in handles {
        let (pq, sd, rd) = handle.await.map_err(|e| e.to_string())??;
        per_question.extend(pq);
        store_durations.extend(sd);
        recall_durations.extend(rd);
    }
    let (mean_p, mean_r, mean_rr, by_qtype) = aggregate(&per_question);
    w(format!(
        "questions_evaluated={} mean_precision@{K}={mean_p:.4} mean_recall@{K}={mean_r:.4} MRR={mean_rr:.4}",
        per_question.len()
    ))?;
    w("per-qtype:".to_string())?;
    for (qtype, b) in &by_qtype {
        w(format!(
            "  {qtype}: n={} P@{K}={:.4} R@{K}={:.4} MRR={:.4}",
            b.questions, b.mean_precision_at_k, b.mean_recall_at_k, b.mrr
        ))?;
    }
    w(fmt_latency(
        "store_latency",
        &LatencySummary::from_durations(&store_durations),
    ))?;
    w(fmt_latency(
        "recall_latency",
        &LatencySummary::from_durations(&recall_durations),
    ))?;

    if retrieval_only {
        return Ok(());
    }

    // 2. Capability ablation ladder, one concurrent task per
    //    (config, conversation) pair; per-question rows are merged per config
    //    and re-aggregated (identical numbers to a sequential run, since each
    //    conversation is an independent haystack).
    w(
        "\n== ablation (baseline -> +composite -> +reranker -> +graph_temporal -> +all) =="
            .to_string(),
    )?;
    let ladder = configs();
    let mut merged: Vec<ConfigReport> = Vec::with_capacity(ladder.len());
    for config in &ladder {
        let handles: Vec<_> = conversations
            .iter()
            .map(|c| {
                let config = *config;
                let c = c.clone();
                tokio::spawn(async move { run_config(&config, &[c], K).await })
            })
            .collect();
        let mut per_question: Vec<QuestionMetrics> = Vec::new();
        for handle in handles {
            let report = handle
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            per_question.extend(report.per_question);
        }
        let (mean_p, mean_r, mean_rr, _) = aggregate(&per_question);
        merged.push(ConfigReport {
            name: config.name.to_string(),
            includes: config.includes.to_string(),
            questions_evaluated: per_question.len(),
            mean_precision_at_k: mean_p,
            mean_recall_at_k: mean_r,
            mrr: mean_rr,
            per_question,
        });
    }
    let baseline = merged.remove(0);
    let rows = merged
        .into_iter()
        .map(|report| AblationRow {
            delta_vs_baseline: MetricDelta {
                precision_at_k: report.mean_precision_at_k - baseline.mean_precision_at_k,
                recall_at_k: report.mean_recall_at_k - baseline.mean_recall_at_k,
                mrr: report.mrr - baseline.mrr,
            },
            config: report,
        })
        .collect();
    let table = AblationTable {
        k: K,
        baseline,
        rows,
    };
    w(ablation::to_markdown(&table))?;

    run_growth_and_qa(&conversations, grow_100k, &mut w).await
}

/// Run ONLY the LLM-gated end-to-end QA phase (`--qa-only`): skips retrieval
/// metrics, the ablation ladder, and the memory-growth phase so a bounded QA
/// sample fits a short wall-clock budget. Numbers come from real judge
/// verdicts; without a configured judge this reports not-run, never a number.
async fn run_qa_only(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w("== QA end-to-end accuracy (LLM-gated, --qa-only) ==".to_string())?;
    match qa::run_qa_gated(conversations, K)
        .await
        .map_err(|e| e.to_string())?
    {
        qa::QaResult::NotRun { reason } => w(format!("QA: {reason}"))?,
        qa::QaResult::Ran(r) => {
            w(format!(
                "QA: graded={} correct={} accuracy={:.4}",
                r.questions, r.correct, r.accuracy
            ))?;
            for (qtype, b) in &r.by_qtype {
                w(format!(
                    "  {qtype}: graded={} correct={} accuracy={:.4}",
                    b.questions, b.correct, b.accuracy
                ))?;
            }
        }
    }
    Ok(())
}

/// Memory-growth measurement and gated QA (phases 3 and 4).
async fn run_growth_and_qa(
    conversations: &[EvalConversation],
    grow_100k: bool,
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    // 3. Memory growth. 100k only when explicitly requested (slow). Each
    //    target's row is printed as soon as it is measured, so a run that is
    //    stopped partway still leaves real numbers for completed targets.
    w("== memory growth (payload-byte proxy, not RSS) ==".to_string())?;
    let targets: &[usize] = if grow_100k {
        &[1_000, 10_000, 100_000]
    } else {
        &[1_000, 10_000]
    };
    for &target in targets {
        let growth = runner::measure_growth(&[target])
            .await
            .map_err(|e| e.to_string())?;
        for g in &growth {
            w(format!(
                "target={} stored={} payload_bytes={} lib_total_size={} store_p50={}us store_p95={}us store_p99={}us probe_search={}us",
                g.target_memories,
                g.stored_memories,
                g.stored_bytes_estimate,
                g.library_reported_total_size,
                g.store_latency.p50_micros,
                g.store_latency.p95_micros,
                g.store_latency.p99_micros,
                g.probe_search_micros
            ))?;
        }
    }
    if !grow_100k {
        w("target=100000: not run (pass --growth-100k to measure)".to_string())?;
    }

    // 4. QA end-to-end accuracy (gated; NotRun without endpoint).
    w("\n== QA end-to-end accuracy (LLM-gated) ==".to_string())?;
    match qa::run_qa_gated(conversations, K)
        .await
        .map_err(|e| e.to_string())?
    {
        qa::QaResult::NotRun { reason } => w(format!("QA: {reason}"))?,
        qa::QaResult::Ran(r) => {
            w(format!(
                "QA: graded={} correct={} accuracy={:.4}",
                r.questions, r.correct, r.accuracy
            ))?;
        }
    }

    Ok(())
}
