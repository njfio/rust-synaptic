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
    // `--agentic-qa` runs ONLY the agentic answer-guided retrieval QA mode:
    // the judge drives retrieval itself (retrieve -> answer-or-search ->
    // retrieve -> ... -> answer -> grade) instead of a single retrieve-then-
    // answer pass. Like `--qa-only`, skips retrieval metrics, ablation, and
    // growth. Requires SYNAPTIC_EVAL_JUDGE=codex; NotRun otherwise (no LLM
    // reasoning-feature judge is wired up for this mode since it needs
    // free-form multi-turn completions, not the fixed answer/grade shape).
    let agentic_qa = args.iter().any(|a| a == "--agentic-qa");
    // `--recall-curve` measures recall@{10,20,50} from a single limit-50 search
    // per question. It answers whether below-rank-10 gold is in the candidate
    // pool (reranking headroom) or absent (first-stage recall problem). No judge.
    let recall_curve = args.iter().any(|a| a == "--recall-curve");
    // `--completeness` measures STRICT full-evidence recall (ALL evidence turns
    // of a question in the top-k, not just any) vs the standard partial recall,
    // broken down by how many evidence turns a question has. It quantifies how
    // much of the "gold-retrieved-but-answer-wrong" QA bucket is really
    // multi-evidence undersupply (retrieval completeness) rather than the judge.
    let completeness = args.iter().any(|a| a == "--completeness");
    // `--forget-curve` measures whether principled forgetting (retained_strength
    // = decay·importance·recency) preserves recall better than random eviction:
    // differentiate access via the question searches, then rank turns by strength
    // and by a deterministic pseudo-random key, keep the top fraction of each,
    // and compare recall@k of the two survivor sets across store sizes.
    let forget_curve = args.iter().any(|a| a == "--forget-curve");
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
    // `--qtype NAME` keeps only questions whose category matches NAME
    // (case-insensitive: multihop|temporal|opendomain|singlehop|abstention),
    // applied BEFORE `--max-questions`. Lets a run target a single category
    // (e.g. abstention, which is clustered at the end of each conversation and
    // otherwise never reached by first-N sampling).
    let qtype_filter: Option<String> = args
        .iter()
        .position(|a| a == "--qtype")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.to_ascii_lowercase());
    // `--facts-from FILE` swaps the haystack from raw conversation turns to
    // LLM-extracted facts (JSON: {"<conv_index>": ["fact", ...]}), for the
    // write-time-extraction experiment. Combine with `--qa-only`/`--agentic-qa`.
    let facts_from: Option<String> = args
        .iter()
        .position(|a| a == "--facts-from")
        .and_then(|i| args.get(i + 1))
        .cloned();
    // `--extract-facts` builds the haystack live with synaptic's own
    // LlmReasoner (needs `--features llm-reasoning` + SYNAPTIC_LLM_URL/MODEL).
    let extract_facts = args.iter().any(|a| a == "--extract-facts");
    // `--reflect` triggers reflection/synthesis after ingestion so insight
    // memories join the QA haystack (measures capability 2 of the v2 spec).
    let reflect = args.iter().any(|a| a == "--reflect");
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
    if let Some(ref want) = qtype_filter {
        for c in &mut conversations {
            c.questions
                .retain(|q| format!("{:?}", q.qtype).to_ascii_lowercase() == *want);
        }
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
    if recall_curve {
        return run_recall_curve(&conversations, &mut w).await;
    }
    if completeness {
        return run_completeness(&conversations, &mut w).await;
    }
    if forget_curve {
        return run_forget_curve(&conversations, &mut w).await;
    }
    // Build the extracted-fact haystack once (from a live LlmReasoner or a
    // JSON file) when a fact-QA mode is requested.
    let facts_map: Option<std::collections::BTreeMap<usize, Vec<String>>> =
        if (qa_only || agentic_qa) && (extract_facts || facts_from.is_some()) {
            if extract_facts {
                Some(extract_facts_via_reasoner(&conversations, &mut w).await?)
            } else {
                Some(load_facts(facts_from.as_deref().unwrap_or_default())?)
            }
        } else {
            None
        };
    if qa_only {
        if let Some(ref facts) = facts_map {
            return run_qa_facts_only(&conversations, facts, &mut w).await;
        }
        if reflect {
            return run_qa_reflect_only(&conversations, &mut w).await;
        }
        return run_qa_only(&conversations, &mut w).await;
    }
    if agentic_qa {
        if let Some(ref facts) = facts_map {
            return run_agentic_facts_only(&conversations, facts, &mut w).await;
        }
        return run_agentic_qa_only(&conversations, &mut w).await;
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

/// Per-question completeness record: `(evidence_count, partial@10, full@10, full@50)`.
type CompletenessRec = (usize, f64, usize, usize);

/// Completeness records for one conversation. `full@10` uses a PRODUCTION
/// limit-10 search (so any final-stage selection like MMR / iterative expansion
/// is exercised); `full@50` uses a limit-50 search (the candidate-pool ceiling).
async fn completeness_one_conversation(
    conversation: EvalConversation,
) -> Result<Vec<CompletenessRec>, String> {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .map_err(|e| e.to_string())?;
    runner::ingest(&conversation, &mut memory)
        .await
        .map_err(|e| e.to_string())?;
    let mut recs = Vec::new();
    for question in &conversation.questions {
        let n_ev = question.evidence_ids.len();
        if n_ev == 0 {
            continue;
        }
        let prod = runner::run_question(question, &memory, 10)
            .await
            .map_err(|e| e.to_string())?;
        let pool = runner::run_question(question, &memory, 50)
            .await
            .map_err(|e| e.to_string())?;
        let relevant: HashSet<&String> = question.evidence_ids.iter().collect();
        let top10: HashSet<&String> = prod.retrieved_ids.iter().take(10).collect();
        let top50: HashSet<&String> = pool.retrieved_ids.iter().take(50).collect();
        let hits10 = relevant.iter().filter(|e| top10.contains(*e)).count();
        let full10 = usize::from(hits10 == n_ev);
        let full50 = usize::from(relevant.iter().all(|e| top50.contains(*e)));
        let partial10 = hits10 as f64 / n_ev as f64;
        recs.push((n_ev, partial10, full10, full50));
    }
    Ok(recs)
}

/// Measure STRICT full-evidence recall (`--completeness`): does the top-k contain
/// ALL of a question's evidence turns (full), and what fraction (partial)?
/// Bucketed by evidence-count so multi-evidence undersupply is visible. One
/// concurrent task per conversation; no judge.
async fn run_completeness(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    // bucket key by evidence count: 1, 2, 3, "4+". Value: (n, partial@10 sum,
    // full@10 count, full@50 count).
    let mut buckets: std::collections::BTreeMap<String, (usize, f64, usize, usize)> =
        std::collections::BTreeMap::new();
    let mut overall = (0usize, 0.0f64, 0usize, 0usize);

    let handles: Vec<_> = conversations
        .iter()
        .cloned()
        .map(|c| tokio::spawn(completeness_one_conversation(c)))
        .collect();
    for handle in handles {
        let recs = handle.await.map_err(|e| e.to_string())??;
        for (n_ev, partial10, full10, full50) in recs {
            let key = if n_ev >= 4 {
                "4+".to_string()
            } else {
                n_ev.to_string()
            };
            let b = buckets.entry(key).or_insert((0, 0.0, 0, 0));
            b.0 += 1;
            b.1 += partial10;
            b.2 += full10;
            b.3 += full50;
            overall.0 += 1;
            overall.1 += partial10;
            overall.2 += full10;
            overall.3 += full50;
        }
    }

    w("== evidence completeness (full = ALL evidence turns in top-k) ==".to_string())?;
    if overall.0 == 0 {
        w("no questions with labeled evidence".to_string())?;
        return Ok(());
    }
    w("by evidence-count: n | partial_recall@10 | full@10 | full@50".to_string())?;
    for (key, (n, psum, f10, f50)) in &buckets {
        w(format!(
            "  {key}-evidence: n={n} partial@10={:.4} full@10={:.4} full@50={:.4}",
            psum / *n as f64,
            *f10 as f64 / *n as f64,
            *f50 as f64 / *n as f64
        ))?;
    }
    w(format!(
        "overall: n={} partial@10={:.4} full@10={:.4} full@50={:.4}",
        overall.0,
        overall.1 / overall.0 as f64,
        overall.2 as f64 / overall.0 as f64,
        overall.3 as f64 / overall.0 as f64
    ))?;
    Ok(())
}

/// `--forget-curve`: measure whether principled forgetting preserves recall
/// better than random eviction. Access is differentiated by running each
/// question's search over the full memory (retrieved turns are "accessed");
/// turns are then ranked by `ForgettingPolicy::retained_strength`
/// (decay·importance·recency) and, separately, by a deterministic pseudo-random
/// key. For each target store fraction we keep the top slice of each ranking,
/// rebuild a fresh memory from the survivors, and compare recall@k. If
/// forgetting > random, the strength ranking is selective; on uniformly-ingested
/// LoCoMo (equal age/importance) the only signal is access, so the two converge.
async fn run_forget_curve(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    use std::collections::{HashMap, HashSet};
    use synaptic::memory::forgetting::ForgettingPolicy;
    use synaptic::{MemoryEntry, MemoryType};

    w(
        "== Forgetting retention curve (recall@k: strength-ranked vs random eviction) =="
            .to_string(),
    )?;
    let policy = ForgettingPolicy::default();
    let fractions = [1.0_f64, 0.75, 0.5, 0.25];
    let mut fg_recall = vec![0.0_f64; fractions.len()];
    let mut rnd_recall = vec![0.0_f64; fractions.len()];
    let mut q_counts = vec![0_usize; fractions.len()];

    // Deterministic FNV-1a hash for a reproducible "random" ordering that is
    // uncorrelated with retained strength.
    fn fnv(key: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in key.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    for conv in conversations {
        let mut full = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(|e| e.to_string())?;
        runner::ingest(conv, &mut full)
            .await
            .map_err(|e| e.to_string())?;

        // Reconstruct the ingested turns (same keys ingest used).
        let mut turns: Vec<(String, String)> = Vec::new();
        for session in &conv.sessions {
            for (i, turn) in session.turns.iter().enumerate() {
                turns.push((runner::turn_memory_key(session, i), turn.text.clone()));
            }
        }

        // Differentiate access: retrieved turns get "accessed".
        let mut access: HashMap<String, u64> = HashMap::new();
        for q in &conv.questions {
            for f in full.search(&q.text, K).await.map_err(|e| e.to_string())? {
                *access.entry(f.entry.key).or_insert(0) += 1;
            }
        }

        // Score each turn by retained strength.
        let mut scored: Vec<(String, String, f64)> = Vec::with_capacity(turns.len());
        for (key, value) in &turns {
            let mut e = MemoryEntry::new(key.clone(), value.clone(), MemoryType::LongTerm);
            for _ in 0..access.get(key).copied().unwrap_or(0) {
                e.mark_accessed();
            }
            let s = policy
                .retained_strength(&e)
                .await
                .map_err(|err| err.to_string())?;
            scored.push((key.clone(), value.clone(), s));
        }
        let n = scored.len();
        let mut by_strength: Vec<usize> = (0..n).collect();
        by_strength.sort_by(|&a, &b| {
            scored[b]
                .2
                .partial_cmp(&scored[a].2)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut by_rand: Vec<usize> = (0..n).collect();
        by_rand.sort_by_key(|&i| fnv(&scored[i].0));

        for (fi, &frac) in fractions.iter().enumerate() {
            let keep = (((frac * n as f64).round()) as usize).clamp(1, n);
            let fg_keep: HashSet<&String> =
                by_strength[..keep].iter().map(|&i| &scored[i].0).collect();
            let rnd_keep: HashSet<&String> =
                by_rand[..keep].iter().map(|&i| &scored[i].0).collect();

            let mut fg_mem = AgentMemory::new(MemoryConfig::default())
                .await
                .map_err(|e| e.to_string())?;
            let mut rnd_mem = AgentMemory::new(MemoryConfig::default())
                .await
                .map_err(|e| e.to_string())?;
            for (key, value, _) in &scored {
                if fg_keep.contains(key) {
                    fg_mem.store(key, value).await.map_err(|e| e.to_string())?;
                }
                if rnd_keep.contains(key) {
                    rnd_mem.store(key, value).await.map_err(|e| e.to_string())?;
                }
            }

            for q in &conv.questions {
                if q.evidence_ids.is_empty() {
                    continue;
                }
                let relevant: HashSet<String> = q.evidence_ids.iter().cloned().collect();
                let fg_keys: Vec<String> = fg_mem
                    .search(&q.text, K)
                    .await
                    .map_err(|e| e.to_string())?
                    .iter()
                    .map(|f| f.entry.key.clone())
                    .collect();
                let rnd_keys: Vec<String> = rnd_mem
                    .search(&q.text, K)
                    .await
                    .map_err(|e| e.to_string())?
                    .iter()
                    .map(|f| f.entry.key.clone())
                    .collect();
                fg_recall[fi] += recall_at_k(&runner::normalize_retrieved(&fg_keys), &relevant, K);
                rnd_recall[fi] +=
                    recall_at_k(&runner::normalize_retrieved(&rnd_keys), &relevant, K);
                q_counts[fi] += 1;
            }
        }
    }

    for (fi, &frac) in fractions.iter().enumerate() {
        let c = q_counts[fi].max(1) as f64;
        w(format!(
            "  keep={:>3.0}%  forgetting recall@{K}={:.4}  random recall@{K}={:.4}  delta={:+.4}  (n={})",
            frac * 100.0,
            fg_recall[fi] / c,
            rnd_recall[fi] / c,
            (fg_recall[fi] - rnd_recall[fi]) / c,
            q_counts[fi]
        ))?;
    }
    Ok(())
}

/// Measure the recall@k curve (`--recall-curve`): one limit-50 search per
/// question, then recall@10/@20/@50 computed from that single ranked list. A
/// large gap between recall@10 and recall@50 means below-rank-10 gold IS in the
/// candidate pool (a reranker over a wider pool could recover it); a flat curve
/// means the gold isn't retrieved at all (a first-stage recall problem).
async fn run_recall_curve(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    const KS: [usize; 3] = [10, 20, 50];
    const KMAX: usize = 50;
    // (n, sum_recall@10, sum_recall@20, sum_recall@50) per qtype + overall.
    let mut overall = [0.0f64; 3];
    let mut n_total = 0usize;
    let mut by_qtype: std::collections::BTreeMap<String, (usize, [f64; 3])> =
        std::collections::BTreeMap::new();

    for conversation in conversations {
        let mut memory = AgentMemory::new(MemoryConfig::default())
            .await
            .map_err(|e| e.to_string())?;
        runner::ingest(conversation, &mut memory)
            .await
            .map_err(|e| e.to_string())?;
        for question in &conversation.questions {
            if question.evidence_ids.is_empty() {
                continue; // no labeled gold → not a recall data point
            }
            let outcome = runner::run_question(question, &memory, KMAX)
                .await
                .map_err(|e| e.to_string())?;
            let relevant: HashSet<String> = question.evidence_ids.iter().cloned().collect();
            let entry = by_qtype
                .entry(format!("{:?}", question.qtype))
                .or_insert((0, [0.0; 3]));
            entry.0 += 1;
            n_total += 1;
            for (i, &k) in KS.iter().enumerate() {
                let r = recall_at_k(&outcome.retrieved_ids, &relevant, k);
                overall[i] += r;
                entry.1[i] += r;
            }
        }
    }

    w("== recall@k curve (single limit-50 search per question) ==".to_string())?;
    if n_total == 0 {
        w("no questions with labeled evidence".to_string())?;
        return Ok(());
    }
    w(format!(
        "overall n={n_total} recall@10={:.4} recall@20={:.4} recall@50={:.4}",
        overall[0] / n_total as f64,
        overall[1] / n_total as f64,
        overall[2] / n_total as f64
    ))?;
    for (qtype, (n, sums)) in &by_qtype {
        w(format!(
            "  {qtype}: n={n} recall@10={:.4} recall@20={:.4} recall@50={:.4}",
            sums[0] / *n as f64,
            sums[1] / *n as f64,
            sums[2] / *n as f64
        ))?;
    }
    Ok(())
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
            write_recall_breakdown(&r.recall_breakdown, &mut *w)?;
            write_faithfulness_breakdown(&r.faithfulness, &mut *w)?;
        }
    }
    Ok(())
}

/// `--qa-only --reflect`: QA after triggering reflection so synthesized insight
/// memories join the haystack. Requires the codex judge.
async fn run_qa_reflect_only(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w("== QA end-to-end accuracy (--qa-only --reflect: +synthesized insights) ==".to_string())?;
    if std::env::var(qa::ENV_JUDGE).ok().as_deref() != Some("codex") {
        w(format!(
            "QA: not run — --reflect requires {}=codex with the codex CLI",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let judge = qa::CodexCliJudge::from_env();
    if !judge.is_available() {
        w(format!(
            "QA: not run — {}=codex but codex CLI not runnable",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let (r, insights) = qa::run_qa_reflect(conversations, &judge, K)
        .await
        .map_err(|e| e.to_string())?;
    w(format!(
        "insights synthesized (stored + retrievable): {insights}"
    ))?;
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
    write_recall_breakdown(&r.recall_breakdown, &mut *w)?;
    write_faithfulness_breakdown(&r.faithfulness, &mut *w)?;
    Ok(())
}

/// Run ONLY the agentic answer-guided retrieval QA mode (`--agentic-qa`):
/// the judge drives retrieval itself instead of a single retrieve-then-
/// answer pass. Requires `SYNAPTIC_EVAL_JUDGE=codex` with an available
/// codex CLI — this mode needs free-form multi-line completions
/// (`ANSWER:`/`SEARCH:`) that the fixed-shape `LlmJudge::answer`/`grade`
/// endpoints don't support, so no `llm-reasoning` fallback is offered;
/// without codex this reports not-run, never a fabricated number.
/// `SYNAPTIC_EVAL_AGENTIC_ROUNDS` overrides the max rounds per question
/// (default 3; see [`qa::agentic_max_rounds`]).
async fn run_agentic_qa_only(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w("== Agentic answer-guided retrieval QA (--agentic-qa) ==".to_string())?;
    let selected = std::env::var(qa::ENV_JUDGE).ok();
    if selected.as_deref() != Some("codex") {
        w(format!(
            "QA: not run — --agentic-qa requires {}=codex with the codex CLI installed",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let judge = qa::CodexCliJudge::from_env();
    if !judge.is_available() {
        w(format!(
            "QA: not run — {}=codex but codex CLI not runnable — install/login \
             the codex CLI or set {}",
            qa::ENV_JUDGE,
            qa::CodexCliJudge::ENV_BIN
        ))?;
        return Ok(());
    }
    let max_rounds = qa::agentic_max_rounds();
    let grounded = qa::grounded_enabled();
    let ground_verify = qa::ground_verify_enabled();
    let report = qa::run_agentic_qa(
        conversations,
        &judge,
        K,
        max_rounds,
        grounded,
        ground_verify,
    )
    .await
    .map_err(|e| e.to_string())?;
    w(format!(
        "QA: graded={} correct={} accuracy={:.4} max_rounds={} mean_rounds_used={:.2} \
         gold_added_by_followup_fraction={:.4} grounded={} ungrounded_overrides={} \
         ground_verify={} support_overrides={}",
        report.questions,
        report.correct,
        report.accuracy,
        report.max_rounds,
        report.mean_rounds_used,
        report.gold_added_by_followup_fraction,
        report.grounded,
        report.ungrounded_overrides,
        report.ground_verify,
        report.support_overrides
    ))?;
    for (qtype, b) in &report.by_qtype {
        w(format!(
            "  {qtype}: graded={} correct={} accuracy={:.4}",
            b.questions, b.correct, b.accuracy
        ))?;
    }
    write_recall_breakdown(&report.recall_breakdown, &mut *w)?;
    write_faithfulness_breakdown(&report.faithfulness, &mut *w)?;
    Ok(())
}

/// Build the fact haystack live, using synaptic's own [`LlmReasoner`] to
/// extract facts from each session (date-prefixed so the extractor anchors
/// dates to when things were said). Requires the `llm-reasoning` feature and
/// `SYNAPTIC_LLM_URL` + `SYNAPTIC_LLM_MODEL` pointing at an OpenAI-compatible
/// endpoint (e.g. the codex shim). Returns the same `{conv_index: [facts]}`
/// map shape as [`load_facts`], so the QA path is identical.
#[cfg(feature = "llm-reasoning")]
async fn extract_facts_via_reasoner(
    conversations: &[EvalConversation],
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<std::collections::BTreeMap<usize, Vec<String>>, String> {
    use synaptic::memory::reasoning::{ExtractionContext, LlmReasoner, MemoryReasoner};
    let reasoner = LlmReasoner::from_env().ok_or_else(|| {
        format!(
            "--extract-facts requires {} + {} (OpenAI-compatible endpoint)",
            LlmReasoner::ENV_URL,
            LlmReasoner::ENV_MODEL
        )
    })?;
    let mut out = std::collections::BTreeMap::new();
    for (ci, conv) in conversations.iter().enumerate() {
        let mut facts: Vec<String> = Vec::new();
        for session in &conv.sessions {
            let date = session.timestamp.as_deref().unwrap_or("");
            let mut text = String::new();
            for turn in &session.turns {
                text.push_str(&format!("[{date}] {}: {}\n", turn.speaker, turn.text));
            }
            let ctx = ExtractionContext {
                source_key: format!("conv{ci}"),
                timestamp: chrono::Utc::now(),
            };
            match reasoner.extract(&text, &ctx).await {
                Ok(ex) => facts.extend(ex.facts.into_iter().map(|f| f.text)),
                Err(e) => w(format!("  extract error conv{ci}: {e}"))?,
            }
        }
        w(format!("conv{ci}: extracted {} facts", facts.len()))?;
        out.insert(ci, facts);
    }
    Ok(out)
}

#[cfg(not(feature = "llm-reasoning"))]
async fn extract_facts_via_reasoner(
    _conversations: &[EvalConversation],
    _w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<std::collections::BTreeMap<usize, Vec<String>>, String> {
    Err("--extract-facts requires building with --features llm-reasoning".to_string())
}

/// Load `--facts-from` JSON (`{"<conv_index>": ["fact", ...]}`) into a map
/// from conversation position to its extracted facts.
fn load_facts(path: &str) -> Result<std::collections::BTreeMap<usize, Vec<String>>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("--facts-from {path}: {e}"))?;
    let parsed: std::collections::BTreeMap<String, Vec<String>> =
        serde_json::from_str(&raw).map_err(|e| format!("--facts-from {path}: {e}"))?;
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in parsed {
        let idx = k
            .parse::<usize>()
            .map_err(|e| format!("--facts-from key {k:?} not a conv index: {e}"))?;
        out.insert(idx, v);
    }
    Ok(out)
}

/// `--qa-only` with an extracted-fact haystack (from `--facts-from` or
/// `--extract-facts`): single-shot QA.
async fn run_qa_facts_only(
    conversations: &[EvalConversation],
    facts: &std::collections::BTreeMap<usize, Vec<String>>,
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w("== QA end-to-end accuracy (--qa-only: extracted-fact haystack) ==".to_string())?;
    let n: usize = facts.values().map(|v| v.len()).sum();
    w(format!("facts: {n} across {} conversations", facts.len()))?;
    if std::env::var(qa::ENV_JUDGE).ok().as_deref() != Some("codex") {
        w(format!(
            "QA: not run — --facts-from requires {}=codex with the codex CLI",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let judge = qa::CodexCliJudge::from_env();
    if !judge.is_available() {
        w(format!(
            "QA: not run — {}=codex but codex CLI not runnable",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let r = qa::run_qa_over_facts(conversations, facts, &judge, K)
        .await
        .map_err(|e| e.to_string())?;
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
    write_faithfulness_breakdown(&r.faithfulness, &mut *w)?;
    Ok(())
}

/// `--agentic-qa` with an extracted-fact haystack: agentic QA.
async fn run_agentic_facts_only(
    conversations: &[EvalConversation],
    facts: &std::collections::BTreeMap<usize, Vec<String>>,
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w("== Agentic QA (--agentic-qa: extracted-fact haystack) ==".to_string())?;
    let n: usize = facts.values().map(|v| v.len()).sum();
    w(format!("facts: {n} across {} conversations", facts.len()))?;
    if std::env::var(qa::ENV_JUDGE).ok().as_deref() != Some("codex") {
        w(format!(
            "QA: not run — fact QA requires {}=codex with the codex CLI",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let judge = qa::CodexCliJudge::from_env();
    if !judge.is_available() {
        w(format!(
            "QA: not run — {}=codex but codex CLI not runnable",
            qa::ENV_JUDGE
        ))?;
        return Ok(());
    }
    let report = qa::run_agentic_qa_over_facts(
        conversations,
        facts,
        &judge,
        K,
        qa::agentic_max_rounds(),
        qa::grounded_enabled(),
        qa::ground_verify_enabled(),
    )
    .await
    .map_err(|e| e.to_string())?;
    w(format!(
        "QA: graded={} correct={} accuracy={:.4} mean_rounds_used={:.2}",
        report.questions, report.correct, report.accuracy, report.mean_rounds_used
    ))?;
    for (qtype, b) in &report.by_qtype {
        w(format!(
            "  {qtype}: graded={} correct={} accuracy={:.4}",
            b.questions, b.correct, b.accuracy
        ))?;
    }
    write_faithfulness_breakdown(&report.faithfulness, &mut *w)?;
    Ok(())
}

/// Print the confabulation-vs-abstention breakdown (see
/// [`qa::FaithfulnessBreakdown`]): abstention-category abstain rate (the
/// cleanest confabulation test), the evidence-missing faithful/confabulated/
/// answered-correct split, and over-abstention on answerable questions.
fn write_faithfulness_breakdown(
    f: &qa::FaithfulnessBreakdown,
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w(format!(
        "  faithfulness: abstention_qtype total={} abstained={} confabulated={}",
        f.abstention_qtype_total, f.abstention_qtype_abstained, f.abstention_qtype_confabulated
    ))?;
    w(format!(
        "  faithfulness: evidence_missing total={} abstained={} confabulated={} answered_correct={}",
        f.evidence_missing_total,
        f.evidence_missing_abstained,
        f.evidence_missing_confabulated,
        f.evidence_missing_answered_correct
    ))?;
    w(format!(
        "  faithfulness: over_abstention={}",
        f.over_abstention
    ))?;
    Ok(())
}

/// Print the gold-retrieval-vs-judge-verdict cross-tab: A/B/C/D counts, the
/// gold-retrieved rate, and — of the wrong answers — the retrieval-bound vs
/// judge-bound fractions. Separates RETRIEVAL-bound loss (gold evidence
/// never recalled) from JUDGE-bound loss (evidence recalled, judge still
/// wrong).
fn write_recall_breakdown(
    tab: &qa::QaRecallBreakdown,
    w: &mut impl FnMut(String) -> Result<(), String>,
) -> Result<(), String> {
    w(format!(
        "  recall-vs-judge cross-tab: a(gold_retrieved&correct)={} b(gold_retrieved&wrong)={} \
         c(no_gold&correct)={} d(no_gold&wrong)={} no_gold_labeled={}",
        tab.a, tab.b, tab.c, tab.d, tab.no_gold_labeled
    ))?;
    w(format!(
        "  gold_retrieved rate={:.4} (of wrong answers: retrieval_bound={:.4} judge_bound={:.4})",
        tab.gold_retrieved_rate(),
        tab.retrieval_bound_fraction(),
        tab.judge_bound_fraction()
    ))?;
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
            write_recall_breakdown(&r.recall_breakdown, &mut *w)?;
        }
    }

    Ok(())
}
