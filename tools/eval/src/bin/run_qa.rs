//! End-to-end QA accuracy on a stratified LoCoMo subset, judged by the
//! `codex` CLI ([`synaptic_eval::qa::CodexCliJudge`]).
//!
//! Usage:
//!   cargo run --release -p synaptic-eval --bin run_qa -- \
//!       [--subset N] [--k K] [--concurrency C] [--data PATH]
//!
//! Pipeline per question: recall up to K memories from an `AgentMemory`
//! ingested with the question's full conversation, ask codex to answer from
//! the recalled snippets only, then ask codex to grade that answer against
//! the gold answer (YES/NO). Accuracy is reported overall and per QType.
//!
//! Honesty contract: every number printed comes from real codex verdicts in
//! this process's run. Judge failures are counted and reported; accuracy is
//! computed over completed (graded) questions only, and the completed count
//! is always printed alongside it. Nothing is fabricated.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use synaptic::{AgentMemory, MemoryConfig};
use synaptic_eval::dataset::{load_locomo, EvalQuestion, QType};
use synaptic_eval::qa::{CodexCliJudge, Judge};
use synaptic_eval::runner::ingest;

const ALL_QTYPES: [QType; 6] = [
    QType::SingleHop,
    QType::MultiHop,
    QType::Temporal,
    QType::KnowledgeUpdate,
    QType::OpenDomain,
    QType::Abstention,
];

struct Args {
    subset: usize,
    k: usize,
    concurrency: usize,
    data: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        subset: 150,
        k: 10,
        concurrency: 4,
        data: PathBuf::from("tools/eval/data/locomo10.json"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("missing value for {name}"));
        match flag.as_str() {
            "--subset" => args.subset = val("--subset")?.parse().map_err(|e| format!("{e}"))?,
            "--k" => args.k = val("--k")?.parse().map_err(|e| format!("{e}"))?,
            "--concurrency" => {
                args.concurrency = val("--concurrency")?.parse().map_err(|e| format!("{e}"))?
            }
            "--data" => args.data = PathBuf::from(val("--data")?),
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if args.subset == 0 || args.k == 0 || args.concurrency == 0 {
        return Err("--subset, --k and --concurrency must be > 0".to_string());
    }
    Ok(args)
}

/// Stratified subset: split the target size evenly across the QTypes present
/// in the data (remainder to the first types), then pick evenly spaced
/// questions within each type (deterministic, no RNG).
fn stratified_subset(
    per_type: &BTreeMap<String, Vec<(usize, EvalQuestion)>>,
    target: usize,
) -> Vec<(usize, EvalQuestion)> {
    let types: Vec<&String> = per_type.keys().collect();
    if types.is_empty() {
        return Vec::new();
    }
    let base = target / types.len();
    let rem = target % types.len();
    let mut selected = Vec::new();
    for (i, ty) in types.iter().enumerate() {
        let pool = &per_type[*ty];
        let quota = (base + usize::from(i < rem)).min(pool.len());
        if quota == 0 {
            continue;
        }
        // Evenly spaced indices across the pool.
        for j in 0..quota {
            let idx = j * pool.len() / quota;
            selected.push(pool[idx].clone());
        }
    }
    selected
}

struct Graded {
    qtype: QType,
    correct: bool,
}

#[tokio::main]
async fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let judge = Arc::new(CodexCliJudge::from_env());
    if !judge.is_available() {
        eprintln!(
            "error: codex CLI not available (checked `codex --version`); not fabricating results"
        );
        std::process::exit(1);
    }

    let conversations = match load_locomo(&args.data) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load {}: {e}", args.data.display());
            std::process::exit(1);
        }
    };
    let total_questions: usize = conversations.iter().map(|c| c.questions.len()).sum();

    // Group all questions by QType, keeping the owning conversation index.
    let mut per_type: BTreeMap<String, Vec<(usize, EvalQuestion)>> = BTreeMap::new();
    for (ci, conv) in conversations.iter().enumerate() {
        for q in &conv.questions {
            per_type
                .entry(format!("{:?}", q.qtype))
                .or_default()
                .push((ci, q.clone()));
        }
    }
    let selected = stratified_subset(&per_type, args.subset);
    eprintln!(
        "dataset: {} conversations, {} questions total; stratified subset: {} questions (target {})",
        conversations.len(),
        total_questions,
        selected.len(),
        args.subset
    );

    // Phase 1: ingest each needed conversation and recall for its selected
    // questions (concurrent across conversations).
    let mut by_conv: BTreeMap<usize, Vec<EvalQuestion>> = BTreeMap::new();
    for (ci, q) in &selected {
        by_conv.entry(*ci).or_default().push(q.clone());
    }
    let mut recall_set = tokio::task::JoinSet::new();
    for (ci, questions) in by_conv {
        let conv = conversations[ci].clone();
        let k = args.k;
        recall_set.spawn(async move {
            let mut memory = AgentMemory::new(MemoryConfig::default())
                .await
                .map_err(|e| format!("memory init failed: {e}"))?;
            ingest(&conv, &mut memory)
                .await
                .map_err(|e| format!("ingest of {} failed: {e}", conv.id))?;
            eprintln!(
                "ingested conversation {} ({} sessions)",
                conv.id,
                conv.sessions.len()
            );
            let mut out = Vec::new();
            for q in questions {
                let recalled: Vec<String> = memory
                    .search(&q.text, k)
                    .await
                    .map_err(|e| format!("search failed: {e}"))?
                    .into_iter()
                    .map(|f| f.entry.value)
                    .collect();
                out.push((q, recalled));
            }
            Ok::<_, String>(out)
        });
    }
    let mut jobs: Vec<(EvalQuestion, Vec<String>)> = Vec::new();
    while let Some(res) = recall_set.join_next().await {
        match res {
            Ok(Ok(mut v)) => jobs.append(&mut v),
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("error: recall task panicked: {e}");
                std::process::exit(1);
            }
        }
    }

    // Phase 2: judge with the codex CLI, bounded concurrency, progress to
    // stderr. Failures are recorded, never converted into a verdict.
    let n = jobs.len();
    eprintln!(
        "recall done; judging {n} questions with codex (concurrency {})",
        args.concurrency
    );
    let mut job_iter = jobs.into_iter();
    let mut set = tokio::task::JoinSet::new();
    let mut done = 0usize;
    let mut failures = 0usize;
    let mut graded: Vec<Graded> = Vec::new();
    let spawn_one = |set: &mut tokio::task::JoinSet<Result<Graded, String>>,
                     judge: Arc<CodexCliJudge>,
                     q: EvalQuestion,
                     recalled: Vec<String>| {
        set.spawn(async move {
            let predicted = judge
                .answer(&q.text, &recalled)
                .await
                .map_err(|e| format!("{} answer: {e}", q.id))?;
            let correct = judge
                .grade(&q.text, &q.gold_answer, &predicted)
                .await
                .map_err(|e| format!("{} grade: {e}", q.id))?;
            Ok(Graded {
                qtype: q.qtype,
                correct,
            })
        });
    };
    for _ in 0..args.concurrency {
        if let Some((q, recalled)) = job_iter.next() {
            spawn_one(&mut set, Arc::clone(&judge), q, recalled);
        }
    }
    while let Some(res) = set.join_next().await {
        done += 1;
        match res {
            Ok(Ok(g)) => graded.push(g),
            Ok(Err(e)) => {
                failures += 1;
                eprintln!("judge failure: {e}");
            }
            Err(e) => {
                failures += 1;
                eprintln!("judge task panicked: {e}");
            }
        }
        eprint!("\rquestion {done}/{n} (failures: {failures})");
        let _ = std::io::stderr().flush();
        if let Some((q, recalled)) = job_iter.next() {
            spawn_one(&mut set, Arc::clone(&judge), q, recalled);
        }
    }
    eprintln!();

    // Aggregate over completed (graded) questions only.
    let completed = graded.len();
    let correct = graded.iter().filter(|g| g.correct).count();
    println!("== End-to-end QA accuracy (LoCoMo stratified subset, codex CLI judge) ==");
    println!(
        "subset target: {} | selected: {n} | completed (graded): {completed} | judge failures: {failures} | k={}",
        args.subset, args.k
    );
    if completed == 0 {
        println!("no questions completed — no accuracy to report");
        std::process::exit(1);
    }
    println!(
        "overall: {correct}/{completed} = {:.1}%",
        100.0 * correct as f64 / completed as f64
    );
    for ty in ALL_QTYPES {
        let of_type: Vec<&Graded> = graded.iter().filter(|g| g.qtype == ty).collect();
        if of_type.is_empty() {
            continue;
        }
        let c = of_type.iter().filter(|g| g.correct).count();
        println!(
            "  {:?}: {c}/{} = {:.1}%",
            ty,
            of_type.len(),
            100.0 * c as f64 / of_type.len() as f64
        );
    }
}
