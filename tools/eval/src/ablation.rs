//! Capability ablation harness: run the LLM-free metric suite (Task 7.2)
//! under a ladder of retrieval configurations and emit a delta table showing
//! what each v2 capability contributes.
//!
//! ## Configs (cumulative ladder — each adds one capability)
//!
//! | name              | retrieval signals                | composite scoring                 | reranker                          |
//! |-------------------|----------------------------------|-----------------------------------|-----------------------------------|
//! | `baseline`        | dense + keyword (RRF fusion)     | off (weights 1.0/0.0/0.0 ≡ pure fused relevance) | none              |
//! | `+composite`      | dense + keyword                  | on (default 0.6/0.2/0.2)          | none                              |
//! | `+reranker`       | dense + keyword                  | on                                | `HeuristicReranker` (no graph)    |
//! | `+graph_temporal` | dense + keyword + graph + temporal | on                              | `HeuristicReranker` (no graph)    |
//! | `+all`            | dense + keyword + graph + temporal | on                              | `HeuristicReranker` (graph-aware) |
//!
//! `baseline` is the closest constructible analog of the pre-v2 pipeline:
//! dense-vector + keyword fusion only, no composite relevance×recency×
//! importance re-scoring (composite weights `{relevance: 1.0, recency: 0.0,
//! importance: 0.0}` make the composite stage the identity on the fused
//! ordering), no reranking, no graph/temporal signals. Result caching is
//! disabled in every config so each question is really re-executed.
//!
//! Each config is built directly from the public pipeline builder
//! (`HybridRetriever` + `PipelineConfig::semantic_focus()` — the same base
//! config `AgentMemory` uses) over the same ingested storage, so the deltas
//! isolate the toggled capability. For `+graph_temporal`/`+all`, ingest also
//! populates a knowledge graph node per stored turn so the graph signal has
//! real structure to score against.
//!
//! **Honesty:** every config is genuinely constructed and executed against
//! the dataset; deltas are `config − baseline` on measured numbers and may be
//! zero or negative on a given dataset. Nothing here fabricates or clamps a
//! delta.

use crate::dataset::EvalConversation;
use crate::metrics::{aggregate, mrr, precision_at_k, recall_at_k, QuestionMetrics};
use crate::runner::{normalize_retrieved, turn_memory_key, RunnerError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use synaptic::memory::embeddings::TfIdfProvider;
use synaptic::memory::knowledge_graph::MemoryKnowledgeGraph;
use synaptic::memory::retrieval::{
    CompositeWeights, DenseVectorRetriever, GraphRetriever, HeuristicReranker, HybridRetriever,
    KeywordRetriever, PipelineConfig, TemporalRetriever,
};
use synaptic::memory::storage::{create_storage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::StorageBackend;

fn mem_err(e: impl std::fmt::Display) -> RunnerError {
    RunnerError::Memory(e.to_string())
}

/// One ablation configuration: which capabilities are enabled on top of the
/// dense+keyword fused baseline.
#[derive(Debug, Clone, Copy)]
pub struct AblationConfig {
    /// Config name as it appears in the emitted table.
    pub name: &'static str,
    /// Human-readable statement of exactly what this config includes.
    pub includes: &'static str,
    /// Composite relevance×recency×importance scoring (default weights).
    /// Off means weights `{1.0, 0.0, 0.0}` — identity on the fused order.
    pub composite: bool,
    /// `HeuristicReranker` over the top-K.
    pub reranker: bool,
    /// Graph + temporal signal retrievers (and knowledge-graph ingest).
    pub graph_temporal: bool,
    /// Hand the knowledge graph to the reranker (graph-proximity feature).
    pub graph_aware_reranker: bool,
}

/// The ablation ladder, baseline first. Each entry documents its contents.
pub fn configs() -> [AblationConfig; 5] {
    [
        AblationConfig {
            name: "baseline",
            includes: "dense-vector + keyword signals, RRF fusion only \
                       (composite weights 1.0/0.0/0.0 = pure fused relevance); \
                       no reranker; no graph/temporal signals",
            composite: false,
            reranker: false,
            graph_temporal: false,
            graph_aware_reranker: false,
        },
        AblationConfig {
            name: "+composite",
            includes: "baseline + composite relevance*recency*importance \
                       scoring (default weights 0.6/0.2/0.2)",
            composite: true,
            reranker: false,
            graph_temporal: false,
            graph_aware_reranker: false,
        },
        AblationConfig {
            name: "+reranker",
            includes: "+composite + HeuristicReranker over the top-K \
                       (term overlap, TF-IDF embedding agreement, recency; \
                       no graph handle)",
            composite: true,
            reranker: true,
            graph_temporal: false,
            graph_aware_reranker: false,
        },
        AblationConfig {
            name: "+graph_temporal",
            includes: "+reranker + graph-relationship and temporal-relevance \
                       signal retrievers (knowledge graph populated at \
                       ingest); reranker still graph-blind",
            composite: true,
            reranker: true,
            graph_temporal: true,
            graph_aware_reranker: false,
        },
        AblationConfig {
            name: "+all",
            includes: "full v2: +graph_temporal with the reranker's \
                       graph-proximity feature wired to the knowledge graph",
            composite: true,
            reranker: true,
            graph_temporal: true,
            graph_aware_reranker: true,
        },
    ]
}

/// Measured metrics for one config over the whole dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigReport {
    /// Config name (matches [`AblationConfig::name`]).
    pub name: String,
    /// What the config includes, verbatim from [`AblationConfig::includes`].
    pub includes: String,
    /// Total questions evaluated (identical across configs by construction).
    pub questions_evaluated: usize,
    /// Mean precision@k across all questions.
    pub mean_precision_at_k: f64,
    /// Mean recall@k across all questions.
    pub mean_recall_at_k: f64,
    /// Mean reciprocal rank across all questions.
    pub mrr: f64,
    /// Per-question detail rows (same schema as the Task 7.2 report).
    pub per_question: Vec<QuestionMetrics>,
}

/// Signed metric difference of a config vs the baseline (`config − baseline`).
/// Real measured numbers only — zero or negative deltas are reported as-is.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricDelta {
    /// Δ mean precision@k.
    pub precision_at_k: f64,
    /// Δ mean recall@k.
    pub recall_at_k: f64,
    /// Δ mean reciprocal rank.
    pub mrr: f64,
}

/// One non-baseline row of the ablation table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationRow {
    /// The config's measured metrics.
    pub config: ConfigReport,
    /// Its metrics minus the baseline's (may be zero or negative).
    pub delta_vs_baseline: MetricDelta,
}

/// The full ablation table: baseline metrics plus per-config deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationTable {
    /// The k used for precision@k / recall@k.
    pub k: usize,
    /// Baseline config metrics (delta reference).
    pub baseline: ConfigReport,
    /// Non-baseline configs, in ladder order, each with its delta.
    pub rows: Vec<AblationRow>,
}

/// Best-effort timestamp parsing for the dataset's verbatim timestamp
/// strings, so recency-sensitive capabilities (composite recency, temporal
/// signal, reranker recency) see real ages instead of "everything ingested
/// now". Supports RFC 3339, the LoCoMo session format (`1:56 pm on 8 May,
/// 2023`) and the LongMemEval haystack-date format (`2023/05/20 (Sat)
/// 02:21`). Unparseable strings leave `created_at` at ingest time.
pub fn parse_timestamp(raw: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(raw) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    let naive = chrono::NaiveDateTime::parse_from_str(raw, "%I:%M %P on %e %B, %Y")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(raw, "%I:%M %p on %e %B, %Y"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(raw, "%Y/%m/%d (%a) %H:%M"))
        .ok()?;
    Some(chrono::DateTime::from_naive_utc_and_offset(
        naive,
        chrono::Utc,
    ))
}

/// One constructed ablation pipeline over its own storage (and, when the
/// config wires graph/temporal signals, its own knowledge graph).
struct AblationPipeline {
    storage: Arc<dyn Storage + Send + Sync>,
    knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
    retriever: HybridRetriever,
}

impl AblationPipeline {
    /// Build the retrieval pipeline for `config` exactly as documented on
    /// [`configs`], mirroring `AgentMemory`'s construction (same
    /// `semantic_focus` base, same retriever types) minus the toggled-off
    /// capabilities.
    async fn build(config: &AblationConfig) -> Result<Self, RunnerError> {
        let storage = create_storage(&StorageBackend::Memory)
            .await
            .map_err(mem_err)?;

        let knowledge_graph = if config.graph_temporal {
            Some(Arc::new(tokio::sync::RwLock::new(
                MemoryKnowledgeGraph::new(synaptic::memory::knowledge_graph::GraphConfig::default()),
            )))
        } else {
            None
        };

        let mut pipeline_config = PipelineConfig::semantic_focus();
        // Re-run every question for real; no cross-question result caching.
        pipeline_config.enable_caching = false;
        if !config.composite {
            // Identity composite: score = normalized fused relevance, which
            // preserves the fused ordering exactly (see pipeline docs).
            pipeline_config = pipeline_config.with_composite_weights(CompositeWeights {
                relevance: 1.0,
                recency: 0.0,
                importance: 0.0,
            });
        }

        let provider = Arc::new(TfIdfProvider::default());
        let mut retriever = HybridRetriever::new(pipeline_config)
            .add_pipeline(Arc::new(DenseVectorRetriever::new(
                Arc::clone(&storage),
                provider,
            )))
            .add_pipeline(Arc::new(KeywordRetriever::new(Arc::clone(&storage))));

        if config.graph_temporal {
            retriever = retriever
                .add_pipeline(Arc::new(GraphRetriever::new(
                    Arc::clone(&storage),
                    knowledge_graph.clone(),
                )))
                .add_pipeline(Arc::new(TemporalRetriever::new(Arc::clone(&storage))));
        }

        if config.reranker {
            let reranker_graph = if config.graph_aware_reranker {
                knowledge_graph.clone()
            } else {
                None
            };
            retriever = retriever.with_reranker(Arc::new(HeuristicReranker::new(
                Some(Arc::new(TfIdfProvider::default())),
                reranker_graph,
            )));
        }

        Ok(Self {
            storage,
            knowledge_graph,
            retriever,
        })
    }

    /// Ingest every turn of `conversation` under the Task 7.2 key scheme
    /// ([`turn_memory_key`]), stamping `created_at` from the dataset
    /// timestamp when it parses, and mirroring each entry into the knowledge
    /// graph when this config wires graph signals.
    async fn ingest(&self, conversation: &EvalConversation) -> Result<(), RunnerError> {
        for session in &conversation.sessions {
            for (i, turn) in session.turns.iter().enumerate() {
                let key = turn_memory_key(session, i);
                let ts = turn.timestamp.as_ref().or(session.timestamp.as_ref());
                let value = match ts {
                    Some(ts) => format!("[{ts}] {}: {}", turn.speaker, turn.text),
                    None => format!("{}: {}", turn.speaker, turn.text),
                };
                let mut entry = MemoryEntry::new(key, value, MemoryType::LongTerm);
                if let Some(created) = ts.and_then(|raw| parse_timestamp(raw)) {
                    entry.metadata.created_at = created;
                    entry.metadata.last_accessed = created;
                }
                self.storage.store(&entry).await.map_err(mem_err)?;
                if let Some(ref kg) = self.knowledge_graph {
                    kg.write()
                        .await
                        .add_memory_node(&entry)
                        .await
                        .map_err(mem_err)?;
                }
            }
        }
        Ok(())
    }
}

/// Run one config over the whole dataset: per conversation, build a fresh
/// pipeline (its own haystack), ingest, run every question through the
/// config's retriever, and score against `evidence_ids` with the Task 7.2
/// metrics.
pub async fn run_config(
    config: &AblationConfig,
    conversations: &[EvalConversation],
    k: usize,
) -> Result<ConfigReport, RunnerError> {
    let mut per_question: Vec<QuestionMetrics> = Vec::new();
    for conversation in conversations {
        let pipeline = AblationPipeline::build(config).await?;
        pipeline.ingest(conversation).await?;
        for question in &conversation.questions {
            let fragments = pipeline
                .retriever
                .search(&question.text, k)
                .await
                .map_err(mem_err)?;
            let keys: Vec<String> = fragments.into_iter().map(|f| f.entry.key).collect();
            let retrieved = normalize_retrieved(&keys);
            let relevant: HashSet<String> = question.evidence_ids.iter().cloned().collect();
            per_question.push(QuestionMetrics {
                question_id: question.id.clone(),
                qtype: format!("{:?}", question.qtype),
                precision_at_k: precision_at_k(&retrieved, &relevant, k),
                recall_at_k: recall_at_k(&retrieved, &relevant, k),
                reciprocal_rank: mrr(&retrieved, &relevant),
            });
        }
    }
    let (mean_p, mean_r, mean_rr, _by_qtype) = aggregate(&per_question);
    Ok(ConfigReport {
        name: config.name.to_string(),
        includes: config.includes.to_string(),
        questions_evaluated: per_question.len(),
        mean_precision_at_k: mean_p,
        mean_recall_at_k: mean_r,
        mrr: mean_rr,
        per_question,
    })
}

/// Run the full ablation ladder and compute each config's delta vs baseline.
/// Every row comes from a real run of that config over `conversations`.
pub async fn run_ablation(
    conversations: &[EvalConversation],
    k: usize,
) -> Result<AblationTable, RunnerError> {
    let ladder = configs();
    let baseline = run_config(&ladder[0], conversations, k).await?;
    let mut rows = Vec::with_capacity(ladder.len() - 1);
    for config in &ladder[1..] {
        let report = run_config(config, conversations, k).await?;
        let delta_vs_baseline = MetricDelta {
            precision_at_k: report.mean_precision_at_k - baseline.mean_precision_at_k,
            recall_at_k: report.mean_recall_at_k - baseline.mean_recall_at_k,
            mrr: report.mrr - baseline.mrr,
        };
        rows.push(AblationRow {
            config: report,
            delta_vs_baseline,
        });
    }
    Ok(AblationTable { k, baseline, rows })
}

/// Render the table as a Markdown delta table (for `docs/evaluation.md`).
pub fn to_markdown(table: &AblationTable) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "| config | recall@{k} | Δrecall | precision@{k} | Δprecision | MRR | ΔMRR |\n",
        k = table.k
    ));
    out.push_str("|---|---|---|---|---|---|---|\n");
    out.push_str(&format!(
        "| {} | {:.4} | — | {:.4} | — | {:.4} | — |\n",
        table.baseline.name,
        table.baseline.mean_recall_at_k,
        table.baseline.mean_precision_at_k,
        table.baseline.mrr
    ));
    for row in &table.rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:+.4} | {:.4} | {:+.4} | {:.4} | {:+.4} |\n",
            row.config.name,
            row.config.mean_recall_at_k,
            row.delta_vs_baseline.recall_at_k,
            row.config.mean_precision_at_k,
            row.delta_vs_baseline.precision_at_k,
            row.config.mrr,
            row.delta_vs_baseline.mrr
        ));
    }
    out
}
