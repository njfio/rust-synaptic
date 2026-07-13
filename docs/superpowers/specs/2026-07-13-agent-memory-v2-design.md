# Agent Memory v2 — Design Spec

**Date:** 2026-07-13
**Branch:** `feature/agent-memory-v2` (off `main` @ db7314f)
**Status:** Approved design; implementation plan is `docs/superpowers/plans/agent-memory-v2.md`.

## Goal

Turn rust-synaptic into a cutting-edge AI-agent memory system and prove it with a real evaluation harness against LongMemEval-S and LoCoMo. Every capability is real, feature-gated, TDD-tested, and honesty-barred: disabled paths fail closed, no "in a real implementation" stubs, and **no performance or quality claim ships without a measured number behind it**.

## State-of-the-art grounding

| System | Idea adopted |
|---|---|
| Generative Agents (Park 2023) | retrieval score = recency·importance·relevance; reflection triggered when accumulated importance crosses a threshold; insights carry pointers to evidence. |
| Mem0 | write path as extract → decide (INSERT/UPDATE/DELETE/NOOP); low-latency, low-token. |
| Zep / Graphiti | bi-temporal KG: event time (`valid_from/valid_to`) + system time (`ingested_at/expired_at`); contradictions **invalidate, not delete**; point-in-time query. |
| HippoRAG | graph-structured multi-hop retrieval (activate the built-but-unwired Graph retriever; relation edges support multi-hop). |
| MemGPT/Letta | tiered memory + eviction (integrate decay with existing promotion tiers). |
| LongMemEval / LoCoMo | the acceptance benchmarks: extraction, multi-session + temporal reasoning, knowledge-update, abstention. |

## Core architectural decision: the `MemoryReasoner` trait

A single trait is the "intelligent core" for extraction, conflict decisions, and synthesis:

```rust
#[async_trait]
pub trait MemoryReasoner: Send + Sync {
    /// Extract atomic facts, entities, and typed relations from raw text.
    async fn extract(&self, text: &str, ctx: &ExtractionContext) -> Result<Extraction>;
    /// Decide how a candidate fact relates to the most-similar existing memories.
    async fn resolve(&self, candidate: &Fact, neighbors: &[ScoredMemory]) -> Result<ConflictResolution>;
    /// Synthesize a higher-level insight from a cluster of memories.
    async fn synthesize(&self, cluster: &[MemoryEntry]) -> Result<Option<Insight>>;
    fn name(&self) -> &str;
}
```

- **`HeuristicReasoner` (default, no new deps):** real rule-based entity extraction (extends the existing keyword/regex extractor in `management/summarization.rs:922` into a proper NER over capitalized spans, dates, numbers, quoted terms, and a small typed lexicon), embedding-based dedup/conflict via the existing `EmbeddingProvider` cosine similarity, and extractive synthesis (representative-sentence selection + template). Fully deterministic → the whole system and eval run offline in CI.
- **`LlmReasoner` (feature `llm-reasoning`, gate b):** prompts an LLM via the existing `reqwest`/Ollama/OpenAI provider surface for extraction/resolution/synthesis. When the feature is off OR no endpoint is reachable, callers fall back to `HeuristicReasoner` — never a fake result, and the fallback is logged, not silent.

`ConflictResolution` enum: `Insert | UpdateInPlace | Supersede { old_id } | Append | NoOp`, each carrying a reason string.

## The seven capabilities and their extension points

### 1. Intelligent write path
New pre-store stage inside `AgentMemory::store_with_report` (`src/lib.rs:362`): `extract → embed-dedup → resolve → apply`. Replaces the Jaccard/concat merge (`knowledge_graph/mod.rs:688`, `merge_content`) with the explicit `ConflictResolution` decision. Extracted entities/relations are written to the KG. A new `reasoning` field is added to `StoreDegradations` (best-effort, logged on failure). Feature: `intelligent-write` (default-on within `agent-memory-v2` group; the extraction quality scales with which reasoner is active).

### 2. Reflection & synthesis
A triggered pass in `consolidation/` that fires when accumulated importance since last reflection crosses `ReflectionConfig.importance_threshold`. Clusters recent memories (embedding k-means / connected-components over similarity), calls `MemoryReasoner::synthesize`, and writes `Insight` memories with `derived_from: Vec<MemoryId>` provenance recorded as KG `Derives` edges. Exposed as `AgentMemory::reflect()`.

### 3. Composite retrieval scoring + reranking
Two additions to the retrieval pipeline:
- **Composite scoring:** wire the present-but-inactive `decay_models.rs` (recency) and `importance_scoring.rs` (importance) into a post-fusion score `score = wr·relevance + wc·recency·decay + wi·importance` with configurable weights; activate the built-but-unwired `GraphRetriever` and `TemporalRetriever` (`src/lib.rs:204`) so all four signals feed fusion.
- **Reranker:** a new `Reranker` trait as a post-fusion stage (slots at `pipeline.rs:382`). `HeuristicReranker` default (feature-weighted cross-features: term-overlap × embedding-agreement × graph-proximity × recency, over the top-K candidate set — real, deterministic). Optional `CrossEncoderReranker` (feature `reranker-model`, gate b) loads a small cross-encoder via candle. Also fix the two known scoring stubs surfaced in the survey: BM25 IDF hardcoded to 1.0 (`strategies.rs:57`) → real corpus IDF; temporal frequency placeholder (`strategies.rs:161`) → real access-frequency.

### 4. Bi-temporal knowledge graph
Add to `Node`/`Edge` (`knowledge_graph/types.rs`): event-time `valid_from: DateTime`, `valid_to: Option<DateTime>`; system-time `ingested_at: DateTime`, `expired_at: Option<DateTime>`. On a `Supersede`, the old edge/fact is invalidated (`valid_to`/`expired_at` set) not removed. New query API `KnowledgeGraph::query_as_of(node, as_of: DateTime)` returns only facts whose validity interval contains `as_of` and that were not expired at `as_of`. Contradiction detection (`Contradicts` relation already exists at `types.rs:44`) drives invalidation by recency-of-truth.

### 5. Principled forgetting / decay
Importance-weighted decay wiring `temporal/decay_models.rs` (Ebbinghaus default) into a `ForgettingPolicy`: each memory's retained-strength decays over time modulated by importance and access; an eviction pass demotes/evicts below a configurable retention floor, integrated with `MemoryPromotionManager` (`src/lib.rs:112`) so decay drives the existing tier transitions (ShortTerm → LongTerm → evicted) rather than a parallel mechanism. Exposed as `AgentMemory::forget(policy)`.

### 6. Agent interface
- **Embedding providers first-class:** wire the candle `MLModelManager` (`integrations/ml_models.rs`) into the `EmbeddingProvider` trait so a local Candle BERT model is a provider peer to the API providers, selectable via `MultiProvider`. Feature `ml-models` (existing).
- **MCP server:** feature `mcp` (gate b — `rmcp` crate). Exposes tools `remember(content, metadata?)`, `recall(query, limit?, as_of?)`, `reflect()`, `forget(policy?)` over stdio JSON-RPC, backed by a shared `AgentMemory`. Disabled feature → the server binary simply isn't built; no fake tool responses.

### 7. Evaluation & validation harness (`tools/eval`)
Runs the **real** LongMemEval-S and LoCoMo datasets (fetched by `tools/eval/fetch_datasets.sh` into a gitignored `tools/eval/data/`; datasets not committed — licensing/size). Two tiers, both honest:

- **LLM-free, reproducible in CI and produced now:**
  - Retrieval precision/recall/MRR vs the datasets' ground-truth evidence spans.
  - Latency p50/p95/p99 for store and recall.
  - Memory-growth curves at 1k/10k/100k memories.
  - **Ablations:** baseline vs +composite-scoring vs +reranker vs +bi-temporal vs +reflection, reported on the retrieval metrics, showing each capability's measured delta.
- **LLM-gated (gate b — needs a configured endpoint):** end-to-end answer accuracy via a `Judge` trait (LLM generates an answer from recalled memories; LLM judge grades vs ground truth). Behind `llm-reasoning`. Produced when an endpoint is available; otherwise `docs/evaluation.md` records it as "not run — requires endpoint," never as an estimate.

`docs/evaluation.md` records only measured numbers, the exact command + machine + dataset version behind each, methodology, and the ablation table. Anything not run is marked not-run.

## Feature-flag layout

New features, all off by default so the default build stays lean:
- `agent-memory-v2` = umbrella enabling the pure-Rust capabilities (`intelligent-write`, composite scoring, bi-temporal KG, forgetting, heuristic reasoner/reranker) — no new deps.
- `llm-reasoning` = `["llm-integration"]` (LLM reasoner + eval judge).
- `reranker-model` = candle cross-encoder (gate b).
- `mcp` = `["dep:rmcp", ...]` (gate b).
Existing `ml-models` reused for the Candle embedding provider.

## Honesty invariants (inherited from prior remediation)

- Every feature-gated path that is disabled returns `MemoryError::feature_disabled` OR falls back to a real weaker implementation — never a fabricated result.
- No `.unwrap()`/`println!` in `src/`; `.expect()` states an invariant; `todo!()`/`unimplemented!()` forbidden.
- No "in a real implementation" / "simulated" markers introduced.
- `docs/evaluation.md` and README cite only measured numbers.

## Testing strategy

Each capability ships real-behavior tests, e.g.:
- write path: extracting "Alice moved to Berlin" then "Alice moved to Munich" yields a `Supersede`, the Berlin fact is invalidated, and `query_as_of(before)` returns Berlin while `query_as_of(after)` returns Munich.
- reflection: a cluster of related raw memories produces an insight whose `derived_from` lists exactly the source ids.
- reranker: a candidate that a single signal over-ranks is demoted below a cross-signal-agreeing candidate.
- forgetting: a low-importance stale memory is evicted while a high-importance one of the same age survives.
- eval harness: retrieval recall on a fixed dataset slice is deterministic and non-trivial (> a random-baseline threshold).

## Known tensions / decisions on record

- **Real-dataset eval:** headline QA-accuracy is LLM-gated (needs endpoint); retrieval/latency/growth/ablation are LLM-free and produced offline. Confirmed acceptable by maintainer.
- **Gate b dependencies** (`rmcp`, candle cross-encoder, LLM client) are presented with pinned versions at first use during execution; proceed with the obvious best option if no response.
