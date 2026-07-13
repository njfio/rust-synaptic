# Agent Memory v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Each phase's later tasks may get a just-in-time sub-plan (writing-plans) when its interfaces firm up.

**Goal:** Turn rust-synaptic into a cutting-edge agent memory system (intelligent write path, reflection, composite retrieval + reranking, bi-temporal KG, principled forgetting, MCP interface) and prove it with a real LongMemEval/LoCoMo evaluation harness.

**Architecture:** A `MemoryReasoner` trait (real local default + optional LLM) drives extraction/conflict-resolution/synthesis; the bi-temporal KG stores facts with event+system validity intervals; the retrieval pipeline gains composite scoring + a reranker stage; decay drives forgetting through the existing promotion tiers; an MCP server exposes remember/recall/reflect/forget; a `tools/eval` harness measures everything against real datasets.

**Tech Stack:** Rust 1.70+, tokio, async-trait, existing `EmbeddingProvider`/`RetrievalPipeline`/`KnowledgeGraph`/`MemoryPromotionManager`; optional candle (reranker/embeddings), reqwest (LLM), rmcp (MCP).

## Global Constraints

- Honesty bar: every disabled feature path returns `MemoryError::feature_disabled` OR falls back to a real weaker impl (logged, never silent) — never a fabricated result. No `todo!()`/`unimplemented!()`. No "in a real implementation"/"simulated" markers. No `.unwrap()`/`println!` in `src/`; `.expect()` states the invariant.
- TDD every task: failing test first (RED evidence), then implement (GREEN), then commit. Real-behavior tests, not mock-assertions.
- One conventional commit per task; ledger line appended to `.superpowers/sdd/progress.md` per completed task.
- Gates per task: `cargo fmt --all -- --check`; `cargo clippy --all-targets --features "<relevant>" -- -D warnings`; the task's tests; `cargo test --lib`. Note which combos only CI (`--all-features`) can verify. `--all-features` is not locally buildable (system libs: alsa/fontconfig/cmake) — CI is the arbiter.
- New features default-OFF; default build stays lean. `agent-memory-v2` umbrella = pure-Rust capabilities, no new deps. `llm-reasoning`, `reranker-model`, `mcp` are additive/gate-b.
- No performance or quality claim ships without a measured number in `docs/evaluation.md`.
- MSRV rust-version = "1.70.0".

## File Structure (new + modified)

- Create `src/memory/reasoning/mod.rs` — `MemoryReasoner` trait, `Extraction`/`Fact`/`Entity`/`Relation`/`ConflictResolution`/`Insight` types.
- Create `src/memory/reasoning/heuristic.rs` — `HeuristicReasoner` (default, deterministic).
- Create `src/memory/reasoning/llm.rs` — `LlmReasoner` (feature `llm-reasoning`).
- Modify `src/memory/knowledge_graph/types.rs` — bi-temporal fields on `Node`/`Edge`.
- Modify `src/memory/knowledge_graph/graph.rs` — invalidation + `query_as_of`.
- Modify `src/memory/knowledge_graph/mod.rs` — replace Jaccard/concat merge with `ConflictResolution` application.
- Modify `src/lib.rs` — write-path reasoning stage; `reflect()`/`forget()`; wire Graph/Temporal retrievers; composite scoring; `reasoning` field on `StoreDegradations`.
- Create `src/memory/retrieval/rerank.rs` — `Reranker` trait + `HeuristicReranker`; `CrossEncoderReranker` (feature `reranker-model`).
- Modify `src/memory/retrieval/strategies.rs` — real BM25 IDF; real temporal access-frequency.
- Modify `src/memory/retrieval/pipeline.rs` — composite score + reranker stage insertion.
- Create `src/memory/reflection.rs` — reflection trigger + clustering + insight write.
- Create `src/memory/forgetting.rs` — `ForgettingPolicy` + eviction wired to promotion.
- Modify `src/memory/store_result.rs` — add `reasoning` degradation field.
- Modify `src/integrations/ml_models.rs` — `EmbeddingProvider` impl for candle model.
- Create `src/bin/synaptic_mcp.rs` + `src/mcp/mod.rs` — MCP server (feature `mcp`).
- Create `tools/eval/` — dataset loaders, runner, metrics, ablation, judge; `fetch_datasets.sh`.
- Create `docs/evaluation.md` — measured results.
- Modify `Cargo.toml` — new features.

---

# PHASE 1 — Reasoning core + intelligent write path

### Task 1.1: `MemoryReasoner` trait and types

**Files:**
- Create: `src/memory/reasoning/mod.rs`
- Modify: `src/memory/mod.rs` (add `pub mod reasoning;`)
- Test: `tests/reasoning_types_tests.rs`

**Interfaces — Produces:**
```rust
pub struct Entity { pub name: String, pub kind: EntityKind, pub span: (usize, usize) }
pub enum EntityKind { Person, Place, Org, Date, Number, Quoted, Term }
pub struct Relation { pub subject: String, pub predicate: String, pub object: String }
pub struct Fact { pub text: String, pub entities: Vec<Entity>, pub relations: Vec<Relation> }
pub struct Extraction { pub facts: Vec<Fact> }
pub struct ExtractionContext { pub source_key: String, pub timestamp: chrono::DateTime<chrono::Utc> }
pub enum ConflictResolution { Insert, UpdateInPlace { reason: String }, Supersede { old_id: String, reason: String }, Append { reason: String }, NoOp { reason: String } }
pub struct Insight { pub text: String, pub derived_from: Vec<String>, pub confidence: f64 }
#[async_trait::async_trait]
pub trait MemoryReasoner: Send + Sync {
    async fn extract(&self, text: &str, ctx: &ExtractionContext) -> crate::error::Result<Extraction>;
    async fn resolve(&self, candidate: &Fact, neighbors: &[(String, f64)]) -> crate::error::Result<ConflictResolution>;
    async fn synthesize(&self, cluster: &[crate::memory::types::MemoryEntry]) -> crate::error::Result<Option<Insight>>;
    fn name(&self) -> &str;
}
```
(`neighbors` is `(memory_id, similarity)` to keep the trait free of retrieval types.)

- [ ] **Step 1: Write the failing test** — `tests/reasoning_types_tests.rs`:
```rust
use synaptic::memory::reasoning::{ConflictResolution, EntityKind, Fact, Entity, Relation};
#[test]
fn conflict_resolution_carries_reason() {
    let r = ConflictResolution::Supersede { old_id: "m1".into(), reason: "newer value".into() };
    match r { ConflictResolution::Supersede { old_id, reason } => { assert_eq!(old_id, "m1"); assert!(reason.contains("newer")); }, _ => panic!("wrong variant") }
}
#[test]
fn fact_holds_entities_and_relations() {
    let f = Fact { text: "Alice lives in Berlin".into(),
        entities: vec![Entity{name:"Alice".into(),kind:EntityKind::Person,span:(0,5)}, Entity{name:"Berlin".into(),kind:EntityKind::Place,span:(15,21)}],
        relations: vec![Relation{subject:"Alice".into(),predicate:"lives_in".into(),object:"Berlin".into()}] };
    assert_eq!(f.entities.len(), 2); assert_eq!(f.relations[0].predicate, "lives_in");
}
```
- [ ] **Step 2: Run — expect FAIL** (`cargo test --test reasoning_types_tests`) — module not found. (Note: the crate denies `clippy::panic` crate-wide via Cargo.toml `[lints]`, so the test file needs `#![allow(clippy::panic)]` like the other integration tests.)
- [ ] **Step 3: Implement** the types + trait above in `src/memory/reasoning/mod.rs` with derives `Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize` on the data types; `pub mod heuristic;` declared but content in 1.2. Add `pub mod reasoning;` to `src/memory/mod.rs` and re-export the trait + key types from there. `async-trait` is already a dependency (check `grep async-trait Cargo.toml`; if absent, add `async-trait = "0.1"`).
- [ ] **Step 4: Run — expect PASS**; plus `cargo test --lib`, fmt, clippy.
- [ ] **Step 5: Commit** — `feat(reasoning): MemoryReasoner trait and extraction/conflict types`.

### Task 1.2: `HeuristicReasoner` — real deterministic extraction

**Files:**
- Create: `src/memory/reasoning/heuristic.rs`
- Test: `tests/heuristic_reasoner_tests.rs`

**Interfaces — Consumes:** Task 1.1 types + `EmbeddingProvider` (`src/memory/embeddings/provider.rs:17`). **Produces:** `pub struct HeuristicReasoner { embedder: Arc<dyn EmbeddingProvider> }` impl `MemoryReasoner`.

Extraction rules (real, deterministic — extend `management/summarization.rs:922` logic): sentence-split; per sentence, entities = capitalized multi-word spans (Person/Org/Place disambiguated by a small lexicon + heuristics: trailing "Inc/Corp/Ltd"→Org, known-place lexicon→Place, else Person if preceded by a person-cue), ISO/`Month DD, YYYY`/numeric dates→Date, bare numbers→Number, quoted spans→Quoted, lexicon terms→Term. Relations = subject-verb-object where subject/object are extracted entities and predicate is the normalized verb lemma (small verb-normalization map: "lives in/moved to/relocated to" → residence predicates, etc.). `resolve`: if max neighbor similarity ≥ `supersede_threshold` (0.85) AND the candidate contradicts (shares subject+predicate, differs object — detected via extracted relations) → `Supersede`; ≥ `dedup_threshold` (0.95) and equal text → `NoOp`; ≥ update threshold (0.85) same subject/predicate/object with more detail → `UpdateInPlace`; else `Insert`. `synthesize`: pick the highest-importance representative sentence(s) across the cluster, template `"Across N related memories: <rep>"`, `derived_from` = cluster ids, `confidence` = mean pairwise similarity.

- [x] **Step 1: Failing test** — `tests/heuristic_reasoner_tests.rs` (use `TfIdfProvider` as embedder):
```rust
// extract entities+relation from "Alice moved to Berlin in 2021."
// assert an entity "Alice" (Person), "Berlin" (Place), a Date "2021", and a relation (Alice, moved_to/residence, Berlin).
// resolve: candidate "Alice moved to Munich" against neighbor list containing the Berlin memory at similarity 0.9 → Supersede.
// synthesize: 3 related memories → Some(Insight) whose derived_from == the 3 ids.
```
- [x] **Step 2: Run — expect FAIL.**
- [x] **Step 3: Implement** `HeuristicReasoner`. Deterministic; no randomness.
- [x] **Step 4: Run — expect PASS**; `cargo test --lib`, fmt, clippy.
- [x] **Step 5: Commit** — `feat(reasoning): deterministic HeuristicReasoner (NER + conflict + synthesis)`.
  - **Deviation note (Task 1.2, revised — stateless interface):** `MemoryReasoner::resolve` takes `neighbors: &[NeighborFact]` where `NeighborFact { id, similarity, text }` carries the neighbor's content. `HeuristicReasoner` is stateless: `extract` mutates no instance state, and `resolve` re-extracts the best neighbor's `text` (via a shared pure `extract_facts` helper) to compare real relations against the candidate. Contradiction (same subject+predicate, differing object) → `Supersede`; equal-text at sim ≥ 0.95 → `NoOp`; same relation with a longer candidate → `UpdateInPlace`. When there is no comparable extracted-relation evidence (or no neighbor), the result is `Insert` — never a blind `Supersede`/`UpdateInPlace` on similarity alone. Relation extraction includes a gap-guard rejecting SVO pairs with an intervening entity. `synthesize` returns `None` for clusters of size < 2. (Task 1.3 must pass neighbor text into `resolve` via `NeighborFact`.)

### Task 1.3: Intelligent write path wired into `store_with_report`

**Files:**
- Modify: `src/lib.rs` (`store_with_report` ~`:362`; `AgentMemory` fields ~`:100`), `src/memory/store_result.rs` (add field), `src/memory/knowledge_graph/mod.rs` (apply resolution)
- Test: `tests/intelligent_write_tests.rs`

**Interfaces — Consumes:** 1.1/1.2. **Produces:** `AgentMemory` holds `reasoner: Arc<dyn MemoryReasoner>` (default `HeuristicReasoner` over the active embedder); `StoreDegradations` gains `pub reasoning: Option<String>`. A new private `AgentMemory::apply_resolution(&mut self, entry, resolution)`.

Write path change: after storage+state write, run `reasoner.extract` on the value; for each fact, embed-dedup against existing (via `EmbeddingManager::get_memory_similarities`), `reasoner.resolve`, then apply: `Insert`→new KG nodes/edges from entities/relations; `Supersede`→invalidate the old KG fact (Phase 4 fields; until Phase 4 lands, mark superseded via a metadata flag and record it) and add new; `UpdateInPlace`→update node; `Append`→existing behavior; `NoOp`→skip. Failures set `degradations.reasoning` and log at warn (best-effort like the other subsystems). Replace the `merge_content` Jaccard/concat call site in `knowledge_graph/mod.rs:688` path so the concat separator is no longer the merge mechanism (keep `merge_content` only as an `Append` helper).

- [x] **Step 1: Failing test** — store "Alice lives in Berlin", then store "Alice lives in Munich"; assert `store_with_report` for the second returns a report indicating a supersede occurred (expose the resolution outcome in the report, e.g. `reasoning: None` on success + a queryable KG state), and that the KG has a Munich residence relation and the Berlin one flagged superseded. (Full bi-temporal assertion lands in Phase 4; here assert the resolution path ran and the KG reflects Munich as current.)
- [x] **Step 2: Run — expect FAIL.**
- [x] **Step 3: Implement.** Keep default behavior identical when extraction yields nothing.
- [x] **Step 4: Run — expect PASS**; `cargo test --lib`, fmt, clippy `--features "security test-utils"`.
- [x] **Step 5: Commit** — `feat(core): intelligent write path with extraction and conflict resolution`.

- **Deviation note (Task 1.3, real-API divergence):**
  - Neighbor discovery does NOT use `EmbeddingManager::get_memory_similarities`: `SimpleEmbedder` TF-IDF is degenerate on small corpora (the first document embeds to an all-zero vector because every IDF is `ln(1)=0`, and `embed_text` mutates the vocabulary per call, so scores drift between queries). Instead `AgentMemory::neighbor_facts` scores all state memories (short- + long-term, excluding the entry being stored) with a deterministic token-set cosine, ties broken by key, top-5 passed as `NeighborFact { id: memory_key, similarity, text: value }`. `NeighborFact.id` is the memory KEY (not a UUID) so `Supersede{old_id}` applies directly against the KG's `memory_to_node` mapping.
  - `SUPERSEDE_THRESHOLD` in `HeuristicReasoner` lowered 0.85 → 0.6: it is only a candidate gate (Supersede/UpdateInPlace still require extracted-relation evidence), and single-slot paraphrases like "Alice lives in Berlin"/"Alice lives in Munich" score 0.75 under the token-set cosine the write path actually supplies. No Task 1.2 test pinned the old boundary.
  - `UpdateInPlace`/`Append` carry no target id; the write path applies them to the best neighbor (same deterministic ordering the reasoner uses to pick its comparison target).
  - Supersede flags the old memory's KG node AND every edge whose `properties["source_memory"]` matches with `properties["superseded_at"]` (Phase 2.2 replaces this flag with `invalidate_edge`). KG state is queryable via `MemoryKnowledgeGraph::relations_for_entity` / `AgentMemory::entity_relations` returning `ExtractedRelation { subject, predicate, object, source_memory, superseded }`.
  - The reasoner field and write-path stage are gated `#[cfg(feature = "embeddings")]` (the `reasoning` module depends on `embeddings`, a default feature) and short-circuit when the knowledge graph is disabled. `merge_content` survives solely as the `Append` helper (`append_memory_content`).

---

# PHASE 2 — Bi-temporal knowledge graph
(Ordered before reflection/reranking because the write path's Supersede needs real invalidation.)

### Task 2.1: Bi-temporal fields on Node/Edge
**Files:** Modify `src/memory/knowledge_graph/types.rs` (`Node` `:114`, `Relationship` `:235`, `Edge` `:300`); Test `tests/bitemporal_types_tests.rs`.
**Produces:** `Node`/`Edge` gain `valid_from: DateTime<Utc>`, `valid_to: Option<DateTime<Utc>>`, `ingested_at: DateTime<Utc>`, `expired_at: Option<DateTime<Utc>>`; constructors default `valid_from`/`ingested_at`=now, `valid_to`/`expired_at`=None; helper `fn is_valid_at(&self, t: DateTime<Utc>) -> bool` (valid_from ≤ t < valid_to.unwrap_or(∞) AND (expired_at.is_none() || t < expired_at)).
- [x] Step 1: failing test — new edge `is_valid_at(now)` true; after `expire_at(t)` set, `is_valid_at(t+1s)` false, `is_valid_at(t-1s)` true.
- [x] Steps 2-4: implement fields + `is_valid_at` + `expire_at`/`invalidate`; serde back-compat via `#[serde(default = "default_bitemporal_start")]` (`DateTime::<Utc>::MIN_UTC` — legacy data reads as "always valid"). No external struct-literal call sites existed; all construction goes through `Node::new`/`Edge::new`. Used `Option::map_or` instead of `is_none_or` (crate MSRV 1.70).
- [x] Step 5: commit — `feat(kg): bi-temporal validity fields on nodes and edges`.

### Task 2.2: Invalidation + point-in-time query
**Files:** Modify `src/memory/knowledge_graph/graph.rs` (add `invalidate_edge`, `neighbors_as_of`), `src/memory/knowledge_graph/mod.rs` (Supersede → real invalidation); Test `tests/bitemporal_query_tests.rs`.
**Produces:** `KnowledgeGraph::invalidate_edge(&self, edge_id: &str, at: DateTime<Utc>) -> Result<bool>`; `KnowledgeGraph::neighbors_as_of(&self, node_id: &str, at: DateTime<Utc>) -> Vec<Edge>` returning only edges valid at `at`; `MemoryKnowledgeGraph::supersede_matching_relations(old_memory_key, fact)` invalidating only the edges whose (subject, predicate) match the superseding fact's relations (per-fact, not per-memory). The `superseded_at` metadata flag and `mark_memory_superseded` were removed; `ExtractedRelation.superseded` now reads bi-temporal validity (`!is_valid_at(now)`).
- [x] Step 1: failing test — add Berlin residence edge at t0; supersede with Munich at t1; `neighbors_as_of(alice, t0)` → Berlin only; `neighbors_as_of(alice, t1+)` → Munich only; Berlin edge still present but invalid (not deleted). Plus: superseding `lives_in` does not invalidate an unrelated `works_at` edge from the same source memory.
- [x] Steps 2-4: implement; wire the Phase-1 `Supersede` to call `invalidate_edge` (via `supersede_matching_relations`) instead of the metadata flag.
- [x] Step 5: commit — `feat(kg): edge invalidation and point-in-time neighbors_as_of query`.

---

# PHASE 3 — Composite retrieval scoring + reranking

### Task 3.1: Real BM25 IDF + temporal access-frequency
**Files:** Modify `src/memory/retrieval/strategies.rs` (`:57` IDF, `:161` frequency); Test extend `tests/retrieval_quality.rs`.
- [ ] Failing test: a rare query term ranks its document above a document matched only on a common term (real IDF); a frequently-accessed memory scores higher on the temporal signal than an equally-recent never-accessed one.
- [ ] Implement corpus IDF (`ln((N - df + 0.5)/(df + 0.5) + 1)`) computed from the candidate corpus; real access-frequency from the access tracker (`importance_scoring.rs` access counts) instead of `relevance_score` placeholder.
- [ ] Commit — `fix(retrieval): real BM25 IDF and access-frequency temporal signal`.

### Task 3.2: Composite scoring + activate Graph/Temporal retrievers
**Files:** Modify `src/lib.rs` (`:204` pipeline build), `src/memory/retrieval/pipeline.rs` (post-fusion composite score); Test `tests/composite_scoring_tests.rs`.
**Produces:** `CompositeWeights { relevance, recency, importance }` (default 0.6/0.2/0.2); post-fusion score `= wr·norm(relevance) + wc·recency_decay + wi·importance`. Register `GraphRetriever` + `TemporalRetriever` in the default `HybridRetriever`.
- [ ] Failing test: given two candidates with equal relevance, the more recent+important one ranks first; a graph-connected candidate surfaces that pure dense/keyword miss.
- [ ] Implement; recency via `decay_models.rs`, importance via `MemoryEntry.metadata.importance`.
- [ ] Commit — `feat(retrieval): composite relevance×recency×importance scoring; wire graph+temporal signals`.

### Task 3.3: Reranker trait + `HeuristicReranker`
**Files:** Create `src/memory/retrieval/rerank.rs`; Modify `pipeline.rs` (`:382` insertion); Test `tests/reranker_tests.rs`.
**Produces:** `#[async_trait] pub trait Reranker { async fn rerank(&self, query: &str, candidates: Vec<ScoredMemory>) -> Result<Vec<ScoredMemory>>; fn name(&self)->&str }`; `HeuristicReranker` scoring cross-features (term-overlap × embedding-agreement × graph-proximity × recency) over top-K. Pipeline gains optional `reranker: Option<Arc<dyn Reranker>>` applied after fusion.
- [ ] Failing test: a candidate over-ranked by one signal alone is demoted below a candidate that multiple signals agree on, after rerank.
- [ ] Implement; deterministic.
- [ ] Commit — `feat(retrieval): Reranker trait and deterministic HeuristicReranker stage`.

### Task 3.4: Optional cross-encoder reranker (gate b — candle)
**Files:** `src/memory/retrieval/rerank.rs` (`CrossEncoderReranker`, feature `reranker-model`); Cargo.toml feature; Test `tests/reranker_model_tests.rs` (feature-gated).
- Gate b: present the cross-encoder model choice + pinned candle version before adding. Disabled feature → type absent; no fake path.
- [ ] Failing test (feature on): reranker loads and produces a monotonic re-ordering on a tiny fixture. Feature off: `HeuristicReranker` remains the default and tests pass.
- [ ] Commit — `feat(retrieval): optional candle cross-encoder reranker (feature reranker-model)`.

---

# PHASE 4 — Reflection & synthesis

### Task 4.1: Reflection trigger + clustering + insight write
**Files:** Create `src/memory/reflection.rs`; Modify `src/lib.rs` (`reflect()`), `src/memory/mod.rs`; Test `tests/reflection_tests.rs`.
**Produces:** `ReflectionConfig { importance_threshold: f64, min_cluster: usize }`; `AgentMemory::reflect(&mut self) -> Result<Vec<Insight>>` — clusters memories accumulated since last reflection (connected-components over embedding similarity ≥ threshold), calls `reasoner.synthesize`, writes each `Insight` as a memory + KG `Derives` edges to sources.
- [ ] Failing test: store 3 memories about one topic + 1 unrelated; `reflect()` returns 1 insight whose `derived_from` == the 3 topic ids and the unrelated memory is not a source; a `Derives` edge exists from insight to each source.
- [ ] Implement; trigger fires only when accumulated importance ≥ threshold (test forces it).
- [ ] Commit — `feat(memory): triggered reflection producing provenance-linked insights`.

---

# PHASE 5 — Principled forgetting / decay

### Task 5.1: ForgettingPolicy + eviction wired to promotion
**Files:** Create `src/memory/forgetting.rs`; Modify `src/lib.rs` (`forget()`), integrate `MemoryPromotionManager` (`:112`); Test `tests/forgetting_tests.rs`.
**Produces:** `ForgettingPolicy { retention_floor: f64, decay: DecayModelType }`; `AgentMemory::forget(&mut self, policy) -> Result<ForgetReport>` computing each memory's retained strength = `decay(age) · importance · recency_of_access`; below `retention_floor` → demote a tier or evict; report lists evicted/demoted ids.
- [ ] Failing test: two same-age memories, one high-importance one low; after `forget`, low-importance one is evicted, high-importance one survives; a recently-accessed low-importance memory survives over a never-accessed one.
- [ ] Implement using `decay_models.rs`; eviction goes through promotion-tier transitions, not a parallel delete path.
- [ ] Commit — `feat(memory): importance-weighted forgetting and eviction via promotion tiers`.

---

# PHASE 6 — Agent interface

### Task 6.1: Candle embedding provider
**Files:** Modify `src/integrations/ml_models.rs` (impl `EmbeddingProvider` for the candle model); Test `tests/candle_embedding_provider_tests.rs` (feature `ml-models`).
- [ ] Failing test (feature on): the candle model, wrapped as `EmbeddingProvider`, returns a fixed-dim embedding and non-trivial cosine between related vs unrelated text; selectable via `MultiProvider`. Feature off: unchanged.
- [ ] Commit — `feat(embeddings): candle model as first-class EmbeddingProvider`.

### Task 6.2: MCP server (gate b — rmcp)
**Files:** Create `src/mcp/mod.rs`, `src/bin/synaptic_mcp.rs`; Cargo.toml feature `mcp`; Test `tests/mcp_tools_tests.rs` (feature `mcp`).
**Produces:** MCP tools `remember{content, metadata?}`, `recall{query, limit?, as_of?}`, `reflect{}`, `forget{policy?}` over stdio JSON-RPC, backed by a shared `AgentMemory`. Gate b: present `rmcp` pinned version before adding.
- [ ] Failing test (feature on): in-process, calling the `remember` tool then `recall` returns the stored content; `recall` with `as_of` respects bi-temporal validity. Feature off: binary/module not built; no fake responses.
- [ ] Commit — `feat(mcp): agent tool server exposing remember/recall/reflect/forget`.

---

# PHASE 7 — Evaluation & validation harness

### Task 7.1: Dataset loaders + fetch script
**Files:** Create `tools/eval/fetch_datasets.sh`, `tools/eval/src/dataset.rs` (or a `tools/eval` crate / `benches` module); `.gitignore` `tools/eval/data/`; Test `tools/eval` unit tests on a tiny committed fixture (a 2-conversation JSON sample structurally identical to the real format, committed as a fixture — NOT the real dataset).
**Produces:** loaders parsing LongMemEval-S and LoCoMo JSON into `EvalConversation { sessions, questions[] with evidence_ids, gold_answer, qtype }`.
- [ ] Failing test: loader parses the committed fixture into the expected typed structure; question types map to the enum.
- [ ] Commit — `feat(eval): LongMemEval/LoCoMo dataset loaders + fetch script`.

### Task 7.2: Runner + retrieval/latency/growth metrics (LLM-free)
**Files:** `tools/eval/src/{runner,metrics}.rs`; Test on fixture.
**Produces:** ingest each conversation into `AgentMemory`, run each question through `recall`, compute retrieval precision/recall/MRR vs `evidence_ids`; latency p50/p95/p99 (store + recall); memory-growth at 1k/10k/100k (synthetic fill using dataset sessions repeated/scaled, documented).
- [ ] Failing test: on the fixture, recall@k for a question whose evidence is ingested is 1.0; MRR computed correctly on a known ranking.
- [ ] Commit — `feat(eval): LLM-free retrieval/latency/growth harness`.

### Task 7.3: Ablation harness
**Files:** `tools/eval/src/ablation.rs`; Test on fixture.
**Produces:** run the metric suite under configs: baseline (pre-v2 pipeline), +composite, +reranker, +bi-temporal, +reflection; emit a delta table.
- [ ] Failing test: two configs on the fixture produce distinct, comparable metric records (structure), and enabling the reranker changes the ranking on a crafted case.
- [ ] Commit — `feat(eval): capability ablation harness`.

### Task 7.4: LLM-gated QA accuracy + Judge (gate b)
**Files:** `tools/eval/src/{qa,judge}.rs` (feature `llm-reasoning`); Test structural (feature-gated).
**Produces:** `Judge` trait; LLM answer-generation from recalled memories + LLM-judge grading vs gold; behind `llm-reasoning`. Feature off → QA accuracy reported as "not run — requires endpoint."
- [ ] Failing test (feature on, mockable judge for the unit test): the QA pipeline records an accuracy for a fixture question; feature off: the runner marks QA as not-run without fabricating a number.
- [ ] Commit — `feat(eval): LLM-gated end-to-end QA accuracy with pluggable Judge`.

### Task 7.5: Produce and commit `docs/evaluation.md`
**Files:** Create `docs/evaluation.md`; run the LLM-free harness on the real datasets (fetched locally) and record measured numbers + the ablation table; mark QA-accuracy not-run-or-run depending on endpoint availability. Update README to cite it (no standalone numbers).
- [ ] Run harness; write only measured numbers with command+machine+dataset-version provenance.
- [ ] Commit — `docs: evaluation results and capability ablation on LongMemEval/LoCoMo`.

---

# PHASE 8 — Adversarial whole-branch validation sweep

### Task 8.1: Final sweep
- Fake-path grep (`feature_disabled` on every disabled path; zero "in a real implementation"/"simulated"/`assert!(true)`); dangling-ref check; revert spot-checks on ≥3 new capability tests (write-path supersede, bi-temporal `neighbors_as_of`, reranker demotion, forgetting eviction — verify each fails when its impl is reverted); full gate battery (fmt, clippy, default build, feature builds that work locally, `cargo test --lib`, capability suites, eval harness on fixture); confirm `docs/evaluation.md` numbers reproduce.
- Summary: capabilities delivered, benchmark results vs baseline, gate status, deviations, sign-off items. No commit unless fixes needed.

---

## Self-Review

- **Spec coverage:** write path (1.3) / reflection (4.1) / composite+rerank (3.1-3.4) / bi-temporal (2.1-2.2) / forgetting (5.1) / MCP+candle (6.1-6.2) / eval (7.1-7.5) — all seven capabilities mapped; honesty invariants in Global Constraints; ablation in 7.3; real-dataset + LLM-gated split in 7.2/7.4/7.5.
- **Type consistency:** `MemoryReasoner`/`ConflictResolution`/`Extraction` (1.1) consumed by 1.2/1.3/4.1; `Reranker`/`ScoredMemory` (3.3) consumed by 3.4/3.2; bi-temporal `is_valid_at`/`neighbors_as_of` (2.1/2.2) consumed by 1.3-supersede and 7.2-`as_of`.
- **Ordering:** Phase 2 (bi-temporal) precedes reflection/forgetting because Supersede needs real invalidation; 1.3 uses a metadata flag only until 2.2 lands, then switches — noted in 1.3/2.2.
- **Gate-b items flagged:** 3.4 (cross-encoder), 6.2 (rmcp), 7.4 (LLM client) each carry an explicit present-choice-first note.
