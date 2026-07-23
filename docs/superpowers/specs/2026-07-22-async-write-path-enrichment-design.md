# Async Write-Path Enrichment — Design Spec

**Date:** 2026-07-22
**Status:** Approved (brainstorm), pending implementation plan

## Problem

rust-synaptic's write path performs **synchronous per-turn LLM enrichment**: every `store()`
awaits the reasoner's `extract → resolve → apply_resolution → mark_superseded → fact-store`
sequence (`reason_over_store`, `src/lib.rs`). With an accessible LLM this is ~5.7 s per
extraction call plus per-fact resolves, making ingest gated on LLM latency: LoCoMo conv-0 took
35–60 min, and an LLM-supersession A/B was measured at ~4 h and aborted (see
`docs/evaluation.md`). This is the identified architectural bottleneck that makes every
write-side LLM capability (distillation, bi-temporal supersession detection, reflection
synthesis) impractical to run or ship at scale.

This spec decouples the fast raw write from the slow LLM enrichment and makes enrichment
**concurrent**, so ingest latency is no longer gated on per-turn LLM latency.

## Goals

1. `store()` returns after the raw write without an inline LLM call (when enrichment is
   deferred/background); enrichment runs later, concurrently.
2. Bounded-concurrent enrichment (default 8) turns N serial LLM calls into ~N/8 wall-clock.
3. Full state consistency: enriched facts and supersessions are in `state`, storage, and KG —
   retrievable and checkpointable, identical semantics to the synchronous path.
4. Best-effort: enrichment errors are logged/counted, never abort the write or the batch.
5. Safe-by-default: enrichment mode is opt-in; default keeps today's exact synchronous behavior.
6. Measure the LoCoMo ingest speedup with a real LLM reasoner, proving the bottleneck is removed.

## Non-Goals

- Persisting the enrichment queue across process restarts (in-memory; crash leaves un-enriched
  entries as plain raw memories — consistent with best-effort).
- Changing the reasoner, extraction/resolution logic, or the distillation/bi-temporal
  semantics — this moves *when/how concurrently* enrichment runs, not *what* it computes.
- Distributed/multi-process enrichment.

## Global Constraints

- Best-effort subsystem: a per-entry enrichment failure sets a counter in the returned report
  and logs, never aborts the batch or the `store()`.
- Default-off equivalence: with enrichment mode `Inline` (default), `store()` behaves
  byte-for-byte like today (enrich synchronously); existing tests and callers unaffected.
- All new enrichment code behind `#[cfg(feature = "embeddings")]`, matching the existing
  write-path reasoning.
- PR-only merge flow; a new branch per PR iteration.
- No fabricated numbers; the measurement uses a real LLM reasoner (codex shim or ollama).

## Design

### Decisions (from brainstorm)

- **Decoupling:** deferred-batch, explicit `enrich_pending()`, plus an opt-in background worker.
- **State:** `state` becomes `Arc<RwLock<AgentState>>` for full consistency (chosen over the
  storage-only gap and the reconcile-pass alternatives).
- **Concurrency:** fully concurrent enrichment on the Arc/RwLock handles (max parallelism),
  bounded by a semaphore.

### Components

1. **Shared state.** `AgentMemory.state: memory::state::AgentState` →
   `Arc<tokio::sync::RwLock<memory::state::AgentState>>`. `AgentState` already derives `Clone`.
   All 23 `self.state.X` sites in `src/lib.rs` become `self.state.read().await.X` /
   `self.state.write().await.X`. `checkpoint_manager.create_checkpoint(&self.state)` (two sites,
   `src/lib.rs:774`, `:1471`) and `should_checkpoint(&self.state)` become
   `&*self.state.read().await`.

2. **Enrichment mode config.** New `MemoryConfig::enrichment_mode: EnrichmentMode` enum:
   - `Inline` **(default)** — today's behavior: `store_with_report` runs `reason_over_store`
     inline (synchronous). No queue, no worker. Byte-for-byte unchanged.
   - `Deferred` — `store()` enqueues; enrichment runs only on `enrich_pending()`.
   - `Background` — `store()` enqueues + notifies a spawned worker that drains continuously.
   New `MemoryConfig::enrichment_concurrency: usize` (default 8) — the semaphore permit count.

3. **Pending queue.** `AgentMemory.enrichment_queue: Arc<Mutex<VecDeque<PendingEnrichment>>>`,
   where `PendingEnrichment { key: String, value: String, timestamp: DateTime<Utc> }`.
   `store_with_report` pushes here (in `Deferred`/`Background`) instead of calling
   `reason_over_store` inline. `::fact{i}` keys (and any store made under the re-entrancy guard)
   are never enqueued.

4. **Enrichment engine.** The existing enrichment methods (`reason_over_store`, `neighbor_facts`,
   `apply_resolution`, `mark_superseded`, `mark_raw_source`) are lifted from `&mut self` onto a
   free-standing async function set that takes the Arc handles:
   `enrich_one(storage: &Arc<dyn Storage>, kg: &Arc<RwLock<KG>>, state: &Arc<RwLock<AgentState>>,
   reasoner: &Arc<dyn MemoryReasoner>, scoring: &Option<Arc<dyn EmbeddingProvider>>, config:
   &EnrichmentParams, entry: &MemoryEntry) -> Result<EnrichmentOutcome>`. Each acquires locks in
   the fixed order **KG → state** for the *shortest* span (mutate, drop); the LLM calls
   (`extract`, `resolve`) happen outside any lock.
   **Fact retrievability:** a fact-store writes `storage.store(fact)` +
   `state.write().await.add_memory(fact)` **and** `scoring.embed(fact.value)` — the last keeps the
   retrieval scoring provider's IDF corpus live so the pipeline/reranker can score the fact (this
   is exactly what `store_with_report` does at `src/lib.rs:681`; `retrieval_scoring_provider` and
   `retrieval_pipeline` are both `Arc`, so this is concurrency-safe). Facts are stored **directly**
   (not via a recursive `store_with_report`), which avoids re-entrancy entirely. The owned
   `embedding_manager` (a separate, already best-effort subsystem, `src/lib.rs:100`) is **not** fed
   for deferred-enriched facts — a documented, minor gap that does not affect pipeline/`storage`
   search retrievability (the real search path). Storage is `Arc<dyn Storage + Send + Sync>` with
   `&self` methods, so concurrent access is sound.

5. **`enrich_pending()`** — `pub async fn enrich_pending(&self) -> EnrichmentReport`. Drains the
   queue, spawns bounded-concurrent (`Semaphore`, `enrichment_concurrency`) `enrich_one` tasks
   over the drained entries, aggregates an `EnrichmentReport { entries, facts_stored,
   supersessions, raw_sources_tagged, errors }`. `&self` (mutation only via Arc/RwLock).

6. **Background worker (`Background` mode).** At construction, spawn a Tokio task holding clones
   of the Arc handles + config + an `Arc<Notify>`. `store()` calls `notify_one()` after
   enqueuing. The worker loops: wait on `Notify`, drain the queue, run `enrich_one` with bounded
   concurrency. `AgentMemory` holds the `JoinHandle` and an `Arc<AtomicBool> shutdown`. New
   `pub async fn shutdown(&self)` sets the flag, notifies, and awaits drain. New
   `pub async fn wait_for_enrichment(&self)` blocks until the queue is empty (poll the queue len
   behind the mutex; for `Background`, also await a drain signal) — for tests/eval determinism.
   `Drop` best-effort signals shutdown (cannot await in `Drop`; the worker observes the flag).

### Data flow

`store(key, value)` (mode = Deferred/Background):
1. Validate; build `MemoryEntry`; `storage.store` + `state.write().await.add_memory` (raw write,
   as today). Temporal/analytics/KG-node/embeddings subsystems run as today.
2. **Skip** the inline `reason_over_store`. Instead, if enrichment is live and the key is not a
   `::fact{i}` key, push `PendingEnrichment` to the queue. In `Background`, `notify_one()`.
3. Return `StoreDegradations` (enrichment degradations now reported later, via the
   `EnrichmentReport`, not the per-store degradations).

`enrich_pending()` / background drain:
1. Lock queue, `drain(..)` into a local `Vec`, unlock.
2. `stream::iter(entries).map(|e| enrich_one(...)).buffer_unordered(concurrency)`.
3. Each `enrich_one`: `extract` (no lock) → for each fact: `neighbor_facts` (storage read) →
   `resolve` (no lock) → apply under KG→state locks (add_extracted_fact / supersede /
   fact-store / mark_raw_source). Re-reads neighbors under the lock it will mutate, so a
   same-batch prior apply is visible (documented: "resolves in completion order").
4. Aggregate outcomes → `EnrichmentReport`.

`search()` is unchanged — it already reads from the pipeline/`storage`, and the superseded/
raw_source filters read `custom_fields` from the returned fragments, so enriched results are
visible once `enrich_pending` (or the worker) has run.

### Error handling

- `enrich_one` returns `Result`; a failure logs at `warn`, increments `EnrichmentReport.errors`,
  and the entry's raw memory remains stored/searchable. The batch continues.
- `Background` worker: same per-entry best-effort; the worker never panics out (errors counted,
  logged; an internal `Arc<Mutex<EnrichmentReport>>` accumulates for observability).
- Lock order KG→state is fixed everywhere to preclude deadlock; locks are never held across an
  `await` on the reasoner.

## Testing

Unit (deterministic; a stub `MemoryReasoner` returning known facts / errors / sleeping):
- Deferred: raw memory retrievable immediately after `store()`; `::fact` memories absent before
  `enrich_pending()`, present after; `state` (shared) contains them (full consistency).
- Concurrency speedup: stub sleeps `d` on each of M entries; `enrich_pending` wall-clock ≈
  `ceil(M/concurrency)·d` (proves parallelism); all facts land exactly once.
- Staleness: two queued entries, same subject, stub resolution supersedes → exactly one
  supersession, current fact wins (proves resolve-under-lock).
- Best-effort: stub errors on one entry → its raw memory intact, others enriched, `errors == 1`,
  batch completes.
- Re-entrancy: `::fact{i}` memories never enqueued/re-enriched.
- Default-off equivalence: `EnrichmentMode::Inline` → `store()` byte-for-byte as today (existing
  `store_extracted_facts_*`, distillation, superseded tests pass unchanged).
- Background: `wait_for_enrichment()` blocks until drained; `shutdown()` drains and stops cleanly.

Measurement (the payoff): eval-harness `--deferred-enrichment` path — ingest all LoCoMo conv-0
turns fast (queue), then one `enrich_pending()` at concurrency 8, using the real LLM reasoner
(ollama qwen or codex shim). Report `ingest wall-clock: synchronous Xs vs deferred Ys (N×)`.
Target: conv-0's ~35–60 min synchronous LLM ingest drops toward `total_LLM_time / 8`.

## Rollout

1. Land the `Arc<RwLock>` state refactor + `EnrichmentMode`/queue/`enrich_pending`/background +
   unit tests, default `Inline` (no behavior change), behind a PR with green substantive CI.
2. Add the eval `--deferred-enrichment` measurement; run it; add the speedup to
   `docs/evaluation.md`.
3. If the speedup is real (expected ~concurrency×), document `Deferred`/`Background` as the
   recommended modes for LLM-reasoner deployments; the write-side capability measurements
   (distillation, LLM supersession) become tractable to re-run.

## Open questions

None blocking. Whether `Drop` should best-effort-drain (it cannot await) vs require an explicit
`shutdown()` is an implementation detail; the plan will use the AtomicBool-observed-by-worker
approach and document that callers should `shutdown().await` for a guaranteed drain.
