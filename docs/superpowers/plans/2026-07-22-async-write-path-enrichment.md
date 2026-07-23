# Async Write-Path Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the fast raw write from slow per-turn LLM enrichment so ingest is no longer gated on LLM latency, with bounded-concurrent enrichment on shared handles.

**Architecture:** `AgentMemory.state` becomes `Arc<RwLock<AgentState>>`. A `MemoryConfig::enrichment_mode` (`Inline` default / `Deferred` / `Background`) controls whether `store()` runs enrichment inline (today's behavior) or enqueues it. The existing `reason_over_store` logic is lifted onto a free async function (`enrich_one`) taking `Arc` handles (storage, kg, state, reasoner, scoring provider); `enrich_pending()` drains the queue with bounded concurrency, and an opt-in background worker drains continuously.

**Tech Stack:** Rust, Tokio (`RwLock`, `Mutex`, `Semaphore`, `Notify`, `spawn`), `futures::stream::buffer_unordered`, the `synaptic` crate.

## Global Constraints

- Default `EnrichmentMode::Inline` behaves byte-for-byte like today (enrich synchronously in `store_with_report`); all existing tests pass unchanged.
- Best-effort: a per-entry enrichment failure logs at `warn`, increments an error counter, never aborts the batch or the `store()`; the raw memory is always stored.
- All new enrichment code behind `#[cfg(feature = "embeddings")]`, matching the existing write-path reasoning.
- Lock order is fixed **KG → state**; locks are never held across a reasoner `await`.
- Fact-stores feed storage + state + the `Arc` scoring provider (`retrieval_scoring_provider`) so facts stay retrievable; the owned `embedding_manager` is not fed for deferred facts (documented minor gap).
- `::fact{i}` keys are never enqueued for enrichment (re-entrancy guard).
- No fabricated numbers; the measurement uses a real LLM reasoner.
- PR-only merge flow; a new branch per PR iteration.

## Verified APIs (confirmed against code, 2026-07-22)
- `Storage`: `async fn store(&self, &MemoryEntry)`, `retrieve(&self, &str) -> Option<MemoryEntry>`, `search(&self, &str, usize)` — all `&self` (Arc-safe). Field: `storage: Arc<dyn memory::storage::Storage + Send + Sync>`.
- `EmbeddingProvider::embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding>`. Field: `retrieval_scoring_provider: Option<Arc<dyn memory::embeddings::provider::EmbeddingProvider>>`.
- `AgentState::add_memory(&mut self, MemoryEntry)`, `has_memory(&self, &str) -> bool`. Derives `Clone`.
- `knowledge_graph: Option<Arc<tokio::sync::RwLock<memory::knowledge_graph::MemoryKnowledgeGraph>>>`.
- `reasoner: Arc<dyn memory::reasoning::MemoryReasoner>`.
- Error: `crate::error::MemoryError::ProcessingError(String)`.
- Enrichment methods to lift: `reason_over_store` (`src/lib.rs:785`), `neighbor_facts` (`:901`), `apply_resolution` (`:1025`), `mark_superseded` (`:863`), `mark_raw_source` (`:843`). Consts `RAW_SOURCE_FIELD`, `SUPERSEDED_FIELD`, `NEIGHBOR_CANDIDATE_LIMIT` on `AgentMemory`.
- `store_with_report` inline enrichment gate: `src/lib.rs:764` (`if self.knowledge_graph.is_some() && !self.storing_facts { reason_over_store }`).

## File Structure

- `src/lib.rs` (modify): state field → `Arc<RwLock<AgentState>>` + all 23 call sites; `EnrichmentMode` reconciliation; queue field; `store_with_report` enqueue branch; `enrich_pending`; background worker fields + `shutdown`/`wait_for_enrichment`; re-export.
- `src/memory/enrichment.rs` (new): `EnrichmentParams`, `EnrichmentReport`, `PendingEnrichment`, the free-function engine `enrich_one` and its helpers (`neighbor_facts_shared`, `apply_resolution_shared`, `mark_field_shared`), lifted from the `&mut self` methods. One clear responsibility: run one entry's enrichment against shared handles.
- `src/memory/mod.rs` (modify): `pub mod enrichment;`.
- `src/lib.rs` `MemoryConfig` (modify): `enrichment_mode`, `enrichment_concurrency`.
- `tools/eval/src/bin/run_eval.rs` + `tools/eval/src/runner.rs` (modify): `--deferred-enrichment` ingest path + timing.

---

### Task 1: `state` → `Arc<RwLock<AgentState>>` (foundation, behavior-preserving)

**Files:**
- Modify: `src/lib.rs` (struct field ~line 3 of struct; 23 `self.state.*` sites; 2 `create_checkpoint(&self.state)` at `:774`,`:1471`; `should_checkpoint(&self.state)` at `:772`; constructor `state: AgentState::...` init)
- Test: existing `src/lib.rs` tests (must stay green — this task changes representation, not behavior)

**Interfaces:**
- Produces: `AgentMemory.state: std::sync::Arc<tokio::sync::RwLock<memory::state::AgentState>>`. Read: `self.state.read().await`. Write: `self.state.write().await`.

- [ ] **Step 1: Change the field type**

In the `AgentMemory` struct definition, change:
```rust
    state: memory::state::AgentState,
```
to:
```rust
    state: std::sync::Arc<tokio::sync::RwLock<memory::state::AgentState>>,
```

- [ ] **Step 2: Change the constructor initializer**

The state local is built at `src/lib.rs:220` (`let state = memory::state::AgentState::new(...)`) and consumed via field-init shorthand `state,` in the final `Self { ... }` (`src/lib.rs:511`). Replace that shorthand `state,` with:
```rust
            state: std::sync::Arc::new(tokio::sync::RwLock::new(state)),
```

- [ ] **Step 3: Update all read/write call sites**

Run `grep -n "self.state." src/lib.rs`. For each site, insert `.read().await` (for `has_memory`, `get_memory`, `peek_memory`, `session_id`, `created_at`, `long_term_memory_count`, `short_term_memory_count`, `total_memory_size`, `get_long_term_memories`) or `.write().await` (for `add_memory`, `clear`, `remove_memory`, `insert_preserving_access`). Example:
```rust
// before: self.state.has_memory(key)
self.state.read().await.has_memory(key)
// before: self.state.add_memory(entry.clone());
self.state.write().await.add_memory(entry.clone());
```
For the two checkpoint sites (`:772`, `:774`, `:1471`), take a read guard first:
```rust
        let state_guard = self.state.read().await;
        if self.checkpoint_manager.should_checkpoint(&state_guard) {
            self.checkpoint_manager.create_checkpoint(&state_guard).await?;
        }
        drop(state_guard);
```
and at `:1471`:
```rust
        let state_guard = self.state.read().await;
        self.checkpoint_manager.create_checkpoint(&state_guard).await
```
**Caution:** do not hold a `state` read guard across a `self.state.write().await` in the same scope (deadlock) — take short guards, drop before the next lock.

- [ ] **Step 4: Build and run the full lib test suite (behavior unchanged)**

Run: `cargo test -p synaptic --lib --features embeddings 2>&1 | tail -8`
Expected: PASS — same count as before this task (the representation changed, behavior did not). If a deadlock hangs a test, find where a read guard is held across a write in the same scope and shorten it.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "refactor(memory): state -> Arc<RwLock<AgentState>> (behavior-preserving)"
```

---

### Task 2: `EnrichmentMode` config + pending queue + `store()` enqueue

**Files:**
- Modify: `src/lib.rs` (`MemoryConfig` struct + `Default`; `AgentMemory` fields; constructor; `store_with_report` enrichment gate `:764`)
- Create: `src/memory/enrichment.rs` (types only, this task)
- Modify: `src/memory/mod.rs` (`pub mod enrichment;`)
- Test: inline `#[cfg(test)]` in `src/lib.rs`

**Interfaces:**
- Produces:
  - `memory::enrichment::EnrichmentMode { Inline, Deferred, Background }` (derives `Debug, Clone, Copy, PartialEq, Eq, Default`; `#[default] Inline`)
  - `memory::enrichment::PendingEnrichment { pub key: String, pub value: String }`
  - `MemoryConfig.enrichment_mode: EnrichmentMode`, `MemoryConfig.enrichment_concurrency: usize` (default 8)
  - `AgentMemory.enrichment_queue: Arc<tokio::sync::Mutex<std::collections::VecDeque<PendingEnrichment>>>`

- [ ] **Step 1: Create the enrichment module with types**

Create `src/memory/enrichment.rs`:
```rust
//! Async/deferred write-path enrichment: the raw store is fast; the slow LLM
//! enrichment (extract -> resolve -> supersede -> fact-store) runs later,
//! concurrently, on shared handles. See docs/superpowers/specs/2026-07-22-*.

/// How write-path LLM enrichment is scheduled relative to `store()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnrichmentMode {
    /// Enrich synchronously inside `store()` (today's behavior). Default.
    #[default]
    Inline,
    /// `store()` enqueues; enrichment runs on an explicit `enrich_pending()`.
    Deferred,
    /// `store()` enqueues + notifies a background worker that drains continuously.
    Background,
}

/// One entry awaiting enrichment (queued by `store()` in Deferred/Background).
#[derive(Debug, Clone)]
pub struct PendingEnrichment {
    pub key: String,
    pub value: String,
}
```
Add to `src/memory/mod.rs` near the other `pub mod` lines:
```rust
pub mod enrichment;
```

- [ ] **Step 2: Write the failing test**

Add to the `src/lib.rs` test module:
```rust
    #[tokio::test]
    async fn enrichment_defaults_to_inline() {
        let config = MemoryConfig::default();
        assert_eq!(
            config.enrichment_mode,
            memory::enrichment::EnrichmentMode::Inline
        );
        assert_eq!(config.enrichment_concurrency, 8);
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn deferred_mode_queues_without_inline_enrich() {
        let config = MemoryConfig {
            enrichment_mode: memory::enrichment::EnrichmentMode::Deferred,
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin").await.unwrap();
        // Deferred: no fact-memory yet (enrichment not run), raw is present.
        assert!(mem.get_entry_for_test("turn1").await.is_some());
        assert!(mem.get_entry_for_test("turn1::fact0").await.is_none());
        assert_eq!(mem.enrichment_queue_len().await, 1);
    }
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p synaptic --lib --features embeddings enrichment_defaults_to_inline deferred_mode_queues 2>&1 | tail -15`
Expected: FAIL — `enrichment_mode` field and `enrichment_queue_len` do not exist.

- [ ] **Step 4: Add config fields + defaults**

In `MemoryConfig` (after `retrieval_excludes_raw_sources`):
```rust
    /// How write-path LLM enrichment is scheduled (see
    /// [`memory::enrichment::EnrichmentMode`]). Default `Inline` (enrich
    /// synchronously inside `store`), byte-for-byte today's behavior.
    pub enrichment_mode: memory::enrichment::EnrichmentMode,
    /// Bounded concurrency for `enrich_pending`/background enrichment. Default 8.
    pub enrichment_concurrency: usize,
```
In `Default for MemoryConfig`:
```rust
            enrichment_mode: memory::enrichment::EnrichmentMode::Inline,
            enrichment_concurrency: 8,
```

- [ ] **Step 5: Add the queue field + test helper + constructor init**

Add to the `AgentMemory` struct:
```rust
    #[cfg(feature = "embeddings")]
    enrichment_queue: std::sync::Arc<
        tokio::sync::Mutex<std::collections::VecDeque<memory::enrichment::PendingEnrichment>>,
    >,
```
In the constructor `Self { ... }`:
```rust
            #[cfg(feature = "embeddings")]
            enrichment_queue: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::VecDeque::new(),
            )),
```
Add a test-only accessor to `impl AgentMemory`:
```rust
    #[cfg(all(feature = "embeddings", test))]
    async fn enrichment_queue_len(&self) -> usize {
        self.enrichment_queue.lock().await.len()
    }
```

- [ ] **Step 6: Branch the `store_with_report` enrichment gate**

Replace the inline-enrichment block at `src/lib.rs:764` (`if self.knowledge_graph.is_some() && !self.storing_facts { ... reason_over_store ... }`) with a mode branch:
```rust
        #[cfg(feature = "embeddings")]
        if self.knowledge_graph.is_some() && !self.storing_facts {
            match self._config.enrichment_mode {
                memory::enrichment::EnrichmentMode::Inline => {
                    if let Err(e) = self.reason_over_store(&entry).await {
                        tracing::warn!(error = %e, "reasoning degraded during store");
                        degradations.reasoning = Some(e.to_string());
                    }
                }
                memory::enrichment::EnrichmentMode::Deferred
                | memory::enrichment::EnrichmentMode::Background => {
                    // Never enqueue a fact-memory (re-entrancy guard already
                    // ensures fact stores set storing_facts, so we never reach
                    // here for them; the key check is belt-and-braces).
                    if !key.contains("::fact") {
                        self.enrichment_queue.lock().await.push_back(
                            memory::enrichment::PendingEnrichment {
                                key: key.to_string(),
                                value: value.to_string(),
                            },
                        );
                        // Background notification wired in Task 5.
                    }
                }
            }
        }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cargo test -p synaptic --lib --features embeddings enrichment_defaults_to_inline deferred_mode_queues 2>&1 | tail -10`
Expected: PASS.
Regression: `cargo test -p synaptic --lib --features embeddings 2>&1 | tail -6` — all pass (Inline default unchanged).

- [ ] **Step 8: Commit**

```bash
git add src/lib.rs src/memory/enrichment.rs src/memory/mod.rs
git commit -m "feat(enrichment): EnrichmentMode config + pending queue + deferred enqueue"
```

---

### Task 3: `enrich_one` engine (lifted onto Arc handles)

**Files:**
- Modify: `src/memory/enrichment.rs` (add the engine)
- Modify: `src/lib.rs` (expose consts needed by the engine, if private)
- Test: inline `#[cfg(test)]` in `src/memory/enrichment.rs`

**Interfaces:**
- Consumes: `PendingEnrichment` (Task 2), the reasoner trait, storage/kg/state/scoring Arc types.
- Produces:
  - `pub struct EnrichmentParams { pub distillation_live: bool, pub exclude_raw_default: bool }` — the per-entry knobs (mirrors `AgentMemory.distillation_live`).
  - `pub struct EnrichmentOutcome { pub facts_stored: usize, pub supersessions: usize, pub raw_tagged: bool }`
  - `pub async fn enrich_one(storage, kg, state, reasoner, scoring, params, key, value) -> Result<EnrichmentOutcome, crate::error::MemoryError>` with exact handle types:
    - `storage: &std::sync::Arc<dyn crate::memory::storage::Storage + Send + Sync>`
    - `kg: &std::sync::Arc<tokio::sync::RwLock<crate::memory::knowledge_graph::MemoryKnowledgeGraph>>`
    - `state: &std::sync::Arc<tokio::sync::RwLock<crate::memory::state::AgentState>>`
    - `reasoner: &std::sync::Arc<dyn crate::memory::reasoning::MemoryReasoner>`
    - `scoring: &Option<std::sync::Arc<dyn crate::memory::embeddings::provider::EmbeddingProvider>>`

- [ ] **Step 1: Write the failing test (stub reasoner, in-memory storage)**

Add a `#[cfg(test)]` module to `src/memory/enrichment.rs`. It builds an in-memory storage (`std::sync::Arc::new(crate::memory::MemoryStorage::new(...))` — confirm its constructor args with `grep -n "impl MemoryStorage\|pub fn new\|pub struct MemoryStorage" src/memory/storage/*.rs`), an empty KG (`MemoryKnowledgeGraph::new(...)`), an `Arc<RwLock<AgentState>>`, a stub reasoner returning one fact, and asserts `enrich_one` stores the raw memory's fact + tags raw. (Mirror the `StubReasoner` shape from `src/lib.rs` tests — `extract` returns one `Fact` with `entities: vec![]`, `resolve` returns `ConflictResolution::Insert`, `synthesize` `Ok(None)`, `name` a literal.)
```rust
#[cfg(test)]
mod tests {
    use super::*;
    // ... construct storage (in-memory), kg (new), state, stub reasoner ...
    #[tokio::test]
    async fn enrich_one_stores_fact_and_tags_raw() {
        // raw "turn1" pre-stored in storage+state
        // enrich_one(..., distillation_live=true, ...) over ("turn1","Alice: Berlin")
        // assert storage has "turn1::fact0"; "turn1" has raw_source custom field
        // assert outcome.facts_stored == 1 && outcome.raw_tagged
    }
}
```
(The implementer fills the exact construction using the crate's in-memory storage constructor — find it with `grep -n "MemoryStorage\|InMemory\|pub fn new" src/memory/storage/mod.rs`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic --lib --features embeddings enrich_one_stores 2>&1 | tail -15`
Expected: FAIL — `enrich_one` not found.

- [ ] **Step 3: Implement the engine by lifting `reason_over_store` and helpers**

Port `reason_over_store` (`src/lib.rs:785`), `neighbor_facts` (`:901`), `apply_resolution` (`:1025`), `mark_superseded` (`:863`), `mark_raw_source` (`:843`) into free functions in `enrichment.rs`, replacing `self.X` with the handle args:
- `self.storage` → `storage`; `self.knowledge_graph` (the `Some(kg)` guard) → `kg` (always present here — enrichment only runs when KG exists); `self.reasoner` → `reasoner`; `self.retrieval_scoring_provider` → `scoring`; `self.distillation_live` → `params.distillation_live`.
- `self.state.add_memory(e)` → `state.write().await.add_memory(e)`; `self.state.has_memory(k)` → `state.read().await.has_memory(k)`.
- Fact-store: instead of the recursive `Box::pin(self.store_with_report(fact_key, fact.text))`, store the fact **directly**:
```rust
async fn store_fact_shared(
    storage: &Arc<dyn Storage + Send + Sync>,
    state: &Arc<RwLock<AgentState>>,
    scoring: &Option<Arc<dyn EmbeddingProvider>>,
    fact_key: &str,
    fact_text: &str,
) -> Result<(), MemoryError> {
    let entry = MemoryEntry::new(fact_key.to_string(), fact_text.to_string(), MemoryType::LongTerm);
    storage.store(&entry).await?;
    state.write().await.add_memory(entry.clone());
    if let Some(p) = scoring {
        // Keep the retrieval scoring corpus live so the pipeline can rank the fact.
        let _ = p.embed(fact_text, None).await;
    }
    Ok(())
}
```
- `mark_superseded` / `mark_raw_source` become `mark_field_shared(storage, state, key, field_const, value)` that retrieves, inserts the custom field, re-stores, and updates state if present. Use the field string literals `"superseded_at"` and `"raw_source"` (matching `AgentMemory::SUPERSEDED_FIELD`/`RAW_SOURCE_FIELD`).
- `neighbor_facts` uses `storage.search` + the token-cosine ranking exactly as `src/lib.rs:901`; port verbatim with `storage` in place of `self.storage`. Keep `NEIGHBOR_CANDIDATE_LIMIT = 20` as a local const.
- Preserve the **failure-ordering invariant**: tag raw source only after ≥1 fact stored.
- Lock discipline: acquire `kg.write().await` only for the KG mutation in `apply_resolution_shared`; never hold it across `reasoner.resolve`.

Assemble `enrich_one` to mirror `reason_over_store`'s control flow (extract → per-fact neighbor+resolve+apply+supersede → store fact if `distillation_live` → tag raw if any stored), returning `EnrichmentOutcome`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic --lib --features embeddings enrich_one 2>&1 | tail -12`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memory/enrichment.rs src/lib.rs
git commit -m "feat(enrichment): enrich_one engine on shared Arc handles"
```

---

### Task 4: `enrich_pending()` bounded-concurrent + `EnrichmentReport`

**Files:**
- Modify: `src/memory/enrichment.rs` (`EnrichmentReport`)
- Modify: `src/lib.rs` (`enrich_pending` method)
- Test: inline `#[cfg(test)]` in `src/lib.rs`

**Interfaces:**
- Produces:
  - `memory::enrichment::EnrichmentReport { pub entries: usize, pub facts_stored: usize, pub supersessions: usize, pub raw_sources_tagged: usize, pub errors: usize }` (derives `Debug, Clone, Default, PartialEq, Eq`)
  - `pub async fn AgentMemory::enrich_pending(&self) -> memory::enrichment::EnrichmentReport`

- [ ] **Step 1: Add `EnrichmentReport` to `enrichment.rs`**

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnrichmentReport {
    pub entries: usize,
    pub facts_stored: usize,
    pub supersessions: usize,
    pub raw_sources_tagged: usize,
    pub errors: usize,
}
```

- [ ] **Step 2: Write the failing tests**

```rust
    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn enrich_pending_processes_queue() {
        let config = MemoryConfig {
            enrichment_mode: memory::enrichment::EnrichmentMode::Deferred,
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: I moved to Berlin").await.unwrap();
        assert!(mem.get_entry_for_test("turn1::fact0").await.is_none());
        let report = mem.enrich_pending().await;
        assert_eq!(report.entries, 1);
        assert_eq!(report.facts_stored, 1);
        assert_eq!(report.errors, 0);
        // fact now retrievable + queue drained
        assert!(mem.get_entry_for_test("turn1::fact0").await.is_some());
        assert_eq!(mem.enrichment_queue_len().await, 0);
    }

    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn enrich_pending_best_effort_on_error() {
        let config = MemoryConfig {
            enrichment_mode: memory::enrichment::EnrichmentMode::Deferred,
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec![],
            fail: true,
        }));
        mem.store("turn1", "Alice: Berlin").await.unwrap();
        let report = mem.enrich_pending().await;
        assert_eq!(report.entries, 1);
        assert_eq!(report.errors, 1);
        // raw memory intact despite enrichment failure
        assert!(mem.get_entry_for_test("turn1").await.is_some());
    }
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p synaptic --lib --features embeddings enrich_pending_processes enrich_pending_best_effort 2>&1 | tail -15`
Expected: FAIL — `enrich_pending` not found.

- [ ] **Step 4: Implement `enrich_pending`**

Add to `impl AgentMemory` (behind `#[cfg(feature = "embeddings")]`):
```rust
    /// Drain the pending-enrichment queue and enrich all entries with bounded
    /// concurrency (`MemoryConfig::enrichment_concurrency`). Best-effort: a
    /// per-entry failure is counted in the report, never aborts the batch.
    /// Only meaningful in Deferred/Background modes (Inline leaves the queue
    /// empty). Takes `&self` — mutation is via the Arc/RwLock handles.
    #[cfg(feature = "embeddings")]
    pub async fn enrich_pending(&self) -> memory::enrichment::EnrichmentReport {
        use futures::stream::{self, StreamExt};
        let Some(ref kg) = self.knowledge_graph else {
            return memory::enrichment::EnrichmentReport::default();
        };
        let pending: Vec<memory::enrichment::PendingEnrichment> = {
            let mut q = self.enrichment_queue.lock().await;
            q.drain(..).collect()
        };
        if pending.is_empty() {
            return memory::enrichment::EnrichmentReport::default();
        }
        let params = memory::enrichment::EnrichmentParams {
            distillation_live: self.distillation_live,
            exclude_raw_default: self._config.retrieval_excludes_raw_sources,
        };
        let concurrency = self._config.enrichment_concurrency.max(1);
        let storage = std::sync::Arc::clone(&self.storage);
        let kg = std::sync::Arc::clone(kg);
        let state = std::sync::Arc::clone(&self.state);
        let reasoner = std::sync::Arc::clone(&self.reasoner);
        let scoring = self.retrieval_scoring_provider.clone();
        let outcomes = stream::iter(pending)
            .map(|p| {
                let (storage, kg, state, reasoner, scoring, params) = (
                    storage.clone(), kg.clone(), state.clone(),
                    reasoner.clone(), scoring.clone(), params.clone(),
                );
                async move {
                    memory::enrichment::enrich_one(
                        &storage, &kg, &state, &reasoner, &scoring, &params, &p.key, &p.value,
                    )
                    .await
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;
        let mut report = memory::enrichment::EnrichmentReport {
            entries: outcomes.len(),
            ..Default::default()
        };
        for o in outcomes {
            match o {
                Ok(oc) => {
                    report.facts_stored += oc.facts_stored;
                    report.supersessions += oc.supersessions;
                    report.raw_sources_tagged += usize::from(oc.raw_tagged);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "enrichment degraded for entry");
                    report.errors += 1;
                }
            }
        }
        report
    }
```
Add `#[derive(Clone)]` to `EnrichmentParams` (Task 3) so it can be cloned per task.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p synaptic --lib --features embeddings enrich_pending 2>&1 | tail -12`
Expected: PASS. Regression: `cargo test -p synaptic --lib --features embeddings 2>&1 | tail -6` all pass.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/memory/enrichment.rs
git commit -m "feat(enrichment): enrich_pending bounded-concurrent + EnrichmentReport"
```

---

### Task 5: Background worker + `shutdown` / `wait_for_enrichment`

**Files:**
- Modify: `src/lib.rs` (`AgentMemory` fields for worker; constructor spawn; `store_with_report` notify; `shutdown`, `wait_for_enrichment`)
- Test: inline `#[cfg(test)]` in `src/lib.rs`

**Interfaces:**
- Produces:
  - `AgentMemory` fields: `enrichment_notify: Arc<tokio::sync::Notify>`, `enrichment_shutdown: Arc<std::sync::atomic::AtomicBool>`, `enrichment_worker: Option<tokio::task::JoinHandle<()>>`
  - `pub async fn AgentMemory::wait_for_enrichment(&self)` — returns when the queue is empty
  - `pub async fn AgentMemory::shutdown(&mut self)` — signal + await worker drain

- [ ] **Step 1: Write the failing test**

```rust
    #[cfg(feature = "embeddings")]
    #[tokio::test]
    async fn background_mode_enriches_without_explicit_call() {
        let config = MemoryConfig {
            enrichment_mode: memory::enrichment::EnrichmentMode::Background,
            store_extracted_facts: true,
            enable_knowledge_graph: true,
            ..Default::default()
        };
        let mut mem = AgentMemory::new(config).await.unwrap();
        mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
            facts: vec!["Alice lives in Berlin".to_string()],
            fail: false,
        }));
        mem.store("turn1", "Alice: Berlin").await.unwrap();
        // No enrich_pending() call; the background worker drains it.
        mem.wait_for_enrichment().await;
        assert!(mem.get_entry_for_test("turn1::fact0").await.is_some());
    }
```
NOTE: the background worker uses the `self.reasoner` captured at spawn time. `set_reasoner_for_test` must therefore either (a) be called before any store, and the worker must read the reasoner from a shared `Arc<ArcSwap>`/re-clone, OR (b) for the test, the worker reads `self.reasoner` via an `Arc` cloned at spawn — but the stub is set after construction. To keep the test valid, make the worker capture an `Arc<tokio::sync::RwLock<Arc<dyn MemoryReasoner>>>` OR have `set_reasoner_for_test` also update the worker's copy. SIMPLEST: store the reasoner behind `Arc<ArcSwapOption>`? To avoid a new dep, make the worker re-fetch the reasoner from a shared `Arc<tokio::sync::Mutex<Arc<dyn MemoryReasoner>>>` that both `set_reasoner_for_test` and the constructor write. Implement that shared cell (`reasoner_cell`) and have both `enrich_pending` and the worker read from it; keep `self.reasoner` as the initial value written into the cell. (This is the one place the plan adds indirection so tests can inject a stub into the background worker.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic --lib --features embeddings background_mode_enriches 2>&1 | tail -12`
Expected: FAIL — `wait_for_enrichment` not found.

- [ ] **Step 3: Implement the worker + shared reasoner cell**

- Add field `reasoner_cell: Arc<tokio::sync::Mutex<Arc<dyn memory::reasoning::MemoryReasoner>>>`, initialized in the constructor from the resolved `reasoner`. Change `set_reasoner_for_test` to also write the cell. Change `enrich_pending` and the worker to read the reasoner from `self.reasoner_cell.lock().await.clone()`.
- Add `enrichment_notify: Arc<Notify>`, `enrichment_shutdown: Arc<AtomicBool>`, `enrichment_worker: Option<JoinHandle<()>>`.
- In the constructor, when `enrichment_mode == Background`, spawn:
```rust
        #[cfg(feature = "embeddings")]
        let enrichment_worker = if config.enrichment_mode
            == memory::enrichment::EnrichmentMode::Background
            && knowledge_graph.is_some()
        {
            let queue = enrichment_queue.clone();
            let notify = enrichment_notify.clone();
            let shutdown = enrichment_shutdown.clone();
            let storage = storage.clone();
            let kg = knowledge_graph.clone().unwrap();
            let state = state.clone();
            let reasoner_cell = reasoner_cell.clone();
            let scoring = retrieval_scoring_provider.clone();
            let params = memory::enrichment::EnrichmentParams {
                distillation_live,
                exclude_raw_default: config.retrieval_excludes_raw_sources,
            };
            let concurrency = config.enrichment_concurrency.max(1);
            Some(tokio::spawn(async move {
                loop {
                    notify.notified().await;
                    if shutdown.load(std::sync::atomic::Ordering::SeqCst)
                        && queue.lock().await.is_empty()
                    {
                        break;
                    }
                    let reasoner = reasoner_cell.lock().await.clone();
                    // drain + buffer_unordered enrich_one (same as enrich_pending's
                    // core; factor into a shared free fn `drain_and_enrich(...)`).
                    memory::enrichment::drain_and_enrich(
                        &queue, &storage, &kg, &state, &reasoner, &scoring, &params, concurrency,
                    ).await;
                }
            }))
        } else {
            None
        };
```
- Factor the drain+enrich core of Task 4 into `pub async fn drain_and_enrich(queue, storage, kg, state, reasoner, scoring, params, concurrency) -> EnrichmentReport` in `enrichment.rs`; have both `enrich_pending` and the worker call it. (DRY — refactor Task 4's body into this fn.)
- In `store_with_report`'s enqueue branch, after `push_back`, add: `if Background { self.enrichment_notify.notify_one(); }`.
- Implement:
```rust
    #[cfg(feature = "embeddings")]
    pub async fn wait_for_enrichment(&self) {
        // Background: nudge the worker and poll until the queue drains.
        loop {
            if self.enrichment_queue.lock().await.is_empty() {
                return;
            }
            self.enrichment_notify.notify_one();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    #[cfg(feature = "embeddings")]
    pub async fn shutdown(&mut self) {
        self.enrichment_shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.enrichment_notify.notify_one();
        if let Some(h) = self.enrichment_worker.take() {
            let _ = h.await;
        }
    }
```
Non-Background modes: `enrichment_worker` is `None`; `wait_for_enrichment` returns immediately (queue only fills in Deferred/Background, and Deferred requires an explicit `enrich_pending`, so `wait_for_enrichment` returning on empty is correct — for Deferred the caller uses `enrich_pending`, not `wait`).

- [ ] **Step 4: Run tests + full regression**

Run: `cargo test -p synaptic --lib --features embeddings background_mode_enriches 2>&1 | tail -10`
Expected: PASS.
Run: `cargo test -p synaptic --lib --features embeddings 2>&1 | tail -6` — all pass.
Run: `cargo build -p synaptic 2>&1 | tail -3` — non-embeddings build compiles.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/memory/enrichment.rs
git commit -m "feat(enrichment): background worker + shutdown/wait_for_enrichment"
```

---

### Task 6: Eval `--deferred-enrichment` measurement + docs

**Files:**
- Modify: `tools/eval/src/runner.rs` (ingest that queues then times `enrich_pending`)
- Modify: `tools/eval/src/bin/run_eval.rs` (`--deferred-enrichment` flag + timing output)
- Modify: `docs/evaluation.md` (result)

This task produces a MEASUREMENT. It uses a real LLM reasoner (ollama qwen or the codex shim via `SYNAPTIC_LLM_URL`).

- [ ] **Step 1: Add a deferred-ingest helper to `runner.rs`**

```rust
/// Ingest all turns with enrichment deferred, then run one bounded-concurrent
/// enrich_pending pass. Returns (raw_ingest_secs, enrich_secs).
pub async fn ingest_deferred(
    conversation: &EvalConversation,
    memory: &mut AgentMemory,
) -> Result<(f64, f64), RunnerError> {
    use std::time::Instant;
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
```

- [ ] **Step 2: Add the `--deferred-enrichment` flag to `run_eval.rs`**

Parse `--deferred-enrichment` in `main`. When set, add a mode that, for conversation 0 (respecting `--max-conversations`), builds an `AgentMemory` with `enrichment_mode: Deferred`, `store_extracted_facts: true`, `enable_knowledge_graph: true`, runs `runner::ingest_deferred`, and — for comparison — also times a synchronous ingest (`enrichment_mode: Inline`, same config, `runner::ingest`). Print:
```
== Deferred-enrichment ingest speedup (conv 0) ==
synchronous (Inline) ingest: {sync}s
deferred: raw {raw}s + enrich {enrich}s = {total}s  (concurrency {c})
speedup: {sync/total:.1}x
```
Requires `SYNAPTIC_LLM_URL`/`SYNAPTIC_LLM_MODEL` (the library LLM reasoner); print a not-run notice if unset (never fabricate).

- [ ] **Step 3: Build and smoke the flag (tiny, no LLM needed to compile)**

Run: `cargo build --release -p synaptic-eval --bin run_eval --features 'synaptic/static-embeddings synaptic/llm-reasoning synaptic-eval/llm-reasoning' 2>&1 | tail -3`
Expected: `Finished`.

- [ ] **Step 4: Run the measurement with a real reasoner**

```bash
source "$SCRATCHPAD/cudaenv.sh"
export SYNAPTIC_LLM_URL='http://localhost:11434/v1' SYNAPTIC_LLM_MODEL='qwen2.5:7b-instruct'
export SYNAPTIC_RETRIEVAL_EMBEDDER=static SYNAPTIC_STATIC_MODEL_DIR=models/potion-base-8M
./target/release/run_eval tools/eval/data/locomo10.json --deferred-enrichment --max-conversations 1
```
Expected: a speedup line. Target: conv-0's synchronous LLM ingest (~tens of minutes) drops toward `total_LLM_time / concurrency`.

- [ ] **Step 5: Document the result in `docs/evaluation.md`**

Add a dated section with the exact synchronous-vs-deferred numbers and the speedup, one honest sentence on what it unblocks (write-side capability re-runs), and caveats (single conversation, local 7B reasoner). Use the measured numbers — do not invent them.

- [ ] **Step 6: Commit**

```bash
git add tools/eval/src/runner.rs tools/eval/src/bin/run_eval.rs docs/evaluation.md
git commit -m "feat(eval): --deferred-enrichment ingest-speedup measurement"
```

---

## Self-Review

**1. Spec coverage:**
- Arc<RwLock> state → Task 1. ✓
- EnrichmentMode config + queue + enqueue → Task 2. ✓
- enrich_one on Arc handles (storage/kg/state/reasoner/scoring) → Task 3. ✓
- enrich_pending bounded-concurrent + report → Task 4. ✓
- Background worker + shutdown + wait_for_enrichment → Task 5. ✓
- Fact retrievability via scoring provider → Task 3 (`store_fact_shared` feeds `scoring.embed`). ✓
- Failure-ordering invariant → Task 3 (tag raw only after ≥1 fact). ✓
- Best-effort → Task 4 test `enrich_pending_best_effort_on_error`. ✓
- Default-off equivalence → Task 2 (`Inline` branch = old `reason_over_store`) + full-suite regression each task. ✓
- Re-entrancy (`::fact` not enqueued) → Task 2 enqueue branch. ✓
- Measurement → Task 6. ✓

**2. Placeholder scan:** Task 3 Step 1 and Task 6 leave the exact in-memory-storage constructor / measured numbers to the implementer — these are genuine "find the constructor" / "measurement output" items, not code placeholders; the surrounding code is complete. Task 5 Step 1 spells out the reasoner-cell indirection needed for the test rather than hand-waving.

**3. Type consistency:** `enrich_one` handle types match the verified field types (Task 3 Interfaces). `EnrichmentReport`/`EnrichmentOutcome`/`EnrichmentParams` names consistent across Tasks 3–5. `drain_and_enrich` (Task 5) factors Task 4's body — Task 4's `enrich_pending` must be refactored to call it in Task 5 (noted). `enrichment_queue_len`/`get_entry_for_test`/`set_reasoner_for_test` are the existing test helpers.

**Pre-implementation note:** Task 1 (the Arc<RwLock> refactor) is the riskiest — deadlocks come from holding a read guard across a write in one scope. Implement and fully green-test Task 1 before any other task. Confirm the in-memory storage constructor name (`grep -n "impl.*Storage for\|pub fn new" src/memory/storage/mod.rs`) for Task 3's test.
