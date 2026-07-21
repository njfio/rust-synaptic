# Write-Time Fact Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLM-distilled facts the primary retrievable units on the default write path when an LLM endpoint is configured, with raw turns retained but excluded from default search — the measured Mem0-parity condition.

**Architecture:** A `DistillationMode` config enum (`Auto`/`On`/`Off`) resolves at construction, via a pure `DistillationPolicy` resolver, into a live on/off decision from the reasoner kind + KG/embeddings availability. When live, `reason_over_store` stores each extracted fact as `{key}::fact{i}` and tags the raw turn `raw_source` **only after ≥1 fact lands**; `search` drops `raw_source`-tagged memories by default. The eval harness gains a config hook so the same full-LoCoMo run can A/B raw-turn vs facts-primary ingestion.

**Tech Stack:** Rust, `synaptic` crate (`src/lib.rs`, `src/memory/reasoning/`), `synaptic-eval` (`tools/eval/`), Tokio async, `cargo test`.

## Global Constraints

- Honesty bar: no fabricated numbers; report negative results; ship distillation default-on ONLY if the full-LoCoMo A/B shows Arm B (distillation) ≥ Arm A (baseline).
- Best-effort subsystem: distillation failure sets a `degradations.*` field, logs, and never aborts the write.
- No behavior change for no-LLM / no-KG deployments: a store there must remain byte-for-byte equivalent to today (raw turns untagged and searchable).
- Backwards compatibility: `store_extracted_facts: bool` stays as a deprecated alias mapped to the new enum.
- Failure-ordering invariant: the raw turn is tagged `raw_source` only AFTER ≥1 fact is successfully stored — never before.
- PR-only merge flow: repo ruleset requires PRs; a NEW branch per PR iteration.
- All distillation code is behind `#[cfg(feature = "embeddings")]`, matching the existing write-path reasoning (`src/lib.rs:245`, `:692`).
- Field/const naming: mirror the superseded pattern verbatim — `RAW_SOURCE_FIELD = "raw_source"` (const), `retrieval_excludes_raw_sources: bool` (config), exactly parallel to `SUPERSEDED_FIELD = "superseded_at"` / `retrieval_excludes_superseded`.

## File Structure

- `src/memory/reasoning/distillation.rs` (new): `DistillationMode` enum, `ResolvedReasonerKind` enum, `DistillationPolicy` resolver — pure logic, no store/IO. One responsibility: decide whether distillation is live and whether `On` is a misconfiguration.
- `src/memory/reasoning/mod.rs` (modify): `pub mod distillation;` + re-exports.
- `src/lib.rs` (modify): config fields (`distillation`, `retrieval_excludes_raw_sources`), constructor records `ResolvedReasonerKind` + computes live-flag, `reason_over_store` tags raw after facts land, `search` filter, `RAW_SOURCE_FIELD` const, `mark_raw_source` helper.
- `tools/eval/src/runner.rs` (modify): `ingest` already stores raw turns; no change needed beyond config (facts happen inside `store`).
- `tools/eval/src/qa.rs` (modify): thread a `MemoryConfig` factory into `run_agentic_qa_impl` so the A/B can choose the write path.
- `tools/eval/src/bin/run_eval.rs` (modify): `--distill` flag selecting the facts-primary config for the agentic-QA run.
- `docs/evaluation.md` (modify): A/B result section.
- `README.md` (modify): note distillation auto-activates with an LLM endpoint.

---

### Task 1: `DistillationPolicy` resolver (pure logic)

**Files:**
- Create: `src/memory/reasoning/distillation.rs`
- Modify: `src/memory/reasoning/mod.rs`
- Test: inline `#[cfg(test)]` module in `distillation.rs`

**Interfaces:**
- Produces:
  - `pub enum DistillationMode { Auto, On, Off }` (derives `Debug, Clone, Copy, PartialEq, Eq`; `Default` → `Auto`)
  - `pub enum ResolvedReasonerKind { Llm, Heuristic }` (derives `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub struct DistillationDecision { pub live: bool }` (derives `Debug, Clone, Copy, PartialEq, Eq`)
  - `pub fn resolve_distillation(mode: DistillationMode, reasoner: ResolvedReasonerKind, prerequisites_met: bool, llm_feature_enabled: bool) -> Result<DistillationDecision, String>` — returns `Err(msg)` for the `On` misconfiguration case (prereqs met + feature enabled + reasoner is Heuristic, i.e. an endpoint was expected but did not resolve).

- [ ] **Step 1: Write the failing tests**

Create `src/memory/reasoning/distillation.rs` with only the test module first:

```rust
//! Distillation policy: decides whether write-time fact distillation is live
//! from the configured mode, the resolved reasoner kind, and whether the write
//! path prerequisites (knowledge graph + embeddings) are available. Pure logic,
//! no store or IO, so it is unit-testable in isolation.

#[cfg(test)]
mod tests {
    use super::*;

    // Auto: live only with an LLM reasoner AND prerequisites.
    #[test]
    fn auto_llm_with_prereqs_is_live() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
        .unwrap();
        assert!(d.live);
    }

    #[test]
    fn auto_heuristic_is_off() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Heuristic,
            true,
            false,
        )
        .unwrap();
        assert!(!d.live);
    }

    #[test]
    fn auto_llm_without_prereqs_is_off() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Llm,
            false,
            true,
        )
        .unwrap();
        assert!(!d.live);
    }

    // Off: never live, regardless of everything else.
    #[test]
    fn off_is_never_live() {
        let d = resolve_distillation(
            DistillationMode::Off,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
        .unwrap();
        assert!(!d.live);
    }

    // On: hard error when prerequisites are absent.
    #[test]
    fn on_without_prereqs_errors() {
        let err = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            false,
            false,
        );
        assert!(err.is_err());
    }

    // On + feature enabled but reasoner resolved Heuristic = endpoint expected
    // but unresolved = misconfiguration = hard error.
    #[test]
    fn on_llm_feature_but_heuristic_reasoner_errors() {
        let err = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            true,
            true,
        );
        assert!(err.is_err());
    }

    // On without the llm feature (heuristic is the only possible reasoner):
    // proceed live with a warning rather than erroring.
    #[test]
    fn on_no_llm_feature_is_live_with_heuristic() {
        let d = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            true,
            false,
        )
        .unwrap();
        assert!(d.live);
    }

    #[test]
    fn on_llm_with_prereqs_is_live() {
        let d = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
        .unwrap();
        assert!(d.live);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (compile error)**

Run: `cargo test -p synaptic --features embeddings distillation:: 2>&1 | tail -20`
Expected: FAIL — `resolve_distillation`, `DistillationMode`, etc. not found.

- [ ] **Step 3: Write the implementation**

Prepend to `src/memory/reasoning/distillation.rs` (above the test module):

```rust
/// How write-time fact distillation is activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistillationMode {
    /// Live when an LLM reasoner is resolved and the write-path prerequisites
    /// (knowledge graph + embeddings) are available; off otherwise. Default.
    #[default]
    Auto,
    /// Require distillation. A hard error at construction when prerequisites are
    /// absent, or when the `llm-reasoning` feature is enabled but no endpoint
    /// resolved (a heuristic reasoner where an LLM was expected). In a build
    /// without the feature, proceeds live with a heuristic reasoner and a
    /// one-time warning.
    On,
    /// Never distill (escape hatch; today's raw-value-only behavior).
    Off,
}

/// Which reasoner the constructor resolved for the write path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedReasonerKind {
    Llm,
    Heuristic,
}

/// Outcome of resolving the distillation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistillationDecision {
    /// Whether the write path should distill facts and tag raw sources.
    pub live: bool,
}

/// Resolve whether distillation is live. `prerequisites_met` is
/// `enable_knowledge_graph && embeddings-available`; `llm_feature_enabled` is
/// whether the `llm-reasoning` cargo feature is compiled in. Returns `Err` only
/// for the `On` misconfiguration cases, where the caller should fail construction.
pub fn resolve_distillation(
    mode: DistillationMode,
    reasoner: ResolvedReasonerKind,
    prerequisites_met: bool,
    llm_feature_enabled: bool,
) -> Result<DistillationDecision, String> {
    match mode {
        DistillationMode::Off => Ok(DistillationDecision { live: false }),
        DistillationMode::Auto => Ok(DistillationDecision {
            live: prerequisites_met && reasoner == ResolvedReasonerKind::Llm,
        }),
        DistillationMode::On => {
            if !prerequisites_met {
                return Err(
                    "DistillationMode::On requires enable_knowledge_graph and the \
                     embeddings feature, but they are not available"
                        .to_string(),
                );
            }
            // Feature enabled but reasoner came back Heuristic => an endpoint was
            // expected (SYNAPTIC_LLM_URL/_MODEL) but did not resolve.
            if llm_feature_enabled && reasoner == ResolvedReasonerKind::Heuristic {
                return Err(
                    "DistillationMode::On with the llm-reasoning feature requires a \
                     configured LLM endpoint (SYNAPTIC_LLM_URL / SYNAPTIC_LLM_MODEL), \
                     but none resolved"
                        .to_string(),
                );
            }
            Ok(DistillationDecision { live: true })
        }
    }
}
```

Add to `src/memory/reasoning/mod.rs` (near the other `pub mod` lines):

```rust
pub mod distillation;
pub use distillation::{
    resolve_distillation, DistillationDecision, DistillationMode, ResolvedReasonerKind,
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic --features embeddings distillation:: 2>&1 | tail -20`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/memory/reasoning/distillation.rs src/memory/reasoning/mod.rs
git commit -m "feat(reasoning): DistillationPolicy resolver for write-time distillation"
```

---

### Task 2: Config fields + deprecated alias + constructor wiring

**Files:**
- Modify: `src/lib.rs` (MemoryConfig struct ~1705, Default impl ~1757, constructor reasoner block ~245, struct fields ~120)
- Test: inline `#[cfg(test)]` in `src/lib.rs` (the module that already holds `store_extracted_facts_*` tests, ~1800)

**Interfaces:**
- Consumes: `DistillationMode`, `ResolvedReasonerKind`, `resolve_distillation` from Task 1.
- Produces:
  - `MemoryConfig.distillation: memory::reasoning::DistillationMode`
  - `MemoryConfig.retrieval_excludes_raw_sources: bool` (default `true`)
  - `MemoryConfig.store_extracted_facts: bool` retained (deprecated; `true` forces `On`)
  - `AgentMemory` private fields: `distillation_live: bool`, `resolved_reasoner_kind: ResolvedReasonerKind`
  - `const RAW_SOURCE_FIELD: &'static str = "raw_source"` on `AgentMemory`

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/lib.rs` (near `store_extracted_facts_off_by_default_keeps_raw_only`, ~1800):

```rust
#[tokio::test]
async fn distillation_defaults_to_auto_off_without_llm() {
    // Default build has no LLM endpoint => Auto resolves to not-live.
    let mem = AgentMemory::new(MemoryConfig::default()).await.unwrap();
    assert!(!mem.distillation_live, "Auto must be off without an LLM reasoner");
}

#[tokio::test]
async fn store_extracted_facts_alias_forces_on() {
    // Deprecated alias: store_extracted_facts=true behaves as DistillationMode::On.
    // With KG+embeddings present and no llm feature endpoint, On is live (heuristic).
    let config = MemoryConfig {
        store_extracted_facts: true,
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mem = AgentMemory::new(config).await.unwrap();
    assert!(mem.distillation_live, "store_extracted_facts=true must make distillation live");
}

#[tokio::test]
async fn retrieval_excludes_raw_sources_defaults_true() {
    let config = MemoryConfig::default();
    assert!(config.retrieval_excludes_raw_sources);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic --features embeddings distillation_defaults_to_auto_off_without_llm 2>&1 | tail -20`
Expected: FAIL — `distillation_live` field and `distillation` config field do not exist.

- [ ] **Step 3: Implement config + constructor**

3a. Add struct fields to `MemoryConfig` (after `retrieval_excludes_superseded`, ~1711):

```rust
    /// Write-time fact distillation activation (see
    /// [`memory::reasoning::DistillationMode`]). Default `Auto`: live when an
    /// LLM reasoner is configured and the knowledge graph + embeddings are
    /// available, off otherwise (raw-value behavior unchanged).
    pub distillation: memory::reasoning::DistillationMode,
    /// Facts-primary retrieval: when `true`, `search` drops memories tagged as
    /// raw sources (the original turns distillation summarized), so retrieval
    /// surfaces distilled facts rather than raw turns. Default `true`. Only has
    /// an effect when distillation is live (untagged raw turns are unaffected).
    pub retrieval_excludes_raw_sources: bool,
```

Update the doc comment on `store_extracted_facts` (~1695) to mark it deprecated:

```rust
    /// DEPRECATED — use [`MemoryConfig::distillation`]. When `true`, forces
    /// `DistillationMode::On` (facts stored as `<key>::fact<N>` memories). Kept
    /// for backwards compatibility; slated for removal in a future major.
    /// Default `false`.
    pub store_extracted_facts: bool,
```

3b. Add to the `Default for MemoryConfig` impl (after `retrieval_excludes_superseded: false,`, ~1758):

```rust
            distillation: memory::reasoning::DistillationMode::Auto,
            retrieval_excludes_raw_sources: true,
```

3c. Add private fields to the `AgentMemory` struct (near `storing_facts: bool,`, ~120):

```rust
    #[cfg(feature = "embeddings")]
    distillation_live: bool,
    #[cfg(feature = "embeddings")]
    resolved_reasoner_kind: memory::reasoning::ResolvedReasonerKind,
```

3d. In the constructor, right after the `reasoner` block (~260), compute the kind + live flag. The reasoner block currently discards which branch it took, so set the kind inside it. Replace the reasoner block (~245-260) with:

```rust
        #[cfg(feature = "embeddings")]
        let (reasoner, resolved_reasoner_kind): (
            Arc<dyn memory::reasoning::MemoryReasoner>,
            memory::reasoning::ResolvedReasonerKind,
        ) = {
            let heuristic: Arc<dyn memory::reasoning::MemoryReasoner> =
                Arc::new(memory::reasoning::HeuristicReasoner::new(Arc::new(
                    memory::embeddings::TfIdfProvider::default(),
                )));
            #[cfg(feature = "llm-reasoning")]
            {
                match memory::reasoning::LlmReasoner::from_env() {
                    Some(llm) => (
                        Arc::new(llm) as Arc<dyn memory::reasoning::MemoryReasoner>,
                        memory::reasoning::ResolvedReasonerKind::Llm,
                    ),
                    None => (heuristic, memory::reasoning::ResolvedReasonerKind::Heuristic),
                }
            }
            #[cfg(not(feature = "llm-reasoning"))]
            {
                (heuristic, memory::reasoning::ResolvedReasonerKind::Heuristic)
            }
        };
```

3e. After the config's `store_extracted_facts`→mode reconciliation and the reasoner block, compute the decision. Add this after the reasoner block (still `#[cfg(feature = "embeddings")]`), where `config` is the incoming `MemoryConfig` (confirm the binding name in the constructor; it is the `config` parameter):

```rust
        #[cfg(feature = "embeddings")]
        let distillation_live = {
            // Deprecated alias: store_extracted_facts=true forces On.
            let mode = if config.store_extracted_facts {
                memory::reasoning::DistillationMode::On
            } else {
                config.distillation
            };
            let prerequisites_met = config.enable_knowledge_graph;
            let llm_feature_enabled = cfg!(feature = "llm-reasoning");
            match memory::reasoning::resolve_distillation(
                mode,
                resolved_reasoner_kind,
                prerequisites_met,
                llm_feature_enabled,
            ) {
                Ok(decision) => {
                    if decision.live
                        && resolved_reasoner_kind
                            == memory::reasoning::ResolvedReasonerKind::Heuristic
                    {
                        tracing::warn!(
                            "write-time distillation is live with the heuristic reasoner; \
                             the measured LoCoMo lift requires an LLM endpoint \
                             (SYNAPTIC_LLM_URL / SYNAPTIC_LLM_MODEL)"
                        );
                    }
                    decision.live
                }
                Err(msg) => {
                    return Err(crate::error::MemoryError::Configuration { message: msg });
                }
            }
        };
```

3f. Add both fields to the `Self { ... }` struct construction at the end of the constructor (near `storing_facts: false,`, ~458):

```rust
            #[cfg(feature = "embeddings")]
            distillation_live,
            #[cfg(feature = "embeddings")]
            resolved_reasoner_kind,
```

Note: if the constructor is not `#[cfg(feature="embeddings")]`-gated as a whole, the two `distillation_live` / `resolved_reasoner_kind` bindings must be produced under the same cfg. Verify `crate::error::SynapticError::ConfigurationError` exists (grep: `grep -n "ConfigurationError" src/error.rs`); if the variant name differs, use the actual config-error variant.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic --features embeddings distillation 2>&1 | tail -20`
Expected: PASS — the three new tests plus the Task-1 tests.

Also confirm no regression:
Run: `cargo test -p synaptic --features embeddings store_extracted_facts 2>&1 | tail -20`
Expected: PASS (existing tests unaffected — the alias still stores facts).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(config): DistillationMode config + constructor live-flag + deprecated alias"
```

---

### Task 3: Tag raw source after facts land + `mark_raw_source` helper

**Files:**
- Modify: `src/lib.rs` — `reason_over_store` (~713), new `mark_raw_source` helper (next to `mark_superseded` ~763), `RAW_SOURCE_FIELD` const (next to `SUPERSEDED_FIELD` ~913), and the gate at ~692.
- Test: inline `#[cfg(test)]` in `src/lib.rs`

**Interfaces:**
- Consumes: `distillation_live` (Task 2), `RAW_SOURCE_FIELD`.
- Produces: after `reason_over_store` stores ≥1 fact, the raw entry has `custom_fields["raw_source"]` set; on extraction/store failure it does not.

- [ ] **Step 1: Write the failing tests**

Add to the `src/lib.rs` test module:

```rust
// A stub reasoner that returns a fixed set of facts (or an error) so the write
// path is testable without an LLM. Placed in the test module.
#[cfg(feature = "embeddings")]
#[derive(Clone)]
struct StubReasoner {
    facts: Vec<String>,
    fail: bool,
}

#[cfg(feature = "embeddings")]
#[async_trait::async_trait]
impl memory::reasoning::MemoryReasoner for StubReasoner {
    async fn extract(
        &self,
        _text: &str,
        _ctx: &memory::reasoning::ExtractionContext,
    ) -> Result<memory::reasoning::Extraction> {
        if self.fail {
            return Err(crate::error::MemoryError::ProcessingError(
                "stub extract failure".to_string(),
            ));
        }
        Ok(memory::reasoning::Extraction {
            facts: self
                .facts
                .iter()
                .map(|t| memory::reasoning::Fact {
                    text: t.clone(),
                    entities: Vec::new(),
                    relations: Vec::new(),
                })
                .collect(),
        })
    }
    async fn resolve(
        &self,
        _candidate: &memory::reasoning::Fact,
        _neighbors: &[memory::reasoning::NeighborFact],
    ) -> Result<memory::reasoning::ConflictResolution> {
        Ok(memory::reasoning::ConflictResolution::Insert)
    }
    async fn synthesize(
        &self,
        _cluster: &[MemoryEntry],
    ) -> Result<Option<memory::reasoning::Insight>> {
        Ok(None)
    }
    fn name(&self) -> &str {
        "StubReasoner"
    }
}

#[tokio::test]
async fn distillation_tags_raw_source_after_facts_land() {
    let config = MemoryConfig {
        store_extracted_facts: true,
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut mem = AgentMemory::new(config).await.unwrap();
    mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
        facts: vec!["Alice lives in Berlin".to_string()],
        fail: false,
    }));
    mem.store("turn1", "Alice: I moved to Berlin last year").await.unwrap();

    // Raw turn is tagged; the fact memory exists and is not tagged.
    let raw = mem.get_entry_for_test("turn1").await.unwrap();
    assert!(raw.metadata.custom_fields.contains_key("raw_source"));
    let fact = mem.get_entry_for_test("turn1::fact0").await.unwrap();
    assert!(!fact.metadata.custom_fields.contains_key("raw_source"));
}

#[tokio::test]
async fn distillation_failure_leaves_raw_untagged() {
    let config = MemoryConfig {
        store_extracted_facts: true,
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut mem = AgentMemory::new(config).await.unwrap();
    mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
        facts: vec![],
        fail: true,
    }));
    // Write still succeeds (best-effort); raw turn stays searchable/untagged.
    mem.store("turn1", "Alice: I moved to Berlin").await.unwrap();
    let raw = mem.get_entry_for_test("turn1").await.unwrap();
    assert!(!raw.metadata.custom_fields.contains_key("raw_source"));
}
```

Add these two test-only helpers to `impl AgentMemory` (guard with `#[cfg(all(feature = "embeddings", test))]` or `#[cfg(feature = "test-utils")]` to match repo convention — check which the existing tests use; the `neighbor_examined_max` helper uses `test-utils`, but these are simpler as `#[cfg(test)]`):

```rust
    #[cfg(all(feature = "embeddings", test))]
    fn set_reasoner_for_test(&mut self, r: Arc<dyn memory::reasoning::MemoryReasoner>) {
        self.reasoner = r;
    }

    #[cfg(all(feature = "embeddings", test))]
    async fn get_entry_for_test(&self, key: &str) -> Option<MemoryEntry> {
        self.storage.retrieve(key).await.ok().flatten()
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic --features embeddings distillation_tags_raw_source_after_facts_land 2>&1 | tail -25`
Expected: FAIL — `set_reasoner_for_test`/`get_entry_for_test` missing, `raw_source` never set.

- [ ] **Step 3: Implement**

3a. Add the const next to `SUPERSEDED_FIELD` (~913):

```rust
    /// Custom-field key marking a memory as a raw source turn that write-time
    /// distillation has summarized into facts. Excluded from `search` when
    /// [`MemoryConfig::retrieval_excludes_raw_sources`] is set.
    #[cfg(feature = "embeddings")]
    const RAW_SOURCE_FIELD: &'static str = "raw_source";
```

3b. Add `mark_raw_source` next to `mark_superseded` (~777):

```rust
    /// Tag a raw source turn as summarized-by-distillation (facts-primary
    /// retrieval). Mirrors [`mark_superseded`]: records `raw_source` in the
    /// entry's custom fields and persists it, without re-running reasoning.
    #[cfg(feature = "embeddings")]
    async fn mark_raw_source(&mut self, key: &str) -> Result<()> {
        if let Some(mut entry) = self.storage.retrieve(key).await? {
            entry
                .metadata
                .custom_fields
                .insert(Self::RAW_SOURCE_FIELD.to_string(), Utc::now().to_rfc3339());
            self.storage.store(&entry).await?;
            if self.state.has_memory(key) {
                self.state.add_memory(entry);
            }
        }
        Ok(())
    }
```

3c. In `reason_over_store` (~713), track whether any fact was stored and tag the raw entry at the end. The existing per-fact loop stores facts behind `self._config.store_extracted_facts` (~744) — change that gate to the live flag and record success. Replace the fact-storage block (~744-752) with:

```rust
            // Persist the extracted fact as its own retrievable memory so
            // search surfaces distilled facts, not just raw turns.
            if self.distillation_live && !fact.text.trim().is_empty() {
                let fact_key = format!("{}::fact{}", entry.key, i);
                self.storing_facts = true;
                // Box::pin breaks the recursive-future size cycle
                // (store_with_report → reason_over_store → store_with_report).
                let result = Box::pin(self.store_with_report(&fact_key, &fact.text)).await;
                self.storing_facts = false;
                result?;
                stored_any_fact = true;
            }
```

Add `let mut stored_any_fact = false;` at the top of `reason_over_store` (after the empty-extraction early return, ~724), and after the `for` loop, before `Ok(())`:

```rust
        // Failure-ordering invariant: tag the raw turn as a summarized source
        // ONLY after ≥1 fact has been stored. If extraction or fact storage
        // failed above (propagated via `?`), we never reach here, so the raw
        // turn stays untagged and searchable — a distillation outage degrades
        // to raw-turn retrieval rather than hiding a turn with no replacement.
        if stored_any_fact {
            self.mark_raw_source(&entry.key).await?;
        }
```

3d. Update the gate at ~692 so reasoning runs when distillation is live (not only when a KG exists — but the KG is a prerequisite, so keep both):

```rust
        #[cfg(feature = "embeddings")]
        if self.knowledge_graph.is_some() && self.distillation_live && !self.storing_facts {
            if let Err(e) = self.reason_over_store(&entry).await {
                tracing::warn!(error = %e, "reasoning degraded during store");
                degradations.reasoning = Some(e.to_string());
            }
        }
```

Note: this changes the previous behavior where `reason_over_store` ran whenever a KG existed (for KG-only supersession side effects) even with facts off. If preserving KG supersession when distillation is off is required, keep the gate as `knowledge_graph.is_some() && !self.storing_facts` and instead rely on the `self.distillation_live` check already inside the fact-storage block (3c) — reasoning runs, but only tags/stores facts when live. **Choose the latter** to avoid regressing KG behavior: revert the gate to its original condition and leave the `distillation_live` checks only inside `reason_over_store`. The `stored_any_fact` tag block already guards tagging.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic --features embeddings distillation 2>&1 | tail -25`
Expected: PASS — both new tests plus prior distillation tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(write-path): tag raw sources after facts land (failure-ordering invariant)"
```

---

### Task 4: `search` filter excludes raw sources

**Files:**
- Modify: `src/lib.rs` — `search` (~1345)
- Test: inline `#[cfg(test)]` in `src/lib.rs`

**Interfaces:**
- Consumes: `RAW_SOURCE_FIELD`, `retrieval_excludes_raw_sources`, `distillation_live`.
- Produces: `search` returns fact memories and excludes `raw_source`-tagged turns by default; with `retrieval_excludes_raw_sources=false` the raw turn reappears.

- [ ] **Step 1: Write the failing tests**

```rust
#[tokio::test]
async fn search_excludes_raw_sources_by_default() {
    let config = MemoryConfig {
        store_extracted_facts: true,
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut mem = AgentMemory::new(config).await.unwrap();
    mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
        facts: vec!["Alice lives in Berlin".to_string()],
        fail: false,
    }));
    mem.store("turn1", "Alice: I moved to Berlin last year").await.unwrap();

    let hits = mem.search("Berlin", 10).await.unwrap();
    let keys: Vec<&str> = hits.iter().map(|f| f.entry.key.as_str()).collect();
    assert!(keys.iter().any(|k| k.contains("::fact")), "fact must be retrievable");
    assert!(!keys.contains(&"turn1"), "raw source must be excluded by default");
}

#[tokio::test]
async fn search_includes_raw_sources_when_disabled() {
    let config = MemoryConfig {
        store_extracted_facts: true,
        enable_knowledge_graph: true,
        retrieval_excludes_raw_sources: false,
        ..Default::default()
    };
    let mut mem = AgentMemory::new(config).await.unwrap();
    mem.set_reasoner_for_test(std::sync::Arc::new(StubReasoner {
        facts: vec!["Alice lives in Berlin".to_string()],
        fail: false,
    }));
    mem.store("turn1", "Alice: I moved to Berlin last year").await.unwrap();

    let hits = mem.search("Berlin", 10).await.unwrap();
    let keys: Vec<&str> = hits.iter().map(|f| f.entry.key.as_str()).collect();
    assert!(keys.contains(&"turn1"), "raw source must reappear when filter disabled");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic --features embeddings search_excludes_raw_sources_by_default 2>&1 | tail -20`
Expected: FAIL — raw `turn1` is still in results (no filter yet).

- [ ] **Step 3: Implement the filter**

In `search` (~1345), right after the existing superseded `retain` block, add:

```rust
        // Facts-primary retrieval: when enabled, drop raw source turns that
        // distillation has summarized, so search surfaces distilled facts.
        #[cfg(feature = "embeddings")]
        if self._config.retrieval_excludes_raw_sources {
            results.retain(|f| {
                !f.entry
                    .metadata
                    .custom_fields
                    .contains_key(Self::RAW_SOURCE_FIELD)
            });
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic --features embeddings search_ 2>&1 | tail -20`
Expected: PASS — both new tests.

Full regression on the crate:
Run: `cargo test -p synaptic --features embeddings 2>&1 | tail -8`
Expected: PASS — all lib tests (no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(search): exclude raw sources by default (facts-primary retrieval)"
```

---

### Task 5: Eval harness A/B write-path switch

**Files:**
- Modify: `tools/eval/src/qa.rs` — `run_agentic_qa` / `run_agentic_qa_impl` (~1827, ~1872) to accept a `MemoryConfig` factory; the internal `AgentMemory::new(MemoryConfig::default())` call (~1888) uses it.
- Modify: `tools/eval/src/bin/run_eval.rs` — `run_agentic_qa_only` (~767) reads a `--distill` flag and builds the facts-primary config.
- Test: inline `#[cfg(test)]` in `run_eval.rs` for the flag parse + config builder.

**Interfaces:**
- Consumes: `synaptic::MemoryConfig`, `DistillationMode`.
- Produces:
  - `qa::run_agentic_qa(conversations, judge, k, max_rounds, grounded, ground_verify, memory_config: MemoryConfig)` — new trailing param.
  - `run_eval` `--distill` flag → `MemoryConfig { distillation: On, enable_knowledge_graph: true, retrieval_excludes_raw_sources: true, ..default }`; absent → `MemoryConfig::default()`.

- [ ] **Step 1: Write the failing test**

Add a `#[cfg(test)]` module to `run_eval.rs`:

```rust
#[cfg(test)]
mod distill_flag_tests {
    use super::distillation_config;
    use synaptic::memory::reasoning::DistillationMode;

    #[test]
    fn distill_flag_builds_facts_primary_config() {
        let cfg = distillation_config(true);
        assert_eq!(cfg.distillation, DistillationMode::On);
        assert!(cfg.enable_knowledge_graph);
        assert!(cfg.retrieval_excludes_raw_sources);
    }

    #[test]
    fn no_distill_flag_is_default() {
        let cfg = distillation_config(false);
        assert_eq!(cfg.distillation, DistillationMode::Auto);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p synaptic-eval --bin run_eval distill_flag 2>&1 | tail -20`
Expected: FAIL — `distillation_config` not found.

- [ ] **Step 3: Implement**

3a. Add the config builder to `run_eval.rs` (near the other helpers):

```rust
/// Build the memory config for the agentic-QA run. `distill=true` selects the
/// facts-primary write path (DistillationMode::On, KG on, raw sources excluded);
/// `false` returns the default config (today's raw-turn behavior).
fn distillation_config(distill: bool) -> synaptic::MemoryConfig {
    if distill {
        synaptic::MemoryConfig {
            distillation: synaptic::memory::reasoning::DistillationMode::On,
            enable_knowledge_graph: true,
            retrieval_excludes_raw_sources: true,
            ..Default::default()
        }
    } else {
        synaptic::MemoryConfig::default()
    }
}
```

3b. In `run_agentic_qa_only` (~767), parse the flag and pass the config. The signature currently takes `conversations` + writer; the flag comes from the top-level `args`. Since `run_agentic_qa_only` does not see `args`, thread a `bool` param: change its signature to `run_agentic_qa_only(conversations, distill: bool, w)` and its caller in `main` (~256) to pass `args.iter().any(|a| a == "--distill")`. Then:

```rust
    let report = qa::run_agentic_qa(
        conversations,
        &judge,
        K,
        max_rounds,
        grounded,
        ground_verify,
        distillation_config(distill),
    )
    .await
    .map_err(|e| e.to_string())?;
```

3c. In `qa.rs`, add the trailing `memory_config: MemoryConfig` param to `run_agentic_qa`, `run_agentic_qa_over_facts`, and `run_agentic_qa_impl`, and use it in place of `MemoryConfig::default()` at ~1888:

```rust
        let mut memory = AgentMemory::new(memory_config.clone())
            .await
            .map_err(mem_err)?;
```

(`MemoryConfig` derives `Clone`; confirm with `grep -n "struct MemoryConfig" -A2 src/lib.rs`. It does.)

3d. Update the other callers of `run_agentic_qa` / `_over_facts` (grep: `rg "run_agentic_qa(_over_facts)?\(" tools/eval`) to pass `MemoryConfig::default()` where they don't opt into distillation, so the crate still compiles.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p synaptic-eval --bin run_eval distill_flag 2>&1 | tail -20`
Expected: PASS — 2 tests.

Build the whole eval crate to catch caller drift:
Run: `cargo build -p synaptic-eval --bin run_eval 2>&1 | tail -5`
Expected: `Finished`.

- [ ] **Step 5: Commit**

```bash
git add tools/eval/src/qa.rs tools/eval/src/bin/run_eval.rs
git commit -m "feat(eval): --distill A/B write-path switch for agentic QA"
```

---

### Task 6: Run the full-LoCoMo A/B and document the result

**Files:**
- Modify: `docs/evaluation.md` (new A/B section at end)
- Modify: `README.md` (distillation note)
- Modify: memory `v2-capability-scorecard.md` (only if the result changes the scorecard claim)

This task produces a MEASUREMENT and documentation, not library code. It uses the codex OAuth shim as the extraction endpoint (the same setup used for the judge — NOT the user's OpenAI API key) and the GPU cross-encoder build.

- [ ] **Step 1: Build the release eval binary with the LLM + GPU features**

```bash
source "$SCRATCHPAD/cudaenv.sh"
cargo build --release -p synaptic-eval --bin run_eval \
  --features 'synaptic/static-embeddings synaptic/reranker-model synaptic/cuda synaptic/llm-reasoning'
```
Expected: `Finished`. (The `llm-reasoning` feature is required so `LlmReasoner::from_env()` can resolve the codex shim endpoint; without it, `DistillationMode::On` will error per Task 1.)

- [ ] **Step 2: Start the codex→OpenAI shim** (per the running-qa-eval memory) and export the LLM env for both the judge and the write-path reasoner:

```bash
export SYNAPTIC_LLM_URL="http://127.0.0.1:<shim-port>/v1"
export SYNAPTIC_LLM_MODEL="<codex-model>"
export SYNAPTIC_EVAL_JUDGE=codex SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS=120 SYNAPTIC_EVAL_QA_CONCURRENCY=12
export SYNAPTIC_RETRIEVAL_EMBEDDER=static SYNAPTIC_STATIC_MODEL_DIR=models/potion-base-8M SYNAPTIC_RETRIEVAL_PRF=1
export SYNAPTIC_RERANKER=cross-encoder SYNAPTIC_RERANKER_MODEL_DIR=models/ms-marco-MiniLM-L6-v2
```

- [ ] **Step 3: Run Arm A (baseline, raw-turn ingestion)** with a dedicated checkpoint:

```bash
export SYNAPTIC_EVAL_QA_CHECKPOINT="$SCRATCHPAD/ab_baseline_ckpt.jsonl"
./target/release/run_eval tools/eval/data/locomo10.json --agentic-qa 2>&1 | tee "$SCRATCHPAD/ab_baseline.out"
```
Expected: a `QA: graded=1986 ... accuracy=...` line. Record the overall + per-category numbers.

- [ ] **Step 4: Run Arm B (distillation, facts-primary)** with its own checkpoint:

```bash
export SYNAPTIC_EVAL_QA_CHECKPOINT="$SCRATCHPAD/ab_distill_ckpt.jsonl"
./target/release/run_eval tools/eval/data/locomo10.json --agentic-qa --distill 2>&1 | tee "$SCRATCHPAD/ab_distill.out"
```
Expected: a `QA: graded=1986 ... accuracy=...` line. Note: Arm B is slower — each turn incurs an LLM extraction call at ingest.

- [ ] **Step 5: Write the A/B section in `docs/evaluation.md`**

Add a dated section with a table: overall + per-category accuracy for Arm A vs Arm B and the delta, plus a one-line honest verdict. If Arm B ≥ Arm A, state distillation is validated as a default; if not, state it is not shipped default-on and why. Use the exact measured numbers — do not invent them. Template:

```markdown
### Write-time distillation A/B on full LoCoMo (2026-07-20)

Same 1,986-q set, same judge (codex) + retrieval (PRF + GPU cross-encoder); only the
write path differs. Arm A ingests raw turns; Arm B distills facts (DistillationMode::On,
facts-primary, raw sources excluded from search), using the codex shim as the extractor.

| category | Arm A (raw) | Arm B (distill) | Δ |
|---|---|---|---|
| Overall | <A> | <B> | <Δ> |
| SingleHop | ... | ... | ... |
| Temporal | ... | ... | ... |
| OpenDomain | ... | ... | ... |
| MultiHop | ... | ... | ... |
| Abstention | ... | ... | ... |

**Verdict:** <one honest sentence — validated default if B ≥ A, else not shipped and why>.
```

- [ ] **Step 6: Update `README.md`** with a short note (only the copy, no numbers unless B≥A):

```markdown
### Write-time fact distillation

When an LLM endpoint is configured (`SYNAPTIC_LLM_URL` / `SYNAPTIC_LLM_MODEL`, with the
`llm-reasoning` feature) and the knowledge graph is enabled, `store()` distills each turn
into fact memories that become the primary retrievable units; the original turns are kept
but excluded from default search. Control it with `MemoryConfig::distillation`
(`Auto` — default, on when an LLM is configured / `On` / `Off`).
```

- [ ] **Step 7: Commit**

```bash
git add docs/evaluation.md README.md
git commit -m "docs(eval): full-LoCoMo write-time distillation A/B result"
```

---

## Self-Review

**1. Spec coverage:**
- Facts-primary retrieval (Approach A, metadata + filter) → Tasks 3, 4. ✓
- `DistillationMode` Auto/On/Off + resolver → Tasks 1, 2. ✓
- Auto default (on when LLM+KG) → Task 2 (constructor). ✓
- Deprecated `store_extracted_facts` alias → Task 2. ✓
- Failure-ordering invariant (tag after facts land) → Task 3 (test `distillation_failure_leaves_raw_untagged`). ✓
- Best-effort semantics (degradations.reasoning) → Task 3 (gate unchanged, error path logs). ✓
- Search filter `retrieval_excludes_raw_sources` → Task 4. ✓
- Eval A/B write-path switch → Task 5. ✓
- Full-LoCoMo before/after, honesty-gated → Task 6. ✓
- No-LLM/no-KG equivalence → Task 2 test `distillation_defaults_to_auto_off_without_llm`. ✓

**2. Placeholder scan:** Task 6 uses `<A>/<B>/<Δ>` — these are intentional measurement outputs to be filled from real runs, not code placeholders. All code steps contain complete code. `<shim-port>` / `<codex-model>` are environment specifics documented in the running-qa-eval memory.

**3. Type consistency:** `DistillationMode` (Task 1) used consistently in Tasks 2, 5. `ResolvedReasonerKind` (Task 1) used in Task 2. `RAW_SOURCE_FIELD` const (Task 3) used in Task 4. `distillation_live` field (Task 2) used in Task 3. `distillation_config` (Task 5) matches its test. Reasoner trait method signatures in the Task-3 stub (`extract`/`resolve`) must match the real `MemoryReasoner` trait — VERIFY against `src/memory/reasoning/mod.rs` before writing the stub (the `Extraction`/`Fact`/`ConflictResolution`/`NeighborFact` shapes are used from real code at `src/lib.rs:714-731`, so they are accurate).

**Verified APIs (confirmed against the code, 2026-07-20):**
- Crate error type is `crate::error::MemoryError` (`src/error.rs:6,16`). Config error variant: `MemoryError::Configuration { message: String }` (`:60`). Processing error: `MemoryError::ProcessingError(String)` (`:178`). The plan uses these exact forms.
- `MemoryReasoner` has FOUR methods: `extract`, `resolve`, `synthesize`, `name` (`src/memory/reasoning/mod.rs:146-159`). The `StubReasoner` implements all four (done above).
- `Fact` fields: `text: String`, `entities: Vec<Entity>`, `relations: Vec<Relation>` (`:64`). The stub sets `entities: Vec::new()`.
- `MemoryConfig.enable_knowledge_graph` **already defaults to `true`** (`src/lib.rs:1723`), so the `Auto` prerequisite is met by default; a default build with an LLM endpoint is live without extra config. The tests set it explicitly only for clarity.
- `MemoryConfig` derives `Clone` (`src/lib.rs:1632`), so `memory_config.clone()` in Task 5 is valid.
