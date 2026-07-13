# Phase 5: Management De-stubbing & Ledger Debt Implementation Plan

> Execute via subagent-driven-development. Same gates/constraints as Phases 0-4.

**Goal:** Eliminate the "In a real implementation" scaffolding (61 sites) and the ledger-debt items via the **implement / delete / demote** rule, so the code either does what it claims, is gone, or is honestly labeled a heuristic.

## Global Constraints (inherit from master plan)
- No `.unwrap()`/`println!` in src/; `.expect()`s state invariants; unsafe forbidden.
- TDD per task; one conventional commit per task.
- Local gates: fmt; clippy `--all-targets --features "security test-utils" -- -D warnings`; the task's tests; `cargo test --lib`. CI = `--all-features` arbiter.
- **Decision rule for each stub:** (a) real data source exists → implement; (b) no consumer in public API/tests → delete; (c) useful heuristic oversold → keep + rename/doc as heuristic (stop calling it "intelligent"/"real").
- Distributed module stubs are OUT OF SCOPE (Phase 6 descopes the whole module).

---

### Task 5.1: Fix the RRF fusion bug (ledger debt)
**Files:** src/memory/retrieval/pipeline.rs (~line 409 combine_scores RRF branch), src/lib.rs (remove the WeightedAverage production override), tests/retrieval_quality.rs.
**Scope:** The RRF branch uses `1.0/(score.recip()+60.0)` (score, not rank) so combined scores (~0.016) fall below min_score (0.1) and filter everything. Fix to real rank-based RRF: `score += 1/(k + rank_i)` summed across retrievers by the item's RANK in each retriever's ordered list (k=60 standard). Restore ReciprocRankFusion as the pipeline default, remove the `WeightedAverage` override in lib.rs and the FIXME comment. The 3 pre-existing `retrieval_quality` failures (test_hybrid_search_with_results, test_multiple_signals_fusion, test_temporal_retriever_recency_bias) must turn GREEN. TDD: a fusion unit test asserting rank-1-in-both-retrievers outranks rank-1-in-one. Commit: "fix(retrieval): correct RRF to rank-based fusion, restore as default".

### Task 5.2: Tokenized storage search (ledger debt)
**Files:** src/memory/storage/memory.rs (search_entries), possibly src/memory/storage/mod.rs, src/lib.rs (remove CandidateWideningStorage if obsolete), tests.
**Scope:** MemoryStorage::search currently requires the whole query phrase verbatim (substring on full query). Change to tokenized matching: split query into terms, a candidate matches if it contains ANY term (OR semantics), rank by number of distinct terms matched then term frequency. This makes the CandidateWideningStorage pipeline wrapper (added in Task 3.1) obsolete — remove it and route the pipeline directly at storage if the widening is no longer needed. Keep dedup + a documented candidate cap. TDD: multi-word query finds a doc containing only some terms; verbatim-phrase requirement is gone. Commit: "feat(storage): tokenized OR-matching search; remove candidate-widening shim".

### Task 5.3: ANN stale-vector invalidation (ledger debt)
**Files:** src/memory/management/mod.rs (index_embedding / the HNSW index path), src/memory/indexing/ if needed, tests/indexed_retrieval_tests.rs.
**Scope:** Re-storing an existing key inserts a new vector (fresh metadata.id) without removing the stale one, so ANN counts drift/double-count. Fix: on re-index of an existing key, remove/supersede the prior vector for that key before inserting the new one (maintain a key→current-uuid map; hnsw_rs may not support deletion — if not, mark the stale uuid as tombstoned and skip tombstoned hits in count/query, documenting the approach). TDD: store key K with embedding A, re-store K with embedding B, assert related-count reflects only B (no double-count). Remove the "KNOWN LIMITATION" doc from Task 3.2. Commit: "fix(indexing): invalidate stale ANN vectors on key overwrite".

### Task 5.4: Sweep `let _ =` error-swallows + remove dead flag (ledger debt)
**Files:** src/lib.rs (the `let _ =` at ~512/552/564/717 — analytics/kg swallows outside store_with_report), src/analytics/mod.rs (dead `enable_visualization` flag).
**Scope:** For each `let _ =` swallow: either propagate via `?`, or log at warn like Task 2.1's store_with_report pattern (capture into an existing degradation channel where one exists; otherwise warn-log with context). Do NOT silently keep swallowing. Remove `AnalyticsConfig.enable_visualization` (gates nothing since Task 1.3 deleted the module) and any references. TDD/verification: existing lib tests stay green; add a focused test if a swallow becomes a propagate. Commit: "refactor(core): stop swallowing analytics/kg errors; drop dead enable_visualization flag".

### Task 5.5: Real lifecycle archival/compression
**Files:** src/memory/management/lifecycle.rs (~10 "In a real implementation" stub methods at 1162-1569), tests.
**Scope:** These are the highest-value stubs. Implement archival/compression/deletion against the real `Storage` trait (backends support get/store/delete/get_all_entries + compression feature exists). For each stub method: implement it against Storage, OR if it has no consumer, delete it, OR if it's a policy hook genuinely pending real infra, demote its doc honestly (no "In a real implementation, this would:"). Report the disposition of each. TDD: archival moves an entry to archived state retrievable as archived; compression reduces stored size for a large entry. Commit: "feat(lifecycle): real archival/compression against Storage; remove stub scaffolding".

### Task 5.6: Drive remaining "In a real implementation" to zero
**Files:** src/memory/management/mod.rs, src/memory/consolidation/*, src/performance/*, src/multimodal/*, src/observability/*, src/cli/*, src/cross_platform/* (all non-distributed "In a real implementation" + hardcoded-placeholder sites: related_count=5, cluster_size=3, 0.3 connectivity, 10% growth, etc.).
**Scope:** Apply implement/delete/demote to every remaining site (grep `grep -rn "In a real implementation" src/ | grep -v distributed`). Hardcoded fake returns (management/mod.rs related_count/cluster_size/connectivity/growth) → wire to real data (e.g. the ANN counter from 3.2, real graph connectivity) or demote to a documented heuristic with a real (if simple) computation. Delete the redundant `MemoryManager` struct (management/mod.rs, unused `_`-prefixed fields) if still present. Acceptance: `grep -rn "In a real implementation" src/ | grep -v distributed` returns ZERO; every remaining "simplified"/"heuristic" marker has a justifying doc comment. This is a LARGE task — the implementer may split into sub-commits by directory, each gated. Commit(s): "refactor(<area>): implement/demote placeholder logic (drive 'in a real implementation' to zero)".

### Task 5.7: Integrate PolicyEngine + ComplianceChecker disposition
**Files:** src/security/mod.rs (SecurityManager), src/security/policy_engine.rs, src/security/audit.rs or wherever ComplianceChecker lives, tests.
**Scope:** Task 4.7 wired PolicyEngine as a public API but nothing in SecurityManager calls it. Integrate: SecurityManager's access decision consults PolicyEngine::evaluate_access for configured policies. ComplianceChecker::check_compliance is an always-compliant stub — implement a real check against configured rules OR demote it honestly (rename, document it returns a static baseline, stop implying real compliance validation). TDD: a Deny policy blocks access through SecurityManager; compliance check reflects a real configured rule (or honest baseline). Commit: "feat(security): integrate PolicyEngine into access decisions; honest ComplianceChecker".

---
## Order
5.1, 5.2 (retrieval) independent. 5.3 independent (indexing). 5.4 (lib/analytics) independent. 5.5, 5.6 (management — SAME files, run 5.5 before 5.6). 5.7 (security) independent. Run same-file tasks sequentially; others may parallelize on disjoint files.
