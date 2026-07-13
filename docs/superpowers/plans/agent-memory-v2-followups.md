# Agent Memory v2 — Follow-ups Plan

> Execute via subagent-driven-development. Same gates/honesty bar as agent-memory-v2. Branch `feature/agent-memory-v2-followups` off `main` @ 5344fa6.

**Goal:** Fix the real performance findings the eval surfaced (search p95 ~14s, superlinear store growth), build the optional LLM reasoner the spec listed, and produce a real (subset) QA-accuracy number via the codex CLI.

## Global Constraints
- Honesty bar (inherited): disabled paths fail closed; no todo!/unimplemented!/"in a real implementation"/"simulated"; no `.unwrap()`/`println!` in src/. TDD; one conventional commit per deliverable; ledger per task. Gates: fmt; clippy `--all-targets --features "security test-utils" -- -D warnings`; task tests; `cargo test --lib`. CI arbitrates `--all-features`.
- Performance fixes MUST preserve behavior: the retrieval-quality suite, composite_scoring, reranker, bitemporal, intelligent_write, forgetting, reflection tests all stay green (correctness before speed). Add micro-benchmark or timing assertions only where deterministic.

## Root-cause map (from profiler)
- **W1** `MemoryStorage::update_stats` full rescan per store (`storage/memory.rs:210,125-141`) → O(n²) store. Fix: incremental `total_size` counter.
- **W4** `neighbor_facts` scans all short+long-term memories per fact (`lib.rs:607-631`) → O(n·facts)/store. Fix: bounded candidate set (reuse storage tokenized search / cap).
- **W2/W3/L4** KG `find_similar_node` + `auto_detect_relationships` full-node scans held under `kg.write()` (`knowledge_graph/mod.rs:660,544,578`; `lib.rs:505,654`) → O(n)/store blocking concurrent search. Fix: bound candidate scan (time/tag bucket), shrink write-lock critical section (compute outside the lock).
- **S1+L1+S3+W5** TF-IDF `embed` rebuilds the whole IDF table under a global `state.write()` on every call, and dense retriever + reranker embed every candidate per query (`tfidf.rs:79,147`; `dense_vector.rs:53,93`; `rerank.rs:216`). Fix: (a) incremental/lazy IDF; (b) read-only query-time embedding (no vocab mutation on queries); (c) content-hash embedding cache reused across dense + rerank so each candidate is embedded at most once.
- **S2** quadruple full-store `storage.search` per query, each re-lowercasing the corpus (`storage/memory.rs:150-195`). Fix: cache lowercased content / share one candidate fetch across pipelines.
- **L2** GraphRetriever holds `kg.read()` across `storage.retrieve().await` (`strategies.rs:402,416`). Fix: collect keys, drop guard, then retrieve.

---

### Task P1: Write-path O(n²) → near-constant (W1, W4, W2/W3/L4)
**Files:** `src/memory/storage/memory.rs` (incremental stats), `src/lib.rs` (neighbor_facts bounding, write-lock scope), `src/memory/knowledge_graph/mod.rs` (bounded candidate scans, shrink lock).
**Scope:**
1. W1: replace the per-store full `update_stats` rescan with an incremental `total_size` counter updated on insert/update/delete/batch/transaction. Keep `MemoryStats` values correct (test: stats after N inserts == sum, and match a full recompute).
2. W4: `neighbor_facts` must NOT scan the whole store — bound it (e.g. reuse the storage tokenized search to fetch top-K candidates by the fact text, cap K≈20), so conflict resolution still works but is O(K) not O(n). Keep the intelligent_write supersede test green.
3. W2/W3/L4: bound the KG per-store similarity/relationship scans to a candidate set (time-window or tag/embedding-bucketed, capped) instead of all nodes, and shrink the `kg.write()` critical section (compute candidates/similarity outside the lock; hold the write lock only for the actual node/edge mutations). Keep KG behavior tests green.
**TDD:** a store-scaling test asserting per-store cost does NOT grow ~linearly with store size — e.g. store 2000 entries, assert the last-100 mean store time is within a small multiple (≤3×) of the first-100 mean (deterministic-enough on a quiet machine; if flaky, assert an operation-count proxy instead — e.g. instrument that neighbor_facts examined ≤K entries, not n). Prefer the op-count proxy for determinism. Plus all existing write-path/KG tests stay green.
**Commits:** may split by offender ("perf(storage): incremental size stats", "perf(core): bound write-path neighbor and KG scans"). Gates each.

### Task P2: Search/embedding hot path (S1/L1/S3/W5, S2, L2)
**Files:** `src/memory/embeddings/providers/tfidf.rs` (+ `simple_embeddings.rs`), `src/memory/retrieval/dense_vector.rs`, `src/memory/retrieval/rerank.rs`, `src/memory/retrieval/strategies.rs`, `src/memory/storage/memory.rs` (search).
**Scope:**
1. TF-IDF read-only query embedding: embedding for scoring must NOT take a write lock or mutate vocabulary — add a read-only embed path (compute term weights against the existing IDF table without rebuilding it). Store-time vocabulary updates may stay, but make IDF recompute incremental/lazy, not a full rebuild per document.
2. Embedding cache: a content-hash-keyed cache (the `compute_content_hash` helper exists) so a candidate/document is embedded at most once across the dense retriever and the reranker within a query (and reused across queries). Dense retriever + HeuristicReranker consult the cache instead of re-embedding.
3. S2: avoid 4 independent full-store lowercasing scans — cache lowercased content on entries or share one candidate fetch across the pipelines (whichever is cleaner without breaking the pipeline abstraction). At minimum, memoize the lowercased form.
4. L2: GraphRetriever must collect related keys, DROP the `kg.read()` guard, THEN `storage.retrieve` — no lock held across awaits.
**TDD:** correctness-preserving — retrieval_quality + composite_scoring + reranker suites stay green (same rankings). Add a test asserting a query embeds each distinct candidate content at most once (cache hit-count via a test-utils counter), and that the query-time embed path does not mutate vocabulary (vocab size unchanged after N scoring embeds). 
**Commits:** may split ("perf(embeddings): read-only query embed + incremental IDF + content-hash cache", "perf(retrieval): reuse embeddings, drop KG lock before awaits, memoize lowercase"). Gates each.

### Task P3: Re-run eval, prove the speedup (measured)
**Files:** `docs/evaluation.md`.
**Scope:** re-run the LLM-free harness/growth on real LoCoMo (`tools/eval/data/locomo10.json`) after P1+P2; record NEW latency p50/p95/p99 and growth 1k/10k (and attempt 100k if now feasible) BESIDE the pre-fix numbers as a before/after table. Confirm retrieval metrics are unchanged (correctness preserved) or note any delta honestly. No fabricated numbers — real re-run only.
**Commit:** "docs(eval): performance before/after and (if feasible) 100k growth".

### Task R1: Optional LlmReasoner (write-path LLM)
**Files:** `src/memory/reasoning/llm.rs` (feature `llm-reasoning`), `src/memory/reasoning/mod.rs` (export), `src/lib.rs` (optional selection).
**Scope:** implement `LlmReasoner` behind `#[cfg(feature = "llm-reasoning")]` implementing `MemoryReasoner` (extract/resolve/synthesize) by prompting an LLM via the existing reqwest/Ollama provider surface (env-configured endpoint, OpenAI-compatible/Ollama). FAIL-OPEN TO HEURISTIC: when the feature is off OR no endpoint is reachable OR a call fails, fall back to `HeuristicReasoner` (logged, never a fabricated extraction, never a hard failure of the write path). Provide a way to select it (config/env). 
**TDD:** feature-off unchanged (HeuristicReasoner default). Feature-on with a mock/stub HTTP (or a trait-level test double) asserts: extract parses the LLM's structured JSON facts; a failed/unreachable endpoint falls back to the heuristic result (assert the fallback path, not a panic or fake). No network in tests.
**Commit:** "feat(reasoning): optional LlmReasoner with heuristic fallback".

### Task Q1: CodexCliJudge + real QA-accuracy subset (IN PROGRESS, tools/eval only)
Already dispatched: `CodexCliJudge` shelling to `codex exec` (gpt-5.6-sol), run a stratified LoCoMo subset, record the real subset QA accuracy in `docs/evaluation.md` labeled explicitly as a subset. Honesty bar: real measured numbers only; subset clearly marked; not comparable to full-set published numbers.

### Task S1(sweep): Follow-up final sweep
Fake-path grep; revert spot-checks on ≥2 new tests (a perf-correctness test + LlmReasoner fallback); dangling refs; full gate battery; confirm evaluation.md before/after + QA numbers are real; summary.

## Ordering
Q1 runs in parallel (tools/eval, disjoint). P1 → P2 (shared files: memory.rs, lib.rs, strategies.rs — sequential). R1 after P1 (shares lib.rs write path). P3 after P1+P2. Sweep last.
