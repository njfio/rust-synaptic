# Retrieval Quality Round — Implementation Plan

> Execute via subagent-driven-development. Same gates/honesty bar as prior rounds. Branch `feature/retrieval-quality` off `main` @ 7e09f79.

**Goal:** Lift the measured retrieval quality that the eval shows is the system's ceiling (recall@10 ≈ 0.33, MRR ≈ 0.30, MultiHop 6.7%, OpenDomain ~9%, QA 16.7%) via three targeted capabilities, each proven with a real LoCoMo ablation.

## Global Constraints
- Honesty bar: disabled/unconfigured paths fail closed OR fall back to the real TF-IDF path (logged, never fabricated); no todo!/unimplemented!/"in a real implementation"/"simulated"; no `.unwrap()`/`println!` in src/. TDD; one conventional commit per deliverable; ledger per task. Gates: fmt; clippy `--all-targets --features "security test-utils" -- -D warnings`; task tests; `cargo test --lib`. CI arbitrates `--all-features`.
- **Default build stays lean & offline:** TF-IDF remains the no-dependency default embedder. Semantic embedding is opt-in (feature/config); when unavailable the pipeline falls back to TF-IDF — never a hard failure.
- **Every quality claim ships with a measured LoCoMo number** (`tools/eval`), before/after, in `docs/evaluation.md`. No unmeasured claims.
- **Enabler:** Ollama `nomic-embed-text` (768-dim) is serving locally and is used as the *measured* semantic embedder; the existing `src/memory/embeddings/providers/ollama.rs` is the client.

## Root findings (from prior rounds + survey)
- Dense retriever uses hash/TF-IDF cosine — lexical only, can't do paraphrase/synonym → kills MultiHop/OpenDomain.
- Multiple embedder instances (SimpleEmbedder in EmbeddingManager, TfIdfProvider for retrieval, reasoner's) — inconsistent; source of the inert-IDF class of bug.
- KG exists but retrieval doesn't traverse it for multi-hop.
- Query is embedded as-is; no decomposition/temporal extraction.

---

### Task 1: Pluggable semantic embedding + unify providers (the #1 lever)
**Files:** `src/lib.rs` (retrieval provider selection), `src/memory/embeddings/` (unify; provider selection), `src/memory/embeddings/providers/ollama.rs` (ensure it implements `EmbeddingProvider` + `embed_for_scoring`), `Cargo.toml` (feature if needed), tests.
**Scope:**
- Make the retrieval pipeline's embedding provider **configurable** (via `MemoryConfig`/builder): default `TfIdfProvider` (offline, no dep), optional a real semantic provider (Ollama `nomic-embed-text`, or the candle model). One shared `Arc<dyn EmbeddingProvider>` used by the dense retriever + reranker + the store-time feed (extends the v3 shared-provider work).
- Ensure `OllamaProvider` implements the full `EmbeddingProvider` trait including `embed_for_scoring` (may just delegate to `embed` since Ollama IDF isn't corpus-relative — a real dense model doesn't need corpus IDF; document that semantic providers ignore the read-only/IDF distinction).
- **Fallback:** if the configured semantic provider is unreachable (Ollama down / no endpoint), fall back to `TfIdfProvider` with a warn — never fail the pipeline.
- Unify: the write-path `EmbeddingManager` and the retrieval provider should share ONE provider instance/type so embeddings are consistent (removes the SimpleEmbedder-vs-TfIdfProvider split). If full unification is too invasive, at minimum make the retrieval provider configurable and document the remaining split.
**TDD:** `tests/semantic_provider_tests.rs` — with a mock/stub `EmbeddingProvider` returning known semantic-like vectors (paraphrase pair close, unrelated far), assert the dense retriever ranks a **paraphrase** of the query above a lexically-overlapping-but-unrelated doc — something TF-IDF cannot do (assert TF-IDF fails this, semantic passes). Fallback test: configured-but-unreachable semantic provider → TF-IDF used, no panic. No network in unit tests.
**Measure (real):** run the LoCoMo retrieval eval with the Ollama semantic provider vs the TF-IDF default; record recall@10/MRR/per-QType before/after in `docs/evaluation.md`. Expect the biggest single recall jump.
**Commit:** "feat(retrieval): pluggable semantic embedding provider (Ollama/candle), TF-IDF fallback".

### Task 2: Multi-hop graph retrieval (HippoRAG-style)
**Files:** `src/memory/retrieval/` (a graph-traversal retriever or an expansion stage), `src/memory/knowledge_graph/`, tests.
**Scope:** Add iterative retrieve→expand→retrieve over the KG: seed with the top semantic hits, expand along KG edges (relations), and re-rank the expanded set — so a fact reachable only via a 2-hop relation surfaces. Optionally a lightweight personalized-PageRank over the seed set. Bounded (cap hops/frontier) and deterministic. Feed into the existing fusion.
**TDD:** `tests/multihop_retrieval_tests.rs` — construct A→B→C relations where the query matches A but the answer is at C; assert multi-hop retrieval surfaces C, which single-hop dense/keyword misses. Deterministic.
**Measure:** LoCoMo ablation — recall on the **MultiHop** QType with vs without multi-hop expansion (the target category, currently 6.7%). Record real delta.
**Commit:** "feat(retrieval): multi-hop graph-expansion retrieval".

### Task 3: Query understanding / decomposition
**Files:** `src/memory/retrieval/` (a query-preprocess stage), tests.
**Scope:** A query-preprocessing stage: (a) temporal-constraint extraction (dates/relative-time → a temporal filter/boost for the TemporalRetriever), (b) multi-part question splitting (conjunctions / "and"/multi-entity questions → sub-queries whose results union). Heuristic/deterministic by default; LLM-optional (reuse `llm-reasoning`) but not required. Feeds the pipeline.
**TDD:** `tests/query_understanding_tests.rs` — a temporal question ("what did X do in 2021") extracts the 2021 constraint and boosts temporally-matching memories; a two-part question splits into two sub-queries whose union covers both answers. Deterministic.
**Measure:** LoCoMo ablation — recall on **Temporal** (currently 30%) and **MultiHop** with vs without query understanding.
**Commit:** "feat(retrieval): query decomposition and temporal-constraint extraction".

### Task 4: Re-measure + honest evaluation
**Files:** `docs/evaluation.md`.
**Scope:** Full LoCoMo ablation with the new capabilities stacked: TF-IDF baseline → +semantic → +multihop → +query-understanding, reporting recall@10/MRR/precision + per-QType, and re-run the codex QA subset. Record real before/after; mark anything not run. Note the semantic numbers require Ollama (state the model + that the default is TF-IDF). No overclaim; if a capability doesn't help a category, report the real (possibly zero) delta.
**Commit:** "docs(eval): retrieval-quality ablation (semantic + multihop + query understanding)".

### Task 5: Final sweep + PR
Fake-path grep; revert spot-checks on ≥2 new tests (semantic-beats-TFIDF, multihop-surfaces-C); dangling refs; full gates; confirm evaluation numbers real; summary + PR.

## Ordering
Task 1 first (foundation + biggest lever); 2 and 3 build on the semantic provider; 4 measures the stack; 5 closes. Tasks 1–3 touch overlapping retrieval files → sequential. Gate-b (a heavyweight model dep) only if candle-bundled is pursued; Ollama needs no new crate (reqwest already present).
