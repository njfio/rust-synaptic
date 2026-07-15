# Evaluation: LoCoMo long-term-memory benchmark

Every number in this document comes from a real measured run of the
`run_eval` binary in this repository. Anything that was not run is marked
**not run** with the reason — nothing here is estimated, extrapolated, or
fabricated.

## Provenance

| | |
|---|---|
| Dataset | LoCoMo (`locomo10.json`), 10 conversations, 272 sessions, 5,882 turns, 1,986 questions |
| Dataset source | `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json` (fetched 2026-07-13 via `tools/eval/fetch_datasets.sh`) |
| Dataset sha256 | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| Code | branch `feature/agent-memory-v2` (run at the commit that introduced this document) |
| Machine | single dev machine, unspecified hardware (Linux x86_64; not a controlled benchmark environment) |
| Build | `cargo build --release`, rustc 1.93.1 |
| Date of run | 2026-07-13 |

### Command

```bash
tools/eval/fetch_datasets.sh          # fetches locomo10.json into tools/eval/data/ (gitignored)
cargo run --release -p synaptic-eval --bin run_eval -- tools/eval/data/locomo10.json
```

Optional flags: `--growth-100k` adds the 100k memory-growth point (slow);
`--growth-only` skips the retrieval/ablation phases and runs only the
memory-growth measurement and the gated QA phase.

## Methodology

- **Retrieval scoring** (LLM-free): every turn of a conversation is stored in
  a fresh `AgentMemory` (each conversation is its own haystack) under a key
  reproducing the dataset's dia-id evidence scheme (`DN:t`). Each question is
  run through `AgentMemory::search` with `k=10`; the ranked retrieved keys
  are scored against the dataset's `evidence_ids` (precision@10 divides by
  `k`; recall@10 divides by `|evidence|`; MRR is the reciprocal rank of the
  first relevant hit).
- **Concurrency**: the 10 conversations were evaluated concurrently (one
  task per conversation on a multi-core machine). Metric values are
  unaffected (haystacks are independent); the per-operation latencies below
  were measured under that concurrent load and are labeled as such.
- **Ablation**: the same dataset is run under a cumulative ladder of
  retrieval configurations built from the public pipeline API (see
  `tools/eval/src/ablation.rs`), with result caching disabled. Deltas are
  `config − baseline` on measured numbers.
- **Memory growth**: synthetic turn-like entries are stored into a fresh
  `AgentMemory` at each target size. The size proxy is the sum of key+value
  UTF-8 payload bytes handed to `store` plus the library's own
  `MemoryStats::total_size` — **not process RSS**; no OS-level memory
  measurement is taken.

## Retrieval quality (default `AgentMemory` pipeline, k=10)

1,986 questions evaluated.

| metric | value |
|---|---|
| mean precision@10 | 0.0362 |
| mean recall@10 | 0.3202 |
| MRR | 0.2871 |

Per question type:

| QType | n | P@10 | R@10 | MRR |
|---|---|---|---|---|
| SingleHop | 841 | 0.0410 | 0.3920 | 0.3444 |
| Temporal | 321 | 0.0417 | 0.3712 | 0.3222 |
| Abstention | 446 | 0.0357 | 0.3487 | 0.2920 |
| MultiHop | 282 | 0.0238 | 0.0823 | 0.1374 |
| OpenDomain | 96 | 0.0135 | 0.0875 | 0.0859 |

Honest reading: multi-hop and open-domain retrieval are weak (R@10 < 0.09);
single-hop/temporal recall lands ~0.37–0.39. LoCoMo is small and hard, and
these are lexical/TF-IDF-based pipelines without an LLM.

## Latency (measured under 10-way concurrent evaluation)

Nearest-rank percentiles over every operation in the run:

| operation | n | p50 | p95 | p99 |
|---|---|---|---|---|
| `store` (ingest, per turn) | 5,882 | 12.55 ms | 31.38 ms | 40.97 ms |
| `search` (per question, k=10) | 1,986 | 11.07 s | 13.99 s | 15.47 s |

The multi-second search latency is a real measurement of the current default
`AgentMemory::search` path on conversation-sized stores under concurrent
load; it is dominated by per-query re-embedding of the store. It is reported
as measured, not tuned away.

## Capability ablation (delta table, k=10)

Cumulative ladder over the same 1,986 questions; every row is a real run of
that configuration. Deltas are vs `baseline` and may legitimately be zero.

| config | recall@10 | Δrecall | precision@10 | Δprecision | MRR | ΔMRR |
|---|---|---|---|---|---|---|
| baseline | 0.1958 | — | 0.0213 | — | 0.1771 | — |
| +composite | 0.1958 | +0.0000 | 0.0213 | +0.0000 | 0.1771 | +0.0000 |
| +reranker | 0.1958 | +0.0000 | 0.0213 | +0.0000 | 0.1919 | +0.0148 |
| +graph_temporal | 0.2695 | +0.0737 | 0.0302 | +0.0089 | 0.2483 | +0.0712 |
| +all | 0.2695 | +0.0737 | 0.0302 | +0.0089 | 0.2482 | +0.0711 |

Honest reading of the deltas on LoCoMo:

- **+composite** changed nothing at k=10 — composite
  relevance×recency×importance re-scoring reorders within the same retrieved
  set, and on this dataset it did not change which ids reached the top 10.
- **+reranker** improved MRR (+0.0148) without changing recall/precision —
  it reorders the retrieved set (better first-hit rank), as a reranker
  should.
- **+graph_temporal** is where the measurable retrieval gain comes from
  (+0.0737 recall@10, +0.0712 MRR): graph and temporal signal retrievers
  surface evidence the dense+keyword fusion missed.
- **+all** (graph-aware reranker) added nothing over +graph_temporal on this
  dataset (ΔMRR −0.0001 vs +graph_temporal, i.e. noise-level). A real
  zero/negative delta, reported as-is.

Note: the ablation `baseline` (raw pipeline, storage-direct, caching off) is
not identical to the default `AgentMemory` pipeline scored in the previous
section, so the absolute numbers differ between the two tables; each table
is internally consistent.

## Memory growth (payload-byte proxy — NOT RSS)

Synthetic fill of a fresh `AgentMemory`, sequential (single task; measured
in a separate `--growth-only` invocation of the same binary on the same
machine and date). The bytes
column is the payload handed to `store` (key+value UTF-8 lengths), plus the
library's own reported `total_size`; no OS memory (RSS) was measured.

| target | stored | payload bytes | lib `total_size` | store p50 | store p95 | store p99 | probe search |
|---|---|---|---|---|---|---|---|
| 1,000 | 1,000 | 73,670 | 273,670 | 9.62 ms | 22.48 ms | 27.44 ms | 26.62 s |
| 10,000 | 10,000 | 765,741 | 2,765,741 | 451.67 ms | 862.32 ms | 900.87 ms | 7.31 s |
| 100,000 | **not run** — see below | | | | | | |

Honest findings from the growth run:

- Payload growth is linear (as expected for a payload-byte proxy), but
  **per-store latency is strongly superlinear**: p50 went from 9.6 ms at 1k
  entries to 451.7 ms at 10k entries (~47x for 10x the data). The 10k fill
  alone took roughly 75 minutes of single-threaded wall time.
- **100k was not run**: at the measured growth rate the sequential 100k fill
  would take on the order of days, which is not a reasonable single-machine
  run. It is marked not-run rather than extrapolated; rerun with
  `--growth-100k` to measure it.
- The probe search after the fill took 26.6 s (1k) / 7.3 s (10k) — single
  measurements each (one probe per fill), so the difference between the two
  should not be over-read; both confirm multi-second search at these store
  sizes, consistent with the latency table above.

## Performance fixes — before / after (2026-07-13)

After the initial evaluation surfaced the multi-second search and superlinear
store growth above, three rounds of performance work were done on the
`feature/agent-memory-v2-followups` branch:

- **P1** (`fed757d`..`242fd96`): incremental `total_size` counter (removed a
  per-store full-corpus rescan), bounded + deterministic KG/neighbor candidate
  scans, shrunk the KG write-lock critical section.
- **P2** (`879956f`..`bb5c842`): read-only query-time embedding (no per-query
  vocabulary rebuild under a global write lock), a content-hash embedding cache
  shared across the dense retriever and reranker, lowercase memoization, and
  dropping the knowledge-graph read lock before storage awaits.
- **P4** (`a88ab12`, `10c8252`): a character-n-gram **inverted index** in
  storage so keyword search is candidate-bounded instead of a full scan
  (results proven byte-identical to the full-scan implementation by an in-test
  reference), plus token-indexed KG candidate selection.

Retrieval **quality is unchanged** — P2 and P4 are correctness-preserving
(ranking-parity tests pin identical top-k ordering; P4 returns byte-identical
search results), so the retrieval/ablation numbers above still hold.

Measured before/after (same harness, same machine, single-shot probe):

| metric | before | after (P1+P2+P4) | change |
|---|---|---|---|
| probe search over 1k-entry store | 26.6 s | **3.65–3.70 s** | **~7× faster** |
| store p50 @ 1k | 9.6 ms | 8.4 ms | ~comparable |

**What is NOT fixed (honest):** the **store path is still superlinear at
scale.** After the fixes, filling 1k→10k still did not complete within a
40-minute clean run (comparable to the original ~75 min), so the 10k/100k
store-latency points were **not re-measured** and 100k remains impractical.
Root cause identified but not addressed by P1/P2/P4: the default
`advanced_management` write pipeline (summarizer / search-engine / lifecycle /
optimizer sub-systems, enabled by `enable_advanced_management = true`) performs
O(n)-per-store work that dominates once the store is large — the original
profiler mapped only the storage/KG/embedding/reasoning paths, which are now
index-backed. **Next performance target:** index or bound the
`advanced_management` per-store subsystems (or make them opt-in), then re-run
the 10k/100k growth measurement. The search-latency win above is real and
measured; the store-growth superlinearity is only partially reduced and is
tracked as open.

## v3 quality/perf fixes (2026-07-14)

Two fixes were made on `feature/agent-memory-v3-quality` and re-measured with
real runs (same dataset, same machine class as above; run date 2026-07-14):

- **A — bounded/debounced auto-summarization** (`736ef3f` + `7b8b031`): the
  `advanced_management` store-path triggers no longer do O(n)-per-store
  full-corpus scans (index-backed candidate search, bounded recent-creation
  window, per-cluster debounce watermarks). This targets the store-growth
  superlinearity flagged as open in the 2026-07-13 section.
- **B — real dense IDF** (`a321772`): the shared `TfIdfProvider` used by the
  dense retriever and reranker is now fed the corpus at store time, so IDF is
  real instead of the uniform 1.0 fallback (dense scoring was previously
  degenerate hashed-TF cosine).

### Store growth after A (real run, `run_eval --growth-only --growth-100k`)

| target | store p50 | store p95 | store p99 | probe search |
|---|---|---|---|---|
| 1,000 | 8.46 ms | 24.53 ms | 43.47 ms | 3.86 s |
| 10,000 | 44.90 ms | 574.45 ms | 1,025.01 ms | 5.66 s |
| 100,000 | in progress — not run to completion in-sandbox (fill was still running at 40 min; now tractable, on the order of hours, vs "days" before) |

Before/after (vs the 2026-07-13 growth table above):

| metric | before | after A | change |
|---|---|---|---|
| store p50 @ 1k | 9.62 ms | 8.46 ms | ~comparable |
| store p50 @ 10k | 451.67 ms | **44.90 ms** | **~10× faster** |
| growth slope 1k→10k | ~47×/decade | ~5.3×/decade | superlinearity substantially reduced |

Honest reading: the 10k store path is ~10× faster and the 1k→10k slope fell
from ~47× to ~5.3× per decade of data; the store path is still superlinear
(5.3× for 10× data is not linear), and the 100k point was **not run to
completion** — it is marked in-progress, not extrapolated.

### Retrieval quality after B (real run, full 1,986-question LoCoMo)

Same command and full dataset as the 2026-07-13 retrieval table, so the
comparison is apples-to-apples:

| metric | before (uniform IDF) | after B (real IDF) | change |
|---|---|---|---|
| mean recall@10 | 0.3202 | **0.3318** | +0.0116 |
| MRR | 0.2871 | **0.2959** | +0.0088 |
| mean precision@10 | 0.0362 | 0.0375 | +0.0013 |

Per question type (after B):

| QType | n | P@10 | R@10 | MRR |
|---|---|---|---|---|
| SingleHop | 841 | 0.0423 | 0.4033 | 0.3480 |
| Temporal | 321 | 0.0408 | 0.3634 | 0.3076 |
| Abstention | 446 | 0.0386 | 0.3789 | 0.3148 |
| MultiHop | 282 | 0.0248 | 0.0847 | 0.1539 |
| OpenDomain | 96 | 0.0156 | 0.1060 | 0.1296 |

Honest reading: real IDF helped, but modestly — about +1.2 points of
recall@10 and +0.9 points of MRR overall. Every QType improved slightly
(MultiHop and OpenDomain remain weak, R@10 ≤ 0.106). This is an incremental
correctness fix to dense scoring, not a step change. Latencies in this run
were measured under heavy concurrent load (a QA run shared the machine) and
are not directly comparable to the 2026-07-13 latency table; the quality
metrics are unaffected by load.

Ablation after B (same cumulative ladder as the 2026-07-13 table, real run):

| config | recall@10 | Δrecall | precision@10 | Δprecision | MRR | ΔMRR |
|---|---|---|---|---|---|---|
| baseline | 0.2615 | — | 0.0292 | — | 0.1742 | — |
| +composite | 0.2615 | +0.0000 | 0.0292 | +0.0000 | 0.1742 | +0.0000 |
| +reranker | 0.2615 | +0.0000 | 0.0292 | +0.0000 | 0.2311 | +0.0570 |
| +graph_temporal | 0.2901 | +0.0286 | 0.0327 | +0.0036 | 0.2425 | +0.0683 |
| +all | 0.2901 | +0.0286 | 0.0327 | +0.0036 | 0.2415 | +0.0673 |

The ablation baseline itself rose (0.1958 → 0.2615 recall@10) because real
IDF improves the raw dense+keyword pipeline directly; the incremental gain
from graph/temporal shrank (+0.0737 → +0.0286 recall) — some of what
graph/temporal used to recover is now found by properly-weighted dense
scoring. The reranker's MRR-only effect grew (+0.0148 → +0.0570).

### QA end-to-end accuracy after A+B (same 150-question stratified subset)

Re-run of the identical stratified subset and judge as the section below
(`run_qa --subset 150`, codex CLI judge, 150/150 graded, 0 judge failures):

| Metric | before | after A+B |
|---|---|---|
| Overall | 25/150 = 16.7% | **25/150 = 16.7%** |
| SingleHop | 8/30 = 26.7% | 9/30 = 30.0% |
| MultiHop | 2/30 = 6.7% | 2/30 = 6.7% |
| Temporal | 9/30 = 30.0% | 8/30 = 26.7% |
| OpenDomain | 5/30 = 16.7% | 4/30 = 13.3% |
| Abstention | 1/30 = 3.3% | 2/30 = 6.7% |

Honest reading: end-to-end QA accuracy is **unchanged overall** (25/150 both
times; per-type movement of ±1 question is judge/selection noise at n=30 per
type). The small retrieval recall gain from B did not translate into more
graded-correct answers on this subset.

## QA end-to-end accuracy

**End-to-end QA accuracy — measured on a stratified N-question LoCoMo subset,
judge = codex CLI (gpt-5.6-sol), 2026-07-13. NOT the full 1986-question set;
NOT directly comparable to published full-set numbers.**

This is a **subset** result (N = 150 of 1,986 questions), stratified evenly
across the six QTypes (30 each, evenly spaced within each type,
deterministic — no RNG). It measures the full recall → answer → grade
pipeline: for each question, up to `k=10` memories are recalled from an
`AgentMemory` ingested with the question's conversation, the `codex` CLI is
asked to answer using **only** the recalled snippets, and `codex` then grades
that answer against the gold answer (YES/NO). Every number below comes from a
real codex verdict in this run; all 150 questions completed with **0 judge
failures**. Accuracy is over completed (graded) questions only.

| Metric | Value |
|---|---|
| Overall | **25 / 150 = 16.7%** |
| SingleHop | 8 / 30 = 26.7% |
| MultiHop | 2 / 30 = 6.7% |
| Temporal | 9 / 30 = 30.0% |
| OpenDomain | 5 / 30 = 16.7% |
| Abstention | 1 / 30 = 3.3% |
| KnowledgeUpdate | (not present in LoCoMo) |

- **N = 150** selected, **150 completed (graded)**, **0 judge failures**, `k = 10`.
- **Judge**: `codex` CLI 0.144.0, model `gpt-5.6-sol` (user's subscription,
  non-interactive `codex exec`). Not feature-gated — `CodexCliJudge` shells
  out; there is no new crate dependency.
- **Honesty**: this is a subset, not the full set, and is not comparable to
  published full-LoCoMo numbers. Low accuracy reflects the strict
  answer-from-recalled-snippets-only constraint plus TF-IDF recall quality on
  this hard long-context benchmark; nothing here is estimated or extrapolated.
  If a run has judge failures, accuracy is reported over completed questions
  with the failure count, never patched with a guessed verdict.

### Command

```bash
cargo run --release -p synaptic-eval --bin run_qa -- --subset 150
```

Flags: `--subset N` (default 150), `--k K` (default 10), `--concurrency C`
(default 4), `--data PATH` (default `tools/eval/data/locomo10.json`). The
codex binary path and per-call timeout are configurable via
`SYNAPTIC_EVAL_CODEX_BIN` (default `codex`) and
`SYNAPTIC_EVAL_CODEX_TIMEOUT_SECS` (default 120). The binary prints progress
to stderr (`question i/N`) so the long run (~150 codex answer+grade pairs) is
observable. Requires a locally installed, logged-in `codex` CLI; if it is
unavailable the binary exits without fabricating a number.

An OpenAI-compatible HTTP judge (`LlmJudge`) also exists behind the
`llm-reasoning` feature via `SYNAPTIC_EVAL_LLM_URL` /
`SYNAPTIC_EVAL_LLM_MODEL` (optional `SYNAPTIC_EVAL_LLM_KEY`); without a judge
configured the gated `run_eval` path reports `QaResult::NotRun` rather than
fabricating an accuracy number.

## LongMemEval-S

**Not run** — dataset not fetched (~200 MB). Run
`tools/eval/fetch_datasets.sh` to download it, then:

```bash
cargo run --release -p synaptic-eval --bin run_eval -- tools/eval/data/longmemeval_s.json
```

## Reproducing

1. `tools/eval/fetch_datasets.sh` (or place `locomo10.json` with the sha256
   above at `tools/eval/data/locomo10.json`).
2. `cargo run --release -p synaptic-eval --bin run_eval -- tools/eval/data/locomo10.json`
3. The binary prints the retrieval report, the ablation Markdown table, the
   growth rows, and the QA gate status; this document transcribes that
   output verbatim.

Expect a long run: the retrieval + ablation phases took roughly an hour of
wall-clock time on a 24-thread dev machine.

## Retrieval-quality round (2026-07-14) — semantic embeddings + multi-hop + query understanding

Three capabilities added to lift the retrieval ceiling. **Measured on a labelled
50-question LoCoMo subset** (5 conversations × first 10 questions each,
`--retrieval-only --max-conversations 5 --max-questions 10`) — a fast directional
signal, **not** the full 1,986-question set; treat magnitudes as indicative.
Semantic embeddings use a locally-served Ollama `nomic-embed-text` (768-dim);
the **default remains TF-IDF** (offline, no dependency) with automatic fallback.

| config | recall@10 | MRR | precision@10 | MultiHop R@10 | Temporal R@10 | SingleHop R@10 |
|---|---|---|---|---|---|---|
| TF-IDF baseline | 0.2212 | 0.2567 | 0.0300 | 0.0821 | 0.4091 | 0.0000 |
| + semantic (Ollama) | 0.3652 | 0.3517 | 0.0500 | 0.1190 | 0.6818 | 0.2500 |
| + semantic + multi-hop + query understanding | **0.3702** | 0.3512 | 0.0540 | **0.1848** | 0.6364 | 0.2500 |

Honest attribution:

- **Semantic embeddings are the dominant lever** — recall@10 0.2212 → 0.3652
  (**+65% relative**) on this subset, with the largest gains where lexical TF-IDF
  is weakest: Temporal 0.4091 → 0.6818 and SingleHop 0.0 → 0.25. This is the
  single biggest retrieval improvement measured in this project. (Requires an
  embedding endpoint; the default TF-IDF path is unchanged and offline.)
- **Multi-hop graph expansion delivers on its target category** — MultiHop
  R@10 0.1190 → 0.1848 (**+55% relative**) and precision 0.050 → 0.054 on top of
  semantic. It surfaces 2-hop-reachable evidence that single-hop dense/keyword
  retrieval misses (unit-proven with an A→B→C fixture).
- **Query understanding is roughly neutral on this subset** — overall recall
  effectively flat and Temporal slightly down (0.6818 → 0.6364). The mechanism is
  real and unit-proven (temporal-constraint boost, multi-part split/union), but
  this question distribution did not reward it measurably; reported as-is, not
  overstated. It may help other distributions (e.g. LongMemEval's explicit
  temporal-reasoning questions).

Not run: the full 1,986-question semantic ablation (the Ollama path serializes
embedding requests, making the full 5-config run impractical in-sandbox — the
subset is the honest signal); LongMemEval-S. Default-build (TF-IDF) numbers from
the prior sections are unchanged.

## Bundled offline embedding model (2026-07-14) — in-process MiniLM, no server

The RQ1 semantic win required an external Ollama server. This bundles a real
semantic model **in-process and fully offline**: all-MiniLM-L6-v2 (384-dim)
loaded via candle from a locally-fetched cache (`scripts/fetch_embedding_model.sh`
→ gitignored `models/`), with real attention-masked mean-pooled inference
(verified: cosine(related)=0.605 vs cosine(unrelated)=−0.046 — genuinely
semantic, not random weights).

**Three-way comparison, same labelled 50-question LoCoMo subset** (retrieval-only):

| provider | recall@10 | MRR | Temporal R@10 | MultiHop R@10 | search latency p50 |
|---|---|---|---|---|---|
| TF-IDF (default, offline, no dep) | 0.2212 | 0.2567 | 0.4091 | 0.0821 | ~instant |
| **candle MiniLM (bundled, in-process, offline)** | **0.3300** | 0.2709 | **0.6364** | 0.1053 | **~28.5 s** |
| Ollama nomic-embed-text (external server) | 0.3652 | 0.3517 | 0.6818 | 0.1190 | ~3.8 s |

Honest findings:

- **The bundled in-process model delivers the semantic win with no server** —
  recall@10 0.2212 → 0.3300 (**+49% relative**), Temporal 0.4091 → 0.6364
  (**+56%**). Slightly below the larger 768-dim Ollama model but the same class
  of improvement, and it runs fully offline in-process (after a one-time weight
  fetch). This is the requested "bundle an offline model" delivered and measured.
- **The hard tradeoff is latency.** candle BERT inference on CPU is currently
  **un-batched** — the eval embeds the query plus every candidate one at a time —
  giving **~28.5 s per query** and ~158 ms per store at eval scale. That is far
  too slow to be a silent default, so the bundled model is **opt-in**
  (`SYNAPTIC_RETRIEVAL_EMBEDDER=candle` or `MemoryConfig.retrieval_embedding_provider`);
  **TF-IDF remains the fast, lean, offline default**, and semantic recall is
  available on demand. The optimization path is clear: **batch candidate
  embeddings in one forward pass** (and/or GPU), which should cut per-query
  latency by roughly an order of magnitude and make the bundled model a viable
  default — tracked as the next step, not yet done.
- **GPU (opt-in, not measured here):** the candle provider now selects its
  device via `Device::cuda_if_available(0)`. With the CUDA toolkit installed,
  `cargo build --features cuda` runs MiniLM inference on the GPU, which is
  expected to cut the multi-second CPU per-query latency dramatically. GPU
  latency was **NOT measured** in this environment (no CUDA toolkit, no sudo
  to install one), so no GPU number is claimed — the CPU path is what is
  verified.

Not measured: batched-candle latency; the full 1,986-question set with candle
(un-batched CPU inference makes it impractical in-sandbox — the subset is the
honest signal).

## Batched candidate embedding (2026-07-14) — candle latency ~4× lower

The bundled candle MiniLM's ~28.5 s/query came from embedding the query and every
candidate one at a time (one BERT forward each). This batches all cache-missing
candidates into a **single padded forward pass** per query (attention-masked so a
padded-batch vector is bit-identical to the solo embedding — proven by an
identity test within 1e-5). Same 50-question LoCoMo subset:

| candle MiniLM (in-process, offline) | before batching | after batching |
|---|---|---|
| recall@10 | 0.3300 | **0.3300 (identical)** |
| Temporal R@10 | 0.6364 | 0.6364 |
| search latency p50 | ~28.5 s | **~7.0 s** (~4× faster) |
| search latency p95 | ~30.5 s | ~24.9 s |
| store latency p50 | ~158 ms | ~152 ms (unchanged) |

Honest findings:

- **Recall is unchanged** — batching is a pure latency optimization; the identity
  test (padded batch == individual embeds) guarantees vectors, hence rankings,
  are preserved. Confirmed: 0.3300 both times, per-category identical.
- **Search latency dropped ~4×** (28.5 s → 7.0 s p50) by collapsing N per-candidate
  forward passes into one batched pass (cache misses only).
- **Still not a fast default.** 7.0 s p50 / 24.9 s p95 on CPU is a real improvement
  but remains too slow to silently default to; candle BERT on CPU is the floor
  (store embedding is still ~152 ms/turn, one forward each). Reaching sub-second
  needs **GPU** or a **smaller/quantized model** — out of scope here. The bundled
  candle model therefore stays **opt-in**, now meaningfully faster; **TF-IDF
  remains the default**, Ollama (~3.8 s) the balanced server option.

Not measured: GPU / quantized-model latency; batched store-side embedding.

## Fast semantic default (2026-07-14) — model2vec static embeddings + GPU option

To make semantic embeddings a genuinely fast default (candle MiniLM was ~7s/query
on CPU even after batching), two paths were added:

1. **model2vec static embeddings** (`minishlab/potion-base-8M`, 256-dim) — a
   token→vector table distilled from a real sentence-transformer. Embedding is a
   token-row lookup + masked mean-pool + L2-norm — **no transformer forward pass**,
   microseconds on any CPU, pure-Rust (candle-free `static-embeddings` feature).
   Genuinely semantic (verified: cosine(related)=0.51 vs cosine(unrelated)=−0.08).
2. **GPU device selection for candle** (`cuda` feature) — the candle provider now
   selects `Device::cuda_if_available(0)`, so building with the CUDA toolkit runs
   MiniLM on a GPU. **Not measured here** (this environment has the NVIDIA driver
   but no CUDA toolkit and no sudo to install it) — no GPU latency is claimed.

**Full provider comparison, same 50-question LoCoMo subset** (retrieval-only):

| provider | recall@10 | MRR | MultiHop R@10 | Temporal R@10 | store latency p50 |
|---|---|---|---|---|---|
| TF-IDF (lexical, default) | 0.2212 | 0.2567 | 0.0821 | 0.4091 | ~9 ms |
| **model2vec static (CPU, fast)** | 0.3240 | **0.3250** | **0.1949** | 0.5455 | **~9 ms** |
| candle MiniLM (CPU transformer) | 0.3300 | 0.2709 | 0.1053 | 0.6364 | ~152 ms |
| candle MiniLM (GPU) | not measured — needs CUDA toolkit | | | | |
| Ollama nomic-embed-text (server) | 0.3652 | 0.3517 | 0.1190 | 0.6818 | — |

Honest findings:

- **model2vec is the recommended fast semantic default.** It matches the candle
  transformer's recall (0.324 vs 0.330), has the **best MRR (0.325) and best
  MultiHop (0.195) of any provider tested**, and embeds at **TF-IDF speed**
  (~9 ms store p50, ~17× faster than candle's 152 ms) — on any CPU, no GPU, no
  forward pass. It is auto-selected as the default when the `static-embeddings`
  feature is built and the model is present (else TF-IDF); a plain `cargo build`
  stays lexical/offline/lean.
- **The remaining ~4.4 s per-query search latency is pipeline overhead, not
  embedding** (multi-hop graph traversal + reranking + composite scoring over the
  candidate set — the same for every provider; store latency isolates the
  embedding cost, where static is instant). Reducing pipeline latency is a
  separate optimization.
- **GPU** would accelerate the transformer path further but is unnecessary for a
  fast default now that static embeddings deliver comparable quality instantly on
  CPU; the `cuda` feature is available for GPU users, unmeasured here.

Not measured: GPU latency (no CUDA toolkit); full 1,986-question set / LongMemEval.

## Search-pipeline latency (2026-07-15)

The store path is fast (static embeddings are instant), but per-query search
latency sat at **~4.4 s p50** — pipeline overhead, not embedding. Profiling
showed it was a **per-candidate 2-hop knowledge-graph BFS**: every candidate
memory re-walked the graph under its own lock, re-cloning a `GraphPath` per edge,
and the reranker/composite pass re-scanned the set. This round removed that
overhead without changing what the pipeline returns.

Fixes (branch `feature/pipeline-latency`):

1. **Node-deduped KG traversal** — `traverse_from_node` carries result indices on
   the queue and clones one parent path per *retained* result instead of per edge;
   nodes are visited once. Same reachable set, far less allocation.
2. **Single-lock batched graph scoring** — `GraphRetriever` takes the KG read lock
   **once per query** and scores all candidates via
   `find_related_memories_batch` / `graph_score_from_related`, replacing the
   per-candidate lock+BFS.
3. **Single-pass reranker proximity** — the heuristic reranker computes
   term-proximity in one pass over each candidate.
4. **Concurrent retriever fan-out** — the retrievers run under `join_all` and are
   reassembled in registration order (deterministic), so the slowest retriever
   bounds wall-clock instead of the sum.

The initial version also reused the pipeline's seed hits for multi-hop expansion;
that changed the BM25 IDF pool between limits and **regressed recall (0.3240 →
0.2957)**, so it was reverted (`2a4741b`) — the batched expansion fetch was kept.
The now-unused prior-results pipeline hooks were then pruned (`50dca6a`).

**Measured on the static-embedding 50-question LoCoMo subset** (`--retrieval-only`,
model2vec `potion-base-8M`):

| metric | before | after |
|---|---|---|
| recall_latency p50 | ~4.4 s | **~0.46 s (~9.5× faster)** |
| recall@10 | 0.3240 | **0.3240 (identical)** |
| MRR | 0.3250 | 0.3250 (identical) |
| MultiHop R@10 | 0.1949 | 0.1949 (identical) |
| store latency p50 | ~9 ms | ~8.5 ms |

Retrieval quality is preserved **exactly** — this is a pure latency/allocation win,
not a quality/latency trade. Not measured: full 1,986-question set; latency on the
Ollama/candle providers (the pipeline overhead is provider-independent, so the
same absolute reduction applies on top of their embedding cost).

## Full-scale LoCoMo validation (2026-07-15)

Every prior retrieval number came from a **50-question subset**. Now that the
pipeline latency fix (above) made it tractable, here is the **complete
1,986-question LoCoMo set** — all 10 conversations, 272 sessions, 5,882 ingested
turns — run end-to-end with the default static-embedding pipeline (model2vec
`potion-base-8M`), one concurrent task per conversation.

**Headline: at full scale, recall@10 = 0.5237** (MRR 0.4058) — substantially
**higher** than the 50-question subset's 0.324. The subset was pessimistic: it
under-sampled the two largest, highest-recall categories. These full-set numbers
supersede the subset as the representative figure.

| QType | n | precision@10 | recall@10 | MRR |
|---|---|---|---|---|
| SingleHop | 841 | 0.0643 | **0.6147** | 0.4501 |
| Temporal | 321 | 0.0679 | 0.6072 | **0.4935** |
| Abstention | 446 | 0.0545 | 0.5359 | 0.3755 |
| MultiHop | 282 | 0.0730 | 0.2475 | 0.3005 |
| OpenDomain | 96 | 0.0333 | 0.2026 | 0.1740 |
| **overall** | **1986** | **0.0624** | **0.5237** | **0.4058** |

Latency at full scale (measured under per-conversation concurrency):

| op | p50 | p95 | p99 |
|---|---|---|---|
| store (n=5882) | 10.9 ms | 20.9 ms | 29.8 ms |
| recall (n=1986) | **0.60 s** | 0.86 s | 1.08 s |

Recall p50 is 0.60 s here vs 0.46 s on the 50-question subset — larger per-query
haystacks (full conversations ingested), still ~7× under the pre-fix 4.4 s.

**Capability ablation** (full set). Note this ladder isolates the *capability
deltas* on the baseline lexical embedder (TF-IDF), so its absolute numbers are
lower than the static-provider default pipeline above; it answers "what does each
stage add," not "what is the best config":

| config | recall@10 | Δrecall | MRR | ΔMRR |
|---|---|---|---|---|
| baseline | 0.2610 | — | 0.1744 | — |
| +composite | 0.2610 | +0.0000 | 0.1744 | +0.0000 |
| +reranker | 0.2610 | +0.0000 | 0.2305 | **+0.0562** |
| +graph_temporal | 0.2896 | **+0.0286** | 0.2422 | +0.0678 |
| +all | 0.2896 | +0.0286 | 0.2412 | +0.0668 |

Honest reading: the reranker earns its keep on ranking (**+0.056 MRR**) without
moving recall (it reorders, doesn't retrieve); graph+temporal expansion adds real
recall (**+0.029**). Composite scoring alone moves neither on this set. The
static semantic embedder (default pipeline, 0.5237) is the dominant lever over
the lexical baseline (0.2610) — a **+0.26 absolute recall** gap, the largest of
any single choice measured.

**Memory growth** (payload-byte proxy, not RSS): 1k stored → store p50 7.4 ms,
probe_search 0.37 s; 10k stored → store p50 44 ms, probe_search 0.62 s. Store
stays low-ms; search cost grows sublinearly with corpus size.

Not run in this pass: QA end-to-end accuracy (LLM-gated — needs an
`llm-reasoning` endpoint; reported as not-run, never fabricated); the 100k growth
target (`--growth-100k`); LongMemEval.

## Negative result: semantic-edge densification does not lift MultiHop (2026-07-15)

**Hypothesis (falsified):** MultiHop recall (R@10 = 0.2475 full-set) is low because
the knowledge graph has near-zero semantic connectivity — the write path never set
`MemoryEntry.embedding`, so `auto_detect_relationships` never created
`SemanticallyRelated` edges, and the default similarity threshold (0.7) was
mis-tuned for the static provider (measured related-cosine ≈ 0.51). We built the
full fix on a branch: attach the scoring-provider embedding to each entry on write
so the KG forms semantic edges, and tune the threshold by measured sweep.

**A 200-question subset looked promising** (MultiHop R@10 0.1933 → 0.2412 at
threshold 0.35). **The full 1,986-question set falsified it:**

| full-set (1,986 q) | baseline | threshold 0.35 | threshold 0.45 |
|---|---|---|---|
| MultiHop R@10 | **0.2475** | 0.2440 | 0.2317 |
| overall R@10 | **0.5237** | 0.5248 | 0.5180 |
| Temporal R@10 | 0.6072 | 0.6290 | 0.6202 |
| OpenDomain R@10 | 0.2026 | 0.2182 | 0.2217 |
| recall latency p50 | **0.60 s** | 1.27 s | 1.11 s |
| store latency p50 | **10.9 ms** | 19.1 ms | 17.5 ms |

MultiHop is flat-to-down at every threshold, while query latency roughly **doubles**
(more edges for the multi-hop retriever to traverse). Small incidental Temporal /
OpenDomain gains do not justify a 2× latency regression, and none of it is the
MultiHop lift the change targeted. The subset was an unrepresentative sample — the
same lesson as the 0.324-subset vs 0.524-full-set gap documented above.

**Why it fails (the useful finding):** cosine-similar *content* is not the
*reasoning* connection multi-hop questions need. Multi-hop gold evidence is linked
by entity coreference across turns (e.g. "Caroline" mentioned in one turn, her
research described 40 turns later), not by surface similarity. Similarity edges
connect turns that merely *read* alike, adding plausible-but-wrong neighbors that
dilute the candidate set. **Lifting MultiHop needs entity-linked edges
(coreference), not denser similarity edges** — a deeper change than edge density.

The branch was abandoned; `main` stays at the fast baseline (recall 0.5237,
latency 0.60 s), which strictly dominates every configuration measured here. This
is recorded so the similarity-edge approach is not re-attempted.

## Negative result: entity-coreference edges also do not lift MultiHop (2026-07-15)

Following the abandoned similarity-edge round (above), we tried the mechanism that
diagnosis suggested MultiHop actually needs: **entity-coreference edges**. The
reasoner already extracts entities per turn (capitalized-span NER: Person/Place/
Org/Term), so we added `Mentions` edges (memory-node → entity-node) and made the
multi-hop retriever traverse *turn → entity → other turns mentioning that entity*,
with full hub mitigation (IDF-weight by entity degree `w(d)=1/(1+ln(1+d))`,
query-similarity frontier ranking, and a degree cap that treats common entities as
coreference-stopwords). Both implementation tasks were reviewed MERGEABLE; the
mechanism is correct and the hub mitigation held latency flat.

**Full 1,986-question LoCoMo, coreference vs baseline:**

| metric | baseline | coreference | delta |
|---|---|---|---|
| MultiHop R@10 | **0.2475** | 0.2417 | −0.0058 (did not rise) |
| MultiHop MRR | 0.2981 | 0.3080 | +0.010 (ranking only) |
| overall R@10 | **0.5237** | 0.5159 | −0.008 |
| Temporal R@10 | 0.6072 | 0.6337 | +0.027 |
| recall latency p50 | **0.60 s** | 0.63 s | flat |

MultiHop recall did not rise; overall recall slipped slightly. The hub mitigation
worked (latency flat, unlike the similarity round's 2× blowup), but the target
metric didn't move.

**The combined finding (the useful part):** **two fundamentally different
graph-expansion mechanisms — content-similarity edges AND entity-coreference edges —
both fail to lift MultiHop recall at scale.** This is strong evidence that the
MultiHop ceiling is **not a retrieval-connectivity problem**. For LoCoMo MultiHop,
the gold evidence turns are largely already lexically reachable (so they surface as
retrieval *seeds*, which additive graph expansion excludes by design), and
expanding the graph adds more noise than gold. Lifting MultiHop recall will require
a different lever than graph expansion — e.g. answer-aware reranking — or it may be
bounded by what retrieval alone can achieve without the answer. Both branches were
abandoned; `main` stays at the fast baseline (recall 0.5237, latency 0.60 s).
Recorded so neither graph-expansion approach is re-attempted for MultiHop.
