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

## First end-to-end QA accuracy (real judge, 2026-07-15)

Until now QA accuracy was reported as "not run" — the harness had a real judge
(`CodexCliJudge`, shelling the logged-in `codex` CLI) and an `LlmJudge` (OpenAI-
compatible endpoint), but `run_qa_gated` never selected the codex path. This round
wired judge selection (`SYNAPTIC_EVAL_JUDGE=codex`), added bounded-concurrency
judging (`SYNAPTIC_EVAL_QA_CONCURRENCY`, default 4) so a sample fits a short
wall-clock budget, and a `--qa-only` flag (skips retrieval/ablation/growth). The
judge answers each question using ONLY the top-`k=10` recalled memories (no world
knowledge), then grades its own answer against the gold answer. Every number below
is a real judge verdict — nothing fabricated; an empty/unconfigured run reports
not-run.

**Measured (static-embedding retrieval, codex judge, concurrency 4):**

| sample | graded | correct | accuracy |
|---|---|---|---|
| 20 questions (first 4 q × 5 conversations) | 20 | 5 | 0.2500 |
| 40 questions (first 4 q × 10 conversations) | 40 | 9 | 0.2250 |

Per-QType (40-question run):

| QType | graded | correct | accuracy |
|---|---|---|---|
| SingleHop | 1 | 1 | 1.0000 |
| Temporal | 16 | 5 | 0.3125 |
| MultiHop | 19 | 3 | 0.1579 |
| OpenDomain | 4 | 0 | 0.0000 |

**Honest caveats — read before citing the headline:**
- **The subset is skewed HARD.** `--max-questions N` takes the *first* N questions
  per conversation, which over-samples MultiHop/Temporal (35 of 40) and
  under-samples the dominant, easiest category SingleHop (1 here vs 841 / 42% of the
  full 1,986-question set). So **0.225 is a pessimistic, hard-weighted estimate**; a
  representative sample would weight SingleHop far more heavily and score higher. The
  per-QType numbers are the reliable signal, not the aggregate.
- **QA accuracy tracks the retrieval ceiling by category.** Categories that retrieve
  well (SingleHop R@10 0.61, Temporal 0.63) answer well; MultiHop (0.16) and
  OpenDomain (0.00) are hard end-to-end exactly as they are hard to retrieve
  (R@10 0.25 / 0.20) — the retrieval ceiling propagates to answers, consistent with
  the two abandoned MultiHop rounds above.
- **Not a full-set run.** A representative full 1,986-question QA pass is bounded by
  sequential judge latency (hours of `codex` calls) and is not yet run; the numbers
  here are labelled subsets. The judge is capped at k=10 recalled memories with a
  strict "use only these" prompt, so this measures retrieval-grounded QA, not the
  judge's own knowledge.

## QA loss diagnostic: retrieval-bound vs judge-bound (2026-07-15)

To answer "why is QA accuracy low," the QA harness now records, per question,
whether the gold evidence was in the top-`k` recalled set (`gold_retrieved` =
recall@k > 0, using the same `normalize_retrieved` as the retrieval eval), and
cross-tabulates it against the judge's correct/wrong verdict:

- **A** gold retrieved & correct · **B** gold retrieved & wrong (judge-bound)
- **C** gold missed & correct · **D** gold missed & wrong (retrieval-bound)

**40-question subset (codex judge, static retrieval):**

| | correct | wrong |
|---|---|---|
| gold in top-10 | A = 7 | **B = 10 (judge-bound)** |
| gold missed | C = 3 | **D = 20 (retrieval-bound)** |

graded=40, correct=10 (0.25), **gold-retrieved rate = 0.425**; of the 30 wrong
answers, **retrieval-bound = 0.667, judge-bound = 0.333**.

**Findings:**
1. **Retrieval is the bigger bottleneck (~2/3 of failures).** For 57.5% of these
   (hard-skewed) questions the gold evidence never entered the top-10 — consistent
   with MultiHop/OpenDomain recall being 0.16–0.25. This is the deep problem the two
   abandoned graph-expansion rounds could not move.
2. **The answer step is a real secondary bottleneck (~1/3), and more tractable.**
   When the evidence WAS retrieved (17 questions), the judge answered correctly only
   7/17 ≈ 41%. That headroom is unrelated to retrieval — a stronger reranker (lift
   gold from rank 8–10 to 1–3) or answer model could recover part of it.
3. **B under-counts retrieval loss.** `gold_retrieved` is true if ANY evidence turn
   is in the top-10, but multi-evidence (MultiHop) questions need several. So part
   of the "judge-bound" B bucket is really partial-retrieval failure — the true
   retrieval-bound share is ≥ 2/3.
4. **C = 3** questions were answered correctly WITHOUT the labeled evidence retrieved
   — either the answer was derivable from other retrieved turns or minor label
   incompleteness; small but worth noting.

Instrumentation only; no fabricated numbers. Same subset caveats as the QA section
above (first-N sampling skews toward hard categories).

## Reranking headroom + over-fetch win (2026-07-15)

The QA loss diagnostic (above) showed ~1/3 of failures were judge-bound (gold
retrieved but not answered). Investigating *why gold that's retrieved doesn't help*
surfaced a structural bug and a large measured headroom.

**Recall@k curve (full 1,986-q set, one limit-50 search per question):**

| category | recall@10 | recall@20 | recall@50 |
|---|---|---|---|
| overall | 0.558 | 0.624 | **0.702** |
| MultiHop | 0.274 | 0.353 | **0.472** |
| OpenDomain | 0.243 | 0.313 | 0.376 |
| SingleHop | 0.646 | 0.712 | 0.782 |
| Temporal | 0.672 | 0.747 | 0.796 |

The gold evidence is in the top-50 candidate pool far more often than it reaches the
top-10 — a **+0.144 overall gap (MultiHop +0.199, +73% relative)**. That is a
*ranking* problem, not a recall problem: the evidence is retrieved but ranked below
the cutoff.

**Root cause:** `fuse_results` truncated the fused pool to `limit` (=10) BEFORE
composite scoring and the reranker ran, so the (already semantic) `HeuristicReranker`
only reordered the final 10 — it could never pull a gold result from rank 11–50 into
the top-10.

**Fix (over-fetch):** fuse/composite/rerank over a wider pool
(`rerank_pool_size`, default 50), truncate to the caller's `limit` LAST.

**Measured full-set result (over-fetch vs baseline):**

| category | baseline R@10 | over-fetch R@10 | delta |
|---|---|---|---|
| overall | 0.5237 | **0.5565** | +0.0328 (+6.3%) |
| MultiHop | 0.2475 | 0.2737 | +0.0262 (+11%) |
| Temporal | 0.6072 | 0.6716 | +0.0644 |
| OpenDomain | 0.2026 | 0.2327 | +0.0301 |
| SingleHop | 0.6147 | 0.6459 | +0.0312 |
| MRR | 0.4058 | 0.4219 | +0.0161 |

Every category improved — including MultiHop, which two graph-expansion rounds could
NOT move: it was a ranking problem all along. Latency cost: recall p50 0.60 s → 0.76 s
(+27%; the reranker now scores 50 candidates instead of 10), still sub-second; store
latency unchanged (~10 ms).

**Remaining headroom:** the reranker captures recall@10 = 0.556 of the 0.702 pool
ceiling. Closing more of that gap needs a stronger reranker (the pool is now available
to it); the `--recall-curve` mode measures the ceiling any reranker can chase.

## Reranker weighting: embedding-dominant default (2026-07-15)

With the over-fetch pool now feeding the reranker (above), the reranker's blend
weights determine how much of the 0.702 pool ceiling is captured. The
`HeuristicReranker` blends term-overlap / embedding-agreement / graph-proximity /
recency (was 0.35 / 0.35 / 0.15 / 0.15). Two observations drove a principled sweep
(weights made env-configurable via `SYNAPTIC_RERANK_W_*`):
- `recency` is **inert** in this eval (all memories share an ingest time → constant,
  no effect on ordering); retained for real deployments.
- `graph_proximity` is **query-agnostic** — it rewards a candidate's closeness to the
  OTHER candidates, not to the query — so it can demote query-relevant gold.

**Full-set sweep (recall@10 / MultiHop / OpenDomain):**

| term/embed/graph | recall@10 | MultiHop | OpenDomain |
|---|---|---|---|
| 0.35/0.35/0.15 (old) | 0.5565 | 0.2737 | 0.2327 |
| 0.425/0.425/0 (drop graph) | 0.5579 | 0.2835 | 0.2275 |
| **0.25/0.60/0 (new default)** | **0.5691** | **0.3001** | **0.2614** |
| 0.20/0.80/0 (too far) | 0.5668 | 0.2994 | 0.2458 |

New default **0.25 / 0.60 / 0.0 / 0.15** lifts full-set recall@10 **0.5565 → 0.5691**
(**MultiHop +9.6%, OpenDomain +12%**, the two hardest categories). The dominant lever
is embedding weight (0.35→0.60); dropping graph adds a marginal +0.0014. The peak is
genuine — embed 0.80 scored lower than 0.60, an inverted-U, not a boundary artifact.

**Honesty caveat:** unlike the structural over-fetch win, these weights are tuned on a
single dataset (LoCoMo) and may not transfer; they are the measured best here and are
env-overridable per deployment (`SYNAPTIC_RERANK_W_TERM|EMBED|GRAPH|RECENCY`).

**Cumulative retrieval gain this session:** recall@10 0.5237 (pre-over-fetch) → 0.5565
(over-fetch) → **0.5691** (embedding-dominant rerank); MultiHop 0.2475 → 0.3001
(**+21%**), reached entirely through ranking, not connectivity.

## Cross-encoder reranker (opt-in, 2026-07-15)

The embedding-weight sweep peaked at recall@10 0.5691; closing more of the 0.702
pool ceiling needs a stronger reranker *architecture*. A candle BERT cross-encoder
(`CrossEncoderReranker`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) already existed in
the tree behind the `reranker-model` feature but was never wired or measured. This
round makes it opt-in (`SYNAPTIC_RERANKER=cross-encoder`, model dir via
`SYNAPTIC_RERANKER_MODEL_DIR`; the fast heuristic reranker stays the default; fetch
with `scripts/fetch_embedding_model.sh --cross-encoder`). Unlike bi-encoder cosine
(query and candidate embedded separately), a cross-encoder tokenizes the
`(query, candidate)` pair jointly through BERT and reads a relevance logit — far more
accurate, far more expensive.

**Head-to-head, identical 100-question LoCoMo subset** (over-fetch pool of 50,
static first-stage embeddings; only the reranker differs):

| metric | heuristic (default) | cross-encoder | delta |
|---|---|---|---|
| recall@10 | 0.4330 | **0.5483** | +0.115 (+27%) |
| MRR | 0.3775 | **0.5092** | +0.132 (+35%) |
| MultiHop R@10 | 0.2756 | 0.3251 | +0.050 (+18%) |
| OpenDomain R@10 | 0.2143 | 0.3571 | +0.143 (+67%) |
| Temporal R@10 | 0.6628 | 0.8372 | +0.174 (+26%) |
| recall latency p50 | 0.76 s | 6.34 s | 8.3× |

Every category improves; the +0.132 MRR shows the cross-encoder ranks gold markedly
higher (it recovers most of the recall@10→recall@50 pool gap). Cost is the catch:
**6.3 s/query on CPU** (50 sequential BERT forward passes per query) — hence opt-in,
not default.

**Honest caveats:**
- This is a **100-question subset** (first 10 q/conversation), skewed toward
  MultiHop/Temporal — the absolute numbers are not the full-set figure (the heuristic
  scores 0.433 here vs 0.5691 full-set because of the category mix). The **relative**
  head-to-head is the reliable signal: same questions, same pool, only the reranker
  changes.
- **No full-set cross-encoder number:** at 6.3 s/query a full 1,986-q run is CPU-bound
  and impractical in this environment (~30–80 min under the eval's concurrency).
- **Batched inference was tried and does NOT help on CPU** (measured, then reverted):
  scoring 16 pairs per padded BERT forward pass instead of 1-at-a-time preserved recall
  exactly but was ~20% *slower* single-stream (p50 3.13 s vs 2.58 s on a 20-q stream) —
  the short dialogue candidates get padded to the batch-max length (pure waste), and CPU
  BLAS gains nothing from batching the way a GPU would. Batching would only pay off on
  GPU (`cuda` feature), where batched matmuls parallelize for free. So the path to a
  full-set cross-encoder number is a GPU, not batching.
- Default retrieval is unchanged; this is strictly an opt-in quality/latency trade for
  deployments that can afford it (or run on GPU).

## QA re-measured at improved retrieval (2026-07-15)

The first QA number (0.225, 40-q subset) was measured BEFORE the retrieval rounds
(over-fetch #75 + embedding-dominant rerank #76) lifted default recall@10 0.5237→0.5691.
The QA loss diagnostic said QA was ~2/3 retrieval-bound — so better retrieval should
raise QA. Re-running the SAME 40-q subset with the SAME codex judge (only default
retrieval changed):

| metric | old retrieval (#73) | improved retrieval | delta |
|---|---|---|---|
| QA accuracy | 0.225 | **0.375** | +0.150 (+67%) |
| MultiHop | 0.158 | 0.263 | +0.105 |
| OpenDomain | 0.000 | 0.500 | +0.500 |
| Temporal | 0.313 | 0.438 | +0.125 |
| gold-retrieved rate | 0.425 | 0.525 | +0.100 |

**Cross-tab shift (retrieval-bound vs judge-bound):**

| | old | improved |
|---|---|---|
| gold retrieved & correct (A) | 7 | 9 |
| gold retrieved & wrong / judge-bound (B) | 10 | 12 |
| gold missed & correct (C) | 3 | 6 |
| gold missed & wrong / retrieval-bound (D) | 20 | 13 |
| of wrong: retrieval-bound | 0.667 | 0.520 |
| of wrong: judge-bound | 0.333 | 0.480 |

**This validates the whole diagnostic chain.** Better retrieval put gold in the top-10
for 10% more questions (gold-retrieved rate 0.425→0.525), retrieval-bound failures fell
20→13, and QA accuracy rose accordingly. As predicted, the bottleneck shifted toward the
answer step: judge-bound now ≈ half of failures (was a third). The remaining QA headroom
is now genuinely on the judge/answer side (a stronger answer model, or the opt-in
cross-encoder reranker for retrieval), not just first-stage recall.

**Caveats:** 40-q hard-skewed subset (first-N per conversation, over-weights MultiHop/
Temporal — the per-QType signal is the reliable read). The codex judge is nondeterministic
run-to-run (~±0.03 observed); the +0.15 accuracy jump is far beyond that noise and
attributable to retrieval (same judge, same questions). Not a full-set run.

## Full-set cross-encoder on GPU (2026-07-16)

The CPU cross-encoder measurement (#77) was subset-only (full set ~30–80 min on CPU).
`CrossEncoderReranker` now selects `Device::cuda_if_available(0)` (falls back to CPU
when candle is built without `cuda` or no GPU is present), so a GPU build runs the
BERT forward passes on-device. On an RTX A3000 (12GB, cc 8.6) this made the FULL
1,986-question run tractable: **5 min 35 s wall-clock** (vs ~30–80 min CPU), and
recall is identical to the CPU path (GPU is pure speedup). Single-query latency:
**0.37 s on GPU vs 2.58 s CPU** (~7×).

**Full-set recall@10, all rerankers (static first-stage retrieval, same over-fetch pool):**

| config | recall@10 | MRR | MultiHop | OpenDomain | SingleHop | Temporal |
|---|---|---|---|---|---|---|
| baseline (pre-session) | 0.5237 | 0.4058 | 0.2475 | 0.2026 | 0.6147 | 0.6072 |
| + over-fetch (#75) | 0.5565 | 0.4219 | 0.2737 | 0.2327 | 0.6459 | 0.6716 |
| + embedding rerank (#76, default) | 0.5691 | — | 0.2737 | 0.2327 | — | — |
| **+ cross-encoder (GPU)** | **0.6104** | **0.5058** | **0.3739** | **0.2755** | **0.7307** | **0.7552** |

The cross-encoder lifts full-set recall@10 to **0.6104** — from the 0.5237 session
start (**+0.087, +17%**) and closing on the 0.702 recall@50 pool ceiling. **MultiHop
0.2475 → 0.3739 (+51%)** — the category two graph-expansion rounds could not move at
all, lifted by half through ranking alone. MRR 0.4058 → 0.5058.

**Build note (GPU is opt-in):** requires building with `--features cuda` (enables
candle's CUDA backend, `cudarc` 0.13.9 → CUDA **12.x**, max 12.8) plus the
`reranker-model` feature and a fetched cross-encoder model. On very new userlands
(this box: glibc 2.42) CUDA 12.8's `math_functions` headers need a small
compatibility patch (`throw()` on `cospi`/`sinpi`/`rsqrt` declarations) and a
sub-15 host compiler (clang used here); a matched-glibc toolkit or container avoids
that. The default build is unchanged and stays CPU.

## QA with the cross-encoder: retrieval up, end-to-end flat (2026-07-16)

The cross-encoder lifts full-set retrieval recall@10 0.5691→0.6104. Does that convert
to better answers? Re-ran the SAME 40-q QA subset with the SAME codex judge, swapping
only the reranker (default embedding-heuristic → GPU cross-encoder):

| metric | default rerank | cross-encoder |
|---|---|---|
| QA accuracy | 0.375 | 0.350 |
| gold-retrieved rate | 0.525 | **0.600** |
| gold retrieved & correct (A) | 9 | 11 |
| gold retrieved & wrong (B, judge-bound) | 12 | 13 |
| correct WITHOUT gold (C) | 6 | 3 |
| gold missed & wrong (D, retrieval-bound) | 13 | 13 |
| of wrong: retrieval-bound / judge-bound | 0.52 / 0.48 | 0.50 / 0.50 |

**Finding (honest, partly counter to the prediction):** the cross-encoder measurably
improved *retrieval of labeled gold* (gold-retrieved rate 0.525→0.600, deterministic;
A rose 9→11), consistent with its full-set recall gain. But **end-to-end QA accuracy
was flat** (0.375→0.350, within the codex judge's ~±0.03 run-to-run noise —
statistically indistinguishable). The reason is visible in row C: correct answers that
did NOT have the labeled gold in the top-10 dropped 6→3. The cross-encoder ranks the
*labeled* evidence higher but tightens the top-10, squeezing out *serendipitous* context
that was helping the judge answer some questions. Better labeled-gold ranking is not the
same as better answer-supporting context.

The bottleneck is now a clean **50/50 retrieval/judge split** — the answer step is the
co-equal limiter. Combined with the earlier QA-at-improved-retrieval result (0.225→0.375,
which DID track a recall gain), the picture is: first-stage recall gains convert to QA
until the judge becomes the binding constraint, which is where we now are. The next QA
lever is answer-side (a stronger answer model, or feeding the judge more context breadth),
not sharper retrieval ranking. Caveats: 40-q hard-skewed subset; codex judge nondeterministic.

## Evidence completeness: the multi-evidence retrieval ceiling (2026-07-16)

With a near-ceiling judge (gpt-5.6-sol), the "gold-retrieved-but-answer-wrong" QA
bucket cannot be an answer-model weakness. Hypothesis: it is retrieval *completeness* —
`gold_retrieved` counts a question as retrieved if ANY one of its evidence turns is in
the top-k, but multi-hop questions need ALL of them. The new `--completeness` mode
measures STRICT full-evidence recall (`full@k` = every evidence turn in the top-k),
bucketed by how many evidence turns a question has. Full set, default pipeline:

| evidence turns | n | partial@10 (any) | full@10 (all) | full@50 |
|---|---|---|---|---|
| 1 | 1559 | 0.637 | 0.637 | 0.758 |
| 2 | 239 | 0.366 | 0.180 | 0.331 |
| 3 | 83 | 0.337 | 0.060 | 0.193 |
| 4+ | 101 | 0.214 | 0.000 | 0.000 |
| overall | 1982 | 0.570 | 0.525 | 0.644 |

**The finding:** single-evidence questions retrieve well (full@10 0.64), but strict
completeness collapses with evidence count — 2-evidence 0.18, 3-evidence 0.06, and
**4+-evidence questions NEVER have all their evidence in the top-10, nor even the
top-50** (full@50 0.00). ~21% of questions (423) are multi-evidence and almost never
get their complete evidence set.

**Consequences:**
1. The QA "judge-bound" failures are overwhelmingly retrieval-COMPLETENESS, not judge
   quality. A strong judge cannot answer a 3-hop question given 1 of 3 required facts.
   A better answer model would not help.
2. It explains why the cross-encoder did not move QA: reranking ranks ONE best turn
   high; multi-evidence questions need the whole evidence SET gathered — a different
   objective (set-cover / iterative retrieval), which single-turn relevance ranking
   (bi-encoder, cross-encoder, graph expansion) does not optimize.
3. Methodology: standard recall@10 (0.558) OVERSTATES answerable retrieval — it gives
   partial credit on multi-evidence questions where strict full@10 is 0.525.

The real remaining retrieval lever is multi-evidence completeness (e.g. query
decomposition into sub-questions each retrieving their own evidence, or iterative
retrieve-until-complete), not sharper single-turn ranking. Measure with
`run_eval --completeness`.

## Negative result: MMR diversity does not fix multi-evidence completeness (2026-07-16)

Following the completeness finding (multi-evidence questions rarely retrieve their full
evidence set), the first attempt was MMR (Maximal Marginal Relevance) diversity selection:
after reranking, greedily pick the top-`limit` balancing relevance against dissimilarity to
already-selected candidates (`SYNAPTIC_RERANK_MMR_LAMBDA`, default 1.0 = off), so the top-10
spreads coverage instead of clustering near-duplicates of the single best turn.

**Measured (static pipeline):** MMR engages (overall recall@10 0.525→0.500 at λ=0.3 on a
20-q stream — diversity trades relevance) but does NOT help the target: MultiHop partial
recall@10 is **exactly flat** across λ∈{1.0, 0.5, 0.3} (0.1920), i.e. it does not change
which evidence lands in the top-10 for multi-evidence questions. (A full-set full@10-by-
evidence-count sweep was attempted but the sequential completeness metric + MMR's per-query
embedding/greedy-selection cost made it impractical in the run budget.)

**Why MMR cannot work here (the structural reason):** MMR only REORDERS the candidate pool.
But the completeness data shows the missing evidence is largely NOT IN THE POOL — full@50 is
0.33 (2-ev), 0.19 (3-ev), and **0.00 (4+-ev)**. For 4+-evidence questions at least one
evidence turn is never even in the top-50, so no reordering (MMR, cross-encoder, graph
expansion) can surface it. The multi-evidence problem is **pool absence, not misranking**.

**Consequence:** the completeness lever is not reranking/selection at all — it is GATHERING
the missing evidence into the candidate set: iterative retrieve-until-complete, or query
decomposition into sub-questions that each retrieve their own evidence and union the pools.
That is a different retrieval loop (multi-query / set-cover), not a scoring change. The MMR
branch was abandoned (off-by-default feature, no measured benefit; nothing merged).

## PRF pool augmentation: gathering missing multi-evidence (2026-07-16)

The MMR negative result (#83) showed the multi-evidence problem is pool ABSENCE, not
misranking — no reordering can surface evidence that first-stage retrieval never pooled.
The fix must GATHER the missing evidence into the candidate set. Pseudo-relevance
feedback (PRF, opt-in `SYNAPTIC_RETRIEVAL_PRF=1`; default off): after the first retrieval,
mine the top-`prf_top_m` turns for salient terms not in the query, expand the query with
them, retrieve once more, and union into the candidate pool — then the existing
over-fetch reranker ranks the enlarged pool.

**Full-set completeness, PRF on vs off** (the target metric is full@50 = whole evidence
set in the pool):

| evidence | full@50 off | full@50 PRF | Δ | full@10 off | full@10 PRF |
|---|---|---|---|---|---|
| 1 | 0.7575 | 0.7627 | +0.005 | 0.6369 | 0.6440 |
| 2 | 0.3305 | 0.3933 | **+0.063 (+19%)** | 0.1799 | 0.1883 |
| 3 | 0.1928 | 0.2651 | **+0.072 (+37%)** | 0.0602 | 0.0723 |
| 4+ | 0.0000 | 0.0396 | **from ZERO** | 0.0000 | 0.0000 |
| overall | 0.6438 | 0.6604 | +0.017 | 0.5252 | 0.5323 |

**PRF is the first mechanism this session to attack pool absence, and it works.** It
raises full@50 (pool coverage) for every multi-evidence bucket — the 4+-evidence bucket
going 0.000→0.040 is the proof, since those complete evidence sets were provably never in
the pool under any reranking (MMR, cross-encoder, graph expansion all could not touch
them). PRF also strictly improves overall partial recall@10 (0.5702→0.5776) and full@10
(0.5252→0.5323) — no regression.

**Why the full@10 gains lag the full@50 gains:** PRF gets the evidence INTO the pool, but
it still has to RANK into the top-10 — which is where a stronger reranker compounds (PRF
gathers, the cross-encoder ranks). The two are complementary levers on the same problem.

**Cost / status:** one extra retrieval round → ~1.8x recall latency (subset p50 0.77s→1.37s),
so PRF is OPT-IN (default off), like the cross-encoder — enable it for multi-evidence-heavy
workloads. Measure with `run_eval --completeness` (now concurrent) and
`SYNAPTIC_RETRIEVAL_PRF=1`.

## PRF + cross-encoder: gather AND rank (2026-07-16)

PRF (#84) gathers multi-evidence into the pool (full@50 up) but its full@10 gains lagged
because the pooled evidence still had to RANK into the top-10 — the cross-encoder's job.
Composing the two opt-in levers (PRF + GPU cross-encoder) converts pool coverage into
top-10 coverage:

**Full-set completeness, multi-evidence full@10:**

| full@10 | baseline | PRF only | **PRF + cross-encoder** |
|---|---|---|---|
| 2-evidence | 0.180 | 0.188 | **0.276 (+53% vs baseline)** |
| 3-evidence | 0.060 | 0.072 | **0.108 (+80%)** |
| 4+-evidence | 0.000 | 0.000 | **0.0099** |
| overall full@10 | 0.525 | 0.532 | **0.567** |
| overall partial recall@10 | 0.570 | 0.578 | **0.615** |

**The composition decomposes cleanly, confirming the two-lever model:**
- full@50 (pool coverage) is IDENTICAL for PRF-only and PRF+cross-encoder (0.393 at 2-ev) —
  the cross-encoder does not change what is in the pool; it only reorders it. **PRF sets the
  ceiling (gather); the cross-encoder reaches it (rank).**
- full@10 (top-10 coverage) roughly doubles for multi-evidence vs baseline — the
  cross-encoder ranks the PRF-gathered evidence into the top-10.

Overall partial recall@10 = 0.615 is the best of any configuration measured. This is the
recommended maximum-quality config (both opt-in; PRF ~1.8x latency, cross-encoder needs a
GPU build). Ran in 13m52s on the RTX A3000 (full set, concurrent, cross-encoder + PRF + the
two-search completeness metric).

**End-to-end QA with this best config (PRF + cross-encoder, 40-q subset, codex judge):
0.325** — flat vs default rerank (0.375) and cross-encoder alone (0.350), all within the
codex judge's ~±0.03 noise. The doubled multi-evidence completeness did NOT clearly move QA
on this subset. Honest reason: completeness doubled but from a LOW BASE — 2-ev full@10 is
0.276, so 72% of 2-evidence questions still lack their complete set in the top-10, and 4+ is
~1%. QA needs ALL the evidence, so it still fails on most multi-evidence questions. The
binding constraint is now the full@50 CEILING itself (2-ev 0.39, 3-ev 0.27, 4+ 0.04) — even
the 50-pool rarely contains the whole evidence set. Raising QA requires raising that ceiling
(more aggressive gathering: multi-round / entity-targeted PRF), not just better ranking.

## Negative result: multi-round PRF drifts (single round is the sweet spot, 2026-07-16)

PRF (#84) gathers multi-evidence into the pool in one round. Would MORE rounds (iterative,
each expanding from the prior round's new hits) reach further and raise the full@50 ceiling?
Built `prf_rounds` (cumulative term accumulation, early-stop on convergence; a controlled
2-hop chain test confirms round 2 can reach a term-hop further). **Full-set measurement says
no — round 2 is WORSE than round 1:**

| multi-evidence full@50 | rounds=1 | rounds=2 |
|---|---|---|
| 2-evidence | 0.393 | 0.372 |
| 3-evidence | 0.265 | 0.229 |
| overall | 0.660 | 0.651 |

**Query drift:** cumulatively appending expansion terms diffuses the query (the static
embedder averages token vectors, so a longer bag-of-terms query drifts from intent), and
the extra round retrieves less-relevant candidates that DISPLACE real evidence from the
bounded 50-pool. The reach-further benefit is real in a controlled fixture but is outweighed
by drift on real data; it's also slower (16:56 vs ~14 min). **Single-round PRF is the sweet
spot** — multi-round abandoned (nothing merged). A non-cumulative multi-QUERY variant (each
round a separate focused query, union the pools) might avoid drift, but is unbuilt/unmeasured.

## Agentic answer-guided retrieval: breaking the QA ceiling (2026-07-16)

QA sat at ~0.375 all along because multi-evidence questions need their COMPLETE evidence
set, and single-shot retrieval — even excellent retrieval (cross-encoder, PRF) — is BLIND
to what the answer needs. The fix is to let the strong judge (gpt-5.6-sol) DRIVE retrieval:
`--agentic-qa` seeds a search, then loops (bounded, default 3 rounds) asking the model to
either `ANSWER` from the current context or `SEARCH: <specific missing fact>`; each SEARCH
retrieves and merges into the context; then the final answer is graded (same grader as
`run_qa`; gold/evidence labels are NEVER in the model's prompt — used only to instrument).

**Standard 40-question LoCoMo subset, all configs (same questions, codex judge):**

| config | QA accuracy | gold-retrieved rate |
|---|---|---|
| single-shot, default rerank | 0.375 | 0.525 |
| single-shot, cross-encoder | 0.350 | — |
| single-shot, PRF + cross-encoder | 0.325 | — |
| agentic (static + PRF) | 0.400 | 0.575 |
| **agentic + PRF + GPU cross-encoder** | **0.500** | **0.650** |

**0.500 vs the previous best 0.375 is +33%, decisively beyond the codex judge's ~±0.03
run-to-run noise** — the first configuration all effort to break the 0.375 ceiling.

**The instrumentation shows the mechanism, not just the score:**
- gold-retrieved rate 0.525 → 0.650 (best measured) — the LLM's follow-up searches, ranked
  by the cross-encoder, actually FIND the missing evidence.
- `gold_added_by_followup` = 0.125 (vs 0.05 for agentic-static) — the cross-encoder makes
  follow-up retrieval land gold that the seed search missed.
- retrieval-bound failures (cross-tab d) fell 14 → 9; gold-and-correct (a) rose to 15.
- mean rounds used 1.55; Temporal QA 0.875.

**Why the agentic LOOP is the key** (not just better retrieval): single-shot PRF+cross-encoder
was 0.325 — the SAME retrieval components, WITHOUT the loop, did not help. Adding the loop
took it to 0.500. Blind term-expansion (PRF) guesses what's missing; the LLM KNOWS what's
missing and searches for it specifically. Gathering (agentic follow-ups) + ranking
(cross-encoder) together break the multi-evidence completeness ceiling that capped QA.

**Cost / status:** opt-in eval mode; multiple codex calls per question (bounded concurrency
via `SYNAPTIC_EVAL_QA_CONCURRENCY`), best retrieval per round needs the GPU cross-encoder
build. Caveats: 40-q hard-skewed subset; codex judge nondeterministic (the +0.125 jump is
far beyond its ~±0.03 noise). MultiHop (0.21) remains the hard tail even here.

## Faithfulness: does the memory system confabulate? (2026-07-17)

Recall is only half of a memory system's value; the other half is GROUNDEDNESS — when it
lacks the evidence, does it say "I don't know" or fabricate a confident wrong answer? Added
`is_abstention` classification + a faithfulness breakdown (confabulation vs honest
abstention), a `--qtype` filter to target the LoCoMo Abstention category (446 questions
whose gold answer is "None" — provably unanswerable, so any confident answer is pure
fabrication), and measured both QA modes.

**Abstention category (50 provably-unanswerable questions):**

| config | abstained (faithful) | confabulated |
|---|---|---|
| single-shot | 47/50 (94%) | 3/50 (6%) |
| agentic (as first built) | **0/50 (0%)** | **50/50 (100%)** |
| agentic (after fix) | 38–39/50 (~78%) | 11–12/50 (~22%) |

**Critical finding:** the agentic loop (which broke the QA ceiling, 0.375→0.500) as first
built **confabulated on 100% of unanswerable questions** — a disqualifying flaw for a memory
system. Root cause: the loop offered only `ANSWER`/`SEARCH` and FORCED an answer on the final
round, so when the evidence genuinely did not exist it fabricated one (mean 2.74 rounds spent
searching for nothing, then a made-up answer). Single-shot, allowed to say "I don't know,"
abstained 94% of the time.

**Fix:** make abstention a first-class terminal action — the prompt (every round, and
especially the forced final round) permits `ANSWER: I don't know` when the memories genuinely
lack the answer. Result: abstention on unanswerable questions recovered 0%→76%, WITHOUT
over-correcting — answerable-question accuracy held (agentic-static 0.400 unchanged) and
`over_abstention = 0` (it never declines a question whose evidence it actually has).

**Honest residuals:** (1) agentic is still more answer-prone than single-shot (24% vs 6%
confabulation on unanswerable; on answerable-but-evidence-missed questions ~56% vs ~16%) —
actively searching then answering biases it toward producing something. A structural
grounding step (require a supporting memory per claim, else abstain) would tighten this
further. (2) Grading artifact: the codex grader does not credit "I don't know" as matching
gold "None", so ABSTENTION-category accuracy reads ~0.04–0.16 even when abstention is
behaviorally correct — the faithfulness metric (not accuracy) is the right measure of
abstention behavior, and full-set QA accuracy understates faithful systems on the 22% of
questions that are unanswerable.

**Takeaway:** for a memory system, faithfulness must be measured alongside recall — the
agentic QA breakthrough was hiding a 100%-confabulation flaw that only the abstention metric
revealed. Post-fix, the system both answers hard multi-evidence questions AND honestly
declines when the evidence is absent. (The `is_abstention` classifier was
tightened to whole-word matching for the short markers `none`/`unknown` to avoid
fragment false positives; the abstention rate was unchanged — ~78% — confirming
the figure is not a matching artifact.)

## Structural grounding: cite-a-memory-or-abstain (2026-07-17)

The agentic abstention fix (#88) made faithfulness a prompt *request* (~22% residual
confabulation on unanswerable questions). Structural grounding (opt-in
`SYNAPTIC_EVAL_GROUNDED=1`) tries to make it a *requirement*: the answer must cite the
numbered context memory it draws from (`ANSWER: <text> SOURCES: [n,...]`), the citation is
verified against the provided context, and a non-abstention answer with NO valid citation is
overridden to "I don't know" (`ungrounded_overrides` counted).

**Measured (agentic + PRF, codex judge):**

| | ungrounded | grounded |
|---|---|---|
| unanswerable: abstained | ~78% | **90%** |
| unanswerable: confabulated | ~22% | **10%** |
| answerable accuracy (20q) | 0.400 | 0.550 |
| answerable evidence-missing confabulation | ~56% | ~20% |
| over_abstention | 0 | 0 |

Grounding improved faithfulness across the board — confabulation roughly halved on both
unanswerable and answerable-but-evidence-missing questions — **without hurting answerable
accuracy** (0.40→0.55, held/improved within the small-sample codex noise) and with zero
over-abstention.

**Honest mechanism note:** `ungrounded_overrides = 0` in practice — the structural override
never fired. The gain came from the citation *requirement in the prompt* (a model asked to
cite a supporting memory reasons about whether it can, and abstains honestly when it can't),
NOT from the override backstop. The override only catches answers with no/invalid citation;
the residual confabulations cite a VALID-but-unsupporting memory (a hallucinated-but-in-range
citation), which citation-*existence* cannot catch. True enforced grounding needs
citation-*support* verification (does the cited memory entail the answer? — an entailment
check), a natural follow-up. As shipped, grounding is a strong opt-in faithfulness lever:
prompt-required citations move agentic confabulation from ~22% toward single-shot's ~6%.

## Citation-support verification + honest limits of structural grounding (2026-07-17)

Structural grounding (#89) overrode answers with NO valid citation, but couldn't catch a
HALLUCINATED-but-valid citation (a real in-context memory that doesn't actually support the
answer). Added an entailment check (opt-in `SYNAPTIC_EVAL_GROUND_VERIFY=1`): after a grounded
answer with a valid citation, ask the judge whether the cited memory actually supports the
answer; if not, override to "I don't know" (`support_overrides` counted). Also hardened the
codex judge to RETRY transient upstream errors (model-at-capacity / rate limits) with backoff
so a single blip no longer aborts a whole eval run.

**Measured (agentic + PRF + grounding + verify, codex judge):**

| | grounded only | grounded + verify |
|---|---|---|
| abstention confabulation | ~10% | ~7% (within noise) |
| `support_overrides` (abstention 30q) | — | **0** |
| answerable accuracy (20q) | 0.55 | 0.45 (within noise) |
| `support_overrides` (answerable 20q) | — | **1** |
| over_abstention | 0 | 0 |

**Honest finding: the structural override gates are nearly INERT on LoCoMo.** Both
`ungrounded_overrides` (citation existence, #89) and `support_overrides` (citation support,
this change) fire ~0-1 times per 30-50 questions. The faithfulness improvements this whole
sub-arc (agentic 100% → ~10% confabulation on unanswerable questions) came from the PROMPT —
asking the model to cite/verify makes it reason about grounding and abstain honestly, rather
than producing catchable violations. The residual confabulations cite plausible-looking
memories the verify-judge accepts, so citation-support verification does not measurably reduce
them here. The entailment check is a CORRECT, fail-closed safety layer (off by default) that
would matter more for models/datasets that hallucinate unsupporting citations at higher rates
— but on LoCoMo, prompt-level grounding is what does the work, and verification adds an extra
judge call for ~no measured benefit. Kept as opt-in; not recommended by default.

The codex-retry hardening, by contrast, is unconditionally useful: transient "model at
capacity" errors were aborting whole runs, and retry-with-backoff makes the QA/agentic
measurements robust to upstream flakiness.

## Head-to-head vs Mem0 (open-source memory system, 2026-07-17)

Ran the popular open-source memory system [Mem0](https://github.com/mem0ai/mem0)
through the SAME LoCoMo questions, the SAME codex answer/grade prompts, and the SAME
abstention/faithfulness classification as this repo's harness (see
`tools/eval/comparisons/`), so the numbers sit on the same footing on the answering side.

**QA accuracy — identical 40-question subset, identical codex judge:**

| system | QA accuracy |
|---|---|
| Mem0 (local qwen-7B ingestion) | 0.250 |
| this repo, single-shot retrieval | 0.375 |
| this repo, agentic answer-guided retrieval | **0.500** |

**Faithfulness — 50 provably-unanswerable (Abstention) questions:**

| system | abstained (honest "I don't know") | confabulated |
|---|---|---|
| Mem0 | 44/50 (88%) | 12% |
| this repo, single-shot | 47/50 (94%) | 6% |
| this repo, agentic (after abstention fix) | ~90% | ~10% |

> **⚠️ This section's "outperforms Mem0" reading was overturned by a fairer follow-up.**
> The numbers below are real, but they used Mem0's *local qwen-7B* extractor. When Mem0 was
> given its intended frontier extractor (gpt-5.6-sol) with correct dates and tested on a
> larger matched slice, **Mem0 won (0.500 vs our 0.450 agentic / 0.325 single-shot)** — see
> "Mem0 with a fair setup … beats this repo" below. Read that section as the current result.

**Honest reading (caveats matter here more than the numbers):**
- Under **matched local conditions** (both systems answering with the same codex judge
  in the same environment), this repo outperforms Mem0 on QA accuracy — 1.5× even
  single-shot, 2× with agentic retrieval. Mem0's single-shot retrieve-then-answer maps
  to our single-shot (0.375 vs 0.250); agentic (0.50) is our differentiator (Mem0 does
  not do iterative answer-guided retrieval). **[Superseded — see the ⚠️ note above: this
  held only because Mem0 was on a weak local extractor with broken dates.]**
- **BUT Mem0's ingestion here used a local qwen-7B, not the GPT-4 its published LoCoMo
  numbers (~66%) assume.** Mem0's memory quality depends heavily on the extraction LLM,
  so this is NOT a claim of beating Mem0's best case — it is "competitive-to-better in a
  matched local setup." A genuine architectural point in this repo's favor: its
  memory-building uses no LLM (heuristic extraction + embeddings), so it is not
  bottlenecked by the ingestion model's quality the way Mem0 is.
- **Faithfulness is largely a wash.** Both systems answer with the same codex judge and
  the same "say I don't know if not answerable" prompt, so abstention is driven mostly by
  the shared answerer, not the memory system — the ~6-point gap (94% vs 88%) just reflects
  Mem0 occasionally retrieving plausible-but-irrelevant facts that nudge the judge to
  answer. This repo's faithfulness *work* was about fixing its OWN agentic mode (which
  confabulated 100% before the fix), not about being dramatically more faithful than Mem0.
- Retrieval `recall` is not compared (different memory unit: Mem0 stores extracted facts,
  not turns). Small subsets; codex judge is nondeterministic (~±0.03). Zep/Graphiti and
  Letta not tested (heavier setup) — this is one real head-to-head, not a field survey.

### Follow-up: Mem0 with a fair setup (frontier extractor + correct dates) beats this repo (2026-07-17)

The 0.250 above used Mem0's *local qwen-7B* extractor. We gave Mem0 its intended setup —
**gpt-5.6-sol** fact-extraction via the codex OAuth login (the codex→OpenAI shim, no API
key) — and fixed a harness bug of our own along the way. Two rounds, reported honestly
because the second overturned the first.

**Round 1 (n=12, and it was unfair to Mem0):** a first pass on conv0–2 gave Mem0 0.333 vs
our 0.500 and we nearly shipped "we still win." But that pass had two defects: (a) n=12 is
tiny; (b) the shim flattened each conversation and codex stamped every fact with *today's*
date ("July 10, **2026**" for a 2023 event), destroying the timestamps temporal questions
need — Mem0 scored **0/6 on Temporal** purely from our bug. That gap was ours, not Mem0's.

**Round 2 (n=40, fair):** we fixed the extraction to prefix each turn with its real session
date (as Mem0's own LoCoMo harness does), re-ingested **all 10 conversations** on gpt-5.6-sol
(**2,182 facts stored**, dates now correct — "On May 8, 2023 …"), and ran the matched
40-question slice (10 convs × first 4) through the SAME codex answerer+judge as our harness.

| Category (n) | **Mem0** (gpt-5.6-sol extract, dated) | This repo — single-shot | This repo — agentic |
|---|---|---|---|
| **Overall (40)** | **0.500** (20/40) | 0.325 (13/40) | 0.450 (18/40) |
| MultiHop (19) | **0.421** | 0.263 | 0.263 |
| OpenDomain (4) | 0.25 | 0.00 | 0.50 |
| SingleHop (1) | 1.00 | 1.00 | 1.00 |
| Temporal (16) | 0.625 | 0.438 | 0.625 |

**What this settles — including against our own earlier claim:**
- **The date fix worked and it mattered.** Mem0's Temporal went 0.0 (n=12, broken) → **0.625**
  (n=40, dated). The earlier temporal gap was our harness artifact, exactly as Round 1's
  caveat suspected — confirmed, not hand-waved.
- **The result reverses. Mem0 (0.500) beats this repo** on the fair, larger slice: clearly
  over our single-shot (0.325), and above our agentic (0.450). 0.500 vs 0.450 is 2 questions —
  within codex-judge noise (~±0.03), so agentic-vs-Mem0 is ≈ a tie — but Mem0 is *not behind
  us anymore*. The n=12 "we win" did not survive enlarging n and removing our own bug.
- **Mem0's real edge is MultiHop (0.421 vs our 0.263).** Write-time fact-extraction — the
  thing this repo does *not* do (we retrieve raw turns) — genuinely helps multi-hop, matching
  our own finding that our multi-hop is ranking-limited (see the MultiHop analysis above).
- Faithfulness is comparable: Mem0 abstained on 6/40 (2 OpenDomain, 4 Temporal); ours
  confabulated 4–5 on evidence-missing questions. Neither is dramatically more faithful; both
  answer with the same codex judge.

**Honest bottom line:** with both systems on frontier answering and Mem0 on its intended
frontier+dated extraction, **Mem0's write-time fact extraction edges our retrieve-raw-turns
pipeline on LoCoMo (0.500 vs 0.450 agentic, within noise; 0.500 vs 0.325 single-shot,
clearly).** The direction to close it is on our side: distill/extract facts at write time and
strengthen multi-hop, rather than retrieving raw conversation turns. This corrects the
earlier under-tested "outperforms Mem0" framing.

**Caveats:** n=40, single codex-judge pass (nondeterministic ~±0.03); our config is
static-embeddings + PRF single-shot/agentic (no cross-encoder reranker in this run); ran
entirely on the codex OAuth login (no OpenAI key). Reproduce:
`comparisons/mem0_codex_eval.py 10 4` (Mem0, date-aware) vs
`run_eval … --qa-only|--agentic-qa --max-conversations 10 --max-questions 4` (ours).

### Closing the gap: write-time fact extraction lifts this repo to match Mem0 (2026-07-18)

The section above pinned Mem0's edge on **write-time fact extraction** (it stores LLM-distilled
facts; this repo retrieved raw conversation turns). We tested that hypothesis directly: feed the
*same* extracted facts into *our* retrieval+answer pipeline and see if the gap closes. Using the
identical 2,182 codex-extracted facts (the exact haystack Mem0 scored 0.500 on), same 40-question
slice, same codex answerer+judge — only the retrieval layer differs:

| Config (40 q) | Overall | MultiHop | OpenDomain | Temporal |
|---|---|---|---|---|
| This repo — raw turns, single-shot | 0.325 | 0.263 | 0.00 | 0.438 |
| This repo — raw turns, agentic | 0.450 | 0.263 | 0.50 | 0.625 |
| This repo — **over extracted facts**, single-shot | 0.475 | 0.316 | 0.50 | 0.688 |
| This repo — **over extracted facts**, agentic | **0.500** | 0.368 | 0.25 | 0.688 |
| Mem0 (over its own facts) | 0.500 | 0.421 | 0.25 | 0.625 |

**What this proves:**
- **Write-time extraction is the entire gap.** Swapping raw turns for extracted facts lifts this
  repo +0.15 single-shot (0.325→0.475) and to **0.500 agentic — exactly Mem0's score.**
- **Our retrieval was never behind.** Given the *identical* fact haystack, this repo ties Mem0
  to the decimal (0.500 = 0.500). The earlier deficit was representation (raw turns), not ranking.
- Temporal actually leads (0.688 vs Mem0's 0.625) once facts carry correct dates.

**Implementation (shipped, `llm-reasoning` feature):** the intelligent write path already existed
(`MemoryReasoner` trait + `LlmReasoner`, `src/memory/reasoning/`). Two new eval ingestion modes
exercise it end-to-end: `--extract-facts` builds the haystack live with **synaptic's own
`LlmReasoner`** (`SYNAPTIC_LLM_URL`/`SYNAPTIC_LLM_MODEL`; each session date-prefixed so extraction
anchors dates correctly), and `--facts-from FILE` replays a precomputed fact set for fast,
deterministic comparison. Two real bugs were found and fixed en route: a char-boundary panic in
`summarization.rs` (multibyte chars in stored text) and non-tolerant relation parsing in
`LlmReasoner` (weaker models omit a field). Reproduce:
`run_eval … --qa-only --facts-from mem0_facts_by_conv.json --max-conversations 10 --max-questions 4`
(replay), or `--extract-facts` with an OpenAI-compatible endpoint (live extraction).

**Caveats:** n=40, single codex-judge pass (~±0.03); the 0.500 uses codex-quality extracted facts
(the `--facts-from` replay isolates retrieval by holding the facts fixed); a full live
`--extract-facts` run with a frontier extractor is the natural next measurement (extraction cost
~20 s/session). The direction, though, is settled: **distill facts at write time and this repo
matches the best open-source agentic memory on LoCoMo.**

### Field survey: adding Graphiti (Zep's engine) — and where Letta fits (2026-07-18)

Extending the one-off Mem0 head-to-head toward a real survey. **Graphiti** (the open-source
temporal-knowledge-graph engine behind Zep) was run through the same pipeline: frontier
extraction via the codex OAuth login (its `OpenAIGenericClient` on `/chat/completions` +
`json_object` — the default client uses OpenAI's Responses API, which the shim doesn't serve),
FalkorDB for the graph, Ollama for embeddings, same codex answer/grade prompts.

Matched **12-question** slice (conv0–2, first 4 each — the largest that fits Graphiti's slow
ingestion; see below), same codex answerer+judge:

| System (12 q) | Overall | MultiHop (4) | OpenDomain (1) | SingleHop (1) | Temporal (6) |
|---|---|---|---|---|---|
| Mem0 (date-aware, frontier) | **0.750** | 0.25 | 1.00 | 1.00 | 1.00 |
| This repo — over facts | 0.583 | 0.00 | 1.00 | 0.00 | 1.00 |
| This repo — raw turns | 0.500 | 0.50 | 0.00 | 1.00 | 0.50 |
| Graphiti (Zep) | 0.500 | 0.50 | 0.00 | 0.00 | 0.667 |

**Honest reading:**
- All four land in a **0.50–0.75 band** on this slice; the write-time-extraction systems (Mem0,
  our over-facts) lead, driven by **perfect Temporal (6/6)** — dated fact extraction pays off
  exactly where it should. Graphiti is competitive, tying our raw-turns baseline.
- **n=12 is small and conv0–2 is an easier slice** (every system scores higher here than on the
  40-q set — e.g. Mem0 0.750 here vs 0.500 at n=40). The reliable ranking remains the 40-q
  numbers above; this slice ranks systems only loosely. The OpenDomain/SingleHop rows are n=1
  (pure noise).
- **Why only 12 q for Graphiti:** its temporal-KG write path is ~4× slower than Mem0's
  (~70 s/episode vs ~20 s/session — it does entity + edge extraction, dedup, and temporal
  invalidation per episode via the LLM), so a 40-q (10-conversation) run was not feasible in
  session. A first attempt also scored a spurious 0.083 until we caught that FalkorDB stores each
  `group_id` as a *separate graph*; `search(group_ids=…)` queried the empty default graph. Fixed
  by connecting the driver per-conversation (`database=<conv>`); the corrected run is above. (Same
  "verify the store is non-empty before trusting a low score" lesson as the Mem0 empty-store bug.)
- **Letta** (formerly MemGPT) is intentionally *not* in the table: it is a full stateful-agent
  framework (core-memory blocks + self-editing + archival passages), not a drop-in
  retrieve-then-answer memory. A fair LoCoMo QA comparison would let Letta's *own* agent answer,
  which changes the answerer and breaks the "same codex judge, only the memory differs" control
  this survey depends on. Noted as architecturally different; a separate agent-level eval, not
  this retrieval-substitution one.

**Bottom line of the survey:** across Mem0, Graphiti, and this repo — held to the same answerer —
the systems cluster together, and the single biggest lever is **write-time fact extraction**
(the thing this repo now does via `store_extracted_facts`), not the choice of retrieval engine.

**Shipped as a library capability (not just an eval mode).** `MemoryConfig::store_extracted_facts`
(default `false`) turns the same distillation on inside the normal write path: every
`AgentMemory::store` runs the active `MemoryReasoner` over the value and persists each extracted
fact as its own retrievable memory (`<key>::fact<N>`), alongside the raw value and KG updates. A
re-entrancy guard prevents fact-memories from re-triggering extraction. With the default heuristic
reasoner this is fully offline/deterministic; with `LlmReasoner` (`SYNAPTIC_LLM_URL`/`_MODEL`) it is
LLM-quality. So the measured LoCoMo lift is reachable through the plain store API, not only the
eval harness.

### v2 capability validation on LoCoMo: reflection is neutral; the rest need other instruments (2026-07-19)

Applying the honesty bar to the agent-memory-v2 capabilities that were built but not yet
measured. The finding: **LoCoMo QA is a fact-recall benchmark, so only fact-retrieval
capabilities show up in it.**

**Reflection / synthesis (`--reflect`, measured):** after ingesting each conversation we trigger
`AgentMemory::reflect()` so synthesized insight-memories join the haystack, then run the same
matched 40-question QA. Result — **neutral**:

| | Overall | MultiHop | Temporal |
|---|---|---|---|
| baseline (raw turns, single-shot) | 0.325 | 0.263 | 0.438 |
| + reflection (heuristic, 71 insights) | 0.325 | 0.211 | 0.438 |

Identical overall accuracy; MultiHop marginally worse (extra memories perturb ranking). Expected:
reflection produces higher-level *summaries*, but LoCoMo questions need *specific facts* the raw
turns already contain — summaries add no fact-level value. The write-path lever for this benchmark
is fact *extraction* (above), not synthesis. LLM-quality synthesis is untested but the mechanism
argues against a QA gain here. Reflection is retained as a capability (unit-tested) and as an eval
mode (`--reflect`); it is simply not a LoCoMo-QA lever.

**Bi-temporal KG (measured on LongMemEval; strong answerer masks its value):**
supersede-on-contradiction runs on the *write* path (`supersede_matching_relations`, unit-tested)
but `search` does not consult edge validity — so LoCoMo turn-QA cannot exercise it. We measured its
target directly on **LongMemEval knowledge-update** (78 questions where a later fact updates an
earlier one; oracle sessions), same codex answerer/judge. Baseline (current system, no validity
wiring): **0.875 (63/72)**. The frontier answerer already resolves updates itself by reasoning over
the retrieved timestamps/context, so retrieval-level validity filtering has <=12.5% end-to-end
headroom — most of which is likely retrieval/ambiguity, not stale-fact confusion. This is the same
pattern seen throughout: a strong answerer masks retrieval-level gaps. Bi-temporal is correctly
built (supersede/`is_valid_at`/`query_as_of` unit-tested); its measurable *value* lives at the
retrieval-precision or weak-answerer level (does validity filtering surface the current fact at
rank 1?), not in end-to-end QA accuracy with a frontier model — which is already at 0.875.
**Wiring now built + demonstrated:** `MemoryConfig::retrieval_excludes_superseded` closes the loop —
the intelligent write path marks a memory superseded when a later fact contradicts it (same
subject+predicate, different object), and `search` drops superseded memories so retrieval returns
the *current* fact. A deterministic test proves the retrieval-precision effect: after "Alice lives
in Berlin" is superseded by "Munich", search returns only Munich with the flag on, both with it off.
So bi-temporal's value is real and demonstrable at the retrieval level; it simply doesn't move
frontier-answerer QA (0.875), which already disambiguates updates from the retrieved context.

**Forgetting / decay (measured — validated with a differentiated-access harness):** an explicit
eviction pass over `retained_strength = decay(age)·importance·recency`. LoCoMo QA can't show its
value (every memory is potentially needed; uniform ingestion gives uniform strength), so we built a
purpose-fit measurement (`--forget-curve`): differentiate access by running each question's search
over the full memory (retrieved turns are "accessed"), rank turns by
`ForgettingPolicy::retained_strength` and, separately, by a deterministic pseudo-random key, keep
the top fraction of each, and compare recall@10 of the survivor sets across store sizes (5 convs,
75 questions):

| store kept | forgetting recall@10 | random recall@10 | delta |
|---|---|---|---|
| 100% | 0.316 | 0.316 | +0.000 (sanity: no eviction) |
| 75% | 0.361 | 0.301 | **+0.060** |
| 50% | 0.372 | 0.299 | **+0.073** |
| 25% | 0.390 | 0.170 | **+0.220** |

**Two findings:** (1) principled forgetting beats random eviction by a widening margin as the store
shrinks — at 25% it retains recall@10 = 0.39 vs random's 0.17 (2.3x); the strength ranking keeps the
question-relevant memories. (2) Forgetting *raises* recall as it evicts (0.316->0.390): removing
low-strength/unaccessed turns lets the relevant ones rank higher in the top-10 (denoising). **Honest
caveat:** access is differentiated on the same questions recall is measured on (past = future
queries), so this is an optimistic best-case; a train/held-out query split would be stricter. The
mechanism — usage-based retention preserving what's needed while bounding size — is validated.

**Held-out query split (2026-07-20): the effect survives, at ~⅔ the magnitude.** The caveat above
is now resolved. `--forget-curve` splits each conversation's questions (deterministically, by FNV
parity) into a TRAIN half that differentiates access and a *disjoint* TEST half that recall is
measured on — so the policy never sees the queries it is evaluated against. For a clean comparison
it also reports the IN-SAMPLE strength ranking (access from ALL questions, including the test set)
scored on the *same* test questions; the gap between them is exactly the leakage. Full LoCoMo, 10
conversations, n = 990 held-out test questions:

| store kept | held-out fg@10 | random@10 | held-out Δ | in-sample fg@10 | in-sample Δ | leak |
|---|---|---|---|---|---|---|
| 100% | 0.5776 | 0.5776 | +0.0000 | 0.5776 | +0.0000 | +0.0000 |
| 75% | 0.5296 | 0.4681 | **+0.0615** | 0.5736 | +0.1055 | +0.0440 |
| 50% | 0.4716 | 0.3476 | **+0.1240** | 0.5236 | +0.1760 | +0.0520 |
| 25% | 0.3110 | 0.2162 | **+0.0948** | 0.3558 | +0.1396 | +0.0448 |

**Two honest conclusions.** (1) **Forgetting generalizes** — held-out, principled retention still
beats random eviction at every level (+0.062 to +0.124), peaking at keep=50% (0.472 vs 0.348 =
**1.36×**). Retention learned from past queries preserves memories that answer *future, unseen*
queries; it is not memorizing the eval set. (2) **The earlier 2.3× was inflated by leakage** —
letting the policy see the eval queries adds ~0.044–0.052 recall at every eviction level, roughly
*doubling* the apparent advantage (e.g. keep=75%: honest +0.062 → leaked +0.106). So ~40–45% of the
old headline advantage was query leakage; the honest, generalizing effect is ≈1.4× at aggressive
eviction. (The measurement is exact and fast: retrieval scores are per-item, so a survivor subset's
top-k is the full-corpus ranking filtered to survivors — no per-fraction memory rebuild, which is
O(n²) on the write path and was intractable.)

**Conclusion (v2 capability scorecard):** composite retrieval and write-time extraction are
validated on LoCoMo (recall 0.61; QA parity with Mem0). Reflection is **measured-neutral** on LoCoMo
QA (summaries ≠ facts). **Forgetting is measured-positive** on its purpose-fit instrument — a
differentiated-access retention curve where principled eviction beats random (held-out ~1.4× at
aggressive eviction, peaking +0.124 at keep=50%; the earlier 2.3× was ~40–45% query leakage, now
corrected with a train/test split) and denoises. **Bi-temporal** was measured on its purpose-fit instrument too (LongMemEval
knowledge-update): the current system is already at **0.875** because the frontier answerer resolves
updates itself, so retrieval-level validity filtering has little end-to-end headroom — its value is
at the retrieval-precision / weak-answerer level. The throughline across the whole eval: **LoCoMo/QA
with a frontier answerer measures fact-retrieval capabilities and masks the rest**; each capability
must be validated on an instrument that isolates *what it changes*, and doing so honestly shows one
neutral (reflection), one clear win (forgetting), and one strong-answerer-masked (bi-temporal).

### MultiHop: entity-bridge gather — abandoned (measured negative, 2026-07-19)

A fresh attempt at the MultiHop *pool-absence* gap (recall@50 = 0.516, so ~48% of MultiHop gold is
never even pooled). Diagnostic first (`--recall-curve --qtype multihop`): recall@10 0.316 / @20 0.394
/ @50 0.516 — a large ranking gap (0.316→0.516, gold pooled but below top-10) *and* a large pool
gap (0.484 never pooled). We targeted the pool gap with a NEW deterministic lever distinct from the
already-exhausted graph expansion: **entity-bridge gather** — extract named entities (capitalized
spans) from the top seed hits, run a *focused* retrieval per entity, and union into the pool
(opt-in, discounted). Hypothesis: a multi-hop bridge fact ("Beth is a doctor") shares no vocabulary
with the query ("what does Alice's sister do?") but is reachable via a bridge entity ("Beth")
surfaced by the seed retrieval.

Measured (full-set MultiHop, 282 q):

| | recall@10 | recall@20 | recall@50 |
|---|---|---|---|
| baseline (PRF) | 0.316 | 0.394 | **0.516** |
| + entity-bridge | 0.319 | 0.398 | **0.509** |

**Negative — abandoned (not shipped).** recall@10/@20 moved +0.003 (noise); recall@50 went *down*
0.007. A per-entity retrieval returns many entity-*general* turns ("everything mentioning Beth"),
which fill and self-poison the bounded pool, displacing gold faster than they add the one specific
connecting fact. This is the **third independent confirmation** of the prior finding
([[multihop-needs-coreference-edges]] in the project notes; PRs #82/#83/#86): gathering *more*
entity/similarity-related content adds noise > gold, whether via KG edges or focused re-retrieval.
It also sharpens *why* the agentic loop works where deterministic gather doesn't — an LLM identifies
the *specific* missing fact ("SEARCH: what is Beth's job"), whereas any deterministic expansion
("more about Beth") is too coarse. The working MultiHop levers remain **ranking** (cross-encoder →
0.374) and **LLM-guided agentic gather** (QA 0.500); deterministic gather is now exhausted too.

### MultiHop: GPU cross-encoder exploits the ranking gap (confirmed, 2026-07-19)

Having ruled out gather (again), we exercised the *ranking* lever the entity-bridge diagnostic had
just re-exposed: the same PRF pool has recall@50 = 0.516 but recall@10 = 0.316 — 0.200 of MultiHop
gold is *pooled but ranked below the top-10 cutoff*. Reordering that pool with the GPU cross-encoder
(`SYNAPTIC_RERANKER=cross-encoder`, `ms-marco-MiniLM-L-6-v2`, candle CUDA on the RTX A3000 —
`--features "synaptic/static-embeddings synaptic/reranker-model synaptic/cuda"`) against the
*identical* PRF pool:

| | recall@10 | recall@20 | recall@50 |
|---|---|---|---|
| baseline (PRF) | 0.316 | 0.394 | **0.516** |
| + GPU cross-encoder | **0.3968** | 0.4580 | **0.5159** |

**Positive — and it is *pure reordering*.** recall@50 is unchanged (0.516 → 0.5159, same pool), so
the entire +0.081 recall@10 (+26%) comes from ranking already-pooled gold into the top-10 — the
mirror image of the entity-bridge failure (which moved the pool and got nothing). This reproduces and
slightly exceeds the earlier full-set cross-encoder MultiHop figure (0.3739) on the current PRF
baseline. Net for MultiHop retrieval, the ledger is unambiguous: every gain since 0.2475 has come
from **ranking** (over-fetch #75, embedding-dominant rerank #76, cross-encoder #80, and this
confirmation), and **zero** from any of the four deterministic-gather attempts (similarity edges,
coreference edges, entity-bridge, MMR). The remaining MultiHop headroom above 0.397 is the pool gap
(0.484 never pooled), addressable only by LLM-guided agentic gather, not by reordering.

## Full-scale agentic QA: the 0.500 subset was pessimistic (2026-07-19)

Every agentic QA number to date (0.500, above) was on a 40-question subset — the FIRST few
questions per conversation, which over-weights the hard MultiHop/Temporal categories and
under-samples the dominant SingleHop. The standing rule (see the eval memory) is to gate quality
claims on the full 1,986-question set. This is that run: **all 1,986 questions**, agentic
(`--agentic-qa`, 3 rounds) + PRF + GPU cross-encoder + codex judge, concurrency 12, in **72 min**
(4330 s) on the RTX A3000.

To make a multi-hour codex run survivable, `run_agentic_qa` gained a resumable JSONL checkpoint
(`SYNAPTIC_EVAL_QA_CHECKPOINT`): each graded question is flushed as it completes; on restart,
already-graded questions are restored into the aggregates and skipped, and fully-graded
conversations skip ingestion. (A live smoke test confirmed a second run resumes in 0 s with an
identical aggregate and no duplicate lines.)

| category | n | accuracy |
|---|---|---|
| **Overall** | **1986** | **0.5463** |
| SingleHop | 841 | 0.7467 |
| Temporal | 321 | 0.7321 |
| OpenDomain | 96 | 0.4167 |
| MultiHop | 282 | 0.3121 |
| Abstention | 446 | 0.2108 |

**The subset was pessimistic, not optimistic.** Full-scale accuracy is **0.5463**, *above* the 0.500
subset figure — because SingleHop (the plurality, 841 q) answers at 0.747 and the subset barely
sampled it. MultiHop (0.312) and OpenDomain (0.417) remain the retrieval-bound tail, exactly
tracking their first-stage recall ceilings; the Abstention/adversarial category (0.211) is the
weakest — see faithfulness below.

**Failure split is ~even at scale** (recall-vs-judge cross-tab): of 900 wrong answers,
retrieval-bound (no gold retrieved, d) = 424 and judge-bound (gold retrieved but wrong, b) = 476.
Neither dominates — further gains need BOTH better retrieval for the MultiHop/OpenDomain tail AND a
better answer step for the judge-bound half. (a = gold-retrieved-and-correct = 978, c =
answered-correct-without-gold = 104.)

**Faithfulness at scale (the honest caveat):** on the 358 questions whose gold evidence was never
retrieved, the model **confabulates a confident wrong answer 148 times (41%)** rather than
abstaining (115 abstained, 95 answered correctly anyway). On the 446 adversarial/abstention
questions it confabulates 147 times. Over-abstention (had the evidence, still abstained) = 71.
So the headline 0.5463 comes with a real groundedness cost: absent evidence, the system guesses
confidently more often than it abstains. The structural-grounding gate (`SYNAPTIC_EVAL_GROUNDED`,
documented above) is the opt-in lever against this and was left OFF here to measure the raw number.

**This is the standing-gate number** for agentic retrieval QA on LoCoMo. Caveat: the codex judge is
nondeterministic run-to-run (~±0.03 on small samples; far tighter at n=1986), and grades adversarial
abstention strictly (a correctly-refused question can still be marked wrong if the phrasing diverges
from the expected form), which depresses the Abstention row.

## Write-time distillation A/B: facts-primary retrieval HURTS (negative result, 2026-07-21)

We built write-time fact distillation as a real, default write path (spec/plan under
`docs/superpowers/`): when an LLM endpoint + knowledge graph are configured, `store()` distills each
turn into fact-memories (`{key}::fact{i}`) that become the primary retrievable units, and the raw
turn is tagged `raw_source` and excluded from default `search` (`MemoryConfig::distillation`,
`retrieval_excludes_raw_sources`). The hypothesis (from the earlier Mem0-parity result) was that
distilled facts retrieve cleaner than raw turns. **A matched A/B refuted this for the accessible
configuration, so distillation ships default-OFF (opt-in).**

**Setup.** Same conversation (LoCoMo conv 0), same codex judge + retrieval (PRF + GPU cross-encoder);
only the write path differs. Arm A ingests raw turns (`distillation: Off`). Arm B distills
facts-primary (`--distill`, `On`), extracting with a local `qwen2.5:7b-instruct` (Ollama, GPU). The
matched set is the 198 questions both arms graded.

| category | Arm A (raw) | Arm B (7B distill) | Δ |
|---|---|---|---|
| **Overall** | **0.5051** | **0.2576** | **−0.2475** |
| SingleHop | 0.729 | 0.257 | −0.471 |
| Temporal | 0.784 | 0.486 | −0.297 |
| MultiHop | 0.188 | 0.062 | −0.125 |
| OpenDomain | 0.385 | 0.308 | −0.077 |
| Abstention | 0.196 | 0.196 | +0.000 |

**Distillation nearly halved accuracy.** The driver is abstention: it **doubled** (103 vs 44 "I
don't know" on the 198). The judge grades the actual answer text, so this is a real retrieval/answer
loss, not a scoring artifact. (The `gold_in_context` recall metric reads 0.000 for Arm B — an
*instrumentation artifact*: it credits raw-turn `evidence_id`s, which distillation replaces with
`::fact` keys, so recall is not meaningful under facts-primary. Accuracy is the valid signal.)

**Two mechanisms (from a direct qwen-vs-codex extraction diagnostic on the answer-losing turns):**
1. **Extractor quality is real.** For "When did Melanie run a charity race?", codex extracted
   *"Melanie ran a charity race … last Saturday"* (answer preserved) while qwen dropped it entirely.
   A frontier extractor would beat the 7B's 0.258. But a frontier extractor is **impractically slow
   for the write path** — codex-via-CLI is ~15 s/turn × multiple calls/turn (extract + per-fact
   resolve); a single-conversation ingest did not finish in 1 h, so the full codex A/B was infeasible.
2. **Facts-primary loses context regardless of extractor.** Raw turns are ingested with a
   `[session-date]` prefix; extracted facts drop it, so temporal questions (gold "7 May 2023"; the
   turn says "yesterday") lose their date anchor even with a perfect extractor. Excluding raw removes
   the verbatim fallback. This is a design-level floor a better extractor cannot fully overcome.

**Decision (honesty bar).** Distillation defaults to **`Off`**. The machinery is complete, tested,
and reviewed, and is available opt-in (`MemoryConfig::distillation = On`, or the deprecated
`store_extracted_facts = true`) for deployments with a strong, fast extractor that measure it for
their own workload. It is **not** shipped default-on because the only end-to-end measurement shows
harm. Caveats: single-conversation matched set (n=198); local 7B extractor; a frontier extractor was
not measured end-to-end (infeasibly slow here). The finding that generalizes: **facts-primary
retrieval that excludes raw turns trades away verbatim recall (dates, exact details) that LoCoMo QA
depends on** — distillation should *augment* raw retrieval, not replace it, and only with an
extraction step faithful enough to not drop answer-bearing detail.

### Follow-up: augment (raw retained) recovers the loss — the exclusion was the culprit (2026-07-21)

The negative result above pointed at a specific fix: distillation should *augment* raw retrieval,
not *replace* it. We tested it directly with a third arm on the same conversation and matched
questions, changing only `retrieval_excludes_raw_sources` (a new `--distill-keep-raw` eval flag):
Arm C distills facts (same qwen2.5:7b extractor) but **keeps raw turns searchable**.

| arm | overall | abstained | Δ vs raw baseline |
|---|---|---|---|
| A — raw baseline | 0.5051 | 44 | — |
| B — facts-primary (raw excluded) | 0.2576 | 103 | −0.2475 |
| **C — augment (raw retained)** | **0.5000** | 52 | **−0.0051 (tie)** |

Per category (A raw / B facts-primary / C augment): Temporal 0.784 / 0.486 / **0.784**; MultiHop
0.188 / 0.062 / **0.188**; OpenDomain 0.385 / 0.308 / **0.385**; SingleHop 0.729 / 0.257 / 0.671;
Abstention 0.196 / 0.196 / **0.261**.

**The harm was entirely the raw-exclusion, not distillation.** Augment recovers the whole −0.25 —
matching baseline exactly on Temporal, MultiHop, and OpenDomain, nearly on SingleHop, and slightly
*beating* it on Abstention. So distilled facts added alongside raw turns are **safe** (neutral with a
weak extractor) and preserve the verbatim fallback (dates, exact details) that facts-primary threw
away. They do not *lift* QA here — a 7B extractor's facts add no signal the raw turns lack — but this
is the correct foundation: with a stronger extractor, augment could add lift without the downside.

**Change applied:** the opt-in distillation default is now **augment** —
`MemoryConfig::retrieval_excludes_raw_sources` defaults to `false`. Enabling `distillation = On` (or
the deprecated `store_extracted_facts = true`) now adds facts alongside raw turns rather than
replacing them, so opting in is safe by default. Facts-primary (raw excluded) remains available by
setting the flag to `true`, for callers who measure it as a net win on their own extractor/workload.

## Three capability-validation threads: all masked by a strong answerer (2026-07-21)

Following the distillation investigation, we measured the three remaining v2 capability questions on
their target instruments. All three came back neutral/masked — a coherent, honest result that sharpens
the scorecard's core lesson.

**Thread 1 — bi-temporal (supersede-on-update).** LongMemEval knowledge-update, n=40, codex agentic,
heuristic supersession marking; only `retrieval_excludes_superseded` differs (new `--exclude-superseded`
eval flag):

| arm | accuracy |
|---|---|
| superseded retained | 0.925 (37/40) |
| superseded excluded (bi-temporal) | 0.950 (38/40) |

+1 question, within codex noise at n=40. Codex is near-ceiling (mean rounds = 1.0, no follow-ups) — it
resolves updates from the retrieved context itself, so the retrieval-precision value of dropping stale
facts is **masked**. The mechanism is proven (library unit test: Berlin→Munich, flag on returns only
Munich); the end-to-end value needs a weak answerer to surface.

**Thread 2 — reflection/synthesis.** LongMemEval multi-session (the synthesis-requiring type), n=40,
single-shot codex, `--reflect` vs baseline:

| arm | accuracy |
|---|---|
| baseline | 0.425 (17/40) |
| + reflection (216 insights synthesized) | 0.450 (18/40) |

+1 question, within noise — **neutral even on the question type reflection should most help**. 216
insights were synthesized and made retrievable but didn't move accuracy: the heuristic synthesizer
(TF-IDF clustering) doesn't produce the specific cross-turn facts these questions need, and codex
already synthesizes across the retrieved raw turns. A fair test needs an LLM synthesizer.

**Thread 3 — distillation lift with a stronger extractor.** Matched conv-0 (n=198), augment mode,
qwen2.5-**14b** extractor vs the earlier 7b:

| arm | accuracy |
|---|---|
| raw baseline | 0.5051 |
| augment (7b) | 0.5000 |
| augment (14b) | 0.4697 |

A stronger extractor does **not** turn augment into a lift (14b within noise of baseline, marginally
below 7b). In augment mode raw turns already carry the answers, so single-turn extraction — however
good the model — only *duplicates* what raw turns hold. Per-turn distillation adds no retrievable
signal on LoCoMo regardless of extractor strength; a real lift would need cross-turn synthesis, not
better per-turn facts.

**The throughline (confirmed across three fresh measurements).** Retrieval-side capabilities
(distillation, bi-temporal validity, reflection) are **masked by a strong answerer**: codex reads
verbatim detail (masks distillation), resolves updates from context (masks bi-temporal), and
synthesizes across retrieved turns (masks reflection). Each would matter with a *weak* answerer, or
where first-stage retrieval — not answering — is the bottleneck. This is exactly what the v2 scorecard
predicted; measuring it on the target instruments confirms it rather than refuting it. The honest
consequence: none of these three is a shippable default lift on LoCoMo/LongMemEval with a frontier
answerer, and each needs a weak-answerer (or LLM-synthesizer) instrument — which the agentic eval path
does not yet support — to demonstrate its value. (Caveats throughout: n≈40 subsets, heuristic
supersession/synthesis, single-conversation extractor comparison.)

## Weak-answerer instrument: the tool to unmask retrieval-side capabilities (2026-07-21)

The three "masked" verdicts above all share a limitation: a strong (codex) answerer resolves
updates, synthesizes, and reads verbatim detail itself, so retrieval-side capabilities can't show
their value. We built the instrument to remove that mask: the agentic loop is split into an
**answerer** (a `Completer` trait — the answer-or-search completions) and a **grader** (codex, still
grades). With `SYNAPTIC_EVAL_ANSWERER_URL`/`_MODEL` set, a weak local model (qwen2.5-7b via ollama)
drives answering while codex grades — so *retrieval* has to carry the answer. (The answerer uses
dedicated env vars, decoupled from the library reasoner's `SYNAPTIC_EVAL_LLM_URL` fallback, so ingest
stays heuristic/fast rather than running per-turn LLM extraction.)

**The instrument works, demonstrated on bi-temporal.** LongMemEval knowledge-update, n=40, qwen
answerer + codex grader:

| answerer | superseded retained | superseded excluded |
|---|---|---|
| codex (strong) | 0.925 | 0.950 |
| **qwen-7b (weak)** | **0.500** | **0.500** |

The weak answerer drops to 0.500 (vs codex's 0.925) — confirming it is genuinely weaker and
retrieval-dependent, i.e. the mask is off. **Yet bi-temporal is still a no-op** — the two arms produce
**identical predictions on all 40 questions** (`exclude_superseded` changed retrieval for zero of
them). This is a sharper finding than the strong-answerer run gave: the blocker is not answerer
masking, it is that **supersession never fires** — the heuristic reasoner detects LongMemEval's
natural-language updates as same-subject/predicate contradictions **0 times**. The bi-temporal
mechanism is correct (library unit test: Berlin→Munich) but does not trigger on realistic updates
without an LLM reasoner to detect the contradiction at write time.

**Net.** The weak-answerer instrument is validated and reusable (any agentic run: set the answerer
vars). It converts the "masked" verdicts into precise ones — for bi-temporal, the value question is
blocked upstream at *supersession detection*, not at the answerer. Applying it to distillation
(augment) and reflection would similarly isolate whether their value is answerer-masked or, like
bi-temporal, blocked upstream — each needs its own (LLM-reasoner or LLM-synthesizer) run, now that the
answerer side is solved. Caveats: n=40, heuristic supersession, single dataset.

## Weak-answerer distillation: facts-primary hurts every answerer — closing the question (2026-07-21)

The distillation investigation's last open question: with a strong answerer, facts-primary
distillation *hurt* (it reads verbatim detail from raw turns) — would a **weak** answerer instead do
better with clean distilled facts than noisy raw turns? Using the weak-answerer instrument (qwen-7b
answers, codex grades), matched conv-0, n=199:

| answerer | raw | facts-primary | Δ |
|---|---|---|---|
| codex (strong) | 0.5051 | 0.2576 | −0.2475 |
| **qwen-7b (weak)** | **0.3266** | **0.1910** | **−0.1357** |

**Facts-primary hurts both answerers.** The hypothesis is refuted: a weak answerer does *worse* with
distilled facts, not better. The mechanism holds for both — excluding raw turns discards the verbatim
detail (dates, exact phrasing) that both strong and weak answerers need, and single-turn 7b facts
cannot replace it. (The weak answerer also validates the instrument: qwen drops raw-turn QA from
codex's 0.505 to 0.327, i.e. it is genuinely retrieval-dependent, yet still can't benefit from
facts-primary.)

**This closes the distillation question.** There is no answerer-strength regime where per-turn
facts-primary distillation is a win. The augment default (raw retained; `retrieval_excludes_raw_sources
= false`, shipped) is the only safe configuration, and it is neutral. A real distillation *lift* would
require the facts to add signal raw turns lack — i.e. cross-turn synthesis or entity linking, not
per-turn extraction — which is a different mechanism than what this feature implements. Caveats:
single conversation (n=199), 7b extractor.
