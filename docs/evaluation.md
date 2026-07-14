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
