# Performance

Measured benchmark results for Synaptic's core memory operations. These are
the only performance numbers in this repository that have actually been run
and recorded; treat anything else as unvalidated.

## Methodology

- **Harness**: criterion 0.5 via `cargo bench --features "security test-utils"
  --bench comprehensive_performance_suite`, release profile.
- **Reduced sampling**: `--sample-size 10 --measurement-time 3` to keep wall
  time reasonable; numbers are therefore *indicative*, not rigorous.
- **Environment**: single dev machine (unspecified hardware), Linux,
  rustc 1.93.1. No isolation (other processes running). Do not compare these
  numbers across machines; use them for order-of-magnitude expectations and
  for detecting regressions on the same machine.
- **Date**: 2026-07-12.
- **Scale**: 1k and 10k entries. The 100k-entry configuration was not run
  (its store phase alone extrapolates to minutes per criterion iteration);
  results at that scale are unknown.

## Results

### Store (`MemoryStorage::store`, batch of N entries per iteration)

| Entries | Total time (median) | Per-entry | Throughput |
|---|---|---|---|
| 1,000 | 13.6 ms | ~13.6 µs | ~73K elem/s |
| 10,000 | 934 ms | ~93 µs | ~10.7K elem/s |

Per-entry cost grows with store size at 10k; store is not O(1) per entry at
that scale in this benchmark.

### Retrieve (`MemoryStorage::retrieve`, N sequential lookups per iteration)

| Entries | Total time (median) | Per-lookup | Throughput |
|---|---|---|---|
| 100 | 15.3 µs | ~153 ns | ~6.6M elem/s |
| 1,000 | 155 µs | ~155 ns | ~6.5M elem/s |
| 10,000 | 1.78 ms | ~178 ns | ~5.6M elem/s |

Key lookup is effectively constant time across these scales.

### Search (`MemoryStorage::search`, corpus of 10,000 entries)

| Result limit | Time per query (median) |
|---|---|
| 10 | 6.22 ms |
| 100 | 6.38 ms |
| 1,000 | 6.71 ms |

Search latency is dominated by scanning/scoring the corpus, not by the
result limit. Note this benchmarks the storage-layer search; `AgentMemory::search`
adds the hybrid (keyword + vector RRF) retrieval pipeline on top when
embeddings are enabled.

## Caveats

- Small sample size (10) and short measurement windows; expect several
  percent run-to-run variance (criterion flagged outliers in some groups).
- Benchmarks exercise the in-memory storage backend; file (Sled) and SQL
  backends will be slower and were not measured here.
- No claims are made about concurrent throughput, sustained load, or
  100k+ entry behavior.

## Reproducing

```bash
cargo bench --features "security test-utils" \
  --bench comprehensive_performance_suite -- \
  --sample-size 10 --measurement-time 3 \
  "storage_operations/memory_storage_(store|retrieve)"

cargo bench --features "security test-utils" \
  --bench comprehensive_performance_suite -- \
  --sample-size 10 --measurement-time 3 \
  "search_operations/search_content"
```

All benches compile with `cargo bench --no-run --features "security test-utils"`
(includes analytics and security bench suites; `comprehensive_benchmarks.rs`
builds but lacks a `harness = false` entry in Cargo.toml, so criterion CLI
options do not reach it — use `comprehensive_performance_suite` instead).
