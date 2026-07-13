//! Evaluation harness for `synaptic` against the LongMemEval-S and LoCoMo
//! long-term-memory benchmarks.
//!
//! The real datasets are fetched (not committed) via `fetch_datasets.sh` into
//! the gitignored `data/` directory; unit tests run against a small
//! hand-authored fixture in `fixtures/` that mirrors both real schemas.

pub mod dataset;
