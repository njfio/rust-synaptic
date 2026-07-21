# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent memory system in Rust: key/value memory with a knowledge graph,
vector embeddings, an **intelligent write path** that distills stored text into
retrievable facts, and a multi-stage retrieval pipeline — hybrid
(keyword + semantic) retrieval with RRF fusion, over-fetch reranking,
pseudo-relevance-feedback (PRF) pool augmentation, and an optional GPU
cross-encoder reranker — plus temporal tracking and optional feature-gated
security primitives.

Retrieval quality and end-to-end answer accuracy are measured on the LoCoMo
long-term-memory benchmark (real dataset, real numbers, honest caveats) — see
[docs/evaluation.md](docs/evaluation.md). Highlights (full 1,986-question set
unless noted): retrieval recall@10 **0.61** (best config); an **agentic
answer-guided retrieval** eval mode that lifts end-to-end QA accuracy to
**0.50** on a labelled subset (a codex-CLI judge; see caveats) while abstaining
("I don't know") rather than confabulating when the evidence is absent — a
faithfulness property measured explicitly. **Write-time fact distillation** is
an opt-in write path (`MemoryConfig::distillation`, default **Off**): a matched
end-to-end A/B showed that a *facts-primary* variant (raw turns excluded from
search) **hurts** QA (0.505 → 0.258 on 198 questions with a local extractor) —
the exclusion, not the distillation, was the culprit — whereas an *augment*
variant that keeps raw turns searchable ties the baseline (0.500). So
distillation ships default-off, and when enabled defaults to the safe augment
mode. Honest head-to-heads vs Mem0 and Graphiti (Zep), and this distillation
investigation — reversing an earlier under-tested claim, then isolating the fix
— are documented with all caveats in [docs/evaluation.md](docs/evaluation.md).

This is a development library (version 0.2.0, not published to crates.io).
The table below states honestly which parts are stable, which are beta, and
which are experimental. There are no simulated or fake fallback code paths:
features that are not really implemented return errors (fail closed) instead
of pretending to work. No quality/performance number appears here that is not
backed by a measured run in [docs/evaluation.md](docs/evaluation.md).

## Module Maturity

| Module | Status | Notes |
|---|---|---|
| Memory store/retrieve (`AgentMemory`) | stable | Core store/retrieve/update with tests; in-memory and file (Sled) backends |
| Storage backends | stable | Memory, file (Sled); SQL (PostgreSQL) behind `sql-storage` |
| Knowledge graph | stable | Node merging, relationship detection, traversal; tested |
| Intelligent write path (`MemoryReasoner`) | beta | On store, the active reasoner extracts facts/entities/relations from the value: entities & relations feed the knowledge graph. Deterministic `HeuristicReasoner` by default (offline); `LlmReasoner` (OpenAI-compatible endpoint) under `llm-reasoning`, with heuristic fallback. |
| Write-time distillation (`MemoryConfig::distillation`) | experimental, **default off** | When `On` (or the deprecated `store_extracted_facts = true`) each extracted fact is persisted as its own retrievable memory (`<key>::fact<N>`). Defaults to **augment** (`retrieval_excludes_raw_sources = false`): facts are added alongside raw turns — a matched LoCoMo A/B tied baseline (0.500). Setting the flag `true` gives facts-primary (raw excluded), which the same A/B measured to **hurt** QA (0.505 → 0.258). See [docs/evaluation.md](docs/evaluation.md) |
| Embeddings | stable | Deterministic local embeddings; used by hybrid retrieval |
| Search / hybrid retrieval | beta | Tokenized keyword + vector + graph + temporal retrievers fused with Reciprocal Rank Fusion (RRF), composite scoring, and a deterministic reranker over an over-fetched candidate pool. Optional: PRF pool augmentation (`SYNAPTIC_RETRIEVAL_PRF`), multi-hop graph expansion, semantic embeddings (`static-embeddings` / `ml-models` / Ollama), and a candle BERT cross-encoder reranker (`reranker-model`, GPU via `cuda`). Measured on LoCoMo — see [docs/evaluation.md](docs/evaluation.md) |
| Evaluation harness (`tools/eval`) | beta | Real LoCoMo/LongMemEval loaders; retrieval metrics (recall/precision/MRR, `--recall-curve`, `--completeness`), memory-growth, and LLM-gated end-to-end QA (`--qa-only`, `--agentic-qa`) with abstention/faithfulness metrics. Every printed number is a real run; QA is gated on a configured judge, never fabricated |
| Checkpoint / restore | beta | Non-destructive restore (snapshot-validate-swap); tested |
| Analytics | beta | Basic behavioral/performance analytics behind `analytics` |
| Security: auth (`security`) | beta | Real argon2 password hashing, TOTP MFA, constant-time API key comparison (`subtle`), deny-by-default policy engine, zeroized keys. Opt-in feature flag |
| Zero-knowledge proofs (`zero-knowledge-proofs`) | beta | Real Poseidon hash + Groth16 proofs (bellman/BLS12-381) with verifier-derived public inputs; soundness attack-tested. Opt-in |
| Homomorphic encryption (`homomorphic-encryption`) | beta (narrow scope) | Real TFHE `FheInt64` encrypt/decrypt/sum/average only. Encrypted search, similarity, and count are descoped and return errors (fail closed) |
| Differential privacy (`security`) | beta | Real Laplace noise via OS RNG; ε-budget accounting is property-tested |
| Distributed (`distributed-experimental`) | experimental | NOT production Raft. Consensus and realtime sync fail closed. Feature was renamed from `distributed` to make this explicit |
| Multimodal (`multimodal`) | experimental | Some processing is real (OpenCV images, Tesseract OCR, tree-sitter code analysis, document parsing); other parts are simple heuristics |
| Cross-platform (`cross-platform`) | experimental | WASM/mobile adapter interfaces exist but platform bridges are not linked or shipped |

## Installation

Clone and build from source:

```bash
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic
cargo build
```

For use in other projects:

```toml
[dependencies]
synaptic = { git = "https://github.com/njfio/rust-synaptic.git" }
```

## Quick Start

This example is a doctest (`cargo test --doc` verifies it compiles and runs):

```rust
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default config: in-memory storage, knowledge graph enabled
    let mut memory = AgentMemory::new(MemoryConfig::default()).await?;

    // Store memories by key
    memory.store("user_name", "Alice").await?;
    memory.store("user_preference", "prefers dark mode").await?;

    // Retrieve by key
    let entry = memory.retrieve("user_name").await?;
    assert_eq!(entry.map(|e| e.value), Some("Alice".to_string()));

    // Search (tokenized keyword + semantic hybrid retrieval)
    let results = memory.search("dark mode", 10).await?;
    assert!(!results.is_empty());

    Ok(())
}
```

## Feature Flags

Default features: `core`, `storage`, `embeddings`, `analytics`, `compression`.

Security features are **opt-in** and gated:

- `security` — argon2 authentication, TOTP MFA, constant-time API keys, policy engine, differential privacy
- `zero-knowledge-proofs` — Poseidon + Groth16 proofs
- `homomorphic-encryption` — TFHE FheInt64 (sum/average only; other encrypted ops fail closed)

Retrieval-quality features (opt-in; the default build stays lean and offline
with a lexical/TF-IDF embedder):

- `static-embeddings` — fast pure-Rust model2vec static embeddings (best
  speed/quality default; fetch the model with `scripts/fetch_embedding_model.sh --potion`)
- `ml-models` — candle transformer embeddings (MiniLM); heavier, CPU-slow
- `reranker-model` — candle BERT cross-encoder reranker (ms-marco-MiniLM);
  strongest ranking, opt-in (`SYNAPTIC_RERANKER=cross-encoder`), GPU-recommended
- `cuda` — candle CUDA backend so `ml-models`/`reranker-model` run on a GPU
- `llm-reasoning` — `LlmReasoner` for the intelligent write path (extract /
  resolve / synthesize via an OpenAI-compatible endpoint, `SYNAPTIC_LLM_URL` /
  `SYNAPTIC_LLM_MODEL`), with deterministic heuristic fallback. Without it the
  offline `HeuristicReasoner` is used

Write-time distillation is a runtime config, not a feature flag: set
`MemoryConfig::distillation = DistillationMode::On` (or the deprecated
`store_extracted_facts = true`) to persist each extracted fact as a retrievable
memory. **Default is `Off`.** When enabled it defaults to **augment**
(`retrieval_excludes_raw_sources = false`): facts are added alongside raw turns,
which a matched LoCoMo A/B measured as neutral vs. the raw baseline (0.500 vs
0.505). Setting `retrieval_excludes_raw_sources = true` gives *facts-primary*
(raw excluded), which the same A/B measured to **hurt** QA (→ 0.258) — use it
only with a strong extractor you have measured on your own workload. Quality
scales with the active reasoner (`LlmReasoner` under `llm-reasoning`, else
`HeuristicReasoner`).

Other optional features: `sql-storage`, `multimodal`, `external-integrations`,
`cross-platform`, `observability`, and `distributed-experimental` (explicitly
experimental, see maturity table). Convenience groups: `full`, `full-experimental`, `minimal`.

## Performance & Evaluation

Measured micro-benchmarks (with methodology and caveats) live in
[docs/performance.md](docs/performance.md). The full retrieval-quality and
end-to-end QA evaluation on the **LoCoMo** long-term-memory benchmark lives in
[docs/evaluation.md](docs/evaluation.md). No other performance or quality
numbers in this repository should be treated as validated.

Headline measured results (see the doc for methodology and the many caveats):

| metric | value | notes |
|---|---|---|
| retrieval recall@10 (full 1,986-q set) | 0.5237 → **0.6104** | baseline → best config (over-fetch + embedding rerank + cross-encoder) |
| MultiHop recall@10 | 0.2475 → **0.3739** | +51%, via ranking (not graph connectivity) |
| search latency p50 | **~0.46–0.76 s** | after a 9.5× pipeline-latency fix |
| end-to-end QA accuracy (40-q subset, codex judge) | 0.375 → **0.50** | single-shot → agentic answer-guided retrieval |
| QA with write-time fact extraction (40-q, codex judge) | 0.325/0.450 → **0.475/0.500** | raw turns → over extracted facts (single-shot/agentic); **ties Mem0's 0.500** on identical facts |
| faithfulness: abstains on unanswerable q | **~90%** | agentic + grounding; confabulates rather than guesses only ~10% |

Run the harness (LLM-free retrieval metrics need no judge):

```bash
cargo build --release -p synaptic-eval --bin run_eval --features synaptic/static-embeddings
SYNAPTIC_RETRIEVAL_EMBEDDER=static SYNAPTIC_STATIC_MODEL_DIR=models/potion-base-8M \
  ./target/release/run_eval tools/eval/data/locomo10.json --retrieval-only
```

End-to-end QA (`--qa-only` / `--agentic-qa`) requires a configured judge
(`SYNAPTIC_EVAL_JUDGE=codex` with the `codex` CLI, or an OpenAI-compatible
endpoint); without one, QA is reported as not-run — never fabricated. The
GPU cross-encoder and agentic modes are documented in `docs/evaluation.md`.

**Head-to-head comparisons.** `docs/evaluation.md` also documents honest,
matched-slice comparisons against [Mem0](https://github.com/mem0ai/mem0) and
Graphiti (Zep) — same LoCoMo questions, same codex answerer+judge, differing
only in the memory system — with reproducible harnesses in
`tools/eval/comparisons/`. All run on the codex OAuth login (no API key). The
write-up includes a result that *reversed* an earlier under-tested "we beat
Mem0" claim once Mem0 was given its intended frontier extractor and correct
dates; the honest conclusion is that write-time fact extraction (not the
retrieval engine) is the dominant lever, and the systems cluster once held to
the same answerer.

**Honesty note:** QA accuracy is measured on labelled subsets (the full
1,986-question judge run is bounded by sequential judge latency) and the codex
judge is nondeterministic (~±0.03); the headline deltas are far beyond that
noise. Retrieval metrics are full-set. Weight-tuned settings (reranker weights,
PRF) are tuned on LoCoMo and may not transfer; the structural over-fetch fix
does.

## Testing

```bash
cargo test --lib                                   # library unit tests (465+)
cargo test                                          # default-feature test suite
cargo test --features "security test-utils"        # include security suites
```

Lints are enforced in CI with `cargo clippy -- -D warnings`, including denies
on `unwrap`, `panic`, and `print` in library code.

## Examples

```bash
cargo run --example basic_usage
cargo run --example knowledge_graph_usage
cargo run --example phase4_security_privacy --features "security"
```

## Project Structure

```text
src/
├── lib.rs                    # AgentMemory entry point
├── memory/                   # Core memory system
│   ├── storage/             # Storage backends
│   ├── knowledge_graph/     # Graph operations
│   ├── management/          # Memory management
│   ├── temporal/            # Temporal tracking
│   ├── retrieval/           # Hybrid retrieval pipeline
│   └── embeddings/          # Vector embeddings
├── analytics/               # Analytics (feature-gated)
├── security/                # Security & privacy (feature-gated)
├── multimodal/              # Multi-modal processing (experimental)
├── distributed/             # Distributed (experimental, fail-closed)
└── cross_platform/          # WASM/mobile adapters (experimental)
```

## Documentation

- [User Guide](docs/user_guide.md)
- [API Guide](docs/api_guide.md)
- [Architecture Guide](docs/architecture.md)
- [Performance](docs/performance.md)
- [Evaluation (LoCoMo benchmark)](docs/evaluation.md)
- [Testing Guide](docs/testing_guide.md)

Generate API docs with `cargo doc --no-deps --open`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.
