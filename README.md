# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent memory system in Rust: key/value memory with a knowledge graph,
vector embeddings, hybrid (keyword + semantic) retrieval, temporal tracking,
and optional feature-gated security primitives.

This is a development library (version 0.2.0, not published to crates.io).
The table below states honestly which parts are stable, which are beta, and
which are experimental. There are no simulated or fake fallback code paths:
features that are not really implemented return errors (fail closed) instead
of pretending to work.

## Module Maturity

| Module | Status | Notes |
|---|---|---|
| Memory store/retrieve (`AgentMemory`) | stable | Core store/retrieve/update with tests; in-memory and file (Sled) backends |
| Storage backends | stable | Memory, file (Sled); SQL (PostgreSQL) behind `sql-storage` |
| Knowledge graph | stable | Node merging, relationship detection, traversal; tested |
| Embeddings | stable | Deterministic local embeddings; used by hybrid retrieval |
| Search / hybrid retrieval | beta | Tokenized keyword + vector search fused with Reciprocal Rank Fusion (RRF); HNSW ANN index used for related-memory counting |
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

Other optional features: `sql-storage`, `multimodal`, `external-integrations`,
`cross-platform`, `observability`, and `distributed-experimental` (explicitly
experimental, see maturity table). Convenience groups: `full`, `full-experimental`, `minimal`.

## Performance

Measured benchmark results (with methodology and caveats) live in
[docs/performance.md](docs/performance.md). Retrieval-quality evaluation on
the LoCoMo long-term-memory benchmark — recall/precision/MRR, latency, memory
growth, and a per-capability ablation — lives in
[docs/evaluation.md](docs/evaluation.md). No other performance or quality
numbers in this repository should be treated as validated.

## Testing

```bash
cargo test --lib                                   # library unit tests (440+)
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
