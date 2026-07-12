# Synaptic Remediation & Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate every fake/simulated implementation, fix core correctness defects, delete dead weight, wire the real retrieval/security machinery into the live paths, and make docs and quality gates honest — turning rust-synaptic into a crate whose claims all survive audit.

**Architecture:** Storage becomes the single source of truth with `AgentState` as a true cache; `AgentMemory::store` surfaces (never swallows) subsystem errors via a degraded-result type; search routes through the existing `retrieval` pipeline + HNSW index; security modules either do real cryptography or return hard errors — no silent plaintext fallbacks, ever.

**Tech Stack:** Rust 1.70+, tokio, sha2, bellman/bls12_381 (Groth16), tfhe, rand (OsRng), hnsw_rs, DashMap, tracing.

## Global Constraints

- `unsafe_code = "forbid"` stays (Cargo.toml `[lints.rust]`).
- No `.unwrap()` in `src/` (CI grep gate); new code must not add bare `.expect("should work")`-style non-messages — every `expect` states the invariant that makes it safe, or use `?`.
- No `println!`/`eprintln!` in `src/` (CI grep gate); use `tracing`.
- All existing green CI gates must stay green after every task: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`, `cargo test --all-features`.
- **Security invariant (new, non-negotiable):** a disabled crypto feature returns `Err(MemoryError::FeatureDisabled { .. })` — it never falls back to fake math that resembles the real operation.
- Commit after every task with conventional-commit messages; never bundle unrelated tasks in one commit.
- MSRV: rust-version = "1.70.0" — do not use features newer than that.

## Phase Map (execution order)

| Phase | Theme | Risk it removes |
|---|---|---|
| 0 | Truth & Safety: kill fake crypto paths | Users harmed by fake security |
| 1 | Delete dead weight | 5k+ LOC maintenance drag, false surface area |
| 2 | Core write-path correctness | Silent data-loss / divergence |
| 3 | Real retrieval on the live path | "Search" that is substring matching |
| 4 | Real security implementations | ZK/HE/DP claims become true |
| 5 | Management-module de-stubbing | Placeholder analytics posing as real |
| 6 | Distributed: real consensus or descope | Hand-rolled "simplified" consensus |
| 7 | Quality-gate re-tightening | Hollow `-D warnings` |
| 8 | Honest docs + validated benchmarks | README overclaim |

Phases 0–3 are fully task-detailed below. Phases 4–8 are scoped with hard acceptance criteria and each gets its own detailed plan doc authored at phase start (they depend on decisions and APIs that land in 0–3). Files for those plans: `docs/superpowers/plans/2026-XX-XX-phaseN-<name>.md`.

---

# PHASE 0 — Truth & Safety

### Task 0.1: Add `FeatureDisabled` error variant

**Files:**
- Modify: `src/error.rs` (the `MemoryError` enum, after the existing variants)
- Test: `tests/security_honesty_tests.rs` (new)

**Interfaces:**
- Produces: `MemoryError::FeatureDisabled { feature: String, operation: String }` and constructor `MemoryError::feature_disabled(feature: &str, operation: &str) -> MemoryError`. All later tasks use this.

- [ ] **Step 1: Write the failing test**

Create `tests/security_honesty_tests.rs`:

```rust
//! Tests asserting that disabled crypto features error instead of faking.

use synaptic::error::MemoryError;

#[test]
fn feature_disabled_error_names_feature_and_operation() {
    let err = MemoryError::feature_disabled("homomorphic-encryption", "encrypt_vector");
    let msg = err.to_string();
    assert!(msg.contains("homomorphic-encryption"));
    assert!(msg.contains("encrypt_vector"));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --test security_honesty_tests`
Expected: FAIL — `no method named feature_disabled` / no variant `FeatureDisabled`.

- [ ] **Step 3: Implement**

In `src/error.rs`, add to the `MemoryError` enum:

```rust
    /// Operation requires a compile-time feature that is not enabled
    #[error("feature '{feature}' is not enabled; refusing to run '{operation}' without real implementation")]
    FeatureDisabled { feature: String, operation: String },
```

And in the `impl MemoryError` block:

```rust
    /// Error for operations that must not silently degrade when their feature is off
    pub fn feature_disabled(feature: &str, operation: &str) -> Self {
        Self::FeatureDisabled {
            feature: feature.to_string(),
            operation: operation.to_string(),
        }
    }
```

- [ ] **Step 4: Run test to verify pass**

Run: `cargo test --test security_honesty_tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/error.rs tests/security_honesty_tests.rs
git commit -m "feat(error): add FeatureDisabled variant for honest feature gating"
```

### Task 0.2: Remove fake homomorphic-encryption fallback

**Files:**
- Modify: `src/security/encryption.rs` — every `#[cfg(not(feature = "homomorphic-encryption"))]` block in `encrypt_vector`, `decrypt_vector`, and the constructor fallback (currently lines ~540-600; the fake is `(value * 1.5 + 42.0)`)
- Test: `tests/security_honesty_tests.rs`

**Interfaces:**
- Consumes: `MemoryError::feature_disabled` from Task 0.1.
- Produces: with the feature off, `encrypt_vector`/`decrypt_vector` return `Err(MemoryError::FeatureDisabled { .. })`.

- [ ] **Step 1: Write the failing test** (only compiled without the feature; the default build)

Append to `tests/security_honesty_tests.rs`:

```rust
#[cfg(all(feature = "security", not(feature = "homomorphic-encryption")))]
mod he_disabled {
    // NOTE (implementation deviation): there is no public `HomomorphicEncryption`
    // type; the real encryptor is the private `HomomorphicContext`, reached only
    // through the public `EncryptionManager::homomorphic_encrypt/decrypt` API.
    // Test through that public API (with an MFA-verified SecurityContext, since
    // the default access policy requires MFA) instead of any test-utils re-export.
    #[tokio::test]
    async fn homomorphic_encrypt_errors_when_feature_disabled() {
        let config = synaptic::security::SecurityConfig::default();
        let key_manager = synaptic::security::key_management::KeyManager::new(&config)
            .await
            .expect("key manager must construct");
        let mut mgr =
            synaptic::security::encryption::EncryptionManager::new(&config, key_manager)
                .await
                .expect("constructor must still work so callers can probe availability");
        let entry = synaptic::memory::types::MemoryEntry::new(
            "k".into(),
            "v".into(),
            synaptic::memory::types::MemoryType::ShortTerm,
        );
        let mut ctx =
            synaptic::security::SecurityContext::new("u".into(), vec!["user".into()]);
        ctx.mfa_verified = true;
        let err = mgr
            .homomorphic_encrypt(&entry, &ctx)
            .await
            .expect_err("must not fake-encrypt");
        assert!(err.to_string().contains("homomorphic-encryption"));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --features security --test security_honesty_tests`
Expected: FAIL — fallback currently returns `Ok(fake_bytes)`.

- [ ] **Step 3: Implement**

Replace the body of each `#[cfg(not(feature = "homomorphic-encryption"))]` block in `encrypt_vector` / `decrypt_vector` with:

```rust
        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            Err(MemoryError::feature_disabled(
                "homomorphic-encryption",
                "encrypt_vector", // or "decrypt_vector" in that method
            ))
        }
```

Delete `extract_numeric_features`/`reconstruct_from_numeric_features` fake paths if they become unreachable; keep the constructor working (it may hold config) but have it log at `warn` that HE operations will error.

- [ ] **Step 4: Run tests**

Run: `cargo test --features security --test security_honesty_tests && cargo test`
Expected: PASS; fix any call sites that relied on fake encryption (they must propagate the error).

- [ ] **Step 5: Commit**

```bash
git add src/security/encryption.rs tests/security_honesty_tests.rs
git commit -m "fix(security)!: homomorphic encryption fails closed when feature disabled"
```

### Task 0.3: Real content hashing in ZK module

**Files:**
- Modify: `src/security/zero_knowledge.rs:335-338` (`hash_content`), and `hash_statement` (line ~675) if it shares the fake scheme
- Test: `tests/security_honesty_tests.rs`

**Interfaces:**
- Produces: `hash_content` returns hex-encoded SHA-256. `sha2 = "0.10"` is already a dependency.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(feature = "security")]
#[test]
fn content_hash_is_sha256_not_length_arithmetic() {
    // Two different strings with the same length must hash differently.
    let a = synaptic::security::zero_knowledge::hash_content_for_test("aaaa");
    let b = synaptic::security::zero_knowledge::hash_content_for_test("bbbb");
    assert_ne!(a, b);
    assert_eq!(a.len(), 64); // hex sha256
}
```

Expose the helper: add to `zero_knowledge.rs` a free function `pub fn hash_content_for_test(content: &str) -> String` (or make the method a free `fn hash_content`) so both the struct and tests use one implementation.

- [ ] **Step 2: Run to verify fail** — `cargo test --features security --test security_honesty_tests` → FAIL (equal-length inputs collide under `len()*17+42`).

- [ ] **Step 3: Implement**

```rust
use sha2::{Digest, Sha256};

fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}
```

Add `hex = "0.4"` to `[dependencies]` if absent (check first: `grep '^hex' Cargo.toml`); otherwise format with `{:x}` via `format!("{:x}", hasher.finalize())`.

- [ ] **Step 4: Run tests** — PASS, plus `cargo test --all-features` to catch signature-generation call sites.

- [ ] **Step 5: Commit**

```bash
git add src/security/zero_knowledge.rs Cargo.toml tests/security_honesty_tests.rs
git commit -m "fix(security): replace fake length-arithmetic hash with SHA-256"
```

### Task 0.4: ZK verify fails closed; remove string-matching "verification"

**Files:**
- Modify: `src/security/zero_knowledge.rs:607-660` (`verify_proof`, both cfg branches)
- Test: `tests/security_honesty_tests.rs`

**Interfaces:**
- Produces: with `zero-knowledge-proofs` **off**: `verify_proof` returns `Err(FeatureDisabled)`. With it **on**: the format-check placeholder (`len > 10 && contains("bellman_proof")`) is replaced by an explicit `Err(MemoryError::feature_disabled("zero-knowledge-proofs(real-verify)", "verify_proof"))` **until Phase 4 lands real Groth16 verification** — an honest error beats a fake `Ok(true)`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(all(feature = "security", not(feature = "zero-knowledge-proofs")))]
#[tokio::test]
async fn zk_verify_errors_when_feature_disabled() {
    // Build a proof via the fallback generator, then confirm verify refuses.
    // Use the module's public manager API (ZeroKnowledgeManager or equivalent
    // exported type in src/security/zero_knowledge.rs).
    // Assert: verify returns Err containing "zero-knowledge".
}
```

Fill in with the module's actual public constructor (read the top of `zero_knowledge.rs` for the exported manager type before writing; the test body must construct, prove, verify, and `expect_err`).

- [ ] **Step 2: Run to verify fail** — currently returns `Ok(bool)`.

- [ ] **Step 3: Implement** — replace both branch bodies:

```rust
        #[cfg(feature = "zero-knowledge-proofs")]
        {
            let statement_hash = self.hash_statement(statement)?;
            if proof.statement_hash != statement_hash {
                return Ok(false);
            }
            // Real Groth16 verification lands in Phase 4 (task 4.2). Until then,
            // refuse rather than approve on a format check.
            Err(MemoryError::feature_disabled(
                "zero-knowledge-proofs(real-verify)",
                "verify_proof",
            ))
        }

        #[cfg(not(feature = "zero-knowledge-proofs"))]
        {
            let _ = statement;
            Err(MemoryError::feature_disabled(
                "zero-knowledge-proofs",
                "verify_proof",
            ))
        }
```

Also delete the hardcoded `Scalar::from(42u64)` secret path in proof generation if it is now unreachable; if generation stays, it must carry a doc comment stating verification is not yet implemented.

- [ ] **Step 4: Run** — `cargo test --all-features`; fix any security-manager call sites that assumed `Ok(true)` (they must propagate).

- [ ] **Step 5: Commit**

```bash
git add src/security/zero_knowledge.rs tests/security_honesty_tests.rs
git commit -m "fix(security)!: ZK verification fails closed instead of string-matching"
```

### Task 0.5: Differential privacy uses OsRng

**Files:**
- Modify: `src/security/privacy.rs` (`NoiseGenerator`, ~line 390; Laplace/exponential mechanisms ~line 294)
- Test: `tests/security_honesty_tests.rs`

- [ ] **Step 1: Failing test** — statistical smoke test: generate 1000 Laplace noise samples with scale 1.0 via the public noise API; assert nonzero variance and that mean is within ±0.5 (catches "returns 0" stubs).
- [ ] **Step 2: Verify fail** (or verify current behavior — if it passes because noise already works, keep the test and continue; only the RNG swap below is required).
- [ ] **Step 3: Implement** — replace `rand::thread_rng()`/`StdRng` in noise generation with `rand::rngs::OsRng`; delete the "In production, use a cryptographically secure RNG" comment.
- [ ] **Step 4: Run** — `cargo test --features security`.
- [ ] **Step 5: Commit** — `git commit -m "fix(security): differential-privacy noise from OsRng"`.

---

# PHASE 1 — Delete Dead Weight

### Task 1.1: Delete unreachable `src/optimization/` and `src/scalability/`

**Files:**
- Delete: `src/optimization/` (1,687 LOC), `src/scalability/` (1,780 LOC)
- Verify-first: neither appears in `src/lib.rs` module declarations.

- [ ] **Step 1: Confirm unreachable** — `grep -rn "crate::optimization\|crate::scalability\|mod optimization\|mod scalability" src/ tests/ benches/ examples/ | grep -v "^src/optimization\|^src/scalability"`. Expected: no hits. If hits appear, stop and report instead of deleting.
- [ ] **Step 2: Delete** — `git rm -r src/optimization src/scalability`
- [ ] **Step 3: Full gate** — `cargo build --all-features && cargo test && cargo clippy --all-targets --all-features -- -D warnings`. Expected: green.
- [ ] **Step 4: Commit** — `git commit -m "chore: delete unreachable optimization/ and scalability/ modules (3.5k LOC dead code)"`

### Task 1.2: Delete orphaned `tests/integration/` placeholder suite

**Files:**
- Delete: `tests/integration/comprehensive_test_suite.rs`, `tests/integration/jupyter_integration_tests.rs`, `tests/integration/jupyter_kernel_tests.rs`, `tests/integration/magic_commands_tests.rs` (the `assert!(true)` placeholders, never wired into any `[[test]]` target)

- [ ] **Step 1: Confirm orphaned** — `grep -rn "integration/" Cargo.toml; grep -rn "mod integration" tests/*.rs`. Expected: no `[[test]]` entry or mod include references these files. Keep any file that IS referenced.
- [ ] **Step 2: Delete confirmed orphans** — `git rm` them.
- [ ] **Step 3: Run** — `cargo test` still green.
- [ ] **Step 4: Commit** — `git commit -m "chore(tests): remove orphaned placeholder integration tests (assert!(true) bodies)"`

### Task 1.3: Consolidate duplicate visualization modules

**Files:**
- Keep: `src/integrations/visualization.rs` (feature-gated behind `visualization`, real plotters backend)
- Delete or fold: `src/analytics/visualization.rs` (1,015 LOC overlap)

- [ ] **Step 1: Map usage** — `grep -rn "analytics::visualization\|analytics::vis" src/ tests/ examples/`. List every consumer.
- [ ] **Step 2: For each consumer, port it** to the integrations module's API (or, if consumers only use types not rendering, move those types into `src/analytics/mod.rs`). Show each edit as a normal Edit with imports updated.
- [ ] **Step 3: Delete** `src/analytics/visualization.rs`; remove its `mod` declaration in `src/analytics/mod.rs`.
- [ ] **Step 4: Gate** — `cargo build --all-features && cargo test --all-features` green.
- [ ] **Step 5: Commit** — `git commit -m "refactor: consolidate visualization into integrations module"`

### Task 1.4: Delete `tests/real_performance_measurement_tests.rs` sleep-benchmarks

- [ ] **Step 1:** Confirm the suite measures `tokio::time::sleep` (lines 36, 78, 123, 374, 434): read the file; any test that measures real code paths gets moved to `tests/performance_tests.rs` instead of deleted.
- [ ] **Step 2:** `git rm tests/real_performance_measurement_tests.rs`; remove its `[[test]]` entry (`performance_suite`) from Cargo.toml if present.
- [ ] **Step 3:** `cargo test` green. Commit: `git commit -m "chore(tests): remove sleep-based fake performance suite"`

---

# PHASE 2 — Core Write-Path Correctness

### Task 2.1: Stop swallowing subsystem errors in `store()`

**Files:**
- Modify: `src/lib.rs:299-371` (`AgentMemory::store`)
- Create: `src/memory/store_result.rs`
- Test: `tests/store_degradation_tests.rs` (new)

**Interfaces:**
- Produces:

```rust
/// Which optional subsystems failed during a store; storage itself succeeded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoreDegradations {
    pub temporal: Option<String>,
    pub analytics: Option<String>,
    pub knowledge_graph: Option<String>,
    pub advanced_management: Option<String>,
    pub embeddings: Option<String>,
}
impl StoreDegradations {
    pub fn is_clean(&self) -> bool { /* all None */ }
}
```

- New method `pub async fn store_with_report(&mut self, key: &str, value: &str) -> Result<StoreDegradations>`; existing `store()` becomes a thin wrapper that calls it, logs each degradation at `warn`, and returns `Ok(())` only if storage succeeded (behavior-compatible), so no caller breaks.

- [ ] **Step 1: Write the failing test**

Create `tests/store_degradation_tests.rs`:

```rust
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::test]
async fn store_with_report_returns_clean_report_on_happy_path() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("default config constructs");
    let report = memory
        .store_with_report("k1", "v1")
        .await
        .expect("storage write succeeds");
    assert!(report.is_clean(), "no subsystem should degrade: {report:?}");
    // and the write is durable:
    let got = memory.retrieve("k1").await.expect("retrieve ok");
    assert!(got.is_some());
}
```

- [ ] **Step 2: Verify fail** — `cargo test --test store_degradation_tests` → no method `store_with_report`.
- [ ] **Step 3: Implement** — add `src/memory/store_result.rs` with the struct above (`is_clean` returns true iff every field is `None`); `pub mod store_result;` in `src/memory/mod.rs` and re-export from `lib.rs`. Rewrite `store` fan-out: each `let _ = subsystem_call` becomes

```rust
        if let Some(ref mut tm) = self.temporal_manager {
            if let Err(e) = tm.track_memory_change(&entry, change_type).await {
                tracing::warn!(error = %e, "temporal tracking degraded during store");
                degradations.temporal = Some(e.to_string());
            }
        }
```

(same pattern for analytics, knowledge_graph, advanced_manager, embeddings — five blocks, each setting its own field). `store()` calls `store_with_report` and discards the report after logging.

- [ ] **Step 4: Run** — `cargo test` full suite green.
- [ ] **Step 5: Commit** — `git commit -m "feat(core): surface subsystem degradations from store instead of swallowing"`

### Task 2.2: Make state a cache, storage the source of truth

**Files:**
- Modify: `src/lib.rs` `store` (write order) and `retrieve` (`:375-472`)
- Test: `tests/store_degradation_tests.rs`

**Interfaces:** unchanged public API; internal ordering contract: **storage write first**, then state insert. On storage `Err`, state must not contain the entry.

- [ ] **Step 1: Failing test** — use a failing-storage double. Check `src/memory/storage/` for an injectable backend; if none exists, add `#[cfg(feature = "test-utils")] pub struct FailingStorage;` in `src/memory/storage/mod.rs` implementing `Storage` with every write method returning `Err(MemoryError::storage("injected failure"))` and reads delegating to an inner `MemoryStorage`. Test: construct `AgentMemory` with it (via the existing storage-injection point in `AgentMemory::new`/config — read `src/lib.rs:84-150` for how `storage` is set and use that), call `store`, assert `Err`, then assert `memory.retrieve("k").await` (which reads state first) returns `Ok(None)` — i.e. state was not polluted.
- [ ] **Step 2: Verify fail** — today `state.add_memory` happens before `storage.store`, so state holds the entry after a storage error.
- [ ] **Step 3: Implement** — reorder in `store_with_report`:

```rust
        self.storage.store(&entry).await?;   // source of truth first
        self.state.add_memory(entry.clone()); // cache second
```

- [ ] **Step 4: Run suite** — green.
- [ ] **Step 5: Commit** — `git commit -m "fix(core): write storage before state cache so failures cannot diverge them"`

### Task 2.3: Non-destructive checkpoint restore

**Files:**
- Modify: `src/lib.rs:496-515` (`restore_checkpoint` — currently `clear()`s storage then repopulates)
- Test: `tests/checkpoint_restore_tests.rs` (new)

- [ ] **Step 1: Failing test** — store 3 entries; checkpoint; store 2 more; restore the checkpoint via a `FailingStorage` variant whose `store` fails **after** `clear` has been called (flag-triggered). Assert the error is returned AND a subsequent `retrieve` of the original 3 keys still succeeds (i.e. restore did not destroy data on failure).
- [ ] **Step 2: Verify fail** — current impl clears then repopulates; injected failure mid-repopulate loses data.
- [ ] **Step 3: Implement** — restore into a staging pass: (1) read checkpoint state, (2) write **all** checkpoint entries to storage first (upserts), (3) compute keys present in storage but absent from checkpoint via `storage.list_keys()` and delete only those, (4) replace `self.state` last. No `clear()`. Any step erroring leaves prior data intact (upserts are idempotent; deletes happen only after all writes succeed).
- [ ] **Step 4: Run** — `cargo test --test checkpoint_restore_tests` and full suite green.
- [ ] **Step 5: Commit** — `git commit -m "fix(core): checkpoint restore is upsert-then-prune, no destructive clear"`

---

# PHASE 3 — Real Retrieval on the Live Path

### Task 3.1: Route `AgentMemory::search` through the retrieval pipeline

**Files:**
- Modify: `src/lib.rs:476-488` (`search`), `src/lib.rs:84-150` (add pipeline field)
- Consume: `src/memory/retrieval/pipeline.rs` — it already exposes `pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>>` (line 272)
- Test: `tests/retrieval_quality.rs` (extend existing)

**Interfaces:**
- `AgentMemory::search` keeps its exact signature. Internally: pipeline when embeddings are enabled and populated, storage substring search as explicit fallback.

- [ ] **Step 1: Failing test** — in `tests/retrieval_quality.rs` add:

```rust
#[tokio::test]
async fn search_ranks_semantic_match_above_substring_noise() {
    let mut memory = AgentMemory::new(MemoryConfig {
        enable_embeddings: true,
        ..Default::default()
    })
    .await
    .expect("config with embeddings constructs");
    memory.store("doc_cat", "felines are small domesticated carnivorous mammals").await.unwrap_or_else(|e| panic!("store: {e}"));
    memory.store("doc_noise", "the word cat appears here but this text is about tax law").await.unwrap_or_else(|e| panic!("store: {e}"));
    let results = memory.search("cat animal pet", 2).await.expect("search ok");
    assert!(!results.is_empty());
    // With TF-IDF/embedding ranking, the feline doc must rank at/above the noise doc.
    assert_eq!(results[0].entry().key(), "doc_cat", "pipeline ranking should beat substring: {results:?}");
}
```

(Adjust accessor names to `MemoryFragment`'s real API — check `src/memory/types.rs` before writing; use whatever getter exposes the key.)

- [ ] **Step 2: Verify fail** — substring search has no ranking; likely returns noise first or misses "felines" entirely for query "cat".
- [ ] **Step 3: Implement** — construct the retrieval pipeline in `AgentMemory::new` when `enable_embeddings` is set (wiring the `EmbeddingManager` the struct already owns — read `pipeline.rs:100-270` for the builder and its stage config, use `RetrievalConfig` semantic defaults from `mod.rs:132`); in `search`, call pipeline when present, else storage search. Keep the input validation exactly as-is.
- [ ] **Step 4: Run** — `cargo test --test retrieval_quality` and full suite green.
- [ ] **Step 5: Commit** — `git commit -m "feat(retrieval): route AgentMemory::search through hybrid retrieval pipeline"`

### Task 3.2: Wire HNSW ANN into `count_related_memories` (kill the O(n) brute force)

**Files:**
- Modify: `src/memory/management/mod.rs:115-180` (BFS/related counting) and the call path that currently brute-forces similarity
- Consume: `src/memory/indexing/` HNSW index (already built, unwired)
- Test: `tests/indexed_retrieval_tests.rs` (extend)

- [ ] **Step 1: Failing test** — behavioral, not perf: insert 200 entries with deterministic embeddings (test-utils embedder); assert `count_related_memories` for a probe returns the same neighbor set as brute-force cosine over the 200 (compute expected in the test) — proving ANN path is wired AND correct. Mark a companion `#[ignore]`d perf test comparing timing at 10k entries.
- [ ] **Step 2: Verify fail** (wire a temporary assertion that the ANN index was consulted — e.g. index hit-counter exposed under `test-utils`).
- [ ] **Step 3: Implement** — replace the linear scan with an index query (`k` = config threshold), falling back to brute force only when index size < 100 (document the constant).
- [ ] **Step 4: Run + commit** — `git commit -m "feat(indexing): use HNSW index for related-memory counting"`

---

# PHASE 4 — Real Security (sub-plan required)

Author `docs/superpowers/plans/2026-XX-XX-phase4-real-security.md` when Phase 0–3 are merged. Hard scope for that plan:

1. **Real Groth16 verify (task 4.1–4.3):** replace demo circuit (`user_secret^2`, hardcoded `Scalar::from(42)`) with a Poseidon-or-SHA-based knowledge-of-preimage circuit over bls12_381; persist proving/verifying keys; `verify_proof` deserializes the proof and calls `bellman::groth16::verify_proof` with a prepared verifying key. Acceptance: round-trip prove/verify passes; a bit-flipped proof **fails**; the Task 0.4 fail-closed error is removed.
2. **TFHE required for HE (4.4):** `homomorphic-encryption` becomes a hard requirement of the `security` feature's HE API (already fail-closed from 0.2); add ciphertext round-trip + additive-operation tests; document precision loss of the ×1000 u32 scaling or replace with `FheInt64`.
3. **Differential privacy correctness (4.5):** property tests (add `proptest` dev-dep) for ε-budget accounting: repeated queries deplete budget; Laplace scale = sensitivity/ε verified statistically.
4. **Key management (4.6):** keys zeroized on drop (`zeroize` crate), never `Debug`-printed; audit `md5` usage — if any security path touches it, replace with sha2; if only non-security fingerprinting, document that at the use site.

Each item is its own TDD task set in the sub-plan with the same step granularity as Phases 0–3.

# PHASE 5 — Management De-stubbing (sub-plan required)

Scope for `phase5-management-destub.md`:

1. Inventory the ~136 placeholder sites: `grep -rn "Placeholder\|placeholder\|In a real implementation\|for now\|simplified" src/memory/management src/memory/consolidation` → checklist in the sub-plan.
2. For each: **implement, delete, or demote.** Decision rule: (a) if a real data source exists (e.g. `related_count = 5` → the Task 3.2 ANN counter), implement it; (b) if the feature has no consumer in the public API or tests, delete it; (c) if it's a heuristic that's genuinely useful but oversold, keep it and rename/doc it as a heuristic (e.g. `merge_content`'s concatenation fallback stays but the docstring stops calling it "intelligent").
3. Kill the redundant `MemoryManager` (`management/mod.rs:36-45`, three unused `_`-prefixed fields) in favor of `AdvancedMemoryManager`.
4. `lifecycle.rs:1162-1421` six stub methods: implement archival/compression against the real `Storage` trait (backends already support it) — these are the highest-value stubs to make real.
5. Acceptance: `grep -c "In a real implementation" src/` reaches 0; every remaining "simplified" marker has a doc comment justifying the simplification.

# PHASE 6 — Distributed: Real Consensus or Descope (sub-plan required; decision gate)

The hand-rolled "simplified consensus" cannot ship as-is. The sub-plan starts with a decision task presenting two costed options to the maintainer:

- **Option A (recommended): descope.** Move `src/distributed/` behind a `distributed-experimental` feature, rename docs accordingly, delete the stubbed realtime/coordination paths ("disabled for now" at coordination.rs:38,73,119). Cost: days. The crate's value is the memory core, not a homegrown Raft.
- **Option B: adopt `openraft`** (the commented-out `raft` dep in Cargo.toml shows this was once intended). Implement `RaftStorage` over the existing `Storage` trait; consensus tests under network partition via `madsim` or in-process transport. Cost: weeks.

Acceptance either way: no code path labeled "simplified for testing" is reachable from a non-`experimental` feature.

# PHASE 7 — Quality-Gate Re-tightening

Sequential lint re-enable, one commit each, in this order (each: flip in Cargo.toml `[lints]`, `cargo clippy --all-targets --all-features -- -D warnings`, fix all findings, commit):

1. `dead_code = "warn"` (after Phase 1 deletions this should be tractable; delete or `#[allow]`-with-justification each finding)
2. `unused_imports`, `unused_variables`, `unused_mut` → default
3. `clippy::complexity` group → default
4. `clippy::perf` group → default
5. `clippy::style` group → default
6. Replace grep gates: `unwrap_used = "deny"`, `panic = "deny"` (tests keep `#[allow]` at mod level); delete `check-unwrap`/`check-println` steps from `quality.yml` once the lints gate CI
7. `.expect()` audit on default-feature paths: every expect message must state an invariant ("config validated at construction") — mechanical sweep of the 560 sites, prioritized: `src/lib.rs`, `src/memory/storage/`, `src/memory/knowledge_graph/`
8. CI: make coverage thresholds consistent (`fail_ci_if_error: true` in both workflows), add `cargo run --example <each wired example>` smoke job, make `check-todos` blocking once count is 0.

# PHASE 8 — Honest Docs & Validated Benchmarks

1. **README rewrite:** delete "zero warnings / production-ready / 1755+ warnings eliminated / 100K+ ops/sec / 161 tests" claims; add per-module maturity table (`stable` — storage, knowledge graph, embeddings / `beta` — retrieval, analytics / `experimental` — security, distributed, multimodal, WASM, mobile). Quick-start example is doctested (`cargo test --doc` gates it).
2. **Benchmarks with numbers:** criterion benches for store/retrieve/search at 1k/10k/100k entries on the real paths (post-Phase 3); publish results table in `docs/performance.md`; README cites that file, nothing else.
3. **Reconcile test counts:** delete `tests/test_config.toml` stale counts; CI badge shows live counts.
4. **Delete stale root files:** `PR_DESCRIPTION.md`, `PR_PHASE3_*.md` (4 files) — historical PR bodies don't belong in the repo root; fold anything still-true into CHANGELOG.
5. **docs/ audit:** for each of the 20 guides, verify code samples compile (move runnable ones into `examples/`); fix or delete per `documentation-accuracy-audit.md` findings; then delete the audit file itself once its items are closed.
6. **CHANGELOG:** cut `v0.2.0` describing the remediation honestly ("removed simulated security fallbacks — breaking", "search now semantic", etc.).

---

## Self-Review Notes

- **Coverage:** every finding from the analysis maps to a task: fake ZK/HE/DP → 0.2–0.5 + Phase 4; dead modules/tests/dup viz/sleep-benches → 1.1–1.4; error swallowing/dual-write/destructive restore → 2.1–2.3; substring search + unwired HNSW → 3.1–3.2; 136 placeholders + god-object cruft → Phase 5; simplified consensus → Phase 6; hollow lints + 560 expects + CI gaps → Phase 7; README/docs/benchmarks/PR files → Phase 8.
- **Type consistency:** `MemoryError::feature_disabled` (0.1) is consumed by 0.2/0.4; `StoreDegradations` (2.1) is consumed by 2.2's reorder; pipeline `search(&self, query, limit)` signature confirmed at `pipeline.rs:272`.
- **Known unknowns flagged in-task** (constructor visibility in 0.2, `MemoryFragment` accessors in 3.1, storage injection point in 2.2) each carry an explicit "read X first" instruction rather than an invented API.
