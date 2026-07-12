# Phase 4: Real Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. **DO NOT EXECUTE until the maintainer resolves the Decision Gates below.**

**Goal:** Replace every fail-closed security stub from Phase 0 with a real implementation (or an explicit, documented descope), so that `security`-feature claims are cryptographically true.

**Architecture:** Groth16 (bellman/bls12_381) knowledge-of-preimage proofs with persisted keys and real verification; TFHE as the only homomorphic path; differential privacy with property-tested ε-accounting; zeroized key material.

**Tech Stack:** bellman 0.14, bls12_381, tfhe, sha2, zeroize, proptest (dev), rand OsRng.

## Decision Gates (maintainer input required before execution)

1. **ZK statement design.** Phase 0 review found `generate_access_proof`/`generate_content_proof` hash an internally-timestamped statement no external caller can reproduce — real verification can never match statements as currently designed. Options:
   - **A (recommended):** caller supplies the statement (or its nonce/timestamp) so verify is reproducible; breaking API change to proof-generation signatures.
   - **B:** proof carries the timestamped statement; verifier checks proof-vs-carried-statement plus a caller-supplied predicate on freshness. Weaker binding, no API break.
2. **Hash-in-circuit choice.** Knowledge-of-preimage circuit needs a SNARK-friendly hash. Options: Poseidon over BLS12-381 scalar field via a vetted gadget crate (**recommended**; small circuit, needs new dependency) vs SHA-256 gadget (huge circuit, slow proving, no new dep). Affects proving time by ~100x.
3. **Homomorphic scope.** TFHE FheUint32 supports the encrypt/decrypt/sum/average ops. `homomorphic_search`/`homomorphic_similarity` over encrypted vectors are research-grade problems — recommend **descoping them permanently** (keep fail-closed error, document as unsupported) rather than shipping a slow toy. Also: `homomorphic_count` currently returns plaintext counts — decide fail-closed vs documented-plaintext-metadata.
4. **Precision contract for HE.** Current scaling is `(value * 1000).abs() as u32` — silently drops sign and truncates. Options: FheInt64 with documented fixed-point scale (recommended) vs keep u32 and document unsigned-millis contract.

## Global Constraints

- All Phase 0 fail-closed behavior stays until the real implementation replacing it is merged in the same task.
- Security invariant unchanged: disabled feature ⇒ `MemoryError::FeatureDisabled`, never fake math.
- No `.unwrap()` in src/; no println!; `.expect()`s state invariants; unsafe forbidden.
- Gates per task: fmt; clippy `--all-targets --features "security zero-knowledge-proofs" -- -D warnings`; the task's test files; `cargo test --lib`. CI covers `--all-features`.
- Tests to RESTORE when real verify lands (they were weakened to expect_err in Phase 0 with "revert in Phase 4" comments): tests/zero_knowledge_tests.rs bellman verify tests → `assert!(is_valid)`; performance test `total_proofs_verified >= 5`; tests/phase4_security_tests.rs ZK tests.

---

### Task 4.1: Persisted Groth16 parameters and key lifecycle

**Files:** Modify src/security/zero_knowledge.rs (key generation/storage); Test tests/zk_groth16_tests.rs (new).
**Scope:** Generate Groth16 parameters once per circuit shape (seeded by OsRng), serialize proving/verifying keys (bellman's `Parameters::write`/`read`), store under a configurable key-store path (default in-memory for tests), expose `prepared_verifying_key()`. TDD: round-trip serialize/deserialize equality; key generation is deterministic given injected RNG only in tests.

### Task 4.2: Real knowledge-of-preimage circuit + verify

**Files:** Modify src/security/zero_knowledge.rs (circuit, generate, verify); Test tests/zk_groth16_tests.rs.
**Scope:** Replace the `user_secret^2` demo circuit and hardcoded `Scalar::from(42)`: circuit proves knowledge of preimage x with H(x) = public input (hash per Decision Gate 2). `generate_proof` proves over the witness derived from the statement (per Decision Gate 1 design); `verify_proof` deserializes proof bytes and calls `groth16::verify_proof` with the prepared key — the Phase 0 fail-closed error is REMOVED here. Acceptance (all must be tests): (a) honest round-trip verifies true; (b) bit-flipped proof fails; (c) wrong public input fails; (d) the Phase 0 weakened tests are restored to `assert!(is_valid)` and pass; (e) proof from key-set A fails under key-set B.

### Task 4.3: Statement reproducibility fix

**Files:** Modify src/security/zero_knowledge.rs (statement construction per Decision Gate 1); tests updated; delete `align_statement_hash` test helper (crutch from Phase 0 review).
**Acceptance:** an external verifier holding only (statement, proof, verifying key) verifies without reaching into generation internals.

### Task 4.4: TFHE-only homomorphic path

**Files:** Modify src/security/encryption.rs; Test tests/homomorphic_encryption_tests.rs.
**Scope:** Under `homomorphic-encryption`: ciphertext round-trip tests, additive-op test (encrypted sum equals plaintext sum within documented precision), precision contract per Decision Gate 4. Search/similarity/count per Decision Gate 3. Feature-off path keeps FeatureDisabled (already done in Phase 0).

### Task 4.5: Differential privacy correctness

**Files:** Modify src/security/privacy.rs; Test tests/security_honesty_tests.rs + new proptest suite tests/dp_property_tests.rs; add proptest dev-dependency.
**Scope:** ε-budget accounting property tests (repeated queries deplete budget; exhausted budget refuses); Laplace scale = sensitivity/ε verified statistically; strengthen Phase 0 smoke test to assert sample variance ≈ 2·scale² (tolerance ±25% at n=5000); exponential mechanism weights verified on a 3-outcome toy distribution.

### Task 4.6: Key hygiene

**Files:** Modify src/security/{encryption,zero_knowledge,key_management}.rs (locate real file names first); Cargo.toml (zeroize).
**Scope:** Key material zeroized on drop (`zeroize::Zeroizing` wrappers); no `Debug` derive exposes key bytes (manual impl redacts); audit `md5` usage (`grep -rn "md5" src/`) — replace with sha2 if any security path touches it, else document fingerprint-only use at the use site.

---

## Execution order & sizing

4.1 → 4.2 → 4.3 are strictly sequential (same file, same abstractions). 4.4, 4.5, 4.6 are independent of the ZK chain and each other. Estimated: 4.1-4.3 are the hard core (days, and proving-time performance needs watching); 4.4-4.6 are each half-day scale.

## Out of scope (recorded for later phases)

- RRF fusion bug fix, ANN stale-vector invalidation, storage verbatim-phrase search → Phase 5 (see .superpowers/sdd/progress.md notes).
- Distributed consensus → Phase 6 decision gate.
