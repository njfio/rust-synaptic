# Phase 4: Real Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Decision gates below are RESOLVED (2026-07-12); execution authorized.

**Goal:** Replace every fail-closed security stub from Phase 0 with a real implementation (or an explicit, documented descope), so that `security`-feature claims are cryptographically true.

**Architecture:** Groth16 (bellman/bls12_381) knowledge-of-preimage proofs with persisted keys and real verification; TFHE as the only homomorphic path; differential privacy with property-tested ε-accounting; zeroized key material.

**Tech Stack:** bellman 0.14, bls12_381, tfhe, sha2, zeroize, proptest (dev), rand OsRng.

## Decision Gates — RESOLVED by maintainer 2026-07-12 (do not re-ask)

1. **ZK statement design: Option A.** Caller supplies the statement/nonce; breaking API change to proof-generation signatures approved. Delete `align_statement_hash` once external verification round-trips.
2. **Hash-in-circuit: Poseidon** via a well-maintained bls12_381 gadget crate; justify crate choice in the task report and pin the version.
3. **Homomorphic scope: descope search/similarity permanently** (fail-closed stays, documented unsupported); **`homomorphic_count` becomes fail-closed too.** Sum/average/encrypt/decrypt get real TFHE implementations.
4. **HE precision: FheInt64 fixed-point** with documented scale; signed values round-trip exactly at the documented precision, with sign-preservation tests.
5. **(Added from final sweep) Real authentication — new Task 4.7:** argon2 password verification, TOTP-based MFA (replace "any 6 digits"), policy_engine.

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

> **Execution notes (2026-07-12, tasks 4.1-4.3 done):** Decision Gate 2 is
> satisfied with `neptune` 13.0.0 (Filecoin's Poseidon over BLS12-381; same
> ff/group 0.13 line as bellman 0.14). neptune's circuit gadget targets
> `bellpepper-core` 0.4, not bellman, so a record-and-replay constraint-system
> adapter (`BellpepperRecorder` in zero_knowledge.rs) bridges the gadget into
> bellman Groth16 — no dependency upgrades needed. The proof envelope now
> carries its Groth16 public input (`ZKProof.public_inputs`, the Poseidon
> digest). `SecurityManager::decrypt_memory` reads the caller statement from
> the `zk_access_statement` context attribute and checks it matches the entry
> and user. Four pre-existing security_suite failures (HE feature-off,
> DP-MFA) remain for tasks 4.4/4.7.

### Task 4.4: TFHE-only homomorphic path

**Files:** Modify src/security/encryption.rs; Test tests/homomorphic_encryption_tests.rs.
**Scope:** Under `homomorphic-encryption`: ciphertext round-trip tests, additive-op test (encrypted sum equals plaintext sum within documented precision), precision contract per Decision Gate 4. Search/similarity/count per Decision Gate 3. Feature-off path keeps FeatureDisabled (already done in Phase 0).

> **Execution note (2026-07-12, task 4.4 done):** Real-API divergence: tfhe
> 0.7 panics at runtime without a seeder feature, so Cargo.toml's tfhe
> dependency gained `seeder_unix`. FheInt64 fixed-point uses
> `FIXED_POINT_SCALE = 1_000_000` (6 decimal digits, exact round-trip at
> that granularity; sign-preserving). Homomorphic ops require
> `tfhe::set_server_key` on the executing thread — installed inside
> `homomorphic_sum`. Average uses TFHE encrypted signed division (truncated
> at SCALE granularity when the sum does not divide evenly). Decision Gate 3
> applied: search/similarity/count fail closed in all builds, documented as
> unsupported by design.

### Task 4.5: Differential privacy correctness

**Files:** Modify src/security/privacy.rs; Test tests/security_honesty_tests.rs + new proptest suite tests/dp_property_tests.rs; add proptest dev-dependency.
**Scope:** ε-budget accounting property tests (repeated queries deplete budget; exhausted budget refuses); Laplace scale = sensitivity/ε verified statistically; strengthen Phase 0 smoke test to assert sample variance ≈ 2·scale² (tolerance ±25% at n=5000); exponential mechanism weights verified on a 3-outcome toy distribution.

### Task 4.6: Key hygiene

**Files:** Modify src/security/{encryption,zero_knowledge,key_management}.rs (locate real file names first); Cargo.toml (zeroize).
**Scope:** Key material zeroized on drop (`zeroize::Zeroizing` wrappers); no `Debug` derive exposes key bytes (manual impl redacts); audit `md5` usage (`grep -rn "md5" src/`) — replace with sha2 if any security path touches it, else document fingerprint-only use at the use site.

### Task 4.7: Real authentication and authorization

**Files:** Modify src/security/access_control.rs (password/API-key/MFA verification), src/security/policy_engine.rs (condition evaluation); Cargo.toml (argon2, a TOTP crate); Test tests/phase4_security_tests.rs (MFA tests) + tests/auth_tests.rs (new).
**Scope:**
- Password verification via `argon2` (PHC-string verify), replacing `password.len() >= 8` at access_control.rs:223. Store/verify against an argon2 hash; provide a hashing helper for provisioning.
- API-key verification replacing `starts_with("sk-")` at :230 — constant-time compare against a stored hash (sha256 of the key), not a prefix check.
- MFA: TOTP (RFC 6238) verification replacing `verify_mfa_token` accepting any 6 digits at :266 — verify against a per-user shared secret with a ±1 time-step window; justify the TOTP crate + pin version in the report.
- policy_engine.rs: remove the catch-all `_ => Ok(true)` in condition evaluation — unknown conditions **deny** (`Ok(false)` or an explicit error). Audit every match arm for other permissive defaults.
- Acceptance tests: correct password/token/key pass; wrong ones fail; unknown policy condition denies; the previously-failing MFA tests in phase4_security_tests.rs (documented in task-hygiene-report.md) turn green. TDD throughout.
- Commit: "feat(security)!: real argon2/TOTP authentication and deny-by-default policy".

---

## Execution order & sizing

4.1 → 4.2 → 4.3 are strictly sequential (same file, same abstractions). 4.4, 4.5, 4.6, 4.7 are independent of the ZK chain and each other. Estimated: 4.1-4.3 are the hard core (proving-time performance needs watching); 4.4-4.7 are each ~half-day scale.

## Out of scope (recorded for later phases)

- RRF fusion bug fix, ANN stale-vector invalidation, storage verbatim-phrase search → Phase 5 (see .superpowers/sdd/progress.md notes).
- Distributed consensus → Phase 6 decision gate.
