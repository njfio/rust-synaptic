//! Groth16 key lifecycle and Poseidon knowledge-of-preimage proof tests.
//!
//! Phase 4 (tasks 4.1-4.3): real zk-SNARK parameter persistence, proving,
//! and verification over the BLS12-381 scalar field. Verification is bound
//! to registered prover commitments: the verifier derives every Groth16
//! public input from its own trusted state, never from the proof envelope.

#![cfg(feature = "zero-knowledge-proofs")]

use std::error::Error;
use synaptic::security::zero_knowledge::ZeroKnowledgeManager;
use synaptic::security::SecurityConfig;

/// Task 4.1: Groth16 parameters serialize and deserialize to an identical
/// byte representation, so keys can be persisted and reloaded.
#[tokio::test]
async fn groth16_parameters_round_trip_bytes() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let manager = ZeroKnowledgeManager::new(&config).await?;

    let params_bytes = manager.groth16_parameter_bytes()?;
    assert!(
        !params_bytes.is_empty(),
        "serialized Groth16 parameters must not be empty"
    );

    // Reconstruct a second manager from the persisted parameters and confirm
    // the round trip is lossless.
    let restored = ZeroKnowledgeManager::from_groth16_parameter_bytes(&config, &params_bytes)?;
    let restored_bytes = restored.groth16_parameter_bytes()?;
    assert_eq!(
        params_bytes, restored_bytes,
        "Groth16 parameter serialization must round-trip byte-identically"
    );

    Ok(())
}

/// Task 4.1: The verifying key is exposed separately (for external
/// verifiers) and survives the same round trip.
#[tokio::test]
async fn groth16_verifying_key_round_trip() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let manager = ZeroKnowledgeManager::new(&config).await?;

    let vk_bytes = manager.verifying_key_bytes()?;
    assert!(
        !vk_bytes.is_empty(),
        "verifying key bytes must not be empty"
    );

    let params_bytes = manager.groth16_parameter_bytes()?;
    let restored = ZeroKnowledgeManager::from_groth16_parameter_bytes(&config, &params_bytes)?;
    assert_eq!(
        vk_bytes,
        restored.verifying_key_bytes()?,
        "verifying key must be identical after parameter round trip"
    );

    Ok(())
}

/// Task 4.1: A configurable key-store path persists parameters on first
/// construction and reloads (rather than regenerates) them afterwards.
#[tokio::test]
async fn groth16_key_store_path_persists_parameters() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let dir = tempfile::tempdir()?;
    let key_path = dir.path().join("groth16-params.bin");

    let first = ZeroKnowledgeManager::new_with_key_store(&config, Some(&key_path)).await?;
    assert!(key_path.exists(), "key store file must be created");
    let first_bytes = first.groth16_parameter_bytes()?;

    // A second manager pointed at the same store must load the same keys
    // instead of generating fresh ones.
    let second = ZeroKnowledgeManager::new_with_key_store(&config, Some(&key_path)).await?;
    assert_eq!(
        first_bytes,
        second.groth16_parameter_bytes()?,
        "key store reload must yield identical parameters"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Task 4.2/soundness fix: Poseidon knowledge-of-registered-secret circuit
// ---------------------------------------------------------------------------

use synaptic::security::zero_knowledge::{AccessStatement, AccessType};
use synaptic::security::SecurityContext;

fn mfa_context(user: &str) -> SecurityContext {
    let mut context = SecurityContext::new(user.to_string(), vec!["admin".to_string()]);
    context.mfa_verified = true;
    context
}

fn statement(user: &str, key: &str) -> AccessStatement {
    AccessStatement {
        memory_key: key.to_string(),
        user_id: user.to_string(),
        access_type: AccessType::Read,
        timestamp: chrono::Utc::now(),
    }
}

/// (a) An honestly generated proof from a registered prover verifies TRUE.
#[tokio::test]
async fn honest_proof_round_trip_verifies() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt = statement("prover", "memory-1");

    let proof = manager.generate_access_proof(&stmt, &context).await?;

    let is_valid = manager.verify_access_proof(&proof, &stmt).await?;
    assert!(is_valid, "honest Groth16 proof must verify true");
    Ok(())
}

/// (b) A bit-flipped proof fails verification.
#[tokio::test]
async fn bit_flipped_proof_fails() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt = statement("prover", "memory-2");

    let mut proof = manager.generate_access_proof(&stmt, &context).await?;

    let mid = proof.proof_data.len() / 2;
    proof.proof_data[mid] ^= 0x01;

    let is_valid = manager.verify_access_proof(&proof, &stmt).await?;
    assert!(!is_valid, "bit-flipped proof must fail verification");
    Ok(())
}

/// (c) A proof claiming a different registered identity fails: the verifier
/// looks up that identity's commitment (a different public input) and the
/// pairing check fails.
#[tokio::test]
async fn wrong_public_input_fails() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    manager.register_prover("other-prover")?;
    let context = mfa_context("prover");
    let stmt = statement("prover", "memory-3");

    let mut proof = manager.generate_access_proof(&stmt, &context).await?;
    proof.prover_id = "other-prover".to_string();

    let is_valid = manager.verify_access_proof(&proof, &stmt).await?;
    assert!(
        !is_valid,
        "proof verified against another identity's commitment must fail"
    );
    Ok(())
}

/// (e) A proof generated under key-set A fails verification under key-set B.
#[tokio::test]
async fn proof_from_other_keyset_fails() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager_a = ZeroKnowledgeManager::new(&config).await?;
    let mut manager_b = ZeroKnowledgeManager::new(&config).await?;
    manager_a.register_prover("prover")?;
    // Give B the same commitment A registered, so the only difference left
    // is the Groth16 key set itself.
    let commitment = manager_a
        .registered_commitment_bytes("prover")
        .ok_or("commitment must exist after registration")?;
    manager_b.register_commitment("prover", &commitment)?;

    let context = mfa_context("prover");
    let stmt = statement("prover", "memory-4");

    let proof = manager_a.generate_access_proof(&stmt, &context).await?;

    assert!(
        manager_a.verify_access_proof(&proof, &stmt).await?,
        "sanity: proof must verify under its own key set"
    );
    assert!(
        !manager_b.verify_access_proof(&proof, &stmt).await?,
        "proof must not verify under an unrelated key set"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Soundness fix: verifier-derived public inputs (registered commitments)
// ---------------------------------------------------------------------------

/// Statement rebinding attack: copy an honest proof for statement A,
/// overwrite its statement_hash to match statement B, and verify against B.
/// The statement-bound public input is derived by the VERIFIER from B, so
/// the pairing check must fail even though the envelope hash matches.
#[tokio::test]
async fn statement_rebinding_rejected() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt_a = statement("prover", "memory-a");
    let stmt_b = statement("prover", "memory-b");

    let mut proof = manager.generate_access_proof(&stmt_a, &context).await?;

    // Attacker recomputes the (public) statement hash for B and rebinds the
    // envelope.
    let serialized_b = serde_json::to_string(&stmt_b)?;
    proof.statement_hash = synaptic::security::zero_knowledge::hash_content_for_test(&serialized_b);

    let is_valid = manager.verify_access_proof(&proof, &stmt_b).await?;
    assert!(
        !is_valid,
        "rebinding an honest proof to another statement must fail cryptographically"
    );
    Ok(())
}

/// Replay attack: an honest proof for statement A must not verify against
/// statement B (without any envelope tampering).
#[tokio::test]
async fn replayed_proof_for_other_statement_rejected() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt_a = statement("prover", "memory-a");
    let stmt_b = statement("prover", "memory-b");

    let proof = manager.generate_access_proof(&stmt_a, &context).await?;
    let is_valid = manager.verify_access_proof(&proof, &stmt_b).await?;
    assert!(!is_valid, "proof for statement A must not verify for B");
    Ok(())
}

/// From-scratch forgery: an attacker who shares the public CRS but has no
/// registered secret picks their own secret, builds a structurally valid
/// proof, and presents it under the victim's identity. The verifier derives
/// the expected commitment from ITS registration store, so the forgery must
/// fail.
#[tokio::test]
async fn forgery_without_registered_secret_rejected() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut verifier = ZeroKnowledgeManager::new(&config).await?;
    verifier.register_prover("victim")?;

    // Attacker clones the public parameters and registers their own secret
    // for the same identity in their own manager.
    let params = verifier.groth16_parameter_bytes()?;
    let mut attacker = ZeroKnowledgeManager::from_groth16_parameter_bytes(&config, &params)?;
    attacker.register_prover("victim")?;

    let context = mfa_context("victim");
    let stmt = statement("victim", "forged-memory");
    let forged = attacker.generate_access_proof(&stmt, &context).await?;

    // Sanity: the forgery is self-consistent for the attacker...
    assert!(attacker.verify_access_proof(&forged, &stmt).await?);
    // ...but the real verifier's registered commitment differs, so it fails.
    let is_valid = verifier.verify_access_proof(&forged, &stmt).await?;
    assert!(
        !is_valid,
        "a proof built without the registered secret must fail verification"
    );
    Ok(())
}

/// A proof claiming an identity with no registered commitment is rejected.
#[tokio::test]
async fn unregistered_prover_rejected() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    manager.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt = statement("prover", "memory-5");

    let mut proof = manager.generate_access_proof(&stmt, &context).await?;
    proof.prover_id = "ghost".to_string();

    let is_valid = manager.verify_access_proof(&proof, &stmt).await?;
    assert!(!is_valid, "unknown prover identity must be rejected");
    Ok(())
}

/// Generation without a registered secret must refuse (the prover has no
/// witness that satisfies the circuit against any registered commitment).
#[tokio::test]
async fn generation_requires_registration() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut manager = ZeroKnowledgeManager::new(&config).await?;
    let context = mfa_context("nobody");
    let stmt = statement("nobody", "memory-6");

    let result = manager.generate_access_proof(&stmt, &context).await;
    assert!(
        result.is_err(),
        "unregistered prover must not obtain proofs"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Hardening: idempotent/reject-if-present commitment registration + freshness
// ---------------------------------------------------------------------------

/// Re-registering the identical commitment for an identity is a no-op
/// success, but registering a *different* commitment is rejected rather
/// than silently overwriting the trusted store.
#[tokio::test]
async fn register_commitment_rejects_conflicting_rebind() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut source = ZeroKnowledgeManager::new(&config).await?;
    source.register_prover("alice")?;
    source.register_prover("mallory")?;
    let alice = source
        .registered_commitment_bytes("alice")
        .ok_or("alice commitment must exist")?;
    let mallory = source
        .registered_commitment_bytes("mallory")
        .ok_or("mallory commitment must exist")?;

    let mut verifier = ZeroKnowledgeManager::new(&config).await?;
    verifier.register_commitment("alice", &alice)?;
    // Same value again: idempotent success.
    verifier.register_commitment("alice", &alice)?;
    // Different value for an already-registered identity: rejected.
    let result = verifier.register_commitment("alice", &mallory);
    assert!(
        result.is_err(),
        "rebinding a registered identity to a different commitment must be rejected"
    );
    // The original commitment is untouched.
    assert_eq!(
        verifier.registered_commitment_bytes("alice"),
        Some(alice),
        "a rejected rebind must not mutate the trusted store"
    );
    Ok(())
}

/// A stale statement is rejected by the freshness window even though the
/// timestamp is what the proof is cryptographically bound to.
#[tokio::test]
async fn expired_statement_is_stale() {
    use synaptic::security::zero_knowledge::{
        AccessStatement, AccessType, MAX_STATEMENT_AGE, MAX_STATEMENT_CLOCK_SKEW,
    };
    let now = chrono::Utc::now();
    let stale = AccessStatement {
        memory_key: "k".to_string(),
        user_id: "u".to_string(),
        access_type: AccessType::Read,
        timestamp: now - MAX_STATEMENT_AGE - chrono::Duration::seconds(1),
    };
    assert!(
        !stale.is_fresh(now),
        "a statement older than MAX_STATEMENT_AGE must be rejected"
    );

    let fresh = AccessStatement {
        timestamp: now - chrono::Duration::seconds(1),
        ..stale.clone()
    };
    assert!(fresh.is_fresh(now), "a recent statement must be accepted");

    let future = AccessStatement {
        timestamp: now + MAX_STATEMENT_CLOCK_SKEW + chrono::Duration::seconds(5),
        ..stale
    };
    assert!(
        !future.is_fresh(now),
        "a statement implausibly far in the future must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Task 4.3: caller-supplied statements; external verification
// ---------------------------------------------------------------------------

/// An external verifier holding only (statement, proof, verifying key,
/// registered commitment) — with no access to the generating manager —
/// verifies the proof.
#[tokio::test]
async fn external_verifier_round_trip() -> Result<(), Box<dyn Error>> {
    let config = SecurityConfig::default();
    let mut prover = ZeroKnowledgeManager::new(&config).await?;
    prover.register_prover("prover")?;
    let context = mfa_context("prover");
    let stmt = statement("prover", "external-memory");

    let proof = prover.generate_access_proof(&stmt, &context).await?;
    let vk_bytes = prover.verifying_key_bytes()?;
    let commitment = prover
        .registered_commitment_bytes("prover")
        .ok_or("commitment must exist after registration")?;
    drop(prover);

    let is_valid = synaptic::security::zero_knowledge::verify_proof_external(
        &stmt,
        &proof,
        &vk_bytes,
        &commitment,
    )?;
    assert!(is_valid, "external verifier must accept an honest proof");

    let other = statement("prover", "different-memory");
    let is_valid = synaptic::security::zero_knowledge::verify_proof_external(
        &other,
        &proof,
        &vk_bytes,
        &commitment,
    )?;
    assert!(
        !is_valid,
        "external verifier must reject a proof bound to another statement"
    );

    Ok(())
}
