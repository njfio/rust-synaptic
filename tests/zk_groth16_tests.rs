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
