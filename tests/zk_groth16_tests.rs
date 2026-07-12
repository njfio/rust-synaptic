//! Groth16 key lifecycle and Poseidon knowledge-of-preimage proof tests.
//!
//! Phase 4 (tasks 4.1-4.3): real zk-SNARK parameter persistence, proving,
//! and verification over the BLS12-381 scalar field.

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
