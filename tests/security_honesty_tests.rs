//! Tests asserting that disabled crypto features error instead of faking.

use synaptic::error::MemoryError;

#[test]
fn feature_disabled_error_names_feature_and_operation() {
    let err = MemoryError::feature_disabled("homomorphic-encryption", "encrypt_vector");
    let msg = err.to_string();
    assert!(msg.contains("homomorphic-encryption"));
    assert!(msg.contains("encrypt_vector"));
}

#[cfg(all(feature = "security", not(feature = "homomorphic-encryption")))]
mod he_disabled {
    use synaptic::memory::types::{MemoryEntry, MemoryType};
    use synaptic::security::encryption::EncryptionManager;
    use synaptic::security::key_management::KeyManager;
    use synaptic::security::{SecurityConfig, SecurityContext};

    /// With the `homomorphic-encryption` cargo feature off, the homomorphic
    /// encryption/decryption paths must fail closed with
    /// `MemoryError::FeatureDisabled` instead of silently returning
    /// fake-encrypted bytes (the old `value * 1.5 + 42.0` fallback).
    #[tokio::test]
    async fn homomorphic_encrypt_errors_when_feature_disabled() {
        let config = SecurityConfig::default();
        let key_manager = KeyManager::new(&config)
            .await
            .expect("key manager construction must still work without the HE feature");
        let mut encryption_manager = EncryptionManager::new(&config, key_manager)
            .await
            .expect("encryption manager must still construct so callers can probe availability");

        let entry = MemoryEntry::new(
            "test-key".to_string(),
            "test-value".to_string(),
            MemoryType::ShortTerm,
        );
        let mut context = SecurityContext::new("test-user".to_string(), vec!["user".to_string()]);
        // Default policy requires MFA; satisfy it so we reach the crypto path
        // instead of failing at the access-control check.
        context.mfa_verified = true;

        let result = encryption_manager
            .homomorphic_encrypt(&entry, &context)
            .await;
        let err = result.expect_err("must not fake-encrypt when the HE feature is disabled");
        assert!(err.to_string().contains("homomorphic-encryption"));
    }

    #[tokio::test]
    async fn homomorphic_decrypt_errors_when_feature_disabled() {
        let config = SecurityConfig::default();
        let key_manager = KeyManager::new(&config)
            .await
            .expect("key manager construction must still work without the HE feature");
        let mut encryption_manager = EncryptionManager::new(&config, key_manager)
            .await
            .expect("encryption manager must still construct so callers can probe availability");

        let mut context = SecurityContext::new("test-user".to_string(), vec!["user".to_string()]);
        // Default policy requires MFA; satisfy it so we reach the crypto path
        // instead of failing at the access-control check.
        context.mfa_verified = true;

        // Build a placeholder encrypted entry directly rather than going through
        // homomorphic_encrypt (which itself now errors closed) so decrypt_vector
        // is exercised independently.
        let encrypted_entry = synaptic::security::EncryptedMemoryEntry {
            id: "test-id".to_string(),
            encrypted_data: vec![0u8; 8],
            encryption_algorithm: "Homomorphic-CKKS".to_string(),
            key_id: "test-key-id".to_string(),
            is_homomorphic: true,
            privacy_level: synaptic::security::PrivacyLevel::Secret,
            created_at: chrono::Utc::now(),
            metadata: synaptic::security::EncryptionMetadata {
                algorithm_version: "2.0".to_string(),
                key_derivation: "Homomorphic-KeyGen".to_string(),
                salt: Vec::new(),
                iv: Vec::new(),
                auth_tag: None,
                compression: Some("CKKS".to_string()),
            },
        };

        let result = encryption_manager
            .homomorphic_decrypt(&encrypted_entry, &context)
            .await;
        let err = result.expect_err("must not fake-decrypt when the HE feature is disabled");
        assert!(err.to_string().contains("homomorphic-encryption"));
    }
}

// Search/similarity have no real homomorphic implementation in ANY build
// (real ones land in Phase 4), so they must fail closed regardless of the
// homomorphic-encryption feature.
#[cfg(feature = "security")]
mod he_search_similarity_fail_closed {
    use synaptic::security::encryption::EncryptionManager;
    use synaptic::security::key_management::KeyManager;
    use synaptic::security::{SecureOperation, SecurityConfig, SecurityContext};

    async fn manager_and_context() -> (EncryptionManager, SecurityContext) {
        let config = SecurityConfig::default();
        let key_manager = KeyManager::new(&config)
            .await
            .expect("key manager must construct");
        let manager = EncryptionManager::new(&config, key_manager)
            .await
            .expect("encryption manager must still construct so callers can probe availability");
        let mut context = SecurityContext::new("test-user".to_string(), vec!["user".to_string()]);
        // Default policy requires MFA; satisfy it so we reach the crypto path.
        context.mfa_verified = true;
        (manager, context)
    }

    #[tokio::test]
    async fn homomorphic_search_fails_closed() {
        let (mut manager, context) = manager_and_context().await;
        let result = manager
            .homomorphic_compute(
                &[],
                SecureOperation::Search {
                    query: "q".to_string(),
                },
                &context,
            )
            .await;
        let err = result.expect_err("must not fake homomorphic search");
        assert!(err.to_string().contains("homomorphic_search"));
    }

    #[tokio::test]
    async fn homomorphic_similarity_fails_closed() {
        let (mut manager, context) = manager_and_context().await;
        let result = manager
            .homomorphic_compute(
                &[],
                SecureOperation::Similarity { threshold: 0.5 },
                &context,
            )
            .await;
        let err = result.expect_err("must not fake homomorphic similarity");
        assert!(err.to_string().contains("homomorphic_similarity"));
    }
}

#[cfg(feature = "security")]
mod zk_content_hash_is_real {
    #[test]
    fn content_hash_is_sha256_not_length_arithmetic() {
        // Two different strings with the same length must hash differently.
        let a = synaptic::security::zero_knowledge::hash_content_for_test("aaaa");
        let b = synaptic::security::zero_knowledge::hash_content_for_test("bbbb");
        assert_ne!(a, b);
        assert_eq!(a.len(), 64); // hex sha256
    }
}

#[cfg(all(feature = "security", not(feature = "zero-knowledge-proofs")))]
mod zk_verify_fails_closed {
    use synaptic::security::zero_knowledge::{AccessStatement, AccessType, ZeroKnowledgeManager};
    use synaptic::security::{SecurityConfig, SecurityContext};

    #[tokio::test]
    async fn zk_verify_errors_when_feature_disabled() {
        let config = SecurityConfig::default();
        let mut manager = ZeroKnowledgeManager::new(&config)
            .await
            .expect("zero-knowledge manager must construct so callers can probe availability");

        let mut context = SecurityContext::new("test-user".to_string(), vec!["user".to_string()]);
        context.mfa_verified = true;

        let statement = AccessStatement {
            memory_key: "memory-1".to_string(),
            user_id: context.user_id.clone(),
            access_type: AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        let proof = manager
            .generate_access_proof(&statement, &context)
            .await
            .expect("fallback proof generation must still succeed");

        let result = manager.verify_access_proof(&proof, &statement).await;
        let err = result.expect_err("must not fake zero-knowledge verification");
        assert!(err.to_string().contains("zero-knowledge"));
    }
}

#[cfg(feature = "security")]
mod dp_noise_is_real_randomness {
    use synaptic::security::privacy::PrivacyManager;
    use synaptic::security::SecurityConfig;

    /// The differential-privacy noise generator must draw from a real
    /// (cryptographically secure) random source rather than a stub that
    /// always returns the same value. Sample many draws of Laplace noise at
    /// scale 1.0 and assert the distribution has nonzero variance and a
    /// mean roughly centered on zero. This catches "returns 0" (or any
    /// constant-value) stubs.
    #[tokio::test]
    async fn laplace_noise_has_nonzero_variance_and_bounded_mean() {
        let config = SecurityConfig::default();
        let manager = PrivacyManager::new(&config)
            .await
            .expect("privacy manager must construct");

        let scale = 1.0;
        let n = 1000usize;
        let samples: Vec<f64> = (0..n)
            .map(|_| manager.sample_laplace_noise(scale))
            .collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(
            variance > 0.0,
            "Laplace noise samples must have nonzero variance (got {variance}); \
             a constant-value stub would produce variance == 0"
        );
        assert!(
            mean.abs() < 0.5,
            "Laplace noise mean should be close to 0 for scale=1.0, got {mean}"
        );

        // Sanity: not all samples identical (would also indicate a stub).
        let all_identical = samples.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_identical, "noise samples must not all be identical");
    }
}
