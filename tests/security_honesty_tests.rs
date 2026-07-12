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
