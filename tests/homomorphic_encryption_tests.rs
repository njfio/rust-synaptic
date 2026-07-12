//! Tests for the real TFHE homomorphic encryption implementation
//!
//! Feature ON: FheInt64 fixed-point round-trip (sign-preserving, exact at
//! FIXED_POINT_SCALE granularity), additive homomorphism (sum/average), and
//! permanently descoped operations (search/similarity/count) failing closed.
//! Feature OFF: all homomorphic paths fail closed with FeatureDisabled.

#![cfg(feature = "security")]

use std::error::Error;
use synaptic::security::{SecurityConfig, SecurityContext, SecurityManager};
use synaptic::{MemoryEntry, MemoryType};

#[cfg(feature = "homomorphic-encryption")]
mod tfhe_tests {
    use super::*;
    use chrono::Utc;
    use synaptic::security::{
        EncryptedMemoryEntry, EncryptionMetadata, PrivacyLevel, SecureOperation,
    };

    async fn manager_and_context() -> Result<(SecurityManager, SecurityContext), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_homomorphic_encryption = true;
        // The tests exercise the crypto path, not MFA enforcement
        config.access_control_policy.require_mfa = false;
        let manager = SecurityManager::new(config).await?;
        let context = SecurityContext::new("test_user".to_string(), vec!["admin".to_string()]);
        Ok((manager, context))
    }

    fn wrap_ciphertext(encrypted_data: Vec<u8>) -> EncryptedMemoryEntry {
        EncryptedMemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            encrypted_data,
            encryption_algorithm: "Homomorphic-TFHE".to_string(),
            key_id: "test-key".to_string(),
            is_homomorphic: true,
            privacy_level: PrivacyLevel::Secret,
            created_at: Utc::now(),
            metadata: EncryptionMetadata {
                algorithm_version: "2.0".to_string(),
                key_derivation: "Homomorphic-KeyGen".to_string(),
                salt: Vec::new(),
                iv: Vec::new(),
                auth_tag: None,
                compression: Some("TFHE".to_string()),
            },
        }
    }

    #[tokio::test]
    async fn test_fixed_point_round_trip_preserves_sign() -> Result<(), Box<dyn Error>> {
        let (manager, context) = manager_and_context().await?;

        // Signed values, including negative and zero; all exact multiples of
        // 1e-6 so the round-trip must be EXACT per the precision contract.
        let original = vec![-3.5, 0.0, 2.25];
        let ciphertext = manager
            .encryption_manager
            .homomorphic_encrypt_vector(&original, &context)
            .await?;
        assert!(!ciphertext.is_empty());
        assert_ne!(ciphertext.len(), original.len() * 8); // real ciphertext expansion

        let decrypted = manager
            .encryption_manager
            .homomorphic_decrypt_vector(&ciphertext, &context)
            .await?;
        assert_eq!(decrypted.len(), original.len());
        for (d, o) in decrypted.iter().zip(&original) {
            assert_eq!(d, o, "fixed-point round-trip must be exact");
        }
        // Explicit sign-preservation assertion
        assert!(
            (decrypted[0] - (-3.5)).abs() < 1e-9,
            "sign must be preserved"
        );
        assert!(decrypted[0] < 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_additive_homomorphism_sum_and_average() -> Result<(), Box<dyn Error>> {
        let (mut manager, context) = manager_and_context().await?;

        let a = vec![1.5, -2.0];
        let b = vec![2.5, 4.0];
        let ct_a = manager
            .encryption_manager
            .homomorphic_encrypt_vector(&a, &context)
            .await?;
        let ct_b = manager
            .encryption_manager
            .homomorphic_encrypt_vector(&b, &context)
            .await?;
        let entries = vec![wrap_ciphertext(ct_a), wrap_ciphertext(ct_b)];

        // Encrypted sum decrypts to the plaintext element-wise sum
        let sum_result = manager
            .encryption_manager
            .homomorphic_compute(&entries, SecureOperation::Sum, &context)
            .await?;
        let sum = manager
            .encryption_manager
            .homomorphic_decrypt_vector(&sum_result.result_data, &context)
            .await?;
        assert_eq!(sum, vec![4.0, 2.0]);

        // Encrypted average decrypts to the plaintext element-wise average
        let avg_result = manager
            .encryption_manager
            .homomorphic_compute(&entries, SecureOperation::Average, &context)
            .await?;
        let avg = manager
            .encryption_manager
            .homomorphic_decrypt_vector(&avg_result.result_data, &context)
            .await?;
        assert_eq!(avg, vec![2.0, 1.0]);

        Ok(())
    }

    #[tokio::test]
    async fn test_descoped_operations_fail_closed_with_feature_on() -> Result<(), Box<dyn Error>> {
        let (mut manager, context) = manager_and_context().await?;

        let ct = manager
            .encryption_manager
            .homomorphic_encrypt_vector(&[1.0], &context)
            .await?;
        let entries = vec![wrap_ciphertext(ct)];

        // Search, similarity, and count are unsupported BY DESIGN and must
        // fail closed even with the homomorphic-encryption feature enabled.
        let ops = vec![
            SecureOperation::Search {
                query: "secret".to_string(),
            },
            SecureOperation::Similarity { threshold: 0.5 },
            SecureOperation::Count,
        ];
        for op in ops {
            let err = manager
                .encryption_manager
                .homomorphic_compute(&entries, op.clone(), &context)
                .await
                .expect_err("descoped homomorphic operation must fail closed");
            let msg = err.to_string();
            assert!(
                msg.contains("unsupported") && msg.contains("homomorphic"),
                "error must name the unsupported homomorphic feature, got: {msg} for {op:?}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_homomorphic_entry_encryption_basic() -> Result<(), Box<dyn Error>> {
        let (mut manager, context) = manager_and_context().await?;

        let memory_entry = MemoryEntry::new(
            "test_numeric_data".to_string(),
            "42.5 -7.25".to_string(),
            MemoryType::ShortTerm,
        );
        let encrypted_entry = manager
            .encryption_manager
            .homomorphic_encrypt(&memory_entry, &context)
            .await?;

        assert!(encrypted_entry.is_homomorphic);
        assert_eq!(encrypted_entry.encryption_algorithm, "Homomorphic-TFHE");
        assert!(!encrypted_entry.encrypted_data.is_empty());
        assert_ne!(
            encrypted_entry.encrypted_data,
            memory_entry.value.as_bytes()
        );

        Ok(())
    }
}

#[cfg(not(feature = "homomorphic-encryption"))]
mod fallback_tests {
    use super::*;
    use synaptic::security::access_control::{AuthenticationCredentials, AuthenticationType};

    #[tokio::test]
    async fn test_fallback_homomorphic_encryption() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_homomorphic_encryption = true;
        // Disable MFA for testing
        config.access_control_policy.require_mfa = false;

        let mut security_manager = SecurityManager::new(config).await?;

        // Add required permissions to the user role
        security_manager
            .access_control
            .add_role(
                "user".to_string(),
                vec![
                    synaptic::security::Permission::ReadMemory,
                    synaptic::security::Permission::WriteMemory,
                ],
            )
            .await?;

        // Provision the user (authentication is real as of Task 4.7) and
        // properly authenticate to create a valid session
        security_manager
            .access_control
            .set_password("test_user", "test_password123")?;
        let credentials = AuthenticationCredentials {
            auth_type: AuthenticationType::Password,
            password: Some("test_password123".to_string()),
            api_key: None,
            certificate: None,
            mfa_token: None,
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test_agent".to_string()),
        };

        let context = security_manager
            .access_control
            .authenticate("test_user".to_string(), credentials)
            .await?;

        let memory_entry = MemoryEntry::new(
            "test_data".to_string(),
            "42.0 100.0".to_string(),
            MemoryType::ShortTerm,
        );

        // With the `homomorphic-encryption` cargo feature off there is no HE
        // backend, so encryption must fail closed instead of fabricating
        // ciphertext.
        let result = security_manager
            .encrypt_memory(&memory_entry, &context)
            .await;
        let err = result.expect_err("must not fake-encrypt when the HE feature is disabled");
        assert!(err.to_string().contains("homomorphic-encryption"));

        Ok(())
    }
}
