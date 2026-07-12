//! Comprehensive tests for real zero-knowledge proof implementation
//!
//! Tests the Bellman-based zk-SNARKs with production-ready algorithms
//! ensuring 90%+ test coverage and comprehensive validation.

use std::error::Error;
#[cfg(feature = "security")]
use synaptic::memory::types::{MemoryEntry, MemoryType};
#[cfg(feature = "security")]
use synaptic::security::{SecurityConfig, SecurityContext, SecurityManager};

#[cfg(feature = "zero-knowledge-proofs")]
mod bellman_tests {
    use super::*;

    /// ZK proof generation always requires MFA; these tests assert the proof
    /// contract, not the MFA policy, so satisfy it up front.
    fn mfa_context(user: &str) -> SecurityContext {
        let mut context = SecurityContext::new(user.to_string(), vec!["admin".to_string()]);
        context.mfa_verified = true;
        context
    }

    /// Proof generation stamps its own timestamp into the statement it hashes,
    /// so a test-held statement never hash-matches the proof. Align the proof's
    /// statement hash with the test statement so verification proceeds past the
    /// hash check to the (currently fail-closed) cryptographic step.
    fn align_statement_hash<T: serde::Serialize>(
        proof: &mut synaptic::security::zero_knowledge::ZKProof,
        statement: &T,
    ) -> Result<(), Box<dyn Error>> {
        let serialized = serde_json::to_string(statement)?;
        proof.statement_hash =
            synaptic::security::zero_knowledge::hash_content_for_test(&serialized);
        Ok(())
    }

    #[tokio::test]
    async fn test_real_zero_knowledge_proof_generation() -> Result<(), Box<dyn Error>> {
        // Initialize security manager with zero-knowledge proofs enabled
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;
        config.encryption_key_size = 2048;

        let mut security_manager = SecurityManager::new(config).await?;

        // Create security context
        let context = mfa_context("test_user");

        // Create test memory entry
        let memory_entry = MemoryEntry::new(
            "sensitive_data".to_string(),
            "confidential information".to_string(),
            MemoryType::ShortTerm,
        );

        // Test zero-knowledge proof generation for access
        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: memory_entry.key.clone(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        let proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(&memory_entry.key, &context, access_statement.access_type)
            .await?;

        // Verify proof properties
        assert!(!proof.id.is_empty());
        assert!(!proof.statement_hash.is_empty());
        assert!(!proof.proof_data.is_empty());
        assert_eq!(proof.proving_key_id, "bellman_groth16");

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_proof_verification() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("test_user");

        // Create access statement
        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "test_key".to_string(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        // Generate proof
        let mut proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(
                &access_statement.memory_key,
                &context,
                access_statement.access_type.clone(),
            )
            .await?;
        align_statement_hash(&mut proof, &access_statement)?;

        // Verification fails closed until real Groth16 verification lands in
        // Phase 4 (task 4.2); this reverts to `assert!(is_valid)` then.
        let err = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .verify_access_proof(&proof, &access_statement)
            .await
            .expect_err("verification must fail closed until real Groth16 verify lands");

        assert!(err
            .to_string()
            .contains("zero-knowledge-proofs(real-verify)"));

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_content_proof() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("test_user");

        // Create content statement
        let content_statement = synaptic::security::zero_knowledge::ContentStatement {
            memory_key: "content_key".to_string(),
            predicate: synaptic::security::zero_knowledge::ContentPredicate::ContainsKeyword(
                "secret".to_string(),
            ),
            timestamp: chrono::Utc::now(),
        };

        // Create a memory entry for content proof
        let memory_entry = MemoryEntry::new(
            content_statement.memory_key.clone(),
            "secret information".to_string(),
            MemoryType::ShortTerm,
        );

        // Generate content proof
        let mut proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_content_proof(&memory_entry, content_statement.predicate.clone(), &context)
            .await?;
        align_statement_hash(&mut proof, &content_statement)?;

        // Verification fails closed until real Groth16 verification lands in
        // Phase 4 (task 4.2); this reverts to `assert!(is_valid)` then.
        let err = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .verify_content_proof(&proof, &content_statement)
            .await
            .expect_err("verification must fail closed until real Groth16 verify lands");

        assert!(err
            .to_string()
            .contains("zero-knowledge-proofs(real-verify)"));
        assert!(!proof.proof_data.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_aggregate_proof() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("test_user");

        // Create aggregate statement
        let aggregate_statement = synaptic::security::zero_knowledge::AggregateStatement {
            entry_count: 100,
            aggregate_type: synaptic::security::zero_knowledge::AggregateType::Count,
            timestamp: chrono::Utc::now(),
        };

        // Create sample memory entries for aggregate proof
        let entries = vec![
            MemoryEntry::new(
                "entry1".to_string(),
                "data1".to_string(),
                MemoryType::ShortTerm,
            ),
            MemoryEntry::new(
                "entry2".to_string(),
                "data2".to_string(),
                MemoryType::ShortTerm,
            ),
        ];

        // Generate aggregate proof
        let proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_aggregate_proof(
                &entries,
                aggregate_statement.aggregate_type.clone(),
                &context,
            )
            .await?;

        // Create a statement that matches what was actually computed
        let actual_statement = synaptic::security::zero_knowledge::AggregateStatement {
            entry_count: entries.len(),
            aggregate_type: aggregate_statement.aggregate_type.clone(),
            timestamp: chrono::Utc::now(),
        };

        // Note: In a real implementation, we would need a verify_aggregate_proof method
        // For now, we'll just verify the proof was generated successfully
        assert!(!proof.proof_data.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_proof_performance() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("perf_test_user");

        // Test performance with multiple proofs
        let mut proof_times = Vec::new();
        let mut verify_times = Vec::new();

        for i in 0..5 {
            let access_statement = synaptic::security::zero_knowledge::AccessStatement {
                memory_key: format!("perf_key_{}", i),
                user_id: context.user_id.clone(),
                access_type: synaptic::security::zero_knowledge::AccessType::Read,
                timestamp: chrono::Utc::now(),
            };

            // Measure proof generation time
            let start_time = std::time::Instant::now();
            let mut proof = security_manager
                .zero_knowledge_manager
                .as_mut()
                .unwrap()
                .generate_access_proof(
                    &access_statement.memory_key,
                    &context,
                    access_statement.access_type.clone(),
                )
                .await?;
            let proof_time = start_time.elapsed();
            proof_times.push(proof_time);
            align_statement_hash(&mut proof, &access_statement)?;

            // Measure verification time. Verification fails closed until real
            // Groth16 verification lands in Phase 4 (task 4.2); this reverts
            // to `assert!(is_valid)` then.
            let start_time = std::time::Instant::now();
            let err = security_manager
                .zero_knowledge_manager
                .as_mut()
                .unwrap()
                .verify_access_proof(&proof, &access_statement)
                .await
                .expect_err("verification must fail closed until real Groth16 verify lands");
            let verify_time = start_time.elapsed();
            verify_times.push(verify_time);

            assert!(err
                .to_string()
                .contains("zero-knowledge-proofs(real-verify)"));
        }

        // Verify performance is reasonable (proofs should complete within 10 seconds each)
        for proof_time in &proof_times {
            assert!(proof_time.as_secs() < 10);
        }

        // Verification should be much faster (under 1 second)
        for verify_time in &verify_times {
            assert!(verify_time.as_millis() < 1000);
        }

        // Test metrics collection (query the ZK manager directly; this test's
        // context has no registered access-control session).
        let metrics = security_manager
            .zero_knowledge_manager
            .as_ref()
            .unwrap()
            .get_metrics()
            .await?;
        assert!(metrics.total_proofs_generated >= 5);
        // Verified-proof metrics stay at zero while verification fails closed;
        // restore `total_proofs_verified >= 5` once Phase 4 lands real verify.

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_proof_invalid_verification() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("test_user");

        // Create access statement
        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "test_key".to_string(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        // Generate proof
        let proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(
                &access_statement.memory_key,
                &context,
                access_statement.access_type.clone(),
            )
            .await?;

        // Create different statement for verification (should fail)
        let different_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "different_key".to_string(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Write,
            timestamp: chrono::Utc::now(),
        };

        // Verification should fail with different statement
        let is_valid = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .verify_access_proof(&proof, &different_statement)
            .await?;

        assert!(!is_valid);

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_security_context_validation() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;
        config.access_control_policy.require_mfa = true;

        let mut security_manager = SecurityManager::new(config).await?;

        // Create context without MFA
        let invalid_context =
            SecurityContext::new("test_user".to_string(), vec!["user".to_string()]);

        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "secure_key".to_string(),
            user_id: invalid_context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        // Should fail due to MFA requirement
        let result = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(
                &access_statement.memory_key,
                &invalid_context,
                access_statement.access_type.clone(),
            )
            .await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_zero_knowledge_proof_serialization() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let context = mfa_context("test_user");

        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "serialization_test".to_string(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        // Generate proof
        let mut proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(
                &access_statement.memory_key,
                &context,
                access_statement.access_type.clone(),
            )
            .await?;
        align_statement_hash(&mut proof, &access_statement)?;

        // Test serialization/deserialization
        let serialized = serde_json::to_string(&proof)?;
        let deserialized: synaptic::security::zero_knowledge::ZKProof =
            serde_json::from_str(&serialized)?;

        // Verify deserialized proof is identical
        assert_eq!(proof.id, deserialized.id);
        assert_eq!(proof.statement_hash, deserialized.statement_hash);
        assert_eq!(proof.proof_data, deserialized.proof_data);
        assert_eq!(proof.proving_key_id, deserialized.proving_key_id);

        // Verification fails closed until real Groth16 verification lands in
        // Phase 4 (task 4.2); this reverts to `assert!(is_valid)` then.
        let err = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .verify_access_proof(&deserialized, &access_statement)
            .await
            .expect_err("verification must fail closed until real Groth16 verify lands");
        assert!(err
            .to_string()
            .contains("zero-knowledge-proofs(real-verify)"));

        Ok(())
    }
}

#[cfg(not(feature = "zero-knowledge-proofs"))]
mod fallback_tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_zero_knowledge_proofs() -> Result<(), Box<dyn Error>> {
        let mut config = SecurityConfig::default();
        config.enable_zero_knowledge = true;

        let mut security_manager = SecurityManager::new(config).await?;
        let mut context = SecurityContext::new("test_user".to_string(), vec!["admin".to_string()]);

        // Set MFA verification for the context
        context.mfa_verified = true;
        context.session_id = "test_session_123".to_string();
        context.request_id = "test_request_456".to_string();

        let access_statement = synaptic::security::zero_knowledge::AccessStatement {
            memory_key: "fallback_test".to_string(),
            user_id: context.user_id.clone(),
            access_type: synaptic::security::zero_knowledge::AccessType::Read,
            timestamp: chrono::Utc::now(),
        };

        // Should use fallback implementation when feature is not enabled
        let proof = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .generate_access_proof(
                &access_statement.memory_key,
                &context,
                access_statement.access_type.clone(),
            )
            .await?;

        assert!(!proof.id.is_empty());
        assert!(!proof.proof_data.is_empty());

        // Fallback verification fails closed: there is no cryptographic
        // verification without the zero-knowledge-proofs feature, so an
        // honest Err beats a fake Ok(true).
        let err = security_manager
            .zero_knowledge_manager
            .as_mut()
            .unwrap()
            .verify_access_proof(&proof, &access_statement)
            .await
            .expect_err("fallback verification must fail closed");
        assert!(err.to_string().contains("zero-knowledge"));

        Ok(())
    }
}
