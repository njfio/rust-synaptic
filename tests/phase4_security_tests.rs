//! Comprehensive tests for Phase 4 Security & Privacy features
//!
//! Tests all advanced security components including homomorphic encryption,
//! zero-knowledge proofs, differential privacy, and advanced access control.

use std::error::Error;
use synaptic::security::{
    access_control::{AuthenticationCredentials, AuthenticationType},
    privacy::{PrivacyQuery, PrivacyQueryType},
    zero_knowledge::{AccessType, ContentPredicate},
    SecureOperation, SecurityConfig, SecurityContext, SecurityManager,
};
use synaptic::{MemoryEntry, MemoryType};

// Helper function to create authenticated security context.
//
// Authentication is real now (Task 4.7): the user's password is provisioned
// as an argon2 hash, a TOTP secret is enrolled, and a valid RFC 6238 token
// for the current time step is presented so `mfa_verified` is genuinely true.
async fn create_authenticated_context(
    security_manager: &mut SecurityManager,
    user_id: &str,
    password: &str,
) -> Result<SecurityContext, Box<dyn Error>> {
    use synaptic::security::access_control::AccessControlManager;

    let totp_secret = AccessControlManager::generate_totp_secret();
    security_manager
        .access_control
        .set_password(user_id, password)?;
    security_manager
        .access_control
        .set_totp_secret(user_id, totp_secret.clone())?;

    let totp = totp_rs::TOTP::new(totp_rs::Algorithm::SHA1, 6, 1, 30, totp_secret)?;
    let mfa_token = totp.generate_current()?;

    let credentials = AuthenticationCredentials {
        auth_type: AuthenticationType::Password,
        password: Some(password.to_string()),
        api_key: None,
        certificate: None,
        mfa_token: Some(mfa_token),
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("test-agent".to_string()),
    };

    let context = security_manager
        .access_control
        .authenticate(user_id.to_string(), credentials)
        .await?;

    Ok(context)
}

#[tokio::test]
async fn test_homomorphic_encryption_basic() -> Result<(), Box<dyn Error>> {
    // Test basic homomorphic encryption functionality
    let mut security_config = SecurityConfig::default();
    security_config.enable_homomorphic_encryption = true;
    security_config.enable_differential_privacy = false;
    security_config.enable_zero_knowledge = false;
    // Disable MFA for testing
    security_config.access_control_policy.require_mfa = false;

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role with required permissions
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![
                synaptic::security::Permission::ReadMemory,
                synaptic::security::Permission::WriteMemory,
                synaptic::security::Permission::ExecuteQueries,
            ],
        )
        .await?;

    // Authenticate user properly
    let context =
        create_authenticated_context(&mut security_manager, "test_user", "password123").await?;

    // Create test memory entry
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test data for homomorphic encryption".to_string(),
        MemoryType::LongTerm,
    );

    // The `homomorphic-encryption` cargo feature is off by default, so the
    // fake CKKS-labelled encryption path must fail closed instead of
    // returning fabricated ciphertext. Revert this assertion to the old
    // "encrypt then decrypt" flow in Phase 4, once a real HE backend lands.
    let result = security_manager.encrypt_memory(&entry, &context).await;
    let err = result.expect_err("must not fake-encrypt when the HE feature is disabled");
    assert!(err.to_string().contains("homomorphic-encryption"));

    Ok(())
}

#[tokio::test]
async fn test_homomorphic_computation() -> Result<(), Box<dyn Error>> {
    // Test secure computation on encrypted data
    let mut security_config = SecurityConfig::default();
    security_config.enable_homomorphic_encryption = true;
    // Disable MFA for testing
    security_config.access_control_policy.require_mfa = false;

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add admin role with required permissions
    security_manager
        .access_control
        .add_role(
            "admin".to_string(),
            vec![
                synaptic::security::Permission::ReadMemory,
                synaptic::security::Permission::WriteMemory,
                synaptic::security::Permission::ExecuteQueries,
            ],
        )
        .await?;

    // Authenticate admin user properly
    let context =
        create_authenticated_context(&mut security_manager, "admin", "adminpass123").await?;

    // Create multiple test entries
    let entries = vec![
        MemoryEntry::new(
            "entry1".to_string(),
            "Data one".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "entry2".to_string(),
            "Data two".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "entry3".to_string(),
            "Data three".to_string(),
            MemoryType::LongTerm,
        ),
    ];

    // Encrypting the first entry must fail closed: the
    // `homomorphic-encryption` cargo feature is disabled, so there is no real
    // ciphertext to compute over. Revert to the full encrypt-then-compute
    // flow in Phase 4, once a real HE backend lands.
    let entry = &entries[0];
    let result = security_manager.encrypt_memory(entry, &context).await;
    let err = result.expect_err("must not fake-encrypt when the HE feature is disabled");
    assert!(err.to_string().contains("homomorphic-encryption"));

    Ok(())
}

#[tokio::test]
async fn test_zero_knowledge_access_proofs() -> Result<(), Box<dyn Error>> {
    // Test zero-knowledge proofs for access control
    let security_config = SecurityConfig {
        enable_zero_knowledge: true,
        ..Default::default()
    };

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![synaptic::security::Permission::ReadMemory],
        )
        .await?;

    // Authenticate user properly
    let mut context =
        create_authenticated_context(&mut security_manager, "test_user", "password123").await?;
    // ZK proof generation always requires MFA; satisfy it so we reach the
    // proof path (this test asserts the proof contract, not the MFA policy).
    context.mfa_verified = true;
    // Register the prover identity (its commitment seeds the verifier's
    // trusted store; proofs from unregistered identities are refused).
    security_manager.register_zk_prover(&context.user_id)?;

    // Caller-supplied statement (task 4.3): the same statement handed to
    // generation verifies the proof.
    let statement = synaptic::security::zero_knowledge::AccessStatement {
        memory_key: "test_memory_key".to_string(),
        user_id: context.user_id.clone(),
        access_type: AccessType::Read,
        timestamp: chrono::Utc::now(),
    };

    let proof = security_manager
        .generate_access_proof(&statement, &context)
        .await?;

    assert!(!proof.id.is_empty());
    assert!(!proof.statement_hash.is_empty());
    assert!(!proof.proof_data.is_empty());

    // Real Groth16 verification (task 4.2): honest proof verifies true.
    let is_valid = security_manager
        .verify_access_proof(&proof, &statement)
        .await?;
    assert!(is_valid);

    Ok(())
}

#[tokio::test]
async fn test_zero_knowledge_content_proofs() -> Result<(), Box<dyn Error>> {
    // Test zero-knowledge proofs for content verification
    let security_config = SecurityConfig {
        enable_zero_knowledge: true,
        ..Default::default()
    };

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![synaptic::security::Permission::ReadMemory],
        )
        .await?;

    // Authenticate user properly
    let context =
        create_authenticated_context(&mut security_manager, "test_user", "password123").await?;

    // Create test entry with specific content
    let entry = MemoryEntry::new(
        "content_test".to_string(),
        "This content contains the keyword secret and is quite long".to_string(),
        MemoryType::LongTerm,
    );

    security_manager.register_zk_prover(&context.user_id)?;

    // Generate content proof for keyword presence (caller-supplied statement)
    let keyword_statement = synaptic::security::zero_knowledge::ContentStatement {
        memory_key: entry.key.clone(),
        predicate: ContentPredicate::ContainsKeyword("secret".to_string()),
        timestamp: chrono::Utc::now(),
    };
    let proof = security_manager
        .generate_content_proof(&entry, &keyword_statement, &context)
        .await?;

    assert!(!proof.id.is_empty());
    assert!(!proof.statement_hash.is_empty());

    // Generate proof for length constraint
    let length_statement = synaptic::security::zero_knowledge::ContentStatement {
        memory_key: entry.key.clone(),
        predicate: ContentPredicate::LengthGreaterThan(30),
        timestamp: chrono::Utc::now(),
    };
    let length_proof = security_manager
        .generate_content_proof(&entry, &length_statement, &context)
        .await?;

    assert!(!length_proof.id.is_empty());
    assert_ne!(proof.statement_hash, length_proof.statement_hash);

    Ok(())
}

#[tokio::test]
async fn test_differential_privacy_basic() -> Result<(), Box<dyn Error>> {
    // Test basic differential privacy functionality
    let security_config = SecurityConfig {
        enable_differential_privacy: true,
        privacy_budget: 5.0,
        ..Default::default()
    };

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![synaptic::security::Permission::ReadMemory],
        )
        .await?;

    // Authenticate user properly
    let context =
        create_authenticated_context(&mut security_manager, "test_user", "password123").await?;

    // Create test entry
    let entry = MemoryEntry::new(
        "privacy_test".to_string(),
        "Sensitive data that needs privacy protection".to_string(),
        MemoryType::LongTerm,
    );

    // Apply differential privacy
    let privatized = security_manager
        .privacy_manager
        .apply_differential_privacy(&entry, &context)
        .await?;

    // The privatized version should be different (with high probability)
    // Note: Due to randomness, this might occasionally be the same
    assert_eq!(privatized.key, entry.key);
    assert_eq!(privatized.memory_type, entry.memory_type);

    Ok(())
}

#[tokio::test]
async fn test_differential_privacy_statistics() -> Result<(), Box<dyn Error>> {
    // Test differentially private statistics
    let security_config = SecurityConfig {
        enable_differential_privacy: true,
        privacy_budget: 10.0,
        ..Default::default()
    };

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![synaptic::security::Permission::ReadMemory],
        )
        .await?;

    // Authenticate user properly
    let context =
        create_authenticated_context(&mut security_manager, "test_user", "password123").await?;

    // Create test entries
    let entries = vec![
        MemoryEntry::new(
            "entry1".to_string(),
            "Short".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "entry2".to_string(),
            "Medium length text".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "entry3".to_string(),
            "This is a much longer text entry".to_string(),
            MemoryType::LongTerm,
        ),
    ];

    // Test count query
    let count_query = PrivacyQuery {
        query_type: PrivacyQueryType::Count,
        sensitivity: 1.0,
        bins: None,
        quantile: None,
    };

    let count_stats = security_manager
        .privacy_manager
        .generate_private_statistics(&entries, count_query, &context)
        .await?;

    assert!(count_stats.result >= 0.0); // Should be non-negative
    assert!(count_stats.epsilon_used > 0.0);
    assert!(count_stats.confidence_interval.0 <= count_stats.confidence_interval.1);

    // Test average query
    let avg_query = PrivacyQuery {
        query_type: PrivacyQueryType::Average,
        sensitivity: 1.0,
        bins: None,
        quantile: None,
    };

    let avg_stats = security_manager
        .privacy_manager
        .generate_private_statistics(&entries, avg_query, &context)
        .await?;

    assert!(avg_stats.epsilon_used > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_privacy_budget_management() -> Result<(), Box<dyn Error>> {
    // Test privacy budget tracking and management
    let security_config = SecurityConfig {
        enable_differential_privacy: true,
        privacy_budget: 2.0, // Small budget for testing
        ..Default::default()
    };

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add user role
    security_manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![synaptic::security::Permission::ReadMemory],
        )
        .await?;

    // Authenticate user properly
    let context =
        create_authenticated_context(&mut security_manager, "budget_test_user", "password123")
            .await?;

    // Check initial budget
    let initial_budget = security_manager
        .privacy_manager
        .get_remaining_budget(&context.user_id)
        .await?;
    assert_eq!(initial_budget, 2.0);

    // Create test entry
    let entry = MemoryEntry::new(
        "budget_test".to_string(),
        "Test data for budget management".to_string(),
        MemoryType::LongTerm,
    );

    // Consume some budget
    let _privatized = security_manager
        .privacy_manager
        .apply_differential_privacy(&entry, &context)
        .await?;

    // Check remaining budget (should be less)
    let remaining_budget = security_manager
        .privacy_manager
        .get_remaining_budget(&context.user_id)
        .await?;
    assert!(remaining_budget < initial_budget);

    Ok(())
}

#[tokio::test]
async fn test_security_metrics_collection() -> Result<(), Box<dyn Error>> {
    // Test comprehensive security metrics collection
    let mut security_config = SecurityConfig::default();
    security_config.enable_homomorphic_encryption = true;
    security_config.enable_differential_privacy = true;
    security_config.enable_zero_knowledge = true;
    // Disable MFA for testing
    security_config.access_control_policy.require_mfa = false;

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add admin role with all permissions
    security_manager
        .access_control
        .add_role(
            "admin".to_string(),
            vec![
                synaptic::security::Permission::ReadMemory,
                synaptic::security::Permission::WriteMemory,
                synaptic::security::Permission::ExecuteQueries,
                synaptic::security::Permission::ViewAuditLogs,
                synaptic::security::Permission::ManageUsers,
            ],
        )
        .await?;

    // Authenticate admin user properly
    let context =
        create_authenticated_context(&mut security_manager, "metrics_user", "adminpass123").await?;

    // Perform various operations to generate metrics
    let entry = MemoryEntry::new(
        "metrics_test".to_string(),
        "Test data for metrics collection".to_string(),
        MemoryType::LongTerm,
    );

    // Encryption operation
    let _encrypted = security_manager.encrypt_memory(&entry, &context).await?;

    // Privacy operation
    let _privatized = security_manager
        .privacy_manager
        .apply_differential_privacy(&entry, &context)
        .await?;

    // Zero-knowledge operation
    security_manager.register_zk_prover(&context.user_id)?;
    let metrics_statement = synaptic::security::zero_knowledge::AccessStatement {
        memory_key: "metrics_test".to_string(),
        user_id: context.user_id.clone(),
        access_type: AccessType::Read,
        timestamp: chrono::Utc::now(),
    };
    let _proof = security_manager
        .generate_access_proof(&metrics_statement, &context)
        .await?;

    // Collect metrics
    let metrics = security_manager.get_security_metrics(&context).await?;

    // Verify metrics are collected
    assert!(metrics.encryption_metrics.total_homomorphic_encryptions > 0);
    assert!(metrics.privacy_metrics.total_privatizations > 0);

    if let Some(ref zk_metrics) = metrics.zero_knowledge_metrics {
        assert!(zk_metrics.total_proofs_generated > 0);
    }

    Ok(())
}

#[tokio::test]
async fn test_integrated_security_workflow() -> Result<(), Box<dyn Error>> {
    // Test complete integrated security workflow
    let mut security_config = SecurityConfig::default();
    security_config.enable_homomorphic_encryption = true;
    security_config.enable_differential_privacy = true;
    security_config.enable_zero_knowledge = true;
    security_config.privacy_budget = 10.0;
    // Disable MFA for testing
    security_config.access_control_policy.require_mfa = false;

    let mut security_manager = SecurityManager::new(security_config).await?;

    // Add admin role with all permissions
    security_manager
        .access_control
        .add_role(
            "admin".to_string(),
            vec![
                synaptic::security::Permission::ReadMemory,
                synaptic::security::Permission::WriteMemory,
                synaptic::security::Permission::ExecuteQueries,
                synaptic::security::Permission::ViewAuditLogs,
                synaptic::security::Permission::ManageUsers,
            ],
        )
        .await?;

    // Authenticate admin user properly
    let context =
        create_authenticated_context(&mut security_manager, "workflow_user", "adminpass123")
            .await?;

    // 1. Create sensitive data
    let sensitive_entry = MemoryEntry::new(
        "workflow_test".to_string(),
        "Highly sensitive financial data: $1,000,000 budget allocation".to_string(),
        MemoryType::LongTerm,
    );

    // 2. Apply differential privacy
    let privatized_entry = security_manager
        .privacy_manager
        .apply_differential_privacy(&sensitive_entry, &context)
        .await?;

    // 3. Encrypt with homomorphic encryption
    let encrypted_entry = security_manager
        .encrypt_memory(&privatized_entry, &context)
        .await?;
    assert!(encrypted_entry.is_homomorphic);

    // 4. Generate zero-knowledge proof for access
    security_manager.register_zk_prover(&context.user_id)?;
    let workflow_statement = synaptic::security::zero_knowledge::AccessStatement {
        memory_key: "workflow_test".to_string(),
        user_id: context.user_id.clone(),
        access_type: AccessType::Read,
        timestamp: chrono::Utc::now(),
    };
    let access_proof = security_manager
        .generate_access_proof(&workflow_statement, &context)
        .await?;

    // 5. Generate content proof
    let workflow_content_statement = synaptic::security::zero_knowledge::ContentStatement {
        memory_key: sensitive_entry.key.clone(),
        predicate: ContentPredicate::ContainsKeyword("financial".to_string()),
        timestamp: chrono::Utc::now(),
    };
    let content_proof = security_manager
        .generate_content_proof(&sensitive_entry, &workflow_content_statement, &context)
        .await?;

    // 6. Perform secure computation
    let computation_result = security_manager
        .secure_compute(&[encrypted_entry.clone()], SecureOperation::Count, &context)
        .await?;

    // 7. Verify all operations completed successfully
    assert!(!access_proof.id.is_empty());
    assert!(!content_proof.id.is_empty());
    assert!(computation_result.privacy_preserved);

    // 8. Collect final metrics
    let final_metrics = security_manager.get_security_metrics(&context).await?;
    assert!(
        final_metrics
            .encryption_metrics
            .total_homomorphic_encryptions
            > 0
    );
    assert!(final_metrics.privacy_metrics.total_privatizations > 0);

    if let Some(ref zk_metrics) = final_metrics.zero_knowledge_metrics {
        assert!(zk_metrics.total_proofs_generated >= 2); // Access + content proofs
    }

    Ok(())
}
