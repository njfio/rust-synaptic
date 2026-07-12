// Phase 4: Security & Privacy Example
// Demonstrates state-of-the-art security features including zero-knowledge architecture,
// homomorphic encryption, differential privacy, and advanced access control

use synaptic::security::{
    privacy::{PrivacyQuery, PrivacyQueryType},
    zero_knowledge::{AccessType, ContentPredicate},
    Permission, SecureOperation, SecurityConfig, SecurityContext, SecurityManager,
};
use synaptic::MemoryEntry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Synaptic Phase 4: Security & Privacy Demo");
    println!("============================================\n");

    // Create advanced security configuration
    let security_config = SecurityConfig {
        enable_zero_knowledge: true,
        enable_homomorphic_encryption: true,
        enable_differential_privacy: true,
        privacy_budget: 10.0,
        encryption_key_size: 256,
        ..Default::default()
    };

    // Initialize security manager
    let mut security_manager = SecurityManager::new(security_config.clone()).await?;
    println!(" Security Manager initialized with all Phase 4 features enabled");

    // Create security contexts for different users
    let admin_context = SecurityContext::new("admin_user".to_string(), vec!["admin".to_string()]);
    let user_context = SecurityContext::new("regular_user".to_string(), vec!["user".to_string()]);

    println!("\n 1. HOMOMORPHIC ENCRYPTION DEMO");
    println!("==================================");

    // Create test memory entries
    let sensitive_entry = MemoryEntry::new(
        "sensitive_data".to_string(),
        "This contains confidential financial information: $50,000 budget".to_string(),
        synaptic::MemoryType::LongTerm,
    );

    let public_entry = MemoryEntry::new(
        "public_data".to_string(),
        "This is public information about our project timeline".to_string(),
        synaptic::MemoryType::ShortTerm,
    );

    // Encrypt memories with homomorphic encryption
    println!("🔒 Encrypting sensitive memory with homomorphic encryption...");
    let encrypted_sensitive = security_manager
        .encrypt_memory(&sensitive_entry, &admin_context)
        .await?;
    println!(
        "    Encrypted with algorithm: {}",
        encrypted_sensitive.encryption_algorithm
    );
    println!("    Privacy level: {:?}", encrypted_sensitive.privacy_level);
    println!("    Is homomorphic: {}", encrypted_sensitive.is_homomorphic);

    println!("🔒 Encrypting public memory with standard encryption...");
    let encrypted_public = security_manager
        .encrypt_memory(&public_entry, &user_context)
        .await?;
    println!(
        "    Encrypted with algorithm: {}",
        encrypted_public.encryption_algorithm
    );

    // Perform secure computation on encrypted data
    println!("\n🧮 Performing secure computation on encrypted data...");
    let encrypted_entries = vec![encrypted_sensitive.clone(), encrypted_public.clone()];
    let computation_result = security_manager
        .secure_compute(&encrypted_entries, SecureOperation::Count, &admin_context)
        .await?;
    println!(
        "    Secure computation completed: {:?}",
        computation_result.operation
    );
    println!(
        "    Privacy preserved: {}",
        computation_result.privacy_preserved
    );

    println!("\n 2. ZERO-KNOWLEDGE PROOFS DEMO");
    println!("=================================");

    // Generate zero-knowledge proof for memory access
    println!(" Generating zero-knowledge proof for memory access...");
    security_manager.register_zk_prover(&admin_context.user_id)?;
    let access_statement = synaptic::security::zero_knowledge::AccessStatement {
        memory_key: "sensitive_data".to_string(),
        user_id: admin_context.user_id.clone(),
        access_type: AccessType::Read,
        timestamp: chrono::Utc::now(),
    };
    let access_proof = security_manager
        .generate_access_proof(&access_statement, &admin_context)
        .await?;
    println!("    Access proof generated: {}", access_proof.id);
    println!("    Statement hash: {}", access_proof.statement_hash);

    // Generate content proof without revealing content
    println!(" Generating zero-knowledge proof for content properties...");
    let content_statement = synaptic::security::zero_knowledge::ContentStatement {
        memory_key: sensitive_entry.key.clone(),
        predicate: ContentPredicate::ContainsKeyword("financial".to_string()),
        timestamp: chrono::Utc::now(),
    };
    let content_proof = security_manager
        .generate_content_proof(&sensitive_entry, &content_statement, &admin_context)
        .await?;
    println!("    Content proof generated: {}", content_proof.id);
    println!("    Proves content contains 'financial' without revealing content");

    println!("\n 3. DIFFERENTIAL PRIVACY DEMO");
    println!("================================");

    // Apply differential privacy to memory entries
    println!("🔒 Applying differential privacy to sensitive data...");
    let privatized_entry = security_manager
        .privacy_manager
        .apply_differential_privacy(&sensitive_entry, &admin_context)
        .await?;
    println!("    Original: {}", sensitive_entry.value);
    println!("    Privatized: {}", privatized_entry.value);

    // Generate differentially private statistics
    println!(" Generating private statistics...");
    let test_entries = vec![sensitive_entry.clone(), public_entry.clone()];
    let privacy_query = PrivacyQuery {
        query_type: PrivacyQueryType::Count,
        sensitivity: 1.0,
        bins: None,
        quantile: None,
    };

    let private_stats = security_manager
        .privacy_manager
        .generate_private_statistics(&test_entries, privacy_query, &admin_context)
        .await?;
    println!("    Private count: {:.2}", private_stats.result);
    println!("    Epsilon used: {}", private_stats.epsilon_used);
    println!(
        "    Confidence interval: ({:.2}, {:.2})",
        private_stats.confidence_interval.0, private_stats.confidence_interval.1
    );

    println!("\n 4. ADVANCED ACCESS CONTROL DEMO");
    println!("===================================");

    // Test access control with different user contexts
    println!(" Testing access control permissions...");

    // Admin should have access
    match security_manager
        .access_control
        .check_permission(&admin_context, Permission::ReadAnalytics)
        .await
    {
        Ok(_) => println!("    Admin has analytics access"),
        Err(e) => println!("    Admin access denied: {}", e),
    }

    // Regular user should not have analytics access
    match security_manager
        .access_control
        .check_permission(&user_context, Permission::ReadAnalytics)
        .await
    {
        Ok(_) => println!("     User has analytics access (unexpected)"),
        Err(_) => println!("    User correctly denied analytics access"),
    }

    println!("\n 5. SECURITY METRICS & MONITORING");
    println!("====================================");

    // Get comprehensive security metrics
    let security_metrics = security_manager
        .get_security_metrics(&admin_context)
        .await?;
    println!(" Security Metrics Summary:");
    println!(
        "    Encryption Operations: {}",
        security_metrics.encryption_metrics.total_encryptions
    );
    println!(
        "    Privacy Operations: {}",
        security_metrics.privacy_metrics.total_privatizations
    );
    println!(
        "    Access Checks: {}",
        security_metrics.access_metrics.total_access_time_ms
    );

    if let Some(ref zk_metrics) = security_metrics.zero_knowledge_metrics {
        println!(
            "    Zero-Knowledge Proofs Generated: {}",
            zk_metrics.total_proofs_generated
        );
        println!(
            "    Zero-Knowledge Proofs Verified: {}",
            zk_metrics.total_proofs_verified
        );
        println!(
            "    Verification Success Rate: {:.1}%",
            zk_metrics.verification_success_rate
        );
    }

    println!("\n 6. PRIVACY BUDGET MANAGEMENT");
    println!("================================");

    // Check remaining privacy budget
    let remaining_budget = security_manager
        .privacy_manager
        .get_remaining_budget(&admin_context.user_id)
        .await?;
    println!(" Privacy Budget Status:");
    println!("    Remaining budget for admin: {:.2}", remaining_budget);
    println!(
        "    Total epsilon consumed: {:.2}",
        security_metrics.privacy_metrics.total_epsilon_consumed
    );

    println!("\n Phase 4 Security & Privacy Demo Complete!");
    println!("=============================================");
    println!("🔒 All advanced security features demonstrated:");
    println!("    Homomorphic Encryption - Compute on encrypted data");
    println!("    Zero-Knowledge Proofs - Verify without revealing");
    println!("    Differential Privacy - Statistical privacy guarantees");
    println!("    Advanced Access Control - Fine-grained permissions");
    println!("    Security Monitoring - Comprehensive metrics");
    println!("    Privacy Budget Management - Controlled privacy spending");

    #[cfg(not(feature = "security"))]
    {
        println!(" Security features not enabled. Please run with:");
        println!("   cargo run --example phase4_security_privacy --features security");
    }

    Ok(())
}
