// Phase 4: Security & Privacy Example
// Demonstrates state-of-the-art security features including zero-knowledge architecture,
// homomorphic encryption, differential privacy, and advanced access control

use synaptic::{AgentMemory, MemoryConfig, MemoryEntry};
use synaptic::security::{
    SecurityManager, SecurityConfig, SecurityContext,
    Permission, SecureOperation,
    access_control::{AuthenticationCredentials, AuthenticationType},
    zero_knowledge::{AccessType, ContentPredicate},
    privacy::{PrivacyQuery, PrivacyQueryType}
};

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
    println!("✅ Security Manager initialized with all Phase 4 features enabled");

    // Create security contexts for different users
    let admin_context = SecurityContext::new("admin_user".to_string(), vec!["admin".to_string()]);
    let user_context = SecurityContext::new("regular_user".to_string(), vec!["user".to_string()]);

    println!("\n🔐 1. HOMOMORPHIC ENCRYPTION DEMO");
    println!("==================================");

    // Create test memory entries
    let sensitive_entry = MemoryEntry::new(
        "sensitive_data".to_string(),
        "This contains confidential financial information: $50,000 budget".to_string(),
        synaptic::MemoryType::LongTerm
    );

    let public_entry = MemoryEntry::new(
        "public_data".to_string(),
        "This is public information about our project timeline".to_string(),
        synaptic::MemoryType::ShortTerm
    );

    // Encrypt memories with homomorphic encryption
    println!("🔒 Encrypting sensitive memory with homomorphic encryption...");
    let encrypted_sensitive = security_manager.encrypt_memory(&sensitive_entry, &admin_context).await?;
    println!("   ✅ Encrypted with algorithm: {}", encrypted_sensitive.encryption_algorithm);
    println!("   ✅ Privacy level: {:?}", encrypted_sensitive.privacy_level);
    println!("   ✅ Is homomorphic: {}", encrypted_sensitive.is_homomorphic);

    println!("🔒 Encrypting public memory with standard encryption...");
    let encrypted_public = security_manager.encrypt_memory(&public_entry, &user_context).await?;
    println!("   ✅ Encrypted with algorithm: {}", encrypted_public.encryption_algorithm);

    // Perform secure computation on encrypted data
    println!("\n🧮 Performing secure computation on encrypted data...");
    let encrypted_entries = vec![encrypted_sensitive.clone(), encrypted_public.clone()];
    let computation_result = security_manager.secure_compute(
        &encrypted_entries,
        SecureOperation::Count,
        &admin_context
    ).await?;
    println!("   ✅ Secure computation completed: {:?}", computation_result.operation);
    println!("   ✅ Privacy preserved: {}", computation_result.privacy_preserved);

    println!("\n🔍 2. ZERO-KNOWLEDGE PROOFS DEMO");
    println!("=================================");

    // Generate zero-knowledge proof for memory access
    println!("🔐 Generating zero-knowledge proof for memory access...");
    let access_proof = security_manager.generate_access_proof(
        "sensitive_data",
        &admin_context,
        AccessType::Read
    ).await?;
    println!("   ✅ Access proof generated: {}", access_proof.id);
    println!("   ✅ Statement hash: {}", access_proof.statement_hash);

    // Generate content proof without revealing content
    println!("🔐 Generating zero-knowledge proof for content properties...");
    let content_proof = security_manager.generate_content_proof(
        &sensitive_entry,
        ContentPredicate::ContainsKeyword("financial".to_string()),
        &admin_context
    ).await?;
    println!("   ✅ Content proof generated: {}", content_proof.id);
    println!("   ✅ Proves content contains 'financial' without revealing content");

    println!("\n📊 3. DIFFERENTIAL PRIVACY DEMO");
    println!("================================");

    // Apply differential privacy to memory entries
    println!("🔒 Applying differential privacy to sensitive data...");
    let privatized_entry = security_manager.privacy_manager
        .apply_differential_privacy(&sensitive_entry, &admin_context).await?;
    println!("   ✅ Original: {}", sensitive_entry.value);
    println!("   ✅ Privatized: {}", privatized_entry.value);

    // Generate differentially private statistics
    println!("📈 Generating private statistics...");
    let test_entries = vec![sensitive_entry.clone(), public_entry.clone()];
    let privacy_query = PrivacyQuery {
        query_type: PrivacyQueryType::Count,
        sensitivity: 1.0,
        bins: None,
        quantile: None,
    };

    let private_stats = security_manager.privacy_manager
        .generate_private_statistics(&test_entries, privacy_query, &admin_context).await?;
    println!("   ✅ Private count: {:.2}", private_stats.result);
    println!("   ✅ Epsilon used: {}", private_stats.epsilon_used);
    println!("   ✅ Confidence interval: ({:.2}, {:.2})",
             private_stats.confidence_interval.0,
             private_stats.confidence_interval.1);

    println!("\n🛡️ 4. ADVANCED ACCESS CONTROL DEMO");
    println!("===================================");

    // Test access control with different user contexts
    println!("🔐 Testing access control permissions...");

    // Admin should have access
    match security_manager.access_control.check_permission(&admin_context, Permission::ReadAnalytics).await {
        Ok(_) => println!("   ✅ Admin has analytics access"),
        Err(e) => println!("   ❌ Admin access denied: {}", e),
    }

    // Regular user should not have analytics access
    match security_manager.access_control.check_permission(&user_context, Permission::ReadAnalytics).await {
        Ok(_) => println!("   ⚠️  User has analytics access (unexpected)"),
        Err(_) => println!("   ✅ User correctly denied analytics access"),
    }

    println!("\n📊 5. SECURITY METRICS & MONITORING");
    println!("====================================");

    // Get comprehensive security metrics
    let security_metrics = security_manager.get_security_metrics(&admin_context).await?;
    println!("🔍 Security Metrics Summary:");
    println!("   📈 Encryption Operations: {}", security_metrics.encryption_metrics.total_encryptions);
    println!("   📈 Privacy Operations: {}", security_metrics.privacy_metrics.total_privatizations);
    println!("   📈 Access Checks: {}", security_metrics.access_metrics.total_access_time_ms);

    if let Some(ref zk_metrics) = security_metrics.zero_knowledge_metrics {
        println!("   📈 Zero-Knowledge Proofs Generated: {}", zk_metrics.total_proofs_generated);
        println!("   📈 Zero-Knowledge Proofs Verified: {}", zk_metrics.total_proofs_verified);
        println!("   📈 Verification Success Rate: {:.1}%", zk_metrics.verification_success_rate);
    }

    println!("\n🎯 6. PRIVACY BUDGET MANAGEMENT");
    println!("================================");

    // Check remaining privacy budget
    let remaining_budget = security_manager.privacy_manager
        .get_remaining_budget(&admin_context.user_id).await?;
    println!("🔍 Privacy Budget Status:");
    println!("   📊 Remaining budget for admin: {:.2}", remaining_budget);
    println!("   📊 Total epsilon consumed: {:.2}", security_metrics.privacy_metrics.total_epsilon_consumed);

    println!("\n✅ Phase 4 Security & Privacy Demo Complete!");
    println!("=============================================");
    println!("🔒 All advanced security features demonstrated:");
    println!("   ✅ Homomorphic Encryption - Compute on encrypted data");
    println!("   ✅ Zero-Knowledge Proofs - Verify without revealing");
    println!("   ✅ Differential Privacy - Statistical privacy guarantees");
    println!("   ✅ Advanced Access Control - Fine-grained permissions");
    println!("   ✅ Security Monitoring - Comprehensive metrics");
    println!("   ✅ Privacy Budget Management - Controlled privacy spending");

    #[cfg(not(feature = "security"))]
    {
        println!("❌ Security features not enabled. Please run with:");
        println!("   cargo run --example phase4_security_privacy --features security");
    }

    Ok(())
}

#[cfg(feature = "security")]
async fn demonstrate_authentication(security_manager: &mut SecurityManager) -> Result<SecurityContext, Box<dyn std::error::Error>> {
    println!("🔐 Authentication & Access Control Demo");
    println!("--------------------------------------");

    // Create authentication credentials
    let credentials = AuthenticationCredentials {
        auth_type: AuthenticationType::Password,
        password: Some("secure_password_123".to_string()),
        api_key: None,
        certificate: None,
        mfa_token: Some("123456".to_string()),
        ip_address: Some("192.168.1.100".to_string()),
        user_agent: Some("Synaptic-Client/1.0".to_string()),
    };

    // Authenticate user
    let context = security_manager.access_control.authenticate("alice_engineer".to_string(), credentials).await?;
    println!("✅ User authenticated successfully");
    println!("   User ID: {}", context.user_id);
    println!("   Session ID: {}", context.session_id);
    println!("   Roles: {:?}", context.roles);
    println!("   MFA Verified: {}", context.mfa_verified);

    // Test permission checks
    let permissions_to_test = vec![
        Permission::ReadMemory,
        Permission::WriteMemory,
        Permission::ReadAnalytics,
        Permission::ManageUsers, // This should fail for regular user
    ];

    for permission in permissions_to_test {
        match security_manager.access_control.check_permission(&context, permission.clone()).await {
            Ok(()) => println!("✅ Permission granted: {:?}", permission),
            Err(_) => println!("❌ Permission denied: {:?}", permission),
        }
    }

    println!();
    Ok(context)
}

#[cfg(feature = "security")]
async fn demonstrate_encryption(
    security_manager: &mut SecurityManager,
    memory: &mut AgentMemory,
    context: &SecurityContext
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 Encryption & Zero-Knowledge Demo");
    println!("----------------------------------");

    // Use the authenticated context passed from the previous step

    // Create sensitive memory entries
    let sensitive_entries = vec![
        MemoryEntry::new("entry1".to_string(), "Patient medical record: John Doe, diagnosed with hypertension".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("entry2".to_string(), "Financial data: Q4 revenue $2.5M, profit margin 15%".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("entry3".to_string(), "Personal information: SSN 123-45-6789, DOB 1985-03-15".to_string(), synaptic::memory::types::MemoryType::LongTerm),
    ];

    let mut encrypted_entries = Vec::new();

    for (i, entry) in sensitive_entries.iter().enumerate() {
        // Encrypt with zero-knowledge architecture
        let encrypted_entry = security_manager.encrypt_memory(entry, context).await?;
        println!("✅ Entry {} encrypted with {} algorithm", 
                 i + 1, encrypted_entry.encryption_algorithm);
        println!("   Privacy Level: {:?}", encrypted_entry.privacy_level);
        println!("   Homomorphic: {}", encrypted_entry.is_homomorphic);
        
        encrypted_entries.push(encrypted_entry);
    }

    // Demonstrate decryption
    for (i, encrypted_entry) in encrypted_entries.iter().enumerate() {
        let decrypted_entry = security_manager.decrypt_memory(encrypted_entry, context).await?;
        println!("✅ Entry {} decrypted successfully", i + 1);
        println!("   Original length: {} chars", sensitive_entries[i].value.len());
        println!("   Decrypted length: {} chars", decrypted_entry.value.len());
    }

    println!();
    Ok(())
}

#[cfg(feature = "security")]
async fn demonstrate_differential_privacy(
    security_manager: &mut SecurityManager,
    memory: &mut AgentMemory
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Differential Privacy Demo");
    println!("---------------------------");

    // Create a dataset of memory entries
    let dataset = vec![
        MemoryEntry::new("user1".to_string(), "User age: 25, salary: $50000".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("user2".to_string(), "User age: 30, salary: $65000".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("user3".to_string(), "User age: 35, salary: $80000".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("user4".to_string(), "User age: 28, salary: $55000".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("user5".to_string(), "User age: 42, salary: $95000".to_string(), synaptic::memory::types::MemoryType::LongTerm),
    ];

    let mut context = SecurityContext::new("data_analyst".to_string(), vec!["analyst".to_string()]);
    context.mfa_verified = true;

    // Apply differential privacy to individual entries
    println!("📊 Applying differential privacy to individual entries:");
    for (i, entry) in dataset.iter().enumerate() {
        let privatized_entry = security_manager.privacy_manager
            .apply_differential_privacy(entry, &context).await?;
        
        println!("   Entry {}: '{}' -> '{}'", 
                 i + 1, 
                 &entry.value[..30.min(entry.value.len())],
                 &privatized_entry.value[..30.min(privatized_entry.value.len())]);
    }

    // Generate differentially private statistics
    use synaptic::security::privacy::{PrivacyQuery, PrivacyQueryType};
    
    let queries = vec![
        PrivacyQuery {
            query_type: PrivacyQueryType::Count,
            sensitivity: 1.0,
            bins: None,
            quantile: None,
        },
        PrivacyQuery {
            query_type: PrivacyQueryType::Average,
            sensitivity: 2.0,
            bins: None,
            quantile: None,
        },
        PrivacyQuery {
            query_type: PrivacyQueryType::Histogram,
            sensitivity: 1.0,
            bins: Some(5),
            quantile: None,
        },
    ];

    println!("\n📈 Generating differentially private statistics:");
    for query in queries {
        let stats = security_manager.privacy_manager
            .generate_private_statistics(&dataset, query.clone(), &context).await?;
        
        println!("   {:?}: {:.2} (ε = {:.2})", 
                 stats.query_type, stats.result, stats.epsilon_used);
        println!("     Confidence interval: ({:.2}, {:.2})", 
                 stats.confidence_interval.0, stats.confidence_interval.1);
    }

    // Show remaining privacy budget
    let remaining_budget = security_manager.privacy_manager
        .get_remaining_budget(&context.user_id).await?;
    println!("\n💰 Remaining privacy budget: {:.2}", remaining_budget);

    println!();
    Ok(())
}

#[cfg(feature = "security")]
async fn demonstrate_homomorphic_computation(
    security_manager: &mut SecurityManager,
    _memory: &mut AgentMemory
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Homomorphic Computation Demo");
    println!("-------------------------------");

    let mut context = SecurityContext::new("compute_user".to_string(), vec!["user".to_string()]);
    context.mfa_verified = true;

    // Create entries with numeric data
    let numeric_entries = vec![
        MemoryEntry::new("q1".to_string(), "Sales data: Q1 revenue 100000, customers 500".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("q2".to_string(), "Sales data: Q2 revenue 120000, customers 600".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("q3".to_string(), "Sales data: Q3 revenue 110000, customers 550".to_string(), synaptic::memory::types::MemoryType::LongTerm),
        MemoryEntry::new("q4".to_string(), "Sales data: Q4 revenue 130000, customers 650".to_string(), synaptic::memory::types::MemoryType::LongTerm),
    ];

    // Encrypt entries with homomorphic encryption
    let mut encrypted_entries = Vec::new();
    for entry in &numeric_entries {
        let encrypted = security_manager.encrypt_memory(entry, &context).await?;
        encrypted_entries.push(encrypted);
    }

    println!("✅ {} entries encrypted with homomorphic encryption", encrypted_entries.len());

    // Perform secure computations on encrypted data
    let operations = vec![
        SecureOperation::Sum,
        SecureOperation::Average,
        SecureOperation::Count,
        SecureOperation::Search { query: "revenue".to_string() },
        SecureOperation::Similarity { threshold: 0.8 },
    ];

    for operation in operations {
        let result = security_manager.secure_compute(
            &encrypted_entries, 
            operation.clone(), 
            &context
        ).await?;
        
        println!("✅ Secure computation completed: {:?}", operation);
        println!("   Computation ID: {}", result.computation_id);
        println!("   Privacy preserved: {}", result.privacy_preserved);
        println!("   Result size: {} bytes", result.result_data.len());
    }

    println!();
    Ok(())
}

#[cfg(feature = "security")]
async fn demonstrate_audit_and_compliance(
    security_manager: &mut SecurityManager
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Audit Logging & Compliance Demo");
    println!("----------------------------------");

    let mut context = SecurityContext::new("audit_user".to_string(), vec!["user".to_string()]);
    context.mfa_verified = true;

    // Generate some audit events
    security_manager.audit_logger.log_authentication_event(
        "audit_user", 
        "password", 
        true, 
        Some("192.168.1.200".to_string())
    ).await?;

    security_manager.audit_logger.log_memory_operation(
        &context, 
        "store_memory", 
        true
    ).await?;

    security_manager.audit_logger.log_access_decision(
        &context, 
        "read_analytics", 
        false
    ).await?;

    // Get security alerts
    let alerts = security_manager.audit_logger.get_security_alerts().await?;
    println!("🚨 Active security alerts: {}", alerts.len());
    for alert in &alerts {
        println!("   Alert: {:?} - {}", alert.alert_type, alert.message);
        println!("   Risk Level: {:?}, Time: {}", alert.risk_level, alert.timestamp);
    }

    // Generate compliance report
    let start_time = chrono::Utc::now() - chrono::Duration::hours(24);
    let end_time = chrono::Utc::now();
    
    let compliance_report = security_manager.audit_logger
        .generate_compliance_report(start_time, end_time).await?;
    
    println!("\n📊 Compliance Report (Last 24 hours):");
    println!("   Total Events: {}", compliance_report.total_events);
    println!("   Authentication Events: {}", compliance_report.authentication_events);
    println!("   Access Control Events: {}", compliance_report.access_control_events);
    println!("   Failed Operations: {}", compliance_report.failed_operations);
    println!("   High Risk Events: {}", compliance_report.high_risk_events);
    println!("   Unique Users: {}", compliance_report.unique_users.len());
    println!("   Compliance Score: {:.1}%", compliance_report.compliance_score);

    println!();
    Ok(())
}

#[cfg(feature = "security")]
async fn show_security_metrics(
    security_manager: &mut SecurityManager
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Security Metrics Summary");
    println!("---------------------------");

    let mut context = SecurityContext::new("metrics_user".to_string(), vec!["admin".to_string()]);
    context.mfa_verified = true;
    let metrics = security_manager.get_security_metrics(&context).await?;

    println!("🔐 Encryption Metrics:");
    println!("   Total Encryptions: {}", metrics.encryption_metrics.total_encryptions);
    println!("   Total Decryptions: {}", metrics.encryption_metrics.total_decryptions);
    println!("   Homomorphic Operations: {}", metrics.encryption_metrics.total_homomorphic_computations);
    println!("   Average Encryption Time: {:.2}ms", metrics.encryption_metrics.average_encryption_time_ms);
    println!("   Success Rate: {:.1}%", metrics.encryption_metrics.encryption_success_rate);

    println!("\n🔒 Privacy Metrics:");
    println!("   Total Privatizations: {}", metrics.privacy_metrics.total_privatizations);
    println!("   Privacy Queries: {}", metrics.privacy_metrics.total_queries);
    println!("   Epsilon Consumed: {:.2}", metrics.privacy_metrics.total_epsilon_consumed);
    println!("   Budget Utilization: {:.1}%", metrics.privacy_metrics.privacy_budget_utilization);

    println!("\n🚪 Access Control Metrics:");
    println!("   Successful Authentications: {}", metrics.access_metrics.total_successful_authentications);
    println!("   Failed Authentications: {}", metrics.access_metrics.total_failed_authentications);
    println!("   Permission Checks: {}", metrics.access_metrics.total_permission_checks);
    println!("   Grant Rate: {:.1}%", metrics.access_metrics.permission_grant_rate);
    println!("   Auth Success Rate: {:.1}%", metrics.access_metrics.authentication_success_rate);

    println!("\n📋 Audit Metrics:");
    println!("   Total Events: {}", metrics.audit_metrics.total_events);
    println!("   Failed Events: {}", metrics.audit_metrics.failed_events);
    println!("   Failure Rate: {:.1}%", metrics.audit_metrics.failure_rate);
    println!("   Events per Hour: {:.1}", metrics.audit_metrics.events_per_hour);

    println!();
    Ok(())
}
