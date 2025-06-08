// Phase 4: Security & Privacy Example
// Demonstrates state-of-the-art security features including zero-knowledge architecture,
// homomorphic encryption, differential privacy, and advanced access control

use synaptic::{AgentMemory, MemoryConfig, MemoryEntry};

#[cfg(feature = "security")]
use synaptic::security::{
    SecurityManager, SecurityConfig, SecurityContext, 
    access_control::{AuthenticationCredentials, AuthenticationType},
    Permission, SecureOperation
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Synaptic Phase 4: Security & Privacy Demo");
    println!("============================================\n");

    #[cfg(feature = "security")]
    {
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

        // Add some basic roles and permissions
        security_manager.access_control.add_role(
            "user".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory]
        ).await?;

        security_manager.access_control.add_role(
            "admin".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory, Permission::ReadAnalytics, Permission::ManageUsers]
        ).await?;

        println!("✅ Security Manager initialized with advanced features");

        // Create memory system with security enabled
        let memory_config = MemoryConfig {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            ..Default::default()
        };

        let mut memory = AgentMemory::new(memory_config).await?;
        println!("✅ Memory system initialized\n");

        // Demonstrate authentication and access control
        let authenticated_context = demonstrate_authentication(&mut security_manager).await?;

        // Demonstrate encryption and zero-knowledge operations
        demonstrate_encryption(&mut security_manager, &mut memory, &authenticated_context).await?;

        // Demonstrate differential privacy
        demonstrate_differential_privacy(&mut security_manager, &mut memory).await?;

        // Demonstrate homomorphic computation
        demonstrate_homomorphic_computation(&mut security_manager, &mut memory).await?;

        // Demonstrate audit logging and compliance
        demonstrate_audit_and_compliance(&mut security_manager).await?;

        // Show security metrics
        show_security_metrics(&mut security_manager).await?;

        println!("\n🎉 Phase 4 Security & Privacy Demo Completed!");
        println!("✅ All advanced security features demonstrated successfully");
    }

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
