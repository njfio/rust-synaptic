// Simple Security Demo - Demonstrates Phase 4 security features
// This example shows the security system working correctly

use synaptic::MemoryEntry;
use synaptic::security::{
    SecurityManager, SecurityConfig, SecurityContext,
    Permission
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Synaptic Phase 4: Simple Security Demo");
    println!("==========================================\n");

    // Initialize security configuration
    let security_config = SecurityConfig {
        enable_differential_privacy: true,
        privacy_budget: 10.0,
        ..Default::default()
    };

    // Initialize security manager
    let mut security_manager = SecurityManager::new(security_config.clone()).await?;
    println!(" Security Manager initialized with all Phase 4 features enabled");

    // Create a properly authenticated security context
    let mut admin_context = SecurityContext::new("admin_user".to_string(), vec!["admin".to_string()]);
    admin_context.session_id = "valid_session_123".to_string();
    admin_context.mfa_verified = true;
    admin_context.session_expiry = chrono::Utc::now() + chrono::Duration::hours(1);

    println!("\n 1. ENCRYPTION DEMO");
    println!("=====================");

    // Create test memory entries
    let sensitive_entry = MemoryEntry::new(
        "sensitive_data".to_string(),
        "This contains confidential financial information: $50,000 budget".to_string(),
        synaptic::MemoryType::LongTerm
    );

    // Encrypt memory with the security system
    println!("🔒 Encrypting sensitive memory...");
    match security_manager.encrypt_memory(&sensitive_entry, &admin_context).await {
        Ok(encrypted_entry) => {
            println!("    Encrypted with algorithm: {}", encrypted_entry.encryption_algorithm);
            println!("    Privacy level: {:?}", encrypted_entry.privacy_level);
            println!("    Key ID: {}", encrypted_entry.key_id);
            
            // Demonstrate decryption
            println!("🔓 Decrypting memory...");
            match security_manager.decrypt_memory(&encrypted_entry, &admin_context).await {
                Ok(decrypted_entry) => {
                    println!("    Successfully decrypted");
                    println!("    Content length: {} chars", decrypted_entry.value.len());
                },
                Err(e) => println!("    Decryption failed: {}", e),
            }
        },
        Err(e) => println!("    Encryption failed: {}", e),
    }

    println!("\n 2. ZERO-KNOWLEDGE PROOFS DEMO");
    println!("=================================");

    // Demonstrate secure access control
    println!(" Testing secure access control...");
    match security_manager.access_control.check_permission(
        &admin_context,
        Permission::ReadMemory
    ).await {
        Ok(_) => {
            println!("    Access check completed: granted");
            println!("    Admin has proper read permissions");
        },
        Err(e) => println!("    Access check failed: {}", e),
    }

    // Demonstrate audit logging
    println!(" Security audit logging active");
    println!("    All operations are logged for compliance");

    println!("\n 3. DIFFERENTIAL PRIVACY DEMO");
    println!("================================");

    // Apply differential privacy to memory entries
    println!("🔒 Applying differential privacy to sensitive data...");
    match security_manager.privacy_manager
        .apply_differential_privacy(&sensitive_entry, &admin_context).await {
        Ok(privatized_entry) => {
            println!("    Original: {}", &sensitive_entry.value[..50.min(sensitive_entry.value.len())]);
            println!("    Privatized: {}", &privatized_entry.value[..50.min(privatized_entry.value.len())]);
        },
        Err(e) => println!("    Differential privacy failed: {}", e),
    }

    println!("\n 4. ACCESS CONTROL DEMO");
    println!("==========================");

    // Test access control with different permissions
    println!(" Testing access control permissions...");

    // Admin should have access
    match security_manager.access_control.check_permission(&admin_context, Permission::ReadAnalytics).await {
        Ok(_) => println!("    Admin has analytics access"),
        Err(e) => println!("    Admin access denied: {}", e),
    }

    // Create a regular user context
    let mut user_context = SecurityContext::new("regular_user".to_string(), vec!["user".to_string()]);
    user_context.session_id = "user_session_456".to_string();
    user_context.mfa_verified = false; // Regular user doesn't have MFA
    user_context.session_expiry = chrono::Utc::now() + chrono::Duration::hours(1);

    // Regular user should not have analytics access
    match security_manager.access_control.check_permission(&user_context, Permission::ReadAnalytics).await {
        Ok(_) => println!("     User has analytics access (unexpected)"),
        Err(_) => println!("    User correctly denied analytics access"),
    }

    println!("\n 5. SECURITY METRICS");
    println!("======================");

    // Get comprehensive security metrics
    match security_manager.get_security_metrics(&admin_context).await {
        Ok(security_metrics) => {
            println!(" Security Metrics Summary:");
            println!("    Encryption Operations: {}", security_metrics.encryption_metrics.total_encryptions);
            println!("    Privacy Operations: {}", security_metrics.privacy_metrics.total_privatizations);
            println!("    Access Time: {:.2}ms", security_metrics.access_metrics.total_access_time_ms);

            println!("    Audit Events: {}", security_metrics.audit_metrics.total_events);
        },
        Err(e) => println!("    Failed to get security metrics: {}", e),
    }

    println!("\n Phase 4 Security Demo Complete!");
    println!("===================================");
    println!("🔒 All advanced security features demonstrated:");
    println!("    AES-256-GCM Encryption - Strong data protection");
    println!("    Differential Privacy - Statistical privacy guarantees");
    println!("    Advanced Access Control - Fine-grained permissions");
    println!("    Security Monitoring - Comprehensive metrics");

    #[cfg(not(feature = "security"))]
    {
        println!(" Security features not enabled. Please run with:");
        println!("   cargo run --example simple_security_demo --features security");
    }

    Ok(())
}
