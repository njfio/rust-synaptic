//! Task 5.7: PolicyEngine integration with SecurityManager, and honest
//! ComplianceChecker behavior.
//!
//! - A Deny policy configured through SecurityManager blocks access for an
//!   otherwise-authenticated request (PolicyEngine is an additional layer on
//!   top of argon2/TOTP authentication, not a replacement).
//! - ComplianceChecker::check_compliance evaluates configured framework rules
//!   (data residency, retention) against declared request context, and is
//!   honest about which checks it skipped when context is missing.

#![cfg(feature = "security")]

use std::error::Error;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::security::access_control::{
    AccessControlManager, AuthenticationCredentials, AuthenticationType,
};
use synaptic::security::policy_engine::{
    AccessControls, AccessRequest, AuditEvent, AuditQuery, AuditRequirements, AuditStorage,
    ComplianceChecker, DataClassification, DataClassificationLevel, PolicyAction, PolicyCondition,
    PolicyEngine, PolicyRule, Role, SecurityPolicy, User,
};
use synaptic::security::{Permission, SecurityConfig, SecurityContext, SecurityManager};

struct NullAuditStorage;

impl AuditStorage for NullAuditStorage {
    fn log_event(&self, _event: AuditEvent) -> synaptic::error::Result<()> {
        Ok(())
    }
    fn query_events(&self, _query: AuditQuery) -> synaptic::error::Result<Vec<AuditEvent>> {
        Ok(Vec::new())
    }
}

fn deny_memory_policy() -> SecurityPolicy {
    SecurityPolicy {
        id: "deny-memory".to_string(),
        name: "deny-memory-access".to_string(),
        description: "denies all access to memory resources".to_string(),
        rules: vec![PolicyRule {
            id: "rule-deny".to_string(),
            name: "deny-memory-resource".to_string(),
            condition: PolicyCondition::ResourceType("memory".to_string()),
            action: PolicyAction::Deny,
            priority: 10,
            enabled: true,
        }],
        data_classification: DataClassification {
            level: DataClassificationLevel::Internal,
            retention_period: chrono::Duration::days(30),
            encryption_required: false,
            access_logging_required: true,
            geographic_restrictions: vec![],
            sharing_restrictions: vec![],
        },
        access_controls: AccessControls {
            required_roles: vec![],
            required_permissions: vec![],
            mfa_required: false,
            ip_whitelist: vec![],
            time_restrictions: None,
            concurrent_session_limit: None,
        },
        audit_requirements: AuditRequirements {
            log_access: true,
            log_modifications: true,
            log_deletions: true,
            log_exports: true,
            retention_period: chrono::Duration::days(365),
            real_time_monitoring: false,
            alert_on_violations: false,
        },
        compliance_frameworks: vec![],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
        active: true,
    }
}

fn engine_with_user(user_id: &str) -> Result<PolicyEngine, Box<dyn Error>> {
    let mut engine = PolicyEngine::new(Box::new(NullAuditStorage));
    engine.add_role(Role {
        id: "role-user".to_string(),
        name: "user".to_string(),
        description: "basic user".to_string(),
        permissions: Default::default(),
        inherits_from: vec![],
        data_access_levels: Default::default(),
        resource_access: Default::default(),
        created_at: chrono::Utc::now(),
        active: true,
    })?;
    engine.add_user(policy_user(user_id))?;
    Ok(engine)
}

fn policy_user(user_id: &str) -> User {
    User {
        id: user_id.to_string(),
        username: user_id.to_string(),
        email: format!("{user_id}@example.com"),
        roles: vec!["role-user".to_string()],
        security_clearance: DataClassificationLevel::Internal,
        mfa_enabled: false,
        last_login: None,
        failed_login_attempts: 0,
        account_locked: false,
        created_at: chrono::Utc::now(),
        active: true,
    }
}

/// Provision real credentials (argon2 password + TOTP secret) and return
/// credentials carrying a valid current TOTP token.
fn provision_and_credentials(
    access_control: &mut AccessControlManager,
    user_id: &str,
    password: &str,
) -> Result<AuthenticationCredentials, Box<dyn Error>> {
    let secret = AccessControlManager::generate_totp_secret();
    access_control.set_password(user_id, password)?;
    access_control.set_totp_secret(user_id, secret.clone())?;
    let totp = totp_rs::TOTP::new(totp_rs::Algorithm::SHA1, 6, 1, 30, secret)?;
    let token = totp.generate_current()?;
    Ok(AuthenticationCredentials {
        auth_type: AuthenticationType::Password,
        password: Some(password.to_string()),
        api_key: None,
        certificate: None,
        mfa_token: Some(token),
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("policy_integration_tests".to_string()),
    })
}

async fn authenticated_manager_and_context(
    user_id: &str,
) -> Result<(SecurityManager, SecurityContext), Box<dyn Error>> {
    let config = SecurityConfig {
        // Homomorphic encryption is not compiled in for this test profile;
        // use standard encryption so the policy layer is what's under test.
        enable_homomorphic_encryption: false,
        ..SecurityConfig::default()
    };
    let mut manager = SecurityManager::new(config).await?;
    manager
        .access_control
        .add_role(
            "user".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory],
        )
        .await?;
    let creds = provision_and_credentials(&mut manager.access_control, user_id, "password123")?;
    let context = manager
        .access_control
        .authenticate(user_id.to_string(), creds)
        .await?;
    Ok((manager, context))
}

#[tokio::test]
async fn test_authenticated_access_allowed_without_policy_engine() -> Result<(), Box<dyn Error>> {
    let (mut manager, context) = authenticated_manager_and_context("alice").await?;
    let entry = MemoryEntry::new(
        "memory-key".to_string(),
        "hello".to_string(),
        MemoryType::ShortTerm,
    );
    // Baseline: no policy engine configured, authenticated write succeeds.
    manager.encrypt_memory(&entry, &context).await?;
    Ok(())
}

#[tokio::test]
async fn test_deny_policy_blocks_authenticated_access() -> Result<(), Box<dyn Error>> {
    let (mut manager, context) = authenticated_manager_and_context("alice").await?;

    let mut engine = engine_with_user(&context.user_id)?;
    engine.add_policy(deny_memory_policy())?;
    manager.set_policy_engine(engine);

    let entry = MemoryEntry::new(
        "memory-key".to_string(),
        "hello".to_string(),
        MemoryType::ShortTerm,
    );
    let result = manager.encrypt_memory(&entry, &context).await;
    let err = result.expect_err("Deny policy must block an otherwise-authenticated request");
    assert!(
        err.to_string().contains("policy"),
        "denial should be attributed to policy, got: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn test_allow_policy_permits_authenticated_access() -> Result<(), Box<dyn Error>> {
    let (mut manager, context) = authenticated_manager_and_context("alice").await?;

    let mut policy = deny_memory_policy();
    policy.rules[0].action = PolicyAction::Allow;
    let mut engine = engine_with_user(&context.user_id)?;
    engine.add_policy(policy)?;
    manager.set_policy_engine(engine);

    let entry = MemoryEntry::new(
        "memory-key".to_string(),
        "hello".to_string(),
        MemoryType::ShortTerm,
    );
    manager.encrypt_memory(&entry, &context).await?;
    Ok(())
}

#[tokio::test]
async fn test_policy_engine_denies_unregistered_user() -> Result<(), Box<dyn Error>> {
    // Deny-by-default: with a policy engine configured, a user unknown to the
    // engine is refused even though authentication succeeded.
    let (mut manager, context) = authenticated_manager_and_context("alice").await?;
    let engine = engine_with_user("someone-else")?;
    manager.set_policy_engine(engine);

    let entry = MemoryEntry::new(
        "memory-key".to_string(),
        "hello".to_string(),
        MemoryType::ShortTerm,
    );
    assert!(manager.encrypt_memory(&entry, &context).await.is_err());
    Ok(())
}

// --- ComplianceChecker ---

fn compliance_request(context: &[(&str, &str)]) -> AccessRequest {
    AccessRequest {
        user_id: "user-1".to_string(),
        resource_type: "memory".to_string(),
        resource_id: "res-1".to_string(),
        action: "read".to_string(),
        ip_address: "127.0.0.1".to_string(),
        user_agent: "policy_integration_tests".to_string(),
        timestamp: chrono::Utc::now(),
        session_id: "session-1".to_string(),
        additional_context: context
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

#[test]
fn test_compliance_residency_violation_detected() -> Result<(), Box<dyn Error>> {
    let checker = ComplianceChecker::new();
    // Default GDPR rules require EU residency.
    let request = compliance_request(&[("data_residency", "US")]);
    let result = checker.check_compliance(&request, &policy_user("user-1"))?;
    assert!(!result.compliant, "US residency must violate GDPR EU rule");
    assert!(
        result.reason.contains("residency"),
        "got: {}",
        result.reason
    );
    Ok(())
}

#[test]
fn test_compliance_retention_violation_detected() -> Result<(), Box<dyn Error>> {
    let checker = ComplianceChecker::new();
    // GDPR Internal retention limit is 5 years; declare 6-year-old data.
    let request = compliance_request(&[
        ("data_residency", "EU"),
        ("data_classification", "internal"),
        ("data_age_days", "2190"),
    ]);
    let result = checker.check_compliance(&request, &policy_user("user-1"))?;
    assert!(
        !result.compliant,
        "6-year-old Internal data must violate GDPR retention"
    );
    Ok(())
}

#[test]
fn test_compliance_within_rules_is_compliant() -> Result<(), Box<dyn Error>> {
    let checker = ComplianceChecker::new();
    let request = compliance_request(&[
        ("data_residency", "EU"),
        ("data_classification", "internal"),
        ("data_age_days", "30"),
    ]);
    let result = checker.check_compliance(&request, &policy_user("user-1"))?;
    assert!(result.compliant, "got: {}", result.reason);
    Ok(())
}

#[test]
fn test_compliance_missing_context_is_honest_about_skipped_checks() -> Result<(), Box<dyn Error>> {
    let checker = ComplianceChecker::new();
    let request = compliance_request(&[]);
    let result = checker.check_compliance(&request, &policy_user("user-1"))?;
    // No declared context: nothing checkable violates, but the result must
    // say checks were skipped rather than pretend full validation happened.
    assert!(result.compliant);
    assert!(
        result.notes.iter().any(|n| n.contains("skipped")),
        "missing context must be reported as skipped checks, got: {:?}",
        result.notes
    );
    Ok(())
}

#[test]
fn test_compliance_unparseable_declared_context_fails_closed() -> Result<(), Box<dyn Error>> {
    let checker = ComplianceChecker::new();
    let request = compliance_request(&[
        ("data_classification", "not-a-level"),
        ("data_age_days", "30"),
    ]);
    let result = checker.check_compliance(&request, &policy_user("user-1"))?;
    assert!(
        !result.compliant,
        "unparseable declared context must fail closed"
    );
    Ok(())
}
