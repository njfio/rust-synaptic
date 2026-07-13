//! Task 4.7: real authentication and authorization tests.
//!
//! Covers argon2 password verification, constant-time API-key verification
//! against stored SHA-256 digests, RFC 6238 TOTP MFA with a ±1 time-step
//! window, and deny-by-default policy-engine condition evaluation.

use std::error::Error;

use synaptic::security::access_control::{
    AccessControlManager, AuthenticationCredentials, AuthenticationType,
};
use synaptic::security::policy_engine::{
    AccessControls, AccessRequest, AuditEvent, AuditQuery, AuditRequirements, AuditStorage,
    DataClassification, DataClassificationLevel, PolicyAction, PolicyCondition, PolicyEngine,
    PolicyRule, Role, SecurityPolicy, TimeWindow, User,
};
use synaptic::security::SecurityConfig;

fn manager_config(require_mfa: bool) -> SecurityConfig {
    let mut config = SecurityConfig::default();
    config.access_control_policy.require_mfa = require_mfa;
    config
}

fn password_credentials(password: &str, mfa_token: Option<String>) -> AuthenticationCredentials {
    AuthenticationCredentials {
        auth_type: AuthenticationType::Password,
        password: Some(password.to_string()),
        api_key: None,
        certificate: None,
        mfa_token,
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("auth_tests".to_string()),
    }
}

fn api_key_credentials(api_key: &str) -> AuthenticationCredentials {
    AuthenticationCredentials {
        auth_type: AuthenticationType::ApiKey,
        password: None,
        api_key: Some(api_key.to_string()),
        certificate: None,
        mfa_token: None,
        ip_address: Some("127.0.0.1".to_string()),
        user_agent: Some("auth_tests".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Password authentication (argon2 PHC verification)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_correct_password_verifies() -> Result<(), Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    manager.set_password("alice", "correct horse battery staple")?;

    let context = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", None),
        )
        .await?;
    assert_eq!(context.user_id, "alice");
    Ok(())
}

#[tokio::test]
async fn test_wrong_password_fails() -> Result<(), Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    manager.set_password("alice", "correct horse battery staple")?;

    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("wrong horse battery staple", None),
        )
        .await;
    assert!(result.is_err(), "wrong password must be rejected");
    Ok(())
}

#[tokio::test]
async fn test_long_password_without_provisioning_fails() -> Result<(), Box<dyn Error>> {
    // The Phase 0 fake accepted any password with len >= 8; a user with no
    // stored credential must now always be denied.
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    let result = manager
        .authenticate(
            "ghost".to_string(),
            password_credentials("password123", None),
        )
        .await;
    assert!(result.is_err(), "unprovisioned user must be rejected");
    Ok(())
}

#[tokio::test]
async fn test_tampered_argon2_hash_fails() -> Result<(), Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    let hash = AccessControlManager::hash_password("correct horse battery staple")?;

    // Corrupt the digest portion (the final `$`-separated field of the PHC
    // string) while keeping the overall PHC shape.
    let digest_start = hash.rfind('$').map(|i| i + 1).unwrap_or(0);
    let (prefix, digest) = hash.split_at(digest_start);
    let tampered_digest: String = digest.chars().rev().collect();
    let tampered = format!("{prefix}{tampered_digest}");
    assert_ne!(tampered, hash);
    manager.set_password_hash("alice", tampered)?;

    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", None),
        )
        .await;
    assert!(
        result.is_err(),
        "tampered stored hash must fail verification"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// API-key authentication (SHA-256 digest, constant-time comparison)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_correct_api_key_passes() -> Result<(), Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    manager.set_api_key("service", "sk-live-0123456789abcdef");

    let context = manager
        .authenticate(
            "service".to_string(),
            api_key_credentials("sk-live-0123456789abcdef"),
        )
        .await?;
    assert_eq!(context.user_id, "service");
    Ok(())
}

#[tokio::test]
async fn test_wrong_api_key_fails_even_with_sk_prefix() -> Result<(), Box<dyn Error>> {
    // The Phase 0 fake accepted anything starting with "sk-". The full key
    // digest must match now, compared via subtle::ConstantTimeEq.
    let mut manager = AccessControlManager::new(&manager_config(false)).await?;
    manager.set_api_key("service", "sk-live-0123456789abcdef");

    for wrong in [
        "sk-live-0123456789abcdeX", // one character off
        "sk-anything-else",         // old prefix fake would have accepted this
        "totally-different",
    ] {
        let result = manager
            .authenticate("service".to_string(), api_key_credentials(wrong))
            .await;
        assert!(result.is_err(), "API key {wrong:?} must be rejected");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MFA (RFC 6238 TOTP, ±1 time-step window)
// ---------------------------------------------------------------------------

const TOTP_STEP: u64 = 30;

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_secs()
}

/// Avoid step-boundary races: if the current 30s step is about to roll over,
/// wait for the next one so all checks land inside a single step.
fn settle_time_step() -> u64 {
    let now = unix_now();
    let into_step = now % TOTP_STEP;
    if into_step >= TOTP_STEP - 5 {
        std::thread::sleep(std::time::Duration::from_secs(TOTP_STEP - into_step + 1));
    }
    unix_now()
}

fn totp_for(secret: &[u8], timestamp: u64) -> Result<String, Box<dyn Error>> {
    let totp = totp_rs::TOTP::new(totp_rs::Algorithm::SHA1, 6, 1, TOTP_STEP, secret.to_vec())?;
    Ok(totp.generate(timestamp))
}

async fn mfa_manager(secret: &[u8]) -> Result<AccessControlManager, Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(true)).await?;
    manager.set_password("alice", "correct horse battery staple")?;
    manager.set_totp_secret("alice", secret.to_vec())?;
    Ok(manager)
}

#[tokio::test]
async fn test_valid_totp_current_step_passes() -> Result<(), Box<dyn Error>> {
    let secret = AccessControlManager::generate_totp_secret();
    let mut manager = mfa_manager(&secret).await?;

    let now = settle_time_step();
    let token = totp_for(&secret, now)?;
    let context = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", Some(token)),
        )
        .await?;
    assert!(context.mfa_verified, "current-step TOTP must verify");
    Ok(())
}

#[tokio::test]
async fn test_totp_previous_and_next_step_pass_within_skew() -> Result<(), Box<dyn Error>> {
    let secret = AccessControlManager::generate_totp_secret();

    let now = settle_time_step();
    for offset in [-(TOTP_STEP as i64), TOTP_STEP as i64] {
        let mut manager = mfa_manager(&secret).await?;
        let timestamp = (now as i64 + offset) as u64;
        let token = totp_for(&secret, timestamp)?;
        let context = manager
            .authenticate(
                "alice".to_string(),
                password_credentials("correct horse battery staple", Some(token)),
            )
            .await?;
        assert!(
            context.mfa_verified,
            "token at offset {offset}s must verify within the ±1-step window"
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_totp_stale_two_steps_back_fails() -> Result<(), Box<dyn Error>> {
    let secret = AccessControlManager::generate_totp_secret();
    let mut manager = mfa_manager(&secret).await?;

    let now = settle_time_step();
    let stale_token = totp_for(&secret, now - 2 * TOTP_STEP)?;
    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", Some(stale_token)),
        )
        .await;
    assert!(result.is_err(), "token two steps stale must be rejected");
    Ok(())
}

#[tokio::test]
async fn test_random_six_digit_token_fails() -> Result<(), Box<dyn Error>> {
    // The Phase 0 fake accepted any 6 ASCII digits. Pick a 6-digit token that
    // provably differs from every token in the accepted ±1-step window.
    let secret = AccessControlManager::generate_totp_secret();
    let mut manager = mfa_manager(&secret).await?;

    let now = settle_time_step();
    let window: Vec<String> = [
        now - TOTP_STEP,
        now,
        now + TOTP_STEP,
        // One extra step on each side in case the step rolls over mid-test.
        now - 2 * TOTP_STEP,
        now + 2 * TOTP_STEP,
    ]
    .iter()
    .map(|t| totp_for(&secret, *t))
    .collect::<Result<_, _>>()?;

    let mut candidate: u32 = 123456;
    while window.contains(&format!("{candidate:06}")) {
        candidate = (candidate + 1) % 1_000_000;
    }
    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials(
                "correct horse battery staple",
                Some(format!("{candidate:06}")),
            ),
        )
        .await;
    assert!(
        result.is_err(),
        "an arbitrary 6-digit token must be rejected"
    );
    Ok(())
}

#[tokio::test]
async fn test_enrolled_totp_user_denied_when_token_omitted() -> Result<(), Box<dyn Error>> {
    // MFA-omission bypass guard: with require_mfa=true and an enrolled TOTP
    // secret, presenting no token must be denied (not a silently unverified
    // session), consistent with how an invalid token hard-fails.
    let secret = AccessControlManager::generate_totp_secret();
    let mut manager = mfa_manager(&secret).await?;

    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", None),
        )
        .await;
    assert!(
        result.is_err(),
        "enrolled-TOTP user with require_mfa and no token must be denied"
    );
    Ok(())
}

#[tokio::test]
async fn test_totp_without_enrolled_secret_fails() -> Result<(), Box<dyn Error>> {
    let mut manager = AccessControlManager::new(&manager_config(true)).await?;
    manager.set_password("alice", "correct horse battery staple")?;

    let result = manager
        .authenticate(
            "alice".to_string(),
            password_credentials("correct horse battery staple", Some("123456".to_string())),
        )
        .await;
    assert!(
        result.is_err(),
        "an MFA token for a user with no enrolled TOTP secret must be rejected"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Policy engine: deny-by-default condition evaluation
// ---------------------------------------------------------------------------

struct NullAuditStorage;

impl AuditStorage for NullAuditStorage {
    fn log_event(&self, _event: AuditEvent) -> synaptic::error::Result<()> {
        Ok(())
    }
    fn query_events(&self, _query: AuditQuery) -> synaptic::error::Result<Vec<AuditEvent>> {
        Ok(Vec::new())
    }
}

fn policy_with_condition(condition: PolicyCondition, action: PolicyAction) -> SecurityPolicy {
    SecurityPolicy {
        id: "policy-1".to_string(),
        name: "test-policy".to_string(),
        description: "policy under test".to_string(),
        rules: vec![PolicyRule {
            id: "rule-1".to_string(),
            name: "rule-under-test".to_string(),
            condition,
            action,
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

fn engine_with_user() -> Result<PolicyEngine, Box<dyn Error>> {
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
    engine.add_user(User {
        id: "user-1".to_string(),
        username: "alice".to_string(),
        email: "alice@example.com".to_string(),
        roles: vec!["role-user".to_string()],
        security_clearance: DataClassificationLevel::Internal,
        mfa_enabled: false,
        last_login: None,
        failed_login_attempts: 0,
        account_locked: false,
        created_at: chrono::Utc::now(),
        active: true,
    })?;
    Ok(engine)
}

fn access_request() -> AccessRequest {
    AccessRequest {
        user_id: "user-1".to_string(),
        resource_type: "memory".to_string(),
        resource_id: "res-1".to_string(),
        action: "read".to_string(),
        ip_address: "127.0.0.1".to_string(),
        user_agent: "auth_tests".to_string(),
        timestamp: chrono::Utc::now(),
        session_id: "session-1".to_string(),
        additional_context: Default::default(),
    }
}

#[test]
fn test_unsupported_policy_condition_denies() -> Result<(), Box<dyn Error>> {
    // A rule keyed on an unimplemented condition (TimeWindow) must fail
    // closed instead of silently matching (the old `_ => Ok(true)`).
    let mut engine = engine_with_user()?;
    let condition = PolicyCondition::TimeWindow(TimeWindow {
        start_time: chrono::NaiveTime::from_hms_opt(9, 0, 0).expect("valid time"),
        end_time: chrono::NaiveTime::from_hms_opt(17, 0, 0).expect("valid time"),
        days_of_week: vec![chrono::Weekday::Mon],
        timezone: "UTC".to_string(),
    });
    engine.add_policy(policy_with_condition(condition, PolicyAction::Allow))?;

    let result = engine.evaluate_access(&access_request());
    assert!(
        result.is_err(),
        "unsupported condition must produce an explicit error, not allow"
    );
    Ok(())
}

#[test]
fn test_supported_condition_still_evaluates() -> Result<(), Box<dyn Error>> {
    // Control: a supported condition (UserRole) evaluates normally, and a
    // matching Deny rule denies.
    let mut engine = engine_with_user()?;
    engine.add_policy(policy_with_condition(
        PolicyCondition::UserRole("user".to_string()),
        PolicyAction::Deny,
    ))?;

    let decision = engine.evaluate_access(&access_request())?;
    assert!(!decision.allowed, "matching Deny rule must deny access");
    Ok(())
}
