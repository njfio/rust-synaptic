//! Advanced Access Control Module
//!
//! Implements role-based access control (RBAC), attribute-based access control (ABAC),
//! multi-factor authentication, and advanced authorization mechanisms.

use crate::error::{MemoryError, Result};
use crate::security::{AbacRule, AccessControlPolicy, Permission, SecurityConfig, SecurityContext};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use subtle::ConstantTimeEq;
use totp_rs::{Algorithm as TotpAlgorithm, TOTP};

/// TOTP parameters (RFC 6238): 6 digits, 30-second step, ±1 step of skew.
const TOTP_DIGITS: usize = 6;
const TOTP_SKEW: u8 = 1;
const TOTP_STEP_SECONDS: u64 = 30;
/// Minimum TOTP shared-secret length (RFC 4226 §4 requires >= 128 bits).
const TOTP_MIN_SECRET_BYTES: usize = 16;

/// Stored credential material for a provisioned user.
///
/// Passwords are stored as argon2id PHC strings, API keys as SHA-256 digests
/// (compared in constant time), and MFA as a raw TOTP shared secret.
#[derive(Clone, Default)]
pub struct StoredCredentials {
    password_hash: Option<String>,
    api_key_sha256: Option<[u8; 32]>,
    totp_secret: Option<Vec<u8>>,
}

impl std::fmt::Debug for StoredCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredCredentials")
            .field(
                "password_hash",
                &self.password_hash.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "api_key_sha256",
                &self.api_key_sha256.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "totp_secret",
                &self.totp_secret.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Access control manager
#[derive(Debug)]
pub struct AccessControlManager {
    policy: AccessControlPolicy,
    active_sessions: HashMap<String, SessionInfo>,
    failed_attempts: HashMap<String, FailedAttemptTracker>,
    credentials: HashMap<String, StoredCredentials>,
    metrics: AccessMetrics,
}

impl AccessControlManager {
    /// Create a new access control manager
    pub async fn new(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            policy: config.access_control_policy.clone(),
            active_sessions: HashMap::new(),
            failed_attempts: HashMap::new(),
            credentials: HashMap::new(),
            metrics: AccessMetrics::default(),
        })
    }

    /// Hash a password into an argon2id PHC string for provisioning/storage.
    pub fn hash_password(password: &str) -> Result<String> {
        let salt = SaltString::generate(&mut rand::rngs::OsRng);
        Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map(|hash| hash.to_string())
            .map_err(|e| MemoryError::access_denied(format!("Password hashing failed: {e}")))
    }

    /// Provision (or rotate) a user's password. The password is stored only
    /// as an argon2id PHC string.
    pub fn set_password(&mut self, user_id: &str, password: &str) -> Result<()> {
        let hash = Self::hash_password(password)?;
        self.set_password_hash(user_id, hash)
    }

    /// Provision a user from an already-computed argon2 PHC string (e.g. one
    /// loaded from a credential store).
    pub fn set_password_hash(&mut self, user_id: &str, phc_hash: String) -> Result<()> {
        if !phc_hash.starts_with('$') {
            return Err(MemoryError::access_denied(
                "Stored password hash must be a PHC-format string".to_string(),
            ));
        }
        self.credentials
            .entry(user_id.to_string())
            .or_default()
            .password_hash = Some(phc_hash);
        Ok(())
    }

    /// Provision (or rotate) a user's API key. Only the SHA-256 digest of the
    /// key is retained; verification is a constant-time digest comparison.
    pub fn set_api_key(&mut self, user_id: &str, api_key: &str) {
        let digest: [u8; 32] = Sha256::digest(api_key.as_bytes()).into();
        self.credentials
            .entry(user_id.to_string())
            .or_default()
            .api_key_sha256 = Some(digest);
    }

    /// Generate a fresh 160-bit TOTP shared secret (RFC 6238 recommends the
    /// HMAC-SHA-1 output size) from the OS CSPRNG.
    pub fn generate_totp_secret() -> Vec<u8> {
        let mut secret = vec![0u8; 20];
        rand::rngs::OsRng.fill_bytes(&mut secret);
        secret
    }

    /// Enroll a user's TOTP shared secret for MFA. The secret must be at
    /// least 128 bits (RFC 4226 §4).
    pub fn set_totp_secret(&mut self, user_id: &str, secret: Vec<u8>) -> Result<()> {
        if secret.len() < TOTP_MIN_SECRET_BYTES {
            return Err(MemoryError::access_denied(format!(
                "TOTP secret must be at least {TOTP_MIN_SECRET_BYTES} bytes"
            )));
        }
        self.credentials
            .entry(user_id.to_string())
            .or_default()
            .totp_secret = Some(secret);
        Ok(())
    }

    /// Authenticate a user and create a security context
    pub async fn authenticate(
        &mut self,
        user_id: String,
        credentials: AuthenticationCredentials,
    ) -> Result<SecurityContext> {
        let start_time = std::time::Instant::now();

        // Check for too many failed attempts
        if self.is_user_locked(&user_id) {
            self.metrics.total_blocked_attempts += 1;
            return Err(MemoryError::access_denied(
                "User account is temporarily locked".to_string(),
            ));
        }

        // Verify credentials
        let auth_result = self.verify_credentials(&user_id, &credentials).await?;

        if !auth_result.success {
            self.record_failed_attempt(&user_id);
            self.metrics.total_failed_authentications += 1;
            return Err(MemoryError::access_denied(
                "Invalid credentials".to_string(),
            ));
        }

        // Create security context
        let mut context = SecurityContext::new(user_id.clone(), auth_result.roles);
        context.attributes = auth_result.attributes;

        // Handle MFA if required
        if self.policy.require_mfa && credentials.mfa_token.is_none() {
            // If the user has an enrolled TOTP secret, omitting the token must
            // NOT silently produce an unverified session — that would be an
            // MFA bypass (the caller could then skip any MFA-gated path that
            // only checks a boolean). Deny, consistent with an invalid token.
            if self.user_has_totp_enrolled(&user_id) {
                self.record_failed_attempt(&user_id);
                self.metrics.total_failed_authentications += 1;
                return Err(MemoryError::access_denied(
                    "MFA required: TOTP token must be provided".to_string(),
                ));
            }
            // No enrolled secret: we cannot challenge what isn't enrolled, so
            // leave mfa_verified=false. Downstream checks (is_mfa_satisfied /
            // validate_session) still treat this as unsatisfied.
            context.mfa_verified = false;
        } else if let Some(ref mfa_token) = credentials.mfa_token {
            if !self.verify_mfa_token(&user_id, mfa_token).await? {
                self.record_failed_attempt(&user_id);
                self.metrics.total_failed_authentications += 1;
                return Err(MemoryError::access_denied("Invalid MFA token".to_string()));
            }
            context.mfa_verified = true;
        }

        // Create session
        let session_info = SessionInfo {
            user_id: user_id.clone(),
            session_id: context.session_id.clone(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            expires_at: context.session_expiry,
            ip_address: credentials.ip_address.clone(),
            user_agent: credentials.user_agent.clone(),
            mfa_verified: context.mfa_verified,
        };

        self.active_sessions
            .insert(context.session_id.clone(), session_info);

        // Clear failed attempts on successful authentication
        self.failed_attempts.remove(&user_id);

        // Update metrics
        self.metrics.total_successful_authentications += 1;
        self.metrics.total_auth_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(context)
    }

    /// Check if a user has a specific permission
    pub async fn check_permission(
        &mut self,
        context: &SecurityContext,
        permission: Permission,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Validate session
        self.validate_session(context).await?;

        // Check RBAC permissions
        let has_rbac_permission = self.check_rbac_permission(context, &permission).await?;

        // Check ABAC rules
        let has_abac_permission = self.check_abac_permission(context, &permission).await?;

        // Permission granted if either RBAC or ABAC allows it
        let permission_granted = has_rbac_permission || has_abac_permission;

        // Update session activity
        if let Some(session) = self.active_sessions.get_mut(&context.session_id) {
            session.last_activity = Utc::now();
        }

        // Update metrics
        self.metrics.total_permission_checks += 1;
        self.metrics.total_access_time_ms += start_time.elapsed().as_millis() as u64;

        if permission_granted {
            self.metrics.total_granted_permissions += 1;
            Ok(())
        } else {
            self.metrics.total_denied_permissions += 1;
            Err(MemoryError::access_denied(format!(
                "Permission denied: {:?} for user {}",
                permission, context.user_id
            )))
        }
    }

    /// Validate and refresh a session
    pub async fn validate_session(&mut self, context: &SecurityContext) -> Result<()> {
        let session = self
            .active_sessions
            .get(&context.session_id)
            .ok_or_else(|| MemoryError::access_denied("Invalid session".to_string()))?;

        // Check if session is expired
        if Utc::now() > session.expires_at {
            self.active_sessions.remove(&context.session_id);
            return Err(MemoryError::access_denied("Session expired".to_string()));
        }

        // Check session timeout
        let timeout_duration =
            chrono::Duration::minutes(self.policy.session_timeout_minutes as i64);
        if Utc::now() > session.last_activity + timeout_duration {
            self.active_sessions.remove(&context.session_id);
            return Err(MemoryError::access_denied("Session timed out".to_string()));
        }

        // Check MFA requirement
        if self.policy.require_mfa && !session.mfa_verified {
            return Err(MemoryError::access_denied(
                "MFA verification required".to_string(),
            ));
        }

        Ok(())
    }

    /// Revoke a session
    pub async fn revoke_session(&mut self, session_id: &str) -> Result<()> {
        self.active_sessions.remove(session_id);
        self.metrics.total_revoked_sessions += 1;
        Ok(())
    }

    /// Add a new role with permissions
    pub async fn add_role(
        &mut self,
        role_name: String,
        permissions: Vec<Permission>,
    ) -> Result<()> {
        self.policy.rbac_rules.insert(role_name, permissions);
        Ok(())
    }

    /// Add a new ABAC rule
    pub async fn add_abac_rule(&mut self, rule: AbacRule) -> Result<()> {
        self.policy.abac_rules.push(rule);
        // Sort by priority (higher priority first)
        self.policy
            .abac_rules
            .sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    /// Get access control metrics
    pub async fn get_metrics(&self) -> Result<AccessMetrics> {
        Ok(self.metrics.clone())
    }

    /// Get active sessions count
    pub async fn get_active_sessions_count(&self) -> Result<usize> {
        Ok(self.active_sessions.len())
    }

    // Private helper methods

    async fn verify_credentials(
        &self,
        user_id: &str,
        credentials: &AuthenticationCredentials,
    ) -> Result<AuthenticationResult> {
        // Verify against provisioned credential material. Users without a
        // stored credential of the requested type are always denied.
        let stored = self.credentials.get(user_id);

        let is_valid = match credentials.auth_type {
            AuthenticationType::Password => {
                // argon2 PHC-string verification against the stored hash.
                match (
                    stored.and_then(|s| s.password_hash.as_deref()),
                    credentials.password.as_deref(),
                ) {
                    (Some(stored_hash), Some(password)) => PasswordHash::new(stored_hash)
                        .map(|parsed| {
                            Argon2::default()
                                .verify_password(password.as_bytes(), &parsed)
                                .is_ok()
                        })
                        .unwrap_or(false),
                    _ => false,
                }
            }
            AuthenticationType::ApiKey => {
                // Constant-time comparison of SHA-256 digests.
                match (
                    stored.and_then(|s| s.api_key_sha256.as_ref()),
                    credentials.api_key.as_deref(),
                ) {
                    (Some(stored_digest), Some(api_key)) => {
                        let presented: [u8; 32] = Sha256::digest(api_key.as_bytes()).into();
                        bool::from(presented.ct_eq(stored_digest))
                    }
                    _ => false,
                }
            }
            AuthenticationType::Certificate => {
                // Fail closed: certificate verification is not implemented
                // (previously any non-empty certificate was accepted).
                false
            }
        };

        if is_valid {
            // Assign roles based on user (simplified)
            let roles =
                if user_id == "admin" || user_id == "workflow_user" || user_id == "metrics_user" {
                    vec!["admin".to_string(), "user".to_string()]
                } else {
                    vec!["user".to_string()]
                };

            // Set user attributes
            let mut attributes = HashMap::new();
            attributes.insert("department".to_string(), "engineering".to_string());
            attributes.insert("clearance_level".to_string(), "confidential".to_string());

            Ok(AuthenticationResult {
                success: true,
                roles,
                attributes,
            })
        } else {
            Ok(AuthenticationResult {
                success: false,
                roles: Vec::new(),
                attributes: HashMap::new(),
            })
        }
    }

    /// Whether the user has an enrolled TOTP secret for MFA.
    fn user_has_totp_enrolled(&self, user_id: &str) -> bool {
        self.credentials
            .get(user_id)
            .is_some_and(|c| c.totp_secret.is_some())
    }

    /// Verify an RFC 6238 TOTP token against the user's enrolled shared
    /// secret, accepting a ±1 time-step window. Users without an enrolled
    /// secret are denied.
    async fn verify_mfa_token(&self, user_id: &str, token: &str) -> Result<bool> {
        let secret = match self
            .credentials
            .get(user_id)
            .and_then(|s| s.totp_secret.clone())
        {
            Some(secret) => secret,
            None => return Ok(false),
        };

        let totp = TOTP::new(
            TotpAlgorithm::SHA1,
            TOTP_DIGITS,
            TOTP_SKEW,
            TOTP_STEP_SECONDS,
            secret,
        )
        .map_err(|e| MemoryError::access_denied(format!("Invalid TOTP configuration: {e}")))?;

        totp.check_current(token)
            .map_err(|e| MemoryError::access_denied(format!("System time error: {e}")))
    }

    async fn check_rbac_permission(
        &self,
        context: &SecurityContext,
        permission: &Permission,
    ) -> Result<bool> {
        for role in &context.roles {
            if let Some(role_permissions) = self.policy.rbac_rules.get(role) {
                if role_permissions.contains(permission) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn check_abac_permission(
        &self,
        context: &SecurityContext,
        permission: &Permission,
    ) -> Result<bool> {
        for rule in &self.policy.abac_rules {
            if !rule.enabled {
                continue;
            }

            if rule.permissions.contains(permission) {
                // Evaluate rule condition (simplified JSON-based evaluation)
                if self
                    .evaluate_abac_condition(&rule.condition, context)
                    .await?
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn evaluate_abac_condition(
        &self,
        condition: &str,
        context: &SecurityContext,
    ) -> Result<bool> {
        // Simplified ABAC condition evaluation
        // In production, use a proper policy evaluation engine

        if condition.contains("department") && condition.contains("engineering") {
            return Ok(context.attributes.get("department") == Some(&"engineering".to_string()));
        }

        if condition.contains("clearance_level") && condition.contains("confidential") {
            let default_clearance = "public".to_string();
            let user_clearance = context
                .attributes
                .get("clearance_level")
                .unwrap_or(&default_clearance);
            return Ok(user_clearance == "confidential" || user_clearance == "secret");
        }

        // Default to false for unknown conditions
        Ok(false)
    }

    fn is_user_locked(&self, user_id: &str) -> bool {
        if let Some(tracker) = self.failed_attempts.get(user_id) {
            tracker.count >= self.policy.max_failed_attempts && Utc::now() < tracker.locked_until
        } else {
            false
        }
    }

    fn record_failed_attempt(&mut self, user_id: &str) {
        let tracker = self
            .failed_attempts
            .entry(user_id.to_string())
            .or_insert_with(|| FailedAttemptTracker {
                count: 0,
                first_attempt: Utc::now(),
                locked_until: Utc::now(),
            });

        tracker.count += 1;

        if tracker.count >= self.policy.max_failed_attempts {
            // Lock for 15 minutes
            tracker.locked_until = Utc::now() + chrono::Duration::minutes(15);
        }
    }
}

/// Authentication credentials
#[derive(Debug, Clone)]
pub struct AuthenticationCredentials {
    pub auth_type: AuthenticationType,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub certificate: Option<Vec<u8>>,
    pub mfa_token: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// Authentication types
#[derive(Debug, Clone)]
pub enum AuthenticationType {
    Password,
    ApiKey,
    Certificate,
}

/// Authentication result
#[derive(Debug)]
struct AuthenticationResult {
    success: bool,
    roles: Vec<String>,
    attributes: HashMap<String, String>,
}

/// Session information
#[derive(Debug, Clone)]
struct SessionInfo {
    // Session audit metadata: populated for forensic/debug inspection even
    // though the manager only reads activity/expiry fields today.
    #[allow(dead_code)]
    user_id: String,
    #[allow(dead_code)]
    session_id: String,
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    #[allow(dead_code)]
    ip_address: Option<String>,
    #[allow(dead_code)]
    user_agent: Option<String>,
    mfa_verified: bool,
}

/// Failed attempt tracker
#[derive(Debug)]
struct FailedAttemptTracker {
    count: u32,
    // Retained for lockout auditing; only `count`/`locked_until` drive logic.
    #[allow(dead_code)]
    first_attempt: DateTime<Utc>,
    locked_until: DateTime<Utc>,
}

/// Access control metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessMetrics {
    pub total_successful_authentications: u64,
    pub total_failed_authentications: u64,
    pub total_blocked_attempts: u64,
    pub total_permission_checks: u64,
    pub total_granted_permissions: u64,
    pub total_denied_permissions: u64,
    pub total_revoked_sessions: u64,
    pub total_auth_time_ms: u64,
    pub total_access_time_ms: u64,
    pub average_auth_time_ms: f64,
    pub average_access_time_ms: f64,
    pub authentication_success_rate: f64,
    pub permission_grant_rate: f64,
}

impl AccessMetrics {
    pub fn calculate_rates(&mut self) {
        let total_auth_attempts =
            self.total_successful_authentications + self.total_failed_authentications;
        if total_auth_attempts > 0 {
            self.authentication_success_rate =
                (self.total_successful_authentications as f64 / total_auth_attempts as f64) * 100.0;
            self.average_auth_time_ms = self.total_auth_time_ms as f64 / total_auth_attempts as f64;
        }

        let total_permission_checks =
            self.total_granted_permissions + self.total_denied_permissions;
        if total_permission_checks > 0 {
            self.permission_grant_rate =
                (self.total_granted_permissions as f64 / total_permission_checks as f64) * 100.0;
            self.average_access_time_ms =
                self.total_access_time_ms as f64 / total_permission_checks as f64;
        }
    }
}
