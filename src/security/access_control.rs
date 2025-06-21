//! Advanced Access Control Module
//! 
//! Implements role-based access control (RBAC), attribute-based access control (ABAC),
//! multi-factor authentication, and advanced authorization mechanisms.

use crate::error::{MemoryError, Result};
use crate::security::{SecurityConfig, SecurityContext, Permission, AbacRule, AccessControlPolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};


/// Access control manager
#[derive(Debug)]
pub struct AccessControlManager {
    #[allow(dead_code)]
    config: SecurityConfig,
    policy: AccessControlPolicy,
    active_sessions: HashMap<String, SessionInfo>,
    failed_attempts: HashMap<String, FailedAttemptTracker>,
    metrics: AccessMetrics,
}

impl AccessControlManager {
    /// Create a new access control manager
    pub async fn new(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            policy: config.access_control_policy.clone(),
            active_sessions: HashMap::new(),
            failed_attempts: HashMap::new(),
            metrics: AccessMetrics::default(),
        })
    }

    /// Authenticate a user and create a security context
    pub async fn authenticate(&mut self, 
        user_id: String, 
        credentials: AuthenticationCredentials
    ) -> Result<SecurityContext> {
        let start_time = std::time::Instant::now();

        // Check for too many failed attempts
        if self.is_user_locked(&user_id) {
            self.metrics.total_blocked_attempts += 1;
            return Err(MemoryError::access_denied("User account is temporarily locked".to_string()));
        }

        // Verify credentials
        let auth_result = self.verify_credentials(&user_id, &credentials).await?;
        
        if !auth_result.success {
            self.record_failed_attempt(&user_id);
            self.metrics.total_failed_authentications += 1;
            return Err(MemoryError::access_denied("Invalid credentials".to_string()));
        }

        // Create security context
        let mut context = SecurityContext::new(user_id.clone(), auth_result.roles);
        context.attributes = auth_result.attributes;

        // Handle MFA if required
        if self.policy.require_mfa && !credentials.mfa_token.is_some() {
            context.mfa_verified = false;
            // In production, would initiate MFA challenge
        } else if let Some(mfa_token) = credentials.mfa_token {
            context.mfa_verified = self.verify_mfa_token(&user_id, &mfa_token).await?;
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

        self.active_sessions.insert(context.session_id.clone(), session_info);
        
        // Clear failed attempts on successful authentication
        self.failed_attempts.remove(&user_id);

        // Update metrics
        self.metrics.total_successful_authentications += 1;
        self.metrics.total_auth_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(context)
    }

    /// Check if a user has a specific permission
    pub async fn check_permission(&mut self, 
        context: &SecurityContext, 
        permission: Permission
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
        let session = self.active_sessions.get(&context.session_id)
            .ok_or_else(|| MemoryError::access_denied("Invalid session".to_string()))?;

        // Check if session is expired
        if Utc::now() > session.expires_at {
            self.active_sessions.remove(&context.session_id);
            return Err(MemoryError::access_denied("Session expired".to_string()));
        }

        // Check session timeout
        let timeout_duration = chrono::Duration::minutes(self.policy.session_timeout_minutes as i64);
        if Utc::now() > session.last_activity + timeout_duration {
            self.active_sessions.remove(&context.session_id);
            return Err(MemoryError::access_denied("Session timed out".to_string()));
        }

        // Check MFA requirement
        if self.policy.require_mfa && !session.mfa_verified {
            return Err(MemoryError::access_denied("MFA verification required".to_string()));
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
    pub async fn add_role(&mut self, role_name: String, permissions: Vec<Permission>) -> Result<()> {
        self.policy.rbac_rules.insert(role_name, permissions);
        Ok(())
    }

    /// Add a new ABAC rule
    pub async fn add_abac_rule(&mut self, rule: AbacRule) -> Result<()> {
        self.policy.abac_rules.push(rule);
        // Sort by priority (higher priority first)
        self.policy.abac_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
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

    async fn verify_credentials(&self, user_id: &str, credentials: &AuthenticationCredentials) -> Result<AuthenticationResult> {
        // In production, this would verify against a secure user database
        // For demo purposes, we'll use simple logic
        
        let is_valid = match credentials.auth_type {
            AuthenticationType::Password => {
                // Simulate password verification
                credentials.password.as_ref().map_or(false, |p| p.len() >= 8)
            },
            AuthenticationType::ApiKey => {
                // Simulate API key verification
                credentials.api_key.as_ref().map_or(false, |k| k.starts_with("sk-"))
            },
            AuthenticationType::Certificate => {
                // Simulate certificate verification
                credentials.certificate.is_some()
            },
        };

        if is_valid {
            // Assign roles based on user (simplified)
            let roles = if user_id == "admin" || user_id == "workflow_user" || user_id == "metrics_user" {
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

    async fn verify_mfa_token(&self, _user_id: &str, token: &str) -> Result<bool> {
        // In production, verify TOTP, SMS, or hardware token
        // For demo, accept tokens that are 6 digits
        Ok(token.len() == 6 && token.chars().all(|c| c.is_ascii_digit()))
    }

    async fn check_rbac_permission(&self, context: &SecurityContext, permission: &Permission) -> Result<bool> {
        for role in &context.roles {
            if let Some(role_permissions) = self.policy.rbac_rules.get(role) {
                if role_permissions.contains(permission) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn check_abac_permission(&self, context: &SecurityContext, permission: &Permission) -> Result<bool> {
        for rule in &self.policy.abac_rules {
            if !rule.enabled {
                continue;
            }

            if rule.permissions.contains(permission) {
                // Evaluate rule condition (simplified JSON-based evaluation)
                if self.evaluate_abac_condition(&rule.condition, context).await? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn evaluate_abac_condition(&self, condition: &str, context: &SecurityContext) -> Result<bool> {
        // Simplified ABAC condition evaluation
        // In production, use a proper policy evaluation engine
        
        if condition.contains("department") && condition.contains("engineering") {
            return Ok(context.attributes.get("department") == Some(&"engineering".to_string()));
        }
        
        if condition.contains("clearance_level") && condition.contains("confidential") {
            let default_clearance = "public".to_string();
            let user_clearance = context.attributes.get("clearance_level").unwrap_or(&default_clearance);
            return Ok(user_clearance == "confidential" || user_clearance == "secret");
        }

        // Default to false for unknown conditions
        Ok(false)
    }

    fn is_user_locked(&self, user_id: &str) -> bool {
        if let Some(tracker) = self.failed_attempts.get(user_id) {
            tracker.count >= self.policy.max_failed_attempts && 
            Utc::now() < tracker.locked_until
        } else {
            false
        }
    }

    fn record_failed_attempt(&mut self, user_id: &str) {
        let tracker = self.failed_attempts.entry(user_id.to_string())
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
        let total_auth_attempts = self.total_successful_authentications + self.total_failed_authentications;
        if total_auth_attempts > 0 {
            self.authentication_success_rate = (self.total_successful_authentications as f64 / total_auth_attempts as f64) * 100.0;
            self.average_auth_time_ms = self.total_auth_time_ms as f64 / total_auth_attempts as f64;
        }

        let total_permission_checks = self.total_granted_permissions + self.total_denied_permissions;
        if total_permission_checks > 0 {
            self.permission_grant_rate = (self.total_granted_permissions as f64 / total_permission_checks as f64) * 100.0;
            self.average_access_time_ms = self.total_access_time_ms as f64 / total_permission_checks as f64;
        }
    }
}
