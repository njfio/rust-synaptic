//! Security and Privacy Module
//! 
//! This module implements state-of-the-art security and privacy features for the Synaptic
//! AI agent memory system, including zero-knowledge architecture, homomorphic encryption,
//! differential privacy, and advanced access control.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

pub mod encryption;
pub mod privacy;
pub mod access_control;
pub mod audit;
pub mod key_management;

/// Security configuration for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable zero-knowledge architecture
    pub enable_zero_knowledge: bool,
    /// Enable homomorphic encryption
    pub enable_homomorphic_encryption: bool,
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Privacy budget for differential privacy
    pub privacy_budget: f64,
    /// Encryption key size in bits
    pub encryption_key_size: usize,
    /// Access control policy
    pub access_control_policy: AccessControlPolicy,
    /// Audit logging configuration
    pub audit_config: AuditConfig,
    /// Key rotation interval in hours
    pub key_rotation_interval_hours: u64,
    /// Enable secure multi-party computation
    pub enable_secure_mpc: bool,
    /// Enable homomorphic operations
    pub enable_homomorphic_ops: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_zero_knowledge: true,
            enable_homomorphic_encryption: true,
            enable_differential_privacy: true,
            privacy_budget: 1.0,
            encryption_key_size: 256,
            access_control_policy: AccessControlPolicy::default(),
            audit_config: AuditConfig::default(),
            key_rotation_interval_hours: 24,
            enable_secure_mpc: true,
            enable_homomorphic_ops: true,
        }
    }
}

/// Access control policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlPolicy {
    /// Default permission level
    pub default_permission: PermissionLevel,
    /// Role-based access control rules
    pub rbac_rules: HashMap<String, Vec<Permission>>,
    /// Attribute-based access control rules
    pub abac_rules: Vec<AbacRule>,
    /// Enable multi-factor authentication
    pub require_mfa: bool,
    /// Session timeout in minutes
    pub session_timeout_minutes: u64,
    /// Maximum failed login attempts
    pub max_failed_attempts: u32,
}

impl Default for AccessControlPolicy {
    fn default() -> Self {
        Self {
            default_permission: PermissionLevel::Read,
            rbac_rules: HashMap::new(),
            abac_rules: Vec::new(),
            require_mfa: true,
            session_timeout_minutes: 60,
            max_failed_attempts: 3,
        }
    }
}

/// Permission levels for access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionLevel {
    None,
    Read,
    Write,
    Admin,
    SuperAdmin,
}

/// Individual permissions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    ReadMemory,
    WriteMemory,
    DeleteMemory,
    ReadAnalytics,
    WriteAnalytics,
    ManageUsers,
    ManageKeys,
    ViewAuditLogs,
    ConfigureSystem,
    ExecuteQueries,
    AccessDistributed,
    ManageIntegrations,
}

/// Attribute-based access control rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacRule {
    pub id: String,
    pub name: String,
    pub condition: String, // JSON-based condition expression
    pub permissions: Vec<Permission>,
    pub priority: u32,
    pub enabled: bool,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log all memory operations
    pub log_memory_operations: bool,
    /// Log authentication events
    pub log_auth_events: bool,
    /// Log access control decisions
    pub log_access_decisions: bool,
    /// Log encryption operations
    pub log_encryption_ops: bool,
    /// Audit log retention period in days
    pub retention_days: u32,
    /// Enable real-time alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_memory_operations: true,
            log_auth_events: true,
            log_access_decisions: true,
            log_encryption_ops: false, // Can be verbose
            retention_days: 90,
            enable_alerting: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for security monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Failed login attempts per hour
    pub failed_logins_per_hour: u32,
    /// Suspicious access patterns per hour
    pub suspicious_access_per_hour: u32,
    /// Encryption failures per hour
    pub encryption_failures_per_hour: u32,
    /// Unauthorized access attempts per hour
    pub unauthorized_attempts_per_hour: u32,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            failed_logins_per_hour: 10,
            suspicious_access_per_hour: 5,
            encryption_failures_per_hour: 3,
            unauthorized_attempts_per_hour: 5,
        }
    }
}

/// Security context for operations
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// User ID
    pub user_id: String,
    /// Session ID
    pub session_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// User attributes for ABAC
    pub attributes: HashMap<String, String>,
    /// Authentication timestamp
    pub auth_timestamp: DateTime<Utc>,
    /// Session expiry
    pub session_expiry: DateTime<Utc>,
    /// MFA verified
    pub mfa_verified: bool,
    /// Request ID for audit trail
    pub request_id: String,
}

impl SecurityContext {
    /// Create a new security context
    pub fn new(user_id: String, roles: Vec<String>) -> Self {
        let now = Utc::now();
        Self {
            user_id,
            session_id: Uuid::new_v4().to_string(),
            roles,
            attributes: HashMap::new(),
            auth_timestamp: now,
            session_expiry: now + chrono::Duration::hours(1),
            mfa_verified: false,
            request_id: Uuid::new_v4().to_string(),
        }
    }

    /// Check if the session is valid
    pub fn is_session_valid(&self) -> bool {
        Utc::now() < self.session_expiry
    }

    /// Check if MFA is required and verified
    pub fn is_mfa_satisfied(&self, require_mfa: bool) -> bool {
        !require_mfa || self.mfa_verified
    }
}

/// Main security manager
#[derive(Debug)]
pub struct SecurityManager {
    config: SecurityConfig,
    pub encryption_manager: encryption::EncryptionManager,
    pub privacy_manager: privacy::PrivacyManager,
    pub access_control: access_control::AccessControlManager,
    pub audit_logger: audit::AuditLogger,
    pub key_manager: key_management::KeyManager,
}

impl SecurityManager {
    /// Create a new security manager
    pub async fn new(config: SecurityConfig) -> Result<Self> {
        let encryption_manager = encryption::EncryptionManager::new(&config).await?;
        let privacy_manager = privacy::PrivacyManager::new(&config).await?;
        let access_control = access_control::AccessControlManager::new(&config).await?;
        let audit_logger = audit::AuditLogger::new(&config.audit_config).await?;
        let key_manager = key_management::KeyManager::new(&config).await?;

        Ok(Self {
            config,
            encryption_manager,
            privacy_manager,
            access_control,
            audit_logger,
            key_manager,
        })
    }

    /// Encrypt a memory entry with zero-knowledge architecture
    pub async fn encrypt_memory(&mut self, 
        entry: &MemoryEntry, 
        context: &SecurityContext
    ) -> Result<EncryptedMemoryEntry> {
        // Check permissions
        self.access_control.check_permission(context, Permission::WriteMemory).await?;

        // Apply differential privacy if enabled
        let processed_entry = if self.config.enable_differential_privacy {
            self.privacy_manager.apply_differential_privacy(entry, context).await?
        } else {
            entry.clone()
        };

        // Encrypt with homomorphic encryption if enabled
        let encrypted_entry = if self.config.enable_homomorphic_encryption {
            self.encryption_manager.homomorphic_encrypt(&processed_entry, context).await?
        } else {
            self.encryption_manager.standard_encrypt(&processed_entry, context).await?
        };

        // Log the operation
        self.audit_logger.log_encryption_operation(context, "encrypt_memory", true).await?;

        Ok(encrypted_entry)
    }

    /// Decrypt a memory entry
    pub async fn decrypt_memory(&mut self, 
        encrypted_entry: &EncryptedMemoryEntry, 
        context: &SecurityContext
    ) -> Result<MemoryEntry> {
        // Check permissions
        self.access_control.check_permission(context, Permission::ReadMemory).await?;

        // Decrypt based on encryption type
        let decrypted_entry = if encrypted_entry.is_homomorphic {
            self.encryption_manager.homomorphic_decrypt(encrypted_entry, context).await?
        } else {
            self.encryption_manager.standard_decrypt(encrypted_entry, context).await?
        };

        // Log the operation
        self.audit_logger.log_encryption_operation(context, "decrypt_memory", true).await?;

        Ok(decrypted_entry)
    }

    /// Perform secure computation on encrypted data
    pub async fn secure_compute(&mut self,
        encrypted_entries: &[EncryptedMemoryEntry],
        operation: SecureOperation,
        context: &SecurityContext
    ) -> Result<EncryptedComputationResult> {
        // Check permissions
        self.access_control.check_permission(context, Permission::ExecuteQueries).await?;

        // Perform homomorphic computation
        let result = self.encryption_manager.homomorphic_compute(
            encrypted_entries,
            operation.clone(),
            context
        ).await?;

        // Log the operation
        self.audit_logger.log_computation_operation(context, &operation, true).await?;

        Ok(result)
    }

    /// Get security metrics
    pub async fn get_security_metrics(&mut self, context: &SecurityContext) -> Result<SecurityMetrics> {
        // Check permissions
        self.access_control.check_permission(context, Permission::ViewAuditLogs).await?;

        let encryption_metrics = self.encryption_manager.get_metrics().await?;
        let privacy_metrics = self.privacy_manager.get_metrics().await?;
        let access_metrics = self.access_control.get_metrics().await?;
        let audit_metrics = self.audit_logger.get_metrics().await?;

        Ok(SecurityMetrics {
            encryption_metrics,
            privacy_metrics,
            access_metrics,
            audit_metrics,
            timestamp: Utc::now(),
        })
    }
}

/// Encrypted memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMemoryEntry {
    pub id: String,
    pub encrypted_data: Vec<u8>,
    pub encryption_algorithm: String,
    pub key_id: String,
    pub is_homomorphic: bool,
    pub privacy_level: PrivacyLevel,
    pub created_at: DateTime<Utc>,
    pub metadata: EncryptionMetadata,
}

/// Privacy levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
}

/// Encryption metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    pub algorithm_version: String,
    pub key_derivation: String,
    pub salt: Vec<u8>,
    pub iv: Vec<u8>,
    pub auth_tag: Option<Vec<u8>>,
    pub compression: Option<String>,
}

/// Secure operations for homomorphic computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecureOperation {
    Sum,
    Average,
    Count,
    Search { query: String },
    Similarity { threshold: f64 },
    Aggregate { function: String },
}

/// Result of encrypted computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedComputationResult {
    pub result_data: Vec<u8>,
    pub operation: SecureOperation,
    pub computation_id: String,
    pub timestamp: DateTime<Utc>,
    pub privacy_preserved: bool,
}

/// Security metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub encryption_metrics: encryption::EncryptionMetrics,
    pub privacy_metrics: privacy::PrivacyMetrics,
    pub access_metrics: access_control::AccessMetrics,
    pub audit_metrics: audit::AuditMetrics,
    pub timestamp: DateTime<Utc>,
}
