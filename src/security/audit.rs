//! Comprehensive Audit Logging Module
//! 
//! Implements enterprise-grade audit logging with real-time monitoring,
//! alerting, and compliance reporting capabilities.

use crate::error::Result;
use crate::security::{SecurityContext, AuditConfig, AlertThresholds, SecureOperation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Audit logger for security events
#[derive(Debug)]
pub struct AuditLogger {
    config: AuditConfig,
    audit_log: VecDeque<AuditEvent>,
    alert_tracker: AlertTracker,
    metrics: AuditMetrics,
}

impl AuditLogger {
    /// Create a new audit logger
    pub async fn new(config: &AuditConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            audit_log: VecDeque::new(),
            alert_tracker: AlertTracker::new(&config.alert_thresholds),
            metrics: AuditMetrics::default(),
        })
    }

    /// Log a memory operation
    pub async fn log_memory_operation(&mut self, 
        context: &SecurityContext, 
        operation: &str, 
        success: bool
    ) -> Result<()> {
        if !self.config.log_memory_operations {
            return Ok(());
        }

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::MemoryOperation,
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            operation: operation.to_string(),
            resource: "memory".to_string(),
            success,
            timestamp: Utc::now(),
            ip_address: None,
            user_agent: None,
            details: HashMap::new(),
            risk_level: if success { RiskLevel::Low } else { RiskLevel::Medium },
        };

        self.add_audit_event(event).await?;
        Ok(())
    }

    /// Log an authentication event
    pub async fn log_authentication_event(&mut self, 
        user_id: &str, 
        auth_type: &str, 
        success: bool,
        ip_address: Option<String>
    ) -> Result<()> {
        if !self.config.log_auth_events {
            return Ok(());
        }

        let mut details = HashMap::new();
        details.insert("auth_type".to_string(), auth_type.to_string());

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::Authentication,
            user_id: user_id.to_string(),
            session_id: "N/A".to_string(),
            operation: if success { "login_success" } else { "login_failure" }.to_string(),
            resource: "authentication".to_string(),
            success,
            timestamp: Utc::now(),
            ip_address,
            user_agent: None,
            details,
            risk_level: if success { RiskLevel::Low } else { RiskLevel::High },
        };

        self.add_audit_event(event).await?;

        // Track failed login attempts for alerting
        if !success {
            self.alert_tracker.record_failed_login(user_id);
        }

        Ok(())
    }

    /// Log an access control decision
    pub async fn log_access_decision(&mut self, 
        context: &SecurityContext, 
        permission: &str, 
        granted: bool
    ) -> Result<()> {
        if !self.config.log_access_decisions {
            return Ok(());
        }

        let mut details = HashMap::new();
        details.insert("permission".to_string(), permission.to_string());
        details.insert("roles".to_string(), context.roles.join(","));

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::AccessControl,
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            operation: if granted { "permission_granted" } else { "permission_denied" }.to_string(),
            resource: "access_control".to_string(),
            success: granted,
            timestamp: Utc::now(),
            ip_address: None,
            user_agent: None,
            details,
            risk_level: if granted { RiskLevel::Low } else { RiskLevel::Medium },
        };

        self.add_audit_event(event).await?;

        // Track unauthorized access attempts
        if !granted {
            self.alert_tracker.record_unauthorized_attempt(&context.user_id);
        }

        Ok(())
    }

    /// Log an encryption operation
    pub async fn log_encryption_operation(&mut self, 
        context: &SecurityContext, 
        operation: &str, 
        success: bool
    ) -> Result<()> {
        if !self.config.log_encryption_ops {
            return Ok(());
        }

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::Encryption,
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            operation: operation.to_string(),
            resource: "encryption".to_string(),
            success,
            timestamp: Utc::now(),
            ip_address: None,
            user_agent: None,
            details: HashMap::new(),
            risk_level: if success { RiskLevel::Low } else { RiskLevel::High },
        };

        self.add_audit_event(event).await?;

        // Track encryption failures for alerting
        if !success {
            self.alert_tracker.record_encryption_failure();
        }

        Ok(())
    }

    /// Log a computation operation
    pub async fn log_computation_operation(&mut self, 
        context: &SecurityContext, 
        operation: &SecureOperation, 
        success: bool
    ) -> Result<()> {
        let mut details = HashMap::new();
        details.insert("operation_type".to_string(), format!("{:?}", operation));

        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::Computation,
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            operation: "secure_computation".to_string(),
            resource: "homomorphic_engine".to_string(),
            success,
            timestamp: Utc::now(),
            ip_address: None,
            user_agent: None,
            details,
            risk_level: RiskLevel::Medium,
        };

        self.add_audit_event(event).await?;
        Ok(())
    }

    /// Log a system event
    pub async fn log_system_event(&mut self, 
        event_type: &str, 
        description: &str, 
        risk_level: RiskLevel
    ) -> Result<()> {
        let event = AuditEvent {
            id: Uuid::new_v4().to_string(),
            event_type: AuditEventType::System,
            user_id: "system".to_string(),
            session_id: "N/A".to_string(),
            operation: event_type.to_string(),
            resource: "system".to_string(),
            success: true,
            timestamp: Utc::now(),
            ip_address: None,
            user_agent: None,
            details: {
                let mut details = HashMap::new();
                details.insert("description".to_string(), description.to_string());
                details
            },
            risk_level,
        };

        self.add_audit_event(event).await?;
        Ok(())
    }

    /// Get audit events within a time range
    pub async fn get_audit_events(&self, 
        start_time: DateTime<Utc>, 
        end_time: DateTime<Utc>
    ) -> Result<Vec<AuditEvent>> {
        let events: Vec<AuditEvent> = self.audit_log.iter()
            .filter(|event| event.timestamp >= start_time && event.timestamp <= end_time)
            .cloned()
            .collect();
        
        Ok(events)
    }

    /// Get audit events for a specific user
    pub async fn get_user_audit_events(&self, 
        user_id: &str, 
        limit: Option<usize>
    ) -> Result<Vec<AuditEvent>> {
        let mut events: Vec<AuditEvent> = self.audit_log.iter()
            .filter(|event| event.user_id == user_id)
            .cloned()
            .collect();
        
        // Sort by timestamp (newest first)
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        if let Some(limit) = limit {
            events.truncate(limit);
        }
        
        Ok(events)
    }

    /// Get security alerts
    pub async fn get_security_alerts(&self) -> Result<Vec<SecurityAlert>> {
        Ok(self.alert_tracker.get_active_alerts())
    }

    /// Get audit metrics
    pub async fn get_metrics(&self) -> Result<AuditMetrics> {
        Ok(self.metrics.clone())
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(&self, 
        start_time: DateTime<Utc>, 
        end_time: DateTime<Utc>
    ) -> Result<ComplianceReport> {
        let events = self.get_audit_events(start_time, end_time).await?;
        
        let mut report = ComplianceReport {
            period_start: start_time,
            period_end: end_time,
            total_events: events.len(),
            authentication_events: 0,
            access_control_events: 0,
            encryption_events: 0,
            failed_operations: 0,
            high_risk_events: 0,
            unique_users: std::collections::HashSet::new(),
            compliance_score: 0.0,
        };

        for event in &events {
            match event.event_type {
                AuditEventType::Authentication => report.authentication_events += 1,
                AuditEventType::AccessControl => report.access_control_events += 1,
                AuditEventType::Encryption => report.encryption_events += 1,
                _ => {}
            }

            if !event.success {
                report.failed_operations += 1;
            }

            if matches!(event.risk_level, RiskLevel::High) {
                report.high_risk_events += 1;
            }

            report.unique_users.insert(event.user_id.clone());
        }

        // Calculate compliance score (simplified)
        let success_rate = if report.total_events > 0 {
            ((report.total_events - report.failed_operations) as f64 / report.total_events as f64) * 100.0
        } else {
            100.0
        };

        let risk_factor = if report.total_events > 0 {
            (report.high_risk_events as f64 / report.total_events as f64) * 100.0
        } else {
            0.0
        };

        report.compliance_score = (success_rate - risk_factor).max(0.0);

        Ok(report)
    }

    // Private helper methods

    async fn add_audit_event(&mut self, event: AuditEvent) -> Result<()> {
        // Add to audit log
        self.audit_log.push_back(event.clone());

        // Maintain log size (keep last 10000 events)
        while self.audit_log.len() > 10000 {
            self.audit_log.pop_front();
        }

        // Update metrics
        self.metrics.total_events += 1;
        match event.event_type {
            AuditEventType::Authentication => self.metrics.authentication_events += 1,
            AuditEventType::AccessControl => self.metrics.access_control_events += 1,
            AuditEventType::MemoryOperation => self.metrics.memory_operation_events += 1,
            AuditEventType::Encryption => self.metrics.encryption_events += 1,
            AuditEventType::Computation => self.metrics.computation_events += 1,
            AuditEventType::System => self.metrics.system_events += 1,
        }

        if !event.success {
            self.metrics.failed_events += 1;
        }

        // Check for alerts if enabled
        if self.config.enable_alerting {
            self.check_for_alerts(&event).await?;
        }

        // In production, would also write to persistent storage
        // and potentially send to SIEM systems

        Ok(())
    }

    async fn check_for_alerts(&mut self, event: &AuditEvent) -> Result<()> {
        // Check various alert conditions
        match event.event_type {
            AuditEventType::Authentication if !event.success => {
                if self.alert_tracker.should_alert_failed_logins() {
                    self.alert_tracker.create_alert(
                        AlertType::ExcessiveFailedLogins,
                        "Excessive failed login attempts detected".to_string(),
                        RiskLevel::High,
                    );
                }
            },
            AuditEventType::AccessControl if !event.success => {
                if self.alert_tracker.should_alert_unauthorized_access() {
                    self.alert_tracker.create_alert(
                        AlertType::UnauthorizedAccess,
                        "Multiple unauthorized access attempts detected".to_string(),
                        RiskLevel::High,
                    );
                }
            },
            AuditEventType::Encryption if !event.success => {
                if self.alert_tracker.should_alert_encryption_failures() {
                    self.alert_tracker.create_alert(
                        AlertType::EncryptionFailure,
                        "Multiple encryption failures detected".to_string(),
                        RiskLevel::Critical,
                    );
                }
            },
            _ => {}
        }

        Ok(())
    }
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: String,
    pub event_type: AuditEventType,
    pub user_id: String,
    pub session_id: String,
    pub operation: String,
    pub resource: String,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub details: HashMap<String, String>,
    pub risk_level: RiskLevel,
}

/// Types of audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    AccessControl,
    MemoryOperation,
    Encryption,
    Computation,
    System,
}

/// Risk levels for events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert tracker for security monitoring
#[derive(Debug)]
struct AlertTracker {
    thresholds: AlertThresholds,
    failed_logins: HashMap<String, u32>,
    unauthorized_attempts: HashMap<String, u32>,
    encryption_failures: u32,
    active_alerts: Vec<SecurityAlert>,
    last_reset: DateTime<Utc>,
}

impl AlertTracker {
    fn new(thresholds: &AlertThresholds) -> Self {
        Self {
            thresholds: thresholds.clone(),
            failed_logins: HashMap::new(),
            unauthorized_attempts: HashMap::new(),
            encryption_failures: 0,
            active_alerts: Vec::new(),
            last_reset: Utc::now(),
        }
    }

    fn record_failed_login(&mut self, user_id: &str) {
        self.reset_counters_if_needed();
        *self.failed_logins.entry(user_id.to_string()).or_insert(0) += 1;
    }

    fn record_unauthorized_attempt(&mut self, user_id: &str) {
        self.reset_counters_if_needed();
        *self.unauthorized_attempts.entry(user_id.to_string()).or_insert(0) += 1;
    }

    fn record_encryption_failure(&mut self) {
        self.reset_counters_if_needed();
        self.encryption_failures += 1;
    }

    fn should_alert_failed_logins(&self) -> bool {
        self.failed_logins.values().sum::<u32>() >= self.thresholds.failed_logins_per_hour
    }

    fn should_alert_unauthorized_access(&self) -> bool {
        self.unauthorized_attempts.values().sum::<u32>() >= self.thresholds.unauthorized_attempts_per_hour
    }

    fn should_alert_encryption_failures(&self) -> bool {
        self.encryption_failures >= self.thresholds.encryption_failures_per_hour
    }

    fn create_alert(&mut self, alert_type: AlertType, message: String, risk_level: RiskLevel) {
        let alert = SecurityAlert {
            id: Uuid::new_v4().to_string(),
            alert_type,
            message,
            risk_level,
            timestamp: Utc::now(),
            acknowledged: false,
        };
        self.active_alerts.push(alert);
    }

    fn get_active_alerts(&self) -> Vec<SecurityAlert> {
        self.active_alerts.clone()
    }

    fn reset_counters_if_needed(&mut self) {
        // Reset counters every hour
        if Utc::now() > self.last_reset + chrono::Duration::hours(1) {
            self.failed_logins.clear();
            self.unauthorized_attempts.clear();
            self.encryption_failures = 0;
            self.last_reset = Utc::now();
        }
    }
}

/// Security alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub id: String,
    pub alert_type: AlertType,
    pub message: String,
    pub risk_level: RiskLevel,
    pub timestamp: DateTime<Utc>,
    pub acknowledged: bool,
}

/// Types of security alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    ExcessiveFailedLogins,
    UnauthorizedAccess,
    EncryptionFailure,
    SuspiciousActivity,
    SystemAnomaly,
}

/// Compliance report
#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_events: usize,
    pub authentication_events: usize,
    pub access_control_events: usize,
    pub encryption_events: usize,
    pub failed_operations: usize,
    pub high_risk_events: usize,
    pub unique_users: std::collections::HashSet<String>,
    pub compliance_score: f64,
}

/// Audit metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditMetrics {
    pub total_events: u64,
    pub authentication_events: u64,
    pub access_control_events: u64,
    pub memory_operation_events: u64,
    pub encryption_events: u64,
    pub computation_events: u64,
    pub system_events: u64,
    pub failed_events: u64,
    pub events_per_hour: f64,
    pub failure_rate: f64,
}

impl AuditMetrics {
    pub fn calculate_rates(&mut self) {
        if self.total_events > 0 {
            self.failure_rate = (self.failed_events as f64 / self.total_events as f64) * 100.0;
        }
        // Events per hour would be calculated based on time window
        self.events_per_hour = self.total_events as f64; // Simplified
    }
}
