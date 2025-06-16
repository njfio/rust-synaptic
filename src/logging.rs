//! Comprehensive logging and tracing infrastructure for the Synaptic memory system
//! 
//! This module provides centralized logging configuration, structured logging,
//! performance tracing, and audit logging capabilities.

use crate::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{Level, Span};
// Tracing subscriber is used in the initialize method
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Comprehensive logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Global log level
    pub level: LogLevel,
    /// Enable structured JSON logging
    pub structured_logging: bool,
    /// Enable performance tracing
    pub enable_performance_tracing: bool,
    /// Enable audit logging
    pub enable_audit_logging: bool,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Log file path (None for stdout only)
    pub log_file: Option<PathBuf>,
    /// Maximum log file size in bytes
    pub max_file_size: u64,
    /// Number of log files to retain
    pub max_files: u32,
    /// Enable log compression
    pub compress_logs: bool,
    /// Custom log format
    pub log_format: LogFormat,
    /// Enable distributed tracing
    pub enable_distributed_tracing: bool,
    /// Trace sampling rate (0.0 to 1.0)
    pub trace_sampling_rate: f64,
    /// Enable real-time log streaming
    pub enable_log_streaming: bool,
    /// Log buffer size for streaming
    pub stream_buffer_size: usize,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
    Full,
}

/// Performance metrics for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation_id: String,
    pub operation_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub memory_usage_bytes: Option<u64>,
    pub cpu_usage_percent: Option<f64>,
    pub success: bool,
    pub error_message: Option<String>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub resource: String,
    pub action: String,
    pub success: bool,
    pub details: HashMap<String, String>,
    pub risk_level: RiskLevel,
}

/// Risk levels for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Centralized logging manager
#[derive(Debug)]
pub struct LoggingManager {
    config: LoggingConfig,
    performance_metrics: Arc<RwLock<Vec<PerformanceMetrics>>>,
    audit_logs: Arc<RwLock<Vec<AuditLogEntry>>>,
    active_spans: Arc<RwLock<HashMap<String, Span>>>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            structured_logging: true,
            enable_performance_tracing: true,
            enable_audit_logging: true,
            enable_metrics: true,
            log_file: None,
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 10,
            compress_logs: true,
            log_format: LogFormat::Pretty,
            enable_distributed_tracing: false,
            trace_sampling_rate: 1.0,
            enable_log_streaming: false,
            stream_buffer_size: 1000,
        }
    }
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

impl LoggingManager {
    /// Create a new logging manager with the given configuration
    pub fn new(config: LoggingConfig) -> Self {
        Self {
            config,
            performance_metrics: Arc::new(RwLock::new(Vec::new())),
            audit_logs: Arc::new(RwLock::new(Vec::new())),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a reference to the logging configuration
    pub fn config(&self) -> &LoggingConfig {
        &self.config
    }

    /// Initialize the global logging infrastructure
    pub fn initialize(&self) -> Result<()> {
        // Try to initialize tracing subscriber, but don't fail if already initialized
        let init_result = match self.config.log_format {
            LogFormat::Json => {
                if let Some(ref log_file) = self.config.log_file {
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(log_file)
                        .map_err(|e| MemoryError::configuration(format!("Failed to open log file: {}", e)))?;

                    tracing_subscriber::fmt()
                        .with_writer(file)
                        .try_init()
                } else {
                    tracing_subscriber::fmt()
                        .try_init()
                }
            }
            LogFormat::Pretty => {
                if let Some(ref log_file) = self.config.log_file {
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(log_file)
                        .map_err(|e| MemoryError::configuration(format!("Failed to open log file: {}", e)))?;

                    tracing_subscriber::fmt()
                        .pretty()
                        .with_writer(file)
                        .try_init()
                } else {
                    tracing_subscriber::fmt()
                        .pretty()
                        .try_init()
                }
            }
            LogFormat::Compact => {
                if let Some(ref log_file) = self.config.log_file {
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(log_file)
                        .map_err(|e| MemoryError::configuration(format!("Failed to open log file: {}", e)))?;

                    tracing_subscriber::fmt()
                        .compact()
                        .with_writer(file)
                        .try_init()
                } else {
                    tracing_subscriber::fmt()
                        .compact()
                        .try_init()
                }
            }
            LogFormat::Full => {
                if let Some(ref log_file) = self.config.log_file {
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(log_file)
                        .map_err(|e| MemoryError::configuration(format!("Failed to open log file: {}", e)))?;

                    tracing_subscriber::fmt()
                        .with_file(true)
                        .with_line_number(true)
                        .with_writer(file)
                        .try_init()
                } else {
                    tracing_subscriber::fmt()
                        .with_file(true)
                        .with_line_number(true)
                        .try_init()
                }
            }
        };

        // Ignore the error if subscriber is already initialized
        match init_result {
            Ok(()) => {
                tracing::info!("Logging system initialized with level: {:?}", self.config.level);
            }
            Err(_) => {
                // Subscriber already initialized, just log a debug message
                tracing::debug!("Logging system already initialized, skipping");
            }
        }

        Ok(())
    }

    /// Start a performance trace for an operation
    pub async fn start_performance_trace(&self, operation_name: &str) -> Result<String> {
        if !self.config.enable_performance_tracing {
            return Ok(String::new());
        }

        let operation_id = Uuid::new_v4().to_string();
        let span = tracing::info_span!("performance_trace", 
            operation_id = %operation_id,
            operation_name = %operation_name
        );

        // Store the span for later use
        {
            let mut spans = self.active_spans.write().await;
            spans.insert(operation_id.clone(), span);
        }

        let metrics = PerformanceMetrics {
            operation_id: operation_id.clone(),
            operation_name: operation_name.to_string(),
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            memory_usage_bytes: None,
            cpu_usage_percent: None,
            success: false,
            error_message: None,
            custom_metrics: HashMap::new(),
        };

        {
            let mut performance_metrics = self.performance_metrics.write().await;
            performance_metrics.push(metrics);
        }

        tracing::info!("Started performance trace for operation: {}", operation_name);
        Ok(operation_id)
    }

    /// End a performance trace
    pub async fn end_performance_trace(&self, operation_id: &str, success: bool, error_message: Option<String>) -> Result<()> {
        if !self.config.enable_performance_tracing || operation_id.is_empty() {
            return Ok(());
        }

        let end_time = Utc::now();

        // Remove and drop the span
        {
            let mut spans = self.active_spans.write().await;
            spans.remove(operation_id);
        }

        // Update performance metrics
        {
            let mut performance_metrics = self.performance_metrics.write().await;
            if let Some(metrics) = performance_metrics.iter_mut().find(|m| m.operation_id == operation_id) {
                metrics.end_time = Some(end_time);
                metrics.duration_ms = Some((end_time - metrics.start_time).num_milliseconds() as u64);
                metrics.success = success;
                metrics.error_message = error_message.clone();
            }
        }

        if success {
            tracing::info!("Performance trace completed successfully for operation_id: {}", operation_id);
        } else {
            tracing::error!("Performance trace completed with error for operation_id: {}: {:?}", operation_id, error_message);
        }

        Ok(())
    }

    /// Log an audit event
    pub async fn log_audit_event(&self, 
        operation: &str,
        user_id: Option<String>,
        session_id: Option<String>,
        resource: &str,
        action: &str,
        success: bool,
        details: HashMap<String, String>,
        risk_level: RiskLevel
    ) -> Result<()> {
        if !self.config.enable_audit_logging {
            return Ok(());
        }

        let audit_entry = AuditLogEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operation: operation.to_string(),
            user_id,
            session_id,
            resource: resource.to_string(),
            action: action.to_string(),
            success,
            details,
            risk_level: risk_level.clone(),
        };

        {
            let mut audit_logs = self.audit_logs.write().await;
            audit_logs.push(audit_entry.clone());
        }

        match risk_level {
            RiskLevel::Low => tracing::info!("Audit event: {} - {} on {}", operation, action, resource),
            RiskLevel::Medium => tracing::warn!("Audit event (MEDIUM): {} - {} on {}", operation, action, resource),
            RiskLevel::High => tracing::error!("Audit event (HIGH): {} - {} on {}", operation, action, resource),
            RiskLevel::Critical => tracing::error!("Audit event (CRITICAL): {} - {} on {}", operation, action, resource),
        }

        Ok(())
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Vec<PerformanceMetrics> {
        let metrics = self.performance_metrics.read().await;
        metrics.clone()
    }

    /// Get audit logs
    pub async fn get_audit_logs(&self) -> Vec<AuditLogEntry> {
        let logs = self.audit_logs.read().await;
        logs.clone()
    }

    /// Clear old metrics and logs
    pub async fn cleanup_old_data(&self, retention_hours: u64) -> Result<()> {
        let cutoff_time = Utc::now() - chrono::Duration::hours(retention_hours as i64);

        // Clean up performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.retain(|m| m.start_time > cutoff_time);
        }

        // Clean up audit logs
        {
            let mut logs = self.audit_logs.write().await;
            logs.retain(|l| l.timestamp > cutoff_time);
        }

        tracing::info!("Cleaned up logging data older than {} hours", retention_hours);
        Ok(())
    }
}
