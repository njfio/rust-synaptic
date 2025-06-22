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

/// Standardized logging macros for consistent log formatting across the codebase
#[macro_export]
macro_rules! log_operation_start {
    ($operation:expr, $($key:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            $($key = $value,)*
            "Starting operation"
        );
    };
}

#[macro_export]
macro_rules! log_operation_success {
    ($operation:expr, $duration:expr, $($key:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            duration_ms = $duration.as_millis(),
            $($key = $value,)*
            "Operation completed successfully"
        );
    };
}

#[macro_export]
macro_rules! log_operation_error {
    ($operation:expr, $error:expr, $($key:ident = $value:expr),*) => {
        tracing::error!(
            operation = $operation,
            error = %$error,
            $($key = $value,)*
            "Operation failed"
        );
    };
}

#[macro_export]
macro_rules! log_performance_metric {
    ($metric_name:expr, $value:expr, $unit:expr) => {
        tracing::info!(
            metric_name = $metric_name,
            value = $value,
            unit = $unit,
            "Performance metric recorded"
        );
    };
}

#[macro_export]
macro_rules! log_memory_operation {
    ($operation:expr, $memory_key:expr, $($key:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            memory_key = $memory_key,
            $($key = $value,)*
            "Memory operation"
        );
    };
}

#[macro_export]
macro_rules! log_security_event {
    ($event_type:expr, $severity:expr, $resource:expr, $($key:ident = $value:expr),*) => {
        match $severity {
            "critical" | "high" => {
                tracing::error!(
                    event_type = $event_type,
                    severity = $severity,
                    resource = $resource,
                    $($key = $value,)*
                    "Security event detected"
                );
            }
            "medium" => {
                tracing::warn!(
                    event_type = $event_type,
                    severity = $severity,
                    resource = $resource,
                    $($key = $value,)*
                    "Security event detected"
                );
            }
            _ => {
                tracing::info!(
                    event_type = $event_type,
                    severity = $severity,
                    resource = $resource,
                    $($key = $value,)*
                    "Security event detected"
                );
            }
        }
    };
}

#[macro_export]
macro_rules! log_query_operation {
    ($query_type:expr, $complexity:expr, $result_count:expr, $duration_ms:expr) => {
        tracing::info!(
            query_type = $query_type,
            complexity = $complexity,
            result_count = $result_count,
            duration_ms = $duration_ms,
            "Query operation completed"
        );
    };
}

#[macro_export]
macro_rules! log_analytics_operation {
    ($operation:expr, $algorithm:expr, $data_size:expr, $($key:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            algorithm = $algorithm,
            data_size = $data_size,
            $($key = $value,)*
            "Analytics operation"
        );
    };
}

#[macro_export]
macro_rules! log_storage_operation {
    ($operation:expr, $backend:expr, $size_bytes:expr, $($key:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            backend = $backend,
            size_bytes = $size_bytes,
            $($key = $value,)*
            "Storage operation"
        );
    };
}

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

impl LoggingConfig {
    /// Create a development-friendly logging configuration
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            structured_logging: false,
            enable_performance_tracing: true,
            enable_audit_logging: false,
            enable_metrics: true,
            log_file: None,
            max_file_size: 50 * 1024 * 1024, // 50MB
            max_files: 5,
            compress_logs: false,
            log_format: LogFormat::Pretty,
            enable_distributed_tracing: false,
            trace_sampling_rate: 1.0,
            enable_log_streaming: false,
            stream_buffer_size: 100,
        }
    }

    /// Create a production-optimized logging configuration
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            structured_logging: true,
            enable_performance_tracing: true,
            enable_audit_logging: true,
            enable_metrics: true,
            log_file: Some(PathBuf::from("synaptic.log")),
            max_file_size: 500 * 1024 * 1024, // 500MB
            max_files: 20,
            compress_logs: true,
            log_format: LogFormat::Json,
            enable_distributed_tracing: true,
            trace_sampling_rate: 0.1, // Sample 10% for performance
            enable_log_streaming: true,
            stream_buffer_size: 10000,
        }
    }

    /// Create a high-performance logging configuration with minimal overhead
    pub fn high_performance() -> Self {
        Self {
            level: LogLevel::Warn,
            structured_logging: false,
            enable_performance_tracing: false,
            enable_audit_logging: false,
            enable_metrics: false,
            log_file: None,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 3,
            compress_logs: false,
            log_format: LogFormat::Compact,
            enable_distributed_tracing: false,
            trace_sampling_rate: 0.01, // Sample 1% only
            enable_log_streaming: false,
            stream_buffer_size: 50,
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

        // Log performance metrics with structured data
        if success {
            tracing::info!(
                operation_id = %operation_id,
                duration_ms = ?(end_time - Utc::now()).num_milliseconds(),
                success = true,
                "Performance trace completed successfully"
            );
        } else {
            tracing::warn!(
                operation_id = %operation_id,
                duration_ms = ?(end_time - Utc::now()).num_milliseconds(),
                success = false,
                error = ?error_message,
                "Performance trace completed with error"
            );
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
            user_id: user_id.clone(),
            session_id: session_id.clone(),
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

        // Log with structured context for better monitoring and alerting
        match risk_level {
            RiskLevel::Low => tracing::info!(
                audit_id = %audit_entry.id,
                operation = %operation,
                action = %action,
                resource = %resource,
                success = success,
                risk_level = "LOW",
                user_id = ?user_id,
                session_id = ?session_id,
                "Audit event recorded"
            ),
            RiskLevel::Medium => tracing::warn!(
                audit_id = %audit_entry.id,
                operation = %operation,
                action = %action,
                resource = %resource,
                success = success,
                risk_level = "MEDIUM",
                user_id = ?user_id,
                session_id = ?session_id,
                "Medium risk audit event"
            ),
            RiskLevel::High => tracing::error!(
                audit_id = %audit_entry.id,
                operation = %operation,
                action = %action,
                resource = %resource,
                success = success,
                risk_level = "HIGH",
                user_id = ?user_id,
                session_id = ?session_id,
                "High risk audit event"
            ),
            RiskLevel::Critical => tracing::error!(
                audit_id = %audit_entry.id,
                operation = %operation,
                action = %action,
                resource = %resource,
                success = success,
                risk_level = "CRITICAL",
                user_id = ?user_id,
                session_id = ?session_id,
                "CRITICAL SECURITY EVENT - Immediate attention required"
            ),
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
        let metrics_removed = {
            let mut metrics = self.performance_metrics.write().await;
            let original_count = metrics.len();
            metrics.retain(|m| m.start_time > cutoff_time);
            original_count - metrics.len()
        };

        // Clean up audit logs
        let logs_removed = {
            let mut logs = self.audit_logs.write().await;
            let original_count = logs.len();
            logs.retain(|l| l.timestamp > cutoff_time);
            original_count - logs.len()
        };

        tracing::info!(
            retention_hours = retention_hours,
            metrics_removed = metrics_removed,
            logs_removed = logs_removed,
            "Cleaned up logging data"
        );
        Ok(())
    }

    /// Export performance metrics to JSON
    pub async fn export_performance_metrics(&self) -> Result<String> {
        let metrics = self.performance_metrics.read().await;
        serde_json::to_string_pretty(&*metrics)
            .map_err(|e| MemoryError::processing_error(format!("Failed to export performance metrics: {}", e)))
    }

    /// Export audit logs to JSON
    pub async fn export_audit_logs(&self) -> Result<String> {
        let logs = self.audit_logs.read().await;
        serde_json::to_string_pretty(&*logs)
            .map_err(|e| MemoryError::processing_error(format!("Failed to export audit logs: {}", e)))
    }

    /// Get performance metrics summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.performance_metrics.read().await;

        let total_operations = metrics.len();
        let successful_operations = metrics.iter().filter(|m| m.success).count();
        let failed_operations = total_operations - successful_operations;

        let avg_duration = if !metrics.is_empty() {
            metrics.iter()
                .filter_map(|m| m.duration_ms)
                .sum::<u64>() as f64 / metrics.len() as f64
        } else {
            0.0
        };

        let operations_by_type = metrics.iter()
            .fold(std::collections::HashMap::new(), |mut acc, m| {
                *acc.entry(m.operation_name.clone()).or_insert(0) += 1;
                acc
            });

        PerformanceSummary {
            total_operations,
            successful_operations,
            failed_operations,
            success_rate: if total_operations > 0 {
                successful_operations as f64 / total_operations as f64
            } else {
                0.0
            },
            avg_duration_ms: avg_duration,
            operations_by_type,
        }
    }

    /// Get audit logs summary
    pub async fn get_audit_summary(&self) -> AuditSummary {
        let logs = self.audit_logs.read().await;

        let total_events = logs.len();
        let successful_events = logs.iter().filter(|l| l.success).count();
        let failed_events = total_events - successful_events;

        let events_by_risk = logs.iter()
            .fold(std::collections::HashMap::new(), |mut acc, l| {
                let risk_str = match l.risk_level {
                    RiskLevel::Low => "low",
                    RiskLevel::Medium => "medium",
                    RiskLevel::High => "high",
                    RiskLevel::Critical => "critical",
                };
                *acc.entry(risk_str.to_string()).or_insert(0) += 1;
                acc
            });

        let events_by_operation = logs.iter()
            .fold(std::collections::HashMap::new(), |mut acc, l| {
                *acc.entry(l.operation.clone()).or_insert(0) += 1;
                acc
            });

        AuditSummary {
            total_events,
            successful_events,
            failed_events,
            events_by_risk,
            events_by_operation,
        }
    }
}

/// Performance metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
    pub operations_by_type: HashMap<String, usize>,
}

/// Audit logs summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total_events: usize,
    pub successful_events: usize,
    pub failed_events: usize,
    pub events_by_risk: HashMap<String, usize>,
    pub events_by_operation: HashMap<String, usize>,
}
