//! Observability Module
//!
//! This module provides comprehensive observability capabilities for the Synaptic
//! memory system including Prometheus metrics, OpenTelemetry tracing, Grafana
//! dashboards, and integrated monitoring solutions following industry best practices.

pub mod prometheus_metrics;
pub mod opentelemetry_tracing;
pub mod grafana_dashboards;
pub mod health_check;

use crate::error::Result;
use prometheus_metrics::{PrometheusMetrics, MetricTimer};
use opentelemetry_tracing::{OpenTelemetryTracing, TracingConfig};
use grafana_dashboards::GrafanaDashboardManager;
use health_check::{HealthCheckManager, HealthCheckConfig, SystemHealthReport};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Comprehensive observability manager for the Synaptic system
#[derive(Clone)]
pub struct ObservabilityManager {
    metrics: PrometheusMetrics,
    tracing: Arc<RwLock<Option<OpenTelemetryTracing>>>,
    dashboards: Arc<RwLock<GrafanaDashboardManager>>,
    health_check: HealthCheckManager,
    config: ObservabilityConfig,
}

/// Configuration for observability features
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub enable_dashboards: bool,
    pub enable_health_checks: bool,
    pub metrics_port: u16,
    pub tracing_config: TracingConfig,
    pub dashboard_refresh_interval: Duration,
    pub health_check_config: HealthCheckConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            enable_dashboards: true,
            enable_health_checks: true,
            metrics_port: 9090,
            tracing_config: TracingConfig::default(),
            dashboard_refresh_interval: Duration::from_secs(30),
            health_check_config: HealthCheckConfig::default(),
        }
    }
}

impl ObservabilityManager {
    /// Create a new observability manager with the provided configuration
    pub async fn new(config: ObservabilityConfig) -> Result<Self> {
        info!("Initializing Synaptic observability manager");
        
        // Initialize Prometheus metrics
        let metrics = if config.enable_metrics {
            PrometheusMetrics::new()?
        } else {
            info!("Metrics collection disabled");
            PrometheusMetrics::new()? // Still create for API compatibility
        };
        
        // Initialize OpenTelemetry tracing
        let tracing = if config.enable_tracing {
            let tracing_instance = OpenTelemetryTracing::new(config.tracing_config.clone()).await?;
            Arc::new(RwLock::new(Some(tracing_instance)))
        } else {
            info!("Distributed tracing disabled");
            Arc::new(RwLock::new(None))
        };
        
        // Initialize Grafana dashboard manager
        let mut dashboard_manager = GrafanaDashboardManager::new();
        if config.enable_dashboards {
            dashboard_manager.create_synaptic_overview_dashboard()?;
        }
        let dashboards = Arc::new(RwLock::new(dashboard_manager));

        // Initialize health check manager
        let health_check = HealthCheckManager::new(config.health_check_config.clone());

        // Start periodic health checks if enabled
        if config.enable_health_checks {
            health_check.start_periodic_checks().await?;
        }

        info!("Observability manager initialized successfully");

        Ok(Self {
            metrics,
            tracing,
            dashboards,
            health_check,
            config,
        })
    }
    
    /// Get the Prometheus metrics instance
    pub fn metrics(&self) -> &PrometheusMetrics {
        &self.metrics
    }
    
    /// Get the OpenTelemetry tracing instance
    pub async fn tracing(&self) -> Option<OpenTelemetryTracing> {
        self.tracing.read().await.clone()
    }
    
    /// Start a memory operation with both metrics and tracing
    pub async fn start_memory_operation(&self, operation: &str, memory_type: &str) -> Result<ObservabilityContext> {
        let timer = if self.config.enable_metrics {
            Some(MetricTimer::new_memory_timer(self.metrics.clone(), operation, memory_type))
        } else {
            None
        };
        
        let span_id = if self.config.enable_tracing {
            if let Some(tracing) = self.tracing.read().await.as_ref() {
                Some(tracing.start_memory_span(operation, memory_type).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        debug!("Started memory operation: {} {}", operation, memory_type);
        
        Ok(ObservabilityContext {
            operation_type: OperationType::Memory,
            timer,
            span_id,
            observability: self.clone(),
        })
    }
    
    /// Start a query operation with both metrics and tracing
    pub async fn start_query_operation(&self, query_type: &str, complexity: &str, result_size: &str) -> Result<ObservabilityContext> {
        let timer = if self.config.enable_metrics {
            Some(MetricTimer::new_query_timer(self.metrics.clone(), query_type, complexity, result_size))
        } else {
            None
        };
        
        let span_id = if self.config.enable_tracing {
            if let Some(tracing) = self.tracing.read().await.as_ref() {
                Some(tracing.start_query_span(query_type, complexity).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        debug!("Started query operation: {} {} {}", query_type, complexity, result_size);
        
        Ok(ObservabilityContext {
            operation_type: OperationType::Query { 
                query_type: query_type.to_string(),
                complexity: complexity.to_string(),
                result_size: result_size.to_string(),
            },
            timer,
            span_id,
            observability: self.clone(),
        })
    }
    
    /// Start an analytics operation with both metrics and tracing
    pub async fn start_analytics_operation(&self, operation: &str, algorithm: &str) -> Result<ObservabilityContext> {
        let span_id = if self.config.enable_tracing {
            if let Some(tracing) = self.tracing.read().await.as_ref() {
                Some(tracing.start_analytics_span(operation, algorithm).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        debug!("Started analytics operation: {} {}", operation, algorithm);
        
        Ok(ObservabilityContext {
            operation_type: OperationType::Analytics {
                operation: operation.to_string(),
                algorithm: algorithm.to_string(),
            },
            timer: None,
            span_id,
            observability: self.clone(),
        })
    }
    
    /// Start a storage operation with both metrics and tracing
    pub async fn start_storage_operation(&self, operation: &str, backend: &str) -> Result<ObservabilityContext> {
        let span_id = if self.config.enable_tracing {
            if let Some(tracing) = self.tracing.read().await.as_ref() {
                Some(tracing.start_storage_span(operation, backend).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        debug!("Started storage operation: {} {}", operation, backend);
        
        Ok(ObservabilityContext {
            operation_type: OperationType::Storage {
                operation: operation.to_string(),
                backend: backend.to_string(),
            },
            timer: None,
            span_id,
            observability: self.clone(),
        })
    }
    
    /// Start a security operation with both metrics and tracing
    pub async fn start_security_operation(&self, operation: &str, resource_type: &str) -> Result<ObservabilityContext> {
        let span_id = if self.config.enable_tracing {
            if let Some(tracing) = self.tracing.read().await.as_ref() {
                Some(tracing.start_security_span(operation, resource_type).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        debug!("Started security operation: {} {}", operation, resource_type);
        
        Ok(ObservabilityContext {
            operation_type: OperationType::Security {
                operation: operation.to_string(),
                resource_type: resource_type.to_string(),
            },
            timer: None,
            span_id,
            observability: self.clone(),
        })
    }
    
    /// Update system health metrics
    pub async fn update_system_health(&self, memory_usage: i64, cpu_usage: f64, active_connections: i64) {
        if self.config.enable_metrics {
            self.metrics.update_system_memory_usage(memory_usage);
            self.metrics.update_system_cpu_usage(cpu_usage);
            self.metrics.update_active_connections(active_connections);
        }
    }
    
    /// Record a security event
    pub async fn record_security_event(&self, event_type: &str, severity: &str, source: &str) {
        if self.config.enable_metrics {
            match severity {
                "critical" | "high" => {
                    self.metrics.record_security_violation(event_type, severity, source);
                }
                _ => {
                    // Record as general security metric
                }
            }
        }
        
        warn!("Security event recorded: {} {} {}", event_type, severity, source);
    }
    
    /// Flush all observability data
    pub async fn flush(&self) -> Result<()> {
        if let Some(tracing) = self.tracing.read().await.as_ref() {
            tracing.flush().await?;
        }
        
        info!("Observability data flushed");
        Ok(())
    }
    
    /// Shutdown observability systems
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down observability systems");
        
        if let Some(tracing) = self.tracing.write().await.take() {
            tracing.shutdown().await?;
        }
        
        info!("Observability shutdown complete");
        Ok(())
    }
    
    /// Get observability configuration
    pub fn config(&self) -> &ObservabilityConfig {
        &self.config
    }

    /// Get the health check manager
    pub fn health_check(&self) -> &HealthCheckManager {
        &self.health_check
    }

    /// Get system health report
    pub async fn get_health_report(&self) -> SystemHealthReport {
        self.health_check.check_all_health().await
    }

    /// Register a health checker
    pub async fn register_health_checker(&self, checker: Box<dyn health_check::HealthChecker>) {
        self.health_check.register_checker(checker).await;
    }

    /// Register a dependency for monitoring
    pub async fn register_dependency(&self, name: String, endpoint: String) {
        self.health_check.register_dependency(name, endpoint).await;
    }
}

/// Context for tracking operations across metrics and tracing
pub struct ObservabilityContext {
    operation_type: OperationType,
    timer: Option<MetricTimer>,
    span_id: Option<String>,
    observability: ObservabilityManager,
}

/// Types of operations being tracked
#[derive(Debug, Clone)]
enum OperationType {
    Memory,
    Query {
        query_type: String,
        complexity: String,
        result_size: String,
    },
    Analytics {
        operation: String,
        algorithm: String,
    },
    Storage {
        operation: String,
        backend: String,
    },
    Security {
        operation: String,
        resource_type: String,
    },
}

impl ObservabilityContext {
    /// Add an attribute to the current span
    pub async fn add_attribute(&self, key: &str, value: &str) -> Result<()> {
        if let Some(span_id) = &self.span_id {
            if let Some(tracing) = self.observability.tracing.read().await.as_ref() {
                tracing.add_span_attribute(span_id, key, value).await?;
            }
        }
        Ok(())
    }
    
    /// Record an error in the current operation
    pub async fn record_error(&self, error: &str, error_type: &str) -> Result<()> {
        if let Some(span_id) = &self.span_id {
            if let Some(tracing) = self.observability.tracing.read().await.as_ref() {
                tracing.record_span_error(span_id, error, error_type).await?;
            }
        }
        Ok(())
    }
    
    /// Finish the operation successfully
    pub async fn finish_success(self, result_size: Option<usize>) -> Result<()> {
        // Finish span
        if let Some(span_id) = &self.span_id {
            if let Some(tracing) = self.observability.tracing.read().await.as_ref() {
                tracing.finish_span_success(span_id, result_size).await?;
            }
        }
        
        // Finish timer and record metrics
        if let Some(timer) = self.timer {
            match self.operation_type {
                OperationType::Memory => {
                    timer.finish_memory_operation("success", result_size.map(|s| s as u64));
                }
                OperationType::Query { .. } => {
                    timer.finish_query_operation("success", "optimized");
                }
                _ => {
                    // Other operation types don't use timers currently
                }
            }
        }
        
        Ok(())
    }
    
    /// Finish the operation with an error
    pub async fn finish_error(self, error: &str, error_type: &str) -> Result<()> {
        // Finish span with error
        if let Some(span_id) = &self.span_id {
            if let Some(tracing) = self.observability.tracing.read().await.as_ref() {
                tracing.finish_span_error(span_id, error, error_type).await?;
            }
        }
        
        // Finish timer and record metrics
        if let Some(timer) = self.timer {
            match self.operation_type {
                OperationType::Memory => {
                    timer.finish_memory_operation("error", None);
                }
                OperationType::Query { .. } => {
                    timer.finish_query_operation("error", "none");
                }
                _ => {
                    // Other operation types don't use timers currently
                }
            }
        }
        
        Ok(())
    }
}
