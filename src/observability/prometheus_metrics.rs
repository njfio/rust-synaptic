//! Prometheus Metrics Collection System
//!
//! This module implements comprehensive Prometheus metrics collection for the Synaptic
//! memory system, providing detailed insights into memory operations, performance
//! indicators, and system health with proper labeling strategies and efficient collection.

use crate::error::Result;
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, IntCounter,
    IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Comprehensive Prometheus metrics collector for Synaptic memory system
#[derive(Clone)]
pub struct PrometheusMetrics {
    registry: Arc<Registry>,
    
    // Memory operation metrics
    memory_operations_total: IntCounterVec,
    memory_operation_duration: HistogramVec,
    memory_size_bytes: HistogramVec,
    memory_nodes_total: IntGaugeVec,
    memory_relationships_total: IntGaugeVec,
    
    // Performance metrics
    query_duration_seconds: HistogramVec,
    query_operations_total: IntCounterVec,
    cache_operations_total: IntCounterVec,
    cache_hit_ratio: GaugeVec,
    
    // System health metrics
    system_memory_usage_bytes: IntGauge,
    system_cpu_usage_percent: Gauge,
    active_connections: IntGauge,
    error_rate: GaugeVec,
    
    // Storage metrics
    storage_operations_total: IntCounterVec,
    storage_operation_duration: HistogramVec,
    storage_size_bytes: IntGaugeVec,
    
    // Analytics metrics
    analytics_operations_total: IntCounterVec,
    analytics_processing_duration: HistogramVec,
    vector_search_duration: HistogramVec,
    
    // Security metrics
    authentication_attempts_total: IntCounterVec,
    authorization_checks_total: IntCounterVec,
    security_violations_total: IntCounterVec,
    
    // Learning metrics
    learning_operations_total: IntCounterVec,
    model_training_duration: HistogramVec,
    model_accuracy: GaugeVec,
    
    // Custom metrics registry
    custom_metrics: Arc<RwLock<HashMap<String, Box<dyn CustomMetric + Send + Sync>>>>,
}

/// Trait for custom metrics that can be registered dynamically
pub trait CustomMetric {
    fn collect(&self) -> Result<Vec<prometheus::proto::MetricFamily>>;
    fn describe(&self) -> Vec<prometheus::proto::MetricFamily>;
}

impl PrometheusMetrics {
    /// Create a new Prometheus metrics collector with comprehensive metric definitions
    pub fn new() -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        // Memory operation metrics
        let memory_operations_total = IntCounterVec::new(
            Opts::new("synaptic_memory_operations_total", "Total number of memory operations")
                .namespace("synaptic")
                .subsystem("memory"),
            &["operation_type", "memory_type", "status"]
        )?;
        
        let memory_operation_duration = HistogramVec::new(
            HistogramOpts::new("synaptic_memory_operation_duration_seconds", "Duration of memory operations")
                .namespace("synaptic")
                .subsystem("memory")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["operation_type", "memory_type"]
        )?;
        
        let memory_size_bytes = HistogramVec::new(
            HistogramOpts::new("synaptic_memory_size_bytes", "Size of memory objects in bytes")
                .namespace("synaptic")
                .subsystem("memory")
                .buckets(vec![1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0, 4194304.0, 16777216.0]),
            &["memory_type", "content_type"]
        )?;
        
        let memory_nodes_total = IntGaugeVec::new(
            Opts::new("synaptic_memory_nodes_total", "Total number of memory nodes")
                .namespace("synaptic")
                .subsystem("memory"),
            &["memory_type", "status"]
        )?;
        
        let memory_relationships_total = IntGaugeVec::new(
            Opts::new("synaptic_memory_relationships_total", "Total number of memory relationships")
                .namespace("synaptic")
                .subsystem("memory"),
            &["relationship_type", "strength_category"]
        )?;
        
        // Performance metrics
        let query_duration_seconds = HistogramVec::new(
            HistogramOpts::new("synaptic_query_duration_seconds", "Duration of query operations")
                .namespace("synaptic")
                .subsystem("performance")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
            &["query_type", "complexity", "result_size"]
        )?;
        
        let query_operations_total = IntCounterVec::new(
            Opts::new("synaptic_query_operations_total", "Total number of query operations")
                .namespace("synaptic")
                .subsystem("performance"),
            &["query_type", "status", "optimization_level"]
        )?;
        
        let cache_operations_total = IntCounterVec::new(
            Opts::new("synaptic_cache_operations_total", "Total number of cache operations")
                .namespace("synaptic")
                .subsystem("performance"),
            &["operation_type", "cache_type", "result"]
        )?;
        
        let cache_hit_ratio = GaugeVec::new(
            Opts::new("synaptic_cache_hit_ratio", "Cache hit ratio")
                .namespace("synaptic")
                .subsystem("performance"),
            &["cache_type", "time_window"]
        )?;
        
        // System health metrics
        let system_memory_usage_bytes = IntGauge::new(
            "synaptic_system_memory_usage_bytes",
            "Current system memory usage in bytes"
        )?;
        
        let system_cpu_usage_percent = Gauge::new(
            "synaptic_system_cpu_usage_percent",
            "Current system CPU usage percentage"
        )?;
        
        let active_connections = IntGauge::new(
            "synaptic_active_connections",
            "Number of active connections"
        )?;
        
        let error_rate = GaugeVec::new(
            Opts::new("synaptic_error_rate", "Error rate by component")
                .namespace("synaptic")
                .subsystem("health"),
            &["component", "error_type", "severity"]
        )?;
        
        // Storage metrics
        let storage_operations_total = IntCounterVec::new(
            Opts::new("synaptic_storage_operations_total", "Total number of storage operations")
                .namespace("synaptic")
                .subsystem("storage"),
            &["operation_type", "storage_backend", "status"]
        )?;
        
        let storage_operation_duration = HistogramVec::new(
            HistogramOpts::new("synaptic_storage_operation_duration_seconds", "Duration of storage operations")
                .namespace("synaptic")
                .subsystem("storage")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
            &["operation_type", "storage_backend"]
        )?;
        
        let storage_size_bytes = IntGaugeVec::new(
            Opts::new("synaptic_storage_size_bytes", "Storage size in bytes")
                .namespace("synaptic")
                .subsystem("storage"),
            &["storage_backend", "data_type"]
        )?;
        
        // Analytics metrics
        let analytics_operations_total = IntCounterVec::new(
            Opts::new("synaptic_analytics_operations_total", "Total number of analytics operations")
                .namespace("synaptic")
                .subsystem("analytics"),
            &["operation_type", "algorithm", "status"]
        )?;
        
        let analytics_processing_duration = HistogramVec::new(
            HistogramOpts::new("synaptic_analytics_processing_duration_seconds", "Duration of analytics processing")
                .namespace("synaptic")
                .subsystem("analytics")
                .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
            &["operation_type", "algorithm", "data_size"]
        )?;
        
        let vector_search_duration = HistogramVec::new(
            HistogramOpts::new("synaptic_vector_search_duration_seconds", "Duration of vector search operations")
                .namespace("synaptic")
                .subsystem("analytics")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["search_type", "index_type", "result_count"]
        )?;
        
        // Security metrics
        let authentication_attempts_total = IntCounterVec::new(
            Opts::new("synaptic_authentication_attempts_total", "Total number of authentication attempts")
                .namespace("synaptic")
                .subsystem("security"),
            &["method", "status", "user_type"]
        )?;
        
        let authorization_checks_total = IntCounterVec::new(
            Opts::new("synaptic_authorization_checks_total", "Total number of authorization checks")
                .namespace("synaptic")
                .subsystem("security"),
            &["resource_type", "action", "status"]
        )?;
        
        let security_violations_total = IntCounterVec::new(
            Opts::new("synaptic_security_violations_total", "Total number of security violations")
                .namespace("synaptic")
                .subsystem("security"),
            &["violation_type", "severity", "source"]
        )?;
        
        // Learning metrics
        let learning_operations_total = IntCounterVec::new(
            Opts::new("synaptic_learning_operations_total", "Total number of learning operations")
                .namespace("synaptic")
                .subsystem("learning"),
            &["operation_type", "algorithm", "status"]
        )?;
        
        let model_training_duration = HistogramVec::new(
            HistogramOpts::new("synaptic_model_training_duration_seconds", "Duration of model training")
                .namespace("synaptic")
                .subsystem("learning")
                .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]),
            &["model_type", "algorithm", "data_size"]
        )?;
        
        let model_accuracy = GaugeVec::new(
            Opts::new("synaptic_model_accuracy", "Model accuracy metrics")
                .namespace("synaptic")
                .subsystem("learning"),
            &["model_type", "metric_type", "dataset"]
        )?;
        
        // Register all metrics
        registry.register(Box::new(memory_operations_total.clone()))?;
        registry.register(Box::new(memory_operation_duration.clone()))?;
        registry.register(Box::new(memory_size_bytes.clone()))?;
        registry.register(Box::new(memory_nodes_total.clone()))?;
        registry.register(Box::new(memory_relationships_total.clone()))?;
        registry.register(Box::new(query_duration_seconds.clone()))?;
        registry.register(Box::new(query_operations_total.clone()))?;
        registry.register(Box::new(cache_operations_total.clone()))?;
        registry.register(Box::new(cache_hit_ratio.clone()))?;
        registry.register(Box::new(system_memory_usage_bytes.clone()))?;
        registry.register(Box::new(system_cpu_usage_percent.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(error_rate.clone()))?;
        registry.register(Box::new(storage_operations_total.clone()))?;
        registry.register(Box::new(storage_operation_duration.clone()))?;
        registry.register(Box::new(storage_size_bytes.clone()))?;
        registry.register(Box::new(analytics_operations_total.clone()))?;
        registry.register(Box::new(analytics_processing_duration.clone()))?;
        registry.register(Box::new(vector_search_duration.clone()))?;
        registry.register(Box::new(authentication_attempts_total.clone()))?;
        registry.register(Box::new(authorization_checks_total.clone()))?;
        registry.register(Box::new(security_violations_total.clone()))?;
        registry.register(Box::new(learning_operations_total.clone()))?;
        registry.register(Box::new(model_training_duration.clone()))?;
        registry.register(Box::new(model_accuracy.clone()))?;
        
        Ok(Self {
            registry,
            memory_operations_total,
            memory_operation_duration,
            memory_size_bytes,
            memory_nodes_total,
            memory_relationships_total,
            query_duration_seconds,
            query_operations_total,
            cache_operations_total,
            cache_hit_ratio,
            system_memory_usage_bytes,
            system_cpu_usage_percent,
            active_connections,
            error_rate,
            storage_operations_total,
            storage_operation_duration,
            storage_size_bytes,
            analytics_operations_total,
            analytics_processing_duration,
            vector_search_duration,
            authentication_attempts_total,
            authorization_checks_total,
            security_violations_total,
            learning_operations_total,
            model_training_duration,
            model_accuracy,
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Get the Prometheus registry for HTTP endpoint exposure
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    // Memory operation metrics

    /// Record a memory operation with timing and metadata
    pub fn record_memory_operation(&self, operation_type: &str, memory_type: &str, duration: Duration, status: &str, size_bytes: Option<u64>) {
        self.memory_operations_total
            .with_label_values(&[operation_type, memory_type, status])
            .inc();

        self.memory_operation_duration
            .with_label_values(&[operation_type, memory_type])
            .observe(duration.as_secs_f64());

        if let Some(size) = size_bytes {
            let content_type = match memory_type {
                "text" => "text",
                "image" => "binary",
                "audio" => "binary",
                "video" => "binary",
                _ => "unknown",
            };

            self.memory_size_bytes
                .with_label_values(&[memory_type, content_type])
                .observe(size as f64);
        }

        debug!("Recorded memory operation: {} {} {} {:?}", operation_type, memory_type, status, duration);
    }

    /// Update memory node counts
    pub fn update_memory_node_count(&self, memory_type: &str, status: &str, count: i64) {
        self.memory_nodes_total
            .with_label_values(&[memory_type, status])
            .set(count);
    }

    /// Update memory relationship counts
    pub fn update_memory_relationship_count(&self, relationship_type: &str, strength_category: &str, count: i64) {
        self.memory_relationships_total
            .with_label_values(&[relationship_type, strength_category])
            .set(count);
    }

    // Performance metrics

    /// Record a query operation with detailed metrics
    pub fn record_query_operation(&self, query_type: &str, complexity: &str, result_size: &str, duration: Duration, status: &str, optimization_level: &str) {
        self.query_operations_total
            .with_label_values(&[query_type, status, optimization_level])
            .inc();

        self.query_duration_seconds
            .with_label_values(&[query_type, complexity, result_size])
            .observe(duration.as_secs_f64());

        debug!("Recorded query operation: {} {} {} {:?}", query_type, status, optimization_level, duration);
    }

    /// Record cache operation
    pub fn record_cache_operation(&self, operation_type: &str, cache_type: &str, result: &str) {
        self.cache_operations_total
            .with_label_values(&[operation_type, cache_type, result])
            .inc();
    }

    /// Update cache hit ratio
    pub fn update_cache_hit_ratio(&self, cache_type: &str, time_window: &str, ratio: f64) {
        self.cache_hit_ratio
            .with_label_values(&[cache_type, time_window])
            .set(ratio);
    }

    // System health metrics

    /// Update system memory usage
    pub fn update_system_memory_usage(&self, bytes: i64) {
        self.system_memory_usage_bytes.set(bytes);
    }

    /// Update system CPU usage
    pub fn update_system_cpu_usage(&self, percent: f64) {
        self.system_cpu_usage_percent.set(percent);
    }

    /// Update active connections count
    pub fn update_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }

    /// Update error rate for a component
    pub fn update_error_rate(&self, component: &str, error_type: &str, severity: &str, rate: f64) {
        self.error_rate
            .with_label_values(&[component, error_type, severity])
            .set(rate);
    }

    // Storage metrics

    /// Record storage operation
    pub fn record_storage_operation(&self, operation_type: &str, storage_backend: &str, duration: Duration, status: &str) {
        self.storage_operations_total
            .with_label_values(&[operation_type, storage_backend, status])
            .inc();

        self.storage_operation_duration
            .with_label_values(&[operation_type, storage_backend])
            .observe(duration.as_secs_f64());

        debug!("Recorded storage operation: {} {} {} {:?}", operation_type, storage_backend, status, duration);
    }

    /// Update storage size
    pub fn update_storage_size(&self, storage_backend: &str, data_type: &str, size_bytes: i64) {
        self.storage_size_bytes
            .with_label_values(&[storage_backend, data_type])
            .set(size_bytes);
    }

    // Analytics metrics

    /// Record analytics operation
    pub fn record_analytics_operation(&self, operation_type: &str, algorithm: &str, duration: Duration, status: &str, data_size: &str) {
        self.analytics_operations_total
            .with_label_values(&[operation_type, algorithm, status])
            .inc();

        self.analytics_processing_duration
            .with_label_values(&[operation_type, algorithm, data_size])
            .observe(duration.as_secs_f64());

        debug!("Recorded analytics operation: {} {} {} {:?}", operation_type, algorithm, status, duration);
    }

    /// Record vector search operation
    pub fn record_vector_search(&self, search_type: &str, index_type: &str, result_count: &str, duration: Duration) {
        self.vector_search_duration
            .with_label_values(&[search_type, index_type, result_count])
            .observe(duration.as_secs_f64());

        debug!("Recorded vector search: {} {} {} {:?}", search_type, index_type, result_count, duration);
    }

    // Security metrics

    /// Record authentication attempt
    pub fn record_authentication_attempt(&self, method: &str, status: &str, user_type: &str) {
        self.authentication_attempts_total
            .with_label_values(&[method, status, user_type])
            .inc();

        debug!("Recorded authentication attempt: {} {} {}", method, status, user_type);
    }

    /// Record authorization check
    pub fn record_authorization_check(&self, resource_type: &str, action: &str, status: &str) {
        self.authorization_checks_total
            .with_label_values(&[resource_type, action, status])
            .inc();

        debug!("Recorded authorization check: {} {} {}", resource_type, action, status);
    }

    /// Record security violation
    pub fn record_security_violation(&self, violation_type: &str, severity: &str, source: &str) {
        self.security_violations_total
            .with_label_values(&[violation_type, severity, source])
            .inc();

        warn!("Recorded security violation: {} {} {}", violation_type, severity, source);
    }

    // Learning metrics

    /// Record learning operation
    pub fn record_learning_operation(&self, operation_type: &str, algorithm: &str, status: &str) {
        self.learning_operations_total
            .with_label_values(&[operation_type, algorithm, status])
            .inc();

        debug!("Recorded learning operation: {} {} {}", operation_type, algorithm, status);
    }

    /// Record model training
    pub fn record_model_training(&self, model_type: &str, algorithm: &str, data_size: &str, duration: Duration) {
        self.model_training_duration
            .with_label_values(&[model_type, algorithm, data_size])
            .observe(duration.as_secs_f64());

        info!("Recorded model training: {} {} {} {:?}", model_type, algorithm, data_size, duration);
    }

    /// Update model accuracy
    pub fn update_model_accuracy(&self, model_type: &str, metric_type: &str, dataset: &str, accuracy: f64) {
        self.model_accuracy
            .with_label_values(&[model_type, metric_type, dataset])
            .set(accuracy);

        info!("Updated model accuracy: {} {} {} {:.4}", model_type, metric_type, dataset, accuracy);
    }

    // Custom metrics management

    /// Register a custom metric
    pub async fn register_custom_metric(&self, name: String, metric: Box<dyn CustomMetric + Send + Sync>) -> Result<()> {
        let mut custom_metrics = self.custom_metrics.write().await;
        custom_metrics.insert(name.clone(), metric);
        info!("Registered custom metric: {}", name);
        Ok(())
    }

    /// Unregister a custom metric
    pub async fn unregister_custom_metric(&self, name: &str) -> Result<()> {
        let mut custom_metrics = self.custom_metrics.write().await;
        if custom_metrics.remove(name).is_some() {
            info!("Unregistered custom metric: {}", name);
            Ok(())
        } else {
            warn!("Attempted to unregister non-existent custom metric: {}", name);
            Err(crate::error::SynapticError::NotFound(format!("Custom metric '{}' not found", name)))
        }
    }

    /// Get all metric families for Prometheus exposition
    pub async fn gather(&self) -> Result<Vec<prometheus::proto::MetricFamily>> {
        let mut families = self.registry.gather();

        // Add custom metrics
        let custom_metrics = self.custom_metrics.read().await;
        for (name, metric) in custom_metrics.iter() {
            match metric.collect() {
                Ok(mut custom_families) => {
                    families.append(&mut custom_families);
                }
                Err(e) => {
                    error!("Failed to collect custom metric '{}': {}", name, e);
                }
            }
        }

        Ok(families)
    }

    /// Reset all metrics (useful for testing)
    pub fn reset_all(&self) {
        // Note: Prometheus metrics don't have a built-in reset method
        // This would typically be used in test scenarios
        warn!("Reset all metrics called - this should only be used in testing");
    }

    /// Get current metric values as a formatted string (for debugging)
    pub async fn debug_dump(&self) -> String {
        let families = match self.gather().await {
            Ok(families) => families,
            Err(e) => {
                return format!("Error gathering metrics: {}", e);
            }
        };

        let mut output = String::new();
        output.push_str("=== Synaptic Prometheus Metrics Debug Dump ===\n\n");

        for family in families {
            output.push_str(&format!("Metric Family: {}\n", family.get_name()));
            output.push_str(&format!("Type: {:?}\n", family.get_field_type()));
            output.push_str(&format!("Help: {}\n", family.get_help()));

            for metric in family.get_metric() {
                output.push_str("  Labels: ");
                for label in metric.get_label() {
                    output.push_str(&format!("{}={} ", label.get_name(), label.get_value()));
                }
                output.push('\n');

                if metric.has_counter() {
                    output.push_str(&format!("  Counter Value: {}\n", metric.get_counter().get_value()));
                } else if metric.has_gauge() {
                    output.push_str(&format!("  Gauge Value: {}\n", metric.get_gauge().get_value()));
                } else if metric.has_histogram() {
                    let hist = metric.get_histogram();
                    output.push_str(&format!("  Histogram Sample Count: {}\n", hist.get_sample_count()));
                    output.push_str(&format!("  Histogram Sample Sum: {}\n", hist.get_sample_sum()));
                }
            }
            output.push('\n');
        }

        output
    }
}

/// Timer utility for measuring operation durations
pub struct MetricTimer {
    start: Instant,
    metrics: PrometheusMetrics,
    operation_type: String,
    labels: Vec<String>,
}

impl MetricTimer {
    /// Create a new timer for memory operations
    pub fn new_memory_timer(metrics: PrometheusMetrics, operation_type: &str, memory_type: &str) -> Self {
        Self {
            start: Instant::now(),
            metrics,
            operation_type: operation_type.to_string(),
            labels: vec![memory_type.to_string()],
        }
    }

    /// Create a new timer for query operations
    pub fn new_query_timer(metrics: PrometheusMetrics, query_type: &str, complexity: &str, result_size: &str) -> Self {
        Self {
            start: Instant::now(),
            metrics,
            operation_type: query_type.to_string(),
            labels: vec![complexity.to_string(), result_size.to_string()],
        }
    }

    /// Finish the timer and record the metric
    pub fn finish_memory_operation(self, status: &str, size_bytes: Option<u64>) {
        let duration = self.start.elapsed();
        self.metrics.record_memory_operation(
            &self.operation_type,
            &self.labels[0],
            duration,
            status,
            size_bytes,
        );
    }

    /// Finish the timer and record the query metric
    pub fn finish_query_operation(self, status: &str, optimization_level: &str) {
        let duration = self.start.elapsed();
        self.metrics.record_query_operation(
            &self.operation_type,
            &self.labels[0],
            &self.labels[1],
            duration,
            status,
            optimization_level,
        );
    }
}
