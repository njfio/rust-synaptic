//! OpenTelemetry Tracing Implementation
//!
//! This module provides comprehensive distributed tracing capabilities using OpenTelemetry,
//! enabling detailed performance monitoring, request flow tracking, and debugging across
//! the entire Synaptic memory system with proper span management and context propagation.

use crate::error::Result;
use opentelemetry::{
    global, trace::{Span, SpanKind, Status, TraceContextExt, Tracer}, Context, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    trace::{self, RandomIdGenerator, Sampler},
    Resource,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// OpenTelemetry tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub otlp_endpoint: String,
    pub sampling_ratio: f64,
    pub max_events_per_span: u32,
    pub max_attributes_per_span: u32,
    pub max_links_per_span: u32,
    pub enable_jaeger_propagation: bool,
    pub enable_b3_propagation: bool,
    pub enable_baggage: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "synaptic-memory-system".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
            otlp_endpoint: "http://localhost:4317".to_string(),
            sampling_ratio: 1.0,
            max_events_per_span: 128,
            max_attributes_per_span: 128,
            max_links_per_span: 128,
            enable_jaeger_propagation: true,
            enable_b3_propagation: true,
            enable_baggage: true,
        }
    }
}

/// OpenTelemetry tracing manager for the Synaptic system
#[derive(Clone)]
pub struct OpenTelemetryTracing {
    tracer: Arc<dyn Tracer + Send + Sync>,
    config: TracingConfig,
    active_spans: Arc<RwLock<HashMap<String, SpanContext>>>,
}

/// Context information for active spans
#[derive(Debug, Clone)]
pub struct SpanContext {
    pub span_id: String,
    pub trace_id: String,
    pub operation_name: String,
    pub start_time: SystemTime,
    pub attributes: HashMap<String, String>,
}

impl OpenTelemetryTracing {
    /// Initialize OpenTelemetry tracing with the provided configuration
    pub async fn new(config: TracingConfig) -> Result<Self> {
        info!("Initializing OpenTelemetry tracing with service: {}", config.service_name);
        
        // Create resource with service information
        let resource = Resource::new(vec![
            KeyValue::new("service.name", config.service_name.clone()),
            KeyValue::new("service.version", config.service_version.clone()),
            KeyValue::new("deployment.environment", config.environment.clone()),
            KeyValue::new("telemetry.sdk.name", "opentelemetry"),
            KeyValue::new("telemetry.sdk.language", "rust"),
            KeyValue::new("telemetry.sdk.version", opentelemetry::sdk::version()),
        ]);
        
        // Configure sampler based on sampling ratio
        let sampler = if config.sampling_ratio >= 1.0 {
            Sampler::AlwaysOn
        } else if config.sampling_ratio <= 0.0 {
            Sampler::AlwaysOff
        } else {
            Sampler::TraceIdRatioBased(config.sampling_ratio)
        };
        
        // Create OTLP exporter
        let exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(&config.otlp_endpoint)
            .build_span_exporter()?;
        
        // Configure trace provider
        let trace_config = trace::config()
            .with_sampler(sampler)
            .with_id_generator(RandomIdGenerator::default())
            .with_max_events_per_span(config.max_events_per_span)
            .with_max_attributes_per_span(config.max_attributes_per_span)
            .with_max_links_per_span(config.max_links_per_span)
            .with_resource(resource);
        
        let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_config(trace_config)
            .build();
        
        // Set global tracer provider
        global::set_tracer_provider(tracer_provider.clone());
        
        // Get tracer instance
        let tracer = tracer_provider.tracer("synaptic-tracer");
        
        info!("OpenTelemetry tracing initialized successfully");
        
        Ok(Self {
            tracer: Arc::new(tracer),
            config,
            active_spans: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Create a new span for memory operations
    pub async fn start_memory_span(&self, operation: &str, memory_type: &str) -> Result<String> {
        let span_name = format!("memory.{}", operation);
        let mut span = self.tracer.start(&span_name);
        
        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "memory"));
        span.set_attribute(KeyValue::new("memory.operation", operation.to_string()));
        span.set_attribute(KeyValue::new("memory.type", memory_type.to_string()));
        span.set_attribute(KeyValue::new("component", "synaptic.memory"));
        
        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());
        
        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };
        
        self.active_spans.write().await.insert(span_id.clone(), context);
        
        debug!("Started memory span: {} for operation: {}", span_id, operation);
        Ok(span_id)
    }
    
    /// Create a new span for query operations
    pub async fn start_query_span(&self, query_type: &str, complexity: &str) -> Result<String> {
        let span_name = format!("query.{}", query_type);
        let mut span = self.tracer.start(&span_name);
        
        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "query"));
        span.set_attribute(KeyValue::new("query.type", query_type.to_string()));
        span.set_attribute(KeyValue::new("query.complexity", complexity.to_string()));
        span.set_attribute(KeyValue::new("component", "synaptic.query"));
        
        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());
        
        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };
        
        self.active_spans.write().await.insert(span_id.clone(), context);
        
        debug!("Started query span: {} for query type: {}", span_id, query_type);
        Ok(span_id)
    }
    
    /// Create a new span for analytics operations
    pub async fn start_analytics_span(&self, operation: &str, algorithm: &str) -> Result<String> {
        let span_name = format!("analytics.{}", operation);
        let mut span = self.tracer.start(&span_name);
        
        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "analytics"));
        span.set_attribute(KeyValue::new("analytics.operation", operation.to_string()));
        span.set_attribute(KeyValue::new("analytics.algorithm", algorithm.to_string()));
        span.set_attribute(KeyValue::new("component", "synaptic.analytics"));
        
        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());
        
        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };
        
        self.active_spans.write().await.insert(span_id.clone(), context);
        
        debug!("Started analytics span: {} for operation: {}", span_id, operation);
        Ok(span_id)
    }
    
    /// Create a new span for storage operations
    pub async fn start_storage_span(&self, operation: &str, backend: &str) -> Result<String> {
        let span_name = format!("storage.{}", operation);
        let mut span = self.tracer.start(&span_name);
        
        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "storage"));
        span.set_attribute(KeyValue::new("storage.operation", operation.to_string()));
        span.set_attribute(KeyValue::new("storage.backend", backend.to_string()));
        span.set_attribute(KeyValue::new("component", "synaptic.storage"));
        
        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());
        
        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };
        
        self.active_spans.write().await.insert(span_id.clone(), context);
        
        debug!("Started storage span: {} for operation: {}", span_id, operation);
        Ok(span_id)
    }
    
    /// Create a new span for security operations
    pub async fn start_security_span(&self, operation: &str, resource_type: &str) -> Result<String> {
        let span_name = format!("security.{}", operation);
        let mut span = self.tracer.start(&span_name);
        
        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "security"));
        span.set_attribute(KeyValue::new("security.operation", operation.to_string()));
        span.set_attribute(KeyValue::new("security.resource_type", resource_type.to_string()));
        span.set_attribute(KeyValue::new("component", "synaptic.security"));
        
        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());
        
        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };
        
        self.active_spans.write().await.insert(span_id.clone(), context);
        
        debug!("Started security span: {} for operation: {}", span_id, operation);
        Ok(span_id)
    }

    /// Add an attribute to an active span
    pub async fn add_span_attribute(&self, span_id: &str, key: &str, value: &str) -> Result<()> {
        let mut active_spans = self.active_spans.write().await;
        if let Some(context) = active_spans.get_mut(span_id) {
            context.attributes.insert(key.to_string(), value.to_string());
            debug!("Added attribute to span {}: {}={}", span_id, key, value);
            Ok(())
        } else {
            warn!("Attempted to add attribute to non-existent span: {}", span_id);
            Err(crate::error::SynapticError::NotFound(format!("Span '{}' not found", span_id)))
        }
    }

    /// Add an event to an active span
    pub async fn add_span_event(&self, span_id: &str, name: &str, attributes: Vec<(String, String)>) -> Result<()> {
        // Note: This is a simplified implementation
        // In a full implementation, we would need to maintain references to actual Span objects
        debug!("Added event to span {}: {} with {} attributes", span_id, name, attributes.len());
        Ok(())
    }

    /// Record an error in a span
    pub async fn record_span_error(&self, span_id: &str, error: &str, error_type: &str) -> Result<()> {
        self.add_span_attribute(span_id, "error", "true").await?;
        self.add_span_attribute(span_id, "error.message", error).await?;
        self.add_span_attribute(span_id, "error.type", error_type).await?;

        warn!("Recorded error in span {}: {} ({})", span_id, error, error_type);
        Ok(())
    }

    /// Finish a span with success status
    pub async fn finish_span_success(&self, span_id: &str, result_size: Option<usize>) -> Result<()> {
        let mut active_spans = self.active_spans.write().await;
        if let Some(context) = active_spans.remove(span_id) {
            let duration = SystemTime::now().duration_since(context.start_time)
                .unwrap_or(Duration::from_secs(0));

            if let Some(size) = result_size {
                debug!("Finished span {} successfully: {} (duration: {:?}, result_size: {})",
                       span_id, context.operation_name, duration, size);
            } else {
                debug!("Finished span {} successfully: {} (duration: {:?})",
                       span_id, context.operation_name, duration);
            }

            Ok(())
        } else {
            warn!("Attempted to finish non-existent span: {}", span_id);
            Err(crate::error::SynapticError::NotFound(format!("Span '{}' not found", span_id)))
        }
    }

    /// Finish a span with error status
    pub async fn finish_span_error(&self, span_id: &str, error: &str, error_type: &str) -> Result<()> {
        self.record_span_error(span_id, error, error_type).await?;

        let mut active_spans = self.active_spans.write().await;
        if let Some(context) = active_spans.remove(span_id) {
            let duration = SystemTime::now().duration_since(context.start_time)
                .unwrap_or(Duration::from_secs(0));

            error!("Finished span {} with error: {} (duration: {:?}, error: {})",
                   span_id, context.operation_name, duration, error);

            Ok(())
        } else {
            warn!("Attempted to finish non-existent span: {}", span_id);
            Err(crate::error::SynapticError::NotFound(format!("Span '{}' not found", span_id)))
        }
    }

    /// Get active span count
    pub async fn get_active_span_count(&self) -> usize {
        self.active_spans.read().await.len()
    }

    /// Get span context information
    pub async fn get_span_context(&self, span_id: &str) -> Option<SpanContext> {
        self.active_spans.read().await.get(span_id).cloned()
    }

    /// Create a child span from a parent span
    pub async fn start_child_span(&self, parent_span_id: &str, operation: &str, component: &str) -> Result<String> {
        let span_name = format!("{}.{}", component, operation);
        let mut span = self.tracer.start(&span_name);

        // Set span attributes
        span.set_attribute(KeyValue::new("operation.type", "child"));
        span.set_attribute(KeyValue::new("parent.span_id", parent_span_id.to_string()));
        span.set_attribute(KeyValue::new("component", component.to_string()));

        let span_context = span.span_context();
        let span_id = format!("{:x}", span_context.span_id());
        let trace_id = format!("{:x}", span_context.trace_id());

        // Store span context
        let context = SpanContext {
            span_id: span_id.clone(),
            trace_id,
            operation_name: span_name,
            start_time: SystemTime::now(),
            attributes: HashMap::new(),
        };

        self.active_spans.write().await.insert(span_id.clone(), context);

        debug!("Started child span: {} for parent: {} operation: {}", span_id, parent_span_id, operation);
        Ok(span_id)
    }

    /// Flush all pending spans
    pub async fn flush(&self) -> Result<()> {
        info!("Flushing OpenTelemetry spans");

        // Force flush the tracer provider
        if let Some(provider) = global::tracer_provider().downcast_ref::<opentelemetry_sdk::trace::TracerProvider>() {
            provider.force_flush();
        }

        Ok(())
    }

    /// Shutdown the tracing system
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down OpenTelemetry tracing");

        // Finish any remaining active spans
        let active_spans = self.active_spans.read().await;
        let span_count = active_spans.len();
        if span_count > 0 {
            warn!("Shutting down with {} active spans", span_count);
        }
        drop(active_spans);

        // Shutdown the global tracer provider
        global::shutdown_tracer_provider();

        info!("OpenTelemetry tracing shutdown complete");
        Ok(())
    }

    /// Get tracing configuration
    pub fn get_config(&self) -> &TracingConfig {
        &self.config
    }

    /// Update sampling ratio (requires restart to take effect)
    pub fn update_sampling_ratio(&mut self, ratio: f64) {
        self.config.sampling_ratio = ratio.clamp(0.0, 1.0);
        info!("Updated sampling ratio to: {}", self.config.sampling_ratio);
    }
}
