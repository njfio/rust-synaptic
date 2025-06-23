// Performance metrics collection and analysis
//
// Provides comprehensive metrics collection, aggregation, and analysis
// for performance monitoring and optimization.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::error::Result;
use super::PerformanceConfig;

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    
    // Latency metrics
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_latency_ms: f64,
    
    // Throughput metrics
    pub throughput_ops_per_sec: f64,
    pub requests_per_sec: f64,
    pub successful_ops_per_sec: f64,
    pub failed_ops_per_sec: f64,
    
    // Resource utilization
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_io_mbps: f64,
    
    // Application-specific metrics
    pub cache_hit_rate: f64,
    pub cache_miss_rate: f64,
    pub index_efficiency: f64,
    pub compression_ratio: f64,
    pub error_rate: f64,
    
    // Quality metrics
    pub availability_percent: f64,
    pub reliability_score: f64,
    pub performance_score: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            max_latency_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            requests_per_sec: 0.0,
            successful_ops_per_sec: 0.0,
            failed_ops_per_sec: 0.0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            memory_usage_percent: 0.0,
            disk_usage_percent: 0.0,
            network_io_mbps: 0.0,
            cache_hit_rate: 0.0,
            cache_miss_rate: 0.0,
            index_efficiency: 1.0,
            compression_ratio: 1.0,
            error_rate: 0.0,
            availability_percent: 100.0,
            reliability_score: 100.0,
            performance_score: 100.0,
        }
    }
}

/// Metrics collector with real-time aggregation
#[derive(Debug)]
pub struct MetricsCollector {
    config: PerformanceConfig,
    current_metrics: Arc<RwLock<PerformanceMetrics>>,
    metrics_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    operation_timings: Arc<RwLock<VecDeque<Duration>>>,
    operation_counters: Arc<RwLock<OperationCounters>>,
    resource_monitor: Arc<RwLock<ResourceMonitor>>,
    is_collecting: Arc<RwLock<bool>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        Ok(Self {
            config,
            current_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            operation_timings: Arc::new(RwLock::new(VecDeque::new())),
            operation_counters: Arc::new(RwLock::new(OperationCounters::new())),
            resource_monitor: Arc::new(RwLock::new(ResourceMonitor::new())),
            is_collecting: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Start metrics collection
    pub async fn start_collection(&self) -> Result<()> {
        let mut is_collecting = self.is_collecting.write().await;
        if *is_collecting {
            return Ok(());
        }
        
        *is_collecting = true;
        
        // Start background collection task
        self.start_background_collection().await?;
        
        Ok(())
    }
    
    /// Stop metrics collection
    pub async fn stop_collection(&self) -> Result<()> {
        let mut is_collecting = self.is_collecting.write().await;
        *is_collecting = false;
        Ok(())
    }
    
    /// Record operation timing
    pub async fn record_operation_timing(&self, duration: Duration) -> Result<()> {
        let mut timings = self.operation_timings.write().await;
        
        // Keep only last 1000 timings
        if timings.len() >= 1000 {
            timings.pop_front();
        }
        
        timings.push_back(duration);
        
        // Update counters
        let mut counters = self.operation_counters.write().await;
        counters.total_operations += 1;
        counters.total_duration += duration;
        
        Ok(())
    }
    
    /// Record operation result
    pub async fn record_operation_result(&self, success: bool) -> Result<()> {
        let mut counters = self.operation_counters.write().await;
        
        if success {
            counters.successful_operations += 1;
        } else {
            counters.failed_operations += 1;
        }
        
        Ok(())
    }
    
    /// Record cache hit/miss
    pub async fn record_cache_access(&self, hit: bool) -> Result<()> {
        let mut counters = self.operation_counters.write().await;
        
        if hit {
            counters.cache_hits += 1;
        } else {
            counters.cache_misses += 1;
        }
        
        Ok(())
    }
    
    /// Get current metrics
    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(self.current_metrics.read().await.clone())
    }
    
    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Result<Vec<PerformanceMetrics>> {
        let history = self.metrics_history.read().await;
        let metrics: Vec<_> = if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.iter().cloned().collect()
        };
        
        Ok(metrics)
    }
    
    /// Start background collection task
    async fn start_background_collection(&self) -> Result<()> {
        let collector = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                let is_collecting = *collector.is_collecting.read().await;
                if !is_collecting {
                    break;
                }
                
                if let Err(e) = collector.collect_metrics().await {
                    eprintln!("Error collecting metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Collect and update metrics
    async fn collect_metrics(&self) -> Result<()> {
        let mut metrics = PerformanceMetrics::default();
        metrics.timestamp = Utc::now();
        
        // Calculate latency metrics
        self.calculate_latency_metrics(&mut metrics).await?;
        
        // Calculate throughput metrics
        self.calculate_throughput_metrics(&mut metrics).await?;
        
        // Get resource utilization
        self.get_resource_utilization(&mut metrics).await?;
        
        // Calculate application-specific metrics
        self.calculate_application_metrics(&mut metrics).await?;
        
        // Calculate quality metrics
        self.calculate_quality_metrics(&mut metrics).await?;
        
        // Update current metrics
        *self.current_metrics.write().await = metrics.clone();
        
        // Add to history
        let mut history = self.metrics_history.write().await;
        if history.len() >= 1000 {
            history.pop_front();
        }
        history.push_back(metrics);
        
        Ok(())
    }
    
    /// Calculate latency metrics from operation timings
    async fn calculate_latency_metrics(&self, metrics: &mut PerformanceMetrics) -> Result<()> {
        let timings = self.operation_timings.read().await;
        
        if timings.is_empty() {
            return Ok(());
        }
        
        let mut sorted_timings: Vec<_> = timings.iter().map(|d| d.as_millis() as f64).collect();
        use crate::error_handling::SafeCompare;
        sorted_timings.sort_by(|a, b| a.safe_partial_cmp(b));
        
        let len = sorted_timings.len();
        
        // Average latency
        metrics.avg_latency_ms = sorted_timings.iter().sum::<f64>() / len as f64;
        
        // Percentiles
        metrics.p50_latency_ms = sorted_timings[len / 2];
        metrics.p95_latency_ms = sorted_timings[(len * 95) / 100];
        metrics.p99_latency_ms = sorted_timings[(len * 99) / 100];
        metrics.max_latency_ms = sorted_timings[len - 1];
        
        Ok(())
    }
    
    /// Calculate throughput metrics
    async fn calculate_throughput_metrics(&self, metrics: &mut PerformanceMetrics) -> Result<()> {
        let counters = self.operation_counters.read().await;
        
        // Calculate rates based on recent activity (last minute)
        let time_window = 60.0; // seconds
        
        metrics.throughput_ops_per_sec = counters.total_operations as f64 / time_window;
        metrics.successful_ops_per_sec = counters.successful_operations as f64 / time_window;
        metrics.failed_ops_per_sec = counters.failed_operations as f64 / time_window;
        metrics.requests_per_sec = metrics.throughput_ops_per_sec;
        
        Ok(())
    }
    
    /// Get resource utilization metrics
    async fn get_resource_utilization(&self, metrics: &mut PerformanceMetrics) -> Result<()> {
        let monitor = self.resource_monitor.read().await;
        
        // In a real implementation, these would collect actual system metrics
        metrics.cpu_usage_percent = monitor.get_cpu_usage().await;
        metrics.memory_usage_mb = monitor.get_memory_usage_mb().await;
        metrics.memory_usage_percent = monitor.get_memory_usage_percent().await;
        metrics.disk_usage_percent = monitor.get_disk_usage_percent().await;
        metrics.network_io_mbps = monitor.get_network_io_mbps().await;
        
        Ok(())
    }
    
    /// Calculate application-specific metrics
    async fn calculate_application_metrics(&self, metrics: &mut PerformanceMetrics) -> Result<()> {
        let counters = self.operation_counters.read().await;
        
        // Cache metrics
        let total_cache_accesses = counters.cache_hits + counters.cache_misses;
        if total_cache_accesses > 0 {
            metrics.cache_hit_rate = counters.cache_hits as f64 / total_cache_accesses as f64;
            metrics.cache_miss_rate = counters.cache_misses as f64 / total_cache_accesses as f64;
        }
        
        // Error rate
        let total_operations = counters.successful_operations + counters.failed_operations;
        if total_operations > 0 {
            metrics.error_rate = counters.failed_operations as f64 / total_operations as f64;
        }
        
        // Mock values for other metrics (would be calculated from actual data)
        metrics.index_efficiency = 0.95;
        metrics.compression_ratio = 0.7;
        
        Ok(())
    }
    
    /// Calculate quality metrics
    async fn calculate_quality_metrics(&self, metrics: &mut PerformanceMetrics) -> Result<()> {
        // Availability (based on error rate)
        metrics.availability_percent = (1.0 - metrics.error_rate) * 100.0;
        
        // Reliability score (composite metric)
        metrics.reliability_score = (metrics.availability_percent + 
            (1.0 - metrics.error_rate) * 100.0) / 2.0;
        
        // Performance score (composite metric)
        let latency_score = (self.config.target_latency_ms / metrics.avg_latency_ms.max(0.1)).min(1.0);
        let throughput_score = (metrics.throughput_ops_per_sec / self.config.target_throughput_ops_per_sec).min(1.0);
        metrics.performance_score = (latency_score + throughput_score) * 50.0;
        
        Ok(())
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            current_metrics: Arc::clone(&self.current_metrics),
            metrics_history: Arc::clone(&self.metrics_history),
            operation_timings: Arc::clone(&self.operation_timings),
            operation_counters: Arc::clone(&self.operation_counters),
            resource_monitor: Arc::clone(&self.resource_monitor),
            is_collecting: Arc::clone(&self.is_collecting),
        }
    }
}

/// Operation counters for metrics calculation
#[derive(Debug, Default)]
pub struct OperationCounters {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_duration: Duration,
}

impl OperationCounters {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Resource monitor for system metrics
#[derive(Debug)]
pub struct ResourceMonitor {
    // In a real implementation, this would contain system monitoring tools
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn get_cpu_usage(&self) -> f64 {
        // Mock implementation - would use actual system monitoring
        45.0
    }
    
    pub async fn get_memory_usage_mb(&self) -> f64 {
        // Mock implementation
        512.0
    }
    
    pub async fn get_memory_usage_percent(&self) -> f64 {
        // Mock implementation
        60.0
    }
    
    pub async fn get_disk_usage_percent(&self) -> f64 {
        // Mock implementation
        75.0
    }
    
    pub async fn get_network_io_mbps(&self) -> f64 {
        // Mock implementation
        10.5
    }
}
