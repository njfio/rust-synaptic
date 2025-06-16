// Performance optimization module
//
// This module provides comprehensive performance optimization capabilities
// including advanced profiling, benchmarking, and optimization strategies.

pub mod profiler;
pub mod benchmarks;
pub mod optimizer;
pub mod metrics;
pub mod cache;
pub mod memory_pool;
pub mod async_executor;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable advanced profiling
    pub enable_profiling: bool,
    
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    
    /// Optimization interval in seconds
    pub optimization_interval_seconds: u64,
    
    /// Performance target thresholds
    pub target_latency_ms: f64,
    pub target_throughput_ops_per_sec: f64,
    pub target_memory_usage_mb: f64,
    pub target_cpu_usage_percent: f64,
    
    /// Cache configuration
    pub cache_size_mb: usize,
    pub cache_ttl_seconds: u64,
    
    /// Memory pool configuration
    pub memory_pool_size_mb: usize,
    pub memory_pool_chunk_size_kb: usize,
    
    /// Async executor configuration
    pub worker_threads: usize,
    pub max_blocking_threads: usize,
    pub thread_keep_alive_seconds: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            enable_auto_optimization: true,
            optimization_interval_seconds: 300, // 5 minutes
            target_latency_ms: 10.0,
            target_throughput_ops_per_sec: 1000.0,
            target_memory_usage_mb: 512.0,
            target_cpu_usage_percent: 70.0,
            cache_size_mb: 128,
            cache_ttl_seconds: 3600, // 1 hour
            memory_pool_size_mb: 64,
            memory_pool_chunk_size_kb: 4,
            worker_threads: 4, // Default to 4 threads
            max_blocking_threads: 512,
            thread_keep_alive_seconds: 60,
        }
    }
}

/// Performance optimization manager
#[derive(Debug)]
pub struct PerformanceManager {
    config: PerformanceConfig,
    profiler: Arc<RwLock<profiler::AdvancedProfiler>>,
    optimizer: Arc<RwLock<optimizer::PerformanceOptimizer>>,
    metrics: Arc<RwLock<metrics::MetricsCollector>>,
    cache: Arc<RwLock<cache::PerformanceCache>>,
    memory_pool: Arc<RwLock<memory_pool::MemoryPool>>,
    executor: Arc<async_executor::AsyncExecutor>,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
}

impl PerformanceManager {
    /// Create a new performance manager
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        let profiler = Arc::new(RwLock::new(
            profiler::AdvancedProfiler::new(config.clone()).await?
        ));
        
        let optimizer = Arc::new(RwLock::new(
            optimizer::PerformanceOptimizer::new(config.clone()).await?
        ));
        
        let metrics = Arc::new(RwLock::new(
            metrics::MetricsCollector::new(config.clone()).await?
        ));
        
        let cache = Arc::new(RwLock::new(
            cache::PerformanceCache::new(config.clone()).await?
        ));
        
        let memory_pool = Arc::new(RwLock::new(
            memory_pool::MemoryPool::new(config.clone()).await?
        ));
        
        let executor = Arc::new(
            async_executor::AsyncExecutor::new(config.clone()).await?
        );
        
        let optimization_history = Arc::new(RwLock::new(Vec::new()));
        
        Ok(Self {
            config,
            profiler,
            optimizer,
            metrics,
            cache,
            memory_pool,
            executor,
            optimization_history,
        })
    }
    
    /// Start performance monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        // Start profiler
        self.profiler.write().await.start().await?;
        
        // Start metrics collection
        self.metrics.write().await.start_collection().await?;
        
        // Start automatic optimization if enabled
        if self.config.enable_auto_optimization {
            self.start_auto_optimization().await?;
        }
        
        Ok(())
    }
    
    /// Stop performance monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        self.profiler.write().await.stop().await?;
        self.metrics.write().await.stop_collection().await?;
        Ok(())
    }
    
    /// Run performance optimization
    pub async fn optimize(&self) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        
        // Collect current metrics
        let current_metrics = self.metrics.read().await.get_current_metrics().await?;
        
        // Run optimization
        let mut optimizer = self.optimizer.write().await;
        let optimization_result = optimizer.optimize(&current_metrics).await?;
        
        // Apply optimizations
        self.apply_optimizations(&optimization_result).await?;
        
        let duration = start_time.elapsed();
        
        let result = OptimizationResult {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            duration,
            optimizations_applied: optimization_result.optimizations.len(),
            performance_improvement: optimization_result.expected_improvement,
            metrics_before: current_metrics.clone(),
            metrics_after: self.metrics.read().await.get_current_metrics().await?,
            details: optimization_result,
        };
        
        // Store in history
        self.optimization_history.write().await.push(result.clone());
        
        Ok(result)
    }
    
    /// Get performance report
    pub async fn get_performance_report(&self) -> Result<PerformanceReport> {
        let current_metrics = self.metrics.read().await.get_current_metrics().await?;
        let profiling_data = self.profiler.read().await.get_profiling_data().await?;
        let optimization_history = self.optimization_history.read().await.clone();
        
        Ok(PerformanceReport {
            timestamp: Utc::now(),
            current_metrics,
            profiling_data,
            optimization_history,
            recommendations: self.generate_recommendations().await?,
        })
    }
    
    /// Start automatic optimization
    async fn start_auto_optimization(&self) -> Result<()> {
        let interval = Duration::from_secs(self.config.optimization_interval_seconds);
        let manager = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = manager.optimize().await {
                    eprintln!("Auto-optimization failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Apply optimizations
    async fn apply_optimizations(&self, result: &optimizer::OptimizationPlan) -> Result<()> {
        for optimization in &result.optimizations {
            match optimization.optimization_type {
                optimizer::OptimizationType::CacheOptimization => {
                    self.cache.write().await.apply_optimization(&optimization.parameters).await?;
                }
                optimizer::OptimizationType::MemoryPoolOptimization => {
                    self.memory_pool.write().await.apply_optimization(&optimization.parameters).await?;
                }
                optimizer::OptimizationType::ExecutorOptimization => {
                    self.executor.apply_optimization(&optimization.parameters).await?;
                }
                _ => {
                    // Handle other optimization types
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate performance recommendations
    async fn generate_recommendations(&self) -> Result<Vec<PerformanceRecommendation>> {
        let current_metrics = self.metrics.read().await.get_current_metrics().await?;
        let mut recommendations = Vec::new();
        
        // Check latency
        if current_metrics.avg_latency_ms > self.config.target_latency_ms {
            recommendations.push(PerformanceRecommendation {
                id: Uuid::new_v4(),
                category: RecommendationCategory::Latency,
                priority: RecommendationPriority::High,
                title: "High Latency Detected".to_string(),
                description: format!(
                    "Current latency ({:.2}ms) exceeds target ({:.2}ms)",
                    current_metrics.avg_latency_ms,
                    self.config.target_latency_ms
                ),
                suggested_actions: vec![
                    "Optimize cache configuration".to_string(),
                    "Increase memory pool size".to_string(),
                    "Review database query performance".to_string(),
                ],
                expected_impact: 25.0,
            });
        }
        
        // Check throughput
        if current_metrics.throughput_ops_per_sec < self.config.target_throughput_ops_per_sec {
            recommendations.push(PerformanceRecommendation {
                id: Uuid::new_v4(),
                category: RecommendationCategory::Throughput,
                priority: RecommendationPriority::Medium,
                title: "Low Throughput Detected".to_string(),
                description: format!(
                    "Current throughput ({:.2} ops/sec) below target ({:.2} ops/sec)",
                    current_metrics.throughput_ops_per_sec,
                    self.config.target_throughput_ops_per_sec
                ),
                suggested_actions: vec![
                    "Increase worker thread count".to_string(),
                    "Optimize async executor configuration".to_string(),
                    "Review bottlenecks in processing pipeline".to_string(),
                ],
                expected_impact: 30.0,
            });
        }
        
        Ok(recommendations)
    }
}

impl Clone for PerformanceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            profiler: Arc::clone(&self.profiler),
            optimizer: Arc::clone(&self.optimizer),
            metrics: Arc::clone(&self.metrics),
            cache: Arc::clone(&self.cache),
            memory_pool: Arc::clone(&self.memory_pool),
            executor: Arc::clone(&self.executor),
            optimization_history: Arc::clone(&self.optimization_history),
        }
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub optimizations_applied: usize,
    pub performance_improvement: f64,
    pub metrics_before: metrics::PerformanceMetrics,
    pub metrics_after: metrics::PerformanceMetrics,
    pub details: optimizer::OptimizationPlan,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub current_metrics: metrics::PerformanceMetrics,
    pub profiling_data: profiler::ProfilingData,
    pub optimization_history: Vec<OptimizationResult>,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Performance recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub id: Uuid,
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub suggested_actions: Vec<String>,
    pub expected_impact: f64,
}

/// Recommendation category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Latency,
    Throughput,
    Memory,
    CPU,
    Cache,
    Database,
    Network,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}
