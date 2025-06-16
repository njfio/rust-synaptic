// Performance optimizer
//
// Provides intelligent performance optimization strategies based on
// real-time metrics and machine learning algorithms.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use super::{PerformanceConfig, metrics::PerformanceMetrics};

/// Performance optimizer with intelligent optimization strategies
#[derive(Debug)]
pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    optimization_strategies: Arc<RwLock<Vec<OptimizationStrategy>>>,
    optimization_history: Arc<RwLock<Vec<OptimizationPlan>>>,
    ml_predictor: Arc<RwLock<MLPredictor>>,
    adaptive_tuner: Arc<RwLock<AdaptiveTuner>>,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        let strategies = vec![
            OptimizationStrategy::new(
                OptimizationType::CacheOptimization,
                "Intelligent Cache Optimization",
                0.8,
            ),
            OptimizationStrategy::new(
                OptimizationType::MemoryPoolOptimization,
                "Memory Pool Optimization",
                0.7,
            ),
            OptimizationStrategy::new(
                OptimizationType::ExecutorOptimization,
                "Async Executor Optimization",
                0.6,
            ),
            OptimizationStrategy::new(
                OptimizationType::IndexOptimization,
                "Index Structure Optimization",
                0.9,
            ),
            OptimizationStrategy::new(
                OptimizationType::CompressionOptimization,
                "Data Compression Optimization",
                0.5,
            ),
        ];
        
        Ok(Self {
            config,
            optimization_strategies: Arc::new(RwLock::new(strategies)),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            ml_predictor: Arc::new(RwLock::new(MLPredictor::new())),
            adaptive_tuner: Arc::new(RwLock::new(AdaptiveTuner::new())),
        })
    }
    
    /// Optimize performance based on current metrics
    pub async fn optimize(&mut self, metrics: &PerformanceMetrics) -> Result<OptimizationPlan> {
        // Analyze current performance
        let analysis = self.analyze_performance(metrics).await?;
        
        // Generate optimization plan
        let plan = self.generate_optimization_plan(&analysis).await?;
        
        // Apply machine learning predictions
        self.apply_ml_predictions(&plan).await?;
        
        // Store in history
        self.optimization_history.write().await.push(plan.clone());
        
        Ok(plan)
    }
    
    /// Analyze current performance metrics
    async fn analyze_performance(&self, metrics: &PerformanceMetrics) -> Result<PerformanceAnalysis> {
        let mut bottlenecks = Vec::new();
        let mut opportunities = Vec::new();
        
        // Analyze latency
        if metrics.avg_latency_ms > self.config.target_latency_ms {
            bottlenecks.push(PerformanceBottleneck {
                component: "Latency".to_string(),
                severity: if metrics.avg_latency_ms > self.config.target_latency_ms * 2.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
                impact: (metrics.avg_latency_ms - self.config.target_latency_ms) / self.config.target_latency_ms,
                description: format!("Latency {:.2}ms exceeds target {:.2}ms", 
                    metrics.avg_latency_ms, self.config.target_latency_ms),
            });
            
            opportunities.push(OptimizationOpportunity {
                optimization_type: OptimizationType::CacheOptimization,
                potential_improvement: 0.3,
                confidence: 0.8,
                description: "Cache optimization can reduce latency".to_string(),
            });
        }
        
        // Analyze throughput
        if metrics.throughput_ops_per_sec < self.config.target_throughput_ops_per_sec {
            bottlenecks.push(PerformanceBottleneck {
                component: "Throughput".to_string(),
                severity: BottleneckSeverity::Medium,
                impact: (self.config.target_throughput_ops_per_sec - metrics.throughput_ops_per_sec) 
                    / self.config.target_throughput_ops_per_sec,
                description: format!("Throughput {:.2} ops/sec below target {:.2} ops/sec",
                    metrics.throughput_ops_per_sec, self.config.target_throughput_ops_per_sec),
            });
            
            opportunities.push(OptimizationOpportunity {
                optimization_type: OptimizationType::ExecutorOptimization,
                potential_improvement: 0.4,
                confidence: 0.7,
                description: "Executor optimization can improve throughput".to_string(),
            });
        }
        
        // Analyze memory usage
        if metrics.memory_usage_mb > self.config.target_memory_usage_mb {
            bottlenecks.push(PerformanceBottleneck {
                component: "Memory".to_string(),
                severity: BottleneckSeverity::Medium,
                impact: (metrics.memory_usage_mb - self.config.target_memory_usage_mb) 
                    / self.config.target_memory_usage_mb,
                description: format!("Memory usage {:.2}MB exceeds target {:.2}MB",
                    metrics.memory_usage_mb, self.config.target_memory_usage_mb),
            });
            
            opportunities.push(OptimizationOpportunity {
                optimization_type: OptimizationType::CompressionOptimization,
                potential_improvement: 0.25,
                confidence: 0.6,
                description: "Compression can reduce memory usage".to_string(),
            });
        }
        
        Ok(PerformanceAnalysis {
            timestamp: Utc::now(),
            metrics: metrics.clone(),
            bottlenecks,
            opportunities,
            overall_score: self.calculate_performance_score(metrics).await?,
        })
    }
    
    /// Generate optimization plan
    async fn generate_optimization_plan(&self, analysis: &PerformanceAnalysis) -> Result<OptimizationPlan> {
        let mut optimizations = Vec::new();
        
        // Sort opportunities by potential improvement
        let mut sorted_opportunities = analysis.opportunities.clone();
        sorted_opportunities.sort_by(|a, b| 
            b.potential_improvement.partial_cmp(&a.potential_improvement).unwrap()
        );
        
        // Generate optimizations for top opportunities
        for opportunity in sorted_opportunities.iter().take(3) {
            let optimization = self.create_optimization(opportunity).await?;
            optimizations.push(optimization);
        }
        
        // Calculate expected improvement
        let expected_improvement = optimizations.iter()
            .map(|opt| opt.expected_improvement)
            .sum::<f64>() / optimizations.len() as f64;
        
        Ok(OptimizationPlan {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            analysis: analysis.clone(),
            optimizations,
            expected_improvement,
            estimated_duration: Duration::from_secs(30), // Estimated optimization time
        })
    }
    
    /// Create optimization from opportunity
    async fn create_optimization(&self, opportunity: &OptimizationOpportunity) -> Result<Optimization> {
        let parameters = match opportunity.optimization_type {
            OptimizationType::CacheOptimization => {
                let mut params = HashMap::new();
                params.insert("cache_size_mb".to_string(), (self.config.cache_size_mb * 2).to_string());
                params.insert("ttl_seconds".to_string(), (self.config.cache_ttl_seconds / 2).to_string());
                params
            }
            OptimizationType::MemoryPoolOptimization => {
                let mut params = HashMap::new();
                params.insert("pool_size_mb".to_string(), (self.config.memory_pool_size_mb * 2).to_string());
                params.insert("chunk_size_kb".to_string(), (self.config.memory_pool_chunk_size_kb * 2).to_string());
                params
            }
            OptimizationType::ExecutorOptimization => {
                let mut params = HashMap::new();
                params.insert("worker_threads".to_string(), (self.config.worker_threads + 2).to_string());
                params.insert("max_blocking_threads".to_string(), (self.config.max_blocking_threads + 100).to_string());
                params
            }
            OptimizationType::IndexOptimization => {
                let mut params = HashMap::new();
                params.insert("rebuild_indexes".to_string(), "true".to_string());
                params.insert("optimize_btree".to_string(), "true".to_string());
                params
            }
            OptimizationType::CompressionOptimization => {
                let mut params = HashMap::new();
                params.insert("compression_level".to_string(), "6".to_string());
                params.insert("compression_algorithm".to_string(), "zstd".to_string());
                params
            }
        };
        
        Ok(Optimization {
            id: Uuid::new_v4(),
            optimization_type: opportunity.optimization_type.clone(),
            description: opportunity.description.clone(),
            parameters,
            expected_improvement: opportunity.potential_improvement,
            confidence: opportunity.confidence,
            estimated_duration: Duration::from_secs(10),
        })
    }
    
    /// Apply machine learning predictions
    async fn apply_ml_predictions(&self, plan: &OptimizationPlan) -> Result<()> {
        let mut predictor = self.ml_predictor.write().await;
        predictor.train_on_plan(plan).await?;
        
        let mut tuner = self.adaptive_tuner.write().await;
        tuner.adjust_parameters(plan).await?;
        
        Ok(())
    }
    
    /// Calculate performance score
    async fn calculate_performance_score(&self, metrics: &PerformanceMetrics) -> Result<f64> {
        let latency_score = (self.config.target_latency_ms / metrics.avg_latency_ms.max(0.1)).min(1.0);
        let throughput_score = (metrics.throughput_ops_per_sec / self.config.target_throughput_ops_per_sec).min(1.0);
        let memory_score = (self.config.target_memory_usage_mb / metrics.memory_usage_mb.max(1.0)).min(1.0);
        let cpu_score = (self.config.target_cpu_usage_percent / metrics.cpu_usage_percent.max(1.0)).min(1.0);
        
        // Weighted average
        let score = (latency_score * 0.3 + throughput_score * 0.3 + memory_score * 0.2 + cpu_score * 0.2) * 100.0;
        Ok(score.max(0.0).min(100.0))
    }
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub optimization_type: OptimizationType,
    pub name: String,
    pub effectiveness: f64,
}

impl OptimizationStrategy {
    pub fn new(optimization_type: OptimizationType, name: &str, effectiveness: f64) -> Self {
        Self {
            optimization_type,
            name: name.to_string(),
            effectiveness,
        }
    }
}

/// Optimization type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    CacheOptimization,
    MemoryPoolOptimization,
    ExecutorOptimization,
    IndexOptimization,
    CompressionOptimization,
}

/// Performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub timestamp: DateTime<Utc>,
    pub metrics: PerformanceMetrics,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub opportunities: Vec<OptimizationOpportunity>,
    pub overall_score: f64,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub component: String,
    pub severity: BottleneckSeverity,
    pub impact: f64,
    pub description: String,
}

/// Bottleneck severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub optimization_type: OptimizationType,
    pub potential_improvement: f64,
    pub confidence: f64,
    pub description: String,
}

/// Optimization plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPlan {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub analysis: PerformanceAnalysis,
    pub optimizations: Vec<Optimization>,
    pub expected_improvement: f64,
    pub estimated_duration: Duration,
}

/// Individual optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optimization {
    pub id: Uuid,
    pub optimization_type: OptimizationType,
    pub description: String,
    pub parameters: HashMap<String, String>,
    pub expected_improvement: f64,
    pub confidence: f64,
    pub estimated_duration: Duration,
}

/// Machine learning predictor for optimization effectiveness
#[derive(Debug)]
pub struct MLPredictor {
    training_data: Vec<OptimizationPlan>,
}

impl MLPredictor {
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
        }
    }
    
    pub async fn train_on_plan(&mut self, plan: &OptimizationPlan) -> Result<()> {
        self.training_data.push(plan.clone());
        
        // Keep only last 100 plans for training
        if self.training_data.len() > 100 {
            self.training_data.remove(0);
        }
        
        Ok(())
    }
    
    pub async fn predict_effectiveness(&self, optimization_type: &OptimizationType) -> Result<f64> {
        // Simple prediction based on historical data
        let relevant_plans: Vec<_> = self.training_data.iter()
            .filter(|plan| plan.optimizations.iter()
                .any(|opt| std::mem::discriminant(&opt.optimization_type) == std::mem::discriminant(optimization_type)))
            .collect();
        
        if relevant_plans.is_empty() {
            return Ok(0.5); // Default prediction
        }
        
        let avg_improvement = relevant_plans.iter()
            .map(|plan| plan.expected_improvement)
            .sum::<f64>() / relevant_plans.len() as f64;
        
        Ok(avg_improvement)
    }
}

/// Adaptive tuner for optimization parameters
#[derive(Debug)]
pub struct AdaptiveTuner {
    parameter_history: HashMap<String, Vec<f64>>,
}

impl AdaptiveTuner {
    pub fn new() -> Self {
        Self {
            parameter_history: HashMap::new(),
        }
    }
    
    pub async fn adjust_parameters(&mut self, plan: &OptimizationPlan) -> Result<()> {
        for optimization in &plan.optimizations {
            for (param_name, param_value) in &optimization.parameters {
                if let Ok(value) = param_value.parse::<f64>() {
                    let history = self.parameter_history.entry(param_name.clone())
                        .or_insert_with(Vec::new);
                    
                    history.push(value);
                    
                    // Keep only last 50 values
                    if history.len() > 50 {
                        history.remove(0);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn get_optimal_parameter(&self, param_name: &str) -> Option<f64> {
        if let Some(history) = self.parameter_history.get(param_name) {
            if !history.is_empty() {
                // Return average of recent values
                let recent_values = &history[history.len().saturating_sub(10)..];
                let avg = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
                return Some(avg);
            }
        }
        
        None
    }
}
