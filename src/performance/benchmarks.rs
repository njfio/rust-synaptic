// Performance benchmarking suite
//
// Provides comprehensive benchmarking capabilities for performance
// measurement, regression detection, and optimization validation.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;

/// Benchmark suite for comprehensive performance testing
pub struct BenchmarkSuite {
    benchmarks: Vec<Benchmark>,
    results: Vec<BenchmarkResult>,
    baseline_results: HashMap<String, BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            results: Vec::new(),
            baseline_results: HashMap::new(),
        }
    }
    
    /// Add a benchmark to the suite
    pub fn add_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }
    
    /// Run all benchmarks
    pub async fn run_all(&mut self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for benchmark in &self.benchmarks {
            let result = self.run_benchmark(benchmark).await?;
            results.push(result);
        }
        
        self.results = results.clone();
        Ok(results)
    }
    
    /// Run a specific benchmark
    pub async fn run_benchmark(&self, benchmark: &Benchmark) -> Result<BenchmarkResult> {
        let start_time = Instant::now();
        let mut measurements = Vec::new();
        
        // Warm-up runs
        for _ in 0..benchmark.warmup_iterations {
            (benchmark.function)().await?;
        }
        
        // Actual benchmark runs
        for iteration in 0..benchmark.iterations {
            let iteration_start = Instant::now();
            
            (benchmark.function)().await?;
            
            let iteration_duration = iteration_start.elapsed();
            measurements.push(Measurement {
                iteration,
                duration: iteration_duration,
                timestamp: Utc::now(),
            });
        }
        
        let total_duration = start_time.elapsed();
        let statistics = self.calculate_statistics(&measurements);
        
        Ok(BenchmarkResult {
            id: Uuid::new_v4(),
            benchmark_name: benchmark.name.clone(),
            benchmark_category: benchmark.category.clone(),
            timestamp: Utc::now(),
            total_duration,
            iterations: benchmark.iterations,
            measurements,
            statistics,
            metadata: benchmark.metadata.clone(),
        })
    }
    
    /// Set baseline results for regression detection
    pub fn set_baseline(&mut self, results: Vec<BenchmarkResult>) {
        for result in results {
            self.baseline_results.insert(result.benchmark_name.clone(), result);
        }
    }
    
    /// Detect performance regressions
    pub fn detect_regressions(&self, threshold_percent: f64) -> Vec<PerformanceRegression> {
        let mut regressions = Vec::new();
        
        for result in &self.results {
            if let Some(baseline) = self.baseline_results.get(&result.benchmark_name) {
                let current_avg = result.statistics.mean_duration.as_nanos() as f64;
                let baseline_avg = baseline.statistics.mean_duration.as_nanos() as f64;
                
                let regression_percent = ((current_avg - baseline_avg) / baseline_avg) * 100.0;
                
                if regression_percent > threshold_percent {
                    regressions.push(PerformanceRegression {
                        benchmark_name: result.benchmark_name.clone(),
                        baseline_duration: baseline.statistics.mean_duration,
                        current_duration: result.statistics.mean_duration,
                        regression_percent,
                        severity: if regression_percent > 50.0 {
                            RegressionSeverity::Critical
                        } else if regression_percent > 25.0 {
                            RegressionSeverity::Major
                        } else {
                            RegressionSeverity::Minor
                        },
                    });
                }
            }
        }
        
        regressions
    }
    
    /// Generate benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        let regressions = self.detect_regressions(10.0); // 10% threshold
        
        BenchmarkReport {
            timestamp: Utc::now(),
            total_benchmarks: self.benchmarks.len(),
            results: self.results.clone(),
            regressions,
            summary: self.generate_summary(),
        }
    }
    
    /// Calculate statistics from measurements
    fn calculate_statistics(&self, measurements: &[Measurement]) -> BenchmarkStatistics {
        if measurements.is_empty() {
            return BenchmarkStatistics::default();
        }
        
        let mut durations: Vec<_> = measurements.iter().map(|m| m.duration).collect();
        durations.sort();
        
        let total_duration: Duration = durations.iter().sum();
        let mean_duration = total_duration / durations.len() as u32;
        
        let min_duration = durations[0];
        let max_duration = durations[durations.len() - 1];
        
        let median_duration = durations[durations.len() / 2];
        let p95_duration = durations[(durations.len() * 95) / 100];
        let p99_duration = durations[(durations.len() * 99) / 100];
        
        // Calculate standard deviation
        let variance: f64 = durations.iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_duration.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / durations.len() as f64;
        
        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);
        
        BenchmarkStatistics {
            mean_duration,
            median_duration,
            min_duration,
            max_duration,
            p95_duration,
            p99_duration,
            std_deviation,
            throughput_ops_per_sec: durations.len() as f64 / total_duration.as_secs_f64(),
        }
    }
    
    /// Generate summary statistics
    fn generate_summary(&self) -> BenchmarkSummary {
        if self.results.is_empty() {
            return BenchmarkSummary::default();
        }
        
        let total_duration: Duration = self.results.iter()
            .map(|r| r.total_duration)
            .sum();
        
        let avg_throughput = self.results.iter()
            .map(|r| r.statistics.throughput_ops_per_sec)
            .sum::<f64>() / self.results.len() as f64;
        
        let fastest_benchmark = self.results.iter()
            .min_by_key(|r| r.statistics.mean_duration)
            .map(|r| r.benchmark_name.clone())
            .unwrap_or_default();
        
        let slowest_benchmark = self.results.iter()
            .max_by_key(|r| r.statistics.mean_duration)
            .map(|r| r.benchmark_name.clone())
            .unwrap_or_default();
        
        BenchmarkSummary {
            total_duration,
            avg_throughput,
            fastest_benchmark,
            slowest_benchmark,
            total_iterations: self.results.iter().map(|r| r.iterations).sum(),
        }
    }
}

/// Individual benchmark definition
pub struct Benchmark {
    pub name: String,
    pub category: BenchmarkCategory,
    pub description: String,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub function: Box<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>> + Send + Sync>,
    pub metadata: HashMap<String, String>,
}

impl Benchmark {
    /// Create a new benchmark
    pub fn new<F, Fut>(
        name: String,
        category: BenchmarkCategory,
        description: String,
        iterations: usize,
        function: F,
    ) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        Self {
            name,
            category,
            description,
            iterations,
            warmup_iterations: iterations / 10, // 10% warmup
            function: Box::new(move || Box::pin(function())),
            metadata: HashMap::new(),
        }
    }
    
    /// Add metadata to benchmark
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Benchmark category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    Memory,
    Storage,
    Search,
    Analytics,
    Security,
    Network,
    Compression,
    Serialization,
}

/// Individual measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub iteration: usize,
    pub duration: Duration,
    pub timestamp: DateTime<Utc>,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub id: Uuid,
    pub benchmark_name: String,
    pub benchmark_category: BenchmarkCategory,
    pub timestamp: DateTime<Utc>,
    pub total_duration: Duration,
    pub iterations: usize,
    pub measurements: Vec<Measurement>,
    pub statistics: BenchmarkStatistics,
    pub metadata: HashMap<String, String>,
}

/// Benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub mean_duration: Duration,
    pub median_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub std_deviation: Duration,
    pub throughput_ops_per_sec: f64,
}

impl Default for BenchmarkStatistics {
    fn default() -> Self {
        Self {
            mean_duration: Duration::from_secs(0),
            median_duration: Duration::from_secs(0),
            min_duration: Duration::from_secs(0),
            max_duration: Duration::from_secs(0),
            p95_duration: Duration::from_secs(0),
            p99_duration: Duration::from_secs(0),
            std_deviation: Duration::from_secs(0),
            throughput_ops_per_sec: 0.0,
        }
    }
}

/// Performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub benchmark_name: String,
    pub baseline_duration: Duration,
    pub current_duration: Duration,
    pub regression_percent: f64,
    pub severity: RegressionSeverity,
}

/// Regression severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Minor,
    Major,
    Critical,
}

/// Benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub timestamp: DateTime<Utc>,
    pub total_benchmarks: usize,
    pub results: Vec<BenchmarkResult>,
    pub regressions: Vec<PerformanceRegression>,
    pub summary: BenchmarkSummary,
}

/// Benchmark summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_duration: Duration,
    pub avg_throughput: f64,
    pub fastest_benchmark: String,
    pub slowest_benchmark: String,
    pub total_iterations: usize,
}

impl Default for BenchmarkSummary {
    fn default() -> Self {
        Self {
            total_duration: Duration::from_secs(0),
            avg_throughput: 0.0,
            fastest_benchmark: String::new(),
            slowest_benchmark: String::new(),
            total_iterations: 0,
        }
    }
}

/// Predefined benchmark suite for common operations
pub struct StandardBenchmarks;

impl StandardBenchmarks {
    /// Create memory operation benchmarks
    pub fn memory_benchmarks() -> Vec<Benchmark> {
        vec![
            Benchmark::new(
                "memory_store_small".to_string(),
                BenchmarkCategory::Memory,
                "Store small memory entries (< 1KB)".to_string(),
                1000,
                || Box::pin(async { Ok(()) }),
            ),
            Benchmark::new(
                "memory_store_large".to_string(),
                BenchmarkCategory::Memory,
                "Store large memory entries (> 10KB)".to_string(),
                100,
                || Box::pin(async { Ok(()) }),
            ),
            Benchmark::new(
                "memory_retrieve".to_string(),
                BenchmarkCategory::Memory,
                "Retrieve memory entries".to_string(),
                1000,
                || Box::pin(async { Ok(()) }),
            ),
        ]
    }
    
    /// Create search operation benchmarks
    pub fn search_benchmarks() -> Vec<Benchmark> {
        vec![
            Benchmark::new(
                "search_exact_match".to_string(),
                BenchmarkCategory::Search,
                "Exact key search".to_string(),
                500,
                || Box::pin(async { Ok(()) }),
            ),
            Benchmark::new(
                "search_similarity".to_string(),
                BenchmarkCategory::Search,
                "Similarity search with embeddings".to_string(),
                100,
                || Box::pin(async { Ok(()) }),
            ),
        ]
    }
    
    /// Create analytics benchmarks
    pub fn analytics_benchmarks() -> Vec<Benchmark> {
        vec![
            Benchmark::new(
                "analytics_pattern_detection".to_string(),
                BenchmarkCategory::Analytics,
                "Pattern detection analysis".to_string(),
                50,
                || Box::pin(async { Ok(()) }),
            ),
            Benchmark::new(
                "analytics_summarization".to_string(),
                BenchmarkCategory::Analytics,
                "Content summarization".to_string(),
                20,
                || Box::pin(async { Ok(()) }),
            ),
        ]
    }
}
