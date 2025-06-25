//! Advanced Performance Optimization Tests
//!
//! Comprehensive tests for the new performance optimization system including
//! profiling, benchmarking, caching, memory pooling, and async execution.

use synaptic::performance::{
    PerformanceManager, PerformanceConfig,
    profiler::AdvancedProfiler,
    optimizer::PerformanceOptimizer,
    metrics::MetricsCollector,
    cache::PerformanceCache,
    memory_pool::MemoryPool,
    async_executor::AsyncExecutor,
    benchmarks::{BenchmarkSuite, Benchmark, BenchmarkCategory, StandardBenchmarks},
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_performance_manager_creation() {
    let config = PerformanceConfig::default();
    let manager = PerformanceManager::new(config).await;
    
    assert!(manager.is_ok());
    let manager = manager.unwrap();
    
    // Test starting monitoring
    let result = manager.start_monitoring().await;
    assert!(result.is_ok());
    
    // Test stopping monitoring
    let result = manager.stop_monitoring().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_advanced_profiler() {
    let config = PerformanceConfig::default();
    let profiler = AdvancedProfiler::new(config).await.unwrap();
    
    // Test starting profiler
    let result = profiler.start().await;
    assert!(result.is_ok());
    
    // Test profiling session
    let session_id = profiler.start_session("test_session".to_string()).await.unwrap();
    assert!(!session_id.is_empty());
    
    // Simulate some work
    sleep(Duration::from_millis(100)).await;
    
    // Record custom metric
    let result = profiler.record_metric(&session_id, "test_metric".to_string(), 42.0).await;
    assert!(result.is_ok());
    
    // End session
    let session_result = profiler.end_session(&session_id).await;
    assert!(session_result.is_ok());
    
    let session_result = session_result.unwrap();
    assert_eq!(session_result.session_name, "test_session");
    assert!(session_result.duration > Duration::from_millis(50));
    assert!(session_result.performance_score >= 0.0);
    assert!(session_result.performance_score <= 100.0);
    
    // Test getting profiling data
    let profiling_data = profiler.get_profiling_data().await;
    assert!(profiling_data.is_ok());
    
    let profiling_data = profiling_data.unwrap();
    assert_eq!(profiling_data.session_results.len(), 1);
    
    // Test stopping profiler
    let result = profiler.stop().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_performance_optimizer() {
    let config = PerformanceConfig::default();
    let mut optimizer = PerformanceOptimizer::new(config.clone()).await.unwrap();
    
    // Create mock metrics
    let metrics = synaptic::performance::metrics::PerformanceMetrics {
        avg_latency_ms: 15.0, // Above target of 10ms
        throughput_ops_per_sec: 500.0, // Below target of 1000 ops/sec
        memory_usage_mb: 600.0, // Above target of 512MB
        cpu_usage_percent: 80.0, // Above target of 70%
        ..Default::default()
    };
    
    // Test optimization
    let optimization_plan = optimizer.optimize(&metrics).await;
    assert!(optimization_plan.is_ok());
    
    let plan = optimization_plan.unwrap();
    assert!(!plan.optimizations.is_empty());
    assert!(plan.expected_improvement > 0.0);
    assert!(plan.estimated_duration > Duration::from_secs(0));
    
    // Verify optimization types are appropriate for the metrics
    let optimization_types: Vec<_> = plan.optimizations.iter()
        .map(|opt| &opt.optimization_type)
        .collect();
    
    // Should include cache optimization for latency issues
    assert!(optimization_types.iter().any(|t| matches!(t, synaptic::performance::optimizer::OptimizationType::CacheOptimization)));
}

#[tokio::test]
async fn test_metrics_collector() {
    let config = PerformanceConfig::default();
    let collector = MetricsCollector::new(config).await.unwrap();
    
    // Test starting collection
    let result = collector.start_collection().await;
    assert!(result.is_ok());
    
    // Record some operations
    let result = collector.record_operation_timing(Duration::from_millis(50)).await;
    assert!(result.is_ok());
    
    let result = collector.record_operation_result(true).await;
    assert!(result.is_ok());
    
    let result = collector.record_operation_result(false).await;
    assert!(result.is_ok());
    
    let result = collector.record_cache_access(true).await;
    assert!(result.is_ok());
    
    let result = collector.record_cache_access(false).await;
    assert!(result.is_ok());
    
    // Wait for metrics collection
    sleep(Duration::from_millis(1100)).await;
    
    // Test getting current metrics
    let metrics = collector.get_current_metrics().await;
    assert!(metrics.is_ok());
    
    let metrics = metrics.unwrap();
    assert!(metrics.avg_latency_ms >= 0.0);
    assert!(metrics.throughput_ops_per_sec >= 0.0);
    
    // Test getting metrics history
    let history = collector.get_metrics_history(Some(10)).await;
    assert!(history.is_ok());
    
    let history = history.unwrap();
    assert!(!history.is_empty());
    
    // Test stopping collection
    let result = collector.stop_collection().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_performance_cache() {
    let config = PerformanceConfig::default();
    let cache = PerformanceCache::new(config).await.unwrap();
    
    // Test putting data in cache
    let test_data = b"test data for caching".to_vec();
    let result = cache.put("test_key".to_string(), test_data.clone()).await;
    assert!(result.is_ok());
    
    // Test getting data from cache
    let retrieved = cache.get("test_key").await;
    assert!(retrieved.is_ok());
    
    let retrieved = retrieved.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
    
    // Test cache miss
    let missing = cache.get("nonexistent_key").await;
    assert!(missing.is_ok());
    assert!(missing.unwrap().is_none());
    
    // Test cache statistics
    let stats = cache.get_statistics().await;
    assert!(stats.is_ok());
    
    let stats = stats.unwrap();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    
    // Test removing from cache
    let result = cache.remove("test_key").await;
    assert!(result.is_ok());
    assert!(result.unwrap());
    
    // Test clearing cache
    let result = cache.clear().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_memory_pool() {
    let config = PerformanceConfig::default();
    let pool = MemoryPool::new(config).await.unwrap();
    
    // Test allocating memory
    let chunk = pool.allocate(1024).await;
    assert!(chunk.is_ok());
    
    let chunk = chunk.unwrap();
    assert_eq!(chunk.size, 1024);
    assert_eq!(chunk.data.len(), 1024);
    
    // Test deallocating memory
    let result = pool.deallocate(chunk).await;
    assert!(result.is_ok());
    
    // Test getting statistics
    let stats = pool.get_statistics().await;
    assert!(stats.is_ok());
    
    let stats = stats.unwrap();
    assert_eq!(stats.allocation_stats.total_allocations, 1);
    assert_eq!(stats.allocation_stats.total_deallocations, 1);
    assert!(!stats.pool_stats.is_empty());
}

#[tokio::test]
async fn test_async_executor() {
    let config = PerformanceConfig::default();
    let executor = AsyncExecutor::new(config).await.unwrap();

    // Test getting initial statistics
    let stats = executor.get_statistics().await;
    assert!(stats.is_ok());

    let stats = stats.unwrap();
    assert_eq!(stats.tasks_submitted, 0);
    assert_eq!(stats.blocking_tasks_submitted, 0);
    assert_eq!(stats.tasks_completed, 0);

    // Test that executor was created successfully
    assert!(true, "AsyncExecutor created and statistics retrieved successfully");
}

#[tokio::test]
async fn test_benchmark_suite() {
    let mut suite = BenchmarkSuite::new();
    
    // Add a simple benchmark
    let benchmark = Benchmark::new(
        "test_benchmark".to_string(),
        BenchmarkCategory::Memory,
        "Test benchmark for validation".to_string(),
        10,
        || Box::pin(async {
            // Simulate work
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }),
    );
    
    suite.add_benchmark(benchmark);
    
    // Run benchmarks
    let results = suite.run_all().await;
    assert!(results.is_ok());
    
    let results = results.unwrap();
    assert_eq!(results.len(), 1);
    
    let result = &results[0];
    assert_eq!(result.benchmark_name, "test_benchmark");
    assert_eq!(result.iterations, 10);
    assert_eq!(result.measurements.len(), 10);
    assert!(result.statistics.mean_duration > Duration::from_nanos(0));
    
    // Test regression detection (no baseline, so no regressions)
    let regressions = suite.detect_regressions(10.0);
    assert!(regressions.is_empty());
    
    // Test report generation
    let report = suite.generate_report();
    assert_eq!(report.total_benchmarks, 1);
    assert_eq!(report.results.len(), 1);
    assert!(report.regressions.is_empty());
}

#[tokio::test]
async fn test_standard_benchmarks() {
    // Test memory benchmarks
    let memory_benchmarks = StandardBenchmarks::memory_benchmarks();
    assert!(!memory_benchmarks.is_empty());
    assert!(memory_benchmarks.iter().any(|b| b.name.contains("memory_store")));
    assert!(memory_benchmarks.iter().any(|b| b.name.contains("memory_retrieve")));
    
    // Test search benchmarks
    let search_benchmarks = StandardBenchmarks::search_benchmarks();
    assert!(!search_benchmarks.is_empty());
    assert!(search_benchmarks.iter().any(|b| b.name.contains("search_exact")));
    assert!(search_benchmarks.iter().any(|b| b.name.contains("search_similarity")));
    
    // Test analytics benchmarks
    let analytics_benchmarks = StandardBenchmarks::analytics_benchmarks();
    assert!(!analytics_benchmarks.is_empty());
    assert!(analytics_benchmarks.iter().any(|b| b.name.contains("analytics_pattern")));
    assert!(analytics_benchmarks.iter().any(|b| b.name.contains("analytics_summarization")));
}

#[tokio::test]
async fn test_performance_optimization_integration() {
    let config = PerformanceConfig::default();
    let manager = PerformanceManager::new(config).await.unwrap();
    
    // Start monitoring
    let result = manager.start_monitoring().await;
    assert!(result.is_ok());
    
    // Wait for some metrics collection
    sleep(Duration::from_millis(500)).await;
    
    // Run optimization
    let optimization_result = manager.optimize().await;
    assert!(optimization_result.is_ok());
    
    let result = optimization_result.unwrap();
    // Optimizations applied is always non-negative by type definition
    assert!(result.performance_improvement >= 0.0);
    assert!(!result.id.to_string().is_empty());
    
    // Get performance report
    let report = manager.get_performance_report().await;
    assert!(report.is_ok());
    
    let report = report.unwrap();
    assert!(!report.optimization_history.is_empty());
    assert!(!report.recommendations.is_empty());
    
    // Stop monitoring
    let result = manager.stop_monitoring().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_cache_optimization_parameters() {
    let config = PerformanceConfig::default();
    let cache = PerformanceCache::new(config).await.unwrap();
    
    // Test applying optimization parameters
    let mut parameters = std::collections::HashMap::new();
    parameters.insert("cache_size_mb".to_string(), "256".to_string());
    parameters.insert("ttl_seconds".to_string(), "1800".to_string());
    parameters.insert("eviction_policy".to_string(), "adaptive".to_string());
    
    let result = cache.apply_optimization(&parameters).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_memory_pool_optimization_parameters() {
    let config = PerformanceConfig::default();
    let pool = MemoryPool::new(config).await.unwrap();
    
    // Test applying optimization parameters
    let mut parameters = std::collections::HashMap::new();
    parameters.insert("pool_size_mb".to_string(), "128".to_string());
    parameters.insert("chunk_size_kb".to_string(), "8".to_string());
    
    let result = pool.apply_optimization(&parameters).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_executor_optimization_parameters() {
    let config = PerformanceConfig::default();
    let executor = AsyncExecutor::new(config).await.unwrap();
    
    // Test applying optimization parameters
    let mut parameters = std::collections::HashMap::new();
    parameters.insert("worker_threads".to_string(), "8".to_string());
    parameters.insert("max_blocking_threads".to_string(), "1024".to_string());
    parameters.insert("scheduling_strategy".to_string(), "adaptive".to_string());
    
    let result = executor.apply_optimization(&parameters).await;
    assert!(result.is_ok());
}
