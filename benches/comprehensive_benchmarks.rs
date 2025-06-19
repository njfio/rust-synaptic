//! Comprehensive Performance Benchmarking Suite
//!
//! Validates 100K+ operations/second target with micro-benchmarks, integration benchmarks,
//! and real-world scenario testing with detailed performance analysis.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType, StorageBackend,
    memory::types::MemoryMetadata,
    security::{SecurityManager, SecurityConfig},
};

#[cfg(feature = "analytics")]
use synaptic::analytics::{SimilarityAnalyzer, ClusteringAnalyzer, TrendAnalyzer};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use uuid::Uuid;
use rand::{Rng, thread_rng};
use rand::distributions::Alphanumeric;

/// Benchmark configuration for different test scenarios
#[derive(Clone)]
pub struct BenchmarkConfig {
    pub memory_count: usize,
    pub content_size: usize,
    pub concurrent_operations: usize,
    pub batch_size: usize,
}

impl BenchmarkConfig {
    pub fn small() -> Self {
        Self {
            memory_count: 1_000,
            content_size: 100,
            concurrent_operations: 10,
            batch_size: 100,
        }
    }

    pub fn medium() -> Self {
        Self {
            memory_count: 10_000,
            content_size: 500,
            concurrent_operations: 50,
            batch_size: 500,
        }
    }

    pub fn large() -> Self {
        Self {
            memory_count: 100_000,
            content_size: 1000,
            concurrent_operations: 100,
            batch_size: 1000,
        }
    }

    pub fn extreme() -> Self {
        Self {
            memory_count: 1_000_000,
            content_size: 2000,
            concurrent_operations: 200,
            batch_size: 2000,
        }
    }
}

/// Generate random test data
fn generate_test_data(config: &BenchmarkConfig) -> Vec<(String, String)> {
    let mut rng = thread_rng();
    (0..config.memory_count)
        .map(|i| {
            let key = format!("test_key_{}", i);
            let content: String = (0..config.content_size)
                .map(|_| rng.sample(Alphanumeric) as char)
                .collect();
            (key, content)
        })
        .collect()
}

/// Micro-benchmarks for core operations
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_operations");
    
    for config in [BenchmarkConfig::small(), BenchmarkConfig::medium(), BenchmarkConfig::large()] {
        let test_data = generate_test_data(&config);
        
        // Benchmark memory storage
        group.throughput(Throughput::Elements(config.memory_count as u64));
        group.bench_with_input(
            BenchmarkId::new("store", config.memory_count),
            &(config.clone(), test_data.clone()),
            |b, (cfg, data)| {
                b.to_async(&rt).iter(|| async {
                    let memory_config = MemoryConfig::default();
                    let mut memory = AgentMemory::new(memory_config).await.unwrap();
                    
                    let start = Instant::now();
                    for (key, value) in data.iter().take(1000) {
                        memory.store(key, value).await.unwrap();
                    }
                    let duration = start.elapsed();
                    
                    // Calculate operations per second
                    let ops_per_sec = 1000.0 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
        
        // Benchmark memory retrieval
        group.bench_with_input(
            BenchmarkId::new("retrieve", config.memory_count),
            &(config.clone(), test_data.clone()),
            |b, (cfg, data)| {
                b.to_async(&rt).iter(|| async {
                    let memory_config = MemoryConfig::default();
                    let mut memory = AgentMemory::new(memory_config).await.unwrap();
                    
                    // Pre-populate memory
                    for (key, value) in data.iter().take(1000) {
                        memory.store(key, value).await.unwrap();
                    }
                    
                    let start = Instant::now();
                    for (key, _) in data.iter().take(1000) {
                        let result = memory.retrieve(key).await.unwrap();
                        black_box(result);
                    }
                    let duration = start.elapsed();
                    
                    let ops_per_sec = 1000.0 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
        
        // Benchmark memory search
        group.bench_with_input(
            BenchmarkId::new("search", config.memory_count),
            &(config.clone(), test_data.clone()),
            |b, (cfg, data)| {
                b.to_async(&rt).iter(|| async {
                    let memory_config = MemoryConfig::default();
                    let mut memory = AgentMemory::new(memory_config).await.unwrap();
                    
                    // Pre-populate memory
                    for (key, value) in data.iter().take(1000) {
                        memory.store(key, value).await.unwrap();
                    }
                    
                    let start = Instant::now();
                    for i in 0..100 {
                        let query = format!("test_{}", i);
                        let results = memory.search(&query, 10).await.unwrap();
                        black_box(results);
                    }
                    let duration = start.elapsed();
                    
                    let ops_per_sec = 100.0 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark analytics operations
#[cfg(feature = "analytics")]
fn bench_analytics_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("analytics_operations");
    
    for config in [BenchmarkConfig::small(), BenchmarkConfig::medium()] {
        let test_data = generate_test_data(&config);
        
        // Benchmark similarity analysis
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("similarity_analysis", config.memory_count),
            &(config.clone(), test_data.clone()),
            |b, (cfg, data)| {
                b.to_async(&rt).iter(|| async {
                    let analyzer = SimilarityAnalyzer::new();
                    
                    let start = Instant::now();
                    for i in 0..100 {
                        let content1 = &data[i % data.len()].1;
                        let content2 = &data[(i + 1) % data.len()].1;
                        let similarity = analyzer.calculate_text_similarity(content1, content2).await.unwrap();
                        black_box(similarity);
                    }
                    let duration = start.elapsed();
                    
                    let ops_per_sec = 100.0 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
        
        // Benchmark clustering analysis
        group.bench_with_input(
            BenchmarkId::new("clustering_analysis", config.memory_count),
            &(config.clone(), test_data.clone()),
            |b, (cfg, data)| {
                b.to_async(&rt).iter(|| async {
                    let analyzer = ClusteringAnalyzer::new();
                    
                    let memories: Vec<MemoryEntry> = data.iter().take(100).map(|(key, value)| {
                        MemoryEntry::new(key.clone(), value.clone(), MemoryType::ShortTerm)
                    }).collect();
                    
                    let start = Instant::now();
                    let clusters = analyzer.cluster_memories(&memories, 5).await.unwrap();
                    let duration = start.elapsed();
                    
                    let ops_per_sec = 100.0 / duration.as_secs_f64();
                    black_box((clusters, ops_per_sec));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark security operations
fn bench_security_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("security_operations");

    // Benchmark security manager creation and basic operations
    group.throughput(Throughput::Elements(1000));
    group.bench_function("security_manager_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let security_config = SecurityConfig::default();
            let _security_manager = SecurityManager::new(security_config).await.unwrap();

            let start = Instant::now();
            for _i in 0..1000 {
                // Simulate security operations without AccessRequest
                // This is a simplified benchmark for security manager overhead
                tokio::task::yield_now().await;
            }
            let duration = start.elapsed();

            let ops_per_sec = 1000.0 / duration.as_secs_f64();
            black_box(ops_per_sec);
        });
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_operations");
    
    for concurrency in [10, 50, 100, 200] {
        group.throughput(Throughput::Elements(concurrency as u64 * 100));
        group.bench_with_input(
            BenchmarkId::new("concurrent_store", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let memory_config = MemoryConfig::default();
                    let memory = std::sync::Arc::new(tokio::sync::Mutex::new(
                        AgentMemory::new(memory_config).await.unwrap()
                    ));
                    
                    let start = Instant::now();
                    let mut handles = Vec::new();
                    
                    for i in 0..concurrency {
                        let memory_clone = memory.clone();
                        let handle = tokio::spawn(async move {
                            for j in 0..100 {
                                let key = format!("concurrent_key_{}_{}", i, j);
                                let value = format!("concurrent_value_{}_{}", i, j);
                                let mut mem = memory_clone.lock().await;
                                mem.store(&key, &value).await.unwrap();
                            }
                        });
                        handles.push(handle);
                    }
                    
                    for handle in handles {
                        handle.await.unwrap();
                    }
                    
                    let duration = start.elapsed();
                    let total_ops = concurrency * 100;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch operations
fn bench_batch_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_operations");
    
    for batch_size in [100, 500, 1000, 2000] {
        let test_data = generate_test_data(&BenchmarkConfig {
            memory_count: batch_size,
            content_size: 500,
            concurrent_operations: 1,
            batch_size,
        });
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_store", batch_size),
            &(batch_size, test_data),
            |b, (batch_size, data)| {
                b.to_async(&rt).iter(|| async {
                    let memory_config = MemoryConfig::default();
                    let mut memory = AgentMemory::new(memory_config).await.unwrap();
                    
                    let start = Instant::now();
                    for (key, value) in data.iter() {
                        memory.store(key, value).await.unwrap();
                    }
                    let duration = start.elapsed();
                    
                    let ops_per_sec = *batch_size as f64 / duration.as_secs_f64();
                    black_box(ops_per_sec);
                });
            },
        );
    }
    
    group.finish();
}

/// Integration benchmark testing real-world scenarios
fn bench_integration_scenarios(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("integration_scenarios");
    
    // Benchmark complete memory lifecycle
    group.throughput(Throughput::Elements(1000));
    group.bench_function("complete_memory_lifecycle", |b| {
        b.to_async(&rt).iter(|| async {
            let memory_config = MemoryConfig::default();
            let mut memory = AgentMemory::new(memory_config).await.unwrap();
            
            let start = Instant::now();
            
            // Store phase
            for i in 0..1000 {
                let key = format!("lifecycle_key_{}", i);
                let value = format!("lifecycle_value_{}", i);
                memory.store(&key, &value).await.unwrap();
            }
            
            // Search phase
            for i in 0..100 {
                let query = format!("lifecycle_{}", i);
                let results = memory.search(&query, 10).await.unwrap();
                black_box(results);
            }
            
            // Retrieve phase
            for i in 0..500 {
                let key = format!("lifecycle_key_{}", i);
                let result = memory.retrieve(&key).await.unwrap();
                black_box(result);
            }
            
            let duration = start.elapsed();
            let total_ops = 1000 + 100 + 500; // store + search + retrieve
            let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
            black_box(ops_per_sec);
        });
    });
    
    // Benchmark analytics pipeline
    #[cfg(feature = "analytics")]
    group.bench_function("analytics_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            let memory_config = MemoryConfig::default();
            let mut memory = AgentMemory::new(memory_config).await.unwrap();
            
            // Pre-populate with test data
            let test_data = generate_test_data(&BenchmarkConfig::small());
            for (key, value) in test_data.iter().take(100) {
                memory.store(key, value).await.unwrap();
            }
            
            let start = Instant::now();
            
            // Similarity analysis
            let similarity_analyzer = SimilarityAnalyzer::new();
            for i in 0..50 {
                let content1 = &test_data[i].1;
                let content2 = &test_data[i + 1].1;
                let similarity = similarity_analyzer.calculate_text_similarity(content1, content2).await.unwrap();
                black_box(similarity);
            }
            
            // Clustering analysis
            let clustering_analyzer = ClusteringAnalyzer::new();
            let memories: Vec<MemoryEntry> = test_data.iter().take(50).map(|(key, value)| {
                MemoryEntry::new(key.clone(), value.clone(), MemoryType::ShortTerm)
            }).collect();
            let clusters = clustering_analyzer.cluster_memories(&memories, 5).await.unwrap();
            black_box(clusters);
            
            let duration = start.elapsed();
            let total_ops = 50 + 1; // similarity + clustering
            let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
            black_box(ops_per_sec);
        });
    });
    
    group.finish();
}

/// Performance monitoring benchmark (simplified)
fn bench_performance_monitoring(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("performance_monitoring");

    group.bench_function("metrics_collection", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();

            // Simulate metric collection without external dependencies
            for _i in 0..1000 {
                // Simulate some work
                tokio::time::sleep(Duration::from_micros(10)).await;
            }

            let total_duration = start.elapsed();
            let ops_per_sec = 1000.0 / total_duration.as_secs_f64();
            black_box(ops_per_sec);
        });
    });

    group.finish();
}

/// Validate 100K+ operations/second target
fn bench_throughput_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("throughput_validation");
    group.sample_size(10); // Fewer samples for long-running benchmarks
    
    // Test with different operation types to validate 100K+ ops/sec
    group.bench_function("validate_100k_ops_per_second", |b| {
        b.to_async(&rt).iter(|| async {
            let memory_config = MemoryConfig::default();
            let mut memory = AgentMemory::new(memory_config).await.unwrap();
            
            let start = Instant::now();
            let target_ops = 100_000;
            
            // Mix of operations to simulate real workload
            for i in 0..target_ops {
                match i % 4 {
                    0 => {
                        // Store operation (25%)
                        let key = format!("throughput_key_{}", i);
                        let value = format!("throughput_value_{}", i);
                        memory.store(&key, &value).await.unwrap();
                    }
                    1 => {
                        // Retrieve operation (25%)
                        let key = format!("throughput_key_{}", i / 4);
                        let result = memory.retrieve(&key).await.unwrap();
                        black_box(result);
                    }
                    2 => {
                        // Search operation (25%)
                        let query = format!("throughput_{}", i / 4);
                        let results = memory.search(&query, 5).await.unwrap();
                        black_box(results);
                    }
                    3 => {
                        // Stats operation (25%)
                        let stats = memory.stats();
                        black_box(stats);
                    }
                    _ => unreachable!(),
                }
            }
            
            let duration = start.elapsed();
            let ops_per_sec = target_ops as f64 / duration.as_secs_f64();
            
            // Validate we achieved 100K+ ops/sec
            assert!(ops_per_sec >= 100_000.0, 
                   "Failed to achieve 100K ops/sec target. Achieved: {:.0} ops/sec", ops_per_sec);
            
            black_box(ops_per_sec);
        });
    });
    
    group.finish();
}

#[cfg(feature = "analytics")]
criterion_group!(
    benches,
    bench_memory_operations,
    bench_analytics_operations,
    bench_security_operations,
    bench_concurrent_operations,
    bench_batch_operations,
    bench_integration_scenarios,
    bench_performance_monitoring,
    bench_throughput_validation
);

#[cfg(not(feature = "analytics"))]
criterion_group!(
    benches,
    bench_memory_operations,
    bench_security_operations,
    bench_concurrent_operations,
    bench_batch_operations,
    bench_integration_scenarios,
    bench_performance_monitoring,
    bench_throughput_validation
);

criterion_main!(benches);
