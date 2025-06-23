use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use synaptic::memory::management::MemoryManager;
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::runtime::Runtime;

/// Create test data for analytics benchmarks
fn create_analytics_test_data(count: usize) -> Vec<MemoryEntry> {
    let content_templates = vec![
        "User interaction with system component {}",
        "Error occurred in module {} during operation",
        "Performance metrics for process {} show improvement",
        "Configuration change applied to service {}",
        "Data processing completed for batch {}",
        "Security event detected in system {}",
        "Network communication established with endpoint {}",
        "Cache invalidation triggered for resource {}",
        "Database query executed for table {}",
        "API request processed for endpoint {}",
    ];
    
    (0..count)
        .map(|i| {
            let template = &content_templates[i % content_templates.len()];
            let content = template.replace("{}", &i.to_string());
            
            MemoryEntry::new(
                format!("analytics_entry_{}", i),
                content,
                if i % 3 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
            )
        })
        .collect()
}

/// Benchmark analytics calculations
fn bench_analytics_calculations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("analytics_calculations");
    
    for data_size in [100, 500, 1000, 2000].iter() {
        let entries = create_analytics_test_data(*data_size);
        
        group.throughput(Throughput::Elements(*data_size as u64));
        
        // Benchmark memory pattern analysis
        group.bench_with_input(
            BenchmarkId::new("memory_pattern_analysis", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let manager = MemoryManager::new(storage).await.unwrap();
                        
                        // Store test data
                        for entry in &entries {
                            manager.store_memory(
                                &entry.value,
                                entry.memory_type.clone(),
                                HashMap::new(),
                            ).await.unwrap();
                        }
                        
                        // Perform analytics
                        manager.analyze_memory_patterns().await.unwrap();
                    })
                })
            },
        );
        
        // Benchmark trend analysis
        group.bench_with_input(
            BenchmarkId::new("trend_analysis", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let manager = MemoryManager::new(storage).await.unwrap();
                        
                        // Store test data with timestamps
                        for (i, entry) in entries.iter().enumerate() {
                            let mut metadata = HashMap::new();
                            metadata.insert("timestamp".to_string(), (i as u64).to_string());
                            
                            manager.store_memory(
                                &entry.value,
                                entry.memory_type.clone(),
                                metadata,
                            ).await.unwrap();
                        }
                        
                        // Perform trend analysis
                        manager.analyze_trends().await.unwrap();
                    })
                })
            },
        );
        
        // Benchmark clustering analysis
        group.bench_with_input(
            BenchmarkId::new("clustering_analysis", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let manager = MemoryManager::new(storage).await.unwrap();
                        
                        // Store test data
                        for entry in &entries {
                            manager.store_memory(
                                &entry.value,
                                entry.memory_type.clone(),
                                HashMap::new(),
                            ).await.unwrap();
                        }
                        
                        // Perform clustering
                        manager.cluster_memories(black_box(5)).await.unwrap();
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark real-time analytics
fn bench_realtime_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("realtime_analytics");
    
    // Benchmark incremental analytics updates
    group.bench_function("incremental_analytics_update", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Initial data
                for i in 0..100 {
                    let content = format!("Initial data entry {}", i);
                    manager.store_memory(
                        &content,
                        MemoryType::ShortTerm,
                        HashMap::new(),
                    ).await.unwrap();
                }
                
                // Simulate real-time updates
                for i in 100..150 {
                    let content = format!("Real-time update entry {}", i);
                    manager.store_memory(
                        black_box(&content),
                        MemoryType::ShortTerm,
                        HashMap::new(),
                    ).await.unwrap();
                    
                    // Update analytics incrementally
                    manager.update_analytics_incremental().await.unwrap();
                }
            })
        })
    });
    
    // Benchmark streaming analytics
    group.bench_function("streaming_analytics", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Simulate streaming data processing
                for batch in 0..10 {
                    let batch_data: Vec<_> = (0..20)
                        .map(|i| format!("Streaming data batch {} item {}", batch, i))
                        .collect();
                    
                    for content in batch_data {
                        manager.store_memory(
                            black_box(&content),
                            MemoryType::ShortTerm,
                            HashMap::new(),
                        ).await.unwrap();
                    }
                    
                    // Process batch analytics
                    manager.process_batch_analytics().await.unwrap();
                }
            })
        })
    });
    
    group.finish();
}

/// Benchmark memory optimization analytics
fn bench_optimization_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("optimization_analytics");
    
    // Benchmark memory usage analysis
    group.bench_function("memory_usage_analysis", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Create varied memory usage patterns
                for i in 0..500 {
                    let size_factor = (i % 10) + 1;
                    let content = "x".repeat(size_factor * 100);
                    
                    manager.store_memory(
                        black_box(&content),
                        if i % 2 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
                        HashMap::new(),
                    ).await.unwrap();
                }
                
                // Analyze memory usage patterns
                manager.analyze_memory_usage().await.unwrap();
            })
        })
    });
    
    // Benchmark performance bottleneck detection
    group.bench_function("bottleneck_detection", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Create performance data
                for i in 0..200 {
                    let mut metadata = HashMap::new();
                    metadata.insert("operation_time".to_string(), (i % 100).to_string());
                    metadata.insert("memory_usage".to_string(), ((i * 1024) % 50000).to_string());
                    
                    let content = format!("Performance data point {}", i);
                    manager.store_memory(
                        &content,
                        MemoryType::ShortTerm,
                        metadata,
                    ).await.unwrap();
                }
                
                // Detect bottlenecks
                manager.detect_performance_bottlenecks().await.unwrap();
            })
        })
    });
    
    group.finish();
}

/// Benchmark predictive analytics
fn bench_predictive_analytics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("predictive_analytics");
    
    // Benchmark memory growth prediction
    group.bench_function("memory_growth_prediction", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Create historical data with growth pattern
                for day in 0..30 {
                    let daily_entries = 50 + (day * 2); // Growing pattern
                    
                    for entry in 0..daily_entries {
                        let mut metadata = HashMap::new();
                        metadata.insert("day".to_string(), day.to_string());
                        metadata.insert("entry_index".to_string(), entry.to_string());
                        
                        let content = format!("Day {} entry {}", day, entry);
                        manager.store_memory(
                            &content,
                            MemoryType::LongTerm,
                            metadata,
                        ).await.unwrap();
                    }
                }
                
                // Predict future memory growth
                manager.predict_memory_growth(black_box(7)).await.unwrap(); // 7 days ahead
            })
        })
    });
    
    // Benchmark access pattern prediction
    group.bench_function("access_pattern_prediction", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Create access pattern data
                for i in 0..300 {
                    let mut metadata = HashMap::new();
                    metadata.insert("access_count".to_string(), ((i % 20) + 1).to_string());
                    metadata.insert("last_access".to_string(), i.to_string());
                    
                    let content = format!("Access pattern data {}", i);
                    manager.store_memory(
                        &content,
                        MemoryType::ShortTerm,
                        metadata,
                    ).await.unwrap();
                }
                
                // Predict access patterns
                manager.predict_access_patterns().await.unwrap();
            })
        })
    });
    
    group.finish();
}

/// Benchmark analytics aggregation
fn bench_analytics_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("analytics_aggregation");
    
    // Benchmark time-series aggregation
    group.bench_function("timeseries_aggregation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Create time-series data
                for hour in 0..24 {
                    for minute in 0..60 {
                        let mut metadata = HashMap::new();
                        metadata.insert("hour".to_string(), hour.to_string());
                        metadata.insert("minute".to_string(), minute.to_string());
                        metadata.insert("value".to_string(), ((hour * 60 + minute) % 100).to_string());
                        
                        let content = format!("Timeseries data {}:{:02}", hour, minute);
                        manager.store_memory(
                            &content,
                            MemoryType::ShortTerm,
                            metadata,
                        ).await.unwrap();
                    }
                }
                
                // Aggregate by hour
                manager.aggregate_timeseries_data(black_box("hour")).await.unwrap();
            })
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_analytics_calculations,
    bench_realtime_analytics,
    bench_optimization_analytics,
    bench_predictive_analytics,
    bench_analytics_aggregation
);
criterion_main!(benches);
