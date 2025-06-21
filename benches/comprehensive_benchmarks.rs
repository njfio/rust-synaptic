//! Simplified Performance Benchmarking Suite
//!
//! Basic benchmarks for core memory operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use synaptic::{AgentMemory, MemoryConfig};
use tokio::runtime::Runtime;

/// Simple memory operations benchmark
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_store_retrieve", |b| {
        b.iter(|| {
            rt.block_on(async {
                let memory_config = MemoryConfig::default();
                let mut memory = AgentMemory::new(memory_config).await.unwrap();
                
                // Store operation
                memory.store("test_key", "test_value").await.unwrap();
                
                // Retrieve operation
                let result = memory.retrieve("test_key").await.unwrap();
                black_box(result);
            })
        })
    });
}

/// Simple search benchmark
fn bench_search_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_search", |b| {
        b.iter(|| {
            rt.block_on(async {
                let memory_config = MemoryConfig::default();
                let mut memory = AgentMemory::new(memory_config).await.unwrap();
                
                // Pre-populate with some data
                for i in 0..10 {
                    let key = format!("search_key_{}", i);
                    let value = format!("search_value_{}", i);
                    memory.store(&key, &value).await.unwrap();
                }
                
                // Search operation
                let results = memory.search("search", 5).await.unwrap();
                black_box(results);
            })
        })
    });
}

/// Batch operations benchmark
fn bench_batch_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_batch_store", |b| {
        b.iter(|| {
            rt.block_on(async {
                let memory_config = MemoryConfig::default();
                let mut memory = AgentMemory::new(memory_config).await.unwrap();
                
                // Batch store operations
                for i in 0..100 {
                    let key = format!("batch_key_{}", i);
                    let value = format!("batch_value_{}", i);
                    memory.store(&key, &value).await.unwrap();
                }
                
                black_box(());
            })
        })
    });
}

criterion_group!(
    benches,
    bench_memory_operations,
    bench_search_operations,
    bench_batch_operations
);
criterion_main!(benches);
