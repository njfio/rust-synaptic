//! Performance benchmarks for memory retrieval operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use synaptic::memory::{
    storage::{memory::MemoryStorage, Storage},
    retrieval::{MemoryRetriever, RetrievalConfig},
    types::{MemoryEntry, MemoryType, MemoryMetadata},
};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Create test memory entries with varying access patterns
fn create_test_entries(count: usize) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    let base_time = chrono::Utc::now() - chrono::Duration::hours(24 * 30); // 30 days ago
    
    for i in 0..count {
        let mut entry = MemoryEntry::new(
            format!("test_key_{}", i),
            format!("Test content for memory entry number {}", i),
            if i % 3 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
        );
        
        // Simulate varying access patterns
        let mut metadata = MemoryMetadata::new();
        metadata.created_at = base_time + chrono::Duration::hours(i as i64);
        metadata.last_accessed = base_time + chrono::Duration::hours((i * 2) as i64);
        metadata.access_count = (i % 100) as u32; // Varying access counts
        metadata.importance = (i as f64 % 10.0) / 10.0; // Varying importance
        
        entry.metadata = metadata;
        entries.push(entry);
    }
    
    entries
}

/// Setup storage with test data
async fn setup_storage_with_data(entry_count: usize) -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());
    let entries = create_test_entries(entry_count);
    
    for entry in entries {
        storage.store(&entry).await.unwrap();
    }
    
    storage
}

/// Benchmark get_recent performance with different dataset sizes
fn bench_get_recent(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("get_recent");
    
    for size in [100, 500, 1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("current_implementation", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let retriever = MemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                );
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_recent(black_box(10)).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark get_frequent performance with different dataset sizes
fn bench_get_frequent(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("get_frequent");
    
    for size in [100, 500, 1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("current_implementation", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let retriever = MemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                );
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_frequent(black_box(10)).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark get_by_tags performance
fn bench_get_by_tags(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("get_by_tags");
    
    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("current_implementation", size),
            size,
            |b, &size| {
                let storage = rt.block_on(async {
                    let storage = Arc::new(MemoryStorage::new());
                    let mut entries = create_test_entries(*size);
                    
                    // Add tags to some entries
                    for (i, entry) in entries.iter_mut().enumerate() {
                        if i % 5 == 0 {
                            entry.metadata.tags.push("important".to_string());
                        }
                        if i % 10 == 0 {
                            entry.metadata.tags.push("urgent".to_string());
                        }
                    }
                    
                    for entry in entries {
                        storage.store(&entry).await.unwrap();
                    }
                    
                    storage
                });
                
                let retriever = MemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                );
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_by_tags(black_box(&["important".to_string()])).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark list_keys operation (underlying bottleneck)
fn bench_list_keys(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("list_keys");
    
    for size in [100, 500, 1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_storage", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                
                b.to_async(&rt).iter(|| async {
                    let result = storage.list_keys().await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark individual retrieve operations
fn bench_retrieve_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("retrieve_operations");
    
    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("sequential_retrieval", size),
            size,
            |b, &size| {
                let (storage, keys) = rt.block_on(async {
                    let storage = setup_storage_with_data(*size).await;
                    let keys = storage.list_keys().await.unwrap();
                    (storage, keys)
                });
                
                b.to_async(&rt).iter(|| async {
                    let mut results = Vec::new();
                    for key in &keys {
                        if let Some(entry) = storage.retrieve(key).await.unwrap() {
                            results.push(entry);
                        }
                    }
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation and sorting overhead
fn bench_sorting_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_overhead");
    
    for size in [100, 500, 1000, 5000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("sort_by_access_time", size),
            size,
            |b, &size| {
                let entries = create_test_entries(*size);
                
                b.iter(|| {
                    let mut entries_copy = entries.clone();
                    entries_copy.sort_by(|a, b| b.last_accessed().cmp(&a.last_accessed()));
                    entries_copy.truncate(10);
                    black_box(entries_copy)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("sort_by_access_count", size),
            size,
            |b, &size| {
                let entries = create_test_entries(*size);
                
                b.iter(|| {
                    let mut entries_copy = entries.clone();
                    entries_copy.sort_by(|a, b| b.access_count().cmp(&a.access_count()));
                    entries_copy.truncate(10);
                    black_box(entries_copy)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_get_recent,
    bench_get_frequent,
    bench_get_by_tags,
    bench_list_keys,
    bench_retrieve_operations,
    bench_sorting_overhead
);
criterion_main!(benches);
