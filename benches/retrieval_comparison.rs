//! Simple performance comparison between original and optimized retrieval

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use synaptic::memory::{
    storage::{memory::MemoryStorage, Storage},
    retrieval::{MemoryRetriever, RetrievalConfig, IndexedMemoryRetriever, IndexingConfig},
    types::{MemoryEntry, MemoryType, MemoryMetadata},
};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Create test memory entries
fn create_test_entries(count: usize) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    let base_time = chrono::Utc::now() - chrono::Duration::hours(24 * 30);
    
    for i in 0..count {
        let mut entry = MemoryEntry::new(
            format!("test_key_{}", i),
            format!("Test content for memory entry number {}", i),
            if i % 3 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
        );
        
        let mut metadata = MemoryMetadata::new();
        metadata.created_at = base_time + chrono::Duration::hours(i as i64);
        metadata.last_accessed = base_time + chrono::Duration::hours((i * 2) as i64);
        metadata.access_count = (i % 100) as u32;
        metadata.importance = (i as f64 % 10.0) / 10.0;
        
        if i % 5 == 0 {
            metadata.tags.push("important".to_string());
        }
        
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

/// Benchmark original vs optimized retrieval
fn bench_retrieval_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("retrieval_comparison");
    
    for size in [100, 500, 1000].iter() {
        // Original retrieval
        group.bench_with_input(
            BenchmarkId::new("original_get_recent", size),
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
        
        // Optimized retrieval (with fallback - no indexes)
        group.bench_with_input(
            BenchmarkId::new("optimized_fallback_get_recent", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let indexing_config = IndexingConfig {
                    enable_access_time_index: false,
                    enable_frequency_index: false,
                    enable_tag_index: false,
                    ..Default::default()
                };
                let retriever = IndexedMemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                    indexing_config,
                );
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_recent(black_box(10)).await.unwrap();
                    black_box(result)
                });
            },
        );
        
        // Optimized retrieval (with indexes)
        group.bench_with_input(
            BenchmarkId::new("optimized_indexed_get_recent", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let retriever = IndexedMemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                    IndexingConfig::default(),
                );
                
                // Initialize indexes
                rt.block_on(retriever.initialize_indexes()).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_recent(black_box(10)).await.unwrap();
                    black_box(result)
                });
            },
        );
        
        // Test frequent retrieval
        group.bench_with_input(
            BenchmarkId::new("original_get_frequent", size),
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
        
        group.bench_with_input(
            BenchmarkId::new("optimized_indexed_get_frequent", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let retriever = IndexedMemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                    IndexingConfig::default(),
                );
                
                rt.block_on(retriever.initialize_indexes()).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_frequent(black_box(10)).await.unwrap();
                    black_box(result)
                });
            },
        );
        
        // Test tag-based retrieval
        group.bench_with_input(
            BenchmarkId::new("original_get_by_tags", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
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
        
        group.bench_with_input(
            BenchmarkId::new("optimized_indexed_get_by_tags", size),
            size,
            |b, &size| {
                let storage = rt.block_on(setup_storage_with_data(size));
                let retriever = IndexedMemoryRetriever::new(
                    storage.clone(),
                    RetrievalConfig::default(),
                    IndexingConfig::default(),
                );
                
                rt.block_on(retriever.initialize_indexes()).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let result = retriever.get_by_tags(black_box(&["important".to_string()])).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache effectiveness
fn bench_cache_effectiveness(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("cache_effectiveness");
    
    let storage = rt.block_on(setup_storage_with_data(1000));
    let retriever = IndexedMemoryRetriever::new(
        storage.clone(),
        RetrievalConfig::default(),
        IndexingConfig::default(),
    );
    
    rt.block_on(retriever.initialize_indexes()).unwrap();
    
    // First call (cold cache)
    group.bench_function("first_call_cold_cache", |b| {
        b.to_async(&rt).iter(|| async {
            let result = retriever.get_recent(black_box(10)).await.unwrap();
            black_box(result)
        });
    });
    
    // Subsequent calls (warm cache)
    group.bench_function("subsequent_call_warm_cache", |b| {
        // Warm up the cache
        rt.block_on(async {
            let _ = retriever.get_recent(10).await.unwrap();
        });
        
        b.to_async(&rt).iter(|| async {
            let result = retriever.get_recent(black_box(10)).await.unwrap();
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_retrieval_comparison, bench_cache_effectiveness);
criterion_main!(benches);
