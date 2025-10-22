//! Benchmark comparing ANN (HNSW) vs brute-force similarity search
//!
//! This benchmark demonstrates the performance benefits of using ANN
//! for large-scale vector similarity search.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use synaptic::memory::embeddings::{EmbeddingProvider, TfIdfProvider};
use synaptic::memory::indexing::{HnswIndex, IndexConfig, VectorIndex};
use synaptic::memory::retrieval::{DenseVectorRetriever, RetrievalPipeline};
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::sync::Arc;
use tokio::runtime::Runtime;

// Helper to create test entries
fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
    MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::ShortTerm)
}

// Helper to create test vector
fn create_test_vector(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim).map(|i| seed + (i as f32 * 0.01)).collect()
}

// Benchmark ANN search with different dataset sizes
fn bench_ann_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ann_search");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Setup: Create and populate index
            let config = IndexConfig::new(128)
                .with_ef_construction(200)
                .with_ef_search(50);
            let mut index = HnswIndex::new(config);

            rt.block_on(async {
                for i in 0..size {
                    let vector = create_test_vector(128, (i % 100) as f32);
                    let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
                    index.add(&vector, entry).await.unwrap();
                }
            });

            // Benchmark: Search
            b.to_async(&rt).iter(|| async {
                let query = create_test_vector(128, 50.0);
                let results = index.search(black_box(&query), black_box(10)).await.unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

// Benchmark brute-force search for comparison
fn bench_bruteforce_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("bruteforce_search");

    for size in [100, 1000, 5000].iter() {
        // Skip 10000 for brute-force as it's too slow
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Setup: Create storage and retriever without index
            let storage = Arc::new(MemoryStorage::new());
            let provider = Arc::new(TfIdfProvider::default());
            let retriever = DenseVectorRetriever::new(storage.clone(), provider.clone());

            rt.block_on(async {
                for i in 0..size {
                    let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
                    storage.store(&entry).await.unwrap();
                }
            });

            // Benchmark: Search (brute-force)
            b.to_async(&rt).iter(|| async {
                let results = retriever
                    .search(black_box("content 50"), black_box(10), None)
                    .await
                    .unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

// Benchmark ANN search with index
fn bench_retriever_with_ann(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("retriever_with_ann");

    for size in [100, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Setup: Create storage and retriever WITH index
            let storage = Arc::new(MemoryStorage::new());
            let provider = Arc::new(TfIdfProvider::default());

            let config = IndexConfig::new(128)
                .with_ef_construction(200)
                .with_ef_search(50);
            let index = Box::new(HnswIndex::new(config));

            let retriever = DenseVectorRetriever::new(storage.clone(), provider.clone())
                .with_vector_index(index);

            // Populate storage and index
            rt.block_on(async {
                for i in 0..size {
                    let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
                    storage.store(&entry).await.unwrap();

                    // Also add to index
                    if let Some(ref index_arc) = retriever.vector_index() {
                        let embedding = provider.embed(&entry.value, None).await.unwrap();
                        let mut idx = index_arc.write();
                        idx.add(embedding.vector(), entry).await.unwrap();
                    }
                }
            });

            // Benchmark: Search with ANN
            b.to_async(&rt).iter(|| async {
                let results = retriever
                    .search(black_box("content 50"), black_box(10), None)
                    .await
                    .unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

// Benchmark index insertion performance
fn bench_index_insertion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("index_insertion");

    for batch_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter_batched(
                    || {
                        // Setup: Create fresh index for each iteration
                        let config = IndexConfig::new(128);
                        HnswIndex::new(config)
                    },
                    |mut index| {
                        // Benchmark: Batch insertion
                        rt.block_on(async {
                            for i in 0..batch_size {
                                let vector = create_test_vector(128, i as f32);
                                let entry = create_test_entry(
                                    &format!("key{}", i),
                                    &format!("content {}", i),
                                );
                                index.add(black_box(&vector), black_box(entry)).await.unwrap();
                            }
                            black_box(index);
                        });
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// Benchmark search quality vs speed tradeoff
fn bench_search_quality_tradeoff(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("search_quality_tradeoff");

    let size = 5000;

    for ef_search in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(ef_search),
            ef_search,
            |b, &ef_search| {
                // Setup: Create index with different ef_search values
                let config = IndexConfig::new(128)
                    .with_ef_construction(200)
                    .with_ef_search(ef_search);
                let mut index = HnswIndex::new(config);

                rt.block_on(async {
                    for i in 0..size {
                        let vector = create_test_vector(128, (i % 100) as f32);
                        let entry =
                            create_test_entry(&format!("key{}", i), &format!("content {}", i));
                        index.add(&vector, entry).await.unwrap();
                    }
                });

                // Benchmark: Search with different quality settings
                b.to_async(&rt).iter(|| async {
                    let query = create_test_vector(128, 50.0);
                    let results = index.search(black_box(&query), black_box(10)).await.unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

// Benchmark index persistence
fn bench_index_persistence(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("index_persistence");

    for size in [1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Setup: Create and populate index
            let config = IndexConfig::new(128);
            let mut index = HnswIndex::new(config);

            rt.block_on(async {
                for i in 0..size {
                    let vector = create_test_vector(128, i as f32);
                    let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
                    index.add(&vector, entry).await.unwrap();
                }
            });

            // Benchmark: Save and load
            b.to_async(&rt).iter(|| async {
                let temp_path = format!("/tmp/bench_index_{}.json", rand::random::<u64>());

                // Save
                index.save(black_box(&temp_path)).await.unwrap();

                // Load
                let config = IndexConfig::new(128);
                let mut new_index = HnswIndex::new(config);
                new_index.load(black_box(&temp_path)).await.unwrap();

                // Cleanup
                let _ = std::fs::remove_file(&temp_path);

                black_box(new_index);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ann_search,
    bench_bruteforce_search,
    bench_retriever_with_ann,
    bench_index_insertion,
    bench_search_quality_tradeoff,
    bench_index_persistence
);
criterion_main!(benches);
