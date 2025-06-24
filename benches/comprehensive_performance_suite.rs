use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use synaptic::memory::management::MemoryManager;
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::file::FileStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::memory::storage::Storage;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Create test memory entries for benchmarking
fn create_test_entries(count: usize) -> Vec<MemoryEntry> {
    (0..count)
        .map(|i| {
            MemoryEntry::new(
                format!("benchmark_key_{}", i),
                format!("Benchmark content for entry {} with some additional text to make it more realistic", i),
                if i % 2 == 0 { MemoryType::LongTerm } else { MemoryType::ShortTerm },
            )
        })
        .collect()
}

/// Benchmark memory storage operations
fn bench_storage_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("storage_operations");
    
    // Test different entry counts
    for entry_count in [100, 1000, 10000].iter() {
        let entries = create_test_entries(*entry_count);
        
        // Benchmark MemoryStorage
        group.throughput(Throughput::Elements(*entry_count as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_storage_store", entry_count),
            entry_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = MemoryStorage::new();
                        for entry in &entries {
                            storage.store(black_box(entry)).await.unwrap();
                        }
                    })
                })
            },
        );
        
        // Benchmark retrieval
        group.bench_with_input(
            BenchmarkId::new("memory_storage_retrieve", entry_count),
            entry_count,
            |b, _| {
                let storage = rt.block_on(async {
                    let storage = MemoryStorage::new();
                    for entry in &entries {
                        storage.store(entry).await.unwrap();
                    }
                    storage
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        for i in 0..*entry_count {
                            let key = format!("benchmark_key_{}", i);
                            storage.retrieve(black_box(&key)).await.unwrap();
                        }
                    })
                })
            },
        );
        
        // Benchmark FileStorage
        group.bench_with_input(
            BenchmarkId::new("file_storage_store", entry_count),
            entry_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = TempDir::new().unwrap();
                        let storage = FileStorage::new(temp_dir.path().join("bench.db")).await.unwrap();
                        for entry in &entries {
                            storage.store(black_box(entry)).await.unwrap();
                        }
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark search operations
fn bench_search_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("search_operations");
    
    // Prepare data
    let entries = create_test_entries(10000);
    let storage = rt.block_on(async {
        let storage = MemoryStorage::new();
        for entry in &entries {
            storage.store(entry).await.unwrap();
        }
        storage
    });
    
    // Test different search terms and limits
    let search_terms = ["benchmark", "content", "entry", "text"];
    let limits = [10, 100, 1000];
    
    for term in search_terms.iter() {
        for limit in limits.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("search_{}", term), limit),
                limit,
                |b, &limit| {
                    b.iter(|| {
                        rt.block_on(async {
                            storage.search(black_box(term), black_box(limit)).await.unwrap()
                        })
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark memory manager operations
fn bench_memory_manager(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_manager");
    
    // Test different scenarios
    for entry_count in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*entry_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("store_memory", entry_count),
            entry_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let manager = MemoryManager::new(storage).await.unwrap();
                        
                        for i in 0..count {
                            let content = format!("Memory content for benchmark {}", i);
                            manager.store_memory(
                                black_box(&content),
                                MemoryType::ShortTerm,
                                std::collections::HashMap::new(),
                            ).await.unwrap();
                        }
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("search_memories", entry_count),
            entry_count,
            |b, &count| {
                let manager = rt.block_on(async {
                    let storage = Arc::new(MemoryStorage::new());
                    let manager = MemoryManager::new(storage).await.unwrap();
                    
                    for i in 0..count {
                        let content = format!("Memory content for benchmark {}", i);
                        manager.store_memory(
                            &content,
                            MemoryType::ShortTerm,
                            std::collections::HashMap::new(),
                        ).await.unwrap();
                    }
                    manager
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        manager.search_memories(black_box("benchmark"), black_box(50)).await.unwrap()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_operations");
    
    // Test different thread counts
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_store", thread_count),
            thread_count,
            |b, &threads| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let mut handles = vec![];
                        
                        for thread_id in 0..threads {
                            let storage_clone = storage.clone();
                            let handle = tokio::spawn(async move {
                                for i in 0..100 {
                                    let entry = MemoryEntry::new(
                                        format!("thread_{}_entry_{}", thread_id, i),
                                        format!("Content from thread {} entry {}", thread_id, i),
                                        MemoryType::ShortTerm,
                                    );
                                    storage_clone.store(black_box(&entry)).await.unwrap();
                                }
                            });
                            handles.push(handle);
                        }
                        
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory consolidation
fn bench_consolidation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("consolidation");
    
    for entry_count in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("consolidate_memories", entry_count),
            entry_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = Arc::new(MemoryStorage::new());
                        let manager = MemoryManager::new(storage).await.unwrap();
                        
                        // Store memories with some duplicates
                        for i in 0..count {
                            let content = if i % 10 == 0 {
                                "Duplicate content for consolidation testing".to_string()
                            } else {
                                format!("Unique content for entry {}", i)
                            };
                            
                            manager.store_memory(
                                &content,
                                MemoryType::LongTerm,
                                std::collections::HashMap::new(),
                            ).await.unwrap();
                        }
                        
                        // Benchmark consolidation
                        manager.consolidate_memories().await.unwrap();
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark similarity calculations
fn bench_similarity_calculations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("similarity");
    
    let entries = create_test_entries(1000);
    
    group.bench_function("calculate_similarity_matrix", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = Arc::new(MemoryStorage::new());
                let manager = MemoryManager::new(storage).await.unwrap();
                
                // Store entries
                for entry in &entries[..100] { // Use subset for performance
                    manager.store_memory(
                        &entry.value,
                        entry.memory_type.clone(),
                        std::collections::HashMap::new(),
                    ).await.unwrap();
                }
                
                // Calculate similarities
                let memories = manager.get_all_memories().await.unwrap();
                for i in 0..memories.len().min(10) {
                    for j in (i+1)..memories.len().min(10) {
                        manager.calculate_similarity(&memories[i], &memories[j]).await.unwrap();
                    }
                }
            })
        })
    });
    
    group.finish();
}

/// Benchmark backup and restore operations
fn bench_backup_restore(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("backup_restore");
    
    for entry_count in [100, 1000, 5000].iter() {
        let entries = create_test_entries(*entry_count);
        
        group.bench_with_input(
            BenchmarkId::new("backup", entry_count),
            entry_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let storage = MemoryStorage::new();
                        for entry in &entries {
                            storage.store(entry).await.unwrap();
                        }
                        
                        let temp_dir = TempDir::new().unwrap();
                        let backup_path = temp_dir.path().join("benchmark_backup.json");
                        storage.backup(backup_path.to_str().unwrap()).await.unwrap();
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("restore", entry_count),
            entry_count,
            |b, _| {
                let backup_path = rt.block_on(async {
                    let storage = MemoryStorage::new();
                    for entry in &entries {
                        storage.store(entry).await.unwrap();
                    }
                    
                    let temp_dir = TempDir::new().unwrap();
                    let backup_path = temp_dir.path().join("benchmark_backup.json");
                    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
                    backup_path
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        let storage = MemoryStorage::new();
                        storage.restore(backup_path.to_str().unwrap()).await.unwrap();
                    })
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_storage_operations,
    bench_search_operations,
    bench_memory_manager,
    bench_concurrent_operations,
    bench_consolidation,
    bench_similarity_calculations,
    bench_backup_restore
);
criterion_main!(benches);
