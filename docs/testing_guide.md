# Synaptic Testing Guide

This guide covers testing strategies, test organization, and best practices for the Synaptic project.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Security Testing](#security-testing)
8. [Test Data Management](#test-data-management)
9. [Continuous Integration](#continuous-integration)

## Testing Overview

Synaptic has a comprehensive test suite with 189+ tests covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Performance Tests**: Benchmarking and load testing
- **Security Tests**: Security feature validation
- **End-to-End Tests**: Complete workflow testing

### Test Statistics

```
Total Tests: 189+
├── Unit Tests: 120+
├── Integration Tests: 45+
├── Performance Tests: 15+
├── Security Tests: 9+
└── Documentation Tests: 5+

Coverage: 90%+ across all modules
```

## Test Organization

### Directory Structure

```
tests/
├── integration/                 # Integration tests
│   ├── memory_integration.rs
│   ├── knowledge_graph_integration.rs
│   └── security_integration.rs
├── performance/                 # Performance tests
│   ├── memory_benchmarks.rs
│   ├── search_benchmarks.rs
│   └── storage_benchmarks.rs
├── security/                    # Security tests
│   ├── encryption_tests.rs
│   ├── access_control_tests.rs
│   └── audit_tests.rs
├── assets/                      # Test data files
│   ├── test_documents/
│   ├── test_images/
│   └── test_audio/
└── test_config.toml            # Test configuration

src/
├── lib.rs                      # Library tests
├── memory/
│   ├── storage/
│   │   └── mod.rs              # Storage unit tests
│   ├── knowledge_graph/
│   │   └── mod.rs              # Graph unit tests
│   └── management/
│       └── mod.rs              # Management unit tests
└── analytics/
    └── mod.rs                  # Analytics unit tests
```

### Test Categories

#### Unit Tests (in `src/`)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_store() {
        // Test individual function
    }

    #[test]
    fn test_validation_logic() {
        // Test pure functions
    }
}
```

#### Integration Tests (in `tests/`)
```rust
// tests/integration_tests.rs
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::test]
async fn test_full_memory_workflow() {
    // Test complete workflows
}
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_memory_store

# Run tests in specific file
cargo test --test integration_tests

# Run tests with specific features
cargo test --features "analytics security"

# Run tests with all features
cargo test --all-features
```

### Test Filtering

```bash
# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test '*'

# Run tests matching pattern
cargo test memory

# Run tests excluding pattern
cargo test -- --skip slow_test

# Run ignored tests
cargo test -- --ignored
```

### Parallel vs Sequential Testing

```bash
# Run tests in parallel (default)
cargo test

# Run tests sequentially
cargo test -- --test-threads=1

# Run specific number of threads
cargo test -- --test-threads=4
```

## Unit Testing

### Memory Storage Tests

```rust
#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::memory::types::{MemoryEntry, MemoryType};

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let storage = MemoryStorage::new();
        let entry = MemoryEntry::new(
            "test_key".to_string(),
            "test_value".to_string(),
            MemoryType::ShortTerm,
        );

        // Test store
        storage.store(&entry).await.unwrap();

        // Test retrieve
        let retrieved = storage.retrieve("test_key").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "test_value");
    }

    #[tokio::test]
    async fn test_update_memory() {
        let storage = MemoryStorage::new();
        let entry = MemoryEntry::new(
            "test_key".to_string(),
            "original_value".to_string(),
            MemoryType::ShortTerm,
        );

        storage.store(&entry).await.unwrap();

        let updated_entry = MemoryEntry::new(
            "test_key".to_string(),
            "updated_value".to_string(),
            MemoryType::ShortTerm,
        );

        storage.update("test_key", &updated_entry).await.unwrap();

        let retrieved = storage.retrieve("test_key").await.unwrap();
        assert_eq!(retrieved.unwrap().value, "updated_value");
    }

    #[tokio::test]
    async fn test_delete_memory() {
        let storage = MemoryStorage::new();
        let entry = MemoryEntry::new(
            "test_key".to_string(),
            "test_value".to_string(),
            MemoryType::ShortTerm,
        );

        storage.store(&entry).await.unwrap();
        storage.delete("test_key").await.unwrap();

        let retrieved = storage.retrieve("test_key").await.unwrap();
        assert!(retrieved.is_none());
    }
}
```

### Knowledge Graph Tests

```rust
#[cfg(test)]
mod knowledge_graph_tests {
    use super::*;

    #[tokio::test]
    async fn test_add_nodes_and_edges() {
        let mut graph = KnowledgeGraph::new().await.unwrap();

        // Add nodes
        graph.add_node("node1", "Content 1").await.unwrap();
        graph.add_node("node2", "Content 2").await.unwrap();

        // Add edge
        graph.add_edge("node1", "node2", 0.8, "related").await.unwrap();

        // Verify relationship
        let related = graph.find_related("node1", 1).await.unwrap();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].target, "node2");
    }

    #[tokio::test]
    async fn test_graph_traversal() {
        let mut graph = KnowledgeGraph::new().await.unwrap();

        // Create a chain: A -> B -> C
        graph.add_node("A", "Node A").await.unwrap();
        graph.add_node("B", "Node B").await.unwrap();
        graph.add_node("C", "Node C").await.unwrap();

        graph.add_edge("A", "B", 0.9, "connects").await.unwrap();
        graph.add_edge("B", "C", 0.8, "connects").await.unwrap();

        // Test traversal
        let path = graph.shortest_path("A", "C").await.unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 3); // A -> B -> C
    }
}
```

### Error Handling Tests

```rust
#[cfg(test)]
mod error_tests {
    use super::*;
    use crate::error::SynapticError;

    #[tokio::test]
    async fn test_invalid_key_error() {
        let storage = MemoryStorage::new();
        
        // Test with empty key
        let result = storage.retrieve("").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SynapticError::ValidationError(msg) => {
                assert!(msg.contains("empty"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[tokio::test]
    async fn test_storage_error_handling() {
        // Test storage backend failures
        let storage = create_failing_storage();
        let entry = MemoryEntry::new(
            "test".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        );

        let result = storage.store(&entry).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SynapticError::StorageError(_) => {
                // Expected error type
            }
            _ => panic!("Expected StorageError"),
        }
    }
}
```

## Integration Testing

### Full System Integration

```rust
// tests/integration_tests.rs
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};
use synaptic::memory::types::{MemoryEntry, MemoryType};

#[tokio::test]
async fn test_complete_memory_workflow() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        enable_consolidation: false, // Disable for predictable testing
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store multiple related memories
    memory.store("ai", "Artificial Intelligence is a field of computer science").await.unwrap();
    memory.store("ml", "Machine Learning is a subset of AI").await.unwrap();
    memory.store("dl", "Deep Learning is a subset of Machine Learning").await.unwrap();

    // Test search functionality
    let results = memory.search("artificial intelligence", 5).await.unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.key == "ai"));

    // Test knowledge graph relationships
    if let Some(graph) = memory.knowledge_graph() {
        let related = graph.find_related("ai", 2).await.unwrap();
        assert!(!related.is_empty());
    }

    // Test memory statistics
    let stats = memory.stats();
    assert_eq!(stats.total_count, 3);
    assert!(stats.total_size > 0);
}
```

### Cross-Feature Integration

```rust
#[tokio::test]
async fn test_analytics_integration() {
    let config = MemoryConfig {
        enable_analytics: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store and access memories to generate analytics data
    for i in 0..10 {
        memory.store(&format!("key_{}", i), &format!("value_{}", i)).await.unwrap();
    }

    // Access some memories multiple times
    for _ in 0..5 {
        memory.retrieve("key_0").await.unwrap();
        memory.retrieve("key_1").await.unwrap();
    }

    // Test analytics
    if let Some(analytics) = memory.analytics() {
        let insights = analytics.generate_insights().await.unwrap();
        assert!(!insights.is_empty());

        let performance = analytics.get_performance_metrics().await.unwrap();
        assert!(performance.total_operations > 0);
    }
}
```

## Performance Testing

### Benchmark Tests

```rust
// benches/memory_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use synaptic::{AgentMemory, MemoryConfig};

fn benchmark_memory_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("memory_store", |b| {
        b.to_async(&rt).iter(|| async {
            let config = MemoryConfig::default();
            let mut memory = AgentMemory::new(config).await.unwrap();
            
            memory.store(
                black_box("benchmark_key"),
                black_box("benchmark_value")
            ).await.unwrap();
        });
    });

    c.bench_function("memory_retrieve", |b| {
        let config = MemoryConfig::default();
        let mut memory = rt.block_on(AgentMemory::new(config)).unwrap();
        rt.block_on(memory.store("key", "value")).unwrap();

        b.to_async(&rt).iter(|| async {
            memory.retrieve(black_box("key")).await.unwrap();
        });
    });

    c.bench_function("memory_search", |b| {
        let config = MemoryConfig::default();
        let mut memory = rt.block_on(AgentMemory::new(config)).unwrap();
        
        // Pre-populate with test data
        rt.block_on(async {
            for i in 0..1000 {
                memory.store(&format!("key_{}", i), &format!("test data {}", i)).await.unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            memory.search(black_box("test"), black_box(10)).await.unwrap();
        });
    });
}

criterion_group!(benches, benchmark_memory_operations);
criterion_main!(benches);
```

### Load Testing

```rust
#[tokio::test]
async fn test_concurrent_operations() {
    let config = MemoryConfig::default();
    let memory = Arc::new(Mutex::new(AgentMemory::new(config).await.unwrap()));
    
    let mut handles = vec![];
    
    // Spawn 100 concurrent operations
    for i in 0..100 {
        let memory_clone = memory.clone();
        let handle = tokio::spawn(async move {
            let mut mem = memory_clone.lock().await;
            mem.store(&format!("concurrent_key_{}", i), &format!("value_{}", i)).await.unwrap();
            mem.retrieve(&format!("concurrent_key_{}", i)).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all data was stored correctly
    let mem = memory.lock().await;
    let stats = mem.stats();
    assert_eq!(stats.total_count, 100);
}
```

## Security Testing

### Encryption Tests

```rust
#[cfg(feature = "security")]
mod security_tests {
    use super::*;
    use synaptic::security::{SecurityManager, SecurityConfig};

    #[tokio::test]
    async fn test_memory_encryption() {
        let config = SecurityConfig::default();
        let mut security = SecurityManager::new(config).await.unwrap();
        
        let entry = MemoryEntry::new(
            "secret".to_string(),
            "sensitive data".to_string(),
            MemoryType::LongTerm,
        );

        // Create authentication context
        let context = create_test_auth_context();

        // Test encryption
        let encrypted = security.encrypt_memory(&entry, &context).await.unwrap();
        assert_ne!(encrypted.value, entry.value); // Should be encrypted

        // Test decryption
        let decrypted = security.decrypt_memory(&encrypted, &context).await.unwrap();
        assert_eq!(decrypted.value, entry.value); // Should match original
    }

    #[tokio::test]
    async fn test_access_control() {
        let config = SecurityConfig::default();
        let mut security = SecurityManager::new(config).await.unwrap();

        // Add roles
        security.access_control.add_role(
            "user".to_string(),
            vec![Permission::ReadMemory, Permission::WriteMemory]
        ).await.unwrap();

        // Test authentication
        let creds = create_test_credentials();
        let context = security.access_control.authenticate("user".to_string(), creds).await.unwrap();

        // Test permission checking
        assert!(security.access_control.check_permission(&context, Permission::ReadMemory).await.is_ok());
        assert!(security.access_control.check_permission(&context, Permission::DeleteMemory).await.is_err());
    }
}
```

## Test Data Management

### Test Fixtures

```rust
// tests/fixtures/mod.rs
use synaptic::memory::types::{MemoryEntry, MemoryType};

pub fn create_test_memory_entries() -> Vec<MemoryEntry> {
    vec![
        MemoryEntry::new(
            "test_1".to_string(),
            "First test entry".to_string(),
            MemoryType::ShortTerm,
        ),
        MemoryEntry::new(
            "test_2".to_string(),
            "Second test entry".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "test_3".to_string(),
            "Third test entry".to_string(),
            MemoryType::Working,
        ),
    ]
}

pub fn create_large_test_dataset(size: usize) -> Vec<MemoryEntry> {
    (0..size)
        .map(|i| MemoryEntry::new(
            format!("large_test_{}", i),
            format!("Large test entry number {}", i),
            MemoryType::ShortTerm,
        ))
        .collect()
}
```

### Test Configuration

```toml
# tests/test_config.toml
[test_settings]
timeout_seconds = 30
max_memory_mb = 512
enable_logging = false

[storage]
backend = "memory"
cleanup_after_test = true

[performance]
benchmark_iterations = 1000
load_test_concurrent_users = 100
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: synaptic_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
        components: rustfmt, clippy

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

    - name: Run tests
      run: |
        cargo test --all-features
        cargo test --no-default-features
        cargo test --features "analytics"
        cargo test --features "security"

    - name: Run clippy
      run: cargo clippy --all-features -- -D warnings

    - name: Check formatting
      run: cargo fmt -- --check

    - name: Run benchmarks
      run: cargo bench --features "full"
```

### Test Coverage

```bash
# Install cargo-tarpaulin for coverage
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --all-features --out Html --output-dir coverage

# Upload to codecov
bash <(curl -s https://codecov.io/bash)
```

This testing guide provides comprehensive coverage of testing strategies and practices for the Synaptic project. The test suite ensures reliability, performance, and security across all components.
