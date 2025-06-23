# Comprehensive Testing Strategy

This document outlines the testing strategy for the Synaptic memory system, covering unit tests, integration tests, performance tests, and error condition tests to ensure 90%+ test coverage and production readiness.

## Testing Philosophy

### Core Principles

1. **Test Pyramid**: More unit tests, fewer integration tests, minimal end-to-end tests
2. **Fast Feedback**: Tests should run quickly and provide immediate feedback
3. **Deterministic**: Tests should be reliable and not flaky
4. **Isolated**: Tests should not depend on external services or state
5. **Comprehensive**: Cover happy paths, error conditions, and edge cases

### Coverage Goals

- **Unit Tests**: 95%+ line coverage
- **Integration Tests**: 90%+ feature coverage
- **Performance Tests**: All critical paths benchmarked
- **Error Condition Tests**: All error paths validated

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual functions and methods in isolation.

**Structure**:
```
tests/
├── unit/
│   ├── memory/
│   │   ├── storage/
│   │   │   ├── memory_storage_tests.rs
│   │   │   ├── file_storage_tests.rs
│   │   │   └── sql_storage_tests.rs
│   │   ├── management/
│   │   │   ├── memory_manager_tests.rs
│   │   │   ├── consolidation_tests.rs
│   │   │   └── optimization_tests.rs
│   │   └── types/
│   │       ├── memory_entry_tests.rs
│   │       └── memory_fragment_tests.rs
│   ├── security/
│   │   ├── encryption_tests.rs
│   │   ├── access_control_tests.rs
│   │   └── audit_tests.rs
│   └── cross_platform/
│       ├── wasm_adapter_tests.rs
│       └── mobile_adapter_tests.rs
```

**Example Unit Test Pattern**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_memory_storage_basic_operations() {
        let storage = MemoryStorage::new();
        let entry = create_test_entry("test_key", "test_value");
        
        // Test store
        let result = storage.store(&entry).await;
        assert!(result.is_ok());
        
        // Test retrieve
        let retrieved = storage.retrieve("test_key").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "test_value");
        
        // Test delete
        let deleted = storage.delete("test_key").await.unwrap();
        assert!(deleted);
        
        // Verify deletion
        let after_delete = storage.retrieve("test_key").await.unwrap();
        assert!(after_delete.is_none());
    }
    
    #[test]
    fn test_memory_entry_validation() {
        // Test valid entry
        let valid_entry = MemoryEntry::new(
            "valid_key".to_string(),
            "valid_content".to_string(),
            MemoryType::ShortTerm,
        );
        assert!(valid_entry.validate().is_ok());
        
        // Test invalid entry (empty key)
        let invalid_entry = MemoryEntry::new(
            "".to_string(),
            "content".to_string(),
            MemoryType::ShortTerm,
        );
        assert!(invalid_entry.validate().is_err());
    }
}
```

### 2. Integration Tests

**Purpose**: Test interactions between components and external systems.

**Structure**:
```
tests/
├── integration/
│   ├── memory_system_integration_tests.rs
│   ├── storage_backend_integration_tests.rs
│   ├── security_integration_tests.rs
│   ├── cross_platform_integration_tests.rs
│   ├── cli_integration_tests.rs
│   └── api_integration_tests.rs
```

**Example Integration Test Pattern**:
```rust
#[tokio::test]
async fn test_memory_manager_with_file_storage() {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path().join("test.db")).await.unwrap();
    let manager = MemoryManager::new(Arc::new(storage)).await.unwrap();
    
    // Test end-to-end memory operations
    let memory_id = manager.store_memory(
        "Test content for integration",
        MemoryType::LongTerm,
        HashMap::new(),
    ).await.unwrap();
    
    // Test retrieval
    let retrieved = manager.get_memory(&memory_id).await.unwrap();
    assert!(retrieved.is_some());
    
    // Test search
    let results = manager.search_memories("integration", 10).await.unwrap();
    assert!(!results.is_empty());
    
    // Test consolidation
    manager.consolidate_memories().await.unwrap();
    
    // Verify memory still exists after consolidation
    let after_consolidation = manager.get_memory(&memory_id).await.unwrap();
    assert!(after_consolidation.is_some());
}
```

### 3. Performance Tests

**Purpose**: Validate performance characteristics and identify bottlenecks.

**Structure**:
```
tests/
├── performance/
│   ├── memory_operations_benchmarks.rs
│   ├── storage_performance_tests.rs
│   ├── search_performance_tests.rs
│   ├── consolidation_performance_tests.rs
│   └── concurrent_access_tests.rs
```

**Example Performance Test Pattern**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_memory_storage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let storage = rt.block_on(async { MemoryStorage::new() });
    
    c.bench_function("memory_store_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                for i in 0..1000 {
                    let entry = create_test_entry(
                        &format!("key_{}", i),
                        &format!("value_{}", i),
                    );
                    storage.store(black_box(&entry)).await.unwrap();
                }
            })
        })
    });
    
    c.bench_function("memory_retrieve_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                for i in 0..1000 {
                    let key = format!("key_{}", i);
                    storage.retrieve(black_box(&key)).await.unwrap();
                }
            })
        })
    });
}

criterion_group!(benches, benchmark_memory_storage);
criterion_main!(benches);
```

**Performance Targets**:
- Memory operations: >100K ops/second
- Search operations: <100ms for 10K entries
- Consolidation: <1s for 1K entries
- Concurrent access: Linear scaling up to 8 threads

### 4. Error Condition Tests

**Purpose**: Validate error handling and recovery mechanisms.

**Structure**:
```
tests/
├── error_conditions/
│   ├── storage_failure_tests.rs
│   ├── network_failure_tests.rs
│   ├── memory_pressure_tests.rs
│   ├── corruption_recovery_tests.rs
│   └── timeout_handling_tests.rs
```

**Example Error Condition Test Pattern**:
```rust
#[tokio::test]
async fn test_storage_failure_recovery() {
    let storage = FailingStorage::new(3); // Fail first 3 operations
    let manager = MemoryManager::new(Arc::new(storage)).await.unwrap();
    
    // First few operations should fail
    for i in 0..3 {
        let result = manager.store_memory(
            &format!("content_{}", i),
            MemoryType::ShortTerm,
            HashMap::new(),
        ).await;
        assert!(result.is_err());
    }
    
    // Subsequent operations should succeed
    let result = manager.store_memory(
        "success_content",
        MemoryType::ShortTerm,
        HashMap::new(),
    ).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    let mut config = MemoryConfig::default();
    config.max_memory_usage = 1024; // Very low limit
    
    let manager = MemoryManager::with_config(config).await.unwrap();
    
    // Fill memory to capacity
    let mut stored_ids = Vec::new();
    for i in 0..100 {
        let large_content = "x".repeat(100); // 100 bytes each
        match manager.store_memory(&large_content, MemoryType::ShortTerm, HashMap::new()).await {
            Ok(id) => stored_ids.push(id),
            Err(_) => break, // Expected when memory is full
        }
    }
    
    // Verify some memories were stored
    assert!(!stored_ids.is_empty());
    
    // Verify memory pressure triggers cleanup
    let stats = manager.get_statistics().await.unwrap();
    assert!(stats.memory_usage_bytes <= config.max_memory_usage);
}
```

### 5. Property-Based Tests

**Purpose**: Test invariants and properties across a wide range of inputs.

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn memory_similarity_properties(
        content1 in "\\PC{1,1000}",
        content2 in "\\PC{1,1000}"
    ) {
        let entry1 = MemoryEntry::new("key1".to_string(), content1.clone(), MemoryType::ShortTerm);
        let entry2 = MemoryEntry::new("key2".to_string(), content2.clone(), MemoryType::ShortTerm);
        let entry3 = MemoryEntry::new("key3".to_string(), content1.clone(), MemoryType::ShortTerm);
        
        let similarity_engine = SimilarityEngine::new();
        
        // Reflexivity: similarity(A, A) = 1.0
        let self_sim = similarity_engine.calculate_similarity(&entry1, &entry1);
        prop_assert!((self_sim - 1.0).abs() < f64::EPSILON);
        
        // Symmetry: similarity(A, B) = similarity(B, A)
        let sim_ab = similarity_engine.calculate_similarity(&entry1, &entry2);
        let sim_ba = similarity_engine.calculate_similarity(&entry2, &entry1);
        prop_assert!((sim_ab - sim_ba).abs() < f64::EPSILON);
        
        // Identical content should have similarity 1.0
        let identical_sim = similarity_engine.calculate_similarity(&entry1, &entry3);
        prop_assert!((identical_sim - 1.0).abs() < f64::EPSILON);
        
        // Similarity should be in range [0, 1]
        prop_assert!(sim_ab >= 0.0 && sim_ab <= 1.0);
    }
}
```

## Test Infrastructure

### 1. Test Utilities

**Common Test Helpers**:
```rust
// tests/common/mod.rs
pub fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
    MemoryEntry::new(
        key.to_string(),
        content.to_string(),
        MemoryType::ShortTerm,
    )
}

pub fn create_test_config() -> MemoryConfig {
    MemoryConfig {
        max_memory_usage: 100 * 1024 * 1024, // 100MB
        consolidation_interval: Duration::from_secs(60),
        enable_encryption: false, // Disable for testing
        ..Default::default()
    }
}

pub async fn create_temp_storage() -> (TempDir, FileStorage) {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path().join("test.db")).await.unwrap();
    (temp_dir, storage)
}
```

### 2. Mock Objects

**Storage Mocks**:
```rust
pub struct MockStorage {
    data: Arc<RwLock<HashMap<String, MemoryEntry>>>,
    fail_operations: Arc<AtomicBool>,
    operation_delay: Duration,
}

impl MockStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            fail_operations: Arc::new(AtomicBool::new(false)),
            operation_delay: Duration::from_millis(0),
        }
    }
    
    pub fn set_failure_mode(&self, should_fail: bool) {
        self.fail_operations.store(should_fail, Ordering::Relaxed);
    }
    
    pub fn set_operation_delay(&mut self, delay: Duration) {
        self.operation_delay = delay;
    }
}

#[async_trait]
impl Storage for MockStorage {
    async fn store(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
        if self.operation_delay > Duration::from_millis(0) {
            tokio::time::sleep(self.operation_delay).await;
        }
        
        if self.fail_operations.load(Ordering::Relaxed) {
            return Err(MemoryError::storage("Mock storage failure"));
        }
        
        let mut data = self.data.write().await;
        data.insert(entry.key.clone(), entry.clone());
        Ok(())
    }
    
    // ... other methods
}
```

### 3. Test Configuration

**Cargo.toml Test Configuration**:
```toml
[dev-dependencies]
tokio-test = "0.4"
tempfile = "3.0"
criterion = "0.5"
proptest = "1.0"
mockall = "0.11"
wiremock = "0.5"

[[bench]]
name = "memory_operations"
harness = false

[[bench]]
name = "storage_performance"
harness = false

[features]
test-utils = []
```

## Continuous Integration

### 1. Test Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run unit tests
        run: cargo test --lib --bins
      
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: cargo test --test '*integration*'
        env:
          DATABASE_URL: postgres://postgres:test@localhost/test
          
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: cargo bench
        
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
      - name: Generate coverage
        run: cargo tarpaulin --out xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 2. Quality Gates

**Coverage Requirements**:
- Minimum 90% line coverage
- Minimum 85% branch coverage
- No decrease in coverage for PRs

**Performance Requirements**:
- No regression >5% in critical paths
- Memory usage within bounds
- No new memory leaks

## Test Execution Strategy

### 1. Local Development

```bash
# Run all tests
cargo test

# Run specific test category
cargo test --test unit
cargo test --test integration
cargo test --test performance

# Run with coverage
cargo tarpaulin --out html

# Run benchmarks
cargo bench
```

### 2. Pre-commit Hooks

```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run fast tests
cargo test --lib --bins --quiet

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy -- -D warnings

# Check for unwrap() in production code
if grep -r "\.unwrap()" src/ --exclude-dir=tests; then
    echo "Error: Found unwrap() calls in production code"
    exit 1
fi
```

This comprehensive testing strategy ensures high-quality, reliable code with excellent coverage and performance validation across all components of the Synaptic memory system.
