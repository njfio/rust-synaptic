# Production-Ready Rust Patterns

This document outlines industry best practices for production-ready Rust applications, focusing on error handling, logging, performance optimization, and reliability patterns used in the Synaptic memory system.

## Error Handling Patterns

### 1. Result Type Usage

**Best Practice**: Use `Result<T, E>` for all fallible operations, never `unwrap()` in production code.

```rust
// ✅ Good - Proper error handling
pub async fn store_memory(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
    self.storage.store(entry).await
        .map_err(|e| MemoryError::storage(format!("Failed to store memory: {}", e)))
}

// ❌ Bad - Can panic in production
pub async fn store_memory(&self, entry: &MemoryEntry) {
    self.storage.store(entry).await.unwrap();
}
```

### 2. Error Type Hierarchy

**Pattern**: Use `thiserror` for library errors, `anyhow` for application errors.

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Storage operation failed: {message}")]
    StorageError { message: String },
    
    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),
    
    #[error("Network error: {source}")]
    NetworkError {
        #[from]
        source: std::io::Error,
    },
}
```

### 3. Error Context and Chaining

**Pattern**: Add context to errors as they propagate up the call stack.

```rust
use anyhow::{Context, Result};

pub async fn process_memory_batch(&self, entries: &[MemoryEntry]) -> Result<()> {
    for (index, entry) in entries.iter().enumerate() {
        self.store_memory(entry).await
            .with_context(|| format!("Failed to process memory entry at index {}", index))?;
    }
    Ok(())
}
```

### 4. Safe Unwrapping Utilities

**Pattern**: Create safe alternatives to `unwrap()` for common scenarios.

```rust
/// Safe unwrap with detailed error context
macro_rules! safe_unwrap {
    ($expr:expr, $msg:expr) => {
        $expr.unwrap_or_else(|| {
            tracing::error!("Safe unwrap failed: {}", $msg);
            panic!("Safe unwrap failed: {}", $msg);
        })
    };
}

/// Safe mutex locking with poisoned lock handling
pub trait SafeMutex<T> {
    fn safe_lock(&self) -> Result<std::sync::MutexGuard<T>, MemoryError>;
}

impl<T> SafeMutex<T> for std::sync::Mutex<T> {
    fn safe_lock(&self) -> Result<std::sync::MutexGuard<T>, MemoryError> {
        self.lock().map_err(|e| {
            MemoryError::concurrency(format!("Mutex lock failed: {}", e))
        })
    }
}
```

## Logging and Observability

### 1. Structured Logging with Tracing

**Best Practice**: Use `tracing` for all logging, never `println!` in production.

```rust
use tracing::{info, warn, error, debug, instrument, Span};

#[instrument(skip(self), fields(key = %key))]
pub async fn retrieve_memory(&self, key: &str) -> Result<Option<MemoryEntry>, MemoryError> {
    debug!("Starting memory retrieval");
    
    let start = std::time::Instant::now();
    let result = self.storage.retrieve(key).await;
    let duration = start.elapsed();
    
    match &result {
        Ok(Some(_)) => {
            info!(
                duration_ms = duration.as_millis(),
                "Memory retrieved successfully"
            );
        },
        Ok(None) => {
            debug!("Memory not found");
        },
        Err(e) => {
            error!(
                error = %e,
                duration_ms = duration.as_millis(),
                "Memory retrieval failed"
            );
        }
    }
    
    result
}
```

### 2. Log Levels and Context

**Pattern**: Use appropriate log levels with rich context.

```rust
// Error: System failures, unrecoverable errors
error!(
    error = %e,
    operation = "memory_consolidation",
    affected_entries = entries.len(),
    "Memory consolidation failed"
);

// Warn: Recoverable issues, degraded performance
warn!(
    cache_hit_rate = cache_stats.hit_rate,
    threshold = 0.8,
    "Cache hit rate below threshold"
);

// Info: Important business events
info!(
    memory_count = count,
    operation = "batch_store",
    duration_ms = duration.as_millis(),
    "Batch memory storage completed"
);

// Debug: Detailed execution flow
debug!(
    query = %query,
    limit = limit,
    "Executing memory search"
);
```

### 3. Metrics and Monitoring

**Pattern**: Integrate with Prometheus and OpenTelemetry for production monitoring.

```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram};

lazy_static! {
    static ref MEMORY_OPERATIONS: Counter = register_counter!(
        "synaptic_memory_operations_total",
        "Total number of memory operations"
    ).unwrap();
    
    static ref OPERATION_DURATION: Histogram = register_histogram!(
        "synaptic_operation_duration_seconds",
        "Duration of memory operations"
    ).unwrap();
    
    static ref ACTIVE_MEMORIES: Gauge = register_gauge!(
        "synaptic_active_memories",
        "Number of active memories in storage"
    ).unwrap();
}

pub async fn store_with_metrics(&self, entry: &MemoryEntry) -> Result<(), MemoryError> {
    let _timer = OPERATION_DURATION.start_timer();
    MEMORY_OPERATIONS.inc();
    
    let result = self.storage.store(entry).await;
    
    if result.is_ok() {
        ACTIVE_MEMORIES.inc();
    }
    
    result
}
```

## Performance Optimization Patterns

### 1. Async/Await Best Practices

**Pattern**: Use async judiciously, avoid blocking in async contexts.

```rust
// ✅ Good - Non-blocking async operations
pub async fn process_memories_concurrent(&self, entries: Vec<MemoryEntry>) -> Result<(), MemoryError> {
    let tasks: Vec<_> = entries.into_iter()
        .map(|entry| {
            let storage = self.storage.clone();
            tokio::spawn(async move {
                storage.store(&entry).await
            })
        })
        .collect();
    
    for task in tasks {
        task.await
            .map_err(|e| MemoryError::concurrency(format!("Task failed: {}", e)))??;
    }
    
    Ok(())
}

// ❌ Bad - Blocking in async context
pub async fn bad_example(&self) -> Result<(), MemoryError> {
    std::thread::sleep(Duration::from_secs(1)); // Blocks entire async runtime
    Ok(())
}
```

### 2. Memory Management

**Pattern**: Use appropriate data structures and avoid unnecessary allocations.

```rust
// ✅ Good - Efficient memory usage
pub fn process_memory_batch(&self, entries: &[MemoryEntry]) -> Result<Vec<ProcessedMemory>, MemoryError> {
    let mut results = Vec::with_capacity(entries.len()); // Pre-allocate
    
    for entry in entries {
        let processed = self.process_single(entry)?;
        results.push(processed);
    }
    
    Ok(results)
}

// ✅ Good - Use references when possible
pub fn calculate_similarity(&self, entry1: &MemoryEntry, entry2: &MemoryEntry) -> f64 {
    // Work with references, avoid cloning
    self.similarity_engine.calculate(&entry1.content, &entry2.content)
}
```

### 3. Caching Strategies

**Pattern**: Implement intelligent caching with proper invalidation.

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;

pub struct MemoryCache {
    cache: Arc<RwLock<LruCache<String, Arc<MemoryEntry>>>>,
    max_size: usize,
}

impl MemoryCache {
    pub async fn get(&self, key: &str) -> Option<Arc<MemoryEntry>> {
        let cache = self.cache.read().await;
        cache.get(key).cloned()
    }
    
    pub async fn insert(&self, key: String, entry: Arc<MemoryEntry>) {
        let mut cache = self.cache.write().await;
        cache.put(key, entry);
    }
    
    pub async fn invalidate(&self, key: &str) {
        let mut cache = self.cache.write().await;
        cache.pop(key);
    }
}
```

## Reliability Patterns

### 1. Circuit Breaker Pattern

**Pattern**: Prevent cascade failures with circuit breakers.

```rust
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct CircuitBreaker {
    failure_count: AtomicU64,
    last_failure: AtomicU64,
    is_open: AtomicBool,
    failure_threshold: u64,
    timeout: Duration,
}

impl CircuitBreaker {
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
    {
        if self.is_open() {
            return Err(/* circuit breaker open error */);
        }
        
        match operation.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            },
            Err(e) => {
                self.on_failure();
                Err(e)
            }
        }
    }
    
    fn is_open(&self) -> bool {
        if !self.is_open.load(Ordering::Relaxed) {
            return false;
        }
        
        let now = Instant::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let last_failure = self.last_failure.load(Ordering::Relaxed);
        
        if now - last_failure > self.timeout.as_secs() {
            self.is_open.store(false, Ordering::Relaxed);
            self.failure_count.store(0, Ordering::Relaxed);
            false
        } else {
            true
        }
    }
}
```

### 2. Retry Patterns

**Pattern**: Implement exponential backoff for transient failures.

```rust
use tokio::time::{sleep, Duration};

pub async fn retry_with_backoff<F, T, E>(
    mut operation: F,
    max_retries: u32,
    base_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
    E: std::fmt::Debug,
{
    let mut delay = base_delay;
    
    for attempt in 0..max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_retries - 1 {
                    return Err(e);
                }
                
                tracing::warn!(
                    attempt = attempt + 1,
                    max_retries = max_retries,
                    delay_ms = delay.as_millis(),
                    error = ?e,
                    "Operation failed, retrying"
                );
                
                sleep(delay).await;
                delay = std::cmp::min(delay * 2, Duration::from_secs(60)); // Cap at 60s
            }
        }
    }
    
    unreachable!()
}
```

### 3. Graceful Shutdown

**Pattern**: Handle shutdown signals gracefully.

```rust
use tokio::signal;
use tokio::sync::broadcast;

pub struct GracefulShutdown {
    shutdown_tx: broadcast::Sender<()>,
}

impl GracefulShutdown {
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        
        let tx = shutdown_tx.clone();
        tokio::spawn(async move {
            let _ = signal::ctrl_c().await;
            tracing::info!("Shutdown signal received");
            let _ = tx.send(());
        });
        
        Self { shutdown_tx }
    }
    
    pub fn subscribe(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }
}

// Usage in main application loop
pub async fn run_application() -> Result<(), Box<dyn std::error::Error>> {
    let shutdown = GracefulShutdown::new();
    let mut shutdown_rx = shutdown.subscribe();
    
    let memory_manager = MemoryManager::new().await?;
    
    tokio::select! {
        result = memory_manager.run() => {
            result?;
        },
        _ = shutdown_rx.recv() => {
            tracing::info!("Graceful shutdown initiated");
            memory_manager.shutdown().await?;
        }
    }
    
    Ok(())
}
```

## Testing Patterns

### 1. Property-Based Testing

**Pattern**: Use property-based testing for complex algorithms.

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn memory_similarity_is_symmetric(
        content1 in "\\PC*",
        content2 in "\\PC*"
    ) {
        let entry1 = MemoryEntry::new("key1".to_string(), content1, MemoryType::ShortTerm);
        let entry2 = MemoryEntry::new("key2".to_string(), content2, MemoryType::ShortTerm);
        
        let similarity_engine = SimilarityEngine::new();
        let sim1 = similarity_engine.calculate_similarity(&entry1, &entry2);
        let sim2 = similarity_engine.calculate_similarity(&entry2, &entry1);
        
        prop_assert!((sim1 - sim2).abs() < f64::EPSILON);
    }
}
```

### 2. Integration Testing

**Pattern**: Test real integrations with proper setup/teardown.

```rust
#[tokio::test]
async fn test_memory_storage_integration() {
    let temp_dir = tempfile::tempdir().unwrap();
    let storage = FileStorage::new(temp_dir.path().join("test.db")).await.unwrap();
    
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "test_content".to_string(),
        MemoryType::LongTerm,
    );
    
    // Test store
    storage.store(&entry).await.unwrap();
    
    // Test retrieve
    let retrieved = storage.retrieve("test_key").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "test_content");
    
    // Test delete
    assert!(storage.delete("test_key").await.unwrap());
    
    // Verify deletion
    let after_delete = storage.retrieve("test_key").await.unwrap();
    assert!(after_delete.is_none());
}
```

## Configuration Management

### 1. Environment-Based Configuration

**Pattern**: Use environment variables with sensible defaults.

```rust
use serde::{Deserialize, Serialize};
use config::{Config, Environment, File};

#[derive(Debug, Deserialize, Serialize)]
pub struct AppConfig {
    pub database_url: String,
    pub log_level: String,
    pub max_connections: u32,
    pub cache_size: usize,
    pub enable_metrics: bool,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let mut cfg = Config::builder()
            .add_source(File::with_name("config/default").required(false))
            .add_source(File::with_name("config/production").required(false))
            .add_source(Environment::with_prefix("SYNAPTIC"))
            .build()?;
        
        cfg.try_deserialize()
    }
}
```

These patterns form the foundation of production-ready Rust applications and are implemented throughout the Synaptic memory system to ensure reliability, performance, and maintainability in production environments.
