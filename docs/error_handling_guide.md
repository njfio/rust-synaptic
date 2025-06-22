# Error Handling Best Practices for Synaptic

This guide outlines the error handling patterns and best practices used throughout the Synaptic codebase to ensure robust, production-ready code.

## Core Principles

1. **Never use `unwrap()` in production code** - Always handle errors gracefully
2. **Use `expect()` only in tests** - With descriptive messages for debugging
3. **Prefer `Result` types** - For operations that can fail
4. **Handle poison errors** - For shared state access
5. **Provide context** - Include meaningful error messages

## Error Handling Utilities

The `error_handling` module provides utilities to replace common `unwrap()` patterns:

### Safe Unwrapping

```rust
use crate::error_handling::SafeUnwrap;

// Instead of: option.unwrap()
let value = option.safe_unwrap("parsing configuration")?;

// Instead of: result.unwrap()
let value = result.safe_unwrap("database connection")?;
```

### Safe Lock Operations

```rust
use crate::error_handling::SafeLock;

// Instead of: lock.read().unwrap()
let guard = lock.safe_read("metrics collection")?;

// Instead of: lock.write().unwrap()
let mut guard = lock.safe_write("cache update")?;
```

### Safe Comparisons

```rust
use crate::error_handling::SafeCompare;

// Instead of: a.partial_cmp(&b).unwrap()
let ordering = a.safe_partial_cmp(&b);

// For sorting:
items.sort_by(|a, b| a.safe_partial_cmp(b));
```

### Safe Collection Access

```rust
use crate::error_handling::SafeCollection;

// Instead of: vec.first().unwrap()
let first = vec.safe_first("processing results")?;

// Instead of: vec[index]
let item = vec.safe_get(index, "accessing cached data")?;
```

## Macros for Common Patterns

```rust
// Safe unwrapping
let value = safe_unwrap!(option, "parsing user input");

// Safe lock access
let guard = safe_lock_read!(metrics_lock, "reading performance metrics");
let mut guard = safe_lock_write!(cache_lock, "updating cache");
let guard = safe_mutex_lock!(state_mutex, "accessing shared state");
```

## Utility Functions

### Division and Percentage Calculations

```rust
use crate::error_handling::utils;

// Safe division
let ratio = utils::safe_divide(numerator, denominator, "calculating hit rate")?;

// Safe percentage
let percent = utils::safe_percentage(hits, total, "cache statistics")?;
```

### Parsing Operations

```rust
// Safe parsing
let number: i32 = utils::safe_parse(input, "user configuration")?;
```

## Error Context Guidelines

Always provide meaningful context in error messages:

### Good Examples

```rust
// Specific operation context
let config = file_content.safe_unwrap("loading database configuration")?;

// Include relevant identifiers
let entry = cache.safe_get(key, &format!("retrieving memory entry '{}'", key))?;

// Describe the operation being performed
let guard = lock.safe_read("collecting performance metrics for dashboard")?;
```

### Bad Examples

```rust
// Too generic
let value = option.safe_unwrap("operation failed")?;

// No context
let item = vec.safe_first("error")?;
```

## Handling Specific Error Types

### Poison Errors

```rust
use crate::error_handling::utils::handle_poison_error;

// Recover from poison errors
let guard = handle_poison_error(lock.read());
```

### Async Operations

```rust
// Handle semaphore acquisition
let permit = semaphore.acquire().await
    .map_err(|e| MemoryError::InvalidOperation(format!("Failed to acquire semaphore: {}", e)))?;

// Handle task spawning
let result = tokio::task::spawn_blocking(|| {
    // blocking work
}).await
.map_err(|e| MemoryError::InvalidOperation(format!("Background task failed: {}", e)))?;
```

### JSON Serialization

```rust
// Handle serialization errors
let json = serde_json::to_string(&data)
    .map_err(|e| MemoryError::InvalidOperation(format!("Failed to serialize data: {}", e)))?;
```

## Testing Error Conditions

Always test error conditions in your code:

```rust
#[tokio::test]
async fn test_error_handling() {
    let empty_vec: Vec<i32> = vec![];
    
    // Test that error is properly returned
    let result = empty_vec.safe_first("test operation");
    assert!(result.is_err());
    
    // Test error message contains context
    let error = result.unwrap_err();
    assert!(error.to_string().contains("test operation"));
}
```

## Migration from unwrap()

When replacing existing `unwrap()` calls:

1. **Identify the operation** - What is being unwrapped?
2. **Add context** - Why might this operation fail?
3. **Choose appropriate utility** - Use the right safe operation
4. **Handle the error** - Propagate with `?` or handle locally
5. **Test the error path** - Ensure error handling works

### Before

```rust
let first_item = items.first().unwrap();
let score_a = a.partial_cmp(&b).unwrap();
let guard = lock.read().unwrap();
```

### After

```rust
let first_item = items.safe_first("processing search results")?;
let score_a = a.safe_partial_cmp(&b);
let guard = lock.safe_read("accessing cached metrics")?;
```

## Performance Considerations

The error handling utilities add minimal overhead:

- **Safe comparisons** - No allocation, just handles NaN cases
- **Safe unwrapping** - Only allocates on error (rare case)
- **Safe locks** - Same performance as manual error handling

## Integration with Tracing

Combine error handling with structured logging:

```rust
match operation.safe_unwrap("critical system operation") {
    Ok(value) => {
        tracing::debug!("Operation succeeded");
        value
    }
    Err(e) => {
        tracing::error!("Critical operation failed: {}", e);
        return Err(e);
    }
}
```

## Summary

By following these patterns, we ensure:

- **No panics in production** - All error conditions are handled
- **Clear error messages** - Easy debugging and monitoring
- **Consistent patterns** - Maintainable codebase
- **Testable error paths** - Reliable error handling

Remember: The goal is not just to avoid panics, but to provide meaningful error information that helps with debugging and monitoring in production environments.

## Advanced Error Handling Patterns

### Retry with Exponential Backoff

For transient failures, use the retry utility:

```rust
use crate::error_handling::utils::retry_with_backoff;

// Retry operation with exponential backoff
let result = retry_with_backoff(
    || async {
        // Your operation here
        storage.store(key, value).await
    },
    max_retries: 3,
    initial_delay_ms: 100,
    "storing memory entry"
).await?;
```

### Circuit Breaker Pattern

Prevent cascading failures with circuit breakers:

```rust
use crate::error_handling::utils::CircuitBreaker;

let circuit_breaker = CircuitBreaker::new(
    failure_threshold: 5,
    recovery_timeout: Duration::from_secs(30)
);

let result = circuit_breaker.call(
    || async {
        external_service.call().await
    },
    "external service call"
).await?;
```

### Timeout Handling

Prevent hanging operations with timeouts:

```rust
use crate::error_handling::utils::with_timeout;

let result = with_timeout(
    || async {
        slow_operation().await
    },
    Duration::from_secs(30),
    "database query"
).await?;
```

### Error Recovery Strategies

Use the error recovery manager for sophisticated error handling:

```rust
use crate::error_handling::recovery::{ErrorRecoveryManager, RecoveryStrategy};

let mut recovery_manager = ErrorRecoveryManager::new();

// Register custom recovery strategies
recovery_manager.register_strategy(
    "storage".to_string(),
    RecoveryStrategy::Retry { max_attempts: 3, delay_ms: 1000 }
);

recovery_manager.register_strategy(
    "not_found".to_string(),
    RecoveryStrategy::Fallback("Using default value".to_string())
);

// Handle errors with recovery
match operation().await {
    Ok(result) => result,
    Err(error) => {
        match recovery_manager.handle_error(&error, || async {
            fallback_operation().await
        }, "critical operation").await? {
            Some(recovered_result) => recovered_result,
            None => default_value, // Use fallback
        }
    }
}
```

## Validation Patterns

### Input Validation

Always validate inputs at system boundaries:

```rust
use crate::error_handling::utils::{
    validate_non_empty_string,
    validate_range,
    validate_collection_size
};

pub async fn store_memory(key: &str, value: &str, memory_type: MemoryType) -> Result<()> {
    // Validate inputs
    validate_non_empty_string(key, "memory key")?;
    validate_non_empty_string(value, "memory value")?;
    validate_range(value.len(), 1, 1_000_000, "memory value length")?;

    // Proceed with operation
    self.storage.store(key, value).await
}
```

### Collection Validation

```rust
pub fn process_batch(entries: &[MemoryEntry]) -> Result<()> {
    validate_collection_size(entries, 1, Some(1000), "memory entries")?;

    for entry in entries {
        validate_non_empty_string(&entry.key, "entry key")?;
        validate_non_empty_string(&entry.value, "entry value")?;
    }

    // Process entries
    Ok(())
}
```

## Error Monitoring and Observability

### Structured Error Logging

```rust
use tracing::{error, warn, info, debug};

match risky_operation().await {
    Ok(result) => {
        info!(
            operation = "memory_store",
            key = %key,
            duration_ms = %duration.as_millis(),
            "Memory stored successfully"
        );
        result
    }
    Err(e) => {
        error!(
            operation = "memory_store",
            key = %key,
            error = %e,
            error_type = %std::any::type_name_of_val(&e),
            "Failed to store memory"
        );
        return Err(e);
    }
}
```

### Error Metrics Collection

```rust
use std::sync::atomic::{AtomicU64, Ordering};

pub struct ErrorMetrics {
    storage_errors: AtomicU64,
    validation_errors: AtomicU64,
    timeout_errors: AtomicU64,
}

impl ErrorMetrics {
    pub fn record_error(&self, error: &MemoryError) {
        match error {
            MemoryError::Storage { .. } => {
                self.storage_errors.fetch_add(1, Ordering::Relaxed);
            }
            MemoryError::Validation { .. } => {
                self.validation_errors.fetch_add(1, Ordering::Relaxed);
            }
            MemoryError::Timeout { .. } => {
                self.timeout_errors.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}
```

## Error Handling in Different Contexts

### CLI Error Handling

```rust
pub async fn handle_cli_command(command: &str) -> Result<()> {
    match execute_command(command).await {
        Ok(result) => {
            println!("✓ {}", result);
            Ok(())
        }
        Err(MemoryError::Validation { message }) => {
            eprintln!("❌ Invalid input: {}", message);
            Err(MemoryError::validation("CLI validation failed"))
        }
        Err(MemoryError::NotFound { key }) => {
            eprintln!("❌ Memory entry '{}' not found", key);
            Err(MemoryError::NotFound { key })
        }
        Err(e) => {
            eprintln!("❌ Operation failed: {}", e);
            Err(e)
        }
    }
}
```

### API Error Handling

```rust
use axum::{http::StatusCode, response::Json};
use serde_json::json;

pub async fn api_error_handler(error: MemoryError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, message) = match error {
        MemoryError::NotFound { key } => (
            StatusCode::NOT_FOUND,
            format!("Memory entry '{}' not found", key)
        ),
        MemoryError::Validation { message } => (
            StatusCode::BAD_REQUEST,
            format!("Validation error: {}", message)
        ),
        MemoryError::Authentication { message } => (
            StatusCode::UNAUTHORIZED,
            format!("Authentication failed: {}", message)
        ),
        MemoryError::Authorization { message } => (
            StatusCode::FORBIDDEN,
            format!("Access denied: {}", message)
        ),
        MemoryError::Timeout { operation } => (
            StatusCode::REQUEST_TIMEOUT,
            format!("Operation '{}' timed out", operation)
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Internal server error".to_string()
        ),
    };

    let body = json!({
        "error": message,
        "status": status.as_u16()
    });

    (status, Json(body))
}
```

## Testing Error Handling

### Comprehensive Error Testing

```rust
#[cfg(test)]
mod error_tests {
    use super::*;

    #[tokio::test]
    async fn test_all_error_paths() {
        // Test validation errors
        let result = store_memory("", "value", MemoryType::ShortTerm).await;
        assert!(matches!(result, Err(MemoryError::Validation { .. })));

        // Test not found errors
        let result = retrieve_memory("nonexistent").await;
        assert!(matches!(result, Err(MemoryError::NotFound { .. })));

        // Test timeout errors
        let result = with_timeout(
            || async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(())
            },
            Duration::from_millis(100),
            "test operation"
        ).await;
        assert!(matches!(result, Err(MemoryError::Timeout { .. })));
    }

    #[test]
    fn test_error_classification() {
        let storage_error = MemoryError::storage("test");
        assert!(storage_error.is_storage_error());

        let not_found_error = MemoryError::NotFound { key: "test".to_string() };
        assert!(not_found_error.is_not_found());
    }

    #[tokio::test]
    async fn test_error_recovery() {
        use crate::error_handling::recovery::*;

        let mut manager = ErrorRecoveryManager::default();
        let error = MemoryError::timeout("test");

        let result = manager.handle_error(&error, || async {
            Ok("recovered")
        }, "test operation").await;

        assert!(result.is_ok());
    }
}
```

## Best Practices Summary

1. **Always provide context** - Include operation details in error messages
2. **Use appropriate error types** - Choose the most specific error variant
3. **Implement retry logic** - For transient failures
4. **Use circuit breakers** - For external service calls
5. **Validate early** - Check inputs at system boundaries
6. **Log structured errors** - Include relevant metadata
7. **Test error paths** - Ensure error handling works correctly
8. **Monitor error rates** - Track error metrics in production
9. **Implement graceful degradation** - Provide fallback behavior
10. **Document error conditions** - Help users understand possible failures

By following these comprehensive error handling patterns, Synaptic maintains high reliability and provides excellent debugging capabilities in production environments.
