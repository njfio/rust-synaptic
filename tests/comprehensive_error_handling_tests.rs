//! Comprehensive error handling tests for the Synaptic memory system

use synaptic::{
    error::{MemoryError, Result},
    error_handling::{
        SafeUnwrap, SafeLock, SafeMutex, SafeCompare, SafeCollection, SafeIterator,
        utils::{
            retry_with_backoff, CircuitBreaker, with_timeout, safe_divide, safe_percentage,
            safe_parse, validate_non_empty_string, validate_range, validate_collection_size,
            handle_poison_error
        },
        recovery::{ErrorRecoveryManager, RecoveryStrategy}
    }
};
use std::sync::{Arc, RwLock, Mutex};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_safe_unwrap_patterns() -> Result<()> {
    // Test Option unwrapping
    let some_value: Option<i32> = Some(42);
    let none_value: Option<i32> = None;
    
    assert_eq!(some_value.safe_unwrap("test option")?, 42);
    assert!(none_value.safe_unwrap("test option").is_err());
    
    // Test Result unwrapping
    let ok_result: std::result::Result<i32, &str> = Ok(42);
    let err_result: std::result::Result<i32, &str> = Err("test error");
    
    assert_eq!(ok_result.safe_unwrap("test result")?, 42);
    assert!(err_result.safe_unwrap("test result").is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_safe_lock_operations() -> Result<()> {
    let rwlock = Arc::new(RwLock::new(42));
    let mutex = Arc::new(Mutex::new(42));
    
    // Test RwLock operations
    {
        let read_guard = rwlock.safe_read("test read lock")?;
        assert_eq!(*read_guard, 42);
    }
    
    {
        let mut write_guard = rwlock.safe_write("test write lock")?;
        *write_guard = 100;
    }
    
    {
        let read_guard = rwlock.safe_read("test read after write")?;
        assert_eq!(*read_guard, 100);
    }
    
    // Test Mutex operations
    {
        let guard = mutex.safe_lock("test mutex lock")?;
        assert_eq!(*guard, 42);
    }
    
    Ok(())
}

#[test]
fn test_safe_compare_operations() {
    let a = 1.0f64;
    let b = 2.0f64;
    let nan = f64::NAN;
    
    // Normal comparisons
    assert_eq!(a.safe_partial_cmp(&b), std::cmp::Ordering::Less);
    assert_eq!(b.safe_partial_cmp(&a), std::cmp::Ordering::Greater);
    assert_eq!(a.safe_partial_cmp(&a), std::cmp::Ordering::Equal);
    
    // NaN comparisons (should default to Equal)
    assert_eq!(nan.safe_partial_cmp(&a), std::cmp::Ordering::Equal);
    assert_eq!(a.safe_partial_cmp(&nan), std::cmp::Ordering::Equal);
    
    // Test with f32
    let a32 = 1.0f32;
    let b32 = 2.0f32;
    assert_eq!(a32.safe_partial_cmp(&b32), std::cmp::Ordering::Less);
}

#[test]
fn test_safe_collection_operations() -> Result<()> {
    let vec = vec![1, 2, 3, 4, 5];
    let empty_vec: Vec<i32> = vec![];
    
    // Test successful operations
    assert_eq!(*vec.safe_first("test first")?, 1);
    assert_eq!(*vec.safe_last("test last")?, 5);
    assert_eq!(*vec.safe_get(2, "test get")?, 3);
    
    // Test error conditions
    assert!(empty_vec.safe_first("empty first").is_err());
    assert!(empty_vec.safe_last("empty last").is_err());
    assert!(vec.safe_get(10, "out of bounds").is_err());
    
    // Test with slices
    let slice = &vec[1..4];
    assert_eq!(*slice.safe_first("slice first")?, 2);
    assert_eq!(*slice.safe_last("slice last")?, 4);
    
    Ok(())
}

#[test]
fn test_safe_iterator_operations() -> Result<()> {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    
    // Test min/max operations
    let min_val = numbers.iter().cloned().safe_min("find minimum")?;
    assert_eq!(min_val, 1);
    
    let max_val = numbers.iter().cloned().safe_max("find maximum")?;
    assert_eq!(max_val, 9);
    
    // Test with empty iterator
    let empty: Vec<i32> = vec![];
    assert!(empty.iter().cloned().safe_min("empty min").is_err());
    assert!(empty.iter().cloned().safe_max("empty max").is_err());
    
    Ok(())
}

#[test]
fn test_utility_functions() -> Result<()> {
    // Test safe division
    assert_eq!(safe_divide(10.0, 2.0, "division test")?, 5.0);
    assert!(safe_divide(10.0, 0.0, "division by zero").is_err());
    
    // Test safe percentage
    assert_eq!(safe_percentage(25.0, 100.0, "percentage test")?, 25.0);
    assert_eq!(safe_percentage(0.0, 0.0, "zero percentage")?, 0.0);
    assert!(safe_percentage(25.0, 0.0, "invalid percentage").is_err());
    
    // Test safe parsing
    let parsed_int: i32 = safe_parse("42", "parse integer")?;
    assert_eq!(parsed_int, 42);
    
    let parsed_float: f64 = safe_parse("3.14", "parse float")?;
    assert_eq!(parsed_float, 3.14);
    
    assert!(safe_parse::<i32>("not_a_number", "invalid parse").is_err());
    
    Ok(())
}

#[test]
fn test_validation_functions() -> Result<()> {
    // Test string validation
    validate_non_empty_string("hello", "test string")?;
    assert!(validate_non_empty_string("", "empty string").is_err());
    assert!(validate_non_empty_string("   ", "whitespace string").is_err());
    
    // Test range validation
    validate_range(5, 1, 10, "valid range")?;
    assert!(validate_range(0, 1, 10, "below range").is_err());
    assert!(validate_range(15, 1, 10, "above range").is_err());
    
    // Test collection size validation
    let vec = vec![1, 2, 3];
    validate_collection_size(&vec, 1, Some(5), "valid collection")?;
    assert!(validate_collection_size(&vec, 5, Some(10), "too small").is_err());
    assert!(validate_collection_size(&vec, 1, Some(2), "too large").is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_retry_with_backoff() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let attempt_count = Arc::new(AtomicUsize::new(0));
    let attempt_count_clone = attempt_count.clone();
    
    // Test successful retry
    let operation = move || {
        let count = attempt_count_clone.fetch_add(1, Ordering::Relaxed) + 1;
        async move {
            if count < 3 {
                Err(MemoryError::timeout("simulated failure"))
            } else {
                Ok(42)
            }
        }
    };
    
    let result = retry_with_backoff(operation, 5, 10, "retry test").await?;
    assert_eq!(result, 42);
    assert_eq!(attempt_count.load(Ordering::Relaxed), 3);
    
    // Test retry exhaustion
    let always_fail = || async {
        Err::<i32, _>(MemoryError::timeout("always fails"))
    };
    
    let result = retry_with_backoff(always_fail, 2, 10, "exhaustion test").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_circuit_breaker() -> Result<()> {
    let circuit_breaker = CircuitBreaker::new(2, Duration::from_millis(100));
    
    // First failure
    let result1 = circuit_breaker.call(|| async {
        Err::<i32, _>(MemoryError::timeout("failure 1"))
    }, "circuit test").await;
    assert!(result1.is_err());
    
    // Second failure - should open circuit
    let result2 = circuit_breaker.call(|| async {
        Err::<i32, _>(MemoryError::timeout("failure 2"))
    }, "circuit test").await;
    assert!(result2.is_err());
    
    // Third call - should be rejected due to open circuit
    let result3 = circuit_breaker.call(|| async {
        Ok(42)
    }, "circuit test").await;
    assert!(result3.is_err());
    assert!(result3.unwrap_err().to_string().contains("Circuit breaker is open"));
    
    // Wait for recovery timeout
    sleep(Duration::from_millis(150)).await;
    
    // Should now allow calls again
    let result4 = circuit_breaker.call(|| async {
        Ok(42)
    }, "circuit test").await?;
    assert_eq!(result4, 42);
    
    Ok(())
}

#[tokio::test]
async fn test_timeout_wrapper() -> Result<()> {
    // Fast operation should succeed
    let result1 = with_timeout(
        || async {
            sleep(Duration::from_millis(10)).await;
            Ok(42)
        },
        Duration::from_millis(100),
        "fast operation"
    ).await?;
    assert_eq!(result1, 42);
    
    // Slow operation should timeout
    let result2 = with_timeout(
        || async {
            sleep(Duration::from_millis(200)).await;
            Ok(42)
        },
        Duration::from_millis(50),
        "slow operation"
    ).await;
    assert!(result2.is_err());
    assert!(result2.unwrap_err().to_string().contains("timed out"));
    
    Ok(())
}

#[tokio::test]
async fn test_error_recovery_manager() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    let mut manager = ErrorRecoveryManager::new();
    
    // Register custom strategies
    manager.register_strategy(
        "timeout".to_string(),
        RecoveryStrategy::Retry { max_attempts: 2, delay_ms: 10 }
    );
    manager.register_strategy(
        "not_found".to_string(),
        RecoveryStrategy::Fallback("default value".to_string())
    );
    manager.register_strategy(
        "validation".to_string(),
        RecoveryStrategy::Fail
    );
    
    // Test retry strategy
    let attempt_count = Arc::new(AtomicUsize::new(0));
    let attempt_count_clone = attempt_count.clone();
    
    let recovery_op = move || {
        let count = attempt_count_clone.fetch_add(1, Ordering::Relaxed) + 1;
        async move {
            if count < 2 {
                Err(MemoryError::timeout("recovery test"))
            } else {
                Ok(42)
            }
        }
    };
    
    let timeout_error = MemoryError::timeout("test");
    let result = manager.handle_error(&timeout_error, recovery_op, "timeout recovery").await?;
    assert_eq!(result, Some(42));
    
    // Test fallback strategy
    let not_found_error = MemoryError::NotFound { key: "test".to_string() };
    let result = manager.handle_error(&not_found_error, || async {
        Ok("fallback")
    }, "not found recovery").await?;
    assert_eq!(result, None); // Fallback returns None
    
    // Test fail strategy
    let validation_error = MemoryError::validation("test");
    let result = manager.handle_error(&validation_error, || async {
        Ok("should not be called")
    }, "validation recovery").await;
    assert!(result.is_err());
    
    Ok(())
}

#[test]
fn test_poison_error_handling() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = mutex.clone();
    
    // Create a poisoned mutex by panicking while holding the lock
    let handle = thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("Intentional panic to poison mutex");
    });
    
    // Wait for the thread to panic
    let _ = handle.join();
    
    // Now the mutex should be poisoned
    let poison_result = mutex.lock();
    assert!(poison_result.is_err());
    
    // Test poison error recovery
    let recovered_guard = handle_poison_error(poison_result);
    assert_eq!(*recovered_guard, 42);
}

#[test]
fn test_error_type_classification() {
    let storage_error = MemoryError::storage("test storage error");
    assert!(storage_error.is_storage_error());
    assert!(!storage_error.is_not_found());
    
    let not_found_error = MemoryError::NotFound { key: "test".to_string() };
    assert!(not_found_error.is_not_found());
    assert!(!not_found_error.is_storage_error());
    
    let serialization_error = MemoryError::Serialization(
        serde_json::from_str::<i32>("invalid json").unwrap_err()
    );
    assert!(serialization_error.is_serialization_error());
    assert!(!serialization_error.is_storage_error());
}

#[tokio::test]
async fn test_comprehensive_error_scenarios() -> Result<()> {
    // Test multiple error handling patterns together
    let data = vec![1, 2, 3, 4, 5];
    
    // Safe collection access with validation
    validate_collection_size(&data, 1, Some(10), "input data")?;
    let first = data.safe_first("getting first element")?;
    let last = data.safe_last("getting last element")?;
    
    // Safe arithmetic operations
    let ratio = safe_divide(*last as f64, *first as f64, "calculating ratio")?;
    let percentage = safe_percentage(*first as f64, data.len() as f64, "calculating percentage")?;
    
    assert_eq!(ratio, 5.0);
    assert_eq!(percentage, 20.0);
    
    // Test with timeout and retry
    let operation_with_recovery = || async {
        // Simulate an operation that might fail
        if rand::random::<f64>() < 0.3 {
            Ok("success")
        } else {
            Err(MemoryError::timeout("random failure"))
        }
    };
    
    let result = retry_with_backoff(operation_with_recovery, 5, 10, "random operation").await;
    // This might succeed or fail depending on random chance, but shouldn't panic
    match result {
        Ok(_) => println!("Operation succeeded"),
        Err(_) => println!("Operation failed after retries"),
    }
    
    Ok(())
}
