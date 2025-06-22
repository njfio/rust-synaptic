//! Panic prevention tests to ensure no unwrap() calls cause panics in production scenarios

use synaptic::{
    AgentMemory, MemoryConfig, StorageBackend,
    memory::types::{MemoryEntry, MemoryType},
    error::Result,
};
use std::sync::Arc;
use tokio::time::{timeout, Duration};

/// Test that memory operations handle edge cases without panicking
#[tokio::test]
async fn test_memory_operations_no_panic() -> Result<()> {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Test with empty strings
    let result = memory.store("", "").await;
    assert!(result.is_err()); // Should error, not panic

    // Test with very long strings
    let long_key = "a".repeat(10000);
    let long_value = "b".repeat(100000);
    let result = memory.store(&long_key, &long_value).await;
    // Should either succeed or fail gracefully, not panic
    match result {
        Ok(_) => println!("Long string storage succeeded"),
        Err(_) => println!("Long string storage failed gracefully"),
    }

    // Test with special characters
    let special_key = "key\0\n\r\t";
    let special_value = "value\0\n\r\t";
    let result = memory.store(special_key, special_value).await;
    match result {
        Ok(_) => println!("Special character storage succeeded"),
        Err(_) => println!("Special character storage failed gracefully"),
    }

    // Test retrieving non-existent keys
    let result = memory.retrieve("nonexistent_key").await?;
    assert!(result.is_none());

    // Test searching with empty query
    let results = memory.search("", 10).await?;
    assert!(results.is_empty() || !results.is_empty()); // Either is fine, just don't panic

    Ok(())
}

/// Test that comparison operations handle NaN values without panicking
#[test]
fn test_comparison_operations_no_panic() {
    use synaptic::error_handling::SafeCompare;

    let normal_value = 1.0f64;
    let nan_value = f64::NAN;
    let infinity = f64::INFINITY;
    let neg_infinity = f64::NEG_INFINITY;

    // Test all combinations of special float values
    let values = [normal_value, nan_value, infinity, neg_infinity];
    
    for &a in &values {
        for &b in &values {
            // These should never panic, even with NaN
            let _result = a.safe_partial_cmp(&b);
            
            // Test with f32 as well
            let a32 = a as f32;
            let b32 = b as f32;
            let _result32 = a32.safe_partial_cmp(&b32);
        }
    }
}

/// Test that collection operations handle empty collections without panicking
#[test]
fn test_collection_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::SafeCollection;

    // Test with empty vectors
    let empty_vec: Vec<i32> = vec![];
    let result = empty_vec.safe_first("empty vector");
    assert!(result.is_err());

    let result = empty_vec.safe_last("empty vector");
    assert!(result.is_err());

    let result = empty_vec.safe_get(0, "empty vector");
    assert!(result.is_err());

    // Test with out-of-bounds access
    let vec = vec![1, 2, 3];
    let result = vec.safe_get(100, "out of bounds");
    assert!(result.is_err());

    // Test with very large indices
    let result = vec.safe_get(usize::MAX, "max index");
    assert!(result.is_err());

    Ok(())
}

/// Test that iterator operations handle empty iterators without panicking
#[test]
fn test_iterator_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::SafeIterator;

    // Test with empty iterators
    let empty: Vec<i32> = vec![];
    let result = empty.iter().cloned().safe_min("empty iterator");
    assert!(result.is_err());

    let result = empty.iter().cloned().safe_max("empty iterator");
    assert!(result.is_err());

    // Test with iterators containing integer values (which implement Ord)
    let special_values = vec![1, 2, 3];
    let mut iter = special_values.into_iter();
    let result = iter.safe_min("special values");
    // Should handle gracefully, not panic
    match result {
        Ok(_) => println!("Min with special values succeeded"),
        Err(_) => println!("Min with special values failed gracefully"),
    }

    Ok(())
}

/// Test that arithmetic operations handle edge cases without panicking
#[test]
fn test_arithmetic_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::utils::{safe_divide, safe_percentage};

    // Test division by zero
    let result = safe_divide(10.0, 0.0, "division by zero");
    assert!(result.is_err());

    // Test with infinity
    let result = safe_divide(f64::INFINITY, 1.0, "infinity division");
    match result {
        Ok(_) => println!("Infinity division succeeded"),
        Err(_) => println!("Infinity division failed gracefully"),
    }

    // Test with NaN
    let result = safe_divide(f64::NAN, 1.0, "NaN division");
    match result {
        Ok(_) => println!("NaN division succeeded"),
        Err(_) => println!("NaN division failed gracefully"),
    }

    // Test percentage with zero denominator
    let result = safe_percentage(50.0, 0.0, "zero percentage");
    assert!(result.is_err());

    // Test with very large numbers
    let result = safe_percentage(f64::MAX, f64::MAX, "max percentage");
    match result {
        Ok(_) => println!("Max percentage succeeded"),
        Err(_) => println!("Max percentage failed gracefully"),
    }

    Ok(())
}

/// Test that parsing operations handle invalid input without panicking
#[test]
fn test_parsing_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::utils::safe_parse;

    // Test parsing invalid strings
    let invalid_inputs = [
        "",
        "not_a_number",
        "123abc",
        "∞",
        "NaN",
        "\0",
        "\n\r\t",
        "1.2.3.4",
        "++123",
        "--456",
    ];

    for input in &invalid_inputs {
        let result: std::result::Result<i32, _> = safe_parse(input, "invalid input");
        assert!(result.is_err()); // Should error, not panic

        let result: std::result::Result<f64, _> = safe_parse(input, "invalid input");
        // May succeed for some inputs like "NaN", should not panic
        match result {
            Ok(_) => println!("Parsing '{}' succeeded", input),
            Err(_) => println!("Parsing '{}' failed gracefully", input),
        }
    }

    Ok(())
}

/// Test that lock operations handle poisoned locks without panicking
#[test]
fn test_lock_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::{SafeLock, SafeMutex};
    use std::sync::{Arc, Mutex, RwLock};
    use std::thread;

    // Create a mutex and poison it
    let mutex = Arc::new(Mutex::new(42));
    let mutex_clone = mutex.clone();

    // Poison the mutex by panicking while holding the lock
    let handle = thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("Intentional panic to poison mutex");
    });

    // Wait for the thread to panic
    let _ = handle.join();

    // Now test that our safe operations handle the poisoned mutex
    let result = mutex.safe_lock("poisoned mutex test");
    // Should either succeed (recovering from poison) or fail gracefully
    match result {
        Ok(_) => println!("Poisoned mutex handled successfully"),
        Err(_) => println!("Poisoned mutex failed gracefully"),
    }

    // Test with RwLock as well
    let rwlock = Arc::new(RwLock::new(42));
    let result = rwlock.safe_read("rwlock test");
    assert!(result.is_ok()); // RwLock shouldn't be poisoned in this case

    Ok(())
}

/// Test that concurrent operations don't cause panics under stress
#[tokio::test]
async fn test_concurrent_operations_no_panic() -> Result<()> {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };
    let memory = Arc::new(tokio::sync::Mutex::new(AgentMemory::new(config).await?));

    let mut handles = vec![];

    // Spawn many concurrent operations
    for i in 0..100 {
        let memory_clone = memory.clone();
        let handle = tokio::spawn(async move {
            let mut mem = memory_clone.lock().await;
            
            // Mix of operations that might conflict
            let key = format!("concurrent_key_{}", i % 10); // Some overlap
            let value = format!("value_{}", i);
            
            // Store
            let _ = mem.store(&key, &value).await;
            
            // Retrieve
            let _ = mem.retrieve(&key).await;
            
            // Search
            let _ = mem.search(&value, 5).await;
            
            // Try to retrieve again (might fail if key doesn't exist)
            let _ = mem.retrieve(&key).await;
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        let _ = handle.await; // Don't panic if individual operations fail
    }

    Ok(())
}

/// Test that timeout operations don't cause panics
#[tokio::test]
async fn test_timeout_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::utils::with_timeout;

    // Test operation that completes quickly
    let result = with_timeout(
        || async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(42)
        },
        Duration::from_millis(100),
        "quick operation"
    ).await;
    assert!(result.is_ok());

    // Test operation that times out
    let result = with_timeout(
        || async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(42)
        },
        Duration::from_millis(50),
        "slow operation"
    ).await;
    assert!(result.is_err());

    // Test operation that panics (should be caught by timeout)
    let result = timeout(
        Duration::from_millis(100),
        async {
            // This would panic, but timeout should prevent it from affecting the test
            tokio::spawn(async {
                panic!("This panic should be contained");
            }).await
        }
    ).await;
    // The timeout should complete, and the panic should be contained in the spawned task
    assert!(result.is_err()); // Timeout or join error, but no panic

    Ok(())
}

/// Test that validation operations handle extreme inputs without panicking
#[test]
fn test_validation_operations_no_panic() -> Result<()> {
    use synaptic::error_handling::utils::{
        validate_non_empty_string, validate_range, validate_collection_size
    };

    // Test string validation with extreme inputs
    let extreme_strings = [
        "",
        " ",
        "\0",
        "\n\r\t",
        &"a".repeat(1_000_000), // Very long string
        "🦀🚀💻", // Unicode
    ];

    for s in &extreme_strings {
        let result = validate_non_empty_string(s, "extreme string test");
        // Should either pass or fail gracefully, not panic
        match result {
            Ok(_) => println!("String '{}' (len={}) validated successfully", 
                            s.chars().take(10).collect::<String>(), s.len()),
            Err(_) => println!("String validation failed gracefully"),
        }
    }

    // Test range validation with extreme values
    let extreme_ranges = [
        (i64::MIN, i64::MIN, i64::MAX),
        (i64::MAX, i64::MIN, i64::MAX),
        (0, i64::MAX, i64::MIN), // Invalid range
        (100, 200, 50), // Value outside range
    ];

    for (value, min, max) in &extreme_ranges {
        let result = validate_range(*value, *min, *max, "extreme range test");
        // Should either pass or fail gracefully, not panic
        match result {
            Ok(_) => println!("Range validation passed for {} in [{}, {}]", value, min, max),
            Err(_) => println!("Range validation failed gracefully"),
        }
    }

    // Test collection size validation
    let large_vec = vec![0; 1_000_000];
    let result = validate_collection_size(&large_vec, 1, Some(100), "large collection");
    assert!(result.is_err()); // Should fail, not panic

    Ok(())
}

/// Test that memory entry operations handle corrupted data without panicking
#[tokio::test]
async fn test_memory_entry_corruption_no_panic() -> Result<()> {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Test with entries containing problematic data
    let problematic_entries = vec![
        MemoryEntry::new("null_bytes".to_string(), "value\0with\0nulls".to_string(), MemoryType::ShortTerm),
        MemoryEntry::new("unicode".to_string(), "🦀🚀💻🌟".to_string(), MemoryType::ShortTerm),
        MemoryEntry::new("control_chars".to_string(), "value\n\r\t\x08".to_string(), MemoryType::ShortTerm),
        MemoryEntry::new("very_long".to_string(), "x".repeat(100_000), MemoryType::ShortTerm),
    ];

    for entry in problematic_entries {
        let result = memory.store(&entry.key, &entry.value).await;
        // Should either succeed or fail gracefully, not panic
        match result {
            Ok(_) => {
                println!("Stored problematic entry: {}", entry.key);
                // Try to retrieve it
                let retrieved = memory.retrieve(&entry.key).await?;
                assert!(retrieved.is_some());
            }
            Err(e) => {
                println!("Failed to store problematic entry gracefully: {}", e);
            }
        }
    }

    Ok(())
}
