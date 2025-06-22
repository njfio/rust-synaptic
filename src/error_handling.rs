//! Error handling utilities and best practices for the Synaptic memory system.
//! 
//! This module provides common error handling patterns and utilities to prevent
//! the use of unwrap() calls in production code.

use crate::error::{MemoryError, Result};
use std::sync::{PoisonError, RwLockReadGuard, RwLockWriteGuard, MutexGuard};
use std::fmt::Debug;

/// Trait for safe unwrapping with context
pub trait SafeUnwrap<T> {
    /// Safely unwrap with a custom error message
    fn safe_unwrap(self, context: &str) -> Result<T>;
    
    /// Safely unwrap with a default value
    fn unwrap_or_default_safe(self) -> T where T: Default;
}

impl<T> SafeUnwrap<T> for Option<T> {
    fn safe_unwrap(self, context: &str) -> Result<T> {
        self.ok_or_else(|| MemoryError::validation(format!("Failed to unwrap Option: {}", context)))
    }

    fn unwrap_or_default_safe(self) -> T where T: Default {
        self.unwrap_or_default()
    }
}

impl<T, E: Debug> SafeUnwrap<T> for std::result::Result<T, E> {
    fn safe_unwrap(self, context: &str) -> Result<T> {
        self.map_err(|e| MemoryError::validation(format!("Failed to unwrap Result in {}: {:?}", context, e)))
    }

    fn unwrap_or_default_safe(self) -> T where T: Default {
        self.unwrap_or_default()
    }
}

/// Safe lock operations that handle poison errors gracefully
pub trait SafeLock<T> {
    /// Safely acquire a read lock
    fn safe_read(&self, context: &str) -> Result<RwLockReadGuard<T>>;
    
    /// Safely acquire a write lock
    fn safe_write(&self, context: &str) -> Result<RwLockWriteGuard<T>>;
}

impl<T> SafeLock<T> for std::sync::RwLock<T> {
    fn safe_read(&self, context: &str) -> Result<RwLockReadGuard<T>> {
        self.read().map_err(|e| {
            MemoryError::concurrency(format!("Failed to acquire read lock in {}: {}", context, e))
        })
    }

    fn safe_write(&self, context: &str) -> Result<RwLockWriteGuard<T>> {
        self.write().map_err(|e| {
            MemoryError::concurrency(format!("Failed to acquire write lock in {}: {}", context, e))
        })
    }
}

/// Safe mutex operations
pub trait SafeMutex<T> {
    /// Safely acquire a mutex lock
    fn safe_lock(&self, context: &str) -> Result<MutexGuard<T>>;
}

impl<T> SafeMutex<T> for std::sync::Mutex<T> {
    fn safe_lock(&self, context: &str) -> Result<MutexGuard<T>> {
        self.lock().map_err(|e| {
            MemoryError::concurrency(format!("Failed to acquire mutex lock in {}: {}", context, e))
        })
    }
}

/// Safe comparison operations for floating point numbers
pub trait SafeCompare {
    /// Safely compare floating point numbers, handling NaN cases
    fn safe_partial_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

impl SafeCompare for f64 {
    fn safe_partial_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl SafeCompare for f32 {
    fn safe_partial_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Safe collection operations
pub trait SafeCollection<T> {
    /// Safely get the first element
    fn safe_first(&self, context: &str) -> Result<&T>;
    
    /// Safely get the last element
    fn safe_last(&self, context: &str) -> Result<&T>;
    
    /// Safely get an element by index
    fn safe_get(&self, index: usize, context: &str) -> Result<&T>;
}

impl<T> SafeCollection<T> for Vec<T> {
    fn safe_first(&self, context: &str) -> Result<&T> {
        self.first().ok_or_else(|| {
            MemoryError::validation(format!("Empty vector in {}", context))
        })
    }

    fn safe_last(&self, context: &str) -> Result<&T> {
        self.last().ok_or_else(|| {
            MemoryError::validation(format!("Empty vector in {}", context))
        })
    }

    fn safe_get(&self, index: usize, context: &str) -> Result<&T> {
        self.get(index).ok_or_else(|| {
            MemoryError::validation(format!("Index {} out of bounds in {}", index, context))
        })
    }
}

impl<T> SafeCollection<T> for [T] {
    fn safe_first(&self, context: &str) -> Result<&T> {
        self.first().ok_or_else(|| {
            MemoryError::validation(format!("Empty slice in {}", context))
        })
    }

    fn safe_last(&self, context: &str) -> Result<&T> {
        self.last().ok_or_else(|| {
            MemoryError::validation(format!("Empty slice in {}", context))
        })
    }

    fn safe_get(&self, index: usize, context: &str) -> Result<&T> {
        self.get(index).ok_or_else(|| {
            MemoryError::validation(format!("Index {} out of bounds in {}", index, context))
        })
    }
}

/// Safe iterator operations
pub trait SafeIterator<T> {
    /// Safely find the minimum value
    fn safe_min(&mut self, context: &str) -> Result<T>;
    
    /// Safely find the maximum value
    fn safe_max(&mut self, context: &str) -> Result<T>;
}

impl<I, T> SafeIterator<T> for I 
where 
    I: Iterator<Item = T>,
    T: Ord,
{
    fn safe_min(&mut self, context: &str) -> Result<T> {
        self.min().ok_or_else(|| {
            MemoryError::validation(format!("Empty iterator in {}", context))
        })
    }

    fn safe_max(&mut self, context: &str) -> Result<T> {
        self.max().ok_or_else(|| {
            MemoryError::validation(format!("Empty iterator in {}", context))
        })
    }
}

/// Utility functions for common error scenarios
pub mod utils {
    use super::*;
    
    /// Handle poison errors by recovering the data
    pub fn handle_poison_error<T>(result: std::result::Result<T, PoisonError<T>>) -> T {
        match result {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("Lock was poisoned, recovering data");
                poisoned.into_inner()
            }
        }
    }
    
    /// Safely parse a string to a number
    pub fn safe_parse<T: std::str::FromStr>(s: &str, context: &str) -> Result<T>
    where
        T::Err: Debug,
    {
        s.parse().map_err(|e| {
            MemoryError::validation(format!("Failed to parse '{}' in {}: {:?}", s, context, e))
        })
    }

    /// Safely divide two numbers, handling division by zero
    pub fn safe_divide(numerator: f64, denominator: f64, context: &str) -> Result<f64> {
        if denominator == 0.0 {
            Err(MemoryError::validation(format!("Division by zero in {}", context)))
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Safely calculate percentage, handling edge cases
    pub fn safe_percentage(part: f64, total: f64, context: &str) -> Result<f64> {
        if total == 0.0 {
            if part == 0.0 {
                Ok(0.0) // 0/0 = 0% by convention
            } else {
                Err(MemoryError::validation(format!("Cannot calculate percentage of {} from total 0 in {}", part, context)))
            }
        } else {
            Ok((part / total) * 100.0)
        }
    }

    /// Retry an operation with exponential backoff
    pub async fn retry_with_backoff<T, F, Fut>(
        operation: F,
        max_retries: usize,
        initial_delay_ms: u64,
        context: &str,
    ) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut delay = initial_delay_ms;
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!("Operation succeeded after {} retries in {}", attempt, context);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < max_retries {
                        tracing::warn!("Operation failed (attempt {}/{}), retrying in {}ms: {}",
                                     attempt + 1, max_retries + 1, delay, context);
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                        delay *= 2; // Exponential backoff
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| MemoryError::unexpected("No error recorded during retry")))
    }

    /// Circuit breaker pattern for fault tolerance
    pub struct CircuitBreaker {
        failure_threshold: usize,
        recovery_timeout: std::time::Duration,
        failure_count: std::sync::atomic::AtomicUsize,
        last_failure_time: std::sync::Mutex<Option<std::time::Instant>>,
        state: std::sync::atomic::AtomicU8, // 0: Closed, 1: Open, 2: HalfOpen
    }

    impl CircuitBreaker {
        pub fn new(failure_threshold: usize, recovery_timeout: std::time::Duration) -> Self {
            Self {
                failure_threshold,
                recovery_timeout,
                failure_count: std::sync::atomic::AtomicUsize::new(0),
                last_failure_time: std::sync::Mutex::new(None),
                state: std::sync::atomic::AtomicU8::new(0), // Closed
            }
        }

        pub async fn call<T, F, Fut>(&self, operation: F, context: &str) -> Result<T>
        where
            F: Fn() -> Fut,
            Fut: std::future::Future<Output = Result<T>>,
        {
            use std::sync::atomic::Ordering;

            let current_state = self.state.load(Ordering::Relaxed);

            // Check if circuit is open
            if current_state == 1 {
                let last_failure = self.last_failure_time.lock().unwrap();
                if let Some(last_time) = *last_failure {
                    if last_time.elapsed() < self.recovery_timeout {
                        return Err(MemoryError::resource_exhausted(
                            format!("Circuit breaker is open for {}", context)
                        ));
                    } else {
                        // Try to transition to half-open
                        self.state.store(2, Ordering::Relaxed);
                    }
                }
            }

            match operation().await {
                Ok(result) => {
                    // Reset on success
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.state.store(0, Ordering::Relaxed); // Closed
                    Ok(result)
                }
                Err(e) => {
                    let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

                    if failures >= self.failure_threshold {
                        self.state.store(1, Ordering::Relaxed); // Open
                        *self.last_failure_time.lock().unwrap() = Some(std::time::Instant::now());
                        tracing::error!("Circuit breaker opened for {} after {} failures", context, failures);
                    }

                    Err(e)
                }
            }
        }
    }

    /// Timeout wrapper for operations
    pub async fn with_timeout<T, F, Fut>(
        operation: F,
        timeout: std::time::Duration,
        context: &str,
    ) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        match tokio::time::timeout(timeout, operation()).await {
            Ok(result) => result,
            Err(_) => Err(MemoryError::timeout(format!("Operation timed out after {:?} in {}", timeout, context))),
        }
    }

    /// Validate input parameters
    pub fn validate_non_empty_string(value: &str, field_name: &str) -> Result<()> {
        if value.trim().is_empty() {
            Err(MemoryError::validation(format!("{} cannot be empty", field_name)))
        } else {
            Ok(())
        }
    }

    /// Validate numeric ranges
    pub fn validate_range<T: PartialOrd + std::fmt::Display>(
        value: T,
        min: T,
        max: T,
        field_name: &str,
    ) -> Result<()> {
        if value < min || value > max {
            Err(MemoryError::validation(format!(
                "{} must be between {} and {}, got {}",
                field_name, min, max, value
            )))
        } else {
            Ok(())
        }
    }

    /// Validate collection size
    pub fn validate_collection_size<T>(
        collection: &[T],
        min_size: usize,
        max_size: Option<usize>,
        collection_name: &str,
    ) -> Result<()> {
        if collection.len() < min_size {
            return Err(MemoryError::validation(format!(
                "{} must contain at least {} items, got {}",
                collection_name, min_size, collection.len()
            )));
        }

        if let Some(max) = max_size {
            if collection.len() > max {
                return Err(MemoryError::validation(format!(
                    "{} must contain at most {} items, got {}",
                    collection_name, max, collection.len()
                )));
            }
        }

        Ok(())
    }
}

/// Macros for common error handling patterns
#[macro_export]
macro_rules! safe_unwrap {
    ($expr:expr, $context:expr) => {
        $expr.safe_unwrap($context)?
    };
}

#[macro_export]
macro_rules! safe_lock_read {
    ($lock:expr, $context:expr) => {
        $lock.safe_read($context)?
    };
}

#[macro_export]
macro_rules! safe_lock_write {
    ($lock:expr, $context:expr) => {
        $lock.safe_write($context)?
    };
}

#[macro_export]
macro_rules! safe_mutex_lock {
    ($mutex:expr, $context:expr) => {
        $mutex.safe_lock($context)?
    };
}

#[macro_export]
macro_rules! validate_input {
    ($condition:expr, $error_msg:expr) => {
        if !($condition) {
            return Err(MemoryError::validation($error_msg));
        }
    };
}

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $error:expr) => {
        if !($condition) {
            return Err($error);
        }
    };
}

#[macro_export]
macro_rules! try_with_context {
    ($expr:expr, $context:expr) => {
        $expr.map_err(|e| MemoryError::unexpected(format!("{}: {}", $context, e)))?
    };
}

#[macro_export]
macro_rules! log_and_return_error {
    ($error:expr, $level:ident) => {{
        tracing::$level!("Error occurred: {}", $error);
        return Err($error);
    }};
}

#[macro_export]
macro_rules! measure_time {
    ($operation:expr, $context:expr) => {{
        let start = std::time::Instant::now();
        let result = $operation;
        let duration = start.elapsed();
        tracing::debug!("Operation '{}' took {:?}", $context, duration);
        result
    }};
}

/// Error recovery strategies
pub mod recovery {
    use super::*;
    use std::collections::HashMap;

    /// Strategy for handling different types of errors
    #[derive(Debug, Clone)]
    pub enum RecoveryStrategy {
        /// Retry the operation
        Retry { max_attempts: usize, delay_ms: u64 },
        /// Use a fallback value
        Fallback(String),
        /// Skip and continue
        Skip,
        /// Fail immediately
        Fail,
    }

    /// Error recovery manager
    pub struct ErrorRecoveryManager {
        strategies: HashMap<String, RecoveryStrategy>,
    }

    impl ErrorRecoveryManager {
        pub fn new() -> Self {
            Self {
                strategies: HashMap::new(),
            }
        }

        pub fn register_strategy(&mut self, error_type: String, strategy: RecoveryStrategy) {
            self.strategies.insert(error_type, strategy);
        }

        pub async fn handle_error<T, F, Fut>(
            &self,
            error: &MemoryError,
            recovery_operation: F,
            context: &str,
        ) -> Result<Option<T>>
        where
            F: Fn() -> Fut,
            Fut: std::future::Future<Output = Result<T>>,
        {
            let error_type = self.classify_error(error);

            if let Some(strategy) = self.strategies.get(&error_type) {
                match strategy {
                    RecoveryStrategy::Retry { max_attempts, delay_ms } => {
                        tracing::info!("Attempting recovery with retry strategy for {}", context);
                        match utils::retry_with_backoff(recovery_operation, *max_attempts, *delay_ms, context).await {
                            Ok(result) => Ok(Some(result)),
                            Err(e) => {
                                tracing::error!("Recovery failed after retries: {}", e);
                                Err(e)
                            }
                        }
                    }
                    RecoveryStrategy::Fallback(fallback_msg) => {
                        tracing::warn!("Using fallback strategy for {}: {}", context, fallback_msg);
                        Ok(None) // Caller should handle fallback logic
                    }
                    RecoveryStrategy::Skip => {
                        tracing::info!("Skipping error in {}", context);
                        Ok(None)
                    }
                    RecoveryStrategy::Fail => {
                        tracing::error!("Failing immediately for error in {}", context);
                        Err(MemoryError::unexpected(format!("Recovery strategy failed for {}", context)))
                    }
                }
            } else {
                tracing::warn!("No recovery strategy found for error type: {}", error_type);
                Err(MemoryError::unexpected(format!("No recovery strategy for {}", context)))
            }
        }

        fn classify_error(&self, error: &MemoryError) -> String {
            match error {
                MemoryError::Storage { .. } => "storage".to_string(),
                MemoryError::NotFound { .. } => "not_found".to_string(),
                MemoryError::Timeout { .. } => "timeout".to_string(),
                MemoryError::Concurrency { .. } => "concurrency".to_string(),
                MemoryError::Validation { .. } => "validation".to_string(),
                MemoryError::ResourceExhausted { .. } => "resource_exhausted".to_string(),
                _ => "unknown".to_string(),
            }
        }
    }

    impl Default for ErrorRecoveryManager {
        fn default() -> Self {
            let mut manager = Self::new();

            // Register default strategies
            manager.register_strategy(
                "timeout".to_string(),
                RecoveryStrategy::Retry { max_attempts: 3, delay_ms: 1000 }
            );
            manager.register_strategy(
                "concurrency".to_string(),
                RecoveryStrategy::Retry { max_attempts: 5, delay_ms: 100 }
            );
            manager.register_strategy(
                "not_found".to_string(),
                RecoveryStrategy::Fallback("Using default value".to_string())
            );
            manager.register_strategy(
                "validation".to_string(),
                RecoveryStrategy::Fail
            );

            manager
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock, Mutex};
    
    #[test]
    fn test_safe_unwrap_option() {
        let some_value: Option<i32> = Some(42);
        let none_value: Option<i32> = None;
        
        assert_eq!(some_value.safe_unwrap("test").unwrap(), 42);
        assert!(none_value.safe_unwrap("test").is_err());
    }
    
    #[test]
    fn test_safe_compare() {
        let a = 1.0f64;
        let b = 2.0f64;
        let nan = f64::NAN;
        
        assert_eq!(a.safe_partial_cmp(&b), std::cmp::Ordering::Less);
        assert_eq!(nan.safe_partial_cmp(&a), std::cmp::Ordering::Equal);
    }
    
    #[test]
    fn test_safe_collection() {
        let vec = vec![1, 2, 3];
        let empty_vec: Vec<i32> = vec![];
        
        assert_eq!(*vec.safe_first("test").unwrap(), 1);
        assert_eq!(*vec.safe_last("test").unwrap(), 3);
        assert!(empty_vec.safe_first("test").is_err());
    }
    
    #[test]
    fn test_safe_divide() {
        assert_eq!(utils::safe_divide(10.0, 2.0, "test").unwrap(), 5.0);
        assert!(utils::safe_divide(10.0, 0.0, "test").is_err());
    }
    
    #[test]
    fn test_safe_percentage() {
        assert_eq!(utils::safe_percentage(25.0, 100.0, "test").unwrap(), 25.0);
        assert_eq!(utils::safe_percentage(0.0, 0.0, "test").unwrap(), 0.0);
        assert!(utils::safe_percentage(25.0, 0.0, "test").is_err());
    }

    #[tokio::test]
    async fn test_retry_with_backoff() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let operation = move || {
            let count = attempt_count_clone.fetch_add(1, Ordering::Relaxed) + 1;
            async move {
                if count < 3 {
                    Err(MemoryError::timeout("test operation"))
                } else {
                    Ok(42)
                }
            }
        };

        let result = utils::retry_with_backoff(operation, 5, 10, "test").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let circuit_breaker = utils::CircuitBreaker::new(2, std::time::Duration::from_millis(100));

        // First failure
        let result1 = circuit_breaker.call(|| async {
            Err::<i32, _>(MemoryError::timeout("test"))
        }, "test").await;
        assert!(result1.is_err());

        // Second failure - should open circuit
        let result2 = circuit_breaker.call(|| async {
            Err::<i32, _>(MemoryError::timeout("test"))
        }, "test").await;
        assert!(result2.is_err());

        // Third call - should be rejected due to open circuit
        let result3 = circuit_breaker.call(|| async {
            Ok(42)
        }, "test").await;
        assert!(result3.is_err());
        assert!(result3.unwrap_err().to_string().contains("Circuit breaker is open"));
    }

    #[tokio::test]
    async fn test_timeout_wrapper() {
        // Fast operation should succeed
        let result1 = utils::with_timeout(
            || async {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                Ok(42)
            },
            std::time::Duration::from_millis(100),
            "test"
        ).await;
        assert!(result1.is_ok());

        // Slow operation should timeout
        let result2 = utils::with_timeout(
            || async {
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                Ok(42)
            },
            std::time::Duration::from_millis(50),
            "test"
        ).await;
        assert!(result2.is_err());
        assert!(result2.unwrap_err().to_string().contains("timed out"));
    }

    #[test]
    fn test_validation_functions() {
        // Test non-empty string validation
        assert!(utils::validate_non_empty_string("hello", "field").is_ok());
        assert!(utils::validate_non_empty_string("", "field").is_err());
        assert!(utils::validate_non_empty_string("   ", "field").is_err());

        // Test range validation
        assert!(utils::validate_range(5, 1, 10, "value").is_ok());
        assert!(utils::validate_range(0, 1, 10, "value").is_err());
        assert!(utils::validate_range(15, 1, 10, "value").is_err());

        // Test collection size validation
        let vec = vec![1, 2, 3];
        assert!(utils::validate_collection_size(&vec, 1, Some(5), "items").is_ok());
        assert!(utils::validate_collection_size(&vec, 5, Some(10), "items").is_err());
        assert!(utils::validate_collection_size(&vec, 1, Some(2), "items").is_err());
    }

    #[tokio::test]
    async fn test_error_recovery_manager() {
        use recovery::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut manager = ErrorRecoveryManager::new();
        manager.register_strategy(
            "timeout".to_string(),
            RecoveryStrategy::Retry { max_attempts: 2, delay_ms: 10 }
        );

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let recovery_op = move || {
            let count = attempt_count_clone.fetch_add(1, Ordering::Relaxed) + 1;
            async move {
                if count < 2 {
                    Err(MemoryError::timeout("test"))
                } else {
                    Ok(42)
                }
            }
        };

        let error = MemoryError::timeout("test");
        let result = manager.handle_error(&error, recovery_op, "test").await;
        assert!(result.is_ok());
    }
}
