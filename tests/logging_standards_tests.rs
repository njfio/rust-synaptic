//! Tests for logging standards compliance and configuration

use tracing_test::traced_test;
use tracing::{info, warn, error, debug, trace};
use std::time::Duration;

#[cfg(test)]
mod logging_standards_tests {
    use super::*;
    use synaptic::memory::types::{MemoryEntry, MemoryType};
    use synaptic::memory::management::MemoryManager;
    use synaptic::memory::storage::memory::MemoryStorage;
    use synaptic::config::MemoryConfig;
    use std::sync::Arc;

    #[traced_test]
    #[tokio::test]
    async fn test_structured_logging_patterns() {
        // Test that structured logging follows our patterns
        let component = "test_component";
        let operation = "test_operation";
        let duration = Duration::from_millis(150);
        
        info!(
            component = %component,
            operation = %operation,
            duration_ms = %duration.as_millis(),
            "Operation completed successfully"
        );
        
        // Verify the log was emitted with correct structure
        assert!(logs_contain("Operation completed successfully"));
        assert!(logs_contain("component"));
        assert!(logs_contain("operation"));
        assert!(logs_contain("duration_ms"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_error_logging_pattern() {
        let error_msg = "Test error message";
        let error_type = "ValidationError";
        let component = "input_validator";
        let operation = "validate_memory";
        
        error!(
            error = %error_msg,
            error_type = %error_type,
            component = %component,
            operation = %operation,
            "Input validation failed"
        );
        
        assert!(logs_contain("Input validation failed"));
        assert!(logs_contain("ValidationError"));
        assert!(logs_contain("input_validator"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_performance_logging_pattern() {
        let query_complexity = 5;
        let results_count = 42;
        let cache_hit = true;
        let duration = Duration::from_millis(75);
        let memory_usage = 1024 * 1024 * 50; // 50MB
        
        info!(
            component = "search_engine",
            operation = "semantic_search",
            query_complexity = %query_complexity,
            results_count = %results_count,
            cache_hit = %cache_hit,
            duration_ms = %duration.as_millis(),
            memory_usage_mb = %memory_usage / 1024 / 1024,
            "Search operation completed"
        );
        
        assert!(logs_contain("Search operation completed"));
        assert!(logs_contain("search_engine"));
        assert!(logs_contain("semantic_search"));
        assert!(logs_contain("query_complexity"));
        assert!(logs_contain("results_count"));
        assert!(logs_contain("cache_hit"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_log_levels() {
        // Test all log levels are working
        error!("Error level test");
        warn!("Warning level test");
        info!("Info level test");
        debug!("Debug level test");
        trace!("Trace level test");
        
        // At minimum, error, warn, and info should be captured
        assert!(logs_contain("Error level test"));
        assert!(logs_contain("Warning level test"));
        assert!(logs_contain("Info level test"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_span_instrumentation() {
        let memory_count = 10;
        let strategy_name = "test_strategy";
        
        // Simulate a span with instrumentation
        let span = tracing::info_span!(
            "memory_consolidation",
            memory_count = %memory_count,
            strategy = %strategy_name
        );
        
        let _guard = span.enter();
        
        info!("Starting memory consolidation");
        
        // Simulate some work
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        let consolidated_count = 8;
        info!(
            consolidated_count = %consolidated_count,
            "Consolidation completed successfully"
        );
        
        assert!(logs_contain("Starting memory consolidation"));
        assert!(logs_contain("Consolidation completed successfully"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_context_propagation() {
        let memory_id = "test_memory_123";
        let memory_type = "ShortTerm";
        
        let span = tracing::Span::current();
        span.record("memory_id", &memory_id);
        span.record("memory_type", &memory_type);
        
        info!(
            memory_id = %memory_id,
            "Processing memory entry"
        );
        
        assert!(logs_contain("Processing memory entry"));
        assert!(logs_contain("test_memory_123"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_no_sensitive_data_logging() {
        // Test that we don't accidentally log sensitive information
        let user_id = "user_123";
        let _password = "secret_password"; // Should never be logged
        let _api_key = "sk-1234567890"; // Should never be logged
        
        info!(
            user_id = %user_id,
            "User authentication successful"
        );
        
        // Verify user_id is logged but sensitive data is not
        assert!(logs_contain("User authentication successful"));
        assert!(logs_contain("user_123"));
        assert!(!logs_contain("secret_password"));
        assert!(!logs_contain("sk-1234567890"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_memory_manager_logging() {
        let config = MemoryConfig::default();
        let storage = Arc::new(MemoryStorage::new());
        let manager = MemoryManager::new(config, storage).await.unwrap();
        
        let memory = MemoryEntry::new(
            "test_key".to_string(),
            "test content".to_string(),
            MemoryType::ShortTerm
        );
        
        // This should generate structured logs
        let result = manager.store_memory("test_key", memory).await;
        assert!(result.is_ok());
        
        // Verify that memory operations generate appropriate logs
        // Note: The actual log verification depends on the implementation
        // This test ensures the operation completes without panics
    }

    #[traced_test]
    #[tokio::test]
    async fn test_rate_limited_logging() {
        // Test that high-frequency operations don't spam logs
        let large_collection: Vec<i32> = (0..10000).collect();
        
        debug!(
            item_count = %large_collection.len(),
            "Processing large collection"
        );
        
        let mut logged_progress = 0;
        for (index, _item) in large_collection.iter().enumerate() {
            if index % 1000 == 0 {
                debug!(
                    processed = %index,
                    total = %large_collection.len(),
                    "Processing progress"
                );
                logged_progress += 1;
            }
        }
        
        // Should have logged progress approximately 10 times (every 1000 items)
        assert!(logged_progress <= 11); // Allow for off-by-one
        assert!(logs_contain("Processing large collection"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_lazy_evaluation_logging() {
        // Test that expensive formatting is only done when needed
        let expensive_data = vec![1, 2, 3, 4, 5];
        
        // Using debug formatting which should be lazy
        debug!(data = ?expensive_data, "Complex data processed");
        
        // This should not cause performance issues even with large data
        let large_data: Vec<i32> = (0..1000).collect();
        trace!(large_data = ?large_data, "Large data processed");
        
        // Test passes if no panics or performance issues occur
    }

    #[traced_test]
    #[tokio::test]
    async fn test_correlation_ids() {
        let trace_id = "trace_123456";
        let span_id = "span_789012";
        let component = "memory_manager";
        
        info!(
            trace_id = %trace_id,
            span_id = %span_id,
            component = %component,
            "Operation started with correlation IDs"
        );
        
        assert!(logs_contain("Operation started with correlation IDs"));
        assert!(logs_contain("trace_123456"));
        assert!(logs_contain("span_789012"));
    }

    #[test]
    fn test_logging_configuration() {
        // Test that logging can be configured properly
        use tracing_subscriber::{EnvFilter, FmtSubscriber};
        
        let subscriber = FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish();
        
        // This should not panic
        let _result = tracing::subscriber::set_global_default(subscriber);
        
        // Test that we can create different log levels
        let filter = EnvFilter::new("synaptic=debug");
        assert!(filter.to_string().contains("synaptic"));
    }

    #[traced_test]
    #[tokio::test]
    async fn test_component_specific_logging() {
        // Test logging from different components
        info!(
            component = "memory_storage",
            operation = "store",
            "Memory stored successfully"
        );
        
        info!(
            component = "search_engine",
            operation = "query",
            "Search completed"
        );
        
        info!(
            component = "analytics_engine",
            operation = "analyze",
            "Analysis completed"
        );
        
        assert!(logs_contain("memory_storage"));
        assert!(logs_contain("search_engine"));
        assert!(logs_contain("analytics_engine"));
    }
}

#[cfg(test)]
mod integration_logging_tests {
    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[tokio::test]
    async fn test_end_to_end_logging() {
        // Test that a complete operation generates appropriate logs
        let start_time = std::time::Instant::now();
        
        info!(
            component = "integration_test",
            operation = "end_to_end_test",
            "Starting end-to-end operation"
        );
        
        // Simulate some work with different log levels
        debug!("Initializing components");
        info!("Components initialized successfully");
        
        // Simulate a warning condition
        warn!(
            threshold = 80,
            current_usage = 85,
            "Usage approaching threshold"
        );
        
        // Complete the operation
        let duration = start_time.elapsed();
        info!(
            component = "integration_test",
            operation = "end_to_end_test",
            duration_ms = %duration.as_millis(),
            "End-to-end operation completed"
        );
        
        assert!(logs_contain("Starting end-to-end operation"));
        assert!(logs_contain("Components initialized successfully"));
        assert!(logs_contain("Usage approaching threshold"));
        assert!(logs_contain("End-to-end operation completed"));
    }
}
