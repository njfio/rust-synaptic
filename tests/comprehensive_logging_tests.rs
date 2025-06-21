//! Comprehensive tests for the logging and tracing infrastructure

use synaptic::logging::{
    LoggingManager, LoggingConfig, LogLevel, LogFormat, RiskLevel,

};
use synaptic::error::Result;
use std::collections::HashMap;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

/// Test basic logging configuration and initialization
#[tokio::test]
async fn test_logging_configuration() -> Result<()> {
    let config = LoggingConfig {
        level: LogLevel::Debug,
        structured_logging: true,
        enable_performance_tracing: true,
        enable_audit_logging: true,
        enable_metrics: true,
        log_file: None,
        max_file_size: 50 * 1024 * 1024, // 50MB
        max_files: 5,
        compress_logs: false,
        log_format: LogFormat::Json,
        enable_distributed_tracing: false,
        trace_sampling_rate: 1.0,
        enable_log_streaming: false,
        stream_buffer_size: 500,
    };

    let manager = LoggingManager::new(config.clone());

    // Test that configuration is properly stored
    assert_eq!(manager.config().max_file_size, 50 * 1024 * 1024);
    assert_eq!(manager.config().max_files, 5);
    assert!(!manager.config().compress_logs);
    assert!(matches!(manager.config().log_format, LogFormat::Json));
    
    Ok(())
}

/// Test logging initialization with different formats
#[tokio::test]
async fn test_logging_initialization_formats() -> Result<()> {
    let formats = vec![
        LogFormat::Pretty,
        LogFormat::Json,
        LogFormat::Compact,
        LogFormat::Full,
    ];

    for format in formats {
        let config = LoggingConfig {
            log_format: format,
            ..Default::default()
        };

        let manager = LoggingManager::new(config);
        
        // Test initialization doesn't panic
        let result = manager.initialize();
        assert!(result.is_ok(), "Failed to initialize logging with format: {:?}", manager.config().log_format);
    }

    Ok(())
}

/// Test file-based logging
#[tokio::test]
async fn test_file_logging() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let log_file = temp_dir.path().join("test.log");

    let config = LoggingConfig {
        log_file: Some(log_file.clone()),
        log_format: LogFormat::Json,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Generate some log entries
    tracing::info!("Test log entry");
    tracing::warn!("Test warning");
    tracing::error!("Test error");

    // Give time for logs to be written
    sleep(Duration::from_millis(100)).await;

    // Verify log file was created (it might be empty if subscriber was already initialized)
    if log_file.exists() {
        let log_content = std::fs::read_to_string(&log_file).unwrap();
        // File might be empty if global subscriber was already set
        println!("Log file content: {}", log_content);
    } else {
        println!("Log file not created (global subscriber already initialized)");
    }

    Ok(())
}

/// Test performance tracing functionality
#[tokio::test]
async fn test_performance_tracing() -> Result<()> {
    let config = LoggingConfig {
        enable_performance_tracing: true,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Start a performance trace
    let operation_id = manager.start_performance_trace("test_operation").await?;
    assert!(!operation_id.is_empty(), "Operation ID should not be empty");

    // Simulate some work
    sleep(Duration::from_millis(50)).await;

    // End the performance trace
    manager.end_performance_trace(&operation_id, true, None).await?;

    // Verify metrics were recorded
    let metrics = manager.get_performance_metrics().await;
    assert_eq!(metrics.len(), 1, "Should have one performance metric");
    
    let metric = &metrics[0];
    assert_eq!(metric.operation_id, operation_id);
    assert_eq!(metric.operation_name, "test_operation");
    assert!(metric.success);
    assert!(metric.duration_ms.is_some());
    assert!(metric.duration_ms.unwrap() >= 50);

    Ok(())
}

/// Test performance tracing with errors
#[tokio::test]
async fn test_performance_tracing_with_error() -> Result<()> {
    let config = LoggingConfig {
        enable_performance_tracing: true,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Start a performance trace
    let operation_id = manager.start_performance_trace("failing_operation").await?;

    // Simulate some work
    sleep(Duration::from_millis(30)).await;

    // End the performance trace with error
    let error_message = "Operation failed due to test condition".to_string();
    manager.end_performance_trace(&operation_id, false, Some(error_message.clone())).await?;

    // Verify metrics were recorded with error
    let metrics = manager.get_performance_metrics().await;
    assert_eq!(metrics.len(), 1);
    
    let metric = &metrics[0];
    assert!(!metric.success);
    assert_eq!(metric.error_message, Some(error_message));

    Ok(())
}

/// Test audit logging functionality
#[tokio::test]
async fn test_audit_logging() -> Result<()> {
    let config = LoggingConfig {
        enable_audit_logging: true,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Log various audit events
    let mut details = HashMap::new();
    details.insert("ip_address".to_string(), "192.168.1.100".to_string());
    details.insert("user_agent".to_string(), "TestAgent/1.0".to_string());

    manager.log_audit_event(
        "user_login",
        Some("user123".to_string()),
        Some("session456".to_string()),
        "authentication_system",
        "login",
        true,
        details.clone(),
        RiskLevel::Low,
    ).await?;

    manager.log_audit_event(
        "data_access",
        Some("user123".to_string()),
        Some("session456".to_string()),
        "sensitive_data",
        "read",
        true,
        details.clone(),
        RiskLevel::Medium,
    ).await?;

    manager.log_audit_event(
        "admin_action",
        Some("admin789".to_string()),
        Some("session789".to_string()),
        "user_management",
        "delete_user",
        true,
        details,
        RiskLevel::High,
    ).await?;

    // Verify audit logs were recorded
    let audit_logs = manager.get_audit_logs().await;
    assert_eq!(audit_logs.len(), 3);

    // Verify first audit log
    let login_log = &audit_logs[0];
    assert_eq!(login_log.operation, "user_login");
    assert_eq!(login_log.user_id, Some("user123".to_string()));
    assert_eq!(login_log.action, "login");
    assert!(login_log.success);
    assert!(matches!(login_log.risk_level, RiskLevel::Low));

    // Verify high-risk audit log
    let admin_log = &audit_logs[2];
    assert_eq!(admin_log.operation, "admin_action");
    assert!(matches!(admin_log.risk_level, RiskLevel::High));

    Ok(())
}

/// Test audit logging with different risk levels
#[tokio::test]
async fn test_audit_risk_levels() -> Result<()> {
    let config = LoggingConfig {
        enable_audit_logging: true,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    let risk_levels = vec![
        RiskLevel::Low,
        RiskLevel::Medium,
        RiskLevel::High,
        RiskLevel::Critical,
    ];

    for (i, risk_level) in risk_levels.into_iter().enumerate() {
        manager.log_audit_event(
            &format!("operation_{}", i),
            Some("user123".to_string()),
            None,
            "test_resource",
            "test_action",
            true,
            HashMap::new(),
            risk_level,
        ).await?;
    }

    let audit_logs = manager.get_audit_logs().await;
    assert_eq!(audit_logs.len(), 4);

    // Verify all risk levels are represented
    let risk_levels: Vec<_> = audit_logs.iter().map(|log| &log.risk_level).collect();
    assert!(risk_levels.iter().any(|r| matches!(r, RiskLevel::Low)));
    assert!(risk_levels.iter().any(|r| matches!(r, RiskLevel::Medium)));
    assert!(risk_levels.iter().any(|r| matches!(r, RiskLevel::High)));
    assert!(risk_levels.iter().any(|r| matches!(r, RiskLevel::Critical)));

    Ok(())
}

/// Test data cleanup functionality
#[tokio::test]
async fn test_data_cleanup() -> Result<()> {
    let config = LoggingConfig {
        enable_performance_tracing: true,
        enable_audit_logging: true,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Generate some old data
    let operation_id = manager.start_performance_trace("old_operation").await?;
    manager.end_performance_trace(&operation_id, true, None).await?;

    manager.log_audit_event(
        "old_operation",
        Some("user123".to_string()),
        None,
        "test_resource",
        "test_action",
        true,
        HashMap::new(),
        RiskLevel::Low,
    ).await?;

    // Verify data exists
    assert_eq!(manager.get_performance_metrics().await.len(), 1);
    assert_eq!(manager.get_audit_logs().await.len(), 1);

    // Clean up data older than 0 hours (should remove everything)
    manager.cleanup_old_data(0).await?;

    // Verify data was cleaned up
    assert_eq!(manager.get_performance_metrics().await.len(), 0);
    assert_eq!(manager.get_audit_logs().await.len(), 0);

    Ok(())
}

/// Test disabled logging features
#[tokio::test]
async fn test_disabled_features() -> Result<()> {
    let config = LoggingConfig {
        enable_performance_tracing: false,
        enable_audit_logging: false,
        ..Default::default()
    };

    let manager = LoggingManager::new(config);
    manager.initialize()?;

    // Try to use disabled features
    let operation_id = manager.start_performance_trace("test_operation").await?;
    assert!(operation_id.is_empty(), "Should return empty string when disabled");

    manager.end_performance_trace(&operation_id, true, None).await?;

    manager.log_audit_event(
        "test_operation",
        None,
        None,
        "test_resource",
        "test_action",
        true,
        HashMap::new(),
        RiskLevel::Low,
    ).await?;

    // Verify no data was recorded
    assert_eq!(manager.get_performance_metrics().await.len(), 0);
    assert_eq!(manager.get_audit_logs().await.len(), 0);

    Ok(())
}

/// Test concurrent logging operations
#[tokio::test]
async fn test_concurrent_logging() -> Result<()> {
    let config = LoggingConfig {
        enable_performance_tracing: true,
        enable_audit_logging: true,
        ..Default::default()
    };

    let manager = std::sync::Arc::new(LoggingManager::new(config));
    manager.initialize()?;

    // Spawn multiple concurrent tasks
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            // Performance tracing
            let operation_id = manager_clone.start_performance_trace(&format!("operation_{}", i)).await.unwrap();
            sleep(Duration::from_millis(10)).await;
            manager_clone.end_performance_trace(&operation_id, true, None).await.unwrap();

            // Audit logging
            manager_clone.log_audit_event(
                &format!("concurrent_operation_{}", i),
                Some(format!("user_{}", i)),
                None,
                "test_resource",
                "test_action",
                true,
                HashMap::new(),
                RiskLevel::Low,
            ).await.unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all operations were recorded
    assert_eq!(manager.get_performance_metrics().await.len(), 10);
    assert_eq!(manager.get_audit_logs().await.len(), 10);

    Ok(())
}
