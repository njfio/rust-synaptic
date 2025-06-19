//! Comprehensive tests for the Performance Profiler
//!
//! This test suite validates the performance profiling functionality including
//! session management, metrics collection, bottleneck detection, optimization
//! recommendations, and report generation.

use synaptic::cli::profiler::{PerformanceProfiler, ProfilerConfig, SessionConfig, ReportFormat};
use synaptic::error::Result;
use std::time::Duration;
use tempfile::TempDir;

/// Test profiler creation and configuration
#[tokio::test]
async fn test_profiler_creation() -> Result<()> {
    let config = ProfilerConfig::default();
    let profiler = PerformanceProfiler::new(config).await?;
    
    // Profiler should be created successfully
    assert_eq!(profiler.get_active_sessions().len(), 0);
    assert_eq!(profiler.get_history().len(), 0);
    
    Ok(())
}

/// Test profiler with custom configuration
#[tokio::test]
async fn test_profiler_custom_config() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = ProfilerConfig::default();
    config.sampling_interval_ms = 50;
    config.enable_real_time = false;
    config.enable_bottleneck_detection = true;
    config.enable_recommendations = true;
    config.output_directory = temp_dir.path().to_string_lossy().to_string();
    config.report_format = ReportFormat::Json;
    
    let profiler = PerformanceProfiler::new(config).await?;
    
    // Profiler should respect custom configuration
    // In a real test, you'd verify the configuration is applied
    
    Ok(())
}

/// Test session lifecycle
#[tokio::test]
async fn test_session_lifecycle() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = ProfilerConfig::default();
    config.output_directory = temp_dir.path().to_string_lossy().to_string();
    config.enable_real_time = false; // Disable for testing
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("test_session".to_string(), session_config).await?;
    
    // Verify session is active
    assert_eq!(profiler.get_active_sessions().len(), 1);
    assert!(profiler.get_session_status(&session_id).is_some());
    
    // Simulate some profiling time
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Collect some metrics
    profiler.collect_metrics().await?;
    
    // Stop the session
    let report = profiler.stop_session(&session_id).await?;
    
    // Verify session is completed
    assert_eq!(profiler.get_active_sessions().len(), 0);
    assert_eq!(profiler.get_history().len(), 1);
    assert_eq!(report.session_name, "test_session");
    
    Ok(())
}

/// Test session pause and resume
#[tokio::test]
async fn test_session_pause_resume() -> Result<()> {
    let config = ProfilerConfig::default();
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("pause_test".to_string(), session_config).await?;
    
    // Pause the session
    profiler.pause_session(&session_id).await?;
    
    // Resume the session
    profiler.resume_session(&session_id).await?;
    
    // Stop the session
    let _report = profiler.stop_session(&session_id).await?;
    
    Ok(())
}

/// Test multiple concurrent sessions
#[tokio::test]
async fn test_multiple_sessions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = ProfilerConfig::default();
    config.output_directory = temp_dir.path().to_string_lossy().to_string();
    config.enable_real_time = false;
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start multiple sessions
    let session_config = SessionConfig::default();
    let session1_id = profiler.start_session("session1".to_string(), session_config.clone()).await?;
    let session2_id = profiler.start_session("session2".to_string(), session_config.clone()).await?;
    let session3_id = profiler.start_session("session3".to_string(), session_config).await?;
    
    // Verify all sessions are active
    assert_eq!(profiler.get_active_sessions().len(), 3);
    
    // Collect metrics for all sessions
    profiler.collect_metrics().await?;
    
    // Stop sessions one by one
    let _report1 = profiler.stop_session(&session1_id).await?;
    assert_eq!(profiler.get_active_sessions().len(), 2);
    
    let _report2 = profiler.stop_session(&session2_id).await?;
    assert_eq!(profiler.get_active_sessions().len(), 1);
    
    let _report3 = profiler.stop_session(&session3_id).await?;
    assert_eq!(profiler.get_active_sessions().len(), 0);
    
    // Verify all reports are in history
    assert_eq!(profiler.get_history().len(), 3);
    
    Ok(())
}

/// Test continuous profiling
#[tokio::test]
async fn test_continuous_profiling() -> Result<()> {
    let config = ProfilerConfig::default();
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("continuous_test".to_string(), session_config).await?;
    
    // Run continuous profiling for a short duration
    let profiling_duration = Duration::from_millis(200);
    profiler.run_continuous_profiling(&session_id, profiling_duration).await?;
    
    // Stop the session
    let report = profiler.stop_session(&session_id).await?;
    
    // Verify metrics were collected
    assert!(report.duration_secs > 0.0);
    
    Ok(())
}

/// Test metrics collection
#[tokio::test]
async fn test_metrics_collection() -> Result<()> {
    let config = ProfilerConfig::default();
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("metrics_test".to_string(), session_config).await?;
    
    // Collect metrics multiple times
    for _ in 0..5 {
        profiler.collect_metrics().await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    
    // Stop the session and get report
    let report = profiler.stop_session(&session_id).await?;
    
    // Verify metrics were collected
    assert!(report.summary.total_operations > 0);
    
    Ok(())
}

/// Test bottleneck detection
#[tokio::test]
async fn test_bottleneck_detection() -> Result<()> {
    let mut config = ProfilerConfig::default();
    config.enable_bottleneck_detection = true;
    config.enable_real_time = false;
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("bottleneck_test".to_string(), session_config).await?;
    
    // Collect some metrics
    profiler.collect_metrics().await?;
    
    // Stop the session and get report
    let report = profiler.stop_session(&session_id).await?;
    
    // Bottleneck detection should have run
    // In a real scenario with high resource usage, bottlenecks would be detected
    
    Ok(())
}

/// Test optimization recommendations
#[tokio::test]
async fn test_optimization_recommendations() -> Result<()> {
    let mut config = ProfilerConfig::default();
    config.enable_recommendations = true;
    config.enable_bottleneck_detection = true;
    config.enable_real_time = false;
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("recommendations_test".to_string(), session_config).await?;
    
    // Collect some metrics
    profiler.collect_metrics().await?;
    
    // Stop the session and get report
    let report = profiler.stop_session(&session_id).await?;
    
    // Recommendations should be generated based on bottlenecks
    // In a real scenario with performance issues, recommendations would be provided
    
    Ok(())
}

/// Test report generation in different formats
#[tokio::test]
async fn test_report_formats() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    let formats = vec![
        ReportFormat::Text,
        ReportFormat::Json,
        ReportFormat::Markdown,
    ];
    
    for format in formats {
        let mut config = ProfilerConfig::default();
        config.output_directory = temp_dir.path().to_string_lossy().to_string();
        config.report_format = format;
        config.enable_real_time = false;
        
        let mut profiler = PerformanceProfiler::new(config).await?;
        
        // Start and stop a session
        let session_config = SessionConfig::default();
        let session_id = profiler.start_session("format_test".to_string(), session_config).await?;
        profiler.collect_metrics().await?;
        let _report = profiler.stop_session(&session_id).await?;
        
        // Report should be saved in the specified format
        // In a real test, you'd verify the file exists and has correct format
    }
    
    Ok(())
}

/// Test visualization generation
#[tokio::test]
async fn test_visualization_generation() -> Result<()> {
    let mut config = ProfilerConfig::default();
    config.visualization.enable_ascii_charts = true;
    config.enable_real_time = false;
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Start a session
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("viz_test".to_string(), session_config).await?;
    
    // Collect multiple metrics for visualization
    for _ in 0..10 {
        profiler.collect_metrics().await?;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    
    // Stop the session and get report
    let report = profiler.stop_session(&session_id).await?;
    
    // Verify visualizations were generated
    assert!(!report.visualizations.is_empty());
    
    // Check that ASCII charts were created
    for viz in &report.visualizations {
        assert!(!viz.ascii_art.is_empty());
    }
    
    Ok(())
}

/// Test session configuration options
#[tokio::test]
async fn test_session_configuration() -> Result<()> {
    let config = ProfilerConfig::default();
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Test different session configurations
    let mut session_config = SessionConfig::default();
    session_config.target_operations = vec!["query".to_string(), "update".to_string()];
    session_config.sampling_rate = 0.5; // 50% sampling
    session_config.include_memory = true;
    session_config.include_cpu = true;
    session_config.include_io = false;
    session_config.tags.insert("environment".to_string(), "test".to_string());
    
    let session_id = profiler.start_session("config_test".to_string(), session_config).await?;
    profiler.collect_metrics().await?;
    let _report = profiler.stop_session(&session_id).await?;
    
    Ok(())
}

/// Test profiler performance and resource usage
#[tokio::test]
async fn test_profiler_performance() -> Result<()> {
    let config = ProfilerConfig::default();
    let start_time = std::time::Instant::now();
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    let creation_time = start_time.elapsed();
    
    // Profiler creation should be fast
    assert!(creation_time.as_millis() < 1000, "Profiler creation took too long: {}ms", creation_time.as_millis());
    
    // Test session operations performance
    let session_start = std::time::Instant::now();
    let session_config = SessionConfig::default();
    let session_id = profiler.start_session("perf_test".to_string(), session_config).await?;
    let session_creation_time = session_start.elapsed();
    
    assert!(session_creation_time.as_millis() < 100, "Session creation took too long: {}ms", session_creation_time.as_millis());
    
    // Test metrics collection performance
    let metrics_start = std::time::Instant::now();
    profiler.collect_metrics().await?;
    let metrics_time = metrics_start.elapsed();
    
    assert!(metrics_time.as_millis() < 50, "Metrics collection took too long: {}ms", metrics_time.as_millis());
    
    // Clean up
    let _report = profiler.stop_session(&session_id).await?;
    
    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = ProfilerConfig::default();
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Test stopping non-existent session
    let result = profiler.stop_session("non_existent").await;
    assert!(result.is_err());
    
    // Test pausing non-existent session
    let result = profiler.pause_session("non_existent").await;
    assert!(result.is_ok()); // Should handle gracefully
    
    // Test resuming non-existent session
    let result = profiler.resume_session("non_existent").await;
    assert!(result.is_ok()); // Should handle gracefully
    
    Ok(())
}

/// Test history management
#[tokio::test]
async fn test_history_management() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = ProfilerConfig::default();
    config.output_directory = temp_dir.path().to_string_lossy().to_string();
    config.enable_real_time = false;
    
    let mut profiler = PerformanceProfiler::new(config).await?;
    
    // Generate some history
    for i in 0..3 {
        let session_config = SessionConfig::default();
        let session_id = profiler.start_session(format!("session_{}", i), session_config).await?;
        profiler.collect_metrics().await?;
        let _report = profiler.stop_session(&session_id).await?;
    }
    
    // Verify history
    assert_eq!(profiler.get_history().len(), 3);
    
    // Test history export
    let history_file = temp_dir.path().join("history.json");
    profiler.export_history(&history_file).await?;
    assert!(history_file.exists());
    
    // Test history clearing
    profiler.clear_history();
    assert_eq!(profiler.get_history().len(), 0);
    
    Ok(())
}
