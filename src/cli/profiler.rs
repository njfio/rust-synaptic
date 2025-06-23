//! Performance Profiling Tools for Synaptic CLI
//!
//! This module implements comprehensive performance profiling capabilities including
//! real-time monitoring, bottleneck identification, optimization recommendations,
//! and detailed metrics collection with visualization.

use crate::error::Result;
use crate::performance::metrics::{PerformanceMetrics, MetricsCollector};
use crate::performance::real_time_monitoring::{RealTimePerformanceMonitor, RealTimeMonitoringConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use tokio::time::interval;

/// Performance profiler for CLI operations
pub struct PerformanceProfiler {
    /// Metrics collector
    metrics_collector: MetricsCollector,
    /// Real-time monitor
    real_time_monitor: RealTimePerformanceMonitor,
    /// Profiling configuration
    config: ProfilerConfig,
    /// Active profiling sessions
    active_sessions: HashMap<String, ProfilingSession>,
    /// Historical profiling data
    history: Vec<ProfileReport>,
}

/// Profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Sampling interval in milliseconds
    pub sampling_interval_ms: u64,
    /// Maximum session duration in seconds
    pub max_session_duration_secs: u64,
    /// Enable real-time monitoring
    pub enable_real_time: bool,
    /// Enable bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Enable optimization recommendations
    pub enable_recommendations: bool,
    /// Output directory for reports
    pub output_directory: String,
    /// Report format
    pub report_format: ReportFormat,
    /// Visualization settings
    pub visualization: VisualizationConfig,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable ASCII charts
    pub enable_ascii_charts: bool,
    /// Chart width
    pub chart_width: usize,
    /// Chart height
    pub chart_height: usize,
    /// Enable color output
    pub enable_colors: bool,
    /// Show percentiles
    pub show_percentiles: bool,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Text,
    Json,
    Html,
    Csv,
    Markdown,
}

/// Profiling session
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    /// Session ID
    pub id: String,
    /// Session name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// Start timestamp
    pub start_timestamp: DateTime<Utc>,
    /// Collected metrics
    pub metrics: Vec<PerformanceMetrics>,
    /// Session configuration
    pub config: SessionConfig,
    /// Current status
    pub status: SessionStatus,
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Target operations to profile
    pub target_operations: Vec<String>,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Include memory profiling
    pub include_memory: bool,
    /// Include CPU profiling
    pub include_cpu: bool,
    /// Include I/O profiling
    pub include_io: bool,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

/// Session status
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Paused,
    Completed,
    Failed(String),
}

/// Profile report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Report ID
    pub id: String,
    /// Session name
    pub session_name: String,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Profiling duration
    pub duration_secs: f64,
    /// Summary statistics
    pub summary: ProfileSummary,
    /// Bottlenecks identified
    pub bottlenecks: Vec<Bottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<Recommendation>,
    /// Detailed metrics
    pub detailed_metrics: DetailedMetrics,
    /// Visualizations
    pub visualizations: Vec<Visualization>,
}

/// Profile summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    /// Total operations
    pub total_operations: u64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Peak latency
    pub peak_latency_ms: f64,
    /// Throughput (ops/sec)
    pub throughput: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// Error rate
    pub error_rate: f64,
    /// Performance score
    pub performance_score: f64,
}

/// Bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Impact assessment
    pub impact: Impact,
    /// Location information
    pub location: Location,
    /// Suggested fixes
    pub suggested_fixes: Vec<String>,
}

/// Bottleneck types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    Cpu,
    Memory,
    Io,
    Network,
    Database,
    Algorithm,
    Concurrency,
    Cache,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impact {
    /// Performance degradation percentage
    pub performance_degradation: f64,
    /// Affected operations
    pub affected_operations: Vec<String>,
    /// Resource waste
    pub resource_waste: f64,
    /// User experience impact
    pub user_experience_impact: String,
}

/// Location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    /// Component name
    pub component: String,
    /// Function or method
    pub function: Option<String>,
    /// File path
    pub file_path: Option<String>,
    /// Line number
    pub line_number: Option<u32>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: Priority,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: ExpectedImprovement,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
    /// Code examples
    pub code_examples: Vec<CodeExample>,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    AlgorithmOptimization,
    CacheImprovement,
    MemoryOptimization,
    ConcurrencyEnhancement,
    DatabaseOptimization,
    NetworkOptimization,
    ConfigurationTuning,
    ArchitecturalChange,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Expected improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    /// Performance gain percentage
    pub performance_gain: f64,
    /// Latency reduction
    pub latency_reduction_ms: f64,
    /// Throughput increase
    pub throughput_increase: f64,
    /// Resource savings
    pub resource_savings: f64,
}

/// Implementation effort
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationEffort {
    /// Estimated hours
    pub estimated_hours: f64,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Required skills
    pub required_skills: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Simple,
    Moderate,
    Complex,
    Expert,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Programming language
    pub language: String,
    /// Code snippet
    pub code: String,
    /// Explanation
    pub explanation: String,
}

/// Detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMetrics {
    /// Latency distribution
    pub latency_distribution: LatencyDistribution,
    /// Throughput over time
    pub throughput_timeline: Vec<TimePoint>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Operation breakdown
    pub operation_breakdown: HashMap<String, OperationMetrics>,
    /// Error analysis
    pub error_analysis: ErrorAnalysis,
}

/// Latency distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistribution {
    /// Percentiles
    pub percentiles: HashMap<String, f64>,
    /// Histogram buckets
    pub histogram: Vec<HistogramBucket>,
    /// Statistical measures
    pub statistics: StatisticalMeasures,
}

/// Histogram bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Count
    pub count: u64,
    /// Frequency
    pub frequency: f64,
}

/// Statistical measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMeasures {
    /// Mean
    pub mean: f64,
    /// Median
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

/// Time point for timeline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage over time
    pub cpu_timeline: Vec<TimePoint>,
    /// Memory usage over time
    pub memory_timeline: Vec<TimePoint>,
    /// I/O operations over time
    pub io_timeline: Vec<TimePoint>,
    /// Network usage over time
    pub network_timeline: Vec<TimePoint>,
}

/// Operation-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Operation name
    pub operation_name: String,
    /// Call count
    pub call_count: u64,
    /// Total time
    pub total_time_ms: f64,
    /// Average time
    pub avg_time_ms: f64,
    /// Min/max times
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// Resource consumption
    pub resource_consumption: HashMap<String, f64>,
}

/// Error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    /// Total errors
    pub total_errors: u64,
    /// Error rate
    pub error_rate: f64,
    /// Error types
    pub error_types: HashMap<String, u64>,
    /// Error timeline
    pub error_timeline: Vec<TimePoint>,
    /// Top error messages
    pub top_errors: Vec<ErrorInfo>,
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error message
    pub message: String,
    /// Occurrence count
    pub count: u64,
    /// First occurrence
    pub first_seen: DateTime<Utc>,
    /// Last occurrence
    pub last_seen: DateTime<Utc>,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
}

/// Visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Visualization {
    /// Visualization type
    pub viz_type: VisualizationType,
    /// Title
    pub title: String,
    /// ASCII art representation
    pub ascii_art: String,
    /// Data points
    pub data_points: Vec<DataPoint>,
    /// Configuration
    pub config: VisualizationConfig,
}

/// Visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    Histogram,
    HeatMap,
    ScatterPlot,
    Timeline,
}

/// Data point for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Label
    pub label: Option<String>,
    /// Color
    pub color: Option<String>,
    /// Size
    pub size: Option<f64>,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sampling_interval_ms: 100,
            max_session_duration_secs: 3600, // 1 hour
            enable_real_time: true,
            enable_bottleneck_detection: true,
            enable_recommendations: true,
            output_directory: "./profiling_reports".to_string(),
            report_format: ReportFormat::Text,
            visualization: VisualizationConfig::default(),
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_ascii_charts: true,
            chart_width: 80,
            chart_height: 20,
            enable_colors: true,
            show_percentiles: true,
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            target_operations: vec!["*".to_string()], // Profile all operations
            sampling_rate: 1.0, // 100% sampling
            include_memory: true,
            include_cpu: true,
            include_io: true,
            tags: HashMap::new(),
        }
    }
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub async fn new(config: ProfilerConfig) -> Result<Self> {
        let performance_config = crate::performance::PerformanceConfig::default();
        let metrics_collector = MetricsCollector::new(performance_config).await?;

        let monitoring_config = RealTimeMonitoringConfig {
            collection_interval_ms: config.sampling_interval_ms,
            enable_alerting: false, // Disable alerts for profiling
            enable_predictive_monitoring: false,
            ..Default::default()
        };
        let real_time_monitor = RealTimePerformanceMonitor::new(monitoring_config).await?;

        Ok(Self {
            metrics_collector,
            real_time_monitor,
            config,
            active_sessions: HashMap::new(),
            history: Vec::new(),
        })
    }

    /// Start a new profiling session
    pub async fn start_session(&mut self, name: String, session_config: SessionConfig) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = ProfilingSession {
            id: session_id.clone(),
            name: name.clone(),
            start_time: Instant::now(),
            start_timestamp: Utc::now(),
            metrics: Vec::new(),
            config: session_config,
            status: SessionStatus::Active,
        };

        self.active_sessions.insert(session_id.clone(), session);

        // Start real-time monitoring if enabled
        if self.config.enable_real_time {
            self.real_time_monitor.start_monitoring().await?;
        }

        tracing::info!("Started profiling session '{}' (ID: {})", name, session_id);
        Ok(session_id)
    }

    /// Stop a profiling session and generate report
    pub async fn stop_session(&mut self, session_id: &str) -> Result<ProfileReport> {
        let mut session = self.active_sessions.remove(session_id)
            .ok_or_else(|| crate::error::MemoryError::InvalidConfiguration {
                message: format!("Session not found: {}", session_id),
            })?;

        session.status = SessionStatus::Completed;
        let duration = session.start_time.elapsed();

        // Stop real-time monitoring if no other sessions are active
        if self.active_sessions.is_empty() && self.config.enable_real_time {
            self.real_time_monitor.stop_monitoring().await?;
        }

        // Generate comprehensive report
        let report = self.generate_report(&session, duration).await?;

        // Save report to history
        self.history.push(report.clone());

        // Save report to file
        self.save_report(&report).await?;

        tracing::info!("Completed profiling session '{}' in {:.2}s", session.name, duration.as_secs_f64());
        Ok(report)
    }

    /// Pause a profiling session
    pub async fn pause_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.status = SessionStatus::Paused;
            tracing::info!("Paused profiling session '{}'", session.name);
        }
        Ok(())
    }

    /// Resume a paused profiling session
    pub async fn resume_session(&mut self, session_id: &str) -> Result<()> {
        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.status = SessionStatus::Active;
            tracing::info!("Resumed profiling session '{}'", session.name);
        }
        Ok(())
    }

    /// Get active sessions
    pub fn get_active_sessions(&self) -> Vec<&ProfilingSession> {
        self.active_sessions.values().collect()
    }

    /// Get session status
    pub fn get_session_status(&self, session_id: &str) -> Option<&SessionStatus> {
        self.active_sessions.get(session_id).map(|s| &s.status)
    }

    /// Collect metrics for active sessions
    pub async fn collect_metrics(&mut self) -> Result<()> {
        if self.active_sessions.is_empty() {
            return Ok(());
        }

        // Collect current metrics
        let current_metrics = self.metrics_collector.get_current_metrics().await?;

        // Add metrics to all active sessions
        for session in self.active_sessions.values_mut() {
            if session.status == SessionStatus::Active {
                session.metrics.push(current_metrics.clone());
            }
        }

        Ok(())
    }

    /// Run continuous profiling for a specified duration
    pub async fn run_continuous_profiling(&mut self, session_id: &str, duration: Duration) -> Result<()> {
        let mut interval = interval(Duration::from_millis(self.config.sampling_interval_ms));
        let end_time = Instant::now() + duration;

        while Instant::now() < end_time {
            interval.tick().await;

            // Check if session is still active
            if let Some(session) = self.active_sessions.get(session_id) {
                if session.status != SessionStatus::Active {
                    break;
                }
            } else {
                break;
            }

            // Collect metrics
            self.collect_metrics().await?;
        }

        Ok(())
    }

    /// Generate comprehensive performance report
    async fn generate_report(&self, session: &ProfilingSession, duration: Duration) -> Result<ProfileReport> {
        let report_id = uuid::Uuid::new_v4().to_string();

        // Calculate summary statistics
        let summary = self.calculate_summary_statistics(&session.metrics, duration).await?;

        // Identify bottlenecks
        let bottlenecks = if self.config.enable_bottleneck_detection {
            self.identify_bottlenecks(&session.metrics).await?
        } else {
            Vec::new()
        };

        // Generate recommendations
        let recommendations = if self.config.enable_recommendations {
            self.generate_recommendations(&summary, &bottlenecks).await?
        } else {
            Vec::new()
        };

        // Create detailed metrics
        let detailed_metrics = self.create_detailed_metrics(&session.metrics).await?;

        // Generate visualizations
        let visualizations = self.generate_visualizations(&session.metrics, &detailed_metrics).await?;

        Ok(ProfileReport {
            id: report_id,
            session_name: session.name.clone(),
            generated_at: Utc::now(),
            duration_secs: duration.as_secs_f64(),
            summary,
            bottlenecks,
            recommendations,
            detailed_metrics,
            visualizations,
        })
    }

    /// Calculate summary statistics
    async fn calculate_summary_statistics(&self, metrics: &[PerformanceMetrics], _duration: Duration) -> Result<ProfileSummary> {
        if metrics.is_empty() {
            return Ok(ProfileSummary {
                total_operations: 0,
                avg_latency_ms: 0.0,
                peak_latency_ms: 0.0,
                throughput: 0.0,
                cpu_utilization: 0.0,
                memory_usage_mb: 0.0,
                error_rate: 0.0,
                performance_score: 0.0,
            });
        }

        let total_operations = metrics.len() as u64;
        let avg_latency_ms = metrics.iter().map(|m| m.avg_latency_ms).sum::<f64>() / metrics.len() as f64;
        let peak_latency_ms = metrics.iter().map(|m| m.avg_latency_ms).fold(0.0, f64::max);
        let throughput = metrics.iter().map(|m| m.throughput_ops_per_sec).sum::<f64>() / metrics.len() as f64;
        let cpu_utilization = metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / metrics.len() as f64;
        let memory_usage_mb = metrics.iter().map(|m| m.memory_usage_percent).sum::<f64>() / metrics.len() as f64;
        let error_rate = metrics.iter().map(|m| m.error_rate).sum::<f64>() / metrics.len() as f64;

        // Calculate performance score (0-100, higher is better)
        let performance_score = self.calculate_performance_score(
            avg_latency_ms,
            throughput,
            cpu_utilization,
            memory_usage_mb,
            error_rate,
        );

        Ok(ProfileSummary {
            total_operations,
            avg_latency_ms,
            peak_latency_ms,
            throughput,
            cpu_utilization,
            memory_usage_mb,
            error_rate,
            performance_score,
        })
    }

    /// Calculate performance score
    fn calculate_performance_score(&self, latency: f64, throughput: f64, cpu: f64, memory: f64, error_rate: f64) -> f64 {
        // Weighted scoring algorithm
        let latency_score = (100.0 - (latency / 10.0).min(100.0)).max(0.0);
        let throughput_score = (throughput / 100.0).min(100.0);
        let cpu_score = (100.0 - cpu).max(0.0);
        let memory_score = (100.0 - memory).max(0.0);
        let error_score = (100.0 - (error_rate * 1000.0)).max(0.0);

        // Weighted average
        latency_score * 0.3 + throughput_score * 0.2 + cpu_score * 0.2 + memory_score * 0.2 + error_score * 0.1
    }

    /// Identify performance bottlenecks
    async fn identify_bottlenecks(&self, metrics: &[PerformanceMetrics]) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();

        if metrics.is_empty() {
            return Ok(bottlenecks);
        }

        // Analyze CPU bottlenecks
        let avg_cpu = metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / metrics.len() as f64;
        if avg_cpu > 80.0 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Cpu,
                severity: if avg_cpu > 95.0 { Severity::Critical } else if avg_cpu > 90.0 { Severity::High } else { Severity::Medium },
                description: format!("High CPU utilization detected: {:.1}%", avg_cpu),
                impact: Impact {
                    performance_degradation: (avg_cpu - 50.0).max(0.0),
                    affected_operations: vec!["All operations".to_string()],
                    resource_waste: avg_cpu - 70.0,
                    user_experience_impact: "Slow response times".to_string(),
                },
                location: Location {
                    component: "CPU".to_string(),
                    function: None,
                    file_path: None,
                    line_number: None,
                },
                suggested_fixes: vec![
                    "Optimize CPU-intensive algorithms".to_string(),
                    "Implement parallel processing".to_string(),
                    "Scale horizontally".to_string(),
                ],
            });
        }

        // Analyze memory bottlenecks
        let avg_memory = metrics.iter().map(|m| m.memory_usage_percent).sum::<f64>() / metrics.len() as f64;
        if avg_memory > 85.0 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Memory,
                severity: if avg_memory > 95.0 { Severity::Critical } else { Severity::High },
                description: format!("High memory utilization detected: {:.1}%", avg_memory),
                impact: Impact {
                    performance_degradation: (avg_memory - 60.0).max(0.0),
                    affected_operations: vec!["Memory-intensive operations".to_string()],
                    resource_waste: avg_memory - 80.0,
                    user_experience_impact: "Potential out-of-memory errors".to_string(),
                },
                location: Location {
                    component: "Memory".to_string(),
                    function: None,
                    file_path: None,
                    line_number: None,
                },
                suggested_fixes: vec![
                    "Implement memory pooling".to_string(),
                    "Optimize data structures".to_string(),
                    "Add memory limits".to_string(),
                ],
            });
        }

        // Analyze latency bottlenecks
        let avg_latency = metrics.iter().map(|m| m.avg_latency_ms).sum::<f64>() / metrics.len() as f64;
        if avg_latency > 100.0 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::Algorithm,
                severity: if avg_latency > 1000.0 { Severity::Critical } else if avg_latency > 500.0 { Severity::High } else { Severity::Medium },
                description: format!("High latency detected: {:.1}ms", avg_latency),
                impact: Impact {
                    performance_degradation: (avg_latency / 10.0).min(100.0),
                    affected_operations: vec!["All operations".to_string()],
                    resource_waste: 0.0,
                    user_experience_impact: "Poor responsiveness".to_string(),
                },
                location: Location {
                    component: "Application".to_string(),
                    function: None,
                    file_path: None,
                    line_number: None,
                },
                suggested_fixes: vec![
                    "Optimize algorithms".to_string(),
                    "Implement caching".to_string(),
                    "Use asynchronous processing".to_string(),
                ],
            });
        }

        Ok(bottlenecks)
    }

    /// Generate optimization recommendations
    async fn generate_recommendations(&self, summary: &ProfileSummary, bottlenecks: &[Bottleneck]) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Generate recommendations based on bottlenecks
        for bottleneck in bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::Cpu => {
                    recommendations.push(Recommendation {
                        recommendation_type: RecommendationType::AlgorithmOptimization,
                        priority: Priority::High,
                        title: "Optimize CPU-intensive operations".to_string(),
                        description: "Reduce CPU utilization by optimizing algorithms and implementing parallel processing".to_string(),
                        expected_improvement: ExpectedImprovement {
                            performance_gain: 25.0,
                            latency_reduction_ms: summary.avg_latency_ms * 0.2,
                            throughput_increase: summary.throughput * 0.15,
                            resource_savings: 20.0,
                        },
                        implementation_effort: ImplementationEffort {
                            estimated_hours: 16.0,
                            complexity: ComplexityLevel::Moderate,
                            required_skills: vec!["Algorithm optimization".to_string(), "Parallel programming".to_string()],
                            dependencies: vec!["Profiling tools".to_string()],
                        },
                        code_examples: vec![
                            CodeExample {
                                title: "Parallel processing with Rayon".to_string(),
                                language: "rust".to_string(),
                                code: r#"use rayon::prelude::*;

// Before: Sequential processing
let results: Vec<_> = data.iter().map(|item| process_item(item)).collect();

// After: Parallel processing
let results: Vec<_> = data.par_iter().map(|item| process_item(item)).collect();"#.to_string(),
                                explanation: "Use parallel iterators to utilize multiple CPU cores".to_string(),
                            }
                        ],
                    });
                },
                BottleneckType::Memory => {
                    recommendations.push(Recommendation {
                        recommendation_type: RecommendationType::MemoryOptimization,
                        priority: Priority::High,
                        title: "Implement memory optimization strategies".to_string(),
                        description: "Reduce memory usage through pooling, efficient data structures, and garbage collection tuning".to_string(),
                        expected_improvement: ExpectedImprovement {
                            performance_gain: 20.0,
                            latency_reduction_ms: summary.avg_latency_ms * 0.1,
                            throughput_increase: summary.throughput * 0.1,
                            resource_savings: 30.0,
                        },
                        implementation_effort: ImplementationEffort {
                            estimated_hours: 12.0,
                            complexity: ComplexityLevel::Moderate,
                            required_skills: vec!["Memory management".to_string(), "Data structures".to_string()],
                            dependencies: vec!["Memory profiler".to_string()],
                        },
                        code_examples: vec![
                            CodeExample {
                                title: "Object pooling pattern".to_string(),
                                language: "rust".to_string(),
                                code: r#"use std::sync::Mutex;
use std::collections::VecDeque;

struct ObjectPool<T> {
    objects: Mutex<VecDeque<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> ObjectPool<T> {
    fn get(&self) -> T {
        self.objects.lock().unwrap().pop_front()
            .unwrap_or_else(|| (self.factory)())
    }

    fn return_object(&self, obj: T) {
        self.objects.lock().unwrap().push_back(obj);
    }
}"#.to_string(),
                                explanation: "Reuse objects to reduce allocation overhead".to_string(),
                            }
                        ],
                    });
                },
                BottleneckType::Algorithm => {
                    recommendations.push(Recommendation {
                        recommendation_type: RecommendationType::CacheImprovement,
                        priority: Priority::Medium,
                        title: "Implement intelligent caching".to_string(),
                        description: "Add caching layers to reduce computation and I/O overhead".to_string(),
                        expected_improvement: ExpectedImprovement {
                            performance_gain: 40.0,
                            latency_reduction_ms: summary.avg_latency_ms * 0.4,
                            throughput_increase: summary.throughput * 0.3,
                            resource_savings: 25.0,
                        },
                        implementation_effort: ImplementationEffort {
                            estimated_hours: 8.0,
                            complexity: ComplexityLevel::Simple,
                            required_skills: vec!["Caching strategies".to_string()],
                            dependencies: vec!["Cache library".to_string()],
                        },
                        code_examples: vec![
                            CodeExample {
                                title: "LRU Cache implementation".to_string(),
                                language: "rust".to_string(),
                                code: r#"use lru::LruCache;
use std::sync::Mutex;

struct CachedService {
    cache: Mutex<LruCache<String, String>>,
}

impl CachedService {
    fn get_data(&self, key: &str) -> Result<String, Box<dyn std::error::Error>> {
        use synaptic::error_handling::SafeMutex;
        let mut cache = self.cache.safe_lock("LRU cache access")?;

        if let Some(value) = cache.get(key) {
            return Ok(value.clone());
        }

        let value = expensive_computation(key);
        cache.put(key.to_string(), value.clone());
        Ok(value)
    }
}"#.to_string(),
                                explanation: "Cache expensive computations to avoid redundant work".to_string(),
                            }
                        ],
                    });
                },
                _ => {
                    // Generic recommendation for other bottleneck types
                    recommendations.push(Recommendation {
                        recommendation_type: RecommendationType::ConfigurationTuning,
                        priority: Priority::Low,
                        title: "Review system configuration".to_string(),
                        description: "Analyze and tune system configuration parameters".to_string(),
                        expected_improvement: ExpectedImprovement {
                            performance_gain: 10.0,
                            latency_reduction_ms: summary.avg_latency_ms * 0.05,
                            throughput_increase: summary.throughput * 0.05,
                            resource_savings: 10.0,
                        },
                        implementation_effort: ImplementationEffort {
                            estimated_hours: 4.0,
                            complexity: ComplexityLevel::Simple,
                            required_skills: vec!["System administration".to_string()],
                            dependencies: vec![],
                        },
                        code_examples: vec![],
                    });
                }
            }
        }

        // Add general recommendations based on performance score
        if summary.performance_score < 50.0 {
            recommendations.push(Recommendation {
                recommendation_type: RecommendationType::ArchitecturalChange,
                priority: Priority::Critical,
                title: "Consider architectural improvements".to_string(),
                description: "The overall performance score is low. Consider major architectural changes".to_string(),
                expected_improvement: ExpectedImprovement {
                    performance_gain: 100.0,
                    latency_reduction_ms: summary.avg_latency_ms * 0.5,
                    throughput_increase: summary.throughput * 0.8,
                    resource_savings: 40.0,
                },
                implementation_effort: ImplementationEffort {
                    estimated_hours: 80.0,
                    complexity: ComplexityLevel::Expert,
                    required_skills: vec!["Architecture design".to_string(), "System design".to_string()],
                    dependencies: vec!["Team consensus".to_string(), "Budget approval".to_string()],
                },
                code_examples: vec![],
            });
        }

        Ok(recommendations)
    }

    /// Create detailed metrics analysis
    async fn create_detailed_metrics(&self, metrics: &[PerformanceMetrics]) -> Result<DetailedMetrics> {
        if metrics.is_empty() {
            return Ok(DetailedMetrics {
                latency_distribution: LatencyDistribution {
                    percentiles: HashMap::new(),
                    histogram: Vec::new(),
                    statistics: StatisticalMeasures {
                        mean: 0.0,
                        median: 0.0,
                        std_dev: 0.0,
                        variance: 0.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                    },
                },
                throughput_timeline: Vec::new(),
                resource_utilization: ResourceUtilization {
                    cpu_timeline: Vec::new(),
                    memory_timeline: Vec::new(),
                    io_timeline: Vec::new(),
                    network_timeline: Vec::new(),
                },
                operation_breakdown: HashMap::new(),
                error_analysis: ErrorAnalysis {
                    total_errors: 0,
                    error_rate: 0.0,
                    error_types: HashMap::new(),
                    error_timeline: Vec::new(),
                    top_errors: Vec::new(),
                },
            });
        }

        // Calculate latency distribution
        let latencies: Vec<f64> = metrics.iter().map(|m| m.avg_latency_ms).collect();
        let latency_distribution = self.calculate_latency_distribution(&latencies);

        // Create timeline data
        let throughput_timeline: Vec<TimePoint> = metrics.iter().enumerate().map(|(i, m)| {
            TimePoint {
                timestamp: Utc::now() - chrono::Duration::seconds((metrics.len() - i) as i64),
                value: m.throughput_ops_per_sec,
                metadata: HashMap::new(),
            }
        }).collect();

        // Create resource utilization timelines
        let cpu_timeline: Vec<TimePoint> = metrics.iter().enumerate().map(|(i, m)| {
            TimePoint {
                timestamp: Utc::now() - chrono::Duration::seconds((metrics.len() - i) as i64),
                value: m.cpu_usage_percent,
                metadata: HashMap::new(),
            }
        }).collect();

        let memory_timeline: Vec<TimePoint> = metrics.iter().enumerate().map(|(i, m)| {
            TimePoint {
                timestamp: Utc::now() - chrono::Duration::seconds((metrics.len() - i) as i64),
                value: m.memory_usage_percent,
                metadata: HashMap::new(),
            }
        }).collect();

        let resource_utilization = ResourceUtilization {
            cpu_timeline,
            memory_timeline,
            io_timeline: Vec::new(), // Placeholder
            network_timeline: Vec::new(), // Placeholder
        };

        // Create operation breakdown (simplified)
        let mut operation_breakdown = HashMap::new();
        operation_breakdown.insert("query_execution".to_string(), OperationMetrics {
            operation_name: "Query Execution".to_string(),
            call_count: metrics.len() as u64,
            total_time_ms: latencies.iter().sum(),
            avg_time_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            min_time_ms: latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            success_rate: 1.0 - (metrics.iter().map(|m| m.error_rate).sum::<f64>() / metrics.len() as f64),
            resource_consumption: HashMap::new(),
        });

        // Create error analysis
        let total_errors = metrics.iter().map(|m| (m.error_rate * 1000.0) as u64).sum();
        let error_rate = metrics.iter().map(|m| m.error_rate).sum::<f64>() / metrics.len() as f64;

        let error_analysis = ErrorAnalysis {
            total_errors,
            error_rate,
            error_types: HashMap::new(), // Placeholder
            error_timeline: Vec::new(), // Placeholder
            top_errors: Vec::new(), // Placeholder
        };

        Ok(DetailedMetrics {
            latency_distribution,
            throughput_timeline,
            resource_utilization,
            operation_breakdown,
            error_analysis,
        })
    }

    /// Calculate latency distribution
    fn calculate_latency_distribution(&self, latencies: &[f64]) -> LatencyDistribution {
        if latencies.is_empty() {
            return LatencyDistribution {
                percentiles: HashMap::new(),
                histogram: Vec::new(),
                statistics: StatisticalMeasures {
                    mean: 0.0,
                    median: 0.0,
                    std_dev: 0.0,
                    variance: 0.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                },
            };
        }

        let mut sorted_latencies = latencies.to_vec();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentiles
        let mut percentiles = HashMap::new();
        let percentile_values = vec![50.0, 75.0, 90.0, 95.0, 99.0, 99.9];

        for p in percentile_values {
            let index = ((p / 100.0) * (sorted_latencies.len() - 1) as f64) as usize;
            percentiles.insert(format!("p{}", p), sorted_latencies[index]);
        }

        // Calculate statistics
        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let median = sorted_latencies[sorted_latencies.len() / 2];

        let variance = latencies.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / latencies.len() as f64;
        let std_dev = variance.sqrt();

        // Create histogram
        let min_val = sorted_latencies[0];
        let max_val = sorted_latencies[sorted_latencies.len() - 1];
        let bucket_count = 10;
        let bucket_size = (max_val - min_val) / bucket_count as f64;

        let mut histogram = Vec::new();
        for i in 0..bucket_count {
            let lower_bound = min_val + (i as f64 * bucket_size);
            let upper_bound = lower_bound + bucket_size;

            let count = latencies.iter()
                .filter(|&&x| x >= lower_bound && x < upper_bound)
                .count() as u64;

            histogram.push(HistogramBucket {
                lower_bound,
                upper_bound,
                count,
                frequency: count as f64 / latencies.len() as f64,
            });
        }

        LatencyDistribution {
            percentiles,
            histogram,
            statistics: StatisticalMeasures {
                mean,
                median,
                std_dev,
                variance,
                skewness: 0.0, // Simplified
                kurtosis: 0.0, // Simplified
            },
        }
    }

    /// Generate visualizations
    async fn generate_visualizations(&self, metrics: &[PerformanceMetrics], detailed_metrics: &DetailedMetrics) -> Result<Vec<Visualization>> {
        let mut visualizations = Vec::new();

        if !self.config.visualization.enable_ascii_charts {
            return Ok(visualizations);
        }

        // Generate latency timeline chart
        let latency_chart = self.create_ascii_line_chart(
            "Latency Over Time",
            &metrics.iter().enumerate().map(|(i, m)| DataPoint {
                x: i as f64,
                y: m.avg_latency_ms,
                label: Some(format!("Sample {}", i)),
                color: None,
                size: None,
            }).collect::<Vec<_>>(),
        );

        visualizations.push(Visualization {
            viz_type: VisualizationType::LineChart,
            title: "Latency Timeline".to_string(),
            ascii_art: latency_chart,
            data_points: metrics.iter().enumerate().map(|(i, m)| DataPoint {
                x: i as f64,
                y: m.avg_latency_ms,
                label: Some(format!("Sample {}", i)),
                color: None,
                size: None,
            }).collect(),
            config: self.config.visualization.clone(),
        });

        // Generate throughput chart
        let throughput_chart = self.create_ascii_line_chart(
            "Throughput Over Time",
            &metrics.iter().enumerate().map(|(i, m)| DataPoint {
                x: i as f64,
                y: m.throughput_ops_per_sec,
                label: Some(format!("Sample {}", i)),
                color: None,
                size: None,
            }).collect::<Vec<_>>(),
        );

        visualizations.push(Visualization {
            viz_type: VisualizationType::LineChart,
            title: "Throughput Timeline".to_string(),
            ascii_art: throughput_chart,
            data_points: metrics.iter().enumerate().map(|(i, m)| DataPoint {
                x: i as f64,
                y: m.throughput_ops_per_sec,
                label: Some(format!("Sample {}", i)),
                color: None,
                size: None,
            }).collect(),
            config: self.config.visualization.clone(),
        });

        // Generate resource utilization chart
        let cpu_chart = self.create_ascii_bar_chart(
            "CPU Utilization",
            &[("CPU", metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / metrics.len() as f64)],
        );

        visualizations.push(Visualization {
            viz_type: VisualizationType::BarChart,
            title: "Resource Utilization".to_string(),
            ascii_art: cpu_chart,
            data_points: vec![DataPoint {
                x: 0.0,
                y: metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / metrics.len() as f64,
                label: Some("CPU".to_string()),
                color: None,
                size: None,
            }],
            config: self.config.visualization.clone(),
        });

        // Generate latency histogram
        let histogram_chart = self.create_ascii_histogram(
            "Latency Distribution",
            &detailed_metrics.latency_distribution.histogram,
        );

        visualizations.push(Visualization {
            viz_type: VisualizationType::Histogram,
            title: "Latency Distribution".to_string(),
            ascii_art: histogram_chart,
            data_points: detailed_metrics.latency_distribution.histogram.iter().map(|bucket| DataPoint {
                x: (bucket.lower_bound + bucket.upper_bound) / 2.0,
                y: bucket.count as f64,
                label: Some(format!("{:.1}-{:.1}ms", bucket.lower_bound, bucket.upper_bound)),
                color: None,
                size: None,
            }).collect(),
            config: self.config.visualization.clone(),
        });

        Ok(visualizations)
    }

    /// Create ASCII line chart
    fn create_ascii_line_chart(&self, title: &str, data_points: &[DataPoint]) -> String {
        if data_points.is_empty() {
            return format!("{}\n(No data available)", title);
        }

        let width = self.config.visualization.chart_width;
        let height = self.config.visualization.chart_height;

        let min_y = data_points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_y = data_points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        let y_range = max_y - min_y;

        let mut chart = String::new();
        chart.push_str(&format!("{}\n", title));
        chart.push_str(&"=".repeat(title.len()));
        chart.push('\n');

        // Create chart grid
        for row in 0..height {
            let y_value = max_y - (row as f64 / (height - 1) as f64) * y_range;

            // Y-axis label
            chart.push_str(&format!("{:8.1} |", y_value));

            // Plot line
            for col in 0..width {
                let x_index = (col as f64 / (width - 1) as f64) * (data_points.len() - 1) as f64;
                let index = x_index.round() as usize;

                if index < data_points.len() {
                    let point_y = data_points[index].y;
                    let normalized_y = (point_y - min_y) / y_range;
                    let chart_y = (1.0 - normalized_y) * (height - 1) as f64;

                    if (chart_y.round() as usize) == row {
                        chart.push('*');
                    } else {
                        chart.push(' ');
                    }
                } else {
                    chart.push(' ');
                }
            }
            chart.push('\n');
        }

        // X-axis
        chart.push_str("         +");
        chart.push_str(&"-".repeat(width));
        chart.push('\n');

        chart
    }

    /// Create ASCII bar chart
    fn create_ascii_bar_chart(&self, title: &str, data: &[(&str, f64)]) -> String {
        if data.is_empty() {
            return format!("{}\n(No data available)", title);
        }

        let width = self.config.visualization.chart_width;
        let max_value = data.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);

        let mut chart = String::new();
        chart.push_str(&format!("{}\n", title));
        chart.push_str(&"=".repeat(title.len()));
        chart.push('\n');

        for (label, value) in data {
            let bar_length = ((value / max_value) * width as f64) as usize;
            chart.push_str(&format!("{:>10} |", label));
            chart.push_str(&"█".repeat(bar_length));
            chart.push_str(&format!(" {:.1}\n", value));
        }

        chart
    }

    /// Create ASCII histogram
    fn create_ascii_histogram(&self, title: &str, buckets: &[HistogramBucket]) -> String {
        if buckets.is_empty() {
            return format!("{}\n(No data available)", title);
        }

        let width = self.config.visualization.chart_width;
        let max_count = buckets.iter().map(|b| b.count).max().unwrap_or(1);

        let mut chart = String::new();
        chart.push_str(&format!("{}\n", title));
        chart.push_str(&"=".repeat(title.len()));
        chart.push('\n');

        for bucket in buckets {
            let bar_length = ((bucket.count as f64 / max_count as f64) * width as f64) as usize;
            chart.push_str(&format!("{:>8.1} |", bucket.lower_bound));
            chart.push_str(&"█".repeat(bar_length));
            chart.push_str(&format!(" {} ({:.1}%)\n", bucket.count, bucket.frequency * 100.0));
        }

        chart
    }

    /// Save report to file
    async fn save_report(&self, report: &ProfileReport) -> Result<()> {
        let output_dir = Path::new(&self.config.output_directory);
        tokio::fs::create_dir_all(output_dir).await?;

        let filename = format!("profile_report_{}_{}.{}",
            report.session_name.replace(' ', "_"),
            report.generated_at.format("%Y%m%d_%H%M%S"),
            match self.config.report_format {
                ReportFormat::Text => "txt",
                ReportFormat::Json => "json",
                ReportFormat::Html => "html",
                ReportFormat::Csv => "csv",
                ReportFormat::Markdown => "md",
            }
        );

        let file_path = output_dir.join(filename);

        let content = match self.config.report_format {
            ReportFormat::Json => serde_json::to_string_pretty(report)?,
            ReportFormat::Text => self.format_text_report(report),
            ReportFormat::Markdown => self.format_markdown_report(report),
            _ => serde_json::to_string_pretty(report)?, // Fallback to JSON
        };

        tokio::fs::write(&file_path, content).await?;
        tracing::info!("Report saved to: {}", file_path.display());

        Ok(())
    }

    /// Format report as text
    fn format_text_report(&self, report: &ProfileReport) -> String {
        let mut output = String::new();

        output.push_str(&format!("Performance Profile Report\n"));
        output.push_str(&format!("==========================\n\n"));
        output.push_str(&format!("Session: {}\n", report.session_name));
        output.push_str(&format!("Generated: {}\n", report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str(&format!("Duration: {:.2} seconds\n\n", report.duration_secs));

        // Summary
        output.push_str("Summary Statistics\n");
        output.push_str("------------------\n");
        output.push_str(&format!("Total Operations: {}\n", report.summary.total_operations));
        output.push_str(&format!("Average Latency: {:.2} ms\n", report.summary.avg_latency_ms));
        output.push_str(&format!("Peak Latency: {:.2} ms\n", report.summary.peak_latency_ms));
        output.push_str(&format!("Throughput: {:.2} ops/sec\n", report.summary.throughput));
        output.push_str(&format!("CPU Utilization: {:.1}%\n", report.summary.cpu_utilization));
        output.push_str(&format!("Memory Usage: {:.1}%\n", report.summary.memory_usage_mb));
        output.push_str(&format!("Error Rate: {:.3}%\n", report.summary.error_rate * 100.0));
        output.push_str(&format!("Performance Score: {:.1}/100\n\n", report.summary.performance_score));

        // Bottlenecks
        if !report.bottlenecks.is_empty() {
            output.push_str("Identified Bottlenecks\n");
            output.push_str("----------------------\n");
            for (i, bottleneck) in report.bottlenecks.iter().enumerate() {
                output.push_str(&format!("{}. {} ({:?})\n", i + 1, bottleneck.description, bottleneck.severity));
                output.push_str(&format!("   Impact: {:.1}% performance degradation\n", bottleneck.impact.performance_degradation));
                output.push_str(&format!("   Suggested fixes:\n"));
                for fix in &bottleneck.suggested_fixes {
                    output.push_str(&format!("   - {}\n", fix));
                }
                output.push('\n');
            }
        }

        // Recommendations
        if !report.recommendations.is_empty() {
            output.push_str("Optimization Recommendations\n");
            output.push_str("----------------------------\n");
            for (i, rec) in report.recommendations.iter().enumerate() {
                output.push_str(&format!("{}. {} ({:?} priority)\n", i + 1, rec.title, rec.priority));
                output.push_str(&format!("   {}\n", rec.description));
                output.push_str(&format!("   Expected improvement: {:.1}% performance gain\n", rec.expected_improvement.performance_gain));
                output.push_str(&format!("   Implementation effort: {:.1} hours ({:?})\n", rec.implementation_effort.estimated_hours, rec.implementation_effort.complexity));
                output.push('\n');
            }
        }

        // Visualizations
        for viz in &report.visualizations {
            output.push_str(&format!("{}\n", viz.title));
            output.push_str(&viz.ascii_art);
            output.push('\n');
        }

        output
    }

    /// Format report as markdown
    fn format_markdown_report(&self, report: &ProfileReport) -> String {
        let mut output = String::new();

        output.push_str(&format!("# Performance Profile Report\n\n"));
        output.push_str(&format!("**Session:** {}\n", report.session_name));
        output.push_str(&format!("**Generated:** {}\n", report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str(&format!("**Duration:** {:.2} seconds\n\n", report.duration_secs));

        // Summary
        output.push_str("## Summary Statistics\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Total Operations | {} |\n", report.summary.total_operations));
        output.push_str(&format!("| Average Latency | {:.2} ms |\n", report.summary.avg_latency_ms));
        output.push_str(&format!("| Peak Latency | {:.2} ms |\n", report.summary.peak_latency_ms));
        output.push_str(&format!("| Throughput | {:.2} ops/sec |\n", report.summary.throughput));
        output.push_str(&format!("| CPU Utilization | {:.1}% |\n", report.summary.cpu_utilization));
        output.push_str(&format!("| Memory Usage | {:.1}% |\n", report.summary.memory_usage_mb));
        output.push_str(&format!("| Error Rate | {:.3}% |\n", report.summary.error_rate * 100.0));
        output.push_str(&format!("| Performance Score | {:.1}/100 |\n\n", report.summary.performance_score));

        // Bottlenecks
        if !report.bottlenecks.is_empty() {
            output.push_str("## Identified Bottlenecks\n\n");
            for (i, bottleneck) in report.bottlenecks.iter().enumerate() {
                output.push_str(&format!("### {}. {} ({:?})\n\n", i + 1, bottleneck.description, bottleneck.severity));
                output.push_str(&format!("**Impact:** {:.1}% performance degradation\n\n", bottleneck.impact.performance_degradation));
                output.push_str("**Suggested fixes:**\n");
                for fix in &bottleneck.suggested_fixes {
                    output.push_str(&format!("- {}\n", fix));
                }
                output.push('\n');
            }
        }

        // Recommendations
        if !report.recommendations.is_empty() {
            output.push_str("## Optimization Recommendations\n\n");
            for (i, rec) in report.recommendations.iter().enumerate() {
                output.push_str(&format!("### {}. {} ({:?} priority)\n\n", i + 1, rec.title, rec.priority));
                output.push_str(&format!("{}\n\n", rec.description));
                output.push_str(&format!("**Expected improvement:** {:.1}% performance gain\n", rec.expected_improvement.performance_gain));
                output.push_str(&format!("**Implementation effort:** {:.1} hours ({:?})\n\n", rec.implementation_effort.estimated_hours, rec.implementation_effort.complexity));

                // Add code examples if available
                for example in &rec.code_examples {
                    output.push_str(&format!("#### {}\n\n", example.title));
                    output.push_str(&format!("```{}\n{}\n```\n\n", example.language, example.code));
                    output.push_str(&format!("{}\n\n", example.explanation));
                }
            }
        }

        // Visualizations
        output.push_str("## Visualizations\n\n");
        for viz in &report.visualizations {
            output.push_str(&format!("### {}\n\n", viz.title));
            output.push_str("```\n");
            output.push_str(&viz.ascii_art);
            output.push_str("```\n\n");
        }

        output
    }

    /// Get profiling history
    pub fn get_history(&self) -> &[ProfileReport] {
        &self.history
    }

    /// Clear profiling history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Export history to file
    pub async fn export_history(&self, file_path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.history)?;
        tokio::fs::write(file_path, content).await?;
        Ok(())
    }
}