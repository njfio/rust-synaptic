//! Real-Time Performance Monitoring System
//!
//! This module implements comprehensive real-time performance monitoring with low-latency
//! metrics collection, anomaly detection, automated alerting, performance baselines,
//! trend analysis, and predictive monitoring capabilities.

use crate::error::Result;
use crate::performance::metrics::{PerformanceMetrics, MetricsCollector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use std::time::Instant;

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitoringConfig {
    /// Metrics collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Maximum metrics history to keep
    pub max_history_size: usize,
    /// Anomaly detection sensitivity (0.0 to 1.0)
    pub anomaly_sensitivity: f64,
    /// Performance baseline window in minutes
    pub baseline_window_minutes: i64,
    /// Alert cooldown period in seconds
    pub alert_cooldown_seconds: u64,
    /// Predictive monitoring window in minutes
    pub prediction_window_minutes: i64,
    /// Enable automated alerting
    pub enable_alerting: bool,
    /// Enable predictive monitoring
    pub enable_predictive_monitoring: bool,
}

impl Default for RealTimeMonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000, // 1 second
            max_history_size: 3600, // 1 hour at 1 second intervals
            anomaly_sensitivity: 0.7,
            baseline_window_minutes: 60, // 1 hour baseline
            alert_cooldown_seconds: 300, // 5 minutes
            prediction_window_minutes: 30, // 30 minutes prediction
            enable_alerting: true,
            enable_predictive_monitoring: true,
        }
    }
}

/// Performance anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    pub id: String,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub metric_name: String,
    pub current_value: f64,
    pub expected_value: f64,
    pub deviation_score: f64,
    pub detected_at: DateTime<Utc>,
    pub description: String,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    Spike,
    Drop,
    Trend,
    Oscillation,
    Flatline,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AnomalySeverity,
    pub title: String,
    pub message: String,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub metrics: HashMap<String, f64>,
    pub anomalies: Vec<PerformanceAnomaly>,
    pub actions_taken: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertType {
    AnomalyDetected,
    ThresholdExceeded,
    PerformanceDegradation,
    ResourceExhaustion,
    PredictiveWarning,
}

/// Performance baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub metric_name: String,
    pub baseline_value: f64,
    pub standard_deviation: f64,
    pub confidence_interval: (f64, f64),
    pub sample_count: usize,
    pub calculated_at: DateTime<Utc>,
    pub valid_until: DateTime<Utc>,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub slope: f64,
    pub r_squared: f64,
    pub prediction: Option<f64>,
    pub confidence: f64,
    pub analyzed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Predictive monitoring result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveMonitoringResult {
    pub metric_name: String,
    pub predicted_value: f64,
    pub prediction_time: DateTime<Utc>,
    pub confidence: f64,
    pub prediction_interval: (f64, f64),
    pub risk_level: RiskLevel,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Real-time performance monitoring system
pub struct RealTimePerformanceMonitor {
    config: RealTimeMonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    metrics_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    trends: Arc<RwLock<HashMap<String, PerformanceTrend>>>,
    anomalies: Arc<RwLock<Vec<PerformanceAnomaly>>>,
    alerts: Arc<RwLock<Vec<PerformanceAlert>>>,
    alert_cooldowns: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    predictions: Arc<RwLock<Vec<PredictiveMonitoringResult>>>,
    monitoring_active: Arc<RwLock<bool>>,
    last_collection: Arc<Mutex<Option<Instant>>>,
}

impl RealTimePerformanceMonitor {
    /// Create a new real-time performance monitor
    pub async fn new(config: RealTimeMonitoringConfig) -> Result<Self> {
        let metrics_collector = Arc::new(MetricsCollector::new(crate::performance::PerformanceConfig::default()).await?);
        
        Ok(Self {
            config,
            metrics_collector,
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            baselines: Arc::new(RwLock::new(HashMap::new())),
            trends: Arc::new(RwLock::new(HashMap::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            alert_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            predictions: Arc::new(RwLock::new(Vec::new())),
            monitoring_active: Arc::new(RwLock::new(false)),
            last_collection: Arc::new(Mutex::new(None)),
        })
    }

    /// Start real-time monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        *self.monitoring_active.write().await = true;
        
        // Start metrics collection loop
        self.start_collection_loop().await?;
        
        // Start anomaly detection loop
        self.start_anomaly_detection_loop().await?;
        
        // Start trend analysis loop
        self.start_trend_analysis_loop().await?;
        
        // Start predictive monitoring if enabled
        if self.config.enable_predictive_monitoring {
            self.start_predictive_monitoring_loop().await?;
        }
        
        tracing::info!("Real-time performance monitoring started");
        Ok(())
    }

    /// Stop real-time monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        *self.monitoring_active.write().await = false;
        tracing::info!("Real-time performance monitoring stopped");
        Ok(())
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics> {
        self.metrics_collector.get_current_metrics().await
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self) -> Vec<PerformanceMetrics> {
        self.metrics_history.read().await.iter().cloned().collect()
    }

    /// Get performance baselines
    pub async fn get_baselines(&self) -> HashMap<String, PerformanceBaseline> {
        self.baselines.read().await.clone()
    }

    /// Get performance trends
    pub async fn get_trends(&self) -> HashMap<String, PerformanceTrend> {
        self.trends.read().await.clone()
    }

    /// Get detected anomalies
    pub async fn get_anomalies(&self) -> Vec<PerformanceAnomaly> {
        self.anomalies.read().await.clone()
    }

    /// Get active alerts
    pub async fn get_alerts(&self) -> Vec<PerformanceAlert> {
        self.alerts.read().await.clone()
    }

    /// Get predictive monitoring results
    pub async fn get_predictions(&self) -> Vec<PredictiveMonitoringResult> {
        self.predictions.read().await.clone()
    }

    /// Force metrics collection
    pub async fn collect_metrics_now(&self) -> Result<PerformanceMetrics> {
        let metrics = self.metrics_collector.get_current_metrics().await?;
        self.add_metrics_to_history(metrics.clone()).await?;
        Ok(metrics)
    }

    /// Add metrics to history
    async fn add_metrics_to_history(&self, metrics: PerformanceMetrics) -> Result<()> {
        let mut history = self.metrics_history.write().await;
        
        // Add new metrics
        history.push_back(metrics);
        
        // Maintain history size limit
        while history.len() > self.config.max_history_size {
            history.pop_front();
        }
        
        Ok(())
    }

    /// Start metrics collection loop
    async fn start_collection_loop(&self) -> Result<()> {
        let monitor = Arc::new(self.clone());
        let interval = std::time::Duration::from_millis(self.config.collection_interval_ms);
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            while *monitor.monitoring_active.read().await {
                interval_timer.tick().await;
                
                // Check if enough time has passed since last collection
                let should_collect = {
                    let mut last_collection = monitor.last_collection.lock().await;
                    let now = Instant::now();
                    
                    if let Some(last) = *last_collection {
                        if now.duration_since(last).as_millis() >= monitor.config.collection_interval_ms as u128 {
                            *last_collection = Some(now);
                            true
                        } else {
                            false
                        }
                    } else {
                        *last_collection = Some(now);
                        true
                    }
                };
                
                if should_collect {
                    if let Err(e) = monitor.collect_metrics_now().await {
                        tracing::error!("Failed to collect metrics: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }

    /// Start anomaly detection loop
    async fn start_anomaly_detection_loop(&self) -> Result<()> {
        let monitor = Arc::new(self.clone());
        let interval = std::time::Duration::from_millis(monitor.config.collection_interval_ms * 2);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            while *monitor.monitoring_active.read().await {
                interval_timer.tick().await;

                if let Err(e) = monitor.detect_anomalies().await {
                    tracing::error!("Failed to detect anomalies: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start trend analysis loop
    async fn start_trend_analysis_loop(&self) -> Result<()> {
        let monitor = Arc::new(self.clone());
        let interval = std::time::Duration::from_secs(60); // Every minute

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            while *monitor.monitoring_active.read().await {
                interval_timer.tick().await;

                if let Err(e) = monitor.analyze_trends().await {
                    tracing::error!("Failed to analyze trends: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start predictive monitoring loop
    async fn start_predictive_monitoring_loop(&self) -> Result<()> {
        let monitor = Arc::new(self.clone());
        let interval = std::time::Duration::from_secs(300); // Every 5 minutes

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            while *monitor.monitoring_active.read().await {
                interval_timer.tick().await;

                if let Err(e) = monitor.generate_predictions().await {
                    tracing::error!("Failed to generate predictions: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Detect performance anomalies
    async fn detect_anomalies(&self) -> Result<()> {
        let history = self.metrics_history.read().await;
        if history.len() < 10 {
            return Ok(()); // Need sufficient history
        }

        let latest_metrics = history.back().unwrap();
        let baselines = self.baselines.read().await;

        let mut new_anomalies = Vec::new();

        // Check each metric for anomalies
        let metrics_to_check = vec![
            ("avg_latency_ms", latest_metrics.avg_latency_ms),
            ("throughput_ops_per_sec", latest_metrics.throughput_ops_per_sec),
            ("cpu_usage_percent", latest_metrics.cpu_usage_percent),
            ("memory_usage_percent", latest_metrics.memory_usage_percent),
            ("error_rate", latest_metrics.error_rate),
            ("cache_hit_rate", latest_metrics.cache_hit_rate),
        ];

        for (metric_name, current_value) in metrics_to_check {
            if let Some(baseline) = baselines.get(metric_name) {
                let deviation = self.calculate_deviation(current_value, baseline);

                if deviation.abs() > self.config.anomaly_sensitivity {
                    let anomaly_type = if deviation > 0.0 {
                        if metric_name == "cache_hit_rate" { AnomalyType::Drop } else { AnomalyType::Spike }
                    } else {
                        if metric_name == "cache_hit_rate" { AnomalyType::Spike } else { AnomalyType::Drop }
                    };

                    let severity = match deviation.abs() {
                        x if x > 3.0 => AnomalySeverity::Critical,
                        x if x > 2.0 => AnomalySeverity::High,
                        x if x > 1.0 => AnomalySeverity::Medium,
                        _ => AnomalySeverity::Low,
                    };

                    let anomaly_description = format!(
                        "{} anomaly detected: {} = {:.2} (expected: {:.2}, deviation: {:.2}σ)",
                        match anomaly_type {
                            AnomalyType::Spike => "Performance spike",
                            AnomalyType::Drop => "Performance drop",
                            _ => "Performance anomaly",
                        },
                        metric_name,
                        current_value,
                        baseline.baseline_value,
                        deviation.abs()
                    );

                    let anomaly = PerformanceAnomaly {
                        id: Uuid::new_v4().to_string(),
                        anomaly_type: anomaly_type.clone(),
                        severity,
                        metric_name: metric_name.to_string(),
                        current_value,
                        expected_value: baseline.baseline_value,
                        deviation_score: deviation.abs(),
                        detected_at: Utc::now(),
                        description: anomaly_description,
                        recommended_actions: self.get_anomaly_recommendations(metric_name, &anomaly_type),
                    };

                    new_anomalies.push(anomaly);
                }
            }
        }

        // Add new anomalies
        if !new_anomalies.is_empty() {
            let mut anomalies = self.anomalies.write().await;
            anomalies.extend(new_anomalies.clone());

            // Keep only recent anomalies (last 1000)
            if anomalies.len() > 1000 {
                anomalies.drain(0..100);
            }

            // Generate alerts if enabled
            if self.config.enable_alerting {
                self.generate_alerts_for_anomalies(&new_anomalies).await?;
            }
        }

        Ok(())
    }

    /// Calculate deviation from baseline in standard deviations
    fn calculate_deviation(&self, current_value: f64, baseline: &PerformanceBaseline) -> f64 {
        if baseline.standard_deviation == 0.0 {
            return 0.0;
        }
        (current_value - baseline.baseline_value) / baseline.standard_deviation
    }

    /// Get recommendations for anomaly type
    fn get_anomaly_recommendations(&self, metric_name: &str, anomaly_type: &AnomalyType) -> Vec<String> {
        match (metric_name, anomaly_type) {
            ("avg_latency_ms", AnomalyType::Spike) => vec![
                "Check for resource contention".to_string(),
                "Review recent code changes".to_string(),
                "Monitor database performance".to_string(),
                "Check network connectivity".to_string(),
            ],
            ("throughput_ops_per_sec", AnomalyType::Drop) => vec![
                "Check system resources".to_string(),
                "Review load balancer configuration".to_string(),
                "Monitor database connections".to_string(),
                "Check for bottlenecks".to_string(),
            ],
            ("cpu_usage_percent", AnomalyType::Spike) => vec![
                "Identify CPU-intensive processes".to_string(),
                "Consider scaling up resources".to_string(),
                "Review algorithm efficiency".to_string(),
                "Check for infinite loops".to_string(),
            ],
            ("memory_usage_percent", AnomalyType::Spike) => vec![
                "Check for memory leaks".to_string(),
                "Review memory allocation patterns".to_string(),
                "Consider increasing memory limits".to_string(),
                "Monitor garbage collection".to_string(),
            ],
            ("error_rate", AnomalyType::Spike) => vec![
                "Review error logs".to_string(),
                "Check external dependencies".to_string(),
                "Validate input data".to_string(),
                "Review recent deployments".to_string(),
            ],
            ("cache_hit_rate", AnomalyType::Drop) => vec![
                "Review cache configuration".to_string(),
                "Check cache eviction policies".to_string(),
                "Monitor cache size limits".to_string(),
                "Review access patterns".to_string(),
            ],
            _ => vec![
                "Monitor system closely".to_string(),
                "Review recent changes".to_string(),
                "Check system logs".to_string(),
            ],
        }
    }

    /// Generate alerts for detected anomalies
    async fn generate_alerts_for_anomalies(&self, anomalies: &[PerformanceAnomaly]) -> Result<()> {
        let mut alerts = self.alerts.write().await;
        let mut cooldowns = self.alert_cooldowns.write().await;
        let now = Utc::now();

        for anomaly in anomalies {
            // Check cooldown
            let cooldown_key = format!("{}_{:?}", anomaly.metric_name, anomaly.anomaly_type);
            if let Some(last_alert) = cooldowns.get(&cooldown_key) {
                let cooldown_duration = Duration::seconds(self.config.alert_cooldown_seconds as i64);
                if now.signed_duration_since(*last_alert) < cooldown_duration {
                    continue; // Still in cooldown
                }
            }

            let alert = PerformanceAlert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::AnomalyDetected,
                severity: anomaly.severity.clone(),
                title: format!("Performance Anomaly: {}", anomaly.metric_name),
                message: anomaly.description.clone(),
                triggered_at: now,
                resolved_at: None,
                metrics: {
                    let mut metrics = HashMap::new();
                    metrics.insert(anomaly.metric_name.clone(), anomaly.current_value);
                    metrics.insert("expected_value".to_string(), anomaly.expected_value);
                    metrics.insert("deviation_score".to_string(), anomaly.deviation_score);
                    metrics
                },
                anomalies: vec![anomaly.clone()],
                actions_taken: vec!["Alert generated".to_string()],
            };

            alerts.push(alert);
            cooldowns.insert(cooldown_key, now);
        }

        // Keep only recent alerts (last 500)
        if alerts.len() > 500 {
            alerts.drain(0..50);
        }

        Ok(())
    }

    /// Analyze performance trends
    async fn analyze_trends(&self) -> Result<()> {
        let history = self.metrics_history.read().await;
        if history.len() < 30 {
            return Ok(()); // Need sufficient history for trend analysis
        }

        let mut trends = self.trends.write().await;

        // Analyze trends for key metrics
        let metrics_to_analyze = vec![
            "avg_latency_ms",
            "throughput_ops_per_sec",
            "cpu_usage_percent",
            "memory_usage_percent",
            "error_rate",
            "cache_hit_rate",
        ];

        for metric_name in metrics_to_analyze {
            let values: Vec<f64> = history.iter().map(|m| self.extract_metric_value(m, metric_name)).collect();
            let trend = self.calculate_trend(&values, metric_name).await?;
            trends.insert(metric_name.to_string(), trend);
        }

        Ok(())
    }

    /// Extract metric value by name
    fn extract_metric_value(&self, metrics: &PerformanceMetrics, metric_name: &str) -> f64 {
        match metric_name {
            "avg_latency_ms" => metrics.avg_latency_ms,
            "throughput_ops_per_sec" => metrics.throughput_ops_per_sec,
            "cpu_usage_percent" => metrics.cpu_usage_percent,
            "memory_usage_percent" => metrics.memory_usage_percent,
            "error_rate" => metrics.error_rate,
            "cache_hit_rate" => metrics.cache_hit_rate,
            _ => 0.0,
        }
    }

    /// Calculate trend for a series of values
    async fn calculate_trend(&self, values: &[f64], metric_name: &str) -> Result<PerformanceTrend> {
        if values.len() < 2 {
            return Ok(PerformanceTrend {
                metric_name: metric_name.to_string(),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                slope: 0.0,
                r_squared: 0.0,
                prediction: None,
                confidence: 0.0,
                analyzed_at: Utc::now(),
            });
        }

        // Calculate linear regression
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum();
        let sum_x_squared: f64 = x_values.iter().map(|x| x * x).sum();
        let _sum_y_squared: f64 = values.iter().map(|y| y * y).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot: f64 = values.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = x_values.iter().zip(values.iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

        // Determine trend direction and strength
        let trend_direction = if slope.abs() < 0.001 {
            TrendDirection::Stable
        } else if slope > 0.0 {
            if metric_name == "error_rate" || metric_name == "avg_latency_ms" {
                TrendDirection::Degrading
            } else {
                TrendDirection::Improving
            }
        } else {
            if metric_name == "error_rate" || metric_name == "avg_latency_ms" {
                TrendDirection::Improving
            } else {
                TrendDirection::Degrading
            }
        };

        let trend_strength = slope.abs() * r_squared;

        // Generate prediction for next time point
        let next_x = values.len() as f64;
        let prediction = slope * next_x + intercept;
        let confidence = r_squared;

        Ok(PerformanceTrend {
            metric_name: metric_name.to_string(),
            trend_direction,
            trend_strength,
            slope,
            r_squared,
            prediction: Some(prediction),
            confidence,
            analyzed_at: Utc::now(),
        })
    }

    /// Generate predictive monitoring results
    async fn generate_predictions(&self) -> Result<()> {
        let history = self.metrics_history.read().await;
        if history.len() < 60 {
            return Ok(()); // Need sufficient history for predictions
        }

        let trends = self.trends.read().await;
        let mut predictions = self.predictions.write().await;

        predictions.clear(); // Clear old predictions

        let prediction_time = Utc::now() + Duration::minutes(self.config.prediction_window_minutes);

        for (metric_name, trend) in trends.iter() {
            if let Some(predicted_value) = trend.prediction {
                let risk_level = self.assess_prediction_risk(metric_name, predicted_value, trend).await;

                let prediction = PredictiveMonitoringResult {
                    metric_name: metric_name.clone(),
                    predicted_value,
                    prediction_time,
                    confidence: trend.confidence,
                    prediction_interval: self.calculate_prediction_interval(predicted_value, trend.confidence),
                    risk_level: risk_level.clone(),
                    recommended_actions: self.get_prediction_recommendations(metric_name, &risk_level),
                };

                predictions.push(prediction);
            }
        }

        Ok(())
    }

    /// Assess risk level for prediction
    async fn assess_prediction_risk(&self, metric_name: &str, predicted_value: f64, trend: &PerformanceTrend) -> RiskLevel {
        // Define risk thresholds for different metrics
        let (warning_threshold, critical_threshold) = match metric_name {
            "avg_latency_ms" => (10.0, 50.0),
            "cpu_usage_percent" => (80.0, 95.0),
            "memory_usage_percent" => (85.0, 95.0),
            "error_rate" => (0.05, 0.1),
            "cache_hit_rate" => (0.7, 0.5), // Lower is worse for cache hit rate
            _ => (0.8, 0.95),
        };

        let is_cache_metric = metric_name == "cache_hit_rate";

        if trend.confidence < 0.3 {
            return RiskLevel::Low; // Low confidence predictions are low risk
        }

        if is_cache_metric {
            if predicted_value < critical_threshold {
                RiskLevel::Critical
            } else if predicted_value < warning_threshold {
                RiskLevel::High
            } else if trend.trend_direction == TrendDirection::Degrading {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            }
        } else {
            if predicted_value > critical_threshold {
                RiskLevel::Critical
            } else if predicted_value > warning_threshold {
                RiskLevel::High
            } else if trend.trend_direction == TrendDirection::Degrading {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            }
        }
    }

    /// Calculate prediction interval
    fn calculate_prediction_interval(&self, predicted_value: f64, confidence: f64) -> (f64, f64) {
        let margin = predicted_value * (1.0 - confidence) * 0.5;
        (predicted_value - margin, predicted_value + margin)
    }

    /// Get recommendations for prediction risk level
    fn get_prediction_recommendations(&self, metric_name: &str, risk_level: &RiskLevel) -> Vec<String> {
        match (metric_name, risk_level) {
            (_, RiskLevel::Critical) => vec![
                "Immediate action required".to_string(),
                "Scale resources proactively".to_string(),
                "Activate incident response".to_string(),
                "Monitor system closely".to_string(),
            ],
            (_, RiskLevel::High) => vec![
                "Prepare for potential issues".to_string(),
                "Review resource allocation".to_string(),
                "Monitor trends closely".to_string(),
                "Consider preventive measures".to_string(),
            ],
            ("avg_latency_ms", RiskLevel::Medium) => vec![
                "Monitor latency trends".to_string(),
                "Review performance optimizations".to_string(),
                "Check for bottlenecks".to_string(),
            ],
            ("cpu_usage_percent", RiskLevel::Medium) => vec![
                "Monitor CPU usage trends".to_string(),
                "Consider CPU optimization".to_string(),
                "Review process efficiency".to_string(),
            ],
            ("memory_usage_percent", RiskLevel::Medium) => vec![
                "Monitor memory usage trends".to_string(),
                "Review memory allocation".to_string(),
                "Check for memory leaks".to_string(),
            ],
            _ => vec![
                "Continue monitoring".to_string(),
                "Maintain current configuration".to_string(),
            ],
        }
    }

    /// Update performance baselines
    pub async fn update_baselines(&self) -> Result<()> {
        let history = self.metrics_history.read().await;
        if history.len() < 60 {
            return Ok(()); // Need sufficient history
        }

        let baseline_window = Duration::minutes(self.config.baseline_window_minutes);
        let cutoff_time = Utc::now() - baseline_window;

        // Filter recent metrics for baseline calculation
        let recent_metrics: Vec<&PerformanceMetrics> = history.iter()
            .filter(|m| m.timestamp > cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return Ok(());
        }

        let mut baselines = self.baselines.write().await;

        // Calculate baselines for key metrics
        let metrics_to_baseline = vec![
            "avg_latency_ms",
            "throughput_ops_per_sec",
            "cpu_usage_percent",
            "memory_usage_percent",
            "error_rate",
            "cache_hit_rate",
        ];

        for metric_name in metrics_to_baseline {
            let values: Vec<f64> = recent_metrics.iter()
                .map(|m| self.extract_metric_value(m, metric_name))
                .collect();

            if !values.is_empty() {
                let baseline = self.calculate_baseline(metric_name, &values).await?;
                baselines.insert(metric_name.to_string(), baseline);
            }
        }

        Ok(())
    }

    /// Calculate baseline for a metric
    async fn calculate_baseline(&self, metric_name: &str, values: &[f64]) -> Result<PerformanceBaseline> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Calculate confidence interval (95%)
        let margin = 1.96 * std_dev / n.sqrt();
        let confidence_interval = (mean - margin, mean + margin);

        Ok(PerformanceBaseline {
            metric_name: metric_name.to_string(),
            baseline_value: mean,
            standard_deviation: std_dev,
            confidence_interval,
            sample_count: values.len(),
            calculated_at: Utc::now(),
            valid_until: Utc::now() + Duration::hours(24), // Valid for 24 hours
        })
    }

    /// Get monitoring status
    pub async fn get_monitoring_status(&self) -> MonitoringStatus {
        let is_active = *self.monitoring_active.read().await;
        let metrics_count = self.metrics_history.read().await.len();
        let anomalies_count = self.anomalies.read().await.len();
        let alerts_count = self.alerts.read().await.len();
        let predictions_count = self.predictions.read().await.len();

        MonitoringStatus {
            is_active,
            metrics_collected: metrics_count,
            anomalies_detected: anomalies_count,
            alerts_generated: alerts_count,
            predictions_available: predictions_count,
            last_collection: Utc::now(), // Simplified
        }
    }
}

/// Monitoring status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringStatus {
    pub is_active: bool,
    pub metrics_collected: usize,
    pub anomalies_detected: usize,
    pub alerts_generated: usize,
    pub predictions_available: usize,
    pub last_collection: DateTime<Utc>,
}

// Implement Clone for RealTimePerformanceMonitor
impl Clone for RealTimePerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_collector: Arc::clone(&self.metrics_collector),
            metrics_history: Arc::clone(&self.metrics_history),
            baselines: Arc::clone(&self.baselines),
            trends: Arc::clone(&self.trends),
            anomalies: Arc::clone(&self.anomalies),
            alerts: Arc::clone(&self.alerts),
            alert_cooldowns: Arc::clone(&self.alert_cooldowns),
            predictions: Arc::clone(&self.predictions),
            monitoring_active: Arc::clone(&self.monitoring_active),
            last_collection: Arc::clone(&self.last_collection),
        }
    }
}
