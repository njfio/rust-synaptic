// Performance Analytics Module
// Advanced performance monitoring and optimization recommendations

use crate::error::Result;
use crate::analytics::{AnalyticsConfig, AnalyticsInsight, InsightType, InsightPriority};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: DateTime<Utc>,
    /// Operations per second
    pub ops_per_second: f64,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Active connections
    pub active_connections: u32,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric_name: String,
    /// Trend direction
    pub trend_direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub trend_strength: f64,
    /// Rate of change per hour
    pub change_rate: f64,
    /// Time period analyzed
    pub analysis_period: Duration,
    /// Confidence in the trend
    pub confidence: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    /// Performance is improving over time
    Improving,
    /// Performance is degrading over time
    Degrading,
    /// Performance is stable
    Stable,
    /// Performance is volatile/unpredictable
    Volatile,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: Uuid,
    /// Optimization category
    pub category: OptimizationCategory,
    /// Priority level
    pub priority: OptimizationPriority,
    /// Title of the recommendation
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected performance impact
    pub expected_impact: PerformanceImpact,
    /// Implementation complexity
    pub complexity: ImplementationComplexity,
    /// Estimated implementation time
    pub estimated_time_hours: f64,
    /// Supporting metrics
    pub supporting_metrics: Vec<String>,
    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
}

/// Optimization categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationCategory {
    /// Caching optimization
    Caching,
    /// Indexing optimization
    Indexing,
    /// Query optimization
    QueryOptimization,
    /// Memory management optimization
    MemoryManagement,
    /// Network optimization
    NetworkOptimization,
    AlgorithmImprovement,
    ResourceScaling,
}

/// Optimization priorities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Expected performance impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Expected throughput improvement (percentage)
    pub throughput_improvement: f64,
    /// Expected latency reduction (percentage)
    pub latency_reduction: f64,
    /// Expected memory savings (percentage)
    pub memory_savings: f64,
    /// Expected CPU savings (percentage)
    pub cpu_savings: f64,
}

/// Implementation complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImplementationComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck ID
    pub id: Uuid,
    /// Component affected
    pub component: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Description
    pub description: String,
    /// Impact on overall performance
    pub performance_impact: f64,
    /// Detection confidence
    pub confidence: f64,
    /// Detected timestamp
    pub detected_at: DateTime<Utc>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Database,
    Cache,
    Algorithm,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum BottleneckSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Performance analytics engine
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Configuration
    _config: AnalyticsConfig,
    /// Performance snapshots history
    snapshots: VecDeque<PerformanceSnapshot>,
    /// Performance trends
    trends: HashMap<String, PerformanceTrend>,
    /// Optimization recommendations
    recommendations: Vec<OptimizationRecommendation>,
    /// Detected bottlenecks
    bottlenecks: Vec<PerformanceBottleneck>,
    /// Baseline performance metrics
    _baseline_metrics: HashMap<String, f64>,
    /// Performance targets
    performance_targets: HashMap<String, f64>,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(config: &AnalyticsConfig) -> Result<Self> {
        let mut performance_targets = HashMap::new();
        performance_targets.insert("ops_per_second".to_string(), 1000.0);
        performance_targets.insert("avg_response_time_ms".to_string(), 1.0);
        performance_targets.insert("cache_hit_rate".to_string(), 0.9);
        performance_targets.insert("error_rate".to_string(), 0.01);

        Ok(Self {
            _config: config.clone(),
            snapshots: VecDeque::new(),
            trends: HashMap::new(),
            recommendations: Vec::new(),
            bottlenecks: Vec::new(),
            _baseline_metrics: HashMap::new(),
            performance_targets,
        })
    }

    /// Record a performance snapshot
    pub async fn record_snapshot(&mut self, snapshot: PerformanceSnapshot) -> Result<()> {
        self.snapshots.push_back(snapshot);

        // Keep only recent snapshots (last 1000)
        if self.snapshots.len() > 1000 {
            self.snapshots.pop_front();
        }

        // Update trends
        self.update_trends().await?;

        // Detect bottlenecks
        self.detect_bottlenecks().await?;

        Ok(())
    }

    /// Update performance trends
    async fn update_trends(&mut self) -> Result<()> {
        if self.snapshots.len() < 10 {
            return Ok(());
        }

        let metrics = vec![
            "ops_per_second",
            "avg_response_time_ms",
            "memory_usage_bytes",
            "cpu_usage_percent",
            "cache_hit_rate",
            "error_rate",
        ];

        for metric in metrics {
            let trend = self.calculate_trend(metric).await?;
            self.trends.insert(metric.to_string(), trend);
        }

        Ok(())
    }

    /// Calculate trend for a specific metric
    async fn calculate_trend(&self, metric_name: &str) -> Result<PerformanceTrend> {
        let values: Vec<f64> = self.snapshots
            .iter()
            .map(|snapshot| self.extract_metric_value(snapshot, metric_name))
            .collect();

        if values.len() < 2 {
            return Ok(PerformanceTrend {
                metric_name: metric_name.to_string(),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                change_rate: 0.0,
                analysis_period: Duration::hours(1),
                confidence: 0.0,
            });
        }

        // Simple linear regression to determine trend
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x_squared_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        let intercept = (y_sum - slope * x_sum) / n;

        // Calculate R-squared for confidence
        let y_mean = y_sum / n;
        let ss_tot: f64 = values.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = values.iter().enumerate()
            .map(|(i, &y)| {
                let predicted = slope * i as f64 + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

        // Determine trend direction and strength
        let (trend_direction, trend_strength) = if slope.abs() < 0.01 {
            (TrendDirection::Stable, 0.0)
        } else if slope > 0.0 {
            if metric_name == "avg_response_time_ms" || metric_name == "error_rate" {
                (TrendDirection::Degrading, slope.abs())
            } else {
                (TrendDirection::Improving, slope.abs())
            }
        } else {
            if metric_name == "avg_response_time_ms" || metric_name == "error_rate" {
                (TrendDirection::Improving, slope.abs())
            } else {
                (TrendDirection::Degrading, slope.abs())
            }
        };

        // Check for volatility
        let volatility = self.calculate_volatility(&values);
        let final_direction = if volatility > 0.5 {
            TrendDirection::Volatile
        } else {
            trend_direction
        };

        Ok(PerformanceTrend {
            metric_name: metric_name.to_string(),
            trend_direction: final_direction,
            trend_strength: trend_strength.min(1.0),
            change_rate: slope,
            analysis_period: Duration::hours(1),
            confidence: r_squared.max(0.0).min(1.0),
        })
    }

    /// Extract metric value from snapshot
    fn extract_metric_value(&self, snapshot: &PerformanceSnapshot, metric_name: &str) -> f64 {
        match metric_name {
            "ops_per_second" => snapshot.ops_per_second,
            "avg_response_time_ms" => snapshot.avg_response_time_ms,
            "memory_usage_bytes" => snapshot.memory_usage_bytes as f64,
            "cpu_usage_percent" => snapshot.cpu_usage_percent,
            "cache_hit_rate" => snapshot.cache_hit_rate,
            "error_rate" => snapshot.error_rate,
            _ => 0.0,
        }
    }

    /// Calculate volatility of values
    fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let std_dev = variance.sqrt();
        if mean > 0.0 {
            std_dev / mean
        } else {
            0.0
        }
    }

    /// Detect performance bottlenecks
    async fn detect_bottlenecks(&mut self) -> Result<()> {
        if let Some(latest_snapshot) = self.snapshots.back() {
            // Check CPU bottleneck
            if latest_snapshot.cpu_usage_percent > 80.0 {
                let bottleneck = PerformanceBottleneck {
                    id: Uuid::new_v4(),
                    component: "CPU".to_string(),
                    bottleneck_type: BottleneckType::CPU,
                    severity: if latest_snapshot.cpu_usage_percent > 95.0 {
                        BottleneckSeverity::Critical
                    } else {
                        BottleneckSeverity::Major
                    },
                    description: format!("High CPU usage: {:.1}%", latest_snapshot.cpu_usage_percent),
                    performance_impact: latest_snapshot.cpu_usage_percent / 100.0,
                    confidence: 0.9,
                    detected_at: Utc::now(),
                };
                self.bottlenecks.push(bottleneck);
            }

            // Check response time bottleneck
            if latest_snapshot.avg_response_time_ms > 10.0 {
                let bottleneck = PerformanceBottleneck {
                    id: Uuid::new_v4(),
                    component: "Response Time".to_string(),
                    bottleneck_type: BottleneckType::Algorithm,
                    severity: if latest_snapshot.avg_response_time_ms > 100.0 {
                        BottleneckSeverity::Critical
                    } else {
                        BottleneckSeverity::Moderate
                    },
                    description: format!("High response time: {:.1}ms", latest_snapshot.avg_response_time_ms),
                    performance_impact: (latest_snapshot.avg_response_time_ms / 100.0).min(1.0),
                    confidence: 0.8,
                    detected_at: Utc::now(),
                };
                self.bottlenecks.push(bottleneck);
            }

            // Check cache hit rate
            if latest_snapshot.cache_hit_rate < 0.7 {
                let bottleneck = PerformanceBottleneck {
                    id: Uuid::new_v4(),
                    component: "Cache".to_string(),
                    bottleneck_type: BottleneckType::Cache,
                    severity: if latest_snapshot.cache_hit_rate < 0.5 {
                        BottleneckSeverity::Major
                    } else {
                        BottleneckSeverity::Moderate
                    },
                    description: format!("Low cache hit rate: {:.1}%", latest_snapshot.cache_hit_rate * 100.0),
                    performance_impact: 1.0 - latest_snapshot.cache_hit_rate,
                    confidence: 0.85,
                    detected_at: Utc::now(),
                };
                self.bottlenecks.push(bottleneck);
            }
        }

        Ok(())
    }

    /// Generate optimization recommendations
    pub async fn generate_recommendations(&mut self) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Analyze trends for recommendations
        for (metric_name, trend) in &self.trends {
            if trend.trend_direction == TrendDirection::Degrading && trend.confidence > 0.7 {
                let recommendation = self.create_trend_based_recommendation(metric_name, trend).await?;
                if let Some(rec) = recommendation {
                    recommendations.push(rec);
                }
            }
        }

        // Analyze bottlenecks for recommendations
        for bottleneck in &self.bottlenecks {
            if bottleneck.severity >= BottleneckSeverity::Moderate {
                let recommendation = self.create_bottleneck_based_recommendation(bottleneck).await?;
                if let Some(rec) = recommendation {
                    recommendations.push(rec);
                }
            }
        }

        self.recommendations.extend(recommendations.clone());
        Ok(recommendations)
    }

    /// Create recommendation based on trend analysis
    async fn create_trend_based_recommendation(&self, metric_name: &str, trend: &PerformanceTrend) -> Result<Option<OptimizationRecommendation>> {
        let recommendation = match metric_name {
            "avg_response_time_ms" => Some(OptimizationRecommendation {
                id: Uuid::new_v4(),
                category: OptimizationCategory::QueryOptimization,
                priority: OptimizationPriority::High,
                title: "Optimize Query Performance".to_string(),
                description: "Response times are increasing. Consider optimizing database queries and adding indexes.".to_string(),
                expected_impact: PerformanceImpact {
                    throughput_improvement: 20.0,
                    latency_reduction: 50.0,
                    memory_savings: 0.0,
                    cpu_savings: 10.0,
                },
                complexity: ImplementationComplexity::Medium,
                estimated_time_hours: 8.0,
                supporting_metrics: vec![
                    format!("Response time trend: {:?}", trend.trend_direction),
                    format!("Change rate: {:.2}ms/hour", trend.change_rate),
                ],
                generated_at: Utc::now(),
            }),
            "cache_hit_rate" => Some(OptimizationRecommendation {
                id: Uuid::new_v4(),
                category: OptimizationCategory::Caching,
                priority: OptimizationPriority::High,
                title: "Improve Cache Strategy".to_string(),
                description: "Cache hit rate is declining. Review cache policies and consider increasing cache size.".to_string(),
                expected_impact: PerformanceImpact {
                    throughput_improvement: 30.0,
                    latency_reduction: 40.0,
                    memory_savings: -10.0, // May use more memory
                    cpu_savings: 15.0,
                },
                complexity: ImplementationComplexity::Low,
                estimated_time_hours: 4.0,
                supporting_metrics: vec![
                    format!("Cache hit rate trend: {:?}", trend.trend_direction),
                    format!("Change rate: {:.3}/hour", trend.change_rate),
                ],
                generated_at: Utc::now(),
            }),
            _ => None,
        };

        Ok(recommendation)
    }

    /// Create recommendation based on bottleneck analysis
    async fn create_bottleneck_based_recommendation(&self, bottleneck: &PerformanceBottleneck) -> Result<Option<OptimizationRecommendation>> {
        let recommendation = match bottleneck.bottleneck_type {
            BottleneckType::CPU => Some(OptimizationRecommendation {
                id: Uuid::new_v4(),
                category: OptimizationCategory::AlgorithmImprovement,
                priority: OptimizationPriority::Critical,
                title: "Optimize CPU-Intensive Operations".to_string(),
                description: "High CPU usage detected. Profile and optimize CPU-intensive algorithms.".to_string(),
                expected_impact: PerformanceImpact {
                    throughput_improvement: 25.0,
                    latency_reduction: 30.0,
                    memory_savings: 0.0,
                    cpu_savings: 40.0,
                },
                complexity: ImplementationComplexity::High,
                estimated_time_hours: 16.0,
                supporting_metrics: vec![
                    format!("CPU usage: {:.1}%", bottleneck.performance_impact * 100.0),
                    format!("Confidence: {:.1}%", bottleneck.confidence * 100.0),
                ],
                generated_at: Utc::now(),
            }),
            BottleneckType::Cache => Some(OptimizationRecommendation {
                id: Uuid::new_v4(),
                category: OptimizationCategory::Caching,
                priority: OptimizationPriority::High,
                title: "Optimize Cache Configuration".to_string(),
                description: "Cache performance issues detected. Review cache size, eviction policies, and access patterns.".to_string(),
                expected_impact: PerformanceImpact {
                    throughput_improvement: 35.0,
                    latency_reduction: 45.0,
                    memory_savings: 0.0,
                    cpu_savings: 20.0,
                },
                complexity: ImplementationComplexity::Medium,
                estimated_time_hours: 6.0,
                supporting_metrics: vec![
                    format!("Cache impact: {:.1}%", bottleneck.performance_impact * 100.0),
                    bottleneck.description.clone(),
                ],
                generated_at: Utc::now(),
            }),
            _ => None,
        };

        Ok(recommendation)
    }

    /// Generate performance insights
    pub async fn generate_insights(&mut self) -> Result<Vec<AnalyticsInsight>> {
        let mut insights = Vec::new();

        // Generate insights from trends
        for (metric_name, trend) in &self.trends {
            if trend.confidence > 0.8 && trend.trend_strength > 0.5 {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::PerformanceOptimization,
                    title: format!("Performance Trend Alert: {}", metric_name),
                    description: format!(
                        "Metric '{}' shows {} trend with {:.1}% confidence",
                        metric_name,
                        format!("{:?}", trend.trend_direction).to_lowercase(),
                        trend.confidence * 100.0
                    ),
                    confidence: trend.confidence,
                    evidence: vec![
                        format!("Trend strength: {:.2}", trend.trend_strength),
                        format!("Change rate: {:.3}/hour", trend.change_rate),
                    ],
                    generated_at: Utc::now(),
                    priority: if trend.trend_direction == TrendDirection::Degrading {
                        InsightPriority::High
                    } else {
                        InsightPriority::Medium
                    },
                };
                insights.push(insight);
            }
        }

        // Generate insights from bottlenecks
        for bottleneck in &self.bottlenecks {
            if bottleneck.severity >= BottleneckSeverity::Major {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::PerformanceOptimization,
                    title: format!("Performance Bottleneck: {}", bottleneck.component),
                    description: bottleneck.description.clone(),
                    confidence: bottleneck.confidence,
                    evidence: vec![
                        format!("Bottleneck type: {:?}", bottleneck.bottleneck_type),
                        format!("Performance impact: {:.1}%", bottleneck.performance_impact * 100.0),
                    ],
                    generated_at: Utc::now(),
                    priority: match bottleneck.severity {
                        BottleneckSeverity::Critical => InsightPriority::Critical,
                        BottleneckSeverity::Major => InsightPriority::High,
                        _ => InsightPriority::Medium,
                    },
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> Option<&PerformanceSnapshot> {
        self.snapshots.back()
    }

    /// Get performance trends
    pub fn get_trends(&self) -> &HashMap<String, PerformanceTrend> {
        &self.trends
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> &[OptimizationRecommendation] {
        &self.recommendations
    }

    /// Get detected bottlenecks
    pub fn get_bottlenecks(&self) -> &[PerformanceBottleneck] {
        &self.bottlenecks
    }

    /// Update performance targets
    pub fn update_targets(&mut self, targets: HashMap<String, f64>) {
        self.performance_targets.extend(targets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_analyzer_creation() {
        let config = AnalyticsConfig::default();
        let analyzer = PerformanceAnalyzer::new(&config);
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_snapshot_recording() {
        let config = AnalyticsConfig::default();
        let mut analyzer = PerformanceAnalyzer::new(&config).unwrap();

        let snapshot = PerformanceSnapshot {
            timestamp: Utc::now(),
            ops_per_second: 500.0,
            avg_response_time_ms: 2.0,
            memory_usage_bytes: 1024 * 1024,
            cpu_usage_percent: 45.0,
            active_connections: 10,
            cache_hit_rate: 0.85,
            error_rate: 0.001,
        };

        let result = analyzer.record_snapshot(snapshot).await;
        assert!(result.is_ok());
        assert_eq!(analyzer.snapshots.len(), 1);
    }

    #[tokio::test]
    async fn test_trend_calculation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = PerformanceAnalyzer::new(&config).unwrap();

        // Add multiple snapshots to establish a trend
        for i in 0..15 {
            let snapshot = PerformanceSnapshot {
                timestamp: Utc::now(),
                ops_per_second: 500.0 + i as f64 * 10.0, // Increasing trend
                avg_response_time_ms: 2.0,
                memory_usage_bytes: 1024 * 1024,
                cpu_usage_percent: 45.0,
                active_connections: 10,
                cache_hit_rate: 0.85,
                error_rate: 0.001,
            };
            analyzer.record_snapshot(snapshot).await.unwrap();
        }

        // Should have calculated trends
        assert!(!analyzer.trends.is_empty());
        
        if let Some(ops_trend) = analyzer.trends.get("ops_per_second") {
            assert_eq!(ops_trend.trend_direction, TrendDirection::Improving);
        }
    }

    #[tokio::test]
    async fn test_bottleneck_detection() {
        let config = AnalyticsConfig::default();
        let mut analyzer = PerformanceAnalyzer::new(&config).unwrap();

        // Create a snapshot with high CPU usage
        let snapshot = PerformanceSnapshot {
            timestamp: Utc::now(),
            ops_per_second: 500.0,
            avg_response_time_ms: 2.0,
            memory_usage_bytes: 1024 * 1024,
            cpu_usage_percent: 95.0, // High CPU usage
            active_connections: 10,
            cache_hit_rate: 0.85,
            error_rate: 0.001,
        };

        analyzer.record_snapshot(snapshot).await.unwrap();

        // Should detect CPU bottleneck
        assert!(!analyzer.bottlenecks.is_empty());
        assert!(analyzer.bottlenecks.iter().any(|b| b.bottleneck_type == BottleneckType::CPU));
    }

    #[tokio::test]
    async fn test_recommendation_generation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = PerformanceAnalyzer::new(&config).unwrap();

        // Add a degrading trend
        let trend = PerformanceTrend {
            metric_name: "avg_response_time_ms".to_string(),
            trend_direction: TrendDirection::Degrading,
            trend_strength: 0.8,
            change_rate: 0.5,
            analysis_period: Duration::hours(1),
            confidence: 0.9,
        };
        analyzer.trends.insert("avg_response_time_ms".to_string(), trend);

        let recommendations = analyzer.generate_recommendations().await.unwrap();
        assert!(!recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_insight_generation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = PerformanceAnalyzer::new(&config).unwrap();

        // Add a high-confidence trend
        let trend = PerformanceTrend {
            metric_name: "ops_per_second".to_string(),
            trend_direction: TrendDirection::Improving,
            trend_strength: 0.9,
            change_rate: 10.0,
            analysis_period: Duration::hours(1),
            confidence: 0.95,
        };
        analyzer.trends.insert("ops_per_second".to_string(), trend);

        let insights = analyzer.generate_insights().await.unwrap();
        assert!(!insights.is_empty());
    }
}
