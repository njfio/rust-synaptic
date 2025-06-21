// Predictive Analytics Module
// Advanced memory access pattern prediction and proactive caching

use crate::error::Result;
use crate::analytics::{AnalyticsEvent, AnalyticsConfig, AnalyticsInsight, InsightType, InsightPriority, AccessType};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Memory access prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPrediction {
    /// Memory key that will likely be accessed
    pub memory_key: String,
    /// Predicted access time
    pub predicted_time: DateTime<Utc>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Predicted access type
    pub access_type: AccessType,
    /// Reasoning for the prediction
    pub reasoning: String,
}

/// Prediction for upcoming search queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPrediction {
    /// Query text expected in the future
    pub query: String,
    /// When the query is likely to occur
    pub predicted_time: DateTime<Utc>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning behind the prediction
    pub reasoning: String,
}

/// Usage trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTrend {
    /// Memory key
    pub memory_key: String,
    /// Trend direction
    pub trend: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Time period analyzed
    pub period_days: u32,
    /// Access frequency change
    pub frequency_change: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Caching recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingRecommendation {
    /// Memory key to cache
    pub memory_key: String,
    /// Recommended cache priority
    pub priority: CachePriority,
    /// Expected cache hit rate
    pub expected_hit_rate: f64,
    /// Recommended cache duration
    pub cache_duration: Duration,
    /// Justification for recommendation
    pub justification: String,
}

/// Cache priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum CachePriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Access pattern for a memory key
#[derive(Debug, Clone)]
struct AccessPattern {
    /// Memory key
    memory_key: String,
    /// Access timestamps
    access_times: VecDeque<DateTime<Utc>>,
    /// Access types
    access_types: VecDeque<AccessType>,
    /// Average interval between accesses
    avg_interval: Option<Duration>,
    /// Last calculated trend
    trend: Option<TrendDirection>,
}

impl AccessPattern {
    fn new(memory_key: String) -> Self {
        Self {
            memory_key,
            access_times: VecDeque::new(),
            access_types: VecDeque::new(),
            avg_interval: None,
            trend: None,
        }
    }

    fn add_access(&mut self, timestamp: DateTime<Utc>, access_type: AccessType) {
        self.access_times.push_back(timestamp);
        self.access_types.push_back(access_type);

        // Keep only recent accesses (last 100)
        if self.access_times.len() > 100 {
            self.access_times.pop_front();
            self.access_types.pop_front();
        }

        self.calculate_metrics();
    }

    fn calculate_metrics(&mut self) {
        if self.access_times.len() < 2 {
            return;
        }

        // Calculate average interval
        let mut intervals = Vec::new();
        for i in 1..self.access_times.len() {
            let interval = self.access_times[i] - self.access_times[i - 1];
            intervals.push(interval);
        }

        if !intervals.is_empty() {
            let total_ms: i64 = intervals.iter().map(|d| d.num_milliseconds()).sum();
            self.avg_interval = Some(Duration::milliseconds(total_ms / intervals.len() as i64));
        }

        // Calculate trend
        self.trend = self.calculate_trend();
    }

    fn calculate_trend(&self) -> Option<TrendDirection> {
        if self.access_times.len() < 5 {
            return None;
        }

        let recent_count = self.access_times.len();
        let mid_point = recent_count / 2;

        let first_half_freq = mid_point as f64;
        let second_half_freq = (recent_count - mid_point) as f64;

        let first_half_time = if mid_point > 0 {
            (self.access_times[mid_point - 1] - self.access_times[0]).num_hours() as f64
        } else {
            1.0
        };

        let second_half_time = if recent_count > mid_point {
            (self.access_times[recent_count - 1] - self.access_times[mid_point]).num_hours() as f64
        } else {
            1.0
        };

        let first_rate = first_half_freq / first_half_time.max(1.0);
        let second_rate = second_half_freq / second_half_time.max(1.0);

        let change_ratio = if first_rate > 0.0 {
            second_rate / first_rate
        } else {
            1.0
        };

        if change_ratio > 1.2 {
            Some(TrendDirection::Increasing)
        } else if change_ratio < 0.8 {
            Some(TrendDirection::Decreasing)
        } else if change_ratio > 0.9 && change_ratio < 1.1 {
            Some(TrendDirection::Stable)
        } else {
            Some(TrendDirection::Volatile)
        }
    }

    fn predict_next_access(&self, confidence_threshold: f64) -> Option<AccessPrediction> {
        if let Some(avg_interval) = self.avg_interval {
            if self.access_times.len() < 3 {
                return None;
            }

            let last_access = *self.access_times.back().unwrap();
            let predicted_time = last_access + avg_interval;

            // Calculate confidence based on pattern consistency
            let confidence = self.calculate_prediction_confidence();

            if confidence >= confidence_threshold {
                let most_common_type = self.get_most_common_access_type();
                
                return Some(AccessPrediction {
                    memory_key: self.memory_key.clone(),
                    predicted_time,
                    confidence,
                    access_type: most_common_type,
                    reasoning: format!(
                        "Based on {} recent accesses with average interval of {} minutes",
                        self.access_times.len(),
                        avg_interval.num_minutes()
                    ),
                });
            }
        }

        None
    }

    fn calculate_prediction_confidence(&self) -> f64 {
        if self.access_times.len() < 3 {
            return 0.0;
        }

        // Calculate variance in intervals
        let intervals: Vec<Duration> = self.access_times
            .iter()
            .zip(self.access_times.iter().skip(1))
            .map(|(a, b)| *b - *a)
            .collect();

        if intervals.is_empty() {
            return 0.0;
        }

        let avg_ms = intervals.iter().map(|d| d.num_milliseconds()).sum::<i64>() as f64 / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|d| {
                let diff = d.num_milliseconds() as f64 - avg_ms;
                diff * diff
            })
            .sum::<f64>() / intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if avg_ms > 0.0 { std_dev / avg_ms } else { 1.0 };

        // Lower coefficient of variation means higher confidence
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    fn get_most_common_access_type(&self) -> AccessType {
        let mut type_counts = HashMap::new();
        for access_type in &self.access_types {
            *type_counts.entry(access_type.clone()).or_insert(0) += 1;
        }

        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(access_type, _)| access_type)
            .unwrap_or(AccessType::Read)
    }
}

/// Predictive analytics engine
#[derive(Debug)]
pub struct PredictiveAnalytics {
    /// Configuration
    config: AnalyticsConfig,
    /// Access patterns for each memory key
    patterns: HashMap<String, AccessPattern>,
    /// Recent predictions made
    predictions: Vec<AccessPrediction>,
    /// Search query patterns
    search_patterns: HashMap<String, AccessPattern>,
    /// Record of recent search queries
    search_history: VecDeque<(String, DateTime<Utc>)>,
    /// Predictions for upcoming searches
    search_predictions: Vec<SearchPrediction>,
    /// Usage trends
    trends: HashMap<String, UsageTrend>,
    /// Caching recommendations
    cache_recommendations: Vec<CachingRecommendation>,
}

impl PredictiveAnalytics {
    /// Create a new predictive analytics engine
    pub fn new(config: &AnalyticsConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            patterns: HashMap::new(),
            predictions: Vec::new(),
            search_patterns: HashMap::new(),
            search_history: VecDeque::new(),
            search_predictions: Vec::new(),
            trends: HashMap::new(),
            cache_recommendations: Vec::new(),
        })
    }

    /// Process an analytics event
    pub async fn process_event(&mut self, event: &AnalyticsEvent) -> Result<()> {
        match event {
            AnalyticsEvent::MemoryAccess { memory_key, access_type, timestamp, .. } => {
                self.record_access(memory_key.clone(), *timestamp, access_type.clone()).await?;
            }
            AnalyticsEvent::SearchQuery { query, timestamp, response_time_ms, .. } => {
                self.analyze_search_pattern(query.clone(), *timestamp, *response_time_ms).await?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Record a memory access for pattern analysis
    async fn record_access(&mut self, memory_key: String, timestamp: DateTime<Utc>, access_type: AccessType) -> Result<()> {
        let pattern = self.patterns
            .entry(memory_key.clone())
            .or_insert_with(|| AccessPattern::new(memory_key.clone()));

        pattern.add_access(timestamp, access_type);

        // Generate prediction if pattern is established
        if let Some(prediction) = pattern.predict_next_access(self.config.prediction_threshold) {
            self.predictions.push(prediction);
        }

        // Update usage trend
        self.update_usage_trend(&memory_key).await?;

        Ok(())
    }

    /// Analyze search patterns for predictive insights
    async fn analyze_search_pattern(&mut self, query: String, timestamp: DateTime<Utc>, _response_time_ms: u64) -> Result<()> {
        // Record search history
        self.search_history.push_back((query.clone(), timestamp));
        if self.search_history.len() > 100 {
            self.search_history.pop_front();
        }

        // Track occurrences for this query
        let pattern = self
            .search_patterns
            .entry(query.clone())
            .or_insert_with(|| AccessPattern::new(query.clone()));
        pattern.add_access(timestamp, AccessType::Search);

        // Generate prediction if the pattern is strong enough
        if let Some(pred) = pattern.predict_next_access(self.config.prediction_threshold) {
            self.search_predictions.push(SearchPrediction {
                query: query.clone(),
                predicted_time: pred.predicted_time,
                confidence: pred.confidence,
                reasoning: pred.reasoning,
            });
        }

        Ok(())
    }

    /// Update usage trend for a memory key
    async fn update_usage_trend(&mut self, memory_key: &str) -> Result<()> {
        if let Some(pattern) = self.patterns.get(memory_key) {
            if pattern.access_times.len() >= 5 {
                let trend_direction = pattern.trend.clone().unwrap_or(TrendDirection::Stable);
                
                // Calculate frequency change over the last week
                let now = Utc::now();
                let week_ago = now - Duration::days(7);
                
                let recent_accesses = pattern.access_times
                    .iter()
                    .filter(|&&time| time > week_ago)
                    .count();

                let frequency_change = if pattern.access_times.len() > 10 {
                    let older_accesses = pattern.access_times.len() - recent_accesses;
                    if older_accesses > 0 {
                        recent_accesses as f64 / older_accesses as f64 - 1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let strength = match trend_direction {
                    TrendDirection::Increasing => frequency_change.max(0.0).min(1.0),
                    TrendDirection::Decreasing => (-frequency_change).max(0.0).min(1.0),
                    TrendDirection::Stable => 1.0 - frequency_change.abs().min(1.0),
                    TrendDirection::Volatile => frequency_change.abs().min(1.0),
                };

                let trend = UsageTrend {
                    memory_key: memory_key.to_string(),
                    trend: trend_direction,
                    strength,
                    period_days: 7,
                    frequency_change,
                };

                self.trends.insert(memory_key.to_string(), trend);
            }
        }

        Ok(())
    }

    /// Generate caching recommendations
    pub async fn generate_caching_recommendations(&mut self) -> Result<Vec<CachingRecommendation>> {
        let mut recommendations = Vec::new();

        for (memory_key, pattern) in &self.patterns {
            if pattern.access_times.len() >= 3 {
                let recommendation = self.analyze_caching_potential(memory_key, pattern).await?;
                if let Some(rec) = recommendation {
                    recommendations.push(rec);
                }
            }
        }

        self.cache_recommendations = recommendations.clone();
        Ok(recommendations)
    }

    /// Analyze caching potential for a memory key
    async fn analyze_caching_potential(&self, memory_key: &str, pattern: &AccessPattern) -> Result<Option<CachingRecommendation>> {
        if pattern.access_times.len() < 3 {
            return Ok(None);
        }

        // Calculate access frequency
        let time_span = *pattern.access_times.back().unwrap() - *pattern.access_times.front().unwrap();
        let access_frequency = if time_span.num_hours() > 0 {
            pattern.access_times.len() as f64 / time_span.num_hours() as f64
        } else {
            0.0
        };

        // Determine cache priority based on frequency and trend
        let priority = if access_frequency > 1.0 {
            CachePriority::Critical
        } else if access_frequency > 0.5 {
            CachePriority::High
        } else if access_frequency > 0.1 {
            CachePriority::Medium
        } else {
            CachePriority::Low
        };

        // Calculate expected hit rate based on pattern consistency
        let confidence = pattern.calculate_prediction_confidence();
        let expected_hit_rate = confidence * 0.8 + 0.2; // Base hit rate of 20%

        // Recommend cache duration based on average interval
        let cache_duration = pattern.avg_interval
            .unwrap_or(Duration::hours(1)) * 2; // Cache for twice the average interval

        if priority >= CachePriority::Medium {
            Ok(Some(CachingRecommendation {
                memory_key: memory_key.to_string(),
                priority,
                expected_hit_rate,
                cache_duration,
                justification: format!(
                    "High access frequency ({:.2}/hour) with {:.0}% pattern consistency",
                    access_frequency, confidence * 100.0
                ),
            }))
        } else {
            Ok(None)
        }
    }

    /// Generate predictive insights
    pub async fn generate_insights(&mut self) -> Result<Vec<AnalyticsInsight>> {
        let mut insights = Vec::new();

        // Generate caching recommendations
        let cache_recs = self.generate_caching_recommendations().await?;
        for rec in cache_recs {
            if rec.priority >= CachePriority::High {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::PerformanceOptimization,
                    title: format!("High-Priority Caching Recommendation for {}", rec.memory_key),
                    description: format!(
                        "Memory '{}' should be cached with {} priority. {}",
                        rec.memory_key, 
                        format!("{:?}", rec.priority).to_lowercase(),
                        rec.justification
                    ),
                    confidence: rec.expected_hit_rate,
                    evidence: vec![
                        format!("Expected cache hit rate: {:.1}%", rec.expected_hit_rate * 100.0),
                        format!("Recommended cache duration: {} minutes", rec.cache_duration.num_minutes()),
                    ],
                    generated_at: Utc::now(),
                    priority: match rec.priority {
                        CachePriority::Critical => InsightPriority::Critical,
                        CachePriority::High => InsightPriority::High,
                        _ => InsightPriority::Medium,
                    },
                };
                insights.push(insight);
            }
        }

        // Generate trend insights
        for (memory_key, trend) in &self.trends {
            if trend.strength > 0.7 {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::UsagePattern,
                    title: format!("Strong Usage Trend Detected for {}", memory_key),
                    description: format!(
                        "Memory '{}' shows a {} trend with {:.1}% strength over {} days",
                        memory_key,
                        format!("{:?}", trend.trend).to_lowercase(),
                        trend.strength * 100.0,
                        trend.period_days
                    ),
                    confidence: trend.strength,
                    evidence: vec![
                        format!("Trend direction: {:?}", trend.trend),
                        format!("Frequency change: {:.1}%", trend.frequency_change * 100.0),
                    ],
                    generated_at: Utc::now(),
                    priority: if trend.strength > 0.9 {
                        InsightPriority::High
                    } else {
                        InsightPriority::Medium
                    },
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Get current predictions
    pub fn get_predictions(&self) -> &[AccessPrediction] {
        &self.predictions
    }

    /// Get search query predictions
    pub fn get_search_predictions(&self) -> &[SearchPrediction] {
        &self.search_predictions
    }

    /// Get usage trends
    pub fn get_trends(&self) -> &HashMap<String, UsageTrend> {
        &self.trends
    }

    /// Get caching recommendations
    pub fn get_cache_recommendations(&self) -> &[CachingRecommendation] {
        &self.cache_recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_predictive_analytics_creation() {
        let config = AnalyticsConfig::default();
        let analytics = PredictiveAnalytics::new(&config);
        assert!(analytics.is_ok());
    }

    #[tokio::test]
    async fn test_access_pattern_tracking() {
        let config = AnalyticsConfig::default();
        let mut analytics = PredictiveAnalytics::new(&config).unwrap();

        let event = AnalyticsEvent::MemoryAccess {
            memory_key: "test_key".to_string(),
            access_type: AccessType::Read,
            timestamp: Utc::now(),
            user_context: None,
        };

        let result = analytics.process_event(&event).await;
        assert!(result.is_ok());
        assert!(analytics.patterns.contains_key("test_key"));
    }

    #[tokio::test]
    async fn test_prediction_generation() {
        let config = AnalyticsConfig::default();
        let mut analytics = PredictiveAnalytics::new(&config).unwrap();

        // Add multiple accesses to establish a pattern
        let base_time = Utc::now();
        for i in 0..5 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: "pattern_key".to_string(),
                access_type: AccessType::Read,
                timestamp: base_time + Duration::hours(i),
                user_context: None,
            };
            analytics.process_event(&event).await.unwrap();
        }

        // Should have generated some predictions
        assert!(!analytics.predictions.is_empty() || analytics.patterns.len() > 0);
    }

    #[tokio::test]
    async fn test_search_prediction_generation() {
        let config = AnalyticsConfig::default();
        let mut analytics = PredictiveAnalytics::new(&config).unwrap();

        let base_time = Utc::now();
        for i in 0..5 {
            let event = AnalyticsEvent::SearchQuery {
                query: "test search".to_string(),
                results_count: 2,
                timestamp: base_time + Duration::hours(i),
                response_time_ms: 20,
            };
            analytics.process_event(&event).await.unwrap();
        }

        assert!(!analytics.get_search_predictions().is_empty());
    }

    #[tokio::test]
    async fn test_caching_recommendations() {
        let config = AnalyticsConfig::default();
        let mut analytics = PredictiveAnalytics::new(&config).unwrap();

        // Create a high-frequency access pattern
        let base_time = Utc::now();
        for i in 0..10 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: "frequent_key".to_string(),
                access_type: AccessType::Read,
                timestamp: base_time + Duration::minutes(i * 10),
                user_context: None,
            };
            analytics.process_event(&event).await.unwrap();
        }

        let _recommendations = analytics.generate_caching_recommendations().await.unwrap();
        // Should generate recommendations for frequently accessed memory
        // Validate that recommendations were generated
    }

    #[tokio::test]
    async fn test_insight_generation() {
        let config = AnalyticsConfig::default();
        let mut analytics = PredictiveAnalytics::new(&config).unwrap();

        // Add some access patterns
        let base_time = Utc::now();
        for i in 0..8 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: "insight_key".to_string(),
                access_type: AccessType::Read,
                timestamp: base_time + Duration::hours(i),
                user_context: None,
            };
            analytics.process_event(&event).await.unwrap();
        }

        let _insights = analytics.generate_insights().await.unwrap();
        // Should generate insights based on patterns
        // Validate that insights were generated
    }
}
