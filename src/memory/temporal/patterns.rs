//! Temporal pattern detection and analysis

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use crate::memory::temporal::TimeRange;
use chrono::{DateTime, Utc, Duration, Weekday, Timelike, Datelike};
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Detected temporal pattern in memory access or creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Unique pattern identifier
    pub id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence in pattern detection (0.0 to 1.0)
    pub confidence: f64,
    /// Time range where pattern was detected
    pub time_range: TimeRange,
    /// Pattern description
    pub description: String,
    /// Supporting evidence
    pub evidence: Vec<PatternEvidence>,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
}

/// Types of temporal patterns
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Daily recurring pattern
    Daily,
    /// Weekly recurring pattern
    Weekly,
    /// Monthly recurring pattern
    Monthly,
    /// Seasonal pattern
    Seasonal,
    /// Burst pattern (high activity in short period)
    Burst,
    /// Gradual increase pattern
    GradualIncrease,
    /// Gradual decrease pattern
    GradualDecrease,
    /// Cyclical pattern with custom period
    Cyclical { period_hours: u64 },
    /// Irregular but significant pattern
    Irregular,
    /// Custom pattern type
    Custom(String),
}

/// Evidence supporting a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEvidence {
    /// Timestamp of the evidence
    pub timestamp: DateTime<Utc>,
    /// Memory key involved
    pub memory_key: String,
    /// Type of activity
    pub activity_type: ActivityType,
    /// Strength of this evidence
    pub strength: f64,
}

/// Types of memory activities
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityType {
    Creation,
    Access,
    Update,
    Search,
    Summarization,
    Deletion,
}

/// Temporal trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTrend {
    /// Trend identifier
    pub id: String,
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength
    pub strength: f64,
    /// Time period analyzed
    pub time_period: TimeRange,
    /// Data points supporting the trend
    pub data_points: Vec<TrendDataPoint>,
    /// Statistical significance
    pub significance: f64,
}

/// Direction of a temporal trend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Cyclical,
}

/// Data point for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub context: String,
}

/// Access pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    /// Memory key
    pub memory_key: String,
    /// Access frequency by hour of day
    pub hourly_distribution: [u32; 24],
    /// Access frequency by day of week
    pub daily_distribution: [u32; 7],
    /// Peak access times
    pub peak_times: Vec<DateTime<Utc>>,
    /// Access clustering information
    pub clustering_info: ClusteringInfo,
}

/// Information about access clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringInfo {
    /// Number of clusters identified
    pub cluster_count: usize,
    /// Average cluster size
    pub avg_cluster_size: f64,
    /// Time between clusters
    pub inter_cluster_time: Duration,
}

/// Pattern detector for identifying temporal patterns
pub struct PatternDetector {
    /// Detected patterns
    patterns: Vec<TemporalPattern>,
    /// Configuration
    config: PatternDetectionConfig,
    /// Pattern detection history
    detection_history: Vec<PatternDetectionRun>,
}

/// Configuration for pattern detection
#[derive(Debug, Clone)]
pub struct PatternDetectionConfig {
    /// Minimum pattern strength to report
    pub min_pattern_strength: f64,
    /// Minimum confidence to report
    pub min_confidence: f64,
    /// Look-back period for pattern detection
    pub lookback_days: u64,
    /// Minimum data points required for pattern
    pub min_data_points: usize,
    /// Enable advanced statistical analysis
    pub enable_statistical_analysis: bool,
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            min_pattern_strength: 0.3,
            min_confidence: 0.6,
            lookback_days: 30,
            min_data_points: 5,
            enable_statistical_analysis: true,
        }
    }
}

/// Record of a pattern detection run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectionRun {
    /// When the detection was run
    pub timestamp: DateTime<Utc>,
    /// Time range analyzed
    pub analyzed_range: TimeRange,
    /// Number of patterns found
    pub patterns_found: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Decision rule for pattern detection
#[derive(Debug, Clone)]
struct DecisionRule {
    pub id: String,
    pub description: String,
    pub conditions: Vec<RuleCondition>,
    pub confidence: f64,
}

/// Rule condition for decision tree-like pattern detection
#[derive(Debug, Clone)]
enum RuleCondition {
    HourRange(u32, u32),
    WeekendOnly,
    StrengthThreshold(f64),
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            config: PatternDetectionConfig::default(),
            detection_history: Vec::new(),
        }
    }

    /// Update patterns based on new memory activity
    pub async fn update_patterns(&mut self, memory_key: &str, _memory: &MemoryEntry) -> Result<()> {
        // Record the activity
        let evidence = PatternEvidence {
            timestamp: Utc::now(),
            memory_key: memory_key.to_string(),
            activity_type: ActivityType::Access, // This should be determined by context
            strength: 1.0,
        };

        // Update existing patterns or detect new ones
        self.update_existing_patterns(&evidence).await?;
        
        Ok(())
    }

    /// Detect patterns within a time range using advanced algorithms
    pub async fn detect_patterns_in_range(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut detected_patterns = Vec::new();

        // Basic pattern detection
        if let Some(daily_pattern) = self.detect_daily_pattern(time_range).await? {
            detected_patterns.push(daily_pattern);
        }

        if let Some(weekly_pattern) = self.detect_weekly_pattern(time_range).await? {
            detected_patterns.push(weekly_pattern);
        }

        // Advanced pattern detection
        let burst_patterns = self.detect_burst_patterns(time_range).await?;
        detected_patterns.extend(burst_patterns);

        let trend_patterns = self.detect_trend_patterns(time_range).await?;
        detected_patterns.extend(trend_patterns);

        // Seasonal patterns with advanced detection
        let seasonal_patterns = self.detect_seasonal_patterns_advanced(time_range).await?;
        detected_patterns.extend(seasonal_patterns);

        // Cyclical patterns with variable periods using FFT-like analysis
        let cyclical_patterns = self.detect_cyclical_patterns_advanced(time_range).await?;
        detected_patterns.extend(cyclical_patterns);

        // Anomaly detection using statistical methods
        let anomaly_patterns = self.detect_anomaly_patterns(time_range).await?;
        detected_patterns.extend(anomaly_patterns);

        // Machine learning-based pattern recognition
        let ml_patterns = self.detect_ml_patterns(time_range).await?;
        detected_patterns.extend(ml_patterns);

        // Complex multi-dimensional patterns
        let complex_patterns = self.detect_complex_patterns(time_range).await?;
        detected_patterns.extend(complex_patterns);

        // Correlation-based patterns
        let correlation_patterns = self.detect_correlation_patterns(time_range).await?;
        detected_patterns.extend(correlation_patterns);

        // Filter by minimum strength and confidence
        detected_patterns.retain(|p| {
            p.strength >= self.config.min_pattern_strength &&
            p.confidence >= self.config.min_confidence
        });

        Ok(detected_patterns)
    }

    /// Detect daily recurring patterns
    async fn detect_daily_pattern(&self, time_range: &TimeRange) -> Result<Option<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points {
            return Ok(None);
        }

        let mut hourly: [u32; 24] = [0; 24];
        for ev in &evidence {
            hourly[ev.timestamp.hour() as usize] += 1;
        }

        let (peak_hour, &peak_count) = hourly
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .unwrap();
        if peak_count == 0 {
            return Ok(None);
        }

        let strength = peak_count as f64 / evidence.len() as f64;
        let mut metadata = HashMap::new();
        metadata.insert("hour_of_day".into(), peak_hour.to_string());

        let pattern = TemporalPattern {
            id: format!("daily_{peak_hour}"),
            pattern_type: PatternType::Daily,
            strength,
            confidence: strength,
            time_range: time_range.clone(),
            description: format!("Activity peaks around {:02}:00", peak_hour),
            evidence: evidence
                .into_iter()
                .filter(|e| e.timestamp.hour() as usize == peak_hour)
                .collect(),
            metadata,
        };

        Ok(Some(pattern))
    }

    /// Detect weekly recurring patterns
    async fn detect_weekly_pattern(&self, time_range: &TimeRange) -> Result<Option<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points {
            return Ok(None);
        }

        let mut daily: [u32; 7] = [0; 7];
        for ev in &evidence {
            daily[ev.timestamp.weekday().num_days_from_monday() as usize] += 1;
        }

        let (peak_day_idx, &peak_count) = daily
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .unwrap();
        if peak_count == 0 {
            return Ok(None);
        }

        let strength = peak_count as f64 / evidence.len() as f64;
        let weekday = Weekday::from_u64(peak_day_idx as u64).unwrap_or(Weekday::Mon);
        let mut metadata = HashMap::new();
        metadata.insert("day_of_week".into(), weekday.num_days_from_monday().to_string());

        let pattern = TemporalPattern {
            id: format!("weekly_{:?}", weekday),
            pattern_type: PatternType::Weekly,
            strength,
            confidence: strength,
            time_range: time_range.clone(),
            description: format!("Activity peaks on {:?}", weekday),
            evidence: evidence
                .into_iter()
                .filter(|e| e.timestamp.weekday() == weekday)
                .collect(),
            metadata,
        };

        Ok(Some(pattern))
    }

    /// Detect burst patterns (high activity in short periods)
    async fn detect_burst_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points {
            return Ok(Vec::new());
        }

        evidence.sort_by_key(|e| e.timestamp);
        let mut clusters: Vec<Vec<PatternEvidence>> = Vec::new();
        for ev in evidence {
            if let Some(cluster) = clusters.last_mut() {
                if ev.timestamp - cluster.last().unwrap().timestamp <= Duration::minutes(60) {
                    cluster.push(ev);
                    continue;
                }
            }
            clusters.push(vec![ev]);
        }

        let mut patterns = Vec::new();
        for cluster in clusters.into_iter().filter(|c| c.len() as usize >= self.config.min_data_points) {
            let start = cluster.first().unwrap().timestamp;
            let end = cluster.last().unwrap().timestamp;
            let strength = cluster.len() as f64 / self.config.min_data_points as f64;
            let pattern = TemporalPattern {
                id: format!("burst_{}_{}", start.timestamp(), end.timestamp()),
                pattern_type: PatternType::Burst,
                strength: strength.min(1.0),
                confidence: 0.8,
                time_range: TimeRange::new(start, end),
                description: format!("Burst of {} events", cluster.len()),
                evidence: cluster,
                metadata: HashMap::new(),
            };
            patterns.push(pattern);
        }
        Ok(patterns)
    }

    /// Detect trend patterns (gradual changes over time)
    async fn detect_trend_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        use std::collections::BTreeMap;
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points {
            return Ok(Vec::new());
        }

        let mut daily_counts: BTreeMap<i64, u32> = BTreeMap::new();
        for ev in &evidence {
            let day = ev.timestamp.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
            *daily_counts.entry(day).or_insert(0) += 1;
        }

        if daily_counts.len() < 2 {
            return Ok(Vec::new());
        }

        let n = daily_counts.len() as f64;
        let times: Vec<f64> = (0..daily_counts.len()).map(|i| i as f64).collect();
        let values: Vec<f64> = daily_counts.values().map(|&v| v as f64).collect();
        let mean_x = (n - 1.0) / 2.0;
        let mean_y = values.iter().sum::<f64>() / n;
        let numerator: f64 = times
            .iter()
            .zip(values.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let denominator: f64 = times.iter().map(|x| (x - mean_x).powi(2)).sum();
        if denominator == 0.0 {
            return Ok(Vec::new());
        }
        let slope = numerator / denominator;

        let pattern_type = if slope > 0.0 {
            PatternType::GradualIncrease
        } else {
            PatternType::GradualDecrease
        };

        let strength = slope.abs().min(1.0);
        let start = *daily_counts.keys().next().unwrap();
        let end = *daily_counts.keys().last().unwrap();
        let pattern = TemporalPattern {
            id: format!("trend_{:?}", pattern_type),
            pattern_type,
            strength,
            confidence: strength,
            time_range: TimeRange::new(DateTime::from_timestamp(start, 0).unwrap(),
                                         DateTime::from_timestamp(end, 0).unwrap()),
            description: "Gradual trend detected".to_string(),
            evidence,
            metadata: HashMap::new(),
        };
        Ok(vec![pattern])
    }

    /// Detect seasonal patterns using advanced statistical analysis
    async fn detect_seasonal_patterns_advanced(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points * 4 { // Need more data for seasonal analysis
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Monthly seasonal analysis
        let monthly_patterns = self.detect_monthly_seasonal_patterns(&evidence, time_range).await?;
        patterns.extend(monthly_patterns);

        // Quarterly seasonal analysis
        let quarterly_patterns = self.detect_quarterly_seasonal_patterns(&evidence, time_range).await?;
        patterns.extend(quarterly_patterns);

        // Custom seasonal periods (e.g., academic year, fiscal year)
        let custom_seasonal_patterns = self.detect_custom_seasonal_patterns(&evidence, time_range).await?;
        patterns.extend(custom_seasonal_patterns);

        Ok(patterns)
    }

    /// Detect monthly seasonal patterns
    async fn detect_monthly_seasonal_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut monthly_counts = [0u32; 12];
        for ev in evidence {
            monthly_counts[(ev.timestamp.month() - 1) as usize] += 1;
        }

        let total_evidence = evidence.len() as f64;
        let expected_per_month = total_evidence / 12.0;

        let mut patterns = Vec::new();

        // Find months with significantly higher activity
        for (month_idx, &count) in monthly_counts.iter().enumerate() {
            let deviation = (count as f64 - expected_per_month) / expected_per_month;
            if deviation > 0.5 { // 50% above average
                let strength = deviation.min(1.0);
                let confidence = self.calculate_seasonal_confidence(count as f64, total_evidence, 12.0);

                let mut metadata = HashMap::new();
                metadata.insert("month".to_string(), (month_idx + 1).to_string());
                metadata.insert("deviation".to_string(), deviation.to_string());

                let pattern = TemporalPattern {
                    id: format!("seasonal_monthly_{}", month_idx + 1),
                    pattern_type: PatternType::Seasonal,
                    strength,
                    confidence,
                    time_range: time_range.clone(),
                    description: format!("Seasonal peak in month {}", month_idx + 1),
                    evidence: evidence.iter()
                        .filter(|e| e.timestamp.month() == (month_idx + 1) as u32)
                        .cloned()
                        .collect(),
                    metadata,
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Detect quarterly seasonal patterns
    async fn detect_quarterly_seasonal_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut quarterly_counts = [0u32; 4];
        for ev in evidence {
            let quarter = ((ev.timestamp.month() - 1) / 3) as usize;
            quarterly_counts[quarter] += 1;
        }

        let total_evidence = evidence.len() as f64;
        let expected_per_quarter = total_evidence / 4.0;

        let mut patterns = Vec::new();

        for (quarter_idx, &count) in quarterly_counts.iter().enumerate() {
            let deviation = (count as f64 - expected_per_quarter) / expected_per_quarter;
            if deviation > 0.3 { // 30% above average
                let strength = deviation.min(1.0);
                let confidence = self.calculate_seasonal_confidence(count as f64, total_evidence, 4.0);

                let mut metadata = HashMap::new();
                metadata.insert("quarter".to_string(), (quarter_idx + 1).to_string());

                let pattern = TemporalPattern {
                    id: format!("seasonal_quarterly_{}", quarter_idx + 1),
                    pattern_type: PatternType::Seasonal,
                    strength,
                    confidence,
                    time_range: time_range.clone(),
                    description: format!("Seasonal peak in Q{}", quarter_idx + 1),
                    evidence: evidence.iter()
                        .filter(|e| ((e.timestamp.month() - 1) / 3) as usize == quarter_idx)
                        .cloned()
                        .collect(),
                    metadata,
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Detect custom seasonal patterns (e.g., academic year, fiscal year)
    async fn detect_custom_seasonal_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Academic year pattern (September to June)
        let academic_pattern = self.detect_academic_year_pattern(evidence, time_range).await?;
        if let Some(pattern) = academic_pattern {
            patterns.push(pattern);
        }

        // Fiscal year pattern (varies by organization, using April-March as example)
        let fiscal_pattern = self.detect_fiscal_year_pattern(evidence, time_range).await?;
        if let Some(pattern) = fiscal_pattern {
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Detect academic year seasonal pattern
    async fn detect_academic_year_pattern(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Option<TemporalPattern>> {
        let academic_months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]; // Sep-Jun
        let mut academic_count = 0;
        let mut non_academic_count = 0;

        for ev in evidence {
            if academic_months.contains(&ev.timestamp.month()) {
                academic_count += 1;
            } else {
                non_academic_count += 1;
            }
        }

        let total = academic_count + non_academic_count;
        if total == 0 {
            return Ok(None);
        }

        let academic_ratio = academic_count as f64 / total as f64;
        let expected_ratio = 10.0 / 12.0; // 10 months out of 12

        if academic_ratio > expected_ratio + 0.1 { // 10% above expected
            let strength = ((academic_ratio - expected_ratio) / (1.0 - expected_ratio)).min(1.0);
            let confidence = self.calculate_seasonal_confidence(academic_count as f64, total as f64, expected_ratio);

            let mut metadata = HashMap::new();
            metadata.insert("academic_ratio".to_string(), academic_ratio.to_string());

            let pattern = TemporalPattern {
                id: "seasonal_academic_year".to_string(),
                pattern_type: PatternType::Seasonal,
                strength,
                confidence,
                time_range: time_range.clone(),
                description: "Academic year seasonal pattern (Sep-Jun)".to_string(),
                evidence: evidence.iter()
                    .filter(|e| academic_months.contains(&e.timestamp.month()))
                    .cloned()
                    .collect(),
                metadata,
            };

            return Ok(Some(pattern));
        }

        Ok(None)
    }

    /// Detect fiscal year seasonal pattern
    async fn detect_fiscal_year_pattern(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Option<TemporalPattern>> {
        // Using April-March fiscal year as example
        let fiscal_q1 = [4, 5, 6]; // Apr-Jun
        let fiscal_q4 = [1, 2, 3]; // Jan-Mar

        let mut q1_count = 0;
        let mut q4_count = 0;
        let mut other_count = 0;

        for ev in evidence {
            let month = ev.timestamp.month();
            if fiscal_q1.contains(&month) {
                q1_count += 1;
            } else if fiscal_q4.contains(&month) {
                q4_count += 1;
            } else {
                other_count += 1;
            }
        }

        let total = q1_count + q4_count + other_count;
        if total == 0 {
            return Ok(None);
        }

        // Check for end-of-fiscal-year pattern (high activity in Q4)
        let q4_ratio = q4_count as f64 / total as f64;
        let expected_q4_ratio = 3.0 / 12.0; // 3 months out of 12

        if q4_ratio > expected_q4_ratio + 0.1 {
            let strength = ((q4_ratio - expected_q4_ratio) / (1.0 - expected_q4_ratio)).min(1.0);
            let confidence = self.calculate_seasonal_confidence(q4_count as f64, total as f64, expected_q4_ratio);

            let mut metadata = HashMap::new();
            metadata.insert("fiscal_q4_ratio".to_string(), q4_ratio.to_string());

            let pattern = TemporalPattern {
                id: "seasonal_fiscal_year_end".to_string(),
                pattern_type: PatternType::Seasonal,
                strength,
                confidence,
                time_range: time_range.clone(),
                description: "Fiscal year-end seasonal pattern (Jan-Mar)".to_string(),
                evidence: evidence.iter()
                    .filter(|e| fiscal_q4.contains(&e.timestamp.month()))
                    .cloned()
                    .collect(),
                metadata,
            };

            return Ok(Some(pattern));
        }

        Ok(None)
    }

    /// Calculate confidence for seasonal patterns
    fn calculate_seasonal_confidence(&self, observed: f64, total: f64, expected_ratio: f64) -> f64 {
        if total == 0.0 {
            return 0.0;
        }

        let observed_ratio = observed / total;
        let deviation = (observed_ratio - expected_ratio).abs();
        let max_deviation = (1.0 - expected_ratio).max(expected_ratio);

        let statistical_confidence = 1.0 - (deviation / max_deviation);
        let sample_size_factor = (total / 100.0).min(1.0); // More confidence with more data

        (statistical_confidence * sample_size_factor).max(0.0).min(1.0)
    }

    /// Detect cyclical patterns with variable periods using spectral analysis
    async fn detect_cyclical_patterns_advanced(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points * 2 {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Convert evidence to time series
        let time_series = self.evidence_to_time_series(&evidence, time_range);

        // Detect periods using autocorrelation
        let autocorr_patterns = self.detect_periods_by_autocorrelation(&time_series, &evidence, time_range).await?;
        patterns.extend(autocorr_patterns);

        // Detect periods using peak analysis
        let peak_patterns = self.detect_periods_by_peak_analysis(&time_series, &evidence, time_range).await?;
        patterns.extend(peak_patterns);

        // Detect harmonic patterns
        let harmonic_patterns = self.detect_harmonic_patterns(&time_series, &evidence, time_range).await?;
        patterns.extend(harmonic_patterns);

        Ok(patterns)
    }

    /// Convert evidence to time series for analysis
    fn evidence_to_time_series(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Vec<f64> {
        let duration_hours = (time_range.end - time_range.start).num_hours() as usize;
        let mut time_series = vec![0.0; duration_hours];

        for ev in evidence {
            let hours_from_start = (ev.timestamp - time_range.start).num_hours() as usize;
            if hours_from_start < time_series.len() {
                time_series[hours_from_start] += ev.strength;
            }
        }

        time_series
    }

    /// Detect periods using autocorrelation analysis
    async fn detect_periods_by_autocorrelation(&self, time_series: &[f64], evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        if time_series.len() < 48 { // Need at least 48 hours of data
            return Ok(patterns);
        }

        // Calculate autocorrelation for different lags
        let max_lag = time_series.len() / 4; // Check up to 1/4 of the series length
        let mut autocorrelations = Vec::new();

        for lag in 1..=max_lag {
            let correlation = self.calculate_autocorrelation(time_series, lag);
            autocorrelations.push((lag, correlation));
        }

        // Find significant peaks in autocorrelation
        let mut peaks = Vec::new();
        for window in autocorrelations.windows(3) {
            let (lag, corr) = window[1];
            if corr > window[0].1 && corr > window[2].1 && corr > 0.3 { // Significant correlation
                peaks.push((lag, corr));
            }
        }

        // Create patterns for significant periods
        for (lag_hours, correlation) in peaks {
            let period_hours = lag_hours as u64;
            let strength = correlation;
            let confidence = correlation * 0.8; // Slightly lower confidence than strength

            let mut metadata = HashMap::new();
            metadata.insert("period_hours".to_string(), period_hours.to_string());
            metadata.insert("autocorrelation".to_string(), correlation.to_string());

            let pattern = TemporalPattern {
                id: format!("cyclical_autocorr_{}", period_hours),
                pattern_type: PatternType::Cyclical { period_hours },
                strength,
                confidence,
                time_range: time_range.clone(),
                description: format!("Cyclical pattern with {}-hour period", period_hours),
                evidence: evidence.to_vec(),
                metadata,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Calculate autocorrelation for a given lag
    fn calculate_autocorrelation(&self, time_series: &[f64], lag: usize) -> f64 {
        if lag >= time_series.len() {
            return 0.0;
        }

        let n = time_series.len() - lag;
        if n == 0 {
            return 0.0;
        }

        // Calculate means
        let mean1 = time_series[..n].iter().sum::<f64>() / n as f64;
        let mean2 = time_series[lag..].iter().sum::<f64>() / n as f64;

        // Calculate correlation
        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..n {
            let diff1 = time_series[i] - mean1;
            let diff2 = time_series[i + lag] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Detect periods using peak analysis
    async fn detect_periods_by_peak_analysis(&self, time_series: &[f64], evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Find peaks in the time series
        let peaks = self.find_peaks_in_time_series(time_series);
        if peaks.len() < 3 {
            return Ok(patterns);
        }

        // Calculate intervals between peaks
        let mut intervals = Vec::new();
        for window in peaks.windows(2) {
            intervals.push(window[1] - window[0]);
        }

        // Find common intervals (periods)
        let mut interval_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &interval in &intervals {
            *interval_counts.entry(interval).or_insert(0) += 1;
        }

        // Create patterns for frequent intervals
        for (&interval, &count) in &interval_counts {
            if count >= 3 && interval >= 2 { // At least 3 occurrences and minimum 2-hour period
                let strength = count as f64 / intervals.len() as f64;
                let confidence = strength * 0.9;

                let mut metadata = HashMap::new();
                metadata.insert("peak_interval_hours".to_string(), interval.to_string());
                metadata.insert("occurrences".to_string(), count.to_string());

                let pattern = TemporalPattern {
                    id: format!("cyclical_peaks_{}", interval),
                    pattern_type: PatternType::Cyclical { period_hours: interval as u64 },
                    strength,
                    confidence,
                    time_range: time_range.clone(),
                    description: format!("Cyclical pattern with {}-hour intervals between peaks", interval),
                    evidence: evidence.to_vec(),
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Find peaks in time series
    fn find_peaks_in_time_series(&self, time_series: &[f64]) -> Vec<usize> {
        let mut peaks = Vec::new();

        if time_series.len() < 3 {
            return peaks;
        }

        // Calculate moving average for baseline
        let window_size = 5.min(time_series.len());
        let mut moving_avg = Vec::new();
        for i in 0..time_series.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(time_series.len());
            let avg = time_series[start..end].iter().sum::<f64>() / (end - start) as f64;
            moving_avg.push(avg);
        }

        // Find peaks above moving average
        for i in 1..time_series.len() - 1 {
            let current = time_series[i];
            let prev = time_series[i - 1];
            let next = time_series[i + 1];
            let baseline = moving_avg[i];

            if current > prev && current > next && current > baseline * 1.2 {
                peaks.push(i);
            }
        }

        peaks
    }

    /// Detect harmonic patterns (multiples of fundamental frequencies)
    async fn detect_harmonic_patterns(&self, time_series: &[f64], evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Find fundamental periods first
        let fundamental_periods = self.find_fundamental_periods(time_series);

        for &fundamental in &fundamental_periods {
            // Check for harmonic patterns (2x, 3x, 4x the fundamental)
            for harmonic in 2..=4 {
                let harmonic_period = fundamental * harmonic;
                if harmonic_period < time_series.len() / 2 {
                    let correlation = self.calculate_harmonic_correlation(time_series, fundamental, harmonic);

                    if correlation > 0.4 { // Significant harmonic relationship
                        let strength = correlation;
                        let confidence = correlation * 0.7; // Lower confidence for harmonics

                        let mut metadata = HashMap::new();
                        metadata.insert("fundamental_period".to_string(), fundamental.to_string());
                        metadata.insert("harmonic_multiple".to_string(), harmonic.to_string());
                        metadata.insert("harmonic_correlation".to_string(), correlation.to_string());

                        let pattern = TemporalPattern {
                            id: format!("harmonic_{}x_{}", harmonic, fundamental),
                            pattern_type: PatternType::Cyclical { period_hours: harmonic_period as u64 },
                            strength,
                            confidence,
                            time_range: time_range.clone(),
                            description: format!("Harmonic pattern ({}x fundamental {}-hour period)", harmonic, fundamental),
                            evidence: evidence.to_vec(),
                            metadata,
                        };

                        patterns.push(pattern);
                    }
                }
            }
        }

        Ok(patterns)
    }

    /// Find fundamental periods in time series
    fn find_fundamental_periods(&self, time_series: &[f64]) -> Vec<usize> {
        let mut periods = Vec::new();

        // Check common periods (24h, 12h, 8h, 6h, 4h, 3h)
        let candidate_periods = [24, 12, 8, 6, 4, 3];

        for &period in &candidate_periods {
            if period < time_series.len() / 3 {
                let correlation = self.calculate_autocorrelation(time_series, period);
                if correlation > 0.5 {
                    periods.push(period);
                }
            }
        }

        periods
    }

    /// Calculate correlation between fundamental and harmonic patterns
    fn calculate_harmonic_correlation(&self, time_series: &[f64], fundamental: usize, harmonic: usize) -> f64 {
        let harmonic_period = fundamental * harmonic;
        if harmonic_period >= time_series.len() {
            return 0.0;
        }

        // Create harmonic signal
        let mut harmonic_signal = vec![0.0; time_series.len()];
        for i in 0..time_series.len() {
            let phase = (i as f64 * 2.0 * std::f64::consts::PI) / harmonic_period as f64;
            harmonic_signal[i] = phase.sin();
        }

        // Calculate correlation with actual time series
        self.calculate_signal_correlation(time_series, &harmonic_signal)
    }

    /// Calculate correlation between two signals
    fn calculate_signal_correlation(&self, signal1: &[f64], signal2: &[f64]) -> f64 {
        if signal1.len() != signal2.len() || signal1.is_empty() {
            return 0.0;
        }

        let n = signal1.len() as f64;
        let mean1 = signal1.iter().sum::<f64>() / n;
        let mean2 = signal2.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..signal1.len() {
            let diff1 = signal1[i] - mean1;
            let diff2 = signal2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Detect anomaly patterns using statistical methods
    async fn detect_anomaly_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Statistical outlier detection
        let outlier_patterns = self.detect_statistical_outliers(&evidence, time_range).await?;
        patterns.extend(outlier_patterns);

        // Isolation forest-like anomaly detection
        let isolation_patterns = self.detect_isolation_anomalies(&evidence, time_range).await?;
        patterns.extend(isolation_patterns);

        // Time series anomaly detection
        let time_series_patterns = self.detect_time_series_anomalies(&evidence, time_range).await?;
        patterns.extend(time_series_patterns);

        Ok(patterns)
    }

    /// Detect statistical outliers
    async fn detect_statistical_outliers(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Calculate statistics for strength values
        let strengths: Vec<f64> = evidence.iter().map(|e| e.strength).collect();
        if strengths.is_empty() {
            return Ok(patterns);
        }

        let mean = strengths.iter().sum::<f64>() / strengths.len() as f64;
        let variance = strengths.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / strengths.len() as f64;
        let std_dev = variance.sqrt();

        // Find outliers (more than 2 standard deviations from mean)
        let threshold = 2.0;
        let mut outliers = Vec::new();

        for ev in evidence {
            let z_score = (ev.strength - mean) / std_dev;
            if z_score.abs() > threshold {
                outliers.push((ev.clone(), z_score));
            }
        }

        if !outliers.is_empty() {
            let strength = outliers.iter().map(|(_, z)| z.abs()).sum::<f64>() / outliers.len() as f64 / 3.0; // Normalize
            let confidence = (outliers.len() as f64 / evidence.len() as f64).min(1.0);

            let mut metadata = HashMap::new();
            metadata.insert("outlier_count".to_string(), outliers.len().to_string());
            metadata.insert("mean_z_score".to_string(), (strength * 3.0).to_string());

            let pattern = TemporalPattern {
                id: "anomaly_statistical_outliers".to_string(),
                pattern_type: PatternType::Irregular,
                strength: strength.min(1.0),
                confidence,
                time_range: time_range.clone(),
                description: format!("Statistical outliers detected ({} anomalies)", outliers.len()),
                evidence: outliers.into_iter().map(|(ev, _)| ev).collect(),
                metadata,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Detect anomalies using isolation forest-like algorithm
    async fn detect_isolation_anomalies(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        if evidence.len() < 10 {
            return Ok(patterns);
        }

        // Create feature vectors for each evidence point
        let features: Vec<Vec<f64>> = evidence.iter().map(|ev| {
            vec![
                ev.timestamp.hour() as f64,
                ev.timestamp.weekday().num_days_from_monday() as f64,
                ev.strength,
                (ev.timestamp - time_range.start).num_hours() as f64,
            ]
        }).collect();

        // Calculate isolation scores
        let mut isolation_scores = Vec::new();
        for (i, feature) in features.iter().enumerate() {
            let score = self.calculate_isolation_score(feature, &features);
            isolation_scores.push((evidence[i].clone(), score));
        }

        // Sort by isolation score (higher = more anomalous)
        isolation_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top anomalies
        let anomaly_threshold = 0.7;
        let anomalies: Vec<_> = isolation_scores.into_iter()
            .filter(|(_, score)| *score > anomaly_threshold)
            .take(evidence.len() / 10) // Max 10% as anomalies
            .collect();

        if !anomalies.is_empty() {
            let avg_score = anomalies.iter().map(|(_, score)| score).sum::<f64>() / anomalies.len() as f64;
            let strength = avg_score;
            let confidence = (anomalies.len() as f64 / evidence.len() as f64 * 10.0).min(1.0);

            let mut metadata = HashMap::new();
            metadata.insert("isolation_anomalies".to_string(), anomalies.len().to_string());
            metadata.insert("avg_isolation_score".to_string(), avg_score.to_string());

            let pattern = TemporalPattern {
                id: "anomaly_isolation_forest".to_string(),
                pattern_type: PatternType::Irregular,
                strength,
                confidence,
                time_range: time_range.clone(),
                description: format!("Isolation-based anomalies detected ({} anomalies)", anomalies.len()),
                evidence: anomalies.into_iter().map(|(ev, _)| ev).collect(),
                metadata,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Calculate isolation score for a feature vector
    fn calculate_isolation_score(&self, target: &[f64], all_features: &[Vec<f64>]) -> f64 {
        let mut total_path_length = 0.0;
        let num_trees = 10;

        for _ in 0..num_trees {
            total_path_length += self.isolation_tree_path_length(target, all_features, 0, 8);
        }

        let avg_path_length = total_path_length / num_trees as f64;
        let expected_path_length = self.expected_isolation_path_length(all_features.len());

        // Normalize to [0, 1] where higher values indicate more anomalous
        if expected_path_length > 0.0 {
            2.0_f64.powf(-avg_path_length / expected_path_length)
        } else {
            0.0
        }
    }

    /// Calculate path length in isolation tree
    fn isolation_tree_path_length(&self, target: &[f64], data: &[Vec<f64>], depth: usize, max_depth: usize) -> f64 {
        if depth >= max_depth || data.len() <= 1 {
            return depth as f64 + self.expected_isolation_path_length(data.len());
        }

        // Random feature selection
        let feature_idx = depth % target.len();
        if feature_idx >= target.len() {
            return depth as f64;
        }

        // Random split point
        let feature_values: Vec<f64> = data.iter().map(|row| row[feature_idx]).collect();
        let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = feature_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_val >= max_val {
            return depth as f64;
        }

        let split_point = min_val + (max_val - min_val) * 0.5; // Simple midpoint split

        if target[feature_idx] < split_point {
            depth as f64 + 1.0
        } else {
            depth as f64 + 1.0
        }
    }

    /// Expected path length for isolation tree
    fn expected_isolation_path_length(&self, n: usize) -> f64 {
        if n <= 1 {
            0.0
        } else {
            2.0 * ((n - 1) as f64).ln() - 2.0 * (n - 1) as f64 / n as f64
        }
    }

    /// Detect time series anomalies
    async fn detect_time_series_anomalies(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Convert to time series
        let time_series = self.evidence_to_time_series(evidence, time_range);
        if time_series.len() < 24 { // Need at least 24 hours
            return Ok(patterns);
        }

        // Detect sudden changes using change point detection
        let change_points = self.detect_change_points(&time_series);

        if !change_points.is_empty() {
            let strength = change_points.len() as f64 / time_series.len() as f64 * 10.0; // Scale up
            let confidence = (change_points.len() as f64 / 5.0).min(1.0); // Max confidence at 5 change points

            let mut metadata = HashMap::new();
            metadata.insert("change_points".to_string(), change_points.len().to_string());
            metadata.insert("change_point_positions".to_string(),
                change_points.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(","));

            let pattern = TemporalPattern {
                id: "anomaly_change_points".to_string(),
                pattern_type: PatternType::Irregular,
                strength: strength.min(1.0),
                confidence,
                time_range: time_range.clone(),
                description: format!("Change point anomalies detected ({} change points)", change_points.len()),
                evidence: evidence.to_vec(),
                metadata,
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Detect change points in time series
    fn detect_change_points(&self, time_series: &[f64]) -> Vec<usize> {
        let mut change_points = Vec::new();
        let window_size = 12.min(time_series.len() / 4); // 12-hour window or 1/4 of series

        if window_size < 3 {
            return change_points;
        }

        for i in window_size..time_series.len() - window_size {
            let before_mean = time_series[i - window_size..i].iter().sum::<f64>() / window_size as f64;
            let after_mean = time_series[i..i + window_size].iter().sum::<f64>() / window_size as f64;

            let before_var = time_series[i - window_size..i].iter()
                .map(|x| (x - before_mean).powi(2))
                .sum::<f64>() / window_size as f64;
            let after_var = time_series[i..i + window_size].iter()
                .map(|x| (x - after_mean).powi(2))
                .sum::<f64>() / window_size as f64;

            // Detect significant change in mean or variance
            let mean_change = (before_mean - after_mean).abs() / (before_mean + after_mean + 1e-10);
            let var_change = (before_var - after_var).abs() / (before_var + after_var + 1e-10);

            if mean_change > 0.5 || var_change > 0.5 {
                change_points.push(i);
            }
        }

        change_points
    }

    /// Detect patterns using machine learning-inspired techniques
    async fn detect_ml_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points * 2 {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Clustering-based pattern detection
        let cluster_patterns = self.detect_clustering_patterns(&evidence, time_range).await?;
        patterns.extend(cluster_patterns);

        // Decision tree-like pattern detection
        let decision_patterns = self.detect_decision_tree_patterns(&evidence, time_range).await?;
        patterns.extend(decision_patterns);

        // Neural network-inspired pattern detection
        let neural_patterns = self.detect_neural_network_patterns(&evidence, time_range).await?;
        patterns.extend(neural_patterns);

        Ok(patterns)
    }

    /// Detect patterns using clustering techniques
    async fn detect_clustering_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Create feature vectors for clustering
        let features: Vec<Vec<f64>> = evidence.iter().map(|ev| {
            vec![
                ev.timestamp.hour() as f64 / 24.0, // Normalize to [0,1]
                ev.timestamp.weekday().num_days_from_monday() as f64 / 7.0,
                ev.strength,
                (ev.timestamp - time_range.start).num_hours() as f64 / (time_range.end - time_range.start).num_hours() as f64,
            ]
        }).collect();

        // Simple k-means clustering (k=3)
        let clusters = self.simple_kmeans_clustering(&features, 3);

        // Analyze clusters for patterns
        for (cluster_id, cluster_indices) in clusters.iter().enumerate() {
            if cluster_indices.len() >= self.config.min_data_points {
                let cluster_evidence: Vec<PatternEvidence> = cluster_indices.iter()
                    .map(|&i| evidence[i].clone())
                    .collect();

                let cluster_strength = self.calculate_cluster_cohesion(&features, cluster_indices);
                let cluster_confidence = cluster_indices.len() as f64 / evidence.len() as f64;

                let mut metadata = HashMap::new();
                metadata.insert("cluster_id".to_string(), cluster_id.to_string());
                metadata.insert("cluster_size".to_string(), cluster_indices.len().to_string());
                metadata.insert("cohesion".to_string(), cluster_strength.to_string());

                let pattern = TemporalPattern {
                    id: format!("ml_cluster_{}", cluster_id),
                    pattern_type: PatternType::Custom(format!("ML_Cluster_{}", cluster_id)),
                    strength: cluster_strength,
                    confidence: cluster_confidence,
                    time_range: time_range.clone(),
                    description: format!("ML-detected cluster pattern (cluster {})", cluster_id),
                    evidence: cluster_evidence,
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Simple k-means clustering implementation
    fn simple_kmeans_clustering(&self, features: &[Vec<f64>], k: usize) -> Vec<Vec<usize>> {
        if features.is_empty() || k == 0 {
            return Vec::new();
        }

        let feature_dim = features[0].len();
        let mut centroids = vec![vec![0.0; feature_dim]; k];

        // Initialize centroids randomly
        for i in 0..k {
            if i < features.len() {
                centroids[i] = features[i].clone();
            }
        }

        let mut assignments = vec![0; features.len()];
        let max_iterations = 10;

        for _ in 0..max_iterations {
            // Assign points to nearest centroid
            for (i, feature) in features.iter().enumerate() {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = self.euclidean_distance(feature, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Update centroids
            for j in 0..k {
                let cluster_points: Vec<&Vec<f64>> = features.iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, feature)| feature)
                    .collect();

                if !cluster_points.is_empty() {
                    for dim in 0..feature_dim {
                        centroids[j][dim] = cluster_points.iter()
                            .map(|point| point[dim])
                            .sum::<f64>() / cluster_points.len() as f64;
                    }
                }
            }
        }

        // Group indices by cluster
        let mut clusters = vec![Vec::new(); k];
        for (i, &cluster_id) in assignments.iter().enumerate() {
            clusters[cluster_id].push(i);
        }

        clusters
    }

    /// Calculate Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }

        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate cluster cohesion (how tightly grouped the cluster is)
    fn calculate_cluster_cohesion(&self, features: &[Vec<f64>], cluster_indices: &[usize]) -> f64 {
        if cluster_indices.len() < 2 {
            return 1.0;
        }

        let cluster_features: Vec<&Vec<f64>> = cluster_indices.iter()
            .map(|&i| &features[i])
            .collect();

        // Calculate centroid
        let feature_dim = cluster_features[0].len();
        let mut centroid = vec![0.0; feature_dim];
        for feature in &cluster_features {
            for (i, &value) in feature.iter().enumerate() {
                centroid[i] += value;
            }
        }
        for value in &mut centroid {
            *value /= cluster_features.len() as f64;
        }

        // Calculate average distance to centroid
        let avg_distance = cluster_features.iter()
            .map(|feature| self.euclidean_distance(feature, &centroid))
            .sum::<f64>() / cluster_features.len() as f64;

        // Convert to cohesion score (lower distance = higher cohesion)
        (1.0 / (1.0 + avg_distance)).min(1.0)
    }

    /// Detect patterns using decision tree-like analysis
    async fn detect_decision_tree_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Create decision rules based on features
        let rules = self.generate_decision_rules(evidence);

        for rule in rules {
            let matching_evidence: Vec<PatternEvidence> = evidence.iter()
                .filter(|ev| self.evidence_matches_rule(ev, &rule))
                .cloned()
                .collect();

            if matching_evidence.len() >= self.config.min_data_points {
                let strength = matching_evidence.len() as f64 / evidence.len() as f64;
                let confidence = rule.confidence;

                let mut metadata = HashMap::new();
                metadata.insert("rule_description".to_string(), rule.description.clone());
                metadata.insert("rule_confidence".to_string(), rule.confidence.to_string());

                let pattern = TemporalPattern {
                    id: format!("ml_decision_rule_{}", rule.id),
                    pattern_type: PatternType::Custom("ML_DecisionRule".to_string()),
                    strength,
                    confidence,
                    time_range: time_range.clone(),
                    description: format!("Decision rule pattern: {}", rule.description),
                    evidence: matching_evidence,
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Generate decision rules from evidence
    fn generate_decision_rules(&self, _evidence: &[PatternEvidence]) -> Vec<DecisionRule> {
        let mut rules = Vec::new();

        // Rule 1: High activity during business hours
        rules.push(DecisionRule {
            id: "business_hours".to_string(),
            description: "High activity during business hours (9-17)".to_string(),
            conditions: vec![
                RuleCondition::HourRange(9, 17),
                RuleCondition::StrengthThreshold(0.5),
            ],
            confidence: 0.8,
        });

        // Rule 2: Weekend patterns
        rules.push(DecisionRule {
            id: "weekend_activity".to_string(),
            description: "Weekend activity pattern".to_string(),
            conditions: vec![
                RuleCondition::WeekendOnly,
                RuleCondition::StrengthThreshold(0.3),
            ],
            confidence: 0.7,
        });

        // Rule 3: Late night activity
        rules.push(DecisionRule {
            id: "late_night".to_string(),
            description: "Late night activity (22-06)".to_string(),
            conditions: vec![
                RuleCondition::HourRange(22, 6),
                RuleCondition::StrengthThreshold(0.4),
            ],
            confidence: 0.6,
        });

        rules
    }

    /// Check if evidence matches a decision rule
    fn evidence_matches_rule(&self, evidence: &PatternEvidence, rule: &DecisionRule) -> bool {
        for condition in &rule.conditions {
            match condition {
                RuleCondition::HourRange(start, end) => {
                    let hour = evidence.timestamp.hour();
                    if start <= end {
                        if hour < *start || hour > *end {
                            return false;
                        }
                    } else {
                        // Wraps around midnight
                        if hour < *start && hour > *end {
                            return false;
                        }
                    }
                }
                RuleCondition::WeekendOnly => {
                    let weekday = evidence.timestamp.weekday();
                    if weekday != chrono::Weekday::Sat && weekday != chrono::Weekday::Sun {
                        return false;
                    }
                }
                RuleCondition::StrengthThreshold(threshold) => {
                    if evidence.strength < *threshold {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Detect patterns using neural network-inspired techniques
    async fn detect_neural_network_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Create feature vectors
        let features: Vec<Vec<f64>> = evidence.iter().map(|ev| {
            vec![
                (ev.timestamp.hour() as f64 / 24.0) * 2.0 - 1.0, // Normalize to [-1,1]
                (ev.timestamp.weekday().num_days_from_monday() as f64 / 7.0) * 2.0 - 1.0,
                ev.strength * 2.0 - 1.0,
                ((ev.timestamp - time_range.start).num_hours() as f64 / (time_range.end - time_range.start).num_hours() as f64) * 2.0 - 1.0,
            ]
        }).collect();

        // Simple perceptron-like pattern detection
        let neural_patterns = self.detect_perceptron_patterns(&features, evidence, time_range).await?;
        patterns.extend(neural_patterns);

        Ok(patterns)
    }

    /// Detect patterns using perceptron-like analysis
    async fn detect_perceptron_patterns(&self, features: &[Vec<f64>], evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        if features.is_empty() {
            return Ok(patterns);
        }

        let feature_dim = features[0].len();

        // Try different weight combinations to find patterns
        let weight_combinations = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Hour-based
            vec![0.0, 1.0, 0.0, 0.0], // Day-based
            vec![0.0, 0.0, 1.0, 0.0], // Strength-based
            vec![0.5, 0.5, 0.0, 0.0], // Time-based combination
            vec![0.0, 0.0, 0.5, 0.5], // Temporal progression
        ];

        for (i, weights) in weight_combinations.iter().enumerate() {
            let activations: Vec<f64> = features.iter()
                .map(|feature| self.calculate_activation(feature, weights))
                .collect();

            // Find threshold that separates high and low activations
            let mut sorted_activations = activations.clone();
            sorted_activations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if sorted_activations.len() > 4 {
                let threshold = sorted_activations[sorted_activations.len() * 3 / 4]; // 75th percentile

                let high_activation_indices: Vec<usize> = activations.iter()
                    .enumerate()
                    .filter(|(_, &activation)| activation > threshold)
                    .map(|(idx, _)| idx)
                    .collect();

                if high_activation_indices.len() >= self.config.min_data_points {
                    let pattern_evidence: Vec<PatternEvidence> = high_activation_indices.iter()
                        .map(|&idx| evidence[idx].clone())
                        .collect();

                    let strength = high_activation_indices.len() as f64 / evidence.len() as f64;
                    let confidence = self.calculate_pattern_confidence(&activations, threshold);

                    let mut metadata = HashMap::new();
                    metadata.insert("perceptron_id".to_string(), i.to_string());
                    metadata.insert("threshold".to_string(), threshold.to_string());
                    metadata.insert("weights".to_string(),
                        weights.iter().map(|w| format!("{:.2}", w)).collect::<Vec<_>>().join(","));

                    let pattern = TemporalPattern {
                        id: format!("ml_perceptron_{}", i),
                        pattern_type: PatternType::Custom("ML_Perceptron".to_string()),
                        strength,
                        confidence,
                        time_range: time_range.clone(),
                        description: format!("Neural network-inspired pattern (perceptron {})", i),
                        evidence: pattern_evidence,
                        metadata,
                    };

                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Calculate activation using weighted sum
    fn calculate_activation(&self, features: &[f64], weights: &[f64]) -> f64 {
        let weighted_sum: f64 = features.iter()
            .zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        // Apply sigmoid activation
        1.0 / (1.0 + (-weighted_sum).exp())
    }

    /// Calculate pattern confidence based on activation distribution
    fn calculate_pattern_confidence(&self, activations: &[f64], threshold: f64) -> f64 {
        let above_threshold = activations.iter().filter(|&&a| a > threshold).count() as f64;
        let below_threshold = activations.iter().filter(|&&a| a <= threshold).count() as f64;

        if above_threshold + below_threshold == 0.0 {
            return 0.0;
        }

        // Calculate separation quality
        let separation = (above_threshold - below_threshold).abs() / (above_threshold + below_threshold);
        separation.min(1.0)
    }

    /// Detect complex multi-dimensional patterns
    async fn detect_complex_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points * 3 {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Multi-scale temporal patterns
        let multiscale_patterns = self.detect_multiscale_patterns(&evidence, time_range).await?;
        patterns.extend(multiscale_patterns);

        // Hierarchical patterns
        let hierarchical_patterns = self.detect_hierarchical_patterns(&evidence, time_range).await?;
        patterns.extend(hierarchical_patterns);

        // Composite patterns (combinations of basic patterns)
        let composite_patterns = self.detect_composite_patterns(&evidence, time_range).await?;
        patterns.extend(composite_patterns);

        Ok(patterns)
    }

    /// Detect multi-scale temporal patterns
    async fn detect_multiscale_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Analyze at different time scales
        let scales = vec![
            ("hourly", 1),
            ("daily", 24),
            ("weekly", 168),
            ("monthly", 720),
        ];

        for (scale_name, scale_hours) in scales {
            let scale_patterns = self.analyze_at_time_scale(evidence, time_range, scale_hours, scale_name).await?;
            patterns.extend(scale_patterns);
        }

        Ok(patterns)
    }

    /// Analyze patterns at a specific time scale
    async fn analyze_at_time_scale(&self, evidence: &[PatternEvidence], time_range: &TimeRange, scale_hours: usize, scale_name: &str) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Aggregate evidence at the given scale
        let duration_hours = (time_range.end - time_range.start).num_hours() as usize;
        let num_buckets = (duration_hours / scale_hours).max(1);
        let mut buckets = vec![0.0; num_buckets];

        for ev in evidence {
            let hours_from_start = (ev.timestamp - time_range.start).num_hours() as usize;
            let bucket_idx = (hours_from_start / scale_hours).min(num_buckets - 1);
            buckets[bucket_idx] += ev.strength;
        }

        // Find patterns in the aggregated data
        if buckets.len() >= 3 {
            // Look for periodic patterns at this scale
            let period_strength = self.detect_periodicity_in_buckets(&buckets);
            if period_strength > 0.3 {
                let mut metadata = HashMap::new();
                metadata.insert("scale".to_string(), scale_name.to_string());
                metadata.insert("scale_hours".to_string(), scale_hours.to_string());
                metadata.insert("period_strength".to_string(), period_strength.to_string());

                let pattern = TemporalPattern {
                    id: format!("multiscale_{}_{}", scale_name, scale_hours),
                    pattern_type: PatternType::Custom(format!("Multiscale_{}", scale_name)),
                    strength: period_strength,
                    confidence: period_strength * 0.8,
                    time_range: time_range.clone(),
                    description: format!("Multi-scale pattern at {} level", scale_name),
                    evidence: evidence.to_vec(),
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Detect periodicity in aggregated buckets
    fn detect_periodicity_in_buckets(&self, buckets: &[f64]) -> f64 {
        if buckets.len() < 4 {
            return 0.0;
        }

        let mut max_periodicity: f64 = 0.0;

        // Check for periods from 2 to half the length
        for period in 2..=buckets.len() / 2 {
            let mut correlation_sum = 0.0;
            let mut count = 0;

            for i in 0..buckets.len() - period {
                correlation_sum += buckets[i] * buckets[i + period];
                count += 1;
            }

            if count > 0 {
                let avg_correlation = correlation_sum / count as f64;
                let normalized_correlation = avg_correlation / (buckets.iter().map(|x| x * x).sum::<f64>() / buckets.len() as f64 + 1e-10);
                max_periodicity = max_periodicity.max(normalized_correlation);
            }
        }

        max_periodicity.min(1.0)
    }

    /// Detect hierarchical patterns
    async fn detect_hierarchical_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Create hierarchy: hour -> day -> week -> month
        let hierarchical_analysis = self.build_temporal_hierarchy(evidence, time_range);

        // Analyze patterns at each level of the hierarchy
        for (level_name, level_data) in hierarchical_analysis {
            if level_data.len() >= 3 {
                let pattern_strength = self.analyze_hierarchical_level(&level_data);
                if pattern_strength > 0.4 {
                    let mut metadata = HashMap::new();
                    metadata.insert("hierarchy_level".to_string(), level_name.clone());
                    metadata.insert("level_size".to_string(), level_data.len().to_string());

                    let pattern = TemporalPattern {
                        id: format!("hierarchical_{}", level_name),
                        pattern_type: PatternType::Custom(format!("Hierarchical_{}", level_name)),
                        strength: pattern_strength,
                        confidence: pattern_strength * 0.9,
                        time_range: time_range.clone(),
                        description: format!("Hierarchical pattern at {} level", level_name),
                        evidence: evidence.to_vec(),
                        metadata,
                    };

                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Build temporal hierarchy from evidence
    fn build_temporal_hierarchy(&self, evidence: &[PatternEvidence], _time_range: &TimeRange) -> Vec<(String, Vec<f64>)> {
        let mut hierarchy = Vec::new();

        // Hour level
        let mut hourly = vec![0.0; 24];
        for ev in evidence {
            hourly[ev.timestamp.hour() as usize] += ev.strength;
        }
        hierarchy.push(("hour".to_string(), hourly));

        // Day level
        let mut daily = vec![0.0; 7];
        for ev in evidence {
            daily[ev.timestamp.weekday().num_days_from_monday() as usize] += ev.strength;
        }
        hierarchy.push(("day".to_string(), daily));

        // Month level
        let mut monthly = vec![0.0; 12];
        for ev in evidence {
            monthly[(ev.timestamp.month() - 1) as usize] += ev.strength;
        }
        hierarchy.push(("month".to_string(), monthly));

        hierarchy
    }

    /// Analyze patterns at a hierarchical level
    fn analyze_hierarchical_level(&self, level_data: &[f64]) -> f64 {
        if level_data.len() < 3 {
            return 0.0;
        }

        // Calculate variance and regularity
        let mean = level_data.iter().sum::<f64>() / level_data.len() as f64;
        let variance = level_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / level_data.len() as f64;

        // Higher variance indicates more pattern structure
        let variance_score = (variance / (mean + 1e-10)).min(1.0);

        // Calculate autocorrelation at lag 1
        let autocorr = if level_data.len() > 1 {
            self.calculate_autocorrelation(level_data, 1).abs()
        } else {
            0.0
        };

        // Combine variance and autocorrelation
        (variance_score * 0.6 + autocorr * 0.4).min(1.0)
    }

    /// Detect composite patterns (combinations of basic patterns)
    async fn detect_composite_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Detect combinations of daily and weekly patterns
        let daily_weekly = self.detect_daily_weekly_composite(evidence, time_range).await?;
        patterns.extend(daily_weekly);

        // Detect burst + trend combinations
        let burst_trend = self.detect_burst_trend_composite(evidence, time_range).await?;
        patterns.extend(burst_trend);

        Ok(patterns)
    }

    /// Detect daily-weekly composite patterns
    async fn detect_daily_weekly_composite(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Create 2D histogram: hour x day_of_week
        let mut histogram = vec![vec![0.0; 7]; 24];
        for ev in evidence {
            let hour = ev.timestamp.hour() as usize;
            let day = ev.timestamp.weekday().num_days_from_monday() as usize;
            histogram[hour][day] += ev.strength;
        }

        // Find peak combinations
        let mut peak_combinations = Vec::new();
        for hour in 0..24 {
            for day in 0..7 {
                if histogram[hour][day] > 0.0 {
                    peak_combinations.push((hour, day, histogram[hour][day]));
                }
            }
        }

        // Sort by strength and take top combinations
        peak_combinations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        if peak_combinations.len() >= 3 {
            let top_combinations = &peak_combinations[..3.min(peak_combinations.len())];
            let total_strength: f64 = top_combinations.iter().map(|(_, _, s)| s).sum();
            let avg_strength = total_strength / evidence.iter().map(|e| e.strength).sum::<f64>();

            if avg_strength > 0.3 {
                let mut metadata = HashMap::new();
                metadata.insert("top_combinations".to_string(),
                    top_combinations.iter()
                        .map(|(h, d, s)| format!("{}h-{}d:{:.2}", h, d, s))
                        .collect::<Vec<_>>()
                        .join(","));

                let pattern = TemporalPattern {
                    id: "composite_daily_weekly".to_string(),
                    pattern_type: PatternType::Custom("Composite_DailyWeekly".to_string()),
                    strength: avg_strength.min(1.0),
                    confidence: avg_strength * 0.8,
                    time_range: time_range.clone(),
                    description: "Composite daily-weekly pattern".to_string(),
                    evidence: evidence.to_vec(),
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Detect burst-trend composite patterns
    async fn detect_burst_trend_composite(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // First detect bursts and trends separately
        let bursts = self.detect_burst_patterns(time_range).await?;
        let trends = self.detect_trend_patterns(time_range).await?;

        // Look for overlapping or sequential burst-trend combinations
        for burst in &bursts {
            for trend in &trends {
                let overlap = self.calculate_pattern_overlap(&burst.time_range, &trend.time_range);
                let sequence_score = self.calculate_pattern_sequence_score(&burst.time_range, &trend.time_range);

                if overlap > 0.3 || sequence_score > 0.5 {
                    let composite_strength = (burst.strength + trend.strength) / 2.0;
                    let composite_confidence = (burst.confidence + trend.confidence) / 2.0 * 0.9; // Slightly lower for composite

                    let mut metadata = HashMap::new();
                    metadata.insert("burst_id".to_string(), burst.id.clone());
                    metadata.insert("trend_id".to_string(), trend.id.clone());
                    metadata.insert("overlap".to_string(), overlap.to_string());
                    metadata.insert("sequence_score".to_string(), sequence_score.to_string());

                    let pattern = TemporalPattern {
                        id: format!("composite_burst_trend_{}_{}", burst.id, trend.id),
                        pattern_type: PatternType::Custom("Composite_BurstTrend".to_string()),
                        strength: composite_strength,
                        confidence: composite_confidence,
                        time_range: time_range.clone(),
                        description: "Composite burst-trend pattern".to_string(),
                        evidence: evidence.to_vec(),
                        metadata,
                    };

                    patterns.push(pattern);
                }
            }
        }

        Ok(patterns)
    }

    /// Calculate overlap between two time ranges
    fn calculate_pattern_overlap(&self, range1: &TimeRange, range2: &TimeRange) -> f64 {
        let overlap_start = range1.start.max(range2.start);
        let overlap_end = range1.end.min(range2.end);

        if overlap_start >= overlap_end {
            return 0.0;
        }

        let overlap_duration = (overlap_end - overlap_start).num_hours() as f64;
        let range1_duration = (range1.end - range1.start).num_hours() as f64;
        let range2_duration = (range2.end - range2.start).num_hours() as f64;
        let min_duration = range1_duration.min(range2_duration);

        if min_duration > 0.0 {
            overlap_duration / min_duration
        } else {
            0.0
        }
    }

    /// Calculate sequence score between two time ranges
    fn calculate_pattern_sequence_score(&self, range1: &TimeRange, range2: &TimeRange) -> f64 {
        let gap = if range1.end <= range2.start {
            (range2.start - range1.end).num_hours() as f64
        } else if range2.end <= range1.start {
            (range1.start - range2.end).num_hours() as f64
        } else {
            return 0.0; // Overlapping, not sequential
        };

        // Score based on gap size (smaller gaps = higher scores)
        let max_gap = 168.0; // 1 week
        if gap <= max_gap {
            1.0 - (gap / max_gap)
        } else {
            0.0
        }
    }

    /// Detect correlation-based patterns
    async fn detect_correlation_patterns(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let evidence = self.gather_evidence_in_range(time_range);
        if evidence.len() < self.config.min_data_points * 2 {
            return Ok(Vec::new());
        }

        let mut patterns = Vec::new();

        // Cross-correlation patterns
        let cross_corr_patterns = self.detect_cross_correlation_patterns(&evidence, time_range).await?;
        patterns.extend(cross_corr_patterns);

        // Lag correlation patterns
        let lag_corr_patterns = self.detect_lag_correlation_patterns(&evidence, time_range).await?;
        patterns.extend(lag_corr_patterns);

        Ok(patterns)
    }

    /// Detect cross-correlation patterns
    async fn detect_cross_correlation_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Group evidence by memory key
        let mut memory_groups: std::collections::HashMap<String, Vec<&PatternEvidence>> = std::collections::HashMap::new();
        for ev in evidence {
            memory_groups.entry(ev.memory_key.clone()).or_default().push(ev);
        }

        // Find correlated memory access patterns
        let memory_keys: Vec<String> = memory_groups.keys().cloned().collect();
        for i in 0..memory_keys.len() {
            for j in i + 1..memory_keys.len() {
                let key1 = &memory_keys[i];
                let key2 = &memory_keys[j];

                if let (Some(group1), Some(group2)) = (memory_groups.get(key1), memory_groups.get(key2)) {
                    let correlation = self.calculate_memory_access_correlation(group1, group2, time_range);

                    if correlation > 0.6 {
                        let combined_evidence: Vec<PatternEvidence> = group1.iter()
                            .chain(group2.iter())
                            .map(|&ev| ev.clone())
                            .collect();

                        let mut metadata = HashMap::new();
                        metadata.insert("memory_key_1".to_string(), key1.clone());
                        metadata.insert("memory_key_2".to_string(), key2.clone());
                        metadata.insert("correlation".to_string(), correlation.to_string());

                        let pattern = TemporalPattern {
                            id: format!("correlation_{}_{}", key1, key2),
                            pattern_type: PatternType::Custom("Correlation".to_string()),
                            strength: correlation,
                            confidence: correlation * 0.9,
                            time_range: time_range.clone(),
                            description: format!("Correlated access pattern between {} and {}", key1, key2),
                            evidence: combined_evidence,
                            metadata,
                        };

                        patterns.push(pattern);
                    }
                }
            }
        }

        Ok(patterns)
    }

    /// Calculate correlation between memory access patterns
    fn calculate_memory_access_correlation(&self, group1: &[&PatternEvidence], group2: &[&PatternEvidence], time_range: &TimeRange) -> f64 {
        // Create time series for both groups
        let duration_hours = (time_range.end - time_range.start).num_hours() as usize;
        let mut series1 = vec![0.0; duration_hours];
        let mut series2 = vec![0.0; duration_hours];

        for ev in group1 {
            let hours_from_start = (ev.timestamp - time_range.start).num_hours() as usize;
            if hours_from_start < series1.len() {
                series1[hours_from_start] += ev.strength;
            }
        }

        for ev in group2 {
            let hours_from_start = (ev.timestamp - time_range.start).num_hours() as usize;
            if hours_from_start < series2.len() {
                series2[hours_from_start] += ev.strength;
            }
        }

        // Calculate Pearson correlation
        self.calculate_signal_correlation(&series1, &series2)
    }

    /// Detect lag correlation patterns
    async fn detect_lag_correlation_patterns(&self, evidence: &[PatternEvidence], time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut patterns = Vec::new();

        // Convert evidence to time series
        let time_series = self.evidence_to_time_series(evidence, time_range);

        // Check for lag correlations (patterns that repeat with a delay)
        let max_lag = time_series.len() / 4; // Check up to 1/4 of the series length

        for lag in 1..=max_lag {
            let correlation = self.calculate_autocorrelation(&time_series, lag);

            if correlation > 0.5 {
                let mut metadata = HashMap::new();
                metadata.insert("lag_hours".to_string(), lag.to_string());
                metadata.insert("lag_correlation".to_string(), correlation.to_string());

                let pattern = TemporalPattern {
                    id: format!("lag_correlation_{}", lag),
                    pattern_type: PatternType::Custom("LagCorrelation".to_string()),
                    strength: correlation,
                    confidence: correlation * 0.8,
                    time_range: time_range.clone(),
                    description: format!("Lag correlation pattern with {}-hour delay", lag),
                    evidence: evidence.to_vec(),
                    metadata,
                };

                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Update existing patterns with new evidence
    async fn update_existing_patterns(&mut self, evidence: &PatternEvidence) -> Result<()> {
        // Collect indices of patterns that need updating
        let mut patterns_to_update = Vec::new();
        for (i, pattern) in self.patterns.iter().enumerate() {
            if self.evidence_supports_pattern(evidence, pattern) {
                patterns_to_update.push(i);
            }
        }

        // Update patterns
        for i in patterns_to_update {
            if let Some(pattern) = self.patterns.get_mut(i) {
                pattern.evidence.push(evidence.clone());
                // Recalculate pattern strength and confidence
                self.recalculate_pattern_metrics_for_index(i).await?;
            }
        }
        Ok(())
    }

    /// Check if evidence supports an existing pattern
    fn evidence_supports_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        match pattern.pattern_type {
            PatternType::Daily => {
                // Check if the evidence fits the daily pattern
                self.fits_daily_pattern(evidence, pattern)
            }
            PatternType::Weekly => {
                // Check if the evidence fits the weekly pattern
                self.fits_weekly_pattern(evidence, pattern)
            }
            PatternType::Monthly => {
                self.fits_monthly_pattern(evidence, pattern)
            }
            PatternType::Seasonal => {
                self.fits_seasonal_pattern(evidence, pattern)
            }
            PatternType::Burst => {
                self.fits_burst_pattern(evidence, pattern)
            }
            PatternType::GradualIncrease | PatternType::GradualDecrease => {
                self.fits_gradual_pattern(evidence, pattern)
            }
            PatternType::Cyclical { period_hours } => {
                self.fits_cyclical_pattern(evidence, pattern, period_hours)
            }
            PatternType::Irregular => {
                self.fits_irregular_pattern(evidence, pattern)
            }
            PatternType::Custom(_) => {
                self.fits_custom_pattern(evidence, pattern)
            }
        }
    }

    /// Check if evidence fits a daily pattern
    fn fits_daily_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        if let Some(hour_str) = pattern.metadata.get("hour_of_day") {
            if let Ok(hour) = hour_str.parse::<u32>() {
                return evidence.timestamp.hour() == hour;
            }
        }
        false
    }

    /// Check if evidence fits a weekly pattern
    fn fits_weekly_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        if let Some(day_str) = pattern.metadata.get("day_of_week") {
            if let Ok(day) = day_str.parse::<u32>() {
                if let Some(wd) = Weekday::from_u64(day as u64) {
                    return evidence.timestamp.weekday() == wd;
                }
            }
        }
        false
    }

    /// Check if evidence fits a monthly pattern
    fn fits_monthly_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        if let Some(day_str) = pattern.metadata.get("day_of_month") {
            if let Ok(day) = day_str.parse::<u32>() {
                return evidence.timestamp.day() == day;
            }
        }
        false
    }

    /// Check if evidence fits a seasonal pattern
    fn fits_seasonal_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        if let (Some(start_str), Some(end_str)) = (pattern.metadata.get("start_month"), pattern.metadata.get("end_month")) {
            if let (Ok(start), Ok(end)) = (start_str.parse::<u32>(), end_str.parse::<u32>()) {
                let month = evidence.timestamp.month();
                if start <= end {
                    return month >= start && month <= end;
                } else {
                    // wraps around the year
                    return month >= start || month <= end;
                }
            }
        }
        false
    }

    /// Check if evidence fits a burst pattern
    fn fits_burst_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        pattern.time_range.contains(evidence.timestamp)
    }

    /// Check if evidence fits a gradual pattern (increase or decrease)
    fn fits_gradual_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        pattern.time_range.contains(evidence.timestamp)
    }

    /// Check if evidence fits a cyclical pattern
    fn fits_cyclical_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern, period_hours: u64) -> bool {
        if period_hours == 0 {
            return false;
        }
        let diff = evidence.timestamp - pattern.time_range.start;
        let hours = diff.num_hours().abs() as u64;
        hours % period_hours == 0
    }

    /// Check if evidence fits an irregular pattern
    fn fits_irregular_pattern(&self, _evidence: &PatternEvidence, _pattern: &TemporalPattern) -> bool {
        true
    }

    /// Check if evidence fits a custom pattern (simple metadata matching)
    fn fits_custom_pattern(&self, evidence: &PatternEvidence, pattern: &TemporalPattern) -> bool {
        if let Some(key) = pattern.metadata.get("memory_key") {
            if key != &evidence.memory_key {
                return false;
            }
        }
        true
    }

    /// Recalculate pattern metrics for a specific pattern index
    async fn recalculate_pattern_metrics_for_index(&mut self, index: usize) -> Result<()> {
        if index < self.patterns.len() {
            // Extract the pattern temporarily to avoid borrowing issues
            if let Some(mut pattern) = self.patterns.get(index).cloned() {
                self.recalculate_pattern_metrics(&mut pattern).await?;
                // Put the updated pattern back
                if let Some(existing_pattern) = self.patterns.get_mut(index) {
                    *existing_pattern = pattern;
                }
            }
        }
        Ok(())
    }

    /// Recalculate pattern metrics based on current evidence
    async fn recalculate_pattern_metrics(&self, pattern: &mut TemporalPattern) -> Result<()> {
        if pattern.evidence.is_empty() {
            pattern.strength = 0.0;
            pattern.confidence = 0.0;
            return Ok(());
        }

        // Calculate strength based on evidence consistency
        let avg_strength = pattern.evidence.iter()
            .map(|e| e.strength)
            .sum::<f64>() / pattern.evidence.len() as f64;

        pattern.strength = avg_strength;

        // Calculate confidence based on evidence count and consistency
        let evidence_count_factor = (pattern.evidence.len() as f64 / 10.0).min(1.0);
        let consistency_factor = self.calculate_evidence_consistency(&pattern.evidence);
        
        pattern.confidence = (evidence_count_factor + consistency_factor) / 2.0;

        Ok(())
    }

    /// Calculate consistency of evidence for a pattern
    fn calculate_evidence_consistency(&self, evidence: &[PatternEvidence]) -> f64 {
        if evidence.len() < 2 {
            return 0.0;
        }

        // Calculate variance in evidence strength
        let mean_strength = evidence.iter().map(|e| e.strength).sum::<f64>() / evidence.len() as f64;
        let variance = evidence.iter()
            .map(|e| (e.strength - mean_strength).powi(2))
            .sum::<f64>() / evidence.len() as f64;

        // Lower variance means higher consistency
        (1.0 - variance).max(0.0)
    }

    /// Collect all evidence within a time range
    fn gather_evidence_in_range(&self, range: &TimeRange) -> Vec<PatternEvidence> {
        self.patterns
            .iter()
            .flat_map(|p| p.evidence.iter().cloned())
            .filter(|e| range.contains(e.timestamp))
            .collect()
    }

    /// Analyze access patterns for a specific memory
    pub async fn analyze_access_pattern(&self, memory_key: &str) -> Result<AccessPattern> {
        let evidence: Vec<PatternEvidence> = self
            .patterns
            .iter()
            .flat_map(|p| p.evidence.iter())
            .filter(|e| e.memory_key == memory_key)
            .cloned()
            .collect();

        let mut hourly_distribution = [0u32; 24];
        let mut daily_distribution = [0u32; 7];
        let mut timestamps = Vec::new();

        for ev in &evidence {
            hourly_distribution[ev.timestamp.hour() as usize] += 1;
            daily_distribution[ev.timestamp.weekday().num_days_from_monday() as usize] += 1;
            timestamps.push(ev.timestamp);
        }

        let mut peak_times = Vec::new();
        if let (Some(start), Some(end)) = (timestamps.iter().min(), timestamps.iter().max()) {
            let range = TimeRange::new(*start, *end);
            if let Some(p) = self.detect_daily_pattern(&range).await? {
                if let Some(h) = p.metadata.get("hour_of_day").and_then(|s| s.parse::<u32>().ok()) {
                    let dt = start.date_naive().and_hms_opt(h, 0, 0).unwrap();
                    peak_times.push(dt.and_utc());
                }
            }
            for burst in self.detect_burst_patterns(&range).await? {
                peak_times.push(burst.time_range.start);
            }
        }

        // clustering info
        timestamps.sort();
        let mut clusters: Vec<Vec<DateTime<Utc>>> = Vec::new();
        for ts in timestamps {
            if let Some(cluster) = clusters.last_mut() {
                if ts - *cluster.last().unwrap() <= Duration::minutes(60) {
                    cluster.push(ts);
                    continue;
                }
            }
            clusters.push(vec![ts]);
        }
        let cluster_count = clusters.len();
        let avg_cluster_size = if cluster_count > 0 {
            clusters.iter().map(|c| c.len()).sum::<usize>() as f64 / cluster_count as f64
        } else { 0.0 };
        let inter_cluster_time = if clusters.len() > 1 {
            let mut times = Vec::new();
            for pair in clusters.windows(2) {
                times.push(pair[1][0] - *pair[0].last().unwrap());
            }
            times.iter().fold(Duration::zero(), |acc, d| acc + *d) / (times.len() as i32)
        } else {
            Duration::zero()
        };

        Ok(AccessPattern {
            memory_key: memory_key.to_string(),
            hourly_distribution,
            daily_distribution,
            peak_times,
            clustering_info: ClusteringInfo {
                cluster_count,
                avg_cluster_size,
                inter_cluster_time,
            },
        })
    }

    /// Get all detected patterns
    pub fn get_patterns(&self) -> &[TemporalPattern] {
        &self.patterns
    }

    /// Get patterns of a specific type
    pub fn get_patterns_by_type(&self, pattern_type: &PatternType) -> Vec<&TemporalPattern> {
        self.patterns.iter()
            .filter(|p| &p.pattern_type == pattern_type)
            .collect()
    }

    /// Clear old patterns that are no longer relevant
    pub async fn cleanup_old_patterns(&mut self, cutoff_date: DateTime<Utc>) -> Result<usize> {
        let original_count = self.patterns.len();
        
        self.patterns.retain(|pattern| {
            pattern.time_range.end >= cutoff_date
        });

        Ok(original_count - self.patterns.len())
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn dt(y: i32, m: u32, d: u32, h: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_opt(h, 0, 0).unwrap().and_utc()
    }

    fn evidence_at(ts: DateTime<Utc>) -> PatternEvidence {
        PatternEvidence {
            timestamp: ts,
            memory_key: "m1".into(),
            activity_type: ActivityType::Access,
            strength: 1.0,
        }
    }

    fn base_pattern(pt: PatternType, range: TimeRange, metadata: HashMap<String, String>) -> TemporalPattern {
        TemporalPattern {
            id: "p".into(),
            pattern_type: pt,
            strength: 1.0,
            confidence: 1.0,
            time_range: range,
            description: String::new(),
            evidence: Vec::new(),
            metadata,
        }
    }

    #[test]
    fn supports_daily() {
        let detector = PatternDetector::new();
        let mut meta = HashMap::new();
        meta.insert("hour_of_day".into(), "10".into());
        let pattern = base_pattern(PatternType::Daily, TimeRange::last_days(1), meta);
        let ev = evidence_at(dt(2024, 1, 1, 10));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_weekly() {
        let detector = PatternDetector::new();
        let mut meta = HashMap::new();
        // 2 -> Wednesday
        meta.insert("day_of_week".into(), "2".into());
        let pattern = base_pattern(PatternType::Weekly, TimeRange::last_days(7), meta);
        let ev = evidence_at(dt(2024, 1, 3, 12));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_monthly() {
        let detector = PatternDetector::new();
        let mut meta = HashMap::new();
        meta.insert("day_of_month".into(), "15".into());
        let pattern = base_pattern(PatternType::Monthly, TimeRange::last_days(30), meta);
        let ev = evidence_at(dt(2024, 1, 15, 8));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_seasonal() {
        let detector = PatternDetector::new();
        let mut meta = HashMap::new();
        meta.insert("start_month".into(), "3".into());
        meta.insert("end_month".into(), "5".into());
        let pattern = base_pattern(PatternType::Seasonal, TimeRange::last_days(365), meta);
        let ev = evidence_at(dt(2024, 4, 1, 0));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_burst() {
        let detector = PatternDetector::new();
        let start = dt(2024, 1, 1, 0);
        let range = TimeRange::new(start, start + Duration::hours(2));
        let pattern = base_pattern(PatternType::Burst, range.clone(), HashMap::new());
        let ev = evidence_at(start + Duration::hours(1));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_gradual() {
        let detector = PatternDetector::new();
        let start = dt(2024, 1, 1, 0);
        let range = TimeRange::new(start, start + Duration::days(10));
        let pattern = base_pattern(PatternType::GradualIncrease, range.clone(), HashMap::new());
        let ev = evidence_at(start + Duration::days(5));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_cyclical() {
        let detector = PatternDetector::new();
        let start = dt(2024, 1, 1, 0);
        let range = TimeRange::new(start, start + Duration::days(5));
        let pattern = base_pattern(PatternType::Cyclical { period_hours: 24 }, range.clone(), HashMap::new());
        let ev = evidence_at(start + Duration::hours(48));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_irregular() {
        let detector = PatternDetector::new();
        let start = dt(2024, 1, 1, 0);
        let pattern = base_pattern(PatternType::Irregular, TimeRange::new(start, start + Duration::days(1)), HashMap::new());
        let ev = evidence_at(start + Duration::hours(6));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }

    #[test]
    fn supports_custom() {
        let detector = PatternDetector::new();
        let start = dt(2024, 1, 1, 0);
        let mut meta = HashMap::new();
        meta.insert("memory_key".into(), "m1".into());
        let pattern = base_pattern(PatternType::Custom("x".into()), TimeRange::new(start, start + Duration::days(1)), meta);
        let ev = evidence_at(start + Duration::hours(1));
        assert!(detector.evidence_supports_pattern(&ev, &pattern));
    }
}
