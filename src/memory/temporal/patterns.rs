//! Temporal pattern detection and analysis

use crate::error::{MemoryError, Result};
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
    pub async fn update_patterns(&mut self, memory_key: &str, memory: &MemoryEntry) -> Result<()> {
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

    /// Detect patterns within a time range
    pub async fn detect_patterns_in_range(&self, time_range: &TimeRange) -> Result<Vec<TemporalPattern>> {
        let mut detected_patterns = Vec::new();

        // Detect daily patterns
        if let Some(daily_pattern) = self.detect_daily_pattern(time_range).await? {
            detected_patterns.push(daily_pattern);
        }

        // Detect weekly patterns
        if let Some(weekly_pattern) = self.detect_weekly_pattern(time_range).await? {
            detected_patterns.push(weekly_pattern);
        }

        // Detect burst patterns
        let burst_patterns = self.detect_burst_patterns(time_range).await?;
        detected_patterns.extend(burst_patterns);

        // Detect trend patterns
        let trend_patterns = self.detect_trend_patterns(time_range).await?;
        detected_patterns.extend(trend_patterns);

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
            let day = ev.timestamp.date_naive().and_hms_opt(0, 0, 0).unwrap().timestamp();
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
            time_range: TimeRange::new(DateTime::<Utc>::from_utc(chrono::NaiveDateTime::from_timestamp_opt(start, 0).unwrap(), Utc),
                                         DateTime::<Utc>::from_utc(chrono::NaiveDateTime::from_timestamp_opt(end, 0).unwrap(), Utc)),
            description: "Gradual trend detected".to_string(),
            evidence,
            metadata: HashMap::new(),
        };
        Ok(vec![pattern])
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
            _ => false, // TODO: Implement other pattern types
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
                    peak_times.push(DateTime::<Utc>::from_utc(dt, Utc));
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
