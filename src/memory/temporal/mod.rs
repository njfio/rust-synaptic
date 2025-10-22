//! Temporal tracking and versioning system for AI agent memory
//!
//! This module provides sophisticated temporal analysis capabilities including:
//! - Memory versioning and change tracking
//! - Differential analysis between memory states
//! - Temporal patterns and trends
//! - Time-based memory evolution

pub mod versioning;
pub mod differential;
pub mod patterns;
pub mod evolution;
pub mod decay_models;

// Re-export commonly used types
pub use versioning::{MemoryVersion, VersionHistory, VersionManager, ChangeType};
pub use differential::{MemoryDiff, DiffAnalyzer, ChangeSet, DiffMetrics};
pub use patterns::{TemporalPattern, PatternDetector, TemporalTrend, AccessPattern};
pub use evolution::{
    MemoryEvolution,
    EvolutionTracker,
    EvolutionMetrics,
    EvolutionEvent,
    GlobalEvolutionMetrics,
    EvolutionData,
};
pub use decay_models::{
    TemporalDecayModels,
    DecayModelType,
    DecayConfig,
    DecayParameters,
    DecayContext,
    DecayResult,
};

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Temporal memory manager that tracks changes over time
pub struct TemporalMemoryManager {
    /// Version manager for tracking memory changes
    version_manager: VersionManager,
    /// Differential analyzer for comparing memory states
    diff_analyzer: DiffAnalyzer,
    /// Pattern detector for identifying temporal patterns
    pattern_detector: PatternDetector,
    /// Evolution tracker for monitoring memory development
    evolution_tracker: EvolutionTracker,
    /// Configuration
    config: TemporalConfig,
}

/// Configuration for temporal tracking
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Maximum number of versions to keep per memory
    pub max_versions_per_memory: usize,
    /// Maximum age of versions to retain
    pub max_version_age_days: u64,
    /// Enable automatic pattern detection
    pub enable_pattern_detection: bool,
    /// Enable evolution tracking
    pub enable_evolution_tracking: bool,
    /// Minimum time between versions for the same memory
    pub min_version_interval_minutes: u64,
    /// Enable differential compression
    pub enable_diff_compression: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            max_versions_per_memory: 100,
            max_version_age_days: 365,
            enable_pattern_detection: true,
            enable_evolution_tracking: true,
            min_version_interval_minutes: 5,
            enable_diff_compression: true,
        }
    }
}

/// Temporal query for analyzing memory changes over time
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    /// Memory key to analyze
    pub memory_key: Option<String>,
    /// Time range for analysis
    pub time_range: Option<TimeRange>,
    /// Types of changes to include
    pub change_types: Vec<ChangeType>,
    /// Minimum significance threshold
    pub min_significance: Option<f64>,
    /// Include pattern analysis
    pub include_patterns: bool,
    /// Include evolution metrics
    pub include_evolution: bool,
}

/// Time range for temporal queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeRange {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }

    pub fn last_hours(hours: u64) -> Self {
        let end = Utc::now();
        let start = end - Duration::hours(hours as i64);
        Self { start, end }
    }

    pub fn last_days(days: u64) -> Self {
        let end = Utc::now();
        let start = end - Duration::days(days as i64);
        Self { start, end }
    }

    pub fn contains(&self, time: DateTime<Utc>) -> bool {
        time >= self.start && time <= self.end
    }

    pub fn duration(&self) -> Duration {
        self.end - self.start
    }
}

/// Result of temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    /// Query that generated this analysis
    pub query: String,
    /// Time range analyzed
    pub time_range: TimeRange,
    /// Version history for the analyzed memories
    pub version_history: Vec<MemoryVersion>,
    /// Detected changes and differences
    pub changes: Vec<ChangeSet>,
    /// Identified temporal patterns
    pub patterns: Vec<TemporalPattern>,
    /// Evolution metrics (per-memory or global)
    pub evolution_metrics: Option<EvolutionData>,
    /// Summary statistics
    pub summary: TemporalSummary,
}

/// Summary statistics for temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSummary {
    /// Total number of versions analyzed
    pub total_versions: usize,
    /// Total number of changes detected
    pub total_changes: usize,
    /// Most active time period
    pub most_active_period: Option<TimeRange>,
    /// Average change frequency (changes per day)
    pub avg_change_frequency: f64,
    /// Most common change type
    pub most_common_change_type: Option<ChangeType>,
    /// Stability score (0.0 = very unstable, 1.0 = very stable)
    pub stability_score: f64,
}

/// Basic usage statistics for the temporal system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalUsageStats {
    /// Total versions stored
    pub total_versions: usize,
    /// Total diffs calculated
    pub total_diffs: usize,
    /// Total evolution events
    pub total_evolution_events: usize,
}

impl TemporalMemoryManager {
    /// Create a new temporal memory manager
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            version_manager: VersionManager::new(config.clone()),
            diff_analyzer: DiffAnalyzer::new(),
            pattern_detector: PatternDetector::new(),
            evolution_tracker: EvolutionTracker::new(),
            config,
        }
    }

    /// Track a new version of a memory entry
    pub async fn track_memory_change(
        &mut self,
        memory: &MemoryEntry,
        change_type: ChangeType,
    ) -> Result<Uuid> {
        // Create a new version
        let version_id = self.version_manager.create_version(memory, &change_type).await?;

        // Analyze differences if this isn't the first version
        if let Some(previous_version) = self.version_manager.get_previous_version(&memory.key).await? {
            let diff = self.diff_analyzer.analyze_difference(&previous_version.memory, memory).await?;
            self.version_manager.attach_diff(version_id, diff).await?;
        }

        // Update pattern detection if enabled
        if self.config.enable_pattern_detection {
            self.pattern_detector.update_patterns(&memory.key, memory).await?;
        }

        // Update evolution tracking if enabled
        if self.config.enable_evolution_tracking {
            self.evolution_tracker.track_change(&memory.key, memory, change_type).await?;
        }

        Ok(version_id)
    }

    /// Get the complete version history for a memory
    pub async fn get_version_history(&self, memory_key: &str) -> Result<VersionHistory> {
        self.version_manager.get_history(memory_key).await
    }

    /// Analyze temporal patterns for a memory or set of memories
    pub async fn analyze_temporal_patterns(&self, query: TemporalQuery) -> Result<TemporalAnalysis> {
        let time_range = query.time_range.unwrap_or_else(|| TimeRange::last_days(30));
        
        // Get version history for the specified time range
        let version_history = if let Some(memory_key) = &query.memory_key {
            self.version_manager.get_history_in_range(memory_key, &time_range).await?
        } else {
            self.version_manager.get_all_history_in_range(&time_range).await?
        };

        // Analyze changes
        let changes = self.diff_analyzer.analyze_changes_in_range(&time_range).await?;

        // Detect patterns if requested
        let patterns = if query.include_patterns {
            self.pattern_detector.detect_patterns_in_range(&time_range).await?
        } else {
            Vec::new()
        };

        // Get evolution metrics if requested
        let evolution_metrics = if query.include_evolution {
            if let Some(memory_key) = &query.memory_key {
                let metrics = self.evolution_tracker.get_metrics(memory_key).await?;
                Some(EvolutionData::PerMemory(metrics))
            } else {
                let metrics = self.evolution_tracker.get_global_metrics().await?;
                Some(EvolutionData::Global(metrics))
            }
        } else {
            None
        };

        // Calculate summary statistics
        let summary = self.calculate_temporal_summary(&version_history, &changes, &time_range);

        Ok(TemporalAnalysis {
            query: format!("Temporal analysis for {:?}", query.memory_key),
            time_range,
            version_history,
            changes,
            patterns,
            evolution_metrics,
            summary,
        })
    }

    /// Compare two memory states and get detailed differences
    pub async fn compare_memory_states(
        &mut self,
        memory1: &MemoryEntry,
        memory2: &MemoryEntry,
    ) -> Result<MemoryDiff> {
        self.diff_analyzer.analyze_difference(memory1, memory2).await
    }

    /// Get memories that have changed within a time range
    pub async fn get_changed_memories(&self, time_range: &TimeRange) -> Result<Vec<String>> {
        self.version_manager.get_changed_memories_in_range(time_range).await
    }

    /// Get the most active memories (most frequently changed)
    pub async fn get_most_active_memories(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        self.version_manager.get_most_active_memories(limit).await
    }

    /// Cleanup old versions based on configuration
    pub async fn cleanup_old_versions(&mut self) -> Result<usize> {
        let cutoff_date = Utc::now() - Duration::days(self.config.max_version_age_days as i64);
        self.version_manager.cleanup_versions_before(cutoff_date).await
    }

    /// Retrieve aggregated usage statistics
    pub async fn get_usage_stats(&self) -> Result<TemporalUsageStats> {
        let total_versions = self.version_manager.total_versions();
        let diff_metrics = self.diff_analyzer.get_metrics();
        let evolution_metrics = self.evolution_tracker.get_global_metrics().await?;

        Ok(TemporalUsageStats {
            total_versions,
            total_diffs: diff_metrics.total_diffs,
            total_evolution_events: evolution_metrics.total_events,
        })
    }

    /// Get differential analysis metrics
    pub fn get_diff_metrics(&self) -> DiffMetrics {
        self.diff_analyzer.get_metrics()
    }

    /// Get global evolution metrics
    pub async fn get_global_evolution_metrics(&self) -> Result<GlobalEvolutionMetrics> {
        self.evolution_tracker.get_global_metrics().await
    }

    /// Get evolution metrics for a specific memory
    pub async fn get_evolution_metrics(&self, memory_key: &str) -> Result<EvolutionMetrics> {
        self.evolution_tracker.get_metrics(memory_key).await
    }

    /// Calculate temporal summary statistics
    pub fn calculate_temporal_summary(
        &self,
        version_history: &[MemoryVersion],
        changes: &[ChangeSet],
        time_range: &TimeRange,
    ) -> TemporalSummary {
        let total_versions = version_history.len();
        let total_changes = changes.len();

        // Calculate average change frequency
        let duration_days = time_range.duration().num_days() as f64;
        let avg_change_frequency = if duration_days > 0.0 {
            total_changes as f64 / duration_days
        } else {
            0.0
        };

        // Find most common change type
        let mut change_type_counts: HashMap<ChangeType, usize> = HashMap::new();
        for version in version_history {
            *change_type_counts.entry(version.change_type.clone()).or_insert(0) += 1;
        }
        let most_common_change_type = change_type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(change_type, _)| change_type);

        // Calculate stability score (inverse of change frequency)
        let stability_score = if avg_change_frequency > 0.0 {
            (1.0 / (1.0 + avg_change_frequency)).min(1.0)
        } else {
            1.0
        };

        let bucket_size = Self::determine_bucket_size(time_range);
        let most_active_period = Self::detect_most_active_period(version_history, bucket_size);

        TemporalSummary {
            total_versions,
            total_changes,
            most_active_period,
            avg_change_frequency,
            most_common_change_type,
            stability_score,
        }
    }

    /// Determine bucket size for activity analysis based on overall range
    fn determine_bucket_size(time_range: &TimeRange) -> Duration {
        if time_range.duration().num_hours() <= 24 {
            Duration::hours(1)
        } else {
            Duration::days(1)
        }
    }

    /// Detect the most active period using the given bucket size
    fn detect_most_active_period(
        version_history: &[MemoryVersion],
        bucket_size: Duration,
    ) -> Option<TimeRange> {

        let mut counts: HashMap<DateTime<Utc>, usize> = HashMap::new();
        for version in version_history {
            let ts = version.created_at.timestamp();
            let bucket_start = ts - (ts % bucket_size.num_seconds());
            if let Some(bucket) = DateTime::from_timestamp(bucket_start, 0) {
                *counts.entry(bucket).or_insert(0) += 1;
            }
        }

        let (start, _) = counts.into_iter().max_by_key(|(_, c)| *c)?;
        Some(TimeRange::new(start, start + bucket_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_config_default() {
        let config = TemporalConfig::default();
        assert_eq!(config.max_versions_per_memory, 100);
        assert_eq!(config.max_version_age_days, 365);
        assert!(config.enable_pattern_detection);
        assert!(config.enable_evolution_tracking);
        assert_eq!(config.min_version_interval_minutes, 5);
        assert!(config.enable_diff_compression);
    }

    #[test]
    fn test_temporal_config_clone() {
        let config1 = TemporalConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.max_versions_per_memory, config2.max_versions_per_memory);
        assert_eq!(config1.max_version_age_days, config2.max_version_age_days);
        assert_eq!(config1.enable_pattern_detection, config2.enable_pattern_detection);
    }

    #[test]
    fn test_temporal_config_custom_values() {
        let config = TemporalConfig {
            max_versions_per_memory: 50,
            max_version_age_days: 180,
            enable_pattern_detection: false,
            enable_evolution_tracking: false,
            min_version_interval_minutes: 10,
            enable_diff_compression: false,
        };

        assert_eq!(config.max_versions_per_memory, 50);
        assert_eq!(config.max_version_age_days, 180);
        assert!(!config.enable_pattern_detection);
        assert!(!config.enable_evolution_tracking);
        assert_eq!(config.min_version_interval_minutes, 10);
        assert!(!config.enable_diff_compression);
    }

    #[test]
    fn test_time_range_new() {
        let start = Utc::now();
        let end = start + Duration::hours(2);
        let range = TimeRange::new(start, end);

        assert_eq!(range.start, start);
        assert_eq!(range.end, end);
    }

    #[test]
    fn test_time_range_last_hours() {
        let range = TimeRange::last_hours(24);
        let duration = range.duration();

        // Should be approximately 24 hours (allowing small timing variance)
        assert!((duration.num_hours() - 24).abs() < 1);
    }

    #[test]
    fn test_time_range_last_days() {
        let range = TimeRange::last_days(7);
        let duration = range.duration();

        // Should be approximately 7 days
        assert!((duration.num_days() - 7).abs() < 1);
    }

    #[test]
    fn test_time_range_contains() {
        let start = Utc::now();
        let end = start + Duration::hours(2);
        let range = TimeRange::new(start, end);

        // Time within range
        let mid_time = start + Duration::hours(1);
        assert!(range.contains(mid_time));

        // Time at boundaries
        assert!(range.contains(start));
        assert!(range.contains(end));

        // Time outside range
        let before = start - Duration::hours(1);
        assert!(!range.contains(before));

        let after = end + Duration::hours(1);
        assert!(!range.contains(after));
    }

    #[test]
    fn test_time_range_duration() {
        let start = Utc::now();
        let end = start + Duration::hours(5);
        let range = TimeRange::new(start, end);

        assert_eq!(range.duration(), Duration::hours(5));
    }

    #[test]
    fn test_time_range_serialization() {
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let range = TimeRange::new(start, end);

        let serialized = serde_json::to_string(&range).unwrap();
        let deserialized: TimeRange = serde_json::from_str(&serialized).unwrap();

        assert_eq!(range.start, deserialized.start);
        assert_eq!(range.end, deserialized.end);
    }

    #[test]
    fn test_time_range_clone() {
        let start = Utc::now();
        let end = start + Duration::hours(1);
        let range1 = TimeRange::new(start, end);
        let range2 = range1.clone();

        assert_eq!(range1.start, range2.start);
        assert_eq!(range1.end, range2.end);
    }

    #[test]
    fn test_temporal_memory_manager_new() {
        let config = TemporalConfig::default();
        let manager = TemporalMemoryManager::new(config.clone());

        // Manager should be initialized with the config
        assert_eq!(manager.config.max_versions_per_memory, config.max_versions_per_memory);
    }

    #[test]
    fn test_temporal_usage_stats_serialization() {
        let stats = TemporalUsageStats {
            total_versions: 100,
            total_diffs: 50,
            total_evolution_events: 25,
        };

        let serialized = serde_json::to_string(&stats).unwrap();
        let deserialized: TemporalUsageStats = serde_json::from_str(&serialized).unwrap();

        assert_eq!(stats.total_versions, deserialized.total_versions);
        assert_eq!(stats.total_diffs, deserialized.total_diffs);
        assert_eq!(stats.total_evolution_events, deserialized.total_evolution_events);
    }

    #[test]
    fn test_temporal_usage_stats_clone() {
        let stats1 = TemporalUsageStats {
            total_versions: 100,
            total_diffs: 50,
            total_evolution_events: 25,
        };

        let stats2 = stats1.clone();
        assert_eq!(stats1.total_versions, stats2.total_versions);
        assert_eq!(stats1.total_diffs, stats2.total_diffs);
    }

    #[test]
    fn test_temporal_summary_serialization() {
        let summary = TemporalSummary {
            total_versions: 50,
            total_changes: 30,
            most_active_period: Some(TimeRange::last_hours(1)),
            avg_change_frequency: 1.5,
            most_common_change_type: Some(ChangeType::Updated),
            stability_score: 0.8,
        };

        let serialized = serde_json::to_string(&summary).unwrap();
        let deserialized: TemporalSummary = serde_json::from_str(&serialized).unwrap();

        assert_eq!(summary.total_versions, deserialized.total_versions);
        assert_eq!(summary.total_changes, deserialized.total_changes);
        assert_eq!(summary.avg_change_frequency, deserialized.avg_change_frequency);
        assert_eq!(summary.stability_score, deserialized.stability_score);
    }

    #[test]
    fn test_temporal_summary_clone() {
        let summary1 = TemporalSummary {
            total_versions: 50,
            total_changes: 30,
            most_active_period: None,
            avg_change_frequency: 1.5,
            most_common_change_type: None,
            stability_score: 0.8,
        };

        let summary2 = summary1.clone();
        assert_eq!(summary1.total_versions, summary2.total_versions);
        assert_eq!(summary1.stability_score, summary2.stability_score);
    }

    #[test]
    fn test_temporal_query_clone() {
        let query1 = TemporalQuery {
            memory_key: Some("test_key".to_string()),
            time_range: Some(TimeRange::last_days(7)),
            change_types: vec![ChangeType::Created, ChangeType::Updated],
            min_significance: Some(0.5),
            include_patterns: true,
            include_evolution: true,
        };

        let query2 = query1.clone();
        assert_eq!(query1.memory_key, query2.memory_key);
        assert_eq!(query1.include_patterns, query2.include_patterns);
        assert_eq!(query1.include_evolution, query2.include_evolution);
    }

    #[test]
    fn test_temporal_analysis_serialization() {
        let time_range = TimeRange::last_hours(24);
        let analysis = TemporalAnalysis {
            query: "test query".to_string(),
            time_range: time_range.clone(),
            version_history: vec![],
            changes: vec![],
            patterns: vec![],
            evolution_metrics: None,
            summary: TemporalSummary {
                total_versions: 0,
                total_changes: 0,
                most_active_period: None,
                avg_change_frequency: 0.0,
                most_common_change_type: None,
                stability_score: 1.0,
            },
        };

        let serialized = serde_json::to_string(&analysis).unwrap();
        let deserialized: TemporalAnalysis = serde_json::from_str(&serialized).unwrap();

        assert_eq!(analysis.query, deserialized.query);
        assert_eq!(analysis.summary.total_versions, deserialized.summary.total_versions);
    }

    #[test]
    fn test_determine_bucket_size_hourly() {
        let range = TimeRange::last_hours(12);
        let bucket_size = TemporalMemoryManager::determine_bucket_size(&range);
        assert_eq!(bucket_size, Duration::hours(1));
    }

    #[test]
    fn test_determine_bucket_size_daily() {
        let range = TimeRange::last_days(7);
        let bucket_size = TemporalMemoryManager::determine_bucket_size(&range);
        assert_eq!(bucket_size, Duration::days(1));
    }

    #[test]
    fn test_time_range_boundary_conditions() {
        let now = Utc::now();
        let range = TimeRange::new(now, now);

        // Zero duration range
        assert_eq!(range.duration(), Duration::zero());
        assert!(range.contains(now));
    }

    #[test]
    fn test_temporal_config_edge_cases() {
        // Test with minimal values
        let config = TemporalConfig {
            max_versions_per_memory: 1,
            max_version_age_days: 1,
            enable_pattern_detection: false,
            enable_evolution_tracking: false,
            min_version_interval_minutes: 1,
            enable_diff_compression: false,
        };

        assert_eq!(config.max_versions_per_memory, 1);
        assert_eq!(config.max_version_age_days, 1);
    }

    #[test]
    fn test_temporal_usage_stats_zero_values() {
        let stats = TemporalUsageStats {
            total_versions: 0,
            total_diffs: 0,
            total_evolution_events: 0,
        };

        assert_eq!(stats.total_versions, 0);
        assert_eq!(stats.total_diffs, 0);
        assert_eq!(stats.total_evolution_events, 0);
    }
}
