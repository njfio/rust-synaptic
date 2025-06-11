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

// Re-export commonly used types
pub use versioning::{MemoryVersion, VersionHistory, VersionManager, ChangeType};
pub use differential::{MemoryDiff, DiffAnalyzer, ChangeSet, DiffMetrics};
pub use patterns::{TemporalPattern, PatternDetector, TemporalTrend, AccessPattern};
pub use evolution::{MemoryEvolution, EvolutionTracker, EvolutionMetrics, EvolutionEvent};

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
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
    /// Evolution metrics
    pub evolution_metrics: Option<EvolutionMetrics>,
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
            if let Some(_memory_key) = &query.memory_key {
                // TODO: Return proper evolution metrics
                None
            } else {
                // TODO: Return global evolution metrics
                None
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

    /// Calculate temporal summary statistics
    fn calculate_temporal_summary(
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

        TemporalSummary {
            total_versions,
            total_changes,
            most_active_period: None, // TODO: Implement period detection
            avg_change_frequency,
            most_common_change_type,
            stability_score,
        }
    }
}
