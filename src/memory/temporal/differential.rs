//! Differential analysis for memory changes

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::memory::temporal::{TimeRange, ChangeType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Detailed difference between two memory entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDiff {
    /// Unique identifier for this diff
    pub id: Uuid,
    /// When this diff was created
    pub created_at: DateTime<Utc>,
    /// Source memory (before change)
    pub from_memory_id: Uuid,
    /// Target memory (after change)
    pub to_memory_id: Uuid,
    /// Content changes
    pub content_changes: ContentDiff,
    /// Metadata changes
    pub metadata_changes: MetadataDiff,
    /// Overall significance score (0.0 to 1.0)
    pub significance_score: f64,
    /// Size of the diff in bytes
    pub diff_size: usize,
    /// Compression ratio if compressed
    pub compression_ratio: Option<f64>,
}

/// Differences in memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDiff {
    /// Type of content change
    pub change_type: ContentChangeType,
    /// Added text segments
    pub additions: Vec<TextSegment>,
    /// Removed text segments
    pub deletions: Vec<TextSegment>,
    /// Modified text segments
    pub modifications: Vec<TextModification>,
    /// Similarity score between old and new content (0.0 to 1.0)
    pub similarity_score: f64,
    /// Length change (positive = growth, negative = shrinkage)
    pub length_delta: i64,
}

/// Types of content changes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentChangeType {
    /// Content was completely replaced
    Replaced,
    /// Content was appended to
    Appended,
    /// Content was prepended to
    Prepended,
    /// Content was inserted in the middle
    Inserted,
    /// Content was partially modified
    Modified,
    /// Content was truncated
    Truncated,
    /// Content was reformatted
    Reformatted,
    /// No content change
    Unchanged,
}

/// A segment of text that was added or removed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSegment {
    /// Position in the text (character index)
    pub position: usize,
    /// Length of the segment
    pub length: usize,
    /// The actual text content
    pub content: String,
    /// Context around this segment
    pub context: Option<String>,
}

/// A modification to existing text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextModification {
    /// Position of the modification
    pub position: usize,
    /// Original text
    pub old_text: String,
    /// New text
    pub new_text: String,
    /// Type of modification
    pub modification_type: ModificationType,
}

/// Types of text modifications
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationType {
    /// Text was substituted
    Substitution,
    /// Text was corrected (spelling, grammar)
    Correction,
    /// Text was expanded with more detail
    Expansion,
    /// Text was condensed
    Condensation,
    /// Text was rephrased
    Rephrase,
}

/// Differences in memory metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDiff {
    /// Changes to tags
    pub tag_changes: TagChanges,
    /// Changes to importance score
    pub importance_change: Option<f64>,
    /// Changes to confidence score
    pub confidence_change: Option<f64>,
    /// Changes to custom fields
    pub custom_field_changes: HashMap<String, FieldChange>,
    /// Changes to memory type
    pub memory_type_change: Option<(String, String)>, // (old, new)
}

/// Changes to tags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagChanges {
    /// Tags that were added
    pub added: Vec<String>,
    /// Tags that were removed
    pub removed: Vec<String>,
    /// Tags that remained unchanged
    pub unchanged: Vec<String>,
}

/// Change to a custom field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldChange {
    /// Field was added
    Added(String),
    /// Field was removed
    Removed(String),
    /// Field value was changed
    Modified { old_value: String, new_value: String },
}

/// Set of changes that occurred together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSet {
    /// Unique identifier
    pub id: Uuid,
    /// When these changes occurred
    pub timestamp: DateTime<Utc>,
    /// Memory key that was changed
    pub memory_key: String,
    /// Type of change
    pub change_type: ChangeType,
    /// Individual diffs in this change set
    pub diffs: Vec<MemoryDiff>,
    /// Overall impact score
    pub impact_score: f64,
    /// Description of the changes
    pub description: String,
}

/// Metrics for differential analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffMetrics {
    /// Total number of diffs analyzed
    pub total_diffs: usize,
    /// Average significance score
    pub avg_significance: f64,
    /// Most common change type
    pub most_common_change: ContentChangeType,
    /// Average content similarity
    pub avg_content_similarity: f64,
    /// Total size of all diffs
    pub total_diff_size: usize,
    /// Compression efficiency
    pub avg_compression_ratio: Option<f64>,
}

/// Differential analyzer for comparing memory states
pub struct DiffAnalyzer {
    /// Cache of recent diffs
    diff_cache: HashMap<(Uuid, Uuid), MemoryDiff>,
    /// Change sets organized by time
    change_sets: Vec<ChangeSet>,
    /// Configuration
    config: DiffConfig,
}

/// Configuration for differential analysis
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Enable content compression for large diffs
    pub enable_compression: bool,
    /// Minimum significance threshold for storing diffs
    pub min_significance_threshold: f64,
    /// Maximum cache size for diffs
    pub max_cache_size: usize,
    /// Enable detailed text analysis
    pub enable_detailed_text_analysis: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            min_significance_threshold: 0.1,
            max_cache_size: 1000,
            enable_detailed_text_analysis: true,
        }
    }
}

impl DiffAnalyzer {
    /// Create a new differential analyzer
    pub fn new() -> Self {
        Self {
            diff_cache: HashMap::new(),
            change_sets: Vec::new(),
            config: DiffConfig::default(),
        }
    }

    /// Analyze the difference between two memory entries
    pub async fn analyze_difference(
        &mut self,
        from_memory: &MemoryEntry,
        to_memory: &MemoryEntry,
    ) -> Result<MemoryDiff> {
        // Check cache first
        let cache_key = (from_memory.id(), to_memory.id());
        if let Some(cached_diff) = self.diff_cache.get(&cache_key) {
            return Ok(cached_diff.clone());
        }

        // Analyze content differences
        let content_changes = self.analyze_content_diff(&from_memory.value, &to_memory.value).await?;

        // Analyze metadata differences
        let metadata_changes = self.analyze_metadata_diff(&from_memory.metadata, &to_memory.metadata).await?;

        // Calculate significance score
        let significance_score = self.calculate_significance_score(&content_changes, &metadata_changes);

        // Calculate diff size
        let diff_size = self.calculate_diff_size(&content_changes, &metadata_changes);

        let diff = MemoryDiff {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            from_memory_id: from_memory.id(),
            to_memory_id: to_memory.id(),
            content_changes,
            metadata_changes,
            significance_score,
            diff_size,
            compression_ratio: None, // TODO: Implement compression
        };

        // Cache the diff if significant enough
        if significance_score >= self.config.min_significance_threshold {
            self.cache_diff(cache_key, diff.clone());
        }

        Ok(diff)
    }

    /// Analyze content differences between two text strings
    async fn analyze_content_diff(&self, old_content: &str, new_content: &str) -> Result<ContentDiff> {
        // Calculate similarity score
        let similarity_score = self.calculate_text_similarity(old_content, new_content);

        // Determine change type
        let change_type = self.determine_content_change_type(old_content, new_content);

        // Find additions, deletions, and modifications
        let (additions, deletions, modifications) = if self.config.enable_detailed_text_analysis {
            self.perform_detailed_text_analysis(old_content, new_content).await?
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Calculate length delta
        let length_delta = new_content.len() as i64 - old_content.len() as i64;

        Ok(ContentDiff {
            change_type,
            additions,
            deletions,
            modifications,
            similarity_score,
            length_delta,
        })
    }

    /// Analyze metadata differences
    async fn analyze_metadata_diff(
        &self,
        old_metadata: &crate::memory::types::MemoryMetadata,
        new_metadata: &crate::memory::types::MemoryMetadata,
    ) -> Result<MetadataDiff> {
        // Analyze tag changes
        let old_tags: HashSet<_> = old_metadata.tags.iter().collect();
        let new_tags: HashSet<_> = new_metadata.tags.iter().collect();
        
        let added: Vec<String> = new_tags.difference(&old_tags).map(|s| s.to_string()).collect();
        let removed: Vec<String> = old_tags.difference(&new_tags).map(|s| s.to_string()).collect();
        let unchanged: Vec<String> = old_tags.intersection(&new_tags).map(|s| s.to_string()).collect();
        
        let tag_changes = TagChanges { added, removed, unchanged };

        // Analyze importance change
        let importance_change = if (old_metadata.importance - new_metadata.importance).abs() > 0.01 {
            Some(new_metadata.importance - old_metadata.importance)
        } else {
            None
        };

        // Analyze confidence change
        let confidence_change = if (old_metadata.confidence - new_metadata.confidence).abs() > 0.01 {
            Some(new_metadata.confidence - old_metadata.confidence)
        } else {
            None
        };

        // Analyze custom field changes
        let mut custom_field_changes = HashMap::new();
        
        // Check for added and modified fields
        for (key, new_value) in &new_metadata.custom_fields {
            match old_metadata.custom_fields.get(key) {
                Some(old_value) if old_value != new_value => {
                    custom_field_changes.insert(
                        key.clone(),
                        FieldChange::Modified {
                            old_value: old_value.clone(),
                            new_value: new_value.clone(),
                        },
                    );
                }
                None => {
                    custom_field_changes.insert(key.clone(), FieldChange::Added(new_value.clone()));
                }
                _ => {} // No change
            }
        }
        
        // Check for removed fields
        for (key, old_value) in &old_metadata.custom_fields {
            if !new_metadata.custom_fields.contains_key(key) {
                custom_field_changes.insert(key.clone(), FieldChange::Removed(old_value.clone()));
            }
        }

        // Check for memory type change (this would be unusual but possible)
        let memory_type_change = None; // Memory type is not in metadata, it's in the entry itself

        Ok(MetadataDiff {
            tag_changes,
            importance_change,
            confidence_change,
            custom_field_changes,
            memory_type_change,
        })
    }

    /// Calculate text similarity using a simple algorithm
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        if text1 == text2 {
            return 1.0;
        }
        
        if text1.is_empty() && text2.is_empty() {
            return 1.0;
        }
        
        if text1.is_empty() || text2.is_empty() {
            return 0.0;
        }

        // Use Jaccard similarity on word level
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Determine the type of content change
    fn determine_content_change_type(&self, old_content: &str, new_content: &str) -> ContentChangeType {
        if old_content == new_content {
            return ContentChangeType::Unchanged;
        }
        
        if old_content.is_empty() {
            return ContentChangeType::Replaced;
        }
        
        if new_content.is_empty() {
            return ContentChangeType::Truncated;
        }
        
        // Check for append
        if new_content.starts_with(old_content) {
            return ContentChangeType::Appended;
        }
        
        // Check for prepend
        if new_content.ends_with(old_content) {
            return ContentChangeType::Prepended;
        }
        
        // Check for insertion (old content is contained in new)
        if new_content.contains(old_content) {
            return ContentChangeType::Inserted;
        }
        
        // Check for truncation (new content is contained in old)
        if old_content.contains(new_content) {
            return ContentChangeType::Truncated;
        }
        
        // Calculate similarity to determine if it's modification or replacement
        let similarity = self.calculate_text_similarity(old_content, new_content);
        if similarity > 0.3 {
            ContentChangeType::Modified
        } else {
            ContentChangeType::Replaced
        }
    }

    /// Perform detailed text analysis to find specific changes
    async fn perform_detailed_text_analysis(
        &self,
        old_text: &str,
        new_text: &str,
    ) -> Result<(Vec<TextSegment>, Vec<TextSegment>, Vec<TextModification>)> {
        // This is a simplified implementation
        // A full implementation would use algorithms like Myers' diff algorithm
        
        let additions = Vec::new(); // TODO: Implement detailed diff
        let deletions = Vec::new(); // TODO: Implement detailed diff
        let modifications = Vec::new(); // TODO: Implement detailed diff
        
        Ok((additions, deletions, modifications))
    }

    /// Calculate significance score for a diff
    fn calculate_significance_score(&self, content_diff: &ContentDiff, metadata_diff: &MetadataDiff) -> f64 {
        let mut score = 0.0;
        
        // Content significance
        match content_diff.change_type {
            ContentChangeType::Replaced => score += 1.0,
            ContentChangeType::Modified => score += 0.7,
            ContentChangeType::Appended | ContentChangeType::Prepended => score += 0.5,
            ContentChangeType::Inserted => score += 0.4,
            ContentChangeType::Truncated => score += 0.6,
            ContentChangeType::Reformatted => score += 0.3,
            ContentChangeType::Unchanged => score += 0.0,
        }
        
        // Length change significance
        let length_change_ratio = content_diff.length_delta.abs() as f64 / 
            (content_diff.length_delta.abs() + 100) as f64; // Normalize
        score += length_change_ratio * 0.3;
        
        // Metadata significance
        if !metadata_diff.tag_changes.added.is_empty() || !metadata_diff.tag_changes.removed.is_empty() {
            score += 0.2;
        }
        
        if metadata_diff.importance_change.is_some() {
            score += 0.1;
        }
        
        if !metadata_diff.custom_field_changes.is_empty() {
            score += 0.1;
        }
        
        score.min(1.0)
    }

    /// Calculate the size of a diff in bytes
    fn calculate_diff_size(&self, content_diff: &ContentDiff, metadata_diff: &MetadataDiff) -> usize {
        let mut size = 0;
        
        // Content diff size
        for addition in &content_diff.additions {
            size += addition.content.len();
        }
        for deletion in &content_diff.deletions {
            size += deletion.content.len();
        }
        for modification in &content_diff.modifications {
            size += modification.old_text.len() + modification.new_text.len();
        }
        
        // Metadata diff size
        for tag in &metadata_diff.tag_changes.added {
            size += tag.len();
        }
        for tag in &metadata_diff.tag_changes.removed {
            size += tag.len();
        }
        
        size
    }

    /// Cache a diff
    fn cache_diff(&mut self, key: (Uuid, Uuid), diff: MemoryDiff) {
        if self.diff_cache.len() >= self.config.max_cache_size {
            // Remove oldest entry (simple FIFO)
            if let Some(oldest_key) = self.diff_cache.keys().next().cloned() {
                self.diff_cache.remove(&oldest_key);
            }
        }
        
        self.diff_cache.insert(key, diff);
    }

    /// Analyze changes within a time range
    pub async fn analyze_changes_in_range(&self, _time_range: &TimeRange) -> Result<Vec<ChangeSet>> {
        // Filter change sets by time range
        Ok(self.change_sets.clone()) // TODO: Implement proper filtering
    }

    /// Get differential metrics
    pub fn get_metrics(&self) -> DiffMetrics {
        let total_diffs = self.diff_cache.len();
        
        if total_diffs == 0 {
            return DiffMetrics {
                total_diffs: 0,
                avg_significance: 0.0,
                most_common_change: ContentChangeType::Unchanged,
                avg_content_similarity: 0.0,
                total_diff_size: 0,
                avg_compression_ratio: None,
            };
        }
        
        let avg_significance = self.diff_cache.values()
            .map(|d| d.significance_score)
            .sum::<f64>() / total_diffs as f64;
        
        let avg_content_similarity = self.diff_cache.values()
            .map(|d| d.content_changes.similarity_score)
            .sum::<f64>() / total_diffs as f64;
        
        let total_diff_size = self.diff_cache.values()
            .map(|d| d.diff_size)
            .sum();
        
        // Find most common change type
        let mut change_type_counts = HashMap::new();
        for diff in self.diff_cache.values() {
            *change_type_counts.entry(diff.content_changes.change_type.clone()).or_insert(0) += 1;
        }
        
        let most_common_change = change_type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(change_type, _)| change_type)
            .unwrap_or(ContentChangeType::Unchanged);
        
        DiffMetrics {
            total_diffs,
            avg_significance,
            most_common_change,
            avg_content_similarity,
            total_diff_size,
            avg_compression_ratio: None, // TODO: Implement compression metrics
        }
    }
}

impl Default for DiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
