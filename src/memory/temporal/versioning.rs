//! Memory versioning and change tracking system

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::memory::temporal::{TemporalConfig, TimeRange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use uuid::Uuid;

/// Types of changes that can occur to memory entries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ChangeType {
    /// Memory was created
    Created,
    /// Memory content was updated
    Updated,
    /// Memory metadata was modified
    MetadataChanged,
    /// Memory was accessed (read)
    Accessed,
    /// Memory was deleted
    Deleted,
    /// Memory was restored from backup
    Restored,
    /// Memory was merged with another memory
    Merged,
    /// Memory was split into multiple memories
    Split,
    /// Memory was summarized
    Summarized,
    /// Memory importance was recalculated
    ImportanceUpdated,
    /// Custom change type
    Custom(String),
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeType::Created => write!(f, "created"),
            ChangeType::Updated => write!(f, "updated"),
            ChangeType::MetadataChanged => write!(f, "metadata_changed"),
            ChangeType::Accessed => write!(f, "accessed"),
            ChangeType::Deleted => write!(f, "deleted"),
            ChangeType::Restored => write!(f, "restored"),
            ChangeType::Merged => write!(f, "merged"),
            ChangeType::Split => write!(f, "split"),
            ChangeType::Summarized => write!(f, "summarized"),
            ChangeType::ImportanceUpdated => write!(f, "importance_updated"),
            ChangeType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// A specific version of a memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVersion {
    /// Unique version identifier
    pub id: Uuid,
    /// The memory entry at this version
    pub memory: MemoryEntry,
    /// When this version was created
    pub created_at: DateTime<Utc>,
    /// Type of change that created this version
    pub change_type: ChangeType,
    /// Version number (incremental)
    pub version_number: u64,
    /// Size of this version in bytes
    pub size_bytes: usize,
    /// Checksum for integrity verification
    pub checksum: String,
    /// Optional change description
    pub change_description: Option<String>,
    /// User or system that made the change
    pub changed_by: Option<String>,
    /// Attached differential data
    pub diff_data: Option<super::differential::MemoryDiff>,
}

impl MemoryVersion {
    /// Create a new memory version
    pub fn new(
        memory: MemoryEntry,
        change_type: ChangeType,
        version_number: u64,
    ) -> Self {
        let size_bytes = memory.estimated_size();
        let checksum = Self::calculate_checksum(&memory);
        
        Self {
            id: Uuid::new_v4(),
            memory,
            created_at: Utc::now(),
            change_type,
            version_number,
            size_bytes,
            checksum,
            change_description: None,
            changed_by: None,
            diff_data: None,
        }
    }

    /// Calculate checksum for a memory entry
    fn calculate_checksum(memory: &MemoryEntry) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        memory.key.hash(&mut hasher);
        memory.value.hash(&mut hasher);
        memory.memory_type.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Verify the integrity of this version
    pub fn verify_integrity(&self) -> bool {
        Self::calculate_checksum(&self.memory) == self.checksum
    }

    /// Get the age of this version
    pub fn age(&self) -> chrono::Duration {
        Utc::now() - self.created_at
    }

    /// Check if this version is significant (not just an access)
    pub fn is_significant(&self) -> bool {
        !matches!(self.change_type, ChangeType::Accessed)
    }
}

/// Complete version history for a memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// Memory key this history belongs to
    pub memory_key: String,
    /// All versions in chronological order (oldest first)
    pub versions: Vec<MemoryVersion>,
    /// Total number of changes
    pub total_changes: usize,
    /// First version timestamp
    pub first_version_at: Option<DateTime<Utc>>,
    /// Last version timestamp
    pub last_version_at: Option<DateTime<Utc>>,
    /// Most common change type
    pub most_common_change: Option<ChangeType>,
}

impl VersionHistory {
    /// Create a new version history
    pub fn new(memory_key: String) -> Self {
        Self {
            memory_key,
            versions: Vec::new(),
            total_changes: 0,
            first_version_at: None,
            last_version_at: None,
            most_common_change: None,
        }
    }

    /// Add a version to the history
    pub fn add_version(&mut self, version: MemoryVersion) {
        if self.first_version_at.is_none() {
            self.first_version_at = Some(version.created_at);
        }
        self.last_version_at = Some(version.created_at);
        
        self.versions.push(version);
        self.total_changes += 1;
        self.update_most_common_change();
    }

    /// Get the latest version
    pub fn latest_version(&self) -> Option<&MemoryVersion> {
        self.versions.last()
    }

    /// Get a specific version by number
    pub fn get_version(&self, version_number: u64) -> Option<&MemoryVersion> {
        self.versions.iter().find(|v| v.version_number == version_number)
    }

    /// Get versions within a time range
    pub fn get_versions_in_range(&self, time_range: &TimeRange) -> Vec<&MemoryVersion> {
        self.versions
            .iter()
            .filter(|v| time_range.contains(v.created_at))
            .collect()
    }

    /// Get the change frequency (changes per day)
    pub fn change_frequency(&self) -> f64 {
        if let (Some(first), Some(last)) = (self.first_version_at, self.last_version_at) {
            let duration_days = (last - first).num_days() as f64;
            if duration_days > 0.0 {
                self.total_changes as f64 / duration_days
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Update the most common change type
    fn update_most_common_change(&mut self) {
        let mut change_counts: HashMap<ChangeType, usize> = HashMap::new();
        for version in &self.versions {
            *change_counts.entry(version.change_type.clone()).or_insert(0) += 1;
        }
        
        self.most_common_change = change_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(change_type, _)| change_type);
    }

    /// Get significant versions only (excluding access-only changes)
    pub fn significant_versions(&self) -> Vec<&MemoryVersion> {
        self.versions.iter().filter(|v| v.is_significant()).collect()
    }

    /// Calculate total size of all versions
    pub fn total_size(&self) -> usize {
        self.versions.iter().map(|v| v.size_bytes).sum()
    }
}

/// Version manager for tracking memory changes
pub struct VersionManager {
    /// Version histories for each memory key
    histories: HashMap<String, VersionHistory>,
    /// Configuration
    config: TemporalConfig,
    /// Version counter for generating version numbers
    version_counters: HashMap<String, u64>,
}

impl VersionManager {
    /// Create a new version manager
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            histories: HashMap::new(),
            config,
            version_counters: HashMap::new(),
        }
    }

    /// Create a new version for a memory entry
    pub async fn create_version(
        &mut self,
        memory: &MemoryEntry,
        change_type: &ChangeType,
    ) -> Result<Uuid> {
        // Check if enough time has passed since last version (for non-significant changes)
        if !matches!(change_type, ChangeType::Created | ChangeType::Updated | ChangeType::Deleted) {
            if let Some(history) = self.histories.get(&memory.key) {
                if let Some(last_version) = history.latest_version() {
                    let time_since_last = Utc::now() - last_version.created_at;
                    let min_interval = chrono::Duration::minutes(self.config.min_version_interval_minutes as i64);
                    if time_since_last < min_interval {
                        // Skip creating a new version if too recent
                        return Ok(last_version.id);
                    }
                }
            }
        }

        // Get or create version counter for this memory
        let version_number = self.version_counters
            .entry(memory.key.clone())
            .and_modify(|counter| *counter += 1)
            .or_insert(1);

        // Create the new version
        let version = MemoryVersion::new(memory.clone(), change_type.clone(), *version_number);
        let version_id = version.id;

        // Add to history
        let history = self.histories
            .entry(memory.key.clone())
            .or_insert_with(|| VersionHistory::new(memory.key.clone()));
        
        history.add_version(version);

        // Cleanup old versions if necessary
        self.cleanup_versions_for_memory(&memory.key).await?;

        Ok(version_id)
    }

    /// Get the version history for a memory
    pub async fn get_history(&self, memory_key: &str) -> Result<VersionHistory> {
        self.histories
            .get(memory_key)
            .cloned()
            .ok_or_else(|| MemoryError::NotFound {
                key: memory_key.to_string(),
            })
    }

    /// Get the previous version of a memory
    pub async fn get_previous_version(&self, memory_key: &str) -> Result<Option<MemoryVersion>> {
        if let Some(history) = self.histories.get(memory_key) {
            if history.versions.len() >= 2 {
                return Ok(Some(history.versions[history.versions.len() - 2].clone()));
            }
        }
        Ok(None)
    }

    /// Get version history within a time range for a specific memory
    pub async fn get_history_in_range(
        &self,
        memory_key: &str,
        time_range: &TimeRange,
    ) -> Result<Vec<MemoryVersion>> {
        if let Some(history) = self.histories.get(memory_key) {
            Ok(history.get_versions_in_range(time_range)
                .into_iter()
                .cloned()
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Get all version history within a time range
    pub async fn get_all_history_in_range(
        &self,
        time_range: &TimeRange,
    ) -> Result<Vec<MemoryVersion>> {
        let mut all_versions = Vec::new();
        
        for history in self.histories.values() {
            all_versions.extend(
                history.get_versions_in_range(time_range)
                    .into_iter()
                    .cloned()
            );
        }
        
        // Sort by creation time
        all_versions.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        
        Ok(all_versions)
    }

    /// Attach differential data to a version
    pub async fn attach_diff(
        &mut self,
        version_id: Uuid,
        diff: super::differential::MemoryDiff,
    ) -> Result<()> {
        for history in self.histories.values_mut() {
            for version in &mut history.versions {
                if version.id == version_id {
                    version.diff_data = Some(diff);
                    return Ok(());
                }
            }
        }
        
        Err(MemoryError::NotFound {
            key: format!("version_{}", version_id),
        })
    }

    /// Get memories that changed within a time range
    pub async fn get_changed_memories_in_range(
        &self,
        time_range: &TimeRange,
    ) -> Result<Vec<String>> {
        let mut changed_memories = Vec::new();
        
        for (memory_key, history) in &self.histories {
            if !history.get_versions_in_range(time_range).is_empty() {
                changed_memories.push(memory_key.clone());
            }
        }
        
        Ok(changed_memories)
    }

    /// Get the most active memories (most frequently changed)
    pub async fn get_most_active_memories(&self, limit: usize) -> Result<Vec<(String, usize)>> {
        let mut activity_scores: Vec<(String, usize)> = self.histories
            .iter()
            .map(|(key, history)| (key.clone(), history.total_changes))
            .collect();
        
        activity_scores.sort_by(|a, b| b.1.cmp(&a.1));
        activity_scores.truncate(limit);
        
        Ok(activity_scores)
    }

    /// Cleanup old versions based on configuration
    pub async fn cleanup_versions_before(&mut self, cutoff_date: DateTime<Utc>) -> Result<usize> {
        let mut total_removed = 0;
        
        for history in self.histories.values_mut() {
            let original_count = history.versions.len();
            history.versions.retain(|v| v.created_at >= cutoff_date);
            total_removed += original_count - history.versions.len();
            
            // Update history metadata
            if !history.versions.is_empty() {
                history.total_changes = history.versions.len();
                history.first_version_at = history.versions.first().map(|v| v.created_at);
                history.last_version_at = history.versions.last().map(|v| v.created_at);
                history.update_most_common_change();
            }
        }
        
        Ok(total_removed)
    }

    /// Cleanup versions for a specific memory based on limits
    async fn cleanup_versions_for_memory(&mut self, memory_key: &str) -> Result<()> {
        if let Some(history) = self.histories.get_mut(memory_key) {
            // Remove excess versions if over the limit
            if history.versions.len() > self.config.max_versions_per_memory {
                let excess = history.versions.len() - self.config.max_versions_per_memory;
                history.versions.drain(0..excess);
                history.total_changes = history.versions.len();
                
                if !history.versions.is_empty() {
                    history.first_version_at = history.versions.first().map(|v| v.created_at);
                    history.update_most_common_change();
                }
            }
        }
        
        Ok(())
    }
}
