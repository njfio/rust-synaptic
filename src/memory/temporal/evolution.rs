//! Memory evolution tracking and analysis

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::memory::temporal::ChangeType;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Tracks the evolution of memories over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvolution {
    /// Memory key being tracked
    pub memory_key: String,
    /// Evolution timeline
    pub timeline: Vec<EvolutionEvent>,
    /// Evolution metrics
    pub metrics: EvolutionMetrics,
    /// First recorded state
    pub initial_state: Option<MemorySnapshot>,
    /// Most recent state
    pub current_state: Option<MemorySnapshot>,
}

/// A snapshot of memory state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// When this snapshot was taken
    pub timestamp: DateTime<Utc>,
    /// Memory content at this time
    pub content: String,
    /// Memory metadata at this time
    pub metadata: MemoryMetadata,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Simplified memory metadata for snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub importance: f64,
    pub confidence: f64,
    pub tags: Vec<String>,
    pub access_count: u64,
}

/// An event in the evolution of a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    /// Unique event identifier
    pub id: Uuid,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Type of change
    pub change_type: ChangeType,
    /// Description of the change
    pub description: String,
    /// Impact score (0.0 to 1.0)
    pub impact_score: f64,
    /// Size of change in bytes
    pub change_size: i64,
    /// Context information
    pub context: HashMap<String, String>,
}

/// Metrics about memory evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Total number of changes
    pub total_changes: usize,
    /// Average time between changes
    pub avg_change_interval: Duration,
    /// Growth rate (content size change per day)
    pub growth_rate: f64,
    /// Stability score (0.0 = very unstable, 1.0 = very stable)
    pub stability_score: f64,
    /// Complexity score (based on content structure)
    pub complexity_score: f64,
    /// Most active period
    pub most_active_period: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Change frequency by type
    pub change_frequency: HashMap<ChangeType, usize>,
}

/// Tracks evolution for multiple memories
pub struct EvolutionTracker {
    /// Evolution data for each memory
    evolutions: HashMap<String, MemoryEvolution>,
    /// Global evolution metrics
    global_metrics: GlobalEvolutionMetrics,
    /// Configuration
    config: EvolutionConfig,
}

/// Global metrics across all memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalEvolutionMetrics {
    /// Total memories being tracked
    pub total_memories: usize,
    /// Total evolution events
    pub total_events: usize,
    /// Average evolution rate across all memories
    pub avg_evolution_rate: f64,
    /// Most evolved memory
    pub most_evolved_memory: Option<String>,
    /// Most stable memory
    pub most_stable_memory: Option<String>,
    /// System-wide growth rate
    pub system_growth_rate: f64,
}

/// Configuration for evolution tracking
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Maximum number of events to keep per memory
    pub max_events_per_memory: usize,
    /// Minimum change size to track (bytes)
    pub min_change_size: usize,
    /// Enable detailed content analysis
    pub enable_content_analysis: bool,
    /// Snapshot interval for periodic captures
    pub snapshot_interval_hours: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            max_events_per_memory: 100,
            min_change_size: 10,
            enable_content_analysis: true,
            snapshot_interval_hours: 24,
        }
    }
}

impl EvolutionTracker {
    /// Create a new evolution tracker
    pub fn new() -> Self {
        Self {
            evolutions: HashMap::new(),
            global_metrics: GlobalEvolutionMetrics {
                total_memories: 0,
                total_events: 0,
                avg_evolution_rate: 0.0,
                most_evolved_memory: None,
                most_stable_memory: None,
                system_growth_rate: 0.0,
            },
            config: EvolutionConfig::default(),
        }
    }

    /// Track a change to a memory
    pub async fn track_change(
        &mut self,
        memory_key: &str,
        memory: &MemoryEntry,
        change_type: ChangeType,
    ) -> Result<()> {
        // Get or create evolution for this memory
        let evolution = self.evolutions
            .entry(memory_key.to_string())
            .or_insert_with(|| MemoryEvolution {
                memory_key: memory_key.to_string(),
                timeline: Vec::new(),
                metrics: EvolutionMetrics::default(),
                initial_state: None,
                current_state: None,
            });

        // Create snapshot of current state
        let snapshot = MemorySnapshot {
            timestamp: Utc::now(),
            content: memory.value.clone(),
            metadata: MemoryMetadata {
                importance: memory.metadata.importance,
                confidence: memory.metadata.confidence,
                tags: memory.metadata.tags.clone(),
                access_count: memory.metadata.access_count,
            },
            size_bytes: memory.value.len(),
        };

        // Set initial state if this is the first change
        if evolution.initial_state.is_none() {
            evolution.initial_state = Some(snapshot.clone());
        }

        // Calculate change impact
        let impact_score = Self::calculate_impact_score_static(evolution, &snapshot, &change_type);
        
        // Calculate change size
        let change_size = if let Some(ref current) = evolution.current_state {
            snapshot.size_bytes as i64 - current.size_bytes as i64
        } else {
            snapshot.size_bytes as i64
        };

        // Create evolution event
        let event = EvolutionEvent {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            change_type: change_type.clone(),
            description: format!("Memory {} was {}", memory_key, change_type),
            impact_score,
            change_size,
            context: HashMap::new(),
        };

        // Add event to timeline
        evolution.timeline.push(event);

        // Update current state
        evolution.current_state = Some(snapshot);

        // Recalculate metrics
        self.recalculate_metrics(memory_key).await?;

        // Update global metrics
        self.update_global_metrics().await?;

        Ok(())
    }

    /// Calculate the impact score of a change
    fn calculate_impact_score_static(
        evolution: &MemoryEvolution,
        new_snapshot: &MemorySnapshot,
        change_type: &ChangeType,
    ) -> f64 {
        let mut impact = 0.0;

        // Base impact by change type
        impact += match change_type {
            ChangeType::Created => 1.0,
            ChangeType::Updated => 0.7,
            ChangeType::Deleted => 0.9,
            ChangeType::Merged => 0.8,
            ChangeType::Split => 0.8,
            ChangeType::Summarized => 0.6,
            _ => 0.3,
        };

        // Size change impact
        if let Some(ref current) = evolution.current_state {
            let size_change_ratio = (new_snapshot.size_bytes as f64 - current.size_bytes as f64).abs() 
                / current.size_bytes.max(1) as f64;
            impact += size_change_ratio * 0.3;
        }

        // Importance change impact
        if let Some(ref current) = evolution.current_state {
            let importance_change = (new_snapshot.metadata.importance - current.metadata.importance).abs();
            impact += importance_change * 0.2;
        }

        impact.min(1.0)
    }

    /// Recalculate metrics for a specific memory
    async fn recalculate_metrics(&mut self, memory_key: &str) -> Result<()> {
        if let Some(evolution) = self.evolutions.get_mut(memory_key) {
            let total_changes = evolution.timeline.len();
            
            // Calculate average change interval
            let avg_change_interval = if total_changes > 1 {
                let first_time = evolution.timeline.first().unwrap().timestamp;
                let last_time = evolution.timeline.last().unwrap().timestamp;
                let total_duration = last_time - first_time;
                Duration::milliseconds(total_duration.num_milliseconds() / (total_changes - 1) as i64)
            } else {
                Duration::zero()
            };

            // Calculate growth rate
            let growth_rate = if let (Some(ref initial), Some(ref current)) = 
                (&evolution.initial_state, &evolution.current_state) {
                let time_diff = (current.timestamp - initial.timestamp).num_days() as f64;
                if time_diff > 0.0 {
                    (current.size_bytes as f64 - initial.size_bytes as f64) / time_diff
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Calculate stability score (inverse of change frequency)
            let stability_score = if total_changes > 0 {
                let days_tracked = if let (Some(ref initial), Some(ref current)) = 
                    (&evolution.initial_state, &evolution.current_state) {
                    (current.timestamp - initial.timestamp).num_days().max(1) as f64
                } else {
                    1.0
                };
                (1.0 / (1.0 + total_changes as f64 / days_tracked)).min(1.0)
            } else {
                1.0
            };

            // Calculate complexity score (based on content length and structure)
            let complexity_score = if let Some(ref current) = evolution.current_state {
                let length_factor = (current.size_bytes as f64 / 1000.0).min(1.0);
                let tag_factor = (current.metadata.tags.len() as f64 / 10.0).min(1.0);
                (length_factor + tag_factor) / 2.0
            } else {
                0.0
            };

            // Count change frequency by type
            let mut change_frequency = HashMap::new();
            for event in &evolution.timeline {
                *change_frequency.entry(event.change_type.clone()).or_insert(0) += 1;
            }

            evolution.metrics = EvolutionMetrics {
                total_changes,
                avg_change_interval,
                growth_rate,
                stability_score,
                complexity_score,
                most_active_period: None, // TODO: Implement period detection
                change_frequency,
            };
        }

        Ok(())
    }

    /// Update global metrics
    async fn update_global_metrics(&mut self) -> Result<()> {
        let total_memories = self.evolutions.len();
        let total_events: usize = self.evolutions.values()
            .map(|e| e.timeline.len())
            .sum();

        let avg_evolution_rate = if total_memories > 0 {
            self.evolutions.values()
                .map(|e| e.metrics.growth_rate)
                .sum::<f64>() / total_memories as f64
        } else {
            0.0
        };

        // Find most evolved memory (highest number of changes)
        let most_evolved_memory = self.evolutions.iter()
            .max_by_key(|(_, e)| e.timeline.len())
            .map(|(key, _)| key.clone());

        // Find most stable memory (highest stability score)
        let most_stable_memory = self.evolutions.iter()
            .max_by(|(_, a), (_, b)| a.metrics.stability_score.partial_cmp(&b.metrics.stability_score).unwrap())
            .map(|(key, _)| key.clone());

        let system_growth_rate = avg_evolution_rate;

        self.global_metrics = GlobalEvolutionMetrics {
            total_memories,
            total_events,
            avg_evolution_rate,
            most_evolved_memory,
            most_stable_memory,
            system_growth_rate,
        };

        Ok(())
    }

    /// Get evolution data for a specific memory
    pub fn get_evolution(&self, memory_key: &str) -> Option<&MemoryEvolution> {
        self.evolutions.get(memory_key)
    }

    /// Get metrics for a specific memory
    pub async fn get_metrics(&self, memory_key: &str) -> Result<EvolutionMetrics> {
        self.evolutions.get(memory_key)
            .map(|e| e.metrics.clone())
            .ok_or_else(|| MemoryError::NotFound {
                key: memory_key.to_string(),
            })
    }

    /// Get global evolution metrics
    pub async fn get_global_metrics(&self) -> Result<GlobalEvolutionMetrics> {
        Ok(self.global_metrics.clone())
    }

    /// Get the most evolved memories
    pub fn get_most_evolved_memories(&self, limit: usize) -> Vec<(String, usize)> {
        let mut memories: Vec<_> = self.evolutions.iter()
            .map(|(key, evolution)| (key.clone(), evolution.timeline.len()))
            .collect();
        
        memories.sort_by(|a, b| b.1.cmp(&a.1));
        memories.truncate(limit);
        memories
    }

    /// Get the most stable memories
    pub fn get_most_stable_memories(&self, limit: usize) -> Vec<(String, f64)> {
        let mut memories: Vec<_> = self.evolutions.iter()
            .map(|(key, evolution)| (key.clone(), evolution.metrics.stability_score))
            .collect();
        
        memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        memories.truncate(limit);
        memories
    }
}

impl Default for EvolutionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EvolutionMetrics {
    fn default() -> Self {
        Self {
            total_changes: 0,
            avg_change_interval: Duration::zero(),
            growth_rate: 0.0,
            stability_score: 1.0,
            complexity_score: 0.0,
            most_active_period: None,
            change_frequency: HashMap::new(),
        }
    }
}
