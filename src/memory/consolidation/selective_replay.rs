//! Selective Replay Manager
//! 
//! Implements intelligent memory replay system with importance-based selection,
//! temporal distribution, and interference minimization.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance, ReplayEntry};
use chrono::{DateTime, Utc, Duration};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;

/// Replay strategy for memory selection
#[derive(Debug, Clone)]
pub enum ReplayStrategy {
    /// Importance-based selection (highest importance first)
    ImportanceBased,
    /// Temporal distribution (spread across time periods)
    TemporalDistribution,
    /// Interference minimization (avoid conflicting memories)
    InterferenceMinimization,
    /// Balanced approach combining multiple strategies
    Balanced,
}

/// Replay scheduling algorithm
#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    /// Fixed interval replay
    FixedInterval,
    /// Exponential backoff based on importance
    ExponentialBackoff,
    /// Adaptive scheduling based on performance
    Adaptive,
    /// Spaced repetition algorithm
    SpacedRepetition,
}

/// Replay performance metrics
#[derive(Debug, Clone)]
pub struct ReplayMetrics {
    /// Total replay operations performed
    pub total_replays: u64,
    /// Average replay effectiveness score
    pub avg_effectiveness: f64,
    /// Memory retention rate after replay
    pub retention_rate: f64,
    /// Interference reduction achieved
    pub interference_reduction: f64,
    /// Last metrics update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Priority queue entry for replay scheduling
#[derive(Debug, Clone)]
struct PriorityReplayEntry {
    entry: ReplayEntry,
    priority: f64,
    scheduled_time: DateTime<Utc>,
}

impl PartialEq for PriorityReplayEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PriorityReplayEntry {}

impl PartialOrd for PriorityReplayEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityReplayEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        use crate::error_handling::SafeCompare;
        // Higher priority first (reverse order for max-heap behavior)
        other.priority.safe_partial_cmp(&self.priority)
    }
}

/// Selective replay manager
#[derive(Debug)]
pub struct SelectiveReplayManager {
    /// Configuration
    config: ConsolidationConfig,
    /// Replay buffer with priority queue
    replay_buffer: BinaryHeap<PriorityReplayEntry>,
    /// Replay history for performance tracking
    replay_history: VecDeque<ReplayEntry>,
    /// Replay strategy
    _strategy: ReplayStrategy,
    /// Scheduling algorithm
    scheduling: SchedulingAlgorithm,
    /// Performance metrics
    metrics: ReplayMetrics,
    /// Temporal distribution tracker
    temporal_distribution: HashMap<String, Vec<DateTime<Utc>>>,
    /// Interference matrix for conflict detection
    interference_matrix: HashMap<String, HashMap<String, f64>>,
}

impl SelectiveReplayManager {
    /// Create a new selective replay manager
    pub fn new(config: &ConsolidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            replay_buffer: BinaryHeap::new(),
            replay_history: VecDeque::new(),
            _strategy: ReplayStrategy::Balanced,
            scheduling: SchedulingAlgorithm::Adaptive,
            metrics: ReplayMetrics {
                total_replays: 0,
                avg_effectiveness: 0.0,
                retention_rate: 0.0,
                interference_reduction: 0.0,
                last_updated: Utc::now(),
            },
            temporal_distribution: HashMap::new(),
            interference_matrix: HashMap::new(),
        })
    }

    /// Add memory to replay buffer with importance-based prioritization
    pub async fn add_to_buffer(&mut self, memory: &MemoryEntry, importance: &MemoryImportance) -> Result<()> {
        // Calculate replay priority based on multiple factors
        let replay_priority = self.calculate_replay_priority(memory, importance).await?;
        
        // Create replay entry
        let replay_entry = ReplayEntry {
            memory: memory.clone(),
            importance_score: importance.importance_score,
            replay_count: 0,
            last_replayed: Utc::now(),
            replay_priority,
        };

        // Calculate scheduled time based on importance and strategy
        let scheduled_time = self.calculate_scheduled_time(&replay_entry).await?;

        // Add to priority queue
        let priority_entry = PriorityReplayEntry {
            entry: replay_entry,
            priority: replay_priority,
            scheduled_time,
        };

        self.replay_buffer.push(priority_entry);

        // Maintain buffer size limit
        self.maintain_buffer_size().await?;

        tracing::debug!("Added memory '{}' to replay buffer with priority {:.3}", 
                       memory.key, replay_priority);

        Ok(())
    }

    /// Perform selective replay based on configured strategy
    pub async fn perform_selective_replay(&mut self) -> Result<()> {
        let batch_size = self.config.replay_batch_size;
        let mut replayed_count = 0;

        tracing::info!("Starting selective replay with batch size {}", batch_size);

        while replayed_count < batch_size && !self.replay_buffer.is_empty() {
            if let Some(priority_entry) = self.replay_buffer.pop() {
                // Check if it's time to replay this memory
                if Utc::now() >= priority_entry.scheduled_time {
                    self.replay_memory(&priority_entry.entry).await?;
                    replayed_count += 1;
                } else {
                    // Put it back if not ready for replay
                    self.replay_buffer.push(priority_entry);
                    break;
                }
            }
        }

        // Update performance metrics
        self.update_replay_metrics(replayed_count).await?;

        tracing::info!("Selective replay completed: {} memories replayed", replayed_count);

        Ok(())
    }

    /// Calculate replay priority using sophisticated algorithms
    async fn calculate_replay_priority(&self, memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        let mut priority_factors = Vec::new();

        // 1. Base importance score
        priority_factors.push(importance.importance_score);

        // 2. Temporal urgency (memories not accessed recently get higher priority)
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours().max(0) as f64;
        let temporal_urgency = (hours_since_access / 168.0).min(1.0); // Normalize to week
        priority_factors.push(temporal_urgency);

        // 3. Forgetting curve factor (Ebbinghaus curve approximation)
        let forgetting_factor = self.calculate_forgetting_curve_factor(memory).await?;
        priority_factors.push(forgetting_factor);

        // 4. Interference risk (memories that might interfere with new learning)
        let interference_risk = self.calculate_interference_risk(&memory.key).await?;
        priority_factors.push(interference_risk);

        // 5. Strategic importance (based on memory type and content)
        let strategic_importance = self.calculate_strategic_importance(memory).await?;
        priority_factors.push(strategic_importance);

        // Weighted combination of priority factors
        let weights = [0.3, 0.2, 0.2, 0.15, 0.15]; // Sum = 1.0
        let weighted_priority: f64 = priority_factors.iter()
            .zip(weights.iter())
            .map(|(factor, weight)| factor * weight)
            .sum();

        Ok(weighted_priority.min(1.0).max(0.0))
    }

    /// Calculate scheduled time for replay based on importance and algorithm
    async fn calculate_scheduled_time(&self, replay_entry: &ReplayEntry) -> Result<DateTime<Utc>> {
        let base_time = Utc::now();

        let delay_hours = match self.scheduling {
            SchedulingAlgorithm::FixedInterval => 24, // Fixed 24-hour interval
            SchedulingAlgorithm::ExponentialBackoff => {
                // Higher importance = shorter delay
                let importance_factor = 1.0 - replay_entry.importance_score;
                (24.0 * (1.0 + importance_factor * 3.0)) as i64 // 24-96 hours
            },
            SchedulingAlgorithm::Adaptive => {
                // Adaptive based on replay history and performance
                self.calculate_adaptive_delay(replay_entry).await? as i64
            },
            SchedulingAlgorithm::SpacedRepetition => {
                // Spaced repetition algorithm (SM-2 inspired)
                self.calculate_spaced_repetition_delay(replay_entry).await? as i64
            },
        };

        Ok(base_time + Duration::hours(delay_hours))
    }

    /// Replay a specific memory entry
    async fn replay_memory(&mut self, replay_entry: &ReplayEntry) -> Result<()> {
        // Simulate memory replay process
        // In a real implementation, this would involve:
        // 1. Retrieving the memory from storage
        // 2. Reinforcing neural pathways
        // 3. Updating memory strength
        // 4. Recording replay event

        // Add small delay to simulate processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

        let mut updated_entry = replay_entry.clone();
        updated_entry.replay_count += 1;
        updated_entry.last_replayed = Utc::now();

        // Update replay priority based on performance
        updated_entry.replay_priority = self.update_replay_priority(&updated_entry).await?;

        // Add to replay history
        self.replay_history.push_back(updated_entry.clone());

        // Maintain history size
        if self.replay_history.len() > 1000 {
            self.replay_history.pop_front();
        }

        // Update temporal distribution tracking
        self.update_temporal_distribution(&replay_entry.memory.key).await?;

        // Update interference matrix
        self.update_interference_matrix(&replay_entry.memory.key).await?;

        self.metrics.total_replays += 1;

        tracing::debug!("Replayed memory '{}' (count: {})", 
                       replay_entry.memory.key, updated_entry.replay_count);

        Ok(())
    }

    /// Calculate forgetting curve factor based on Ebbinghaus curve
    async fn calculate_forgetting_curve_factor(&self, memory: &MemoryEntry) -> Result<f64> {
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours().max(0) as f64;
        
        // Ebbinghaus forgetting curve: R = e^(-t/S)
        // Where R = retention, t = time, S = strength of memory
        let memory_strength = memory.metadata.importance * 100.0; // Scale importance to hours
        let retention = (-hours_since_access / memory_strength).exp();
        
        // Return forgetting factor (1 - retention)
        Ok((1.0 - retention).min(1.0).max(0.0))
    }

    /// Calculate interference risk for a memory
    async fn calculate_interference_risk(&self, memory_key: &str) -> Result<f64> {
        if let Some(interference_map) = self.interference_matrix.get(memory_key) {
            // Calculate average interference with other memories
            let avg_interference = if interference_map.is_empty() {
                0.0
            } else {
                interference_map.values().sum::<f64>() / interference_map.len() as f64
            };
            Ok(avg_interference)
        } else {
            Ok(0.0) // No interference data available
        }
    }

    /// Calculate strategic importance based on memory characteristics
    async fn calculate_strategic_importance(&self, memory: &MemoryEntry) -> Result<f64> {
        let mut importance_factors = Vec::new();

        // Memory type importance
        let type_importance = match memory.memory_type {
            crate::memory::types::MemoryType::LongTerm => 0.8,
            crate::memory::types::MemoryType::ShortTerm => 0.4,
        };
        importance_factors.push(type_importance);

        // Content length factor (moderate length preferred)
        let content_length = memory.value.len() as f64;
        let length_factor = if content_length > 100.0 && content_length < 1000.0 {
            0.8
        } else {
            0.5
        };
        importance_factors.push(length_factor);

        // Tag-based importance
        let tag_importance = if memory.metadata.tags.is_empty() {
            0.3
        } else {
            (memory.metadata.tags.len() as f64 / 10.0).min(1.0)
        };
        importance_factors.push(tag_importance);

        // Calculate average importance
        let avg_importance = importance_factors.iter().sum::<f64>() / importance_factors.len() as f64;
        Ok(avg_importance)
    }

    /// Calculate adaptive delay based on performance history
    async fn calculate_adaptive_delay(&self, _replay_entry: &ReplayEntry) -> Result<f64> {
        // Simplified adaptive algorithm
        // In practice, this would analyze replay effectiveness and adjust delays
        let base_delay = 24.0; // 24 hours
        let performance_factor = self.metrics.avg_effectiveness;
        
        // Better performance = longer delays (memory is stable)
        let adaptive_delay = base_delay * (1.0 + performance_factor);
        Ok(adaptive_delay)
    }

    /// Calculate spaced repetition delay using SM-2 inspired algorithm
    async fn calculate_spaced_repetition_delay(&self, replay_entry: &ReplayEntry) -> Result<f64> {
        let replay_count = replay_entry.replay_count as f64;
        let importance = replay_entry.importance_score;
        
        // SM-2 inspired intervals: 1, 6, 24, 72, 168 hours...
        let base_intervals = [1.0, 6.0, 24.0, 72.0, 168.0];
        let interval_index = (replay_count as usize).min(base_intervals.len() - 1);
        let base_interval = base_intervals[interval_index];
        
        // Adjust based on importance (higher importance = more frequent replay)
        let importance_factor = 2.0 - importance; // 1.0 to 2.0 range
        let adjusted_interval = base_interval * importance_factor;
        
        Ok(adjusted_interval)
    }

    /// Update replay priority based on performance
    async fn update_replay_priority(&self, replay_entry: &ReplayEntry) -> Result<f64> {
        let base_priority = replay_entry.replay_priority;
        let replay_count = replay_entry.replay_count as f64;
        
        // Decrease priority with successful replays (diminishing returns)
        let decay_factor = 1.0 / (1.0 + replay_count * 0.1);
        let updated_priority = base_priority * decay_factor;
        
        Ok(updated_priority.max(0.1)) // Minimum priority threshold
    }

    /// Maintain replay buffer size within configured limits
    async fn maintain_buffer_size(&mut self) -> Result<()> {
        while self.replay_buffer.len() > self.config.max_replay_buffer_size {
            // Remove lowest priority entries
            if let Some(_) = self.replay_buffer.pop() {
                // Entry removed
            }
        }
        Ok(())
    }

    /// Update replay performance metrics
    async fn update_replay_metrics(&mut self, replayed_count: usize) -> Result<()> {
        // Calculate effectiveness based on replay success
        let effectiveness = if replayed_count > 0 {
            replayed_count as f64 / self.config.replay_batch_size as f64
        } else {
            0.0
        };

        // Update running average
        let total_ops = self.metrics.total_replays as f64;
        if total_ops > 0.0 {
            self.metrics.avg_effectiveness = 
                (self.metrics.avg_effectiveness * (total_ops - 1.0) + effectiveness) / total_ops;
        } else {
            self.metrics.avg_effectiveness = effectiveness;
        }

        self.metrics.last_updated = Utc::now();
        Ok(())
    }

    /// Update temporal distribution tracking
    async fn update_temporal_distribution(&mut self, memory_key: &str) -> Result<()> {
        let now = Utc::now();
        self.temporal_distribution
            .entry(memory_key.to_string())
            .or_insert_with(Vec::new)
            .push(now);
        Ok(())
    }

    /// Update interference matrix
    async fn update_interference_matrix(&mut self, memory_key: &str) -> Result<()> {
        // Simplified interference calculation
        // In practice, this would analyze semantic similarity and temporal proximity
        use rand::Rng;
        let interference_score = rand::thread_rng().gen::<f64>() * 0.5; // Random interference for simulation

        self.interference_matrix
            .entry(memory_key.to_string())
            .or_insert_with(HashMap::new)
            .insert("global".to_string(), interference_score);

        Ok(())
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// Get replay metrics
    pub fn get_metrics(&self) -> &ReplayMetrics {
        &self.metrics
    }

    /// Get replay history
    pub fn get_replay_history(&self, limit: usize) -> Vec<&ReplayEntry> {
        self.replay_history.iter().rev().take(limit).collect()
    }

    /// Force immediate replay for testing purposes
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn force_immediate_replay(&mut self) -> Result<()> {
        let batch_size = self.config.replay_batch_size;
        let mut replayed_count = 0;

        tracing::info!("Starting forced immediate replay with batch size {}", batch_size);

        while replayed_count < batch_size && !self.replay_buffer.is_empty() {
            if let Some(priority_entry) = self.replay_buffer.pop() {
                // Force replay regardless of scheduled time
                self.replay_memory(&priority_entry.entry).await?;
                replayed_count += 1;
            }
        }

        // Update performance metrics
        self.update_replay_metrics(replayed_count).await?;

        tracing::info!("Forced immediate replay completed: {} memories replayed", replayed_count);

        Ok(())
    }

    /// Make all scheduled replays immediate for testing purposes
    pub fn make_all_replays_immediate(&mut self) {
        let mut temp_buffer = Vec::new();
        while let Some(mut priority_entry) = self.replay_buffer.pop() {
            priority_entry.scheduled_time = Utc::now(); // Schedule immediately for testing
            temp_buffer.push(priority_entry);
        }
        for entry in temp_buffer {
            self.replay_buffer.push(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_selective_replay_manager_creation() {
        let config = ConsolidationConfig::default();
        let manager = SelectiveReplayManager::new(&config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_add_to_buffer() {
        let config = ConsolidationConfig::default();
        let mut manager = SelectiveReplayManager::new(&config).unwrap();

        let memory = MemoryEntry::new(
            "test_key".to_string(),
            "Test content".to_string(),
            MemoryType::LongTerm
        );

        let importance = MemoryImportance {
            memory_key: "test_key".to_string(),
            importance_score: 0.8,
            access_frequency: 0.7,
            recency_score: 0.6,
            centrality_score: 0.5,
            uniqueness_score: 0.4,
            temporal_consistency: 0.3,
            calculated_at: Utc::now(),
            fisher_information: None,
        };

        let result = manager.add_to_buffer(&memory, &importance).await;
        assert!(result.is_ok());
        assert_eq!(manager.buffer_size(), 1);
    }

    #[tokio::test]
    async fn test_selective_replay() {
        let config = ConsolidationConfig::default();
        let mut manager = SelectiveReplayManager::new(&config).unwrap();

        // Add some memories to buffer with immediate scheduling for testing
        for i in 0..5 {
            let memory = MemoryEntry::new(
                format!("key_{}", i),
                format!("Content {}", i),
                MemoryType::LongTerm
            );

            let importance = MemoryImportance {
                memory_key: format!("key_{}", i),
                importance_score: 0.5 + (i as f64 * 0.1),
                access_frequency: 0.5,
                recency_score: 0.5,
                centrality_score: 0.5,
                uniqueness_score: 0.5,
                temporal_consistency: 0.5,
                calculated_at: Utc::now(),
                fisher_information: None,
            };

            // Create replay entry with immediate scheduling for testing
            let replay_entry = ReplayEntry {
                memory: memory.clone(),
                importance_score: importance.importance_score,
                replay_count: 0,
                last_replayed: Utc::now(),
                replay_priority: 0.8,
            };

            // Add to priority queue with immediate scheduling
            let priority_entry = PriorityReplayEntry {
                entry: replay_entry,
                priority: 0.8,
                scheduled_time: Utc::now(), // Schedule immediately for testing
            };

            manager.replay_buffer.push(priority_entry);
        }

        let result = manager.perform_selective_replay().await;
        assert!(result.is_ok());
        assert!(manager.get_metrics().total_replays > 0);
        assert!(manager.get_replay_history(10).len() > 0);
    }
}
