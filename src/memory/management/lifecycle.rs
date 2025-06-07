//! Memory lifecycle management

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Manages the lifecycle of memories from creation to archival
pub struct MemoryLifecycleManager {
    /// Lifecycle policies
    policies: Vec<LifecyclePolicy>,
    /// Memory lifecycle states
    memory_states: HashMap<String, MemoryLifecycleState>,
    /// Lifecycle events
    events: Vec<LifecycleEvent>,
}

/// A policy that governs memory lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecyclePolicy {
    /// Policy identifier
    pub id: String,
    /// Policy name
    pub name: String,
    /// Conditions that trigger this policy
    pub conditions: Vec<LifecycleCondition>,
    /// Actions to take when conditions are met
    pub actions: Vec<LifecycleAction>,
    /// Whether this policy is active
    pub active: bool,
    /// Priority (higher numbers = higher priority)
    pub priority: u32,
}

/// Conditions that can trigger lifecycle actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCondition {
    /// Memory age exceeds threshold
    AgeExceeds { days: u64 },
    /// Memory hasn't been accessed for a period
    NotAccessedFor { days: u64 },
    /// Memory importance below threshold
    ImportanceBelow { threshold: f64 },
    /// Memory size exceeds threshold
    SizeExceeds { bytes: usize },
    /// Memory has specific tags
    HasTags { tags: Vec<String> },
    /// Memory access count below threshold
    AccessCountBelow { count: u64 },
    /// Custom condition
    Custom { condition: String },
}

/// Actions that can be taken in the lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleAction {
    /// Archive the memory
    Archive,
    /// Delete the memory
    Delete,
    /// Compress the memory
    Compress,
    /// Move to long-term storage
    MoveToLongTerm,
    /// Reduce importance
    ReduceImportance { factor: f64 },
    /// Add warning tag
    AddWarningTag { tag: String },
    /// Summarize the memory
    Summarize,
    /// Custom action
    Custom { action: String },
}

/// Current lifecycle state of a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLifecycleState {
    /// Memory key
    pub memory_key: String,
    /// Current stage in lifecycle
    pub stage: MemoryStage,
    /// When this state was last updated
    pub last_updated: DateTime<Utc>,
    /// Lifecycle events for this memory
    pub events: Vec<LifecycleEvent>,
    /// Warnings or notices
    pub warnings: Vec<String>,
    /// Next scheduled action
    pub next_action: Option<ScheduledAction>,
}

/// Stages in memory lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryStage {
    /// Newly created memory
    Created,
    /// Active memory being used
    Active,
    /// Memory showing signs of aging
    Aging,
    /// Memory marked for review
    UnderReview,
    /// Memory scheduled for archival
    ScheduledForArchival,
    /// Memory has been archived
    Archived,
    /// Memory scheduled for deletion
    ScheduledForDeletion,
    /// Memory has been deleted
    Deleted,
}

/// A scheduled action in the lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledAction {
    /// Action to be taken
    pub action: LifecycleAction,
    /// When the action should be executed
    pub scheduled_time: DateTime<Utc>,
    /// Reason for the action
    pub reason: String,
}

/// An event in the memory lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleEvent {
    /// Event identifier
    pub id: Uuid,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Memory key involved
    pub memory_key: String,
    /// Type of event
    pub event_type: LifecycleEventType,
    /// Event description
    pub description: String,
    /// Policy that triggered this event (if any)
    pub triggered_by_policy: Option<String>,
}

/// Types of lifecycle events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleEventType {
    Created,
    Accessed,
    Updated,
    StageChanged,
    PolicyTriggered,
    ActionExecuted,
    Archived,
    Deleted,
    Warning,
}

impl MemoryLifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        Self {
            policies: Self::create_default_policies(),
            memory_states: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Track memory creation
    pub async fn track_memory_creation(&mut self, memory: &MemoryEntry) -> Result<()> {
        let state = MemoryLifecycleState {
            memory_key: memory.key.clone(),
            stage: MemoryStage::Created,
            last_updated: Utc::now(),
            events: Vec::new(),
            warnings: Vec::new(),
            next_action: None,
        };

        self.memory_states.insert(memory.key.clone(), state);

        let event = LifecycleEvent {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            memory_key: memory.key.clone(),
            event_type: LifecycleEventType::Created,
            description: "Memory created".to_string(),
            triggered_by_policy: None,
        };

        self.events.push(event);
        self.evaluate_policies(&memory.key).await?;

        Ok(())
    }

    /// Track memory update
    pub async fn track_memory_update(&mut self, memory: &MemoryEntry) -> Result<()> {
        if let Some(state) = self.memory_states.get_mut(&memory.key) {
            state.last_updated = Utc::now();
            
            // Update stage if appropriate
            if state.stage == MemoryStage::Created {
                state.stage = MemoryStage::Active;
            }

            let event = LifecycleEvent {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                memory_key: memory.key.clone(),
                event_type: LifecycleEventType::Updated,
                description: "Memory updated".to_string(),
                triggered_by_policy: None,
            };

            self.events.push(event);
            self.evaluate_policies(&memory.key).await?;
        }

        Ok(())
    }

    /// Track memory deletion
    pub async fn track_memory_deletion(&mut self, memory_key: &str) -> Result<()> {
        if let Some(state) = self.memory_states.get_mut(memory_key) {
            state.stage = MemoryStage::Deleted;
            state.last_updated = Utc::now();

            let event = LifecycleEvent {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                memory_key: memory_key.to_string(),
                event_type: LifecycleEventType::Deleted,
                description: "Memory deleted".to_string(),
                triggered_by_policy: None,
            };

            self.events.push(event);
        }

        Ok(())
    }

    /// Evaluate policies for a specific memory
    async fn evaluate_policies(&mut self, memory_key: &str) -> Result<()> {
        // TODO: Implement policy evaluation logic
        // This would check each policy's conditions against the memory
        // and execute actions if conditions are met
        Ok(())
    }

    /// Create default lifecycle policies
    fn create_default_policies() -> Vec<LifecyclePolicy> {
        vec![
            LifecyclePolicy {
                id: "archive_old_memories".to_string(),
                name: "Archive Old Memories".to_string(),
                conditions: vec![
                    LifecycleCondition::AgeExceeds { days: 365 },
                    LifecycleCondition::NotAccessedFor { days: 90 },
                ],
                actions: vec![LifecycleAction::Archive],
                active: true,
                priority: 1,
            },
            LifecyclePolicy {
                id: "delete_low_importance".to_string(),
                name: "Delete Low Importance Memories".to_string(),
                conditions: vec![
                    LifecycleCondition::ImportanceBelow { threshold: 0.1 },
                    LifecycleCondition::NotAccessedFor { days: 180 },
                ],
                actions: vec![LifecycleAction::Delete],
                active: true,
                priority: 2,
            },
        ]
    }

    /// Get lifecycle state for a memory
    pub fn get_memory_state(&self, memory_key: &str) -> Option<&MemoryLifecycleState> {
        self.memory_states.get(memory_key)
    }

    /// Get all memories in a specific stage
    pub fn get_memories_in_stage(&self, stage: &MemoryStage) -> Vec<&MemoryLifecycleState> {
        self.memory_states.values()
            .filter(|state| &state.stage == stage)
            .collect()
    }

    /// Get lifecycle events for a memory
    pub fn get_memory_events(&self, memory_key: &str) -> Vec<&LifecycleEvent> {
        self.events.iter()
            .filter(|event| event.memory_key == memory_key)
            .collect()
    }

    /// Get recent lifecycle events
    pub fn get_recent_events(&self, limit: usize) -> Vec<&LifecycleEvent> {
        let mut events = self.events.iter().collect::<Vec<_>>();
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        events.truncate(limit);
        events
    }

    /// Add a custom lifecycle policy
    pub fn add_policy(&mut self, policy: LifecyclePolicy) {
        self.policies.push(policy);
        // Sort by priority (highest first)
        self.policies.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a lifecycle policy
    pub fn remove_policy(&mut self, policy_id: &str) -> bool {
        if let Some(pos) = self.policies.iter().position(|p| p.id == policy_id) {
            self.policies.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get all active policies
    pub fn get_active_policies(&self) -> Vec<&LifecyclePolicy> {
        self.policies.iter().filter(|p| p.active).collect()
    }
}

impl Default for MemoryLifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}
