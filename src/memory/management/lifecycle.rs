//! Memory lifecycle management

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use lz4_flex;

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

/// Report from automated lifecycle management run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManagementReport {
    /// Total number of memories evaluated
    pub total_memories_evaluated: usize,
    /// Number of memories archived
    pub memories_archived: usize,
    /// Number of memories deleted
    pub memories_deleted: usize,
    /// Number of memories compressed
    pub memories_compressed: usize,
    /// Number of policies triggered
    pub policies_triggered: usize,
    /// Number of warnings added
    pub warnings_added: usize,
    /// Errors encountered during processing
    pub errors: Vec<String>,
    /// Duration of the management run in milliseconds
    pub duration_ms: u64,
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
    pub async fn track_memory_creation(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<()> {
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
        self.evaluate_policies(storage, &memory.key).await?;

        Ok(())
    }

    /// Track memory update
    pub async fn track_memory_update(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<()> {
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
            self.evaluate_policies(storage, &memory.key).await?;
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

    /// Evaluate policies for a specific memory using comprehensive lifecycle management
    /// Implements automated archival, retention policies, and cleanup with sophisticated logic
    async fn evaluate_policies(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
    ) -> Result<()> {
        tracing::debug!("Evaluating lifecycle policies for memory: {}", memory_key);
        let start_time = std::time::Instant::now();

        // Get current memory state
        let memory_state = match self.memory_states.get(memory_key) {
            Some(state) => state.clone(),
            None => return Ok(()), // No state to evaluate
        };

        // Get active policies sorted by priority (clone to avoid borrowing issues)
        let active_policies: Vec<LifecyclePolicy> = self.get_active_policies().into_iter().cloned().collect();
        let mut actions_to_execute = Vec::new();

        for policy in active_policies {
            tracing::debug!("Evaluating policy '{}' for memory '{}'", policy.name, memory_key);

            // Check if all conditions are met
            let conditions_met = self.evaluate_policy_conditions(storage, &policy, memory_key, &memory_state).await?;

            if conditions_met {
                tracing::info!("Policy '{}' triggered for memory '{}'", policy.name, memory_key);

                // Record policy trigger event
                let event = LifecycleEvent {
                    id: Uuid::new_v4(),
                    timestamp: Utc::now(),
                    memory_key: memory_key.to_string(),
                    event_type: LifecycleEventType::PolicyTriggered,
                    description: format!("Policy '{}' triggered", policy.name),
                    triggered_by_policy: Some(policy.id.clone()),
                };
                self.events.push(event);

                // Add actions to execution queue
                for action in &policy.actions {
                    actions_to_execute.push((action.clone(), policy.id.clone()));
                }

                // Break after first matching policy (highest priority wins)
                break;
            }
        }

        // Execute actions
        for (action, policy_id) in actions_to_execute {
            self.execute_lifecycle_action(storage, memory_key, &action, &policy_id).await?;
        }

        let duration = start_time.elapsed();
        tracing::debug!("Policy evaluation completed for memory '{}' in {:?}", memory_key, duration);

        Ok(())
    }

    /// Evaluate conditions for a specific policy against a memory
    async fn evaluate_policy_conditions(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        policy: &LifecyclePolicy,
        memory_key: &str,
        memory_state: &MemoryLifecycleState,
    ) -> Result<bool> {
        for condition in &policy.conditions {
            let condition_met = self.evaluate_single_condition(storage, condition, memory_key, memory_state).await?;
            if !condition_met {
                tracing::debug!("Condition {:?} not met for memory '{}'", condition, memory_key);
                return Ok(false);
            }
        }

        tracing::debug!("All conditions met for policy '{}' on memory '{}'", policy.name, memory_key);
        Ok(true)
    }

    /// Evaluate a single condition against a memory
    async fn evaluate_single_condition(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        condition: &LifecycleCondition,
        memory_key: &str,
        memory_state: &MemoryLifecycleState,
    ) -> Result<bool> {
        let now = Utc::now();

        match condition {
            LifecycleCondition::AgeExceeds { days } => {
                let age_threshold = Duration::days(*days as i64);
                let memory_age = now - memory_state.last_updated;
                Ok(memory_age > age_threshold)
            }

            LifecycleCondition::NotAccessedFor { days } => {
                // Find last access event
                let last_access = self.events.iter()
                    .filter(|e| e.memory_key == memory_key && e.event_type == LifecycleEventType::Accessed)
                    .max_by_key(|e| e.timestamp);

                let last_access_time = last_access
                    .map(|e| e.timestamp)
                    .unwrap_or(memory_state.last_updated);

                let time_since_access = now - last_access_time;
                let threshold = Duration::days(*days as i64);
                Ok(time_since_access > threshold)
            }

            LifecycleCondition::ImportanceBelow { threshold } => {
                // Get the actual memory entry to check importance
                if let Some(memory) = storage.retrieve(memory_key).await? {
                    Ok(memory.metadata.importance < *threshold)
                } else {
                    // Memory not found, consider it as low importance
                    Ok(true)
                }
            }

            LifecycleCondition::SizeExceeds { bytes } => {
                // Get the actual memory entry to check size
                if let Some(memory) = storage.retrieve(memory_key).await? {
                    let memory_size = memory.value.len() +
                        memory.metadata.tags.iter().map(|t| t.len()).sum::<usize>() +
                        memory.key.len();
                    Ok(memory_size > *bytes)
                } else {
                    // Memory not found, consider it as not exceeding size
                    Ok(false)
                }
            }

            LifecycleCondition::HasTags { tags } => {
                // Get the actual memory entry to check tags
                if let Some(memory) = storage.retrieve(memory_key).await? {
                    let has_matching_tag = tags.iter().any(|tag| memory.metadata.tags.contains(tag));
                    Ok(has_matching_tag)
                } else {
                    // Memory not found, consider it as not having tags
                    Ok(false)
                }
            }

            LifecycleCondition::AccessCountBelow { count } => {
                let access_count = self.events.iter()
                    .filter(|e| e.memory_key == memory_key && e.event_type == LifecycleEventType::Accessed)
                    .count() as u64;
                Ok(access_count < *count)
            }

            LifecycleCondition::Custom { condition } => {
                // Custom condition evaluation would be implemented here
                tracing::debug!("Evaluating custom condition: {}", condition);
                Ok(false) // Default to false for custom conditions
            }
        }
    }

    /// Execute a lifecycle action on a memory
    async fn execute_lifecycle_action(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
        action: &LifecycleAction,
        policy_id: &str,
    ) -> Result<()> {
        tracing::info!("Executing lifecycle action {:?} on memory '{}' (policy: {})", action, memory_key, policy_id);

        match action {
            LifecycleAction::Archive => {
                self.archive_memory(memory_key).await?;
            }

            LifecycleAction::Delete => {
                self.delete_memory(memory_key).await?;
            }

            LifecycleAction::Compress => {
                self.compress_memory(storage, memory_key).await?;
            }

            LifecycleAction::MoveToLongTerm => {
                self.move_to_long_term_storage(storage, memory_key).await?;
            }

            LifecycleAction::ReduceImportance { factor } => {
                self.reduce_memory_importance(storage, memory_key, *factor).await?;
            }

            LifecycleAction::AddWarningTag { tag } => {
                self.add_warning_tag(memory_key, tag).await?;
            }

            LifecycleAction::Summarize => {
                self.summarize_memory(storage, memory_key).await?;
            }

            LifecycleAction::Custom { action } => {
                self.execute_custom_action(memory_key, action).await?;
            }
        }

        // Record action execution event
        let event = LifecycleEvent {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            memory_key: memory_key.to_string(),
            event_type: LifecycleEventType::ActionExecuted,
            description: format!("Executed action: {:?}", action),
            triggered_by_policy: Some(policy_id.to_string()),
        };
        self.events.push(event);

        Ok(())
    }

    /// Archive a memory (move to archived stage)
    async fn archive_memory(&mut self, memory_key: &str) -> Result<()> {
        if let Some(state) = self.memory_states.get_mut(memory_key) {
            state.stage = MemoryStage::Archived;
            state.last_updated = Utc::now();

            let event = LifecycleEvent {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                memory_key: memory_key.to_string(),
                event_type: LifecycleEventType::Archived,
                description: "Memory archived".to_string(),
                triggered_by_policy: None,
            };
            self.events.push(event);

            tracing::info!("Memory '{}' archived successfully", memory_key);
        }
        Ok(())
    }

    /// Delete a memory (move to deleted stage)
    async fn delete_memory(&mut self, memory_key: &str) -> Result<()> {
        if let Some(state) = self.memory_states.get_mut(memory_key) {
            state.stage = MemoryStage::Deleted;
            state.last_updated = Utc::now();

            let event = LifecycleEvent {
                id: Uuid::new_v4(),
                timestamp: Utc::now(),
                memory_key: memory_key.to_string(),
                event_type: LifecycleEventType::Deleted,
                description: "Memory deleted by lifecycle policy".to_string(),
                triggered_by_policy: None,
            };
            self.events.push(event);

            tracing::info!("Memory '{}' deleted by lifecycle policy", memory_key);
        }
        Ok(())
    }

    /// Compress a memory using LZ4 compression algorithm
    async fn compress_memory(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
    ) -> Result<()> {
        // Retrieve the memory from storage
        if let Some(mut memory) = storage.retrieve(memory_key).await? {
            let original_size = memory.value.len();

            // Compress the memory content using LZ4
            let compressed_data = lz4_flex::compress_prepend_size(memory.value.as_bytes());
            let compressed_size = compressed_data.len();

            // Update memory with compressed content (using hex encoding for simplicity)
            memory.value = format!("COMPRESSED:{}", hex::encode(&compressed_data));
            memory.metadata.tags.push("compressed".to_string());
            memory.metadata.last_accessed = chrono::Utc::now();

            // Store the compressed memory back
            storage.store(&memory).await?;

            // Update lifecycle state
            if let Some(state) = self.memory_states.get_mut(memory_key) {
                state.last_updated = chrono::Utc::now();
                state.warnings.push(format!("Compressed: {} -> {} bytes ({:.1}% reduction)",
                    original_size, compressed_size,
                    (1.0 - compressed_size as f64 / original_size as f64) * 100.0));
            }

            tracing::info!("Memory '{}' compressed: {} -> {} bytes ({:.1}% reduction)",
                memory_key, original_size, compressed_size,
                (1.0 - compressed_size as f64 / original_size as f64) * 100.0);
        } else {
            tracing::warn!("Memory '{}' not found for compression", memory_key);
        }

        Ok(())
    }

    /// Move memory to long-term storage with metadata updates
    async fn move_to_long_term_storage(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
    ) -> Result<()> {
        // Retrieve the memory from storage
        if let Some(mut memory) = storage.retrieve(memory_key).await? {
            // Mark memory as moved to long-term storage
            memory.metadata.tags.push("long_term_storage".to_string());
            memory.metadata.tags.push("archived".to_string());
            memory.metadata.last_accessed = chrono::Utc::now();

            // Reduce importance for long-term storage
            memory.metadata.importance *= 0.5;

            // Store the updated memory back
            storage.store(&memory).await?;

            // Update lifecycle state
            if let Some(state) = self.memory_states.get_mut(memory_key) {
                state.stage = MemoryStage::Archived;
                state.last_updated = chrono::Utc::now();
                state.warnings.push("Moved to long-term storage".to_string());
            }

            tracing::info!("Memory '{}' moved to long-term storage successfully", memory_key);
        } else {
            tracing::warn!("Memory '{}' not found for long-term storage move", memory_key);
        }

        Ok(())
    }

    /// Reduce memory importance by a factor with storage update
    async fn reduce_memory_importance(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
        factor: f64,
    ) -> Result<()> {
        // Retrieve the memory from storage
        if let Some(mut memory) = storage.retrieve(memory_key).await? {
            let original_importance = memory.metadata.importance;

            // Reduce importance by the given factor
            memory.metadata.importance *= factor;
            memory.metadata.importance = memory.metadata.importance.max(0.0).min(1.0); // Clamp to [0, 1]
            memory.metadata.last_accessed = chrono::Utc::now();

            // Add tag to indicate importance reduction
            memory.metadata.tags.push(format!("importance_reduced_{:.2}", factor));

            // Store the updated memory back
            storage.store(&memory).await?;

            // Update lifecycle state
            if let Some(state) = self.memory_states.get_mut(memory_key) {
                state.last_updated = chrono::Utc::now();
                state.warnings.push(format!("Importance reduced: {:.3} -> {:.3} (factor: {:.2})",
                    original_importance, memory.metadata.importance, factor));
            }

            tracing::info!("Memory '{}' importance reduced: {:.3} -> {:.3} (factor: {:.2})",
                memory_key, original_importance, memory.metadata.importance, factor);
        } else {
            tracing::warn!("Memory '{}' not found for importance reduction", memory_key);
        }

        Ok(())
    }

    /// Add a warning tag to memory
    async fn add_warning_tag(&mut self, memory_key: &str, tag: &str) -> Result<()> {
        if let Some(state) = self.memory_states.get_mut(memory_key) {
            state.warnings.push(format!("Warning: {}", tag));
            state.last_updated = Utc::now();

            tracing::info!("Warning tag '{}' added to memory '{}'", tag, memory_key);
        }
        Ok(())
    }

    /// Summarize a memory using intelligent content reduction
    async fn summarize_memory(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
    ) -> Result<()> {
        // Retrieve the memory from storage
        if let Some(mut memory) = storage.retrieve(memory_key).await? {
            let original_length = memory.value.len();

            // Create a simple but effective summary
            let summary = self.create_memory_summary(&memory.value);
            let summary_length = summary.len();

            // Update memory with summary
            memory.value = summary;
            memory.metadata.tags.push("summarized".to_string());
            memory.metadata.tags.push(format!("original_length_{}", original_length));
            memory.metadata.last_accessed = chrono::Utc::now();

            // Reduce importance slightly since it's now summarized
            memory.metadata.importance *= 0.9;

            // Store the summarized memory back
            storage.store(&memory).await?;

            // Update lifecycle state
            if let Some(state) = self.memory_states.get_mut(memory_key) {
                state.last_updated = chrono::Utc::now();
                state.warnings.push(format!("Summarized: {} -> {} chars ({:.1}% reduction)",
                    original_length, summary_length,
                    (1.0 - summary_length as f64 / original_length as f64) * 100.0));
            }

            tracing::info!("Memory '{}' summarized: {} -> {} chars ({:.1}% reduction)",
                memory_key, original_length, summary_length,
                (1.0 - summary_length as f64 / original_length as f64) * 100.0);
        } else {
            tracing::warn!("Memory '{}' not found for summarization", memory_key);
        }

        Ok(())
    }

    /// Create a summary of memory content using multiple strategies
    fn create_memory_summary(&self, content: &str) -> String {
        if content.len() <= 200 {
            return content.to_string(); // Already short enough
        }

        // Strategy 1: Extract first and last sentences
        let sentences: Vec<&str> = content.split('.').filter(|s| !s.trim().is_empty()).collect();
        if sentences.len() >= 2 {
            let first_sentence = sentences[0].trim();
            let last_sentence = sentences[sentences.len() - 1].trim();

            if first_sentence.len() + last_sentence.len() < content.len() / 2 {
                return format!("{}. ... {}", first_sentence, last_sentence);
            }
        }

        // Strategy 2: Extract key phrases (words longer than 4 characters)
        let key_words: Vec<&str> = content.split_whitespace()
            .filter(|word| word.len() > 4 && word.chars().all(|c| c.is_alphabetic()))
            .take(10)
            .collect();

        if !key_words.is_empty() {
            return format!("Key concepts: {}", key_words.join(", "));
        }

        // Strategy 3: Simple truncation with ellipsis
        let truncated = &content[..content.len().min(200)];
        format!("{}...", truncated)
    }

    /// Execute a custom action
    async fn execute_custom_action(&mut self, memory_key: &str, action: &str) -> Result<()> {
        // Custom action execution would be implemented here
        // This could involve calling external systems, APIs, etc.

        tracing::info!("Custom action '{}' executed on memory '{}' (placeholder implementation)", action, memory_key);
        Ok(())
    }

    /// Run automated lifecycle management for all memories
    pub async fn run_automated_lifecycle_management(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
    ) -> Result<LifecycleManagementReport> {
        tracing::info!("Starting automated lifecycle management run");
        let start_time = std::time::Instant::now();

        let mut report = LifecycleManagementReport {
            total_memories_evaluated: 0,
            memories_archived: 0,
            memories_deleted: 0,
            memories_compressed: 0,
            policies_triggered: 0,
            warnings_added: 0,
            errors: Vec::new(),
            duration_ms: 0,
        };

        // Get all memory keys to evaluate
        let memory_keys: Vec<String> = self.memory_states.keys().cloned().collect();
        report.total_memories_evaluated = memory_keys.len();

        for memory_key in memory_keys {
            match self.evaluate_policies(storage, &memory_key).await {
                Ok(()) => {
                    // Count actions taken (simplified)
                    let start_time_utc = Utc::now() - chrono::Duration::milliseconds(start_time.elapsed().as_millis() as i64);
                    let recent_events = self.events.iter()
                        .filter(|e| e.memory_key == memory_key && e.timestamp > start_time_utc)
                        .collect::<Vec<_>>();

                    for event in recent_events {
                        match event.event_type {
                            LifecycleEventType::PolicyTriggered => report.policies_triggered += 1,
                            LifecycleEventType::Archived => report.memories_archived += 1,
                            LifecycleEventType::Deleted => report.memories_deleted += 1,
                            LifecycleEventType::ActionExecuted => {
                                if event.description.contains("compress") {
                                    report.memories_compressed += 1;
                                }
                                if event.description.contains("Warning") {
                                    report.warnings_added += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    report.errors.push(format!("Error evaluating memory '{}': {}", memory_key, e));
                }
            }
        }

        report.duration_ms = start_time.elapsed().as_millis() as u64;

        tracing::info!(
            "Automated lifecycle management completed: {} memories evaluated, {} archived, {} deleted, {} compressed in {}ms",
            report.total_memories_evaluated, report.memories_archived, report.memories_deleted,
            report.memories_compressed, report.duration_ms
        );

        Ok(report)
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
