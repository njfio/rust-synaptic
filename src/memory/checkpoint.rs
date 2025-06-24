//! Checkpointing system for agent state management

use crate::error::{MemoryError, Result, MemoryErrorExt};
use crate::memory::state::AgentState;
use crate::memory::storage::Storage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Checkpoint manager for creating and restoring agent state snapshots
pub struct CheckpointManager {
    /// Storage backend for checkpoints
    storage: Arc<dyn Storage + Send + Sync>,
    /// Checkpoint metadata cache
    metadata_cache: Arc<RwLock<HashMap<Uuid, CheckpointMetadata>>>,
    /// Configuration
    config: CheckpointConfig,
}

/// Configuration for checkpoint management
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between automatic checkpoints (number of operations)
    pub auto_checkpoint_interval: usize,
    /// Maximum number of checkpoints to retain
    pub max_checkpoints: usize,
    /// Enable compression for checkpoint data
    pub enable_compression: bool,
    /// Checkpoint retention policy
    pub retention_policy: RetentionPolicy,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            auto_checkpoint_interval: 100,
            max_checkpoints: 50,
            enable_compression: true,
            retention_policy: RetentionPolicy::KeepRecent { count: 10 },
        }
    }
}

/// Checkpoint retention policies
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Keep the most recent N checkpoints
    KeepRecent { count: usize },
    /// Keep checkpoints newer than specified duration
    KeepByAge { max_age_hours: u64 },
    /// Keep checkpoints based on importance scoring
    KeepByImportance { max_count: usize },
    /// Custom retention logic
    Custom,
}

/// Metadata for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint identifier
    pub id: Uuid,
    /// When this checkpoint was created
    pub created_at: DateTime<Utc>,
    /// Session ID this checkpoint belongs to
    pub session_id: Uuid,
    /// State version at time of checkpoint
    pub state_version: u64,
    /// Size of the checkpoint data in bytes
    pub size_bytes: usize,
    /// Number of memories in the checkpoint
    pub memory_count: usize,
    /// Checkpoint importance score (0.0 to 1.0)
    pub importance: f64,
    /// Optional description or tags
    pub description: Option<String>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

impl CheckpointMetadata {
    pub fn new(session_id: Uuid, state_version: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            created_at: Utc::now(),
            session_id,
            state_version,
            size_bytes: 0,
            memory_count: 0,
            importance: 0.5,
            description: None,
            custom_fields: HashMap::new(),
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
}

/// A complete checkpoint containing state and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
    /// Serialized agent state
    pub state_data: Vec<u8>,
}

impl Checkpoint {
    pub fn new(state: &AgentState, metadata: CheckpointMetadata) -> Result<Self> {
        let state_data = bincode::serialize(state)
            .checkpoint_context("Failed to serialize agent state")?;

        Ok(Self {
            metadata,
            state_data,
        })
    }

    pub fn restore_state(&self) -> Result<AgentState> {
        bincode::deserialize(&self.state_data)
            .checkpoint_context("Failed to deserialize agent state")
    }

    pub fn size(&self) -> usize {
        self.state_data.len() + 
        bincode::serialized_size(&self.metadata).unwrap_or(0) as usize
    }
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(
        checkpoint_interval: usize,
        storage: Arc<dyn Storage + Send + Sync>,
    ) -> Self {
        let config = CheckpointConfig {
            auto_checkpoint_interval: checkpoint_interval,
            ..Default::default()
        };

        Self {
            storage,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create a new checkpoint manager with custom configuration
    pub fn with_config(
        storage: Arc<dyn Storage + Send + Sync>,
        config: CheckpointConfig,
    ) -> Self {
        Self {
            storage,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if a checkpoint should be created based on the current state
    pub fn should_checkpoint(&self, state: &AgentState) -> bool {
        // Simple heuristic: checkpoint every N operations (based on version)
        state.version() % self.config.auto_checkpoint_interval as u64 == 0
    }

    /// Create a checkpoint of the current agent state
    pub async fn create_checkpoint(&self, state: &AgentState) -> Result<Uuid> {
        let mut metadata = CheckpointMetadata::new(state.session_id(), state.version());
        metadata.memory_count = state.short_term_memory_count() + state.long_term_memory_count();

        let checkpoint = Checkpoint::new(state, metadata.clone())?;
        metadata.size_bytes = checkpoint.size();

        // Store the checkpoint
        let checkpoint_key = format!("checkpoint_{}", metadata.id);
        #[cfg(feature = "bincode")]
        let serialized_checkpoint = bincode::serialize(&checkpoint)
            .checkpoint_context("Failed to serialize checkpoint")?;

        #[cfg(not(feature = "bincode"))]
        let serialized_checkpoint = serde_json::to_vec(&checkpoint)
            .checkpoint_context("Failed to serialize checkpoint")?;

        // Create a memory entry for the checkpoint
        let checkpoint_entry = crate::memory::types::MemoryEntry::new(
            checkpoint_key,
            hex::encode(&serialized_checkpoint),
            crate::memory::types::MemoryType::LongTerm,
        );

        self.storage.store(&checkpoint_entry).await
            .checkpoint_context("Failed to store checkpoint")?;

        // Update metadata cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(metadata.id, metadata.clone());
        }

        // Apply retention policy
        self.apply_retention_policy().await?;

        Ok(metadata.id)
    }

    /// Restore agent state from a checkpoint
    pub async fn restore_checkpoint(&self, checkpoint_id: Uuid) -> Result<AgentState> {
        let checkpoint_key = format!("checkpoint_{}", checkpoint_id);
        
        let checkpoint_entry = self.storage.retrieve(&checkpoint_key).await
            .checkpoint_context("Failed to retrieve checkpoint")?
            .ok_or_else(|| MemoryError::NotFound {
                key: checkpoint_key,
            })?;

        let checkpoint_bytes = hex::decode(&checkpoint_entry.value)
            .checkpoint_context("Failed to decode checkpoint hex")?;
        #[cfg(feature = "bincode")]
        let checkpoint: Checkpoint = bincode::deserialize(&checkpoint_bytes)
            .checkpoint_context("Failed to deserialize checkpoint")?;

        #[cfg(not(feature = "bincode"))]
        let checkpoint: Checkpoint = serde_json::from_slice(&checkpoint_bytes)
            .checkpoint_context("Failed to deserialize checkpoint")?;

        checkpoint.restore_state()
    }

    /// List all available checkpoints for a session
    pub async fn list_checkpoints(&self, session_id: Option<Uuid>) -> Result<Vec<CheckpointMetadata>> {
        let cache = self.metadata_cache.read().await;
        let mut checkpoints: Vec<CheckpointMetadata> = cache.values().cloned().collect();

        // Filter by session ID if provided
        if let Some(session_id) = session_id {
            checkpoints.retain(|metadata| metadata.session_id == session_id);
        }

        // Sort by creation time (newest first)
        checkpoints.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(checkpoints)
    }

    /// Get checkpoint metadata by ID
    pub async fn get_checkpoint_metadata(&self, checkpoint_id: Uuid) -> Result<Option<CheckpointMetadata>> {
        let cache = self.metadata_cache.read().await;
        Ok(cache.get(&checkpoint_id).cloned())
    }

    /// Delete a specific checkpoint
    pub async fn delete_checkpoint(&self, checkpoint_id: Uuid) -> Result<bool> {
        let checkpoint_key = format!("checkpoint_{}", checkpoint_id);
        
        let deleted = self.storage.delete(&checkpoint_key).await
            .checkpoint_context("Failed to delete checkpoint")?;

        if deleted {
            let mut cache = self.metadata_cache.write().await;
            cache.remove(&checkpoint_id);
        }

        Ok(deleted)
    }

    /// Get the most recent checkpoint for a session
    pub async fn get_latest_checkpoint(&self, session_id: Uuid) -> Result<Option<CheckpointMetadata>> {
        let checkpoints = self.list_checkpoints(Some(session_id)).await?;
        Ok(checkpoints.into_iter().next())
    }

    /// Create a named checkpoint with custom metadata
    pub async fn create_named_checkpoint(
        &self,
        state: &AgentState,
        name: String,
        importance: Option<f64>,
    ) -> Result<Uuid> {
        let mut metadata = CheckpointMetadata::new(state.session_id(), state.version())
            .with_description(name);

        if let Some(importance) = importance {
            metadata = metadata.with_importance(importance);
        }

        metadata.memory_count = state.short_term_memory_count() + state.long_term_memory_count();

        let checkpoint = Checkpoint::new(state, metadata.clone())?;
        metadata.size_bytes = checkpoint.size();

        let checkpoint_key = format!("checkpoint_{}", metadata.id);
        let serialized_checkpoint = bincode::serialize(&checkpoint)
            .checkpoint_context("Failed to serialize named checkpoint")?;

        let checkpoint_entry = crate::memory::types::MemoryEntry::new(
            checkpoint_key,
            hex::encode(&serialized_checkpoint),
            crate::memory::types::MemoryType::LongTerm,
        );

        self.storage.store(&checkpoint_entry).await
            .checkpoint_context("Failed to store named checkpoint")?;

        let checkpoint_id = metadata.id;

        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(metadata.id, metadata);
        }

        Ok(checkpoint_id)
    }

    /// Apply the configured retention policy
    async fn apply_retention_policy(&self) -> Result<()> {
        match &self.config.retention_policy {
            RetentionPolicy::KeepRecent { count } => {
                self.apply_keep_recent_policy(*count).await
            }
            RetentionPolicy::KeepByAge { max_age_hours } => {
                self.apply_keep_by_age_policy(*max_age_hours).await
            }
            RetentionPolicy::KeepByImportance { max_count } => {
                self.apply_keep_by_importance_policy(*max_count).await
            }
            RetentionPolicy::Custom => {
                // Custom retention logic would be implemented by the user
                Ok(())
            }
        }
    }

    /// Keep only the most recent N checkpoints
    async fn apply_keep_recent_policy(&self, keep_count: usize) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        let mut checkpoints: Vec<CheckpointMetadata> = cache.values().cloned().collect();
        
        if checkpoints.len() <= keep_count {
            return Ok(());
        }

        // Sort by creation time (newest first)
        checkpoints.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Remove old checkpoints
        for checkpoint in checkpoints.iter().skip(keep_count) {
            let checkpoint_key = format!("checkpoint_{}", checkpoint.id);
            let _ = self.storage.delete(&checkpoint_key).await;
            cache.remove(&checkpoint.id);
        }

        Ok(())
    }

    /// Keep only checkpoints newer than the specified age
    async fn apply_keep_by_age_policy(&self, max_age_hours: u64) -> Result<()> {
        let cutoff_time = Utc::now() - chrono::Duration::hours(max_age_hours as i64);
        let mut cache = self.metadata_cache.write().await;
        
        let expired_ids: Vec<Uuid> = cache
            .values()
            .filter(|metadata| metadata.created_at < cutoff_time)
            .map(|metadata| metadata.id)
            .collect();

        for checkpoint_id in expired_ids {
            let checkpoint_key = format!("checkpoint_{}", checkpoint_id);
            let _ = self.storage.delete(&checkpoint_key).await;
            cache.remove(&checkpoint_id);
        }

        Ok(())
    }

    /// Keep checkpoints based on importance scoring
    async fn apply_keep_by_importance_policy(&self, max_count: usize) -> Result<()> {
        let mut cache = self.metadata_cache.write().await;
        let mut checkpoints: Vec<CheckpointMetadata> = cache.values().cloned().collect();
        
        if checkpoints.len() <= max_count {
            return Ok(());
        }

        // Sort by importance (highest first)
        checkpoints.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

        // Remove low-importance checkpoints
        for checkpoint in checkpoints.iter().skip(max_count) {
            let checkpoint_key = format!("checkpoint_{}", checkpoint.id);
            let _ = self.storage.delete(&checkpoint_key).await;
            cache.remove(&checkpoint.id);
        }

        Ok(())
    }
}
