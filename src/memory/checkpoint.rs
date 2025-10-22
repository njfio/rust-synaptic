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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.auto_checkpoint_interval, 100);
        assert_eq!(config.max_checkpoints, 50);
        assert!(config.enable_compression);
        match config.retention_policy {
            RetentionPolicy::KeepRecent { count } => assert_eq!(count, 10),
            _ => panic!("Expected KeepRecent retention policy"),
        }
    }

    #[test]
    fn test_checkpoint_config_clone() {
        let config1 = CheckpointConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.auto_checkpoint_interval, config2.auto_checkpoint_interval);
        assert_eq!(config1.max_checkpoints, config2.max_checkpoints);
        assert_eq!(config1.enable_compression, config2.enable_compression);
    }

    #[test]
    fn test_checkpoint_metadata_new() {
        let session_id = Uuid::new_v4();
        let state_version = 42;
        let metadata = CheckpointMetadata::new(session_id, state_version);

        assert_eq!(metadata.session_id, session_id);
        assert_eq!(metadata.state_version, state_version);
        assert_eq!(metadata.size_bytes, 0);
        assert_eq!(metadata.memory_count, 0);
        assert_eq!(metadata.importance, 0.5);
        assert!(metadata.description.is_none());
        assert!(metadata.custom_fields.is_empty());
    }

    #[test]
    fn test_checkpoint_metadata_with_description() {
        let session_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(session_id, 1)
            .with_description("Test checkpoint".to_string());

        assert_eq!(metadata.description, Some("Test checkpoint".to_string()));
    }

    #[test]
    fn test_checkpoint_metadata_with_importance() {
        let session_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(session_id, 1)
            .with_importance(0.8);

        assert_eq!(metadata.importance, 0.8);
    }

    #[test]
    fn test_checkpoint_metadata_with_importance_clamping() {
        let session_id = Uuid::new_v4();

        // Test upper bound clamping
        let metadata1 = CheckpointMetadata::new(session_id, 1)
            .with_importance(1.5);
        assert_eq!(metadata1.importance, 1.0);

        // Test lower bound clamping
        let metadata2 = CheckpointMetadata::new(session_id, 1)
            .with_importance(-0.5);
        assert_eq!(metadata2.importance, 0.0);
    }

    #[test]
    fn test_checkpoint_metadata_builder_chaining() {
        let session_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(session_id, 10)
            .with_description("Important checkpoint".to_string())
            .with_importance(0.9);

        assert_eq!(metadata.description, Some("Important checkpoint".to_string()));
        assert_eq!(metadata.importance, 0.9);
        assert_eq!(metadata.state_version, 10);
    }

    #[test]
    fn test_checkpoint_metadata_clone() {
        let session_id = Uuid::new_v4();
        let metadata1 = CheckpointMetadata::new(session_id, 5)
            .with_description("Test".to_string());

        let metadata2 = metadata1.clone();
        assert_eq!(metadata1.id, metadata2.id);
        assert_eq!(metadata1.session_id, metadata2.session_id);
        assert_eq!(metadata1.state_version, metadata2.state_version);
        assert_eq!(metadata1.description, metadata2.description);
    }

    #[test]
    fn test_checkpoint_metadata_serialization() {
        let session_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(session_id, 100)
            .with_description("Serialization test".to_string())
            .with_importance(0.75);

        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: CheckpointMetadata = serde_json::from_str(&serialized).unwrap();

        assert_eq!(metadata.id, deserialized.id);
        assert_eq!(metadata.session_id, deserialized.session_id);
        assert_eq!(metadata.state_version, deserialized.state_version);
        assert_eq!(metadata.importance, deserialized.importance);
        assert_eq!(metadata.description, deserialized.description);
    }

    #[test]
    fn test_retention_policy_keep_recent() {
        let policy = RetentionPolicy::KeepRecent { count: 5 };
        match policy {
            RetentionPolicy::KeepRecent { count } => assert_eq!(count, 5),
            _ => panic!("Expected KeepRecent"),
        }
    }

    #[test]
    fn test_retention_policy_keep_by_age() {
        let policy = RetentionPolicy::KeepByAge { max_age_hours: 24 };
        match policy {
            RetentionPolicy::KeepByAge { max_age_hours } => assert_eq!(max_age_hours, 24),
            _ => panic!("Expected KeepByAge"),
        }
    }

    #[test]
    fn test_retention_policy_keep_by_importance() {
        let policy = RetentionPolicy::KeepByImportance { max_count: 100 };
        match policy {
            RetentionPolicy::KeepByImportance { max_count } => assert_eq!(max_count, 100),
            _ => panic!("Expected KeepByImportance"),
        }
    }

    #[test]
    fn test_retention_policy_clone() {
        let policy1 = RetentionPolicy::KeepRecent { count: 10 };
        let policy2 = policy1.clone();

        match (policy1, policy2) {
            (RetentionPolicy::KeepRecent { count: c1 }, RetentionPolicy::KeepRecent { count: c2 }) => {
                assert_eq!(c1, c2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_checkpoint_new_and_restore() {
        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);
        let metadata = CheckpointMetadata::new(session_id, state.version());

        let checkpoint = Checkpoint::new(&state, metadata).unwrap();
        let restored_state = checkpoint.restore_state().unwrap();

        assert_eq!(state.session_id(), restored_state.session_id());
        assert_eq!(state.version(), restored_state.version());
    }

    #[test]
    fn test_checkpoint_size() {
        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);
        let metadata = CheckpointMetadata::new(session_id, state.version());

        let checkpoint = Checkpoint::new(&state, metadata).unwrap();
        let size = checkpoint.size();

        // Size should be greater than 0
        assert!(size > 0);
    }

    #[test]
    fn test_checkpoint_clone() {
        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);
        let metadata = CheckpointMetadata::new(session_id, state.version());

        let checkpoint1 = Checkpoint::new(&state, metadata).unwrap();
        let checkpoint2 = checkpoint1.clone();

        assert_eq!(checkpoint1.metadata.id, checkpoint2.metadata.id);
        assert_eq!(checkpoint1.state_data.len(), checkpoint2.state_data.len());
    }

    #[test]
    fn test_checkpoint_serialization() {
        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);
        let metadata = CheckpointMetadata::new(session_id, state.version());

        let checkpoint = Checkpoint::new(&state, metadata).unwrap();

        let serialized = serde_json::to_string(&checkpoint).unwrap();
        let deserialized: Checkpoint = serde_json::from_str(&serialized).unwrap();

        assert_eq!(checkpoint.metadata.id, deserialized.metadata.id);
        assert_eq!(checkpoint.state_data.len(), deserialized.state_data.len());
    }

    #[tokio::test]
    async fn test_checkpoint_manager_new() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(50, storage);

        assert_eq!(manager.config.auto_checkpoint_interval, 50);
    }

    #[tokio::test]
    async fn test_checkpoint_manager_with_config() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let config = CheckpointConfig {
            auto_checkpoint_interval: 25,
            max_checkpoints: 100,
            enable_compression: false,
            retention_policy: RetentionPolicy::KeepByAge { max_age_hours: 48 },
        };

        let manager = CheckpointManager::with_config(storage, config.clone());

        assert_eq!(manager.config.auto_checkpoint_interval, 25);
        assert_eq!(manager.config.max_checkpoints, 100);
        assert!(!manager.config.enable_compression);
    }

    #[test]
    fn test_checkpoint_manager_should_checkpoint() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(10, storage);

        let session_id = Uuid::new_v4();
        let mut state = AgentState::new(session_id);

        // At version 1, should not checkpoint (1 % 10 != 0)
        assert!(!manager.should_checkpoint(&state));

        // Simulate state updates to reach version 10
        for _ in 0..9 {
            state.add_memory(crate::memory::types::MemoryEntry::new(
                format!("key_{}", Uuid::new_v4()),
                "test".to_string(),
                crate::memory::types::MemoryType::ShortTerm,
            ));
        }

        // At version 10, should checkpoint (10 % 10 == 0)
        assert!(manager.should_checkpoint(&state));
    }

    #[tokio::test]
    async fn test_checkpoint_manager_create_and_restore() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(10, storage);

        let session_id = Uuid::new_v4();
        let mut state = AgentState::new(session_id);

        // Add some data to the state
        state.add_memory(crate::memory::types::MemoryEntry::new(
            "test_key".to_string(),
            "test_value".to_string(),
            crate::memory::types::MemoryType::ShortTerm,
        ));

        // Create checkpoint
        let checkpoint_id = manager.create_checkpoint(&state).await.unwrap();

        // Restore checkpoint
        let restored_state = manager.restore_checkpoint(checkpoint_id).await.unwrap();

        assert_eq!(state.session_id(), restored_state.session_id());
        assert!(restored_state.has_memory("test_key"));
    }

    #[tokio::test]
    async fn test_checkpoint_manager_list_checkpoints() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(10, storage);

        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);

        // Create multiple checkpoints
        let _ = manager.create_checkpoint(&state).await.unwrap();
        let _ = manager.create_checkpoint(&state).await.unwrap();

        let checkpoints = manager.list_checkpoints(Some(session_id)).await.unwrap();
        assert_eq!(checkpoints.len(), 2);
    }

    #[tokio::test]
    async fn test_checkpoint_manager_get_checkpoint_metadata() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(10, storage);

        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);

        let checkpoint_id = manager.create_checkpoint(&state).await.unwrap();
        let metadata = manager.get_checkpoint_metadata(checkpoint_id).await.unwrap();

        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().id, checkpoint_id);
    }

    #[tokio::test]
    async fn test_checkpoint_manager_delete_checkpoint() {
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let manager = CheckpointManager::new(10, storage);

        let session_id = Uuid::new_v4();
        let state = AgentState::new(session_id);

        let checkpoint_id = manager.create_checkpoint(&state).await.unwrap();
        let deleted = manager.delete_checkpoint(checkpoint_id).await.unwrap();

        assert!(deleted);

        let metadata = manager.get_checkpoint_metadata(checkpoint_id).await.unwrap();
        assert!(metadata.is_none());
    }
}
