//! # Offline-First Support
//!
//! Offline-first capabilities for the Synaptic memory system, enabling local storage,
//! conflict resolution, and seamless online/offline transitions.

use super::{
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, PlatformInfo, Platform,
    PerformanceProfile, StorageBackend, StorageStats,
};
use crate::error::MemoryError as SynapticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Offline storage adapter
pub struct OfflineAdapter {
    /// Local storage backend
    storage: Arc<RwLock<LocalStorage>>,
    
    /// Offline configuration
    config: OfflineConfig,
    
    /// Sync queue for when online
    sync_queue: Arc<RwLock<Vec<SyncOperation>>>,
    
    /// Conflict resolution strategy
    conflict_resolver: Box<dyn ConflictResolver>,
}

/// Offline-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineConfig {
    /// Local storage directory
    pub storage_dir: PathBuf,
    
    /// Enable automatic sync when online
    pub auto_sync: bool,
    
    /// Enable conflict detection
    pub enable_conflict_detection: bool,
    
    /// Maximum offline storage size (bytes)
    pub max_offline_storage: usize,
    
    /// Sync queue size limit
    pub max_sync_queue_size: usize,
    
    /// Enable data compression
    pub enable_compression: bool,
    
    /// Enable encryption for local storage
    pub enable_encryption: bool,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("./synaptic_offline"),
            auto_sync: true,
            enable_conflict_detection: true,
            max_offline_storage: 500 * 1024 * 1024, // 500MB
            max_sync_queue_size: 1000,
            enable_compression: true,
            enable_encryption: false,
        }
    }
}

/// Local storage implementation
#[derive(Debug)]
struct LocalStorage {
    /// In-memory storage for fast access
    memory_store: HashMap<String, StoredItem>,
    
    /// File system storage for persistence
    file_store: HashMap<String, PathBuf>,
    
    /// Storage directory
    storage_dir: PathBuf,
    
    /// Total storage used
    total_size: usize,
}

/// Stored item with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredItem {
    /// Item data
    data: Vec<u8>,
    
    /// Creation timestamp
    created_at: u64,
    
    /// Last modified timestamp
    modified_at: u64,
    
    /// Version for conflict resolution
    version: u64,
    
    /// Checksum for integrity
    checksum: String,
    
    /// Sync status
    sync_status: SyncStatus,
}

/// Sync status for offline items
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum SyncStatus {
    /// Item is synced with remote
    Synced,
    
    /// Item needs to be uploaded
    PendingUpload,
    
    /// Item needs to be downloaded
    PendingDownload,
    
    /// Item has conflicts
    Conflicted,
    
    /// Item is being synced
    Syncing,
}

/// Sync operation for queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperation {
    /// Upload local item to remote
    Upload {
        key: String,
        data: Vec<u8>,
        version: u64,
    },
    
    /// Download remote item to local
    Download {
        key: String,
        remote_version: u64,
    },
    
    /// Delete item from remote
    Delete {
        key: String,
    },
    
    /// Resolve conflict
    ResolveConflict {
        key: String,
        local_version: u64,
        remote_version: u64,
        resolution: ConflictResolution,
    },
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Use local version
    UseLocal,
    
    /// Use remote version
    UseRemote,
    
    /// Merge versions (if possible)
    Merge,
    
    /// Create new version
    CreateNew,
}

/// Trait for conflict resolution
pub trait ConflictResolver: Send + Sync {
    /// Resolve conflict between local and remote versions
    fn resolve_conflict(
        &self,
        key: &str,
        local_item: &StoredItem,
        remote_item: &StoredItem,
    ) -> Result<ConflictResolution, SynapticError>;
}

/// Default conflict resolver (last-write-wins)
#[derive(Debug)]
pub struct LastWriteWinsResolver;

impl ConflictResolver for LastWriteWinsResolver {
    fn resolve_conflict(
        &self,
        _key: &str,
        local_item: &StoredItem,
        remote_item: &StoredItem,
    ) -> Result<ConflictResolution, SynapticError> {
        if local_item.modified_at > remote_item.modified_at {
            Ok(ConflictResolution::UseLocal)
        } else {
            Ok(ConflictResolution::UseRemote)
        }
    }
}

impl LocalStorage {
    /// Create new local storage
    fn new(storage_dir: PathBuf) -> Result<Self, SynapticError> {
        // Create storage directory if it doesn't exist
        std::fs::create_dir_all(&storage_dir)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create storage directory: {}", e)))?;

        Ok(Self {
            memory_store: HashMap::new(),
            file_store: HashMap::new(),
            storage_dir,
            total_size: 0,
        })
    }

    /// Store item
    fn store(&mut self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate checksum
        let checksum = format!("{:x}", md5::compute(data));

        let item = StoredItem {
            data: data.to_vec(),
            created_at: now,
            modified_at: now,
            version: now, // Use timestamp as version
            checksum,
            sync_status: SyncStatus::PendingUpload,
        };

        // Store in memory
        self.memory_store.insert(key.to_string(), item.clone());

        // Store to file system
        let file_path = self.storage_dir.join(format!("{}.dat", key));
        let serialized = bincode::serialize(&item)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to serialize item: {}", e)))?;

        std::fs::write(&file_path, serialized)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to write file: {}", e)))?;

        self.file_store.insert(key.to_string(), file_path);
        self.total_size += data.len();

        Ok(())
    }

    /// Retrieve item
    fn retrieve(&mut self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        // Check memory first
        if let Some(item) = self.memory_store.get(key) {
            return Ok(Some(item.data.clone()));
        }

        // Check file system
        if let Some(file_path) = self.file_store.get(key) {
            if file_path.exists() {
                let serialized = std::fs::read(file_path)
                    .map_err(|e| SynapticError::ProcessingError(format!("Failed to read file: {}", e)))?;

                let item: StoredItem = bincode::deserialize(&serialized)
                    .map_err(|e| SynapticError::ProcessingError(format!("Failed to deserialize item: {}", e)))?;

                // Verify checksum
                let checksum = format!("{:x}", md5::compute(&item.data));
                if checksum != item.checksum {
                    return Err(SynapticError::ProcessingError("Data corruption detected".to_string()));
                }

                // Cache in memory
                self.memory_store.insert(key.to_string(), item.clone());

                return Ok(Some(item.data));
            }
        }

        Ok(None)
    }

    /// Delete item
    fn delete(&mut self, key: &str) -> Result<bool, SynapticError> {
        let mut deleted = false;

        // Remove from memory
        if let Some(item) = self.memory_store.remove(key) {
            self.total_size = self.total_size.saturating_sub(item.data.len());
            deleted = true;
        }

        // Remove from file system
        if let Some(file_path) = self.file_store.remove(key) {
            if file_path.exists() {
                std::fs::remove_file(file_path)
                    .map_err(|e| SynapticError::ProcessingError(format!("Failed to delete file: {}", e)))?;
                deleted = true;
            }
        }

        Ok(deleted)
    }

    /// List all keys
    fn list_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.memory_store.keys().cloned().collect();
        keys.extend(self.file_store.keys().cloned());
        keys.sort();
        keys.dedup();
        keys
    }

    /// Get storage statistics
    fn get_stats(&self) -> StorageStats {
        let item_count = self.memory_store.len().max(self.file_store.len());
        let average_item_size = if item_count > 0 {
            self.total_size / item_count
        } else {
            0
        };

        StorageStats {
            used_storage: self.total_size,
            available_storage: 1024 * 1024 * 1024, // 1GB estimate
            item_count,
            average_item_size,
            backend: StorageBackend::FileSystem,
        }
    }

    /// Load existing data from storage directory
    fn load_existing(&mut self) -> Result<(), SynapticError> {
        if !self.storage_dir.exists() {
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.storage_dir)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to read storage directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to read directory entry: {}", e)))?;
            
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("dat") {
                if let Some(key) = path.file_stem().and_then(|s| s.to_str()) {
                    let serialized = std::fs::read(&path)
                        .map_err(|e| SynapticError::ProcessingError(format!("Failed to read file: {}", e)))?;

                    let item: StoredItem = bincode::deserialize(&serialized)
                        .map_err(|e| SynapticError::ProcessingError(format!("Failed to deserialize item: {}", e)))?;

                    self.total_size += item.data.len();
                    self.file_store.insert(key.to_string(), path);
                }
            }
        }

        Ok(())
    }
}

impl OfflineAdapter {
    /// Create new offline adapter
    pub fn new() -> Result<Self, SynapticError> {
        let config = OfflineConfig::default();
        let mut storage = LocalStorage::new(config.storage_dir.clone())?;
        storage.load_existing()?;

        Ok(Self {
            storage: Arc::new(RwLock::new(storage)),
            config,
            sync_queue: Arc::new(RwLock::new(Vec::new())),
            conflict_resolver: Box::new(LastWriteWinsResolver),
        })
    }

    /// Add operation to sync queue
    pub fn queue_sync_operation(&self, operation: SyncOperation) -> Result<(), SynapticError> {
        let mut queue = self.sync_queue.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire sync queue lock: {}", e)))?;

        if queue.len() >= self.config.max_sync_queue_size {
            // Remove oldest operation
            queue.remove(0);
        }

        queue.push(operation);
        Ok(())
    }

    /// Get pending sync operations
    pub fn get_pending_operations(&self) -> Result<Vec<SyncOperation>, SynapticError> {
        let queue = self.sync_queue.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire sync queue lock: {}", e)))?;
        
        Ok(queue.clone())
    }

    /// Clear sync queue
    pub fn clear_sync_queue(&self) -> Result<(), SynapticError> {
        let mut queue = self.sync_queue.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire sync queue lock: {}", e)))?;
        
        queue.clear();
        Ok(())
    }

    /// Check if online (placeholder implementation)
    pub fn is_online(&self) -> bool {
        // In a real implementation, this would check network connectivity
        true
    }

    /// Sync with remote (placeholder implementation)
    pub async fn sync_with_remote(&self) -> Result<(), SynapticError> {
        if !self.is_online() {
            return Ok(()); // Skip sync if offline
        }

        let operations = self.get_pending_operations()?;
        
        for operation in operations {
            match operation {
                SyncOperation::Upload { key, data, version } => {
                    // Upload to remote storage
                    println!("Uploading {} (version {})", key, version);
                }
                SyncOperation::Download { key, remote_version } => {
                    // Download from remote storage
                    println!("Downloading {} (version {})", key, remote_version);
                }
                SyncOperation::Delete { key } => {
                    // Delete from remote storage
                    println!("Deleting {}", key);
                }
                SyncOperation::ResolveConflict { key, resolution, .. } => {
                    // Resolve conflict
                    println!("Resolving conflict for {} with strategy {:?}", key, resolution);
                }
            }
        }

        self.clear_sync_queue()?;
        Ok(())
    }
}

impl CrossPlatformAdapter for OfflineAdapter {
    fn initialize(&mut self, _config: &PlatformConfig) -> Result<(), SynapticError> {
        // Offline adapter is already initialized in new()
        Ok(())
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        storage.store(key, data)?;

        // Queue sync operation
        self.queue_sync_operation(SyncOperation::Upload {
            key: key.to_string(),
            data: data.to_vec(),
            version: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })?;

        Ok(())
    }

    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        storage.retrieve(key)
    }

    fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        let result = storage.delete(key)?;

        if result {
            // Queue sync operation
            self.queue_sync_operation(SyncOperation::Delete {
                key: key.to_string(),
            })?;
        }

        Ok(result)
    }

    fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        let storage = self.storage.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        Ok(storage.list_keys())
    }

    fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        let storage = self.storage.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        Ok(storage.get_stats())
    }

    fn supports_feature(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::FileSystemAccess => true,
            PlatformFeature::NetworkAccess => true,
            PlatformFeature::BackgroundProcessing => true,
            PlatformFeature::PushNotifications => false,
            PlatformFeature::HardwareAcceleration => true,
            PlatformFeature::MultiThreading => true,
            PlatformFeature::LargeMemoryAllocation => true,
        }
    }

    fn get_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            platform: Platform::Desktop, // Assuming desktop for offline adapter
            version: "1.0.0".to_string(),
            available_memory: 1024 * 1024 * 1024, // 1GB
            available_storage: 10 * 1024 * 1024 * 1024, // 10GB
            supported_features: vec![
                PlatformFeature::FileSystemAccess,
                PlatformFeature::NetworkAccess,
                PlatformFeature::BackgroundProcessing,
                PlatformFeature::HardwareAcceleration,
                PlatformFeature::MultiThreading,
                PlatformFeature::LargeMemoryAllocation,
            ],
            performance_profile: PerformanceProfile {
                cpu_score: 0.9,
                memory_score: 0.9,
                storage_score: 0.8,
                network_score: 0.7,
                battery_optimization: false,
            },
        }
    }
}
