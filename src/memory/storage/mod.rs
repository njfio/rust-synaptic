//! Storage backends for the memory system

pub mod memory;
pub mod file;

use crate::error::Result;
use crate::memory::types::{MemoryEntry, MemoryFragment};
use crate::StorageBackend;
use async_trait::async_trait;
use std::sync::Arc;

/// Trait defining the storage interface for memory entries
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store a memory entry
    async fn store(&self, entry: &MemoryEntry) -> Result<()>;

    /// Retrieve a memory entry by key
    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>>;

    /// Search for memory entries matching a query
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>>;

    /// Update an existing memory entry
    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()>;

    /// Delete a memory entry
    async fn delete(&self, key: &str) -> Result<bool>;

    /// List all stored keys
    async fn list_keys(&self) -> Result<Vec<String>>;

    /// Get the total number of stored entries
    async fn count(&self) -> Result<usize>;

    /// Clear all stored entries
    async fn clear(&self) -> Result<()>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Get storage statistics
    async fn stats(&self) -> Result<StorageStats>;

    /// Perform maintenance operations (compaction, cleanup, etc.)
    async fn maintenance(&self) -> Result<()>;

    /// Create a backup of the storage
    async fn backup(&self, path: &str) -> Result<()>;

    /// Restore from a backup
    async fn restore(&self, path: &str) -> Result<()>;

    /// Get all entries (for analysis and processing)
    async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>>;
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub average_entry_size: f64,
    pub storage_type: String,
    pub last_maintenance: Option<chrono::DateTime<chrono::Utc>>,
    pub fragmentation_ratio: f64,
}

impl StorageStats {
    pub fn new(storage_type: String) -> Self {
        Self {
            total_entries: 0,
            total_size_bytes: 0,
            average_entry_size: 0.0,
            storage_type,
            last_maintenance: None,
            fragmentation_ratio: 0.0,
        }
    }
}

/// Create a storage backend based on the configuration
#[tracing::instrument(skip(backend))]
pub async fn create_storage(backend: &StorageBackend) -> Result<Arc<dyn Storage + Send + Sync>> {
    tracing::info!("Creating storage backend: {:?}", backend);

    match backend {
        StorageBackend::Memory => {
            tracing::debug!("Initializing in-memory storage");
            Ok(Arc::new(memory::MemoryStorage::new()))
        }
        StorageBackend::File { path } => {
            tracing::debug!("Initializing file storage at path: {:?}", path);
            Ok(Arc::new(file::FileStorage::new(path).await?))
        }
        #[cfg(feature = "sql-storage")]
        StorageBackend::Sql { connection_string } => {
            tracing::debug!("Initializing SQL storage with connection string");
            Ok(Arc::new(crate::integrations::database::DatabaseClient::new(
                crate::integrations::database::DatabaseConfig {
                    database_url: connection_string.to_string(),
                    max_connections: 10,
                    connect_timeout_secs: 30,
                    schema: "public".to_string(),
                    ssl_mode: "prefer".to_string(),
                }
            ).await?))
        }
    }
}

/// Batch operations for efficient bulk storage operations
#[async_trait]
pub trait BatchStorage: Storage {
    /// Store multiple entries in a single operation
    async fn store_batch(&self, entries: &[MemoryEntry]) -> Result<()>;

    /// Retrieve multiple entries by keys
    async fn retrieve_batch(&self, keys: &[String]) -> Result<Vec<Option<MemoryEntry>>>;

    /// Delete multiple entries by keys
    async fn delete_batch(&self, keys: &[String]) -> Result<usize>;
}

/// Storage configuration options
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub max_entry_size: usize,
    pub cache_size: usize,
    pub sync_interval_seconds: u64,
    pub backup_interval_hours: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_compression: false,
            enable_encryption: false,
            max_entry_size: 1024 * 1024, // 1MB
            cache_size: 1000,
            sync_interval_seconds: 60,
            backup_interval_hours: 24,
        }
    }
}

/// Storage transaction for atomic operations
pub struct StorageTransaction {
    operations: Vec<StorageOperation>,
}

#[derive(Debug, Clone)]
pub enum StorageOperation {
    Store { key: String, entry: MemoryEntry },
    Update { key: String, entry: MemoryEntry },
    Delete { key: String },
}

impl StorageTransaction {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn store(&mut self, key: String, entry: MemoryEntry) {
        self.operations.push(StorageOperation::Store { key, entry });
    }

    pub fn update(&mut self, key: String, entry: MemoryEntry) {
        self.operations.push(StorageOperation::Update { key, entry });
    }

    pub fn delete(&mut self, key: String) {
        self.operations.push(StorageOperation::Delete { key });
    }

    pub fn operations(&self) -> &[StorageOperation] {
        &self.operations
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

impl Default for StorageTransaction {
    fn default() -> Self {
        Self::new()
    }
}

/// Transactional storage trait for atomic operations
#[async_trait]
pub trait TransactionalStorage: Storage {
    /// Execute a transaction atomically
    async fn execute_transaction(&self, transaction: StorageTransaction) -> Result<()>;

    /// Begin a new transaction
    async fn begin_transaction(&self) -> Result<Box<dyn TransactionHandle>>;
}

/// Handle for managing an active transaction
#[async_trait]
pub trait TransactionHandle: Send + Sync {
    /// Store an entry in the transaction
    async fn store(&mut self, key: &str, entry: &MemoryEntry) -> Result<()>;

    /// Update an entry in the transaction
    async fn update(&mut self, key: &str, entry: &MemoryEntry) -> Result<()>;

    /// Delete an entry in the transaction
    async fn delete(&mut self, key: &str) -> Result<()>;

    /// Commit the transaction
    async fn commit(self: Box<Self>) -> Result<()>;

    /// Rollback the transaction
    async fn rollback(self: Box<Self>) -> Result<()>;
}

/// Storage middleware for adding functionality like caching, compression, etc.
pub struct StorageMiddleware {
    inner: Arc<dyn Storage + Send + Sync>,
    _config: StorageConfig,
}

impl StorageMiddleware {
    pub fn new(inner: Arc<dyn Storage + Send + Sync>, config: StorageConfig) -> Self {
        Self { inner, _config: config }
    }
}

#[async_trait]
impl Storage for StorageMiddleware {
    async fn store(&self, entry: &MemoryEntry) -> Result<()> {
        // Apply middleware logic (compression, encryption, etc.)
        self.inner.store(entry).await
    }

    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.inner.retrieve(key).await
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        self.inner.search(query, limit).await
    }

    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()> {
        self.inner.update(key, entry).await
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        self.inner.delete(key).await
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        self.inner.list_keys().await
    }

    async fn count(&self) -> Result<usize> {
        self.inner.count().await
    }

    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key).await
    }

    async fn stats(&self) -> Result<StorageStats> {
        self.inner.stats().await
    }

    async fn maintenance(&self) -> Result<()> {
        self.inner.maintenance().await
    }

    async fn backup(&self, path: &str) -> Result<()> {
        self.inner.backup(path).await
    }

    async fn restore(&self, path: &str) -> Result<()> {
        self.inner.restore(path).await
    }

    async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>> {
        self.inner.get_all_entries().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::{MemoryEntry, MemoryMetadata, MemoryType};
    use uuid::Uuid;

    #[test]
    fn test_storage_stats_new() {
        let stats = StorageStats::new("memory".to_string());
        assert_eq!(stats.storage_type, "memory");
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert_eq!(stats.average_entry_size, 0.0);
        assert!(stats.last_maintenance.is_none());
        assert_eq!(stats.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_storage_stats_clone() {
        let stats1 = StorageStats::new("file".to_string());
        let stats2 = stats1.clone();
        assert_eq!(stats1.storage_type, stats2.storage_type);
        assert_eq!(stats1.total_entries, stats2.total_entries);
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(!config.enable_compression);
        assert!(!config.enable_encryption);
        assert_eq!(config.max_entry_size, 1024 * 1024);
        assert_eq!(config.cache_size, 1000);
        assert_eq!(config.sync_interval_seconds, 60);
        assert_eq!(config.backup_interval_hours, 24);
    }

    #[test]
    fn test_storage_config_clone() {
        let config1 = StorageConfig::default();
        let config2 = config1.clone();
        assert_eq!(config1.enable_compression, config2.enable_compression);
        assert_eq!(config1.max_entry_size, config2.max_entry_size);
    }

    #[test]
    fn test_storage_transaction_new() {
        let transaction = StorageTransaction::new();
        assert!(transaction.is_empty());
        assert_eq!(transaction.len(), 0);
    }

    #[test]
    fn test_storage_transaction_default() {
        let transaction = StorageTransaction::default();
        assert!(transaction.is_empty());
        assert_eq!(transaction.len(), 0);
    }

    #[test]
    fn test_storage_transaction_store() {
        let mut transaction = StorageTransaction::new();
        let entry = create_test_entry();
        transaction.store("test_key".to_string(), entry.clone());

        assert!(!transaction.is_empty());
        assert_eq!(transaction.len(), 1);

        let ops = transaction.operations();
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            StorageOperation::Store { key, .. } => {
                assert_eq!(key, "test_key");
            }
            _ => panic!("Expected Store operation"),
        }
    }

    #[test]
    fn test_storage_transaction_update() {
        let mut transaction = StorageTransaction::new();
        let entry = create_test_entry();
        transaction.update("update_key".to_string(), entry.clone());

        assert_eq!(transaction.len(), 1);

        let ops = transaction.operations();
        match &ops[0] {
            StorageOperation::Update { key, .. } => {
                assert_eq!(key, "update_key");
            }
            _ => panic!("Expected Update operation"),
        }
    }

    #[test]
    fn test_storage_transaction_delete() {
        let mut transaction = StorageTransaction::new();
        transaction.delete("delete_key".to_string());

        assert_eq!(transaction.len(), 1);

        let ops = transaction.operations();
        match &ops[0] {
            StorageOperation::Delete { key } => {
                assert_eq!(key, "delete_key");
            }
            _ => panic!("Expected Delete operation"),
        }
    }

    #[test]
    fn test_storage_transaction_multiple_operations() {
        let mut transaction = StorageTransaction::new();
        let entry = create_test_entry();

        transaction.store("key1".to_string(), entry.clone());
        transaction.update("key2".to_string(), entry.clone());
        transaction.delete("key3".to_string());

        assert_eq!(transaction.len(), 3);
        assert!(!transaction.is_empty());

        let ops = transaction.operations();
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn test_storage_operation_store_clone() {
        let entry = create_test_entry();
        let op = StorageOperation::Store {
            key: "test".to_string(),
            entry: entry.clone(),
        };
        let op_clone = op.clone();

        match (op, op_clone) {
            (StorageOperation::Store { key: k1, .. }, StorageOperation::Store { key: k2, .. }) => {
                assert_eq!(k1, k2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_storage_operation_update_clone() {
        let entry = create_test_entry();
        let op = StorageOperation::Update {
            key: "test".to_string(),
            entry: entry.clone(),
        };
        let op_clone = op.clone();

        match (op, op_clone) {
            (StorageOperation::Update { key: k1, .. }, StorageOperation::Update { key: k2, .. }) => {
                assert_eq!(k1, k2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_storage_operation_delete_clone() {
        let op = StorageOperation::Delete {
            key: "test".to_string(),
        };
        let op_clone = op.clone();

        match (op, op_clone) {
            (StorageOperation::Delete { key: k1 }, StorageOperation::Delete { key: k2 }) => {
                assert_eq!(k1, k2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[tokio::test]
    async fn test_create_storage_memory_backend() {
        let backend = StorageBackend::Memory;
        let result = create_storage(&backend).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_storage_file_backend() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_storage.db");

        let backend = StorageBackend::File {
            path: test_path.to_string_lossy().to_string(),
        };

        let result = create_storage(&backend).await;
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_file(test_path);
    }

    #[test]
    fn test_storage_middleware_creation() {
        let memory_storage = Arc::new(memory::MemoryStorage::new());
        let config = StorageConfig::default();
        let middleware = StorageMiddleware::new(memory_storage, config);

        // Middleware should be created successfully
        assert!(Arc::strong_count(&middleware.inner) >= 1);
    }

    #[tokio::test]
    async fn test_storage_middleware_delegates_to_inner() {
        let memory_storage = Arc::new(memory::MemoryStorage::new());
        let config = StorageConfig::default();
        let middleware = StorageMiddleware::new(memory_storage.clone(), config);

        let entry = create_test_entry();

        // Store through middleware
        let result = middleware.store(&entry).await;
        assert!(result.is_ok());

        // Verify count through middleware
        let count_result = middleware.count().await;
        assert!(count_result.is_ok());
        assert_eq!(count_result.unwrap(), 1);
    }

    #[test]
    fn test_storage_stats_with_data() {
        let mut stats = StorageStats::new("file".to_string());
        stats.total_entries = 100;
        stats.total_size_bytes = 1024 * 100;
        stats.average_entry_size = 1024.0;
        stats.fragmentation_ratio = 0.15;
        stats.last_maintenance = Some(chrono::Utc::now());

        assert_eq!(stats.total_entries, 100);
        assert_eq!(stats.total_size_bytes, 102400);
        assert_eq!(stats.average_entry_size, 1024.0);
        assert_eq!(stats.fragmentation_ratio, 0.15);
        assert!(stats.last_maintenance.is_some());
    }

    #[test]
    fn test_storage_config_custom_values() {
        let config = StorageConfig {
            enable_compression: true,
            enable_encryption: true,
            max_entry_size: 5 * 1024 * 1024,
            cache_size: 5000,
            sync_interval_seconds: 30,
            backup_interval_hours: 12,
        };

        assert!(config.enable_compression);
        assert!(config.enable_encryption);
        assert_eq!(config.max_entry_size, 5 * 1024 * 1024);
        assert_eq!(config.cache_size, 5000);
        assert_eq!(config.sync_interval_seconds, 30);
        assert_eq!(config.backup_interval_hours, 12);
    }

    #[test]
    fn test_storage_transaction_operations_order() {
        let mut transaction = StorageTransaction::new();
        let entry = create_test_entry();

        transaction.store("key1".to_string(), entry.clone());
        transaction.update("key2".to_string(), entry.clone());
        transaction.delete("key3".to_string());

        let ops = transaction.operations();

        // Verify order is preserved
        match &ops[0] {
            StorageOperation::Store { key, .. } => assert_eq!(key, "key1"),
            _ => panic!("Expected Store at index 0"),
        }

        match &ops[1] {
            StorageOperation::Update { key, .. } => assert_eq!(key, "key2"),
            _ => panic!("Expected Update at index 1"),
        }

        match &ops[2] {
            StorageOperation::Delete { key } => assert_eq!(key, "key3"),
            _ => panic!("Expected Delete at index 2"),
        }
    }

    // Helper function to create test entries
    fn create_test_entry() -> MemoryEntry {
        MemoryEntry {
            id: Uuid::new_v4(),
            content: "test content".to_string(),
            memory_type: MemoryType::ShortTerm,
            metadata: MemoryMetadata::new(),
            created_at: chrono::Utc::now(),
            accessed_at: chrono::Utc::now(),
            embedding: None,
        }
    }
}
