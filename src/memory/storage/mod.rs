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
    config: StorageConfig,
}

impl StorageMiddleware {
    pub fn new(inner: Arc<dyn Storage + Send + Sync>, config: StorageConfig) -> Self {
        Self { inner, config }
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
