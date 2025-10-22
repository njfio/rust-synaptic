//! In-memory storage implementation for the memory system

use crate::error::{MemoryError, Result};
use crate::memory::storage::{Storage, StorageStats, BatchStorage, StorageTransaction, TransactionalStorage, TransactionHandle};
use crate::memory::types::{MemoryEntry, MemoryFragment};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;

/// Backup data structure for in-memory storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryStorageBackup {
    /// All memory entries
    entries: Vec<MemoryEntry>,
    /// Original storage creation timestamp
    created_at: DateTime<Utc>,
    /// Backup creation timestamp
    backup_timestamp: DateTime<Utc>,
    /// Backup format version
    version: String,
    /// Number of entries in backup
    entry_count: usize,
    /// Total size of all entries in bytes
    total_size: usize,
    /// Backup metadata
    metadata: BackupMetadata,
}

/// Metadata for backup operations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackupMetadata {
    /// Backup creation method (manual, automatic, etc.)
    creation_method: String,
    /// Compression used (if any)
    compression: Option<String>,
    /// Checksum for integrity verification
    checksum: String,
    /// Additional custom metadata
    custom_fields: HashMap<String, String>,
}

/// Options for backup operations
#[derive(Debug, Clone)]
pub struct BackupOptions {
    /// Creation method identifier
    pub creation_method: String,
    /// Compression algorithm to use
    pub compression: Option<String>,
    /// Whether to format JSON with pretty printing
    pub pretty_print: bool,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
}

impl Default for BackupOptions {
    fn default() -> Self {
        Self {
            creation_method: "manual".to_string(),
            compression: None,
            pretty_print: true,
            custom_fields: HashMap::new(),
        }
    }
}

/// Options for restore operations
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    /// Whether to merge with existing data or replace it
    pub merge_strategy: bool,
    /// Whether to overwrite existing entries when merging
    pub overwrite_existing: bool,
    /// Whether to verify backup integrity
    pub verify_integrity: bool,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            merge_strategy: false,
            overwrite_existing: false,
            verify_integrity: true,
        }
    }
}

/// In-memory storage implementation using DashMap for thread-safe concurrent access
pub struct MemoryStorage {
    /// Main storage for memory entries (wrapped in Arc for transaction support)
    entries: Arc<DashMap<String, MemoryEntry>>,
    /// Statistics tracking
    stats: RwLock<StorageStats>,
    /// Creation timestamp
    created_at: DateTime<Utc>,
}

impl MemoryStorage {
    /// Create a new in-memory storage instance
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            stats: RwLock::new(StorageStats::new("memory".to_string())),
            created_at: Utc::now(),
        }
    }

    /// Get the number of stored entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the storage is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Update internal statistics
    fn update_stats(&self) {
        let mut stats = self.stats.write();
        stats.total_entries = self.entries.len();
        
        let total_size: usize = self.entries
            .iter()
            .map(|entry| entry.value().estimated_size())
            .sum();
        
        stats.total_size_bytes = total_size;
        stats.average_entry_size = if stats.total_entries > 0 {
            total_size as f64 / stats.total_entries as f64
        } else {
            0.0
        };
    }

    /// Perform simple text-based search
    fn search_entries(&self, query: &str, limit: usize) -> Vec<MemoryFragment> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for entry_ref in self.entries.iter() {
            let entry = entry_ref.value();
            let content_lower = entry.value.to_lowercase();
            
            // Simple relevance scoring based on query term frequency
            let relevance_score = if content_lower.contains(&query_lower) {
                // Count occurrences of query terms
                let query_terms: Vec<&str> = query_lower.split_whitespace().collect();
                let mut score = 0.0;
                
                for term in query_terms {
                    let occurrences = content_lower.matches(term).count();
                    score += occurrences as f64;
                }
                
                // Normalize by content length
                score / content_lower.len() as f64
            } else {
                0.0
            };

            if relevance_score > 0.0 {
                let fragment = MemoryFragment::new(entry.clone(), relevance_score);
                results.push(fragment);
            }

            if results.len() >= limit {
                break;
            }
        }

        // Sort by relevance score (highest first)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    #[tracing::instrument(skip(self, entry), fields(key = %entry.key))]
    async fn store(&self, entry: &MemoryEntry) -> Result<()> {
        tracing::debug!("Storing entry in memory storage");
        self.entries.insert(entry.key.clone(), entry.clone());
        self.update_stats();
        tracing::debug!("Entry stored successfully");
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(key = %key))]
    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>> {
        tracing::debug!("Retrieving entry from memory storage");
        let result = self.entries.get(key).map(|entry_ref| {
            // Clone only when we actually have an entry to avoid unnecessary operations
            entry_ref.value().clone()
        });

        if result.is_some() {
            tracing::debug!("Entry found in memory storage");
        } else {
            tracing::debug!("Entry not found in memory storage");
        }
        Ok(result)
    }

    #[tracing::instrument(skip(self, query), fields(query_len = query.len(), limit = limit))]
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        tracing::debug!("Searching entries in memory storage");
        let results = self.search_entries(query, limit);
        tracing::debug!("Search completed, found {} results", results.len());
        Ok(results)
    }

    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()> {
        if self.entries.contains_key(key) {
            self.entries.insert(key.to_string(), entry.clone());
            self.update_stats();
            Ok(())
        } else {
            Err(MemoryError::NotFound {
                key: key.to_string(),
            })
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let removed = self.entries.remove(key).is_some();
        if removed {
            self.update_stats();
        }
        Ok(removed)
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        Ok(self.entries.iter().map(|entry| entry.key().clone()).collect())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.entries.len())
    }

    async fn clear(&self) -> Result<()> {
        self.entries.clear();
        self.update_stats();
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.entries.contains_key(key))
    }

    async fn stats(&self) -> Result<StorageStats> {
        self.update_stats();
        Ok(self.stats.read().clone())
    }

    async fn maintenance(&self) -> Result<()> {
        // For in-memory storage, maintenance is essentially a no-op
        // but we can update statistics
        self.update_stats();
        Ok(())
    }

    async fn backup(&self, path: &str) -> Result<()> {
        self.backup_with_options(path, BackupOptions::default()).await
    }

    async fn restore(&self, path: &str) -> Result<()> {
        self.restore_with_options(path, RestoreOptions::default()).await
    }

    async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>> {
        Ok(self.entries.iter().map(|entry| entry.value().clone()).collect())
    }
}

impl MemoryStorage {
    /// Create a backup with specific options
    pub async fn backup_with_options(&self, path: &str, options: BackupOptions) -> Result<()> {
        tracing::info!("Creating backup of in-memory storage to: {} with options: {:?}", path, options);

        // Collect all entries
        let entries: Vec<MemoryEntry> = self.entries
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Calculate total size
        let total_size = entries.iter().map(|e| e.estimated_size()).sum();

        // Create backup metadata
        let json_data = serde_json::to_string_pretty(&entries)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize entries for checksum: {}", e)))?;

        let checksum = self.calculate_checksum(&json_data);

        let metadata = BackupMetadata {
            creation_method: options.creation_method.clone(),
            compression: options.compression.clone(),
            checksum,
            custom_fields: options.custom_fields.clone(),
        };

        // Create backup data structure
        let backup_data = MemoryStorageBackup {
            entries,
            created_at: self.created_at,
            backup_timestamp: Utc::now(),
            version: "1.1".to_string(), // Updated version for enhanced format
            entry_count: self.entries.len(),
            total_size,
            metadata,
        };

        // Serialize to JSON
        let json_data = if options.pretty_print {
            serde_json::to_string_pretty(&backup_data)
        } else {
            serde_json::to_string(&backup_data)
        }.map_err(|e| MemoryError::storage(format!("Failed to serialize backup data: {}", e)))?;

        // Apply compression if requested
        let final_data = if options.compression.is_some() {
            self.compress_data(&json_data)?
        } else {
            json_data.into_bytes()
        };

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| MemoryError::storage(format!("Failed to create backup directory: {}", e)))?;
        }

        // Write to file
        tokio::fs::write(path, final_data).await
            .map_err(|e| MemoryError::storage(format!("Failed to write backup file: {}", e)))?;

        tracing::info!(
            "Successfully created backup with {} entries, total size: {} bytes, compression: {:?}",
            backup_data.entry_count,
            backup_data.total_size,
            options.compression
        );
        Ok(())
    }

    /// Restore from backup with specific options
    pub async fn restore_with_options(&self, path: &str, options: RestoreOptions) -> Result<()> {
        tracing::info!("Restoring in-memory storage from: {} with options: {:?}", path, options);

        // Read backup file
        let file_data = tokio::fs::read(path).await
            .map_err(|e| MemoryError::storage(format!("Failed to read backup file: {}", e)))?;

        // Decompress if needed
        let json_data = if self.is_compressed_data(&file_data) {
            String::from_utf8(self.decompress_data(&file_data)?)
                .map_err(|e| MemoryError::storage(format!("Failed to decode decompressed data: {}", e)))?
        } else {
            String::from_utf8(file_data)
                .map_err(|e| MemoryError::storage(format!("Failed to decode backup file: {}", e)))?
        };

        // Try to deserialize as new format first, then fall back to old format
        let backup_data = match serde_json::from_str::<MemoryStorageBackup>(&json_data) {
            Ok(data) => data,
            Err(_) => {
                // Try legacy format (version 1.0)
                tracing::warn!("Attempting to restore from legacy backup format");
                self.restore_legacy_format(&json_data)?
            }
        };

        // Validate backup version compatibility
        if !self.is_version_compatible(&backup_data.version) {
            return Err(MemoryError::storage(format!(
                "Unsupported backup version: {}. Supported versions: 1.0, 1.1",
                backup_data.version
            )));
        }

        // Verify checksum if available
        if backup_data.version == "1.1" {
            self.verify_backup_integrity(&backup_data)?;
        }

        // Handle merge vs replace strategy
        if options.merge_strategy {
            tracing::info!("Merging backup entries with existing data");
            // Don't clear existing entries, just add/update from backup
        } else {
            tracing::info!("Replacing all existing data with backup");
            self.entries.clear();
        }

        // Restore entries
        let mut restored_count = 0;
        let mut skipped_count = 0;

        for entry in backup_data.entries {
            if options.merge_strategy && self.entries.contains_key(&entry.key) && !options.overwrite_existing {
                skipped_count += 1;
                continue;
            }

            self.entries.insert(entry.key.clone(), entry);
            restored_count += 1;
        }

        // Update statistics
        self.update_stats();

        tracing::info!(
            "Successfully restored {} entries from backup (skipped: {}, total in backup: {})",
            restored_count,
            skipped_count,
            backup_data.entry_count
        );
        Ok(())
    }
    /// Calculate checksum for data integrity verification
    fn calculate_checksum(&self, data: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Compress data using simple compression
    #[cfg(feature = "compression")]
    fn compress_data(&self, data: &str) -> Result<Vec<u8>> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data.as_bytes())
            .map_err(|e| MemoryError::storage(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| MemoryError::storage(format!("Compression finalization failed: {}", e)))
    }

    /// Compress data (no-op when compression feature is disabled)
    #[cfg(not(feature = "compression"))]
    fn compress_data(&self, data: &str) -> Result<Vec<u8>> {
        Ok(data.as_bytes().to_vec())
    }

    /// Decompress data
    #[cfg(feature = "compression")]
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| MemoryError::storage(format!("Decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    /// Decompress data (no-op when compression feature is disabled)
    #[cfg(not(feature = "compression"))]
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    /// Check if data is compressed
    fn is_compressed_data(&self, data: &[u8]) -> bool {
        // Check for gzip magic number
        data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
    }

    /// Check if backup version is compatible
    fn is_version_compatible(&self, version: &str) -> bool {
        matches!(version, "1.0" | "1.1")
    }

    /// Verify backup integrity using checksum
    fn verify_backup_integrity(&self, backup_data: &MemoryStorageBackup) -> Result<()> {
        let entries_json = serde_json::to_string_pretty(&backup_data.entries)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize entries for verification: {}", e)))?;

        let calculated_checksum = self.calculate_checksum(&entries_json);

        if calculated_checksum != backup_data.metadata.checksum {
            return Err(MemoryError::storage(format!(
                "Backup integrity verification failed. Expected checksum: {}, calculated: {}",
                backup_data.metadata.checksum,
                calculated_checksum
            )));
        }

        tracing::info!("Backup integrity verification passed");
        Ok(())
    }

    /// Restore from legacy backup format (version 1.0)
    fn restore_legacy_format(&self, json_data: &str) -> Result<MemoryStorageBackup> {
        // Try to parse as legacy format
        #[derive(Deserialize)]
        struct LegacyBackup {
            entries: Vec<MemoryEntry>,
            created_at: DateTime<Utc>,
            backup_timestamp: DateTime<Utc>,
            version: String,
            entry_count: usize,
        }

        let legacy_backup: LegacyBackup = serde_json::from_str(json_data)
            .map_err(|e| MemoryError::storage(format!("Failed to parse legacy backup format: {}", e)))?;

        // Convert to new format
        let total_size = legacy_backup.entries.iter().map(|e| e.estimated_size()).sum();
        let checksum = self.calculate_checksum(&serde_json::to_string(&legacy_backup.entries).unwrap_or_default());

        Ok(MemoryStorageBackup {
            entries: legacy_backup.entries,
            created_at: legacy_backup.created_at,
            backup_timestamp: legacy_backup.backup_timestamp,
            version: legacy_backup.version,
            entry_count: legacy_backup.entry_count,
            total_size,
            metadata: BackupMetadata {
                creation_method: "legacy".to_string(),
                compression: None,
                checksum,
                custom_fields: HashMap::new(),
            },
        })
    }

    /// Get backup information without restoring
    pub async fn get_backup_info(&self, path: &str) -> Result<BackupInfo> {
        let file_data = tokio::fs::read(path).await
            .map_err(|e| MemoryError::storage(format!("Failed to read backup file: {}", e)))?;

        let json_data = if self.is_compressed_data(&file_data) {
            String::from_utf8(self.decompress_data(&file_data)?)
                .map_err(|e| MemoryError::storage(format!("Failed to decode decompressed data: {}", e)))?
        } else {
            String::from_utf8(file_data)
                .map_err(|e| MemoryError::storage(format!("Failed to decode backup file: {}", e)))?
        };

        let backup_data: MemoryStorageBackup = serde_json::from_str(&json_data)
            .map_err(|e| MemoryError::storage(format!("Failed to deserialize backup data: {}", e)))?;

        Ok(BackupInfo {
            version: backup_data.version,
            entry_count: backup_data.entry_count,
            total_size: backup_data.total_size,
            created_at: backup_data.created_at,
            backup_timestamp: backup_data.backup_timestamp,
            creation_method: backup_data.metadata.creation_method,
            compression: backup_data.metadata.compression,
            is_compressed: self.is_compressed_data(&tokio::fs::read(path).await.unwrap_or_default()),
        })
    }
}

/// Information about a backup file
#[derive(Debug, Clone)]
pub struct BackupInfo {
    pub version: String,
    pub entry_count: usize,
    pub total_size: usize,
    pub created_at: DateTime<Utc>,
    pub backup_timestamp: DateTime<Utc>,
    pub creation_method: String,
    pub compression: Option<String>,
    pub is_compressed: bool,
}

#[async_trait]
impl BatchStorage for MemoryStorage {
    async fn store_batch(&self, entries: &[MemoryEntry]) -> Result<()> {
        for entry in entries {
            self.entries.insert(entry.key.clone(), entry.clone());
        }
        self.update_stats();
        Ok(())
    }

    async fn retrieve_batch(&self, keys: &[String]) -> Result<Vec<Option<MemoryEntry>>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(self.entries.get(key).map(|entry| entry.value().clone()));
        }
        Ok(results)
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        for key in keys {
            if self.entries.remove(key).is_some() {
                deleted_count += 1;
            }
        }
        if deleted_count > 0 {
            self.update_stats();
        }
        Ok(deleted_count)
    }
}

/// In-memory transaction handle
///
/// # Transaction Semantics
///
/// This handle provides ACID-like transaction semantics for in-memory storage:
///
/// - **Atomicity**: All operations in a transaction are applied together on commit,
///   or none are applied if the transaction is rolled back or dropped.
/// - **Consistency**: The transaction maintains a consistent view of operations.
/// - **Isolation**: Uncommitted changes are not visible to other readers until commit.
/// - **Durability**: N/A for in-memory storage (see file/SQL storage for persistence).
///
/// ## Usage
///
/// ```rust,no_run
/// # use synaptic::memory::storage::{Storage, TransactionalStorage};
/// # use synaptic::memory::storage::memory::MemoryStorage;
/// # use synaptic::memory::types::{MemoryEntry, MemoryType};
/// # use std::sync::Arc;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let storage = Arc::new(MemoryStorage::new());
///
/// // Begin a transaction
/// let mut transaction = storage.begin_transaction().await?;
///
/// // Perform operations
/// let entry = MemoryEntry::new("key".to_string(), "value".to_string(), MemoryType::ShortTerm);
/// transaction.store("key", &entry).await?;
/// transaction.update("other_key", &entry).await?;
/// transaction.delete("old_key").await?;
///
/// // Commit all operations atomically
/// transaction.commit().await?;
/// # Ok(())
/// # }
/// ```
///
/// ## Important Notes
///
/// - The transaction holds a reference to the live storage's entry map.
/// - Changes are buffered in-memory until commit.
/// - Dropping a transaction without calling commit() or rollback() discards all changes.
/// - Concurrent transactions are supported via DashMap's thread-safe operations.
pub struct MemoryTransactionHandle {
    /// Reference to the live storage's entry map (not a copy)
    entries: Arc<DashMap<String, MemoryEntry>>,
    /// Buffered operations to apply on commit
    operations: Vec<(String, Option<MemoryEntry>)>, // (key, entry) - None means delete
    /// Whether the transaction has been committed or rolled back
    committed: bool,
}

impl MemoryTransactionHandle {
    fn new(entries: Arc<DashMap<String, MemoryEntry>>) -> Self {
        Self {
            entries,
            operations: Vec::new(),
            committed: false,
        }
    }
}

#[async_trait]
impl TransactionHandle for MemoryTransactionHandle {
    async fn store(&mut self, key: &str, entry: &MemoryEntry) -> Result<()> {
        self.operations.push((key.to_string(), Some(entry.clone())));
        Ok(())
    }

    async fn update(&mut self, key: &str, entry: &MemoryEntry) -> Result<()> {
        // For in-memory storage, update is the same as store
        self.operations.push((key.to_string(), Some(entry.clone())));
        Ok(())
    }

    async fn delete(&mut self, key: &str) -> Result<()> {
        self.operations.push((key.to_string(), None));
        Ok(())
    }

    async fn commit(mut self: Box<Self>) -> Result<()> {
        for (key, entry_opt) in &self.operations {
            match entry_opt {
                Some(entry) => {
                    self.entries.insert(key.clone(), entry.clone());
                }
                None => {
                    self.entries.remove(key);
                }
            }
        }
        self.committed = true;
        Ok(())
    }

    async fn rollback(mut self: Box<Self>) -> Result<()> {
        // For in-memory storage, rollback is just clearing the operations
        self.operations.clear();
        self.committed = true;
        Ok(())
    }
}

#[async_trait]
impl TransactionalStorage for MemoryStorage {
    async fn execute_transaction(&self, transaction: StorageTransaction) -> Result<()> {
        // Execute all operations atomically
        for operation in transaction.operations() {
            match operation {
                crate::memory::storage::StorageOperation::Store { key, entry } => {
                    self.entries.insert(key.clone(), entry.clone());
                }
                crate::memory::storage::StorageOperation::Update { key, entry } => {
                    self.entries.insert(key.clone(), entry.clone());
                }
                crate::memory::storage::StorageOperation::Delete { key } => {
                    self.entries.remove(key);
                }
            }
        }
        self.update_stats();
        Ok(())
    }

    async fn begin_transaction(&self) -> Result<Box<dyn TransactionHandle>> {
        Ok(Box::new(MemoryTransactionHandle::new(Arc::clone(&self.entries))))
    }
}
