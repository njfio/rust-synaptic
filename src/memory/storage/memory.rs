//! In-memory storage implementation for the memory system

use crate::error::{MemoryError, Result};
use crate::memory::storage::{Storage, StorageStats, BatchStorage, StorageTransaction, TransactionalStorage, TransactionHandle};
use crate::memory::types::{MemoryEntry, MemoryFragment};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// In-memory storage implementation using DashMap for thread-safe concurrent access
pub struct MemoryStorage {
    /// Main storage for memory entries
    entries: DashMap<String, MemoryEntry>,
    /// Statistics tracking
    stats: RwLock<StorageStats>,
    /// Creation timestamp
    created_at: DateTime<Utc>,
}

impl MemoryStorage {
    /// Create a new in-memory storage instance
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
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
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
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
        let result = self.entries.get(key).map(|entry| entry.value().clone());
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

    async fn backup(&self, _path: &str) -> Result<()> {
        // In-memory storage cannot be backed up to disk
        // This would require serialization to a file
        Err(MemoryError::storage(
            "In-memory storage does not support backup operations"
        ))
    }

    async fn restore(&self, _path: &str) -> Result<()> {
        // In-memory storage cannot restore from disk
        Err(MemoryError::storage(
            "In-memory storage does not support restore operations"
        ))
    }
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
pub struct MemoryTransactionHandle {
    storage: Arc<MemoryStorage>,
    operations: Vec<(String, Option<MemoryEntry>)>, // (key, entry) - None means delete
    committed: bool,
}

impl MemoryTransactionHandle {
    fn new(storage: Arc<MemoryStorage>) -> Self {
        Self {
            storage,
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
                    self.storage.entries.insert(key.clone(), entry.clone());
                }
                None => {
                    self.storage.entries.remove(key);
                }
            }
        }
        self.storage.update_stats();
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
        Ok(Box::new(MemoryTransactionHandle::new(Arc::new(
            MemoryStorage::new()
        ))))
    }
}
