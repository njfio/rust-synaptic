//! File-based storage implementation using Sled embedded database

use crate::error::{MemoryError, Result, MemoryErrorExt};
use crate::memory::storage::{Storage, StorageStats, BatchStorage};
use crate::memory::types::{MemoryEntry, MemoryFragment};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// File-based storage implementation using Sled embedded database
pub struct FileStorage {
    /// Sled database instance
    db: sled::Db,
    /// Storage path
    path: PathBuf,
    /// Statistics cache
    stats_cache: Arc<RwLock<Option<(StorageStats, DateTime<Utc>)>>>,
    /// Cache TTL in seconds
    stats_cache_ttl: u64,
}

impl FileStorage {
    /// Create a new file storage instance
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Ensure the directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await
                .storage_context("Failed to create storage directory")?;
        }

        let db = sled::open(&path)
            .storage_context("Failed to open Sled database")?;

        Ok(Self {
            db,
            path,
            stats_cache: Arc::new(RwLock::new(None)),
            stats_cache_ttl: 60, // 1 minute cache TTL
        })
    }

    /// Serialize a memory entry to bytes
    fn serialize_entry(&self, entry: &MemoryEntry) -> Result<Vec<u8>> {
        bincode::serialize(entry)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize entry: {}", e)))
    }

    /// Deserialize bytes to a memory entry
    fn deserialize_entry(&self, bytes: &[u8]) -> Result<MemoryEntry> {
        bincode::deserialize(bytes)
            .map_err(|e| MemoryError::storage(format!("Failed to deserialize entry: {}", e)))
    }

    /// Update and cache storage statistics
    async fn update_stats_cache(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::new("file".to_string());
        stats.total_entries = self.db.len();
        
        let mut total_size = 0usize;
        for result in self.db.iter() {
            let (key, value) = result.storage_context("Failed to iterate over database")?;
            total_size += key.len() + value.len();
        }
        
        stats.total_size_bytes = total_size;
        stats.average_entry_size = if stats.total_entries > 0 {
            total_size as f64 / stats.total_entries as f64
        } else {
            0.0
        };
        
        stats.last_maintenance = Some(Utc::now());
        
        // Cache the stats
        let mut cache = self.stats_cache.write().await;
        *cache = Some((stats.clone(), Utc::now()));
        
        Ok(stats)
    }

    /// Get cached statistics or compute new ones
    async fn get_stats_cached(&self) -> Result<StorageStats> {
        let cache = self.stats_cache.read().await;
        
        if let Some((stats, cached_at)) = cache.as_ref() {
            let age = Utc::now() - *cached_at;
            if age.num_seconds() < self.stats_cache_ttl as i64 {
                return Ok(stats.clone());
            }
        }
        
        drop(cache);
        self.update_stats_cache().await
    }

    /// Perform simple text-based search across all entries
    async fn search_entries(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for result in self.db.iter() {
            let (_key, value) = result.storage_context("Failed to iterate over database")?;
            let entry = self.deserialize_entry(&value)?;
            
            let content_lower = entry.value.to_lowercase();
            
            // Simple relevance scoring
            let relevance_score = if content_lower.contains(&query_lower) {
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
                let fragment = MemoryFragment::new(entry, relevance_score);
                results.push(fragment);
            }

            if results.len() >= limit {
                break;
            }
        }

        // Sort by relevance score (highest first)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Get the database path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush all pending writes to disk
    pub async fn flush(&self) -> Result<()> {
        self.db.flush_async().await
            .storage_context("Failed to flush database to disk")?;
        Ok(())
    }

    /// Get database size on disk
    pub fn size_on_disk(&self) -> Result<u64> {
        self.db.size_on_disk()
            .storage_context("Failed to get database size")
    }

    /// Export all data to a JSON file
    pub async fn export_to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut entries = Vec::new();
        
        for result in self.db.iter() {
            let (_key, value) = result.storage_context("Failed to iterate over database")?;
            let entry = self.deserialize_entry(&value)?;
            entries.push(entry);
        }

        let json = serde_json::to_string_pretty(&entries)
            .storage_context("Failed to serialize entries to JSON")?;

        tokio::fs::write(path, json).await
            .storage_context("Failed to write JSON export file")?;

        Ok(())
    }

    /// Import data from a JSON file
    pub async fn import_from_json<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let json = tokio::fs::read_to_string(path).await
            .storage_context("Failed to read JSON import file")?;

        let entries: Vec<MemoryEntry> = serde_json::from_str(&json)
            .storage_context("Failed to deserialize JSON data")?;

        let mut imported_count = 0;
        for entry in entries {
            let serialized = self.serialize_entry(&entry)?;
            self.db.insert(&entry.key, serialized)
                .storage_context("Failed to insert imported entry")?;
            imported_count += 1;
        }

        self.db.flush_async().await
            .storage_context("Failed to flush imported data")?;

        // Invalidate stats cache
        let mut cache = self.stats_cache.write().await;
        *cache = None;

        Ok(imported_count)
    }
}

#[async_trait]
impl Storage for FileStorage {
    async fn store(&self, entry: &MemoryEntry) -> Result<()> {
        let serialized = self.serialize_entry(entry)?;
        self.db.insert(&entry.key, serialized)
            .storage_context("Failed to store entry")?;
        
        // Invalidate stats cache
        let mut cache = self.stats_cache.write().await;
        *cache = None;
        
        Ok(())
    }

    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>> {
        match self.db.get(key).storage_context("Failed to retrieve entry")? {
            Some(bytes) => Ok(Some(self.deserialize_entry(&bytes)?)),
            None => Ok(None),
        }
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        self.search_entries(query, limit).await
    }

    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()> {
        if self.db.contains_key(key).storage_context("Failed to check key existence")? {
            let serialized = self.serialize_entry(entry)?;
            self.db.insert(key, serialized)
                .storage_context("Failed to update entry")?;
            
            // Invalidate stats cache
            let mut cache = self.stats_cache.write().await;
            *cache = None;
            
            Ok(())
        } else {
            Err(MemoryError::NotFound {
                key: key.to_string(),
            })
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let removed = self.db.remove(key)
            .storage_context("Failed to delete entry")?
            .is_some();
        
        if removed {
            // Invalidate stats cache
            let mut cache = self.stats_cache.write().await;
            *cache = None;
        }
        
        Ok(removed)
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        for result in self.db.iter() {
            let (key, _) = result.storage_context("Failed to iterate over keys")?;
            keys.push(String::from_utf8_lossy(&key).to_string());
        }
        Ok(keys)
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.db.len())
    }

    async fn clear(&self) -> Result<()> {
        self.db.clear().storage_context("Failed to clear database")?;
        
        // Invalidate stats cache
        let mut cache = self.stats_cache.write().await;
        *cache = None;
        
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.db.contains_key(key)
            .storage_context("Failed to check key existence")
    }

    async fn stats(&self) -> Result<StorageStats> {
        self.get_stats_cached().await
    }

    async fn maintenance(&self) -> Result<()> {
        // Perform database maintenance
        self.db.flush_async().await
            .storage_context("Failed to flush database during maintenance")?;
        
        // Update stats cache
        self.update_stats_cache().await?;
        
        Ok(())
    }

    async fn backup(&self, path: &str) -> Result<()> {
        self.export_to_json(path).await
    }

    async fn restore(&self, path: &str) -> Result<()> {
        self.import_from_json(path).await?;
        Ok(())
    }

    async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>> {
        let mut entries = Vec::new();

        for result in self.db.iter() {
            let (_key, value) = result.storage_context("Failed to iterate over database")?;
            let entry = self.deserialize_entry(&value)?;
            entries.push(entry);
        }

        Ok(entries)
    }
}

#[async_trait]
impl BatchStorage for FileStorage {
    async fn store_batch(&self, entries: &[MemoryEntry]) -> Result<()> {
        let mut batch = sled::Batch::default();
        
        for entry in entries {
            let serialized = self.serialize_entry(entry)?;
            batch.insert(entry.key.as_bytes(), serialized);
        }
        
        self.db.apply_batch(batch)
            .storage_context("Failed to apply batch store operation")?;
        
        // Invalidate stats cache
        let mut cache = self.stats_cache.write().await;
        *cache = None;
        
        Ok(())
    }

    async fn retrieve_batch(&self, keys: &[String]) -> Result<Vec<Option<MemoryEntry>>> {
        let mut results = Vec::with_capacity(keys.len());
        
        for key in keys {
            match self.db.get(key).storage_context("Failed to retrieve entry in batch")? {
                Some(bytes) => results.push(Some(self.deserialize_entry(&bytes)?)),
                None => results.push(None),
            }
        }
        
        Ok(results)
    }

    async fn delete_batch(&self, keys: &[String]) -> Result<usize> {
        let mut batch = sled::Batch::default();
        let mut deleted_count = 0;
        
        for key in keys {
            if self.db.contains_key(key).storage_context("Failed to check key existence")? {
                batch.remove(key.as_bytes());
                deleted_count += 1;
            }
        }
        
        if deleted_count > 0 {
            self.db.apply_batch(batch)
                .storage_context("Failed to apply batch delete operation")?;
            
            // Invalidate stats cache
            let mut cache = self.stats_cache.write().await;
            *cache = None;
        }
        
        Ok(deleted_count)
    }
}
