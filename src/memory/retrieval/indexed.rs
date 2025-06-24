//! Optimized memory retrieval with indexing and caching
//!
//! This module provides high-performance memory retrieval operations using
//! secondary indexes and multi-level caching to avoid expensive full table scans.

use crate::{
    error::Result,
    memory::{
        types::MemoryEntry,
        storage::Storage,
        retrieval::RetrievalConfig,
    },
};
use std::{
    collections::{BTreeMap, HashMap, HashSet, BinaryHeap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use dashmap::DashMap;

/// Configuration for indexing and caching behavior
#[derive(Debug, Clone)]
pub struct IndexingConfig {
    pub enable_access_time_index: bool,
    pub enable_frequency_index: bool,
    pub enable_tag_index: bool,
    pub hot_cache_size: usize,
    pub query_cache_ttl_seconds: u64,
    pub index_maintenance_interval_seconds: u64,
    pub frequency_index_rebuild_threshold: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            enable_access_time_index: true,
            enable_frequency_index: true,
            enable_tag_index: true,
            hot_cache_size: 1000,
            query_cache_ttl_seconds: 300, // 5 minutes
            index_maintenance_interval_seconds: 60,
            frequency_index_rebuild_threshold: 100,
        }
    }
}

/// Index for efficient access time-based queries
#[derive(Debug, Default)]
pub struct AccessTimeIndex {
    /// BTreeMap for efficient range queries and ordering
    /// Maps timestamp to set of keys with that timestamp
    index: BTreeMap<DateTime<Utc>, HashSet<String>>,
    /// Reverse mapping for efficient updates
    reverse_index: HashMap<String, DateTime<Utc>>,
}

impl AccessTimeIndex {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add or update an entry in the index
    pub fn update(&mut self, key: String, timestamp: DateTime<Utc>) {
        // Remove old entry if exists
        if let Some(old_timestamp) = self.reverse_index.get(&key) {
            if let Some(keys) = self.index.get_mut(old_timestamp) {
                keys.remove(&key);
                if keys.is_empty() {
                    self.index.remove(old_timestamp);
                }
            }
        }
        
        // Add new entry
        self.index.entry(timestamp).or_default().insert(key.clone());
        self.reverse_index.insert(key, timestamp);
    }
    
    /// Remove an entry from the index
    pub fn remove(&mut self, key: &str) {
        if let Some(timestamp) = self.reverse_index.remove(key) {
            if let Some(keys) = self.index.get_mut(&timestamp) {
                keys.remove(key);
                if keys.is_empty() {
                    self.index.remove(&timestamp);
                }
            }
        }
    }
    
    /// Get the most recently accessed keys
    pub fn get_most_recent_keys(&self, limit: usize) -> Vec<String> {
        let mut result = Vec::new();
        
        // Iterate in reverse order (most recent first)
        for (_, keys) in self.index.iter().rev() {
            for key in keys {
                result.push(key.clone());
                if result.len() >= limit {
                    return result;
                }
            }
        }
        
        result
    }
    
    /// Get keys within a time range
    pub fn get_keys_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<String> {
        let mut result = Vec::new();
        
        for (_timestamp, keys) in self.index.range(start..=end) {
            result.extend(keys.iter().cloned());
        }
        
        result
    }
}

/// Index for efficient frequency-based queries
#[derive(Debug)]
pub struct AccessFrequencyIndex {
    /// Heap for efficient top-k queries (max-heap)
    frequency_heap: BinaryHeap<(u32, String)>,
    /// Quick lookup for access counts
    key_to_count: HashMap<String, u32>,
    /// Flag indicating if heap needs rebuilding
    dirty: bool,
    /// Threshold for triggering rebuild
    rebuild_threshold: usize,
    /// Counter for updates since last rebuild
    updates_since_rebuild: usize,
}

impl AccessFrequencyIndex {
    pub fn new(rebuild_threshold: usize) -> Self {
        Self {
            frequency_heap: BinaryHeap::new(),
            key_to_count: HashMap::new(),
            dirty: false,
            rebuild_threshold,
            updates_since_rebuild: 0,
        }
    }
    
    /// Update access count for a key
    pub fn update(&mut self, key: String, count: u32) {
        let old_count = self.key_to_count.insert(key.clone(), count);
        
        // Mark as dirty if this is a significant change
        if old_count.map_or(true, |old| old != count) {
            self.dirty = true;
            self.updates_since_rebuild += 1;
        }
    }
    
    /// Remove a key from the index
    pub fn remove(&mut self, key: &str) {
        if self.key_to_count.remove(key).is_some() {
            self.dirty = true;
            self.updates_since_rebuild += 1;
        }
    }
    
    /// Get the most frequently accessed keys
    pub fn get_most_frequent_keys(&mut self, limit: usize) -> Vec<String> {
        // Rebuild heap if dirty and threshold exceeded
        if self.dirty && self.updates_since_rebuild >= self.rebuild_threshold {
            self.rebuild_heap();
        }
        
        // Extract top k elements without consuming the heap
        let mut temp_heap = self.frequency_heap.clone();
        let mut result = Vec::new();
        
        for _ in 0..limit {
            if let Some((_, key)) = temp_heap.pop() {
                result.push(key);
            } else {
                break;
            }
        }
        
        result
    }
    
    /// Rebuild the heap from current counts
    fn rebuild_heap(&mut self) {
        self.frequency_heap.clear();
        
        for (key, count) in &self.key_to_count {
            self.frequency_heap.push((*count, key.clone()));
        }
        
        self.dirty = false;
        self.updates_since_rebuild = 0;
    }
    
    /// Get access count for a key
    pub fn get_count(&self, key: &str) -> u32 {
        self.key_to_count.get(key).copied().unwrap_or(0)
    }
}

/// Inverted index for tag-based queries
#[derive(Debug, Default)]
pub struct TagIndex {
    /// Maps tag to set of keys that have that tag
    tag_to_keys: HashMap<String, HashSet<String>>,
    /// Maps key to set of tags it has
    key_to_tags: HashMap<String, HashSet<String>>,
}

impl TagIndex {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update tags for a key
    pub fn update(&mut self, key: String, tags: HashSet<String>) {
        // Remove old associations
        if let Some(old_tags) = self.key_to_tags.get(&key) {
            for tag in old_tags {
                if let Some(keys) = self.tag_to_keys.get_mut(tag) {
                    keys.remove(&key);
                    if keys.is_empty() {
                        self.tag_to_keys.remove(tag);
                    }
                }
            }
        }
        
        // Add new associations
        for tag in &tags {
            self.tag_to_keys.entry(tag.clone()).or_default().insert(key.clone());
        }
        
        self.key_to_tags.insert(key, tags);
    }
    
    /// Remove a key from the index
    pub fn remove(&mut self, key: &str) {
        if let Some(tags) = self.key_to_tags.remove(key) {
            for tag in tags {
                if let Some(keys) = self.tag_to_keys.get_mut(&tag) {
                    keys.remove(key);
                    if keys.is_empty() {
                        self.tag_to_keys.remove(&tag);
                    }
                }
            }
        }
    }
    
    /// Get keys that have any of the specified tags
    pub fn get_keys_with_any_tags(&self, tags: &[String]) -> Vec<String> {
        let mut result = HashSet::new();
        
        for tag in tags {
            if let Some(keys) = self.tag_to_keys.get(tag) {
                result.extend(keys.iter().cloned());
            }
        }
        
        result.into_iter().collect()
    }
    
    /// Get keys that have all of the specified tags
    pub fn get_keys_with_all_tags(&self, tags: &[String]) -> Vec<String> {
        if tags.is_empty() {
            return Vec::new();
        }
        
        // Start with keys that have the first tag
        let mut result: HashSet<String> = if let Some(keys) = self.tag_to_keys.get(&tags[0]) {
            keys.clone()
        } else {
            return Vec::new();
        };
        
        // Intersect with keys that have each subsequent tag
        for tag in &tags[1..] {
            if let Some(keys) = self.tag_to_keys.get(tag) {
                result = result.intersection(keys).cloned().collect();
            } else {
                return Vec::new();
            }
        }
        
        result.into_iter().collect()
    }
}

/// LRU cache for hot data
#[derive(Debug)]
pub struct HotDataCache {
    entries: DashMap<String, (MemoryEntry, Instant)>,
    access_order: RwLock<VecDeque<String>>,
    max_size: usize,
}

impl HotDataCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: DashMap::new(),
            access_order: RwLock::new(VecDeque::new()),
            max_size,
        }
    }
    
    /// Get an entry from cache
    pub async fn get(&self, key: &str) -> Option<MemoryEntry> {
        if let Some(entry) = self.entries.get(key) {
            let (memory_entry, _) = entry.value();
            
            // Update access order
            let mut order = self.access_order.write().await;
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
            }
            order.push_back(key.to_string());
            
            Some(memory_entry.clone())
        } else {
            None
        }
    }
    
    /// Put an entry in cache
    pub async fn put(&self, key: String, entry: MemoryEntry) {
        let now = Instant::now();
        
        // Add to cache
        self.entries.insert(key.clone(), (entry, now));
        
        // Update access order and evict if necessary
        let mut order = self.access_order.write().await;
        if let Some(pos) = order.iter().position(|k| k == &key) {
            order.remove(pos);
        }
        order.push_back(key);
        
        // Evict oldest entries if over capacity
        while order.len() > self.max_size {
            if let Some(oldest_key) = order.pop_front() {
                self.entries.remove(&oldest_key);
            }
        }
    }
    
    /// Remove an entry from cache
    pub async fn remove(&self, key: &str) {
        self.entries.remove(key);
        
        let mut order = self.access_order.write().await;
        if let Some(pos) = order.iter().position(|k| k == key) {
            order.remove(pos);
        }
    }
    
    /// Clear all entries
    pub async fn clear(&self) {
        self.entries.clear();
        self.access_order.write().await.clear();
    }
}

/// Cache for query results with TTL
#[derive(Debug)]
pub struct QueryResultCache {
    recent_results: RwLock<HashMap<usize, (Vec<MemoryEntry>, Instant)>>,
    frequent_results: RwLock<HashMap<usize, (Vec<MemoryEntry>, Instant)>>,
    tag_results: RwLock<HashMap<Vec<String>, (Vec<MemoryEntry>, Instant)>>,
    ttl: Duration,
}

impl QueryResultCache {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            recent_results: RwLock::new(HashMap::new()),
            frequent_results: RwLock::new(HashMap::new()),
            tag_results: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }
    
    /// Get cached recent results
    pub async fn get_recent(&self, limit: usize) -> Option<Vec<MemoryEntry>> {
        let cache = self.recent_results.read().await;
        if let Some((results, timestamp)) = cache.get(&limit) {
            if timestamp.elapsed() < self.ttl {
                return Some(results.clone());
            }
        }
        None
    }
    
    /// Cache recent results
    pub async fn put_recent(&self, limit: usize, results: Vec<MemoryEntry>) {
        let mut cache = self.recent_results.write().await;
        cache.insert(limit, (results, Instant::now()));
    }
    
    /// Get cached frequent results
    pub async fn get_frequent(&self, limit: usize) -> Option<Vec<MemoryEntry>> {
        let cache = self.frequent_results.read().await;
        if let Some((results, timestamp)) = cache.get(&limit) {
            if timestamp.elapsed() < self.ttl {
                return Some(results.clone());
            }
        }
        None
    }
    
    /// Cache frequent results
    pub async fn put_frequent(&self, limit: usize, results: Vec<MemoryEntry>) {
        let mut cache = self.frequent_results.write().await;
        cache.insert(limit, (results, Instant::now()));
    }
    
    /// Clean up expired entries
    pub async fn cleanup_expired(&self) {
        let now = Instant::now();
        
        {
            let mut cache = self.recent_results.write().await;
            cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.ttl);
        }
        
        {
            let mut cache = self.frequent_results.write().await;
            cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.ttl);
        }
        
        {
            let mut cache = self.tag_results.write().await;
            cache.retain(|_, (_, timestamp)| now.duration_since(*timestamp) < self.ttl);
        }
    }
}

/// High-performance memory retriever with indexing and caching
pub struct IndexedMemoryRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    config: RetrievalConfig,
    indexing_config: IndexingConfig,

    // Indexes
    access_time_index: RwLock<AccessTimeIndex>,
    frequency_index: RwLock<AccessFrequencyIndex>,
    tag_index: RwLock<TagIndex>,

    // Caches
    hot_cache: HotDataCache,
    query_cache: QueryResultCache,

    // Background maintenance
    maintenance_handle: RwLock<Option<tokio::task::JoinHandle<()>>>,
}

impl IndexedMemoryRetriever {
    /// Create a new indexed memory retriever
    pub fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        config: RetrievalConfig,
        indexing_config: IndexingConfig,
    ) -> Self {
        let retriever = Self {
            storage,
            config,
            indexing_config: indexing_config.clone(),
            access_time_index: RwLock::new(AccessTimeIndex::new()),
            frequency_index: RwLock::new(AccessFrequencyIndex::new(
                indexing_config.frequency_index_rebuild_threshold,
            )),
            tag_index: RwLock::new(TagIndex::new()),
            hot_cache: HotDataCache::new(indexing_config.hot_cache_size),
            query_cache: QueryResultCache::new(indexing_config.query_cache_ttl_seconds),
            maintenance_handle: RwLock::new(None),
        };

        retriever
    }

    /// Start background maintenance tasks
    pub async fn start_maintenance(&self) {
        let query_cache = Arc::new(QueryResultCache::new(self.indexing_config.query_cache_ttl_seconds));
        let interval = Duration::from_secs(self.indexing_config.index_maintenance_interval_seconds);

        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Cleanup expired cache entries
                query_cache.cleanup_expired().await;

                tracing::debug!("Performed index maintenance");
            }
        });

        *self.maintenance_handle.write().await = Some(handle);
    }

    /// Stop background maintenance
    pub async fn stop_maintenance(&self) {
        if let Some(handle) = self.maintenance_handle.write().await.take() {
            handle.abort();
        }
    }

    /// Get recently accessed memories with O(log n + k) complexity
    pub async fn get_recent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        // 1. Check query cache first
        if let Some(cached) = self.query_cache.get_recent(limit).await {
            tracing::debug!("Cache hit for get_recent({})", limit);
            return Ok(cached);
        }

        // 2. Use access time index for efficient retrieval
        let recent_keys = if self.indexing_config.enable_access_time_index {
            let index = self.access_time_index.read().await;
            index.get_most_recent_keys(limit)
        } else {
            // Fallback to storage-level operation
            self.fallback_get_recent_keys(limit).await?
        };

        // 3. Batch retrieve entries with hot cache
        let mut entries = Vec::new();
        let mut keys_to_fetch = Vec::new();

        // Check hot cache first
        for key in &recent_keys {
            if let Some(entry) = self.hot_cache.get(key).await {
                entries.push(entry);
            } else {
                keys_to_fetch.push(key.clone());
            }
        }

        // Fetch remaining entries from storage
        if !keys_to_fetch.is_empty() {
            for key in &keys_to_fetch {
                if let Some(entry) = self.storage.retrieve(key).await? {
                    // Add to hot cache
                    self.hot_cache.put(key.clone(), entry.clone()).await;
                    entries.push(entry);
                }
            }
        }

        // Sort by access time to maintain order
        entries.sort_by(|a, b| b.last_accessed().cmp(&a.last_accessed()));
        entries.truncate(limit);

        // 4. Cache results
        self.query_cache.put_recent(limit, entries.clone()).await;

        tracing::debug!("Retrieved {} recent entries", entries.len());
        Ok(entries)
    }

    /// Get frequently accessed memories with optimized performance
    pub async fn get_frequent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        // 1. Check query cache first
        if let Some(cached) = self.query_cache.get_frequent(limit).await {
            tracing::debug!("Cache hit for get_frequent({})", limit);
            return Ok(cached);
        }

        // 2. Use frequency index for efficient retrieval
        let frequent_keys = if self.indexing_config.enable_frequency_index {
            let mut index = self.frequency_index.write().await;
            index.get_most_frequent_keys(limit)
        } else {
            // Fallback to storage-level operation
            self.fallback_get_frequent_keys(limit).await?
        };

        // 3. Batch retrieve entries with hot cache
        let mut entries = Vec::new();

        for key in &frequent_keys {
            if let Some(entry) = self.hot_cache.get(key).await {
                entries.push(entry);
            } else if let Some(entry) = self.storage.retrieve(key).await? {
                // Add to hot cache
                self.hot_cache.put(key.clone(), entry.clone()).await;
                entries.push(entry);
            }
        }

        // Sort by access count to maintain order
        entries.sort_by(|a, b| b.access_count().cmp(&a.access_count()));
        entries.truncate(limit);

        // 4. Cache results
        self.query_cache.put_frequent(limit, entries.clone()).await;

        tracing::debug!("Retrieved {} frequent entries", entries.len());
        Ok(entries)
    }

    /// Get memories by tags with O(1) tag lookup
    pub async fn get_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryEntry>> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }

        // Use tag index for efficient lookup
        let keys = if self.indexing_config.enable_tag_index {
            let index = self.tag_index.read().await;
            index.get_keys_with_any_tags(tags)
        } else {
            // Fallback to full scan
            self.fallback_get_keys_by_tags(tags).await?
        };

        // Batch retrieve entries
        let mut entries = Vec::new();
        for key in &keys {
            if let Some(entry) = self.hot_cache.get(key).await {
                entries.push(entry);
            } else if let Some(entry) = self.storage.retrieve(key).await? {
                // Add to hot cache
                self.hot_cache.put(key.clone(), entry.clone()).await;
                entries.push(entry);
            }
        }

        tracing::debug!("Retrieved {} entries by tags {:?}", entries.len(), tags);
        Ok(entries)
    }

    /// Update indexes when an entry is stored
    pub async fn on_entry_stored(&self, entry: &MemoryEntry) {
        if self.indexing_config.enable_access_time_index {
            let mut index = self.access_time_index.write().await;
            index.update(entry.key.clone(), entry.last_accessed());
        }

        if self.indexing_config.enable_frequency_index {
            let mut index = self.frequency_index.write().await;
            index.update(entry.key.clone(), entry.access_count() as u32);
        }

        if self.indexing_config.enable_tag_index {
            let mut index = self.tag_index.write().await;
            let tags: HashSet<String> = entry.tags().iter().cloned().collect();
            index.update(entry.key.clone(), tags);
        }

        // Invalidate relevant caches
        self.invalidate_caches().await;
    }

    /// Update indexes when an entry is deleted
    pub async fn on_entry_deleted(&self, key: &str) {
        if self.indexing_config.enable_access_time_index {
            let mut index = self.access_time_index.write().await;
            index.remove(key);
        }

        if self.indexing_config.enable_frequency_index {
            let mut index = self.frequency_index.write().await;
            index.remove(key);
        }

        if self.indexing_config.enable_tag_index {
            let mut index = self.tag_index.write().await;
            index.remove(key);
        }

        // Remove from hot cache
        self.hot_cache.remove(key).await;

        // Invalidate relevant caches
        self.invalidate_caches().await;
    }

    /// Invalidate query caches
    async fn invalidate_caches(&self) {
        // For now, we'll clear all caches on any change
        // In a more sophisticated implementation, we could be more selective
        *self.query_cache.recent_results.write().await = HashMap::new();
        *self.query_cache.frequent_results.write().await = HashMap::new();
        *self.query_cache.tag_results.write().await = HashMap::new();
    }

    /// Fallback method for getting recent keys without index
    async fn fallback_get_recent_keys(&self, limit: usize) -> Result<Vec<String>> {
        let keys = self.storage.list_keys().await?;
        let mut entries = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                entries.push(entry);
            }
        }

        entries.sort_by(|a, b| b.last_accessed().cmp(&a.last_accessed()));
        entries.truncate(limit);

        Ok(entries.into_iter().map(|e| e.key).collect())
    }

    /// Fallback method for getting frequent keys without index
    async fn fallback_get_frequent_keys(&self, limit: usize) -> Result<Vec<String>> {
        let keys = self.storage.list_keys().await?;
        let mut entries = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                entries.push(entry);
            }
        }

        entries.sort_by(|a, b| b.access_count().cmp(&a.access_count()));
        entries.truncate(limit);

        Ok(entries.into_iter().map(|e| e.key).collect())
    }

    /// Fallback method for getting keys by tags without index
    async fn fallback_get_keys_by_tags(&self, tags: &[String]) -> Result<Vec<String>> {
        let keys = self.storage.list_keys().await?;
        let mut result = Vec::new();

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                if tags.iter().any(|tag| entry.has_tag(tag)) {
                    result.push(key);
                }
            }
        }

        Ok(result)
    }

    /// Initialize indexes from existing storage data
    pub async fn initialize_indexes(&self) -> Result<()> {
        tracing::info!("Initializing indexes from storage...");

        let keys = self.storage.list_keys().await?;
        let mut initialized_count = 0;

        for key in keys {
            if let Some(entry) = self.storage.retrieve(&key).await? {
                self.on_entry_stored(&entry).await;
                initialized_count += 1;
            }
        }

        tracing::info!("Initialized indexes with {} entries", initialized_count);
        Ok(())
    }

    /// Get index statistics for monitoring
    pub async fn get_index_stats(&self) -> IndexStats {
        let access_time_entries = self.access_time_index.read().await.reverse_index.len();
        let frequency_entries = self.frequency_index.read().await.key_to_count.len();
        let tag_entries = self.tag_index.read().await.key_to_tags.len();
        let hot_cache_entries = self.hot_cache.entries.len();

        IndexStats {
            access_time_index_entries: access_time_entries,
            frequency_index_entries: frequency_entries,
            tag_index_entries: tag_entries,
            hot_cache_entries,
            hot_cache_max_size: self.hot_cache.max_size,
        }
    }
}

/// Statistics about index usage and performance
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub access_time_index_entries: usize,
    pub frequency_index_entries: usize,
    pub tag_index_entries: usize,
    pub hot_cache_entries: usize,
    pub hot_cache_max_size: usize,
}
