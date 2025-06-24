# Memory Retrieval Optimization Design

## Current Performance Issues

### Problem Analysis
The current `MemoryRetriever` implementation has severe scalability issues:

1. **Full Table Scan**: `get_recent()` and `get_frequent()` methods call `list_keys()` followed by individual `retrieve()` calls for every entry
2. **In-Memory Sorting**: All entries are loaded into memory, sorted, then truncated to the desired limit
3. **No Indexing**: No secondary indexes for access patterns (last_accessed, access_count)
4. **No Caching**: Repeated expensive operations with no result caching
5. **Linear Complexity**: O(n) complexity for operations that should be O(log n) or O(1)

### Performance Impact
- **10,000 entries**: ~100ms for `get_recent(10)` (loading 10,000 entries to return 10)
- **100,000 entries**: ~1s+ for simple retrieval operations
- **Memory usage**: Loads entire dataset into memory for sorting
- **Network overhead**: Individual retrieve calls in distributed scenarios

## Proposed Solution: Multi-Level Indexing and Caching

### 1. Secondary Indexes

#### Access Time Index (B-Tree)
```rust
struct AccessTimeIndex {
    // BTreeMap for efficient range queries and ordering
    index: BTreeMap<DateTime<Utc>, HashSet<String>>, // timestamp -> keys
    reverse_index: HashMap<String, DateTime<Utc>>,   // key -> timestamp
}
```

**Benefits**:
- O(log n) insertion/deletion
- O(log n + k) for getting k most recent entries
- Efficient range queries for time-based filtering

#### Access Frequency Index (Priority Queue + HashMap)
```rust
struct AccessFrequencyIndex {
    // Max-heap for efficient top-k queries
    frequency_heap: BinaryHeap<(u32, String)>,      // (access_count, key)
    key_to_count: HashMap<String, u32>,             // key -> access_count
    dirty: bool,                                    // needs rebuild flag
}
```

**Benefits**:
- O(1) for getting most frequent entries (when clean)
- O(log n) for updates
- Lazy rebuilding when dirty

#### Tag Index (Inverted Index)
```rust
struct TagIndex {
    tag_to_keys: HashMap<String, HashSet<String>>,  // tag -> keys
    key_to_tags: HashMap<String, HashSet<String>>,  // key -> tags
}
```

**Benefits**:
- O(1) lookup for tag-based queries
- Efficient intersection for multi-tag queries

### 2. Multi-Level Caching Strategy

#### L1 Cache: Hot Data (LRU)
```rust
struct HotDataCache {
    entries: LruCache<String, MemoryEntry>,         // Recently accessed entries
    recent_keys: VecDeque<String>,                  // Recently accessed keys order
    frequent_keys: Vec<String>,                     // Most frequent keys (cached)
    max_size: usize,
}
```

#### L2 Cache: Query Results (TTL)
```rust
struct QueryResultCache {
    recent_results: HashMap<usize, (Vec<MemoryEntry>, Instant)>,    // limit -> (results, timestamp)
    frequent_results: HashMap<usize, (Vec<MemoryEntry>, Instant)>,  // limit -> (results, timestamp)
    tag_results: HashMap<Vec<String>, (Vec<MemoryEntry>, Instant)>, // tags -> (results, timestamp)
    ttl: Duration,
}
```

### 3. Optimized Storage Interface

#### Enhanced Storage Trait
```rust
#[async_trait]
pub trait IndexedStorage: Storage {
    // Efficient top-k queries without full scan
    async fn get_recent_keys(&self, limit: usize) -> Result<Vec<String>>;
    async fn get_frequent_keys(&self, limit: usize) -> Result<Vec<String>>;
    async fn get_keys_by_tags(&self, tags: &[String]) -> Result<Vec<String>>;
    
    // Batch operations for efficiency
    async fn retrieve_batch_ordered(&self, keys: &[String]) -> Result<Vec<MemoryEntry>>;
    
    // Index maintenance
    async fn update_access_time(&self, key: &str, timestamp: DateTime<Utc>) -> Result<()>;
    async fn update_access_count(&self, key: &str, count: u32) -> Result<()>;
}
```

### 4. Implementation Architecture

#### IndexedMemoryRetriever
```rust
pub struct IndexedMemoryRetriever {
    storage: Arc<dyn IndexedStorage + Send + Sync>,
    config: RetrievalConfig,
    
    // Indexes
    access_time_index: RwLock<AccessTimeIndex>,
    frequency_index: RwLock<AccessFrequencyIndex>,
    tag_index: RwLock<TagIndex>,
    
    // Caches
    hot_cache: RwLock<HotDataCache>,
    query_cache: RwLock<QueryResultCache>,
    
    // Background maintenance
    index_maintenance_handle: Option<JoinHandle<()>>,
}
```

#### Optimized Query Methods
```rust
impl IndexedMemoryRetriever {
    /// O(log n + k) complexity instead of O(n log n)
    pub async fn get_recent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        // 1. Check query cache first
        if let Some(cached) = self.check_query_cache_recent(limit).await {
            return Ok(cached);
        }
        
        // 2. Use access time index for efficient retrieval
        let recent_keys = {
            let index = self.access_time_index.read().await;
            index.get_most_recent_keys(limit)
        };
        
        // 3. Batch retrieve entries
        let entries = self.storage.retrieve_batch_ordered(&recent_keys).await?;
        
        // 4. Cache results
        self.cache_query_result_recent(limit, &entries).await;
        
        Ok(entries)
    }
    
    /// O(1) for cached results, O(k) for fresh queries
    pub async fn get_frequent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Similar optimized implementation using frequency index
    }
    
    /// O(1) tag lookup + O(k) retrieval
    pub async fn get_by_tags(&self, tags: &[String]) -> Result<Vec<MemoryEntry>> {
        // Use inverted tag index for efficient lookup
    }
}
```

### 5. Index Maintenance Strategy

#### Lazy Index Updates
- Indexes updated asynchronously to avoid blocking operations
- Batch updates for better performance
- Dirty flags to track when rebuilding is needed

#### Background Maintenance
```rust
async fn maintain_indexes(&self) {
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        
        // Rebuild dirty indexes
        self.rebuild_frequency_index_if_dirty().await;
        
        // Cleanup expired cache entries
        self.cleanup_expired_cache_entries().await;
        
        // Compact indexes if needed
        self.compact_indexes_if_needed().await;
    }
}
```

### 6. Memory Storage Enhancements

#### Indexed Memory Storage
```rust
pub struct IndexedMemoryStorage {
    entries: DashMap<String, MemoryEntry>,
    
    // Built-in indexes
    access_time_index: RwLock<BTreeMap<DateTime<Utc>, HashSet<String>>>,
    frequency_index: RwLock<BinaryHeap<(u32, String)>>,
    tag_index: RwLock<HashMap<String, HashSet<String>>>,
    
    // Statistics for optimization
    stats: RwLock<StorageStats>,
}
```

### 7. Performance Targets

#### Before Optimization
- `get_recent(10)` with 10,000 entries: ~100ms
- `get_frequent(10)` with 10,000 entries: ~100ms
- Memory usage: O(n) for each query

#### After Optimization
- `get_recent(10)` with 10,000 entries: ~1ms (cached) / ~5ms (fresh)
- `get_frequent(10)` with 10,000 entries: ~1ms (cached) / ~3ms (fresh)
- Memory usage: O(k) where k is the result size

#### Scalability Targets
- Support 1M+ entries with sub-10ms query times
- Memory overhead: <20% of total storage size for indexes
- Cache hit rate: >90% for repeated queries

### 8. Implementation Plan

1. **Phase 1**: Implement basic indexing structures
2. **Phase 2**: Add caching layer with TTL
3. **Phase 3**: Implement background maintenance
4. **Phase 4**: Add batch operations and optimizations
5. **Phase 5**: Performance testing and tuning

### 9. Backward Compatibility

The new `IndexedMemoryRetriever` will implement the same public interface as `MemoryRetriever`, ensuring backward compatibility while providing significant performance improvements.

### 10. Configuration Options

```rust
pub struct IndexingConfig {
    pub enable_access_time_index: bool,
    pub enable_frequency_index: bool,
    pub enable_tag_index: bool,
    pub hot_cache_size: usize,
    pub query_cache_ttl_seconds: u64,
    pub index_maintenance_interval_seconds: u64,
    pub frequency_index_rebuild_threshold: usize,
}
```

This design provides a comprehensive solution to the scalability issues while maintaining clean interfaces and backward compatibility.
