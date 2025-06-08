// Real Redis Cache Integration
// Implements actual Redis caching for memory entries and analytics data

use crate::error::{Result, MemoryError};
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,
    /// Connection pool size
    pub pool_size: u32,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Default TTL for cached entries in seconds
    pub default_ttl_secs: u64,
    /// Key prefix for namespacing
    pub key_prefix: String,
    /// Enable compression
    pub compression: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:11111".to_string()),
            pool_size: 10,
            connect_timeout_secs: 5,
            default_ttl_secs: 3600, // 1 hour
            key_prefix: "synaptic:".to_string(),
            compression: true,
        }
    }
}

/// Real Redis client
#[derive(Debug)]
pub struct RedisClient {
    config: RedisConfig,
    #[cfg(feature = "distributed")]
    connection_manager: Option<redis::aio::ConnectionManager>,
    metrics: RedisMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct RedisMetrics {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_sets: u64,
    pub cache_deletes: u64,
    pub total_operations: u64,
    pub avg_response_time_ms: f64,
    pub connection_errors: u64,
}

impl RedisClient {
    /// Create a new Redis client with real connection
    pub async fn new(config: RedisConfig) -> Result<Self> {
        #[cfg(feature = "distributed")]
        {
            let client = redis::Client::open(config.url.as_str())
                .map_err(|e| MemoryError::storage(format!("Failed to create Redis client: {}", e)))?;

            let connection_manager = redis::aio::ConnectionManager::new(client).await
                .map_err(|e| MemoryError::storage(format!("Failed to connect to Redis: {}", e)))?;

            Ok(Self {
                config,
                connection_manager: Some(connection_manager),
                metrics: RedisMetrics::default(),
            })
        }

        #[cfg(not(feature = "distributed"))]
        {
            Ok(Self {
                config,
                metrics: RedisMetrics::default(),
            })
        }
    }

    /// Cache a memory entry
    #[cfg(feature = "distributed")]
    pub async fn cache_memory(&mut self, key: &str, entry: &MemoryEntry, ttl_secs: Option<u64>) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}memory:{}", self.config.key_prefix, key);
        let ttl = ttl_secs.unwrap_or(self.config.default_ttl_secs);

        // Serialize the memory entry
        let serialized = if self.config.compression {
            self.compress_and_serialize(entry)?
        } else {
            serde_json::to_vec(entry)
                .map_err(|e| MemoryError::storage(format!("Failed to serialize memory entry: {}", e)))?
        };

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            conn.set_ex(&cache_key, serialized, ttl as usize).await
                .map_err(|e| MemoryError::storage(format!("Failed to cache memory entry: {}", e)))?;
        }

        self.update_metrics(start_time, CacheOperation::Set);
        Ok(())
    }

    /// Retrieve a cached memory entry
    #[cfg(feature = "distributed")]
    pub async fn get_cached_memory(&mut self, key: &str) -> Result<Option<MemoryEntry>> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}memory:{}", self.config.key_prefix, key);

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let cached_data: Option<Vec<u8>> = conn.get(&cache_key).await
                .map_err(|e| MemoryError::storage(format!("Failed to get cached memory: {}", e)))?;

            if let Some(data) = cached_data {
                let entry = if self.config.compression {
                    self.decompress_and_deserialize(&data)?
                } else {
                    serde_json::from_slice(&data)
                        .map_err(|e| MemoryError::storage(format!("Failed to deserialize memory entry: {}", e)))?
                };

                self.update_metrics(start_time, CacheOperation::Hit);
                return Ok(Some(entry));
            }
        }

        self.update_metrics(start_time, CacheOperation::Miss);
        Ok(None)
    }

    /// Cache analytics data
    #[cfg(feature = "distributed")]
    pub async fn cache_analytics(&mut self, key: &str, data: &serde_json::Value, ttl_secs: Option<u64>) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}analytics:{}", self.config.key_prefix, key);
        let ttl = ttl_secs.unwrap_or(self.config.default_ttl_secs);

        let serialized = serde_json::to_vec(data)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize analytics data: {}", e)))?;

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            conn.set_ex(&cache_key, serialized, ttl as usize).await
                .map_err(|e| MemoryError::storage(format!("Failed to cache analytics data: {}", e)))?;
        }

        self.update_metrics(start_time, CacheOperation::Set);
        Ok(())
    }

    /// Get cached analytics data
    #[cfg(feature = "distributed")]
    pub async fn get_cached_analytics(&mut self, key: &str) -> Result<Option<serde_json::Value>> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}analytics:{}", self.config.key_prefix, key);

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let cached_data: Option<Vec<u8>> = conn.get(&cache_key).await
                .map_err(|e| MemoryError::storage(format!("Failed to get cached analytics: {}", e)))?;

            if let Some(data) = cached_data {
                let value = serde_json::from_slice(&data)
                    .map_err(|e| MemoryError::storage(format!("Failed to deserialize analytics data: {}", e)))?;

                self.update_metrics(start_time, CacheOperation::Hit);
                return Ok(Some(value));
            }
        }

        self.update_metrics(start_time, CacheOperation::Miss);
        Ok(None)
    }

    /// Delete cached entry
    #[cfg(feature = "distributed")]
    pub async fn delete_cached(&mut self, key: &str) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}*:{}", self.config.key_prefix, key);

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            // Get all keys matching the pattern
            let keys: Vec<String> = conn.keys(&cache_key).await
                .map_err(|e| MemoryError::storage(format!("Failed to get keys for deletion: {}", e)))?;

            if !keys.is_empty() {
                conn.del(&keys).await
                    .map_err(|e| MemoryError::storage(format!("Failed to delete cached entries: {}", e)))?;
            }
        }

        self.update_metrics(start_time, CacheOperation::Delete);
        Ok(())
    }

    /// Cache embeddings
    #[cfg(feature = "distributed")]
    pub async fn cache_embedding(&mut self, text: &str, embedding: &[f32], ttl_secs: Option<u64>) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}embedding:{}", self.config.key_prefix, self.hash_text(text));
        let ttl = ttl_secs.unwrap_or(self.config.default_ttl_secs * 24); // Embeddings last longer

        let serialized = serde_json::to_vec(embedding)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize embedding: {}", e)))?;

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            conn.set_ex(&cache_key, serialized, ttl as usize).await
                .map_err(|e| MemoryError::storage(format!("Failed to cache embedding: {}", e)))?;
        }

        self.update_metrics(start_time, CacheOperation::Set);
        Ok(())
    }

    /// Get cached embedding
    #[cfg(feature = "distributed")]
    pub async fn get_cached_embedding(&mut self, text: &str) -> Result<Option<Vec<f32>>> {
        let start_time = std::time::Instant::now();
        
        let cache_key = format!("{}embedding:{}", self.config.key_prefix, self.hash_text(text));

        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let cached_data: Option<Vec<u8>> = conn.get(&cache_key).await
                .map_err(|e| MemoryError::storage(format!("Failed to get cached embedding: {}", e)))?;

            if let Some(data) = cached_data {
                let embedding = serde_json::from_slice(&data)
                    .map_err(|e| MemoryError::storage(format!("Failed to deserialize embedding: {}", e)))?;

                self.update_metrics(start_time, CacheOperation::Hit);
                return Ok(Some(embedding));
            }
        }

        self.update_metrics(start_time, CacheOperation::Miss);
        Ok(None)
    }

    /// Get cache statistics
    #[cfg(feature = "distributed")]
    pub async fn get_cache_stats(&mut self) -> Result<CacheStats> {
        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let info: String = conn.info("memory").await
                .map_err(|e| MemoryError::storage(format!("Failed to get Redis info: {}", e)))?;

            // Parse Redis memory info
            let mut used_memory = 0u64;
            let mut max_memory = 0u64;
            
            for line in info.lines() {
                if line.starts_with("used_memory:") {
                    used_memory = line.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
                } else if line.starts_with("maxmemory:") {
                    max_memory = line.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
                }
            }

            let hit_rate = if self.metrics.cache_hits + self.metrics.cache_misses > 0 {
                self.metrics.cache_hits as f64 / (self.metrics.cache_hits + self.metrics.cache_misses) as f64
            } else {
                0.0
            };

            return Ok(CacheStats {
                hit_rate,
                used_memory_bytes: used_memory,
                max_memory_bytes: max_memory,
                total_keys: self.get_key_count().await?,
                evictions: 0, // Would need to parse from Redis info
            });
        }

        Ok(CacheStats::default())
    }

    #[cfg(feature = "distributed")]
    async fn get_key_count(&mut self) -> Result<u64> {
        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let keys: Vec<String> = conn.keys(format!("{}*", self.config.key_prefix)).await
                .map_err(|e| MemoryError::storage(format!("Failed to count keys: {}", e)))?;
            
            return Ok(keys.len() as u64);
        }
        Ok(0)
    }

    /// Compress and serialize data
    #[cfg(feature = "distributed")]
    fn compress_and_serialize(&self, entry: &MemoryEntry) -> Result<Vec<u8>> {
        let serialized = serde_json::to_vec(entry)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize for compression: {}", e)))?;

        // For now, just return serialized data without compression
        // In production, you would use actual LZ4 compression
        Ok(serialized)
    }

    /// Decompress and deserialize data
    #[cfg(feature = "distributed")]
    fn decompress_and_deserialize(&self, data: &[u8]) -> Result<MemoryEntry> {
        // For now, just deserialize directly
        // In production, you would decompress first
        let entry = serde_json::from_slice(data)
            .map_err(|e| MemoryError::storage(format!("Failed to deserialize data: {}", e)))?;

        Ok(entry)
    }

    /// Hash text for consistent cache keys
    fn hash_text(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Health check for Redis connection
    pub async fn health_check(&self) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            if let Some(ref conn) = self.connection_manager {
                use redis::AsyncCommands;
                let mut conn_clone = conn.clone();
                
                let _: String = conn_clone.ping().await
                    .map_err(|e| MemoryError::storage(format!("Redis health check failed: {}", e)))?;
            }
        }
        Ok(())
    }

    /// Shutdown Redis connection
    pub async fn shutdown(&mut self) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            self.connection_manager = None;
        }
        Ok(())
    }

    /// Get Redis metrics
    pub fn get_metrics(&self) -> &RedisMetrics {
        &self.metrics
    }

    /// Clear all cached data
    #[cfg(feature = "distributed")]
    pub async fn clear_cache(&mut self) -> Result<()> {
        if let Some(ref mut conn) = self.connection_manager {
            use redis::AsyncCommands;
            
            let keys: Vec<String> = conn.keys(format!("{}*", self.config.key_prefix)).await
                .map_err(|e| MemoryError::storage(format!("Failed to get keys for clearing: {}", e)))?;

            if !keys.is_empty() {
                conn.del(&keys).await
                    .map_err(|e| MemoryError::storage(format!("Failed to clear cache: {}", e)))?;
            }
        }
        Ok(())
    }

    fn update_metrics(&mut self, start_time: std::time::Instant, operation: CacheOperation) {
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        
        self.metrics.total_operations += 1;
        self.metrics.avg_response_time_ms = 
            (self.metrics.avg_response_time_ms * (self.metrics.total_operations - 1) as f64 + elapsed_ms) 
            / self.metrics.total_operations as f64;

        match operation {
            CacheOperation::Hit => self.metrics.cache_hits += 1,
            CacheOperation::Miss => self.metrics.cache_misses += 1,
            CacheOperation::Set => self.metrics.cache_sets += 1,
            CacheOperation::Delete => self.metrics.cache_deletes += 1,
        }
    }
}

#[derive(Debug)]
enum CacheOperation {
    Hit,
    Miss,
    Set,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub hit_rate: f64,
    pub used_memory_bytes: u64,
    pub max_memory_bytes: u64,
    pub total_keys: u64,
    pub evictions: u64,
}

// Stub implementations for when distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
impl RedisClient {
    pub async fn cache_memory(&mut self, _key: &str, _entry: &MemoryEntry, _ttl_secs: Option<u64>) -> Result<()> {
        Err(MemoryError::configuration("Redis feature not enabled"))
    }

    pub async fn get_cached_memory(&mut self, _key: &str) -> Result<Option<MemoryEntry>> {
        Ok(None)
    }

    pub async fn cache_analytics(&mut self, _key: &str, _data: &serde_json::Value, _ttl_secs: Option<u64>) -> Result<()> {
        Err(MemoryError::configuration("Redis feature not enabled"))
    }

    pub async fn get_cached_analytics(&mut self, _key: &str) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    pub async fn delete_cached(&mut self, _key: &str) -> Result<()> {
        Ok(())
    }

    pub async fn cache_embedding(&mut self, _text: &str, _embedding: &[f32], _ttl_secs: Option<u64>) -> Result<()> {
        Err(MemoryError::configuration("Redis feature not enabled"))
    }

    pub async fn get_cached_embedding(&mut self, _text: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    pub async fn get_cache_stats(&mut self) -> Result<CacheStats> {
        Ok(CacheStats::default())
    }

    pub async fn clear_cache(&mut self) -> Result<()> {
        Ok(())
    }
}
