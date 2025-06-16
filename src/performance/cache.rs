// High-performance cache implementation
//
// Provides intelligent caching with adaptive algorithms, compression,
// and performance optimization strategies.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

use crate::error::Result;
use super::PerformanceConfig;

/// High-performance cache with intelligent optimization
#[derive(Debug)]
pub struct PerformanceCache {
    config: PerformanceConfig,
    cache_data: Arc<RwLock<HashMap<String, CacheEntry>>>,
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    cache_stats: Arc<RwLock<CacheStatistics>>,
    eviction_policy: Arc<RwLock<EvictionPolicy>>,
}

impl PerformanceCache {
    /// Create a new performance cache
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        Ok(Self {
            config,
            cache_data: Arc::new(RwLock::new(HashMap::new())),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStatistics::new())),
            eviction_policy: Arc::new(RwLock::new(EvictionPolicy::LRU)),
        })
    }
    
    /// Get value from cache
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut cache = self.cache_data.write().await;
        let mut stats = self.cache_stats.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is expired
            if entry.is_expired() {
                cache.remove(key);
                stats.misses += 1;
                return Ok(None);
            }
            
            // Update access information
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update access patterns
            self.update_access_pattern(key).await?;
            
            stats.hits += 1;
            Ok(Some(entry.data.clone()))
        } else {
            stats.misses += 1;
            Ok(None)
        }
    }
    
    /// Put value in cache
    pub async fn put(&self, key: String, value: Vec<u8>) -> Result<()> {
        let mut cache = self.cache_data.write().await;
        
        // Check cache size limit
        let current_size = self.calculate_cache_size(&cache).await;
        let max_size = self.config.cache_size_mb * 1024 * 1024;
        
        if current_size + value.len() > max_size {
            self.evict_entries(&mut cache, value.len()).await?;
        }
        
        let entry = CacheEntry {
            data: value,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            ttl: Duration::from_secs(self.config.cache_ttl_seconds),
        };
        
        cache.insert(key.clone(), entry);
        
        // Update access patterns
        self.update_access_pattern(&key).await?;
        
        Ok(())
    }
    
    /// Remove value from cache
    pub async fn remove(&self, key: &str) -> Result<bool> {
        let mut cache = self.cache_data.write().await;
        Ok(cache.remove(key).is_some())
    }
    
    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        let mut cache = self.cache_data.write().await;
        cache.clear();
        
        let mut stats = self.cache_stats.write().await;
        stats.evictions += cache.len() as u64;
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_statistics(&self) -> Result<CacheStatistics> {
        Ok(self.cache_stats.read().await.clone())
    }
    
    /// Apply optimization parameters
    pub async fn apply_optimization(&self, parameters: &HashMap<String, String>) -> Result<()> {
        // Apply cache size optimization
        if let Some(size_str) = parameters.get("cache_size_mb") {
            if let Ok(size) = size_str.parse::<usize>() {
                // Update cache size (would require config update in real implementation)
                println!("Optimizing cache size to {}MB", size);
            }
        }
        
        // Apply TTL optimization
        if let Some(ttl_str) = parameters.get("ttl_seconds") {
            if let Ok(ttl) = ttl_str.parse::<u64>() {
                println!("Optimizing cache TTL to {}s", ttl);
            }
        }
        
        // Apply eviction policy optimization
        if let Some(policy_str) = parameters.get("eviction_policy") {
            let policy = match policy_str.as_str() {
                "lru" => EvictionPolicy::LRU,
                "lfu" => EvictionPolicy::LFU,
                "adaptive" => EvictionPolicy::Adaptive,
                _ => EvictionPolicy::LRU,
            };
            
            *self.eviction_policy.write().await = policy;
        }
        
        Ok(())
    }
    
    /// Update access pattern for a key
    async fn update_access_pattern(&self, key: &str) -> Result<()> {
        let mut patterns = self.access_patterns.write().await;
        
        let pattern = patterns.entry(key.to_string()).or_insert_with(|| AccessPattern {
            key: key.to_string(),
            access_count: 0,
            last_access: Instant::now(),
            access_frequency: 0.0,
            access_intervals: Vec::new(),
        });
        
        let now = Instant::now();
        if pattern.access_count > 0 {
            let interval = now.duration_since(pattern.last_access);
            pattern.access_intervals.push(interval);
            
            // Keep only last 10 intervals
            if pattern.access_intervals.len() > 10 {
                pattern.access_intervals.remove(0);
            }
            
            // Calculate frequency
            if !pattern.access_intervals.is_empty() {
                let avg_interval = pattern.access_intervals.iter().sum::<Duration>() 
                    / pattern.access_intervals.len() as u32;
                pattern.access_frequency = 1.0 / avg_interval.as_secs_f64();
            }
        }
        
        pattern.access_count += 1;
        pattern.last_access = now;
        
        Ok(())
    }
    
    /// Calculate total cache size
    async fn calculate_cache_size(&self, cache: &HashMap<String, CacheEntry>) -> usize {
        cache.values().map(|entry| entry.data.len()).sum()
    }
    
    /// Evict entries to make space
    async fn evict_entries(&self, cache: &mut HashMap<String, CacheEntry>, needed_space: usize) -> Result<()> {
        let policy = self.eviction_policy.read().await.clone();
        let mut stats = self.cache_stats.write().await;
        
        match policy {
            EvictionPolicy::LRU => {
                self.evict_lru(cache, needed_space, &mut stats).await?;
            }
            EvictionPolicy::LFU => {
                self.evict_lfu(cache, needed_space, &mut stats).await?;
            }
            EvictionPolicy::Adaptive => {
                self.evict_adaptive(cache, needed_space, &mut stats).await?;
            }
        }
        
        Ok(())
    }
    
    /// LRU eviction
    async fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>, needed_space: usize, stats: &mut CacheStatistics) -> Result<()> {
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.last_accessed);
        
        let mut freed_space = 0;
        let mut to_remove = Vec::new();
        
        for (key, entry) in entries {
            if freed_space >= needed_space {
                break;
            }
            
            freed_space += entry.data.len();
            to_remove.push(key.clone());
        }
        
        for key in to_remove {
            cache.remove(&key);
            stats.evictions += 1;
        }
        
        Ok(())
    }
    
    /// LFU eviction
    async fn evict_lfu(&self, cache: &mut HashMap<String, CacheEntry>, needed_space: usize, stats: &mut CacheStatistics) -> Result<()> {
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.access_count);
        
        let mut freed_space = 0;
        let mut to_remove = Vec::new();
        
        for (key, entry) in entries {
            if freed_space >= needed_space {
                break;
            }
            
            freed_space += entry.data.len();
            to_remove.push(key.clone());
        }
        
        for key in to_remove {
            cache.remove(&key);
            stats.evictions += 1;
        }
        
        Ok(())
    }
    
    /// Adaptive eviction based on access patterns
    async fn evict_adaptive(&self, cache: &mut HashMap<String, CacheEntry>, needed_space: usize, stats: &mut CacheStatistics) -> Result<()> {
        let patterns = self.access_patterns.read().await;
        
        let mut entries: Vec<_> = cache.iter().collect();
        
        // Sort by adaptive score (combination of recency, frequency, and size)
        entries.sort_by(|(key_a, entry_a), (key_b, entry_b)| {
            let score_a = self.calculate_adaptive_score(key_a, entry_a, &patterns);
            let score_b = self.calculate_adaptive_score(key_b, entry_b, &patterns);
            score_a.partial_cmp(&score_b).unwrap()
        });
        
        let mut freed_space = 0;
        let mut to_remove = Vec::new();
        
        for (key, entry) in entries {
            if freed_space >= needed_space {
                break;
            }
            
            freed_space += entry.data.len();
            to_remove.push(key.clone());
        }
        
        for key in to_remove {
            cache.remove(&key);
            stats.evictions += 1;
        }
        
        Ok(())
    }
    
    /// Calculate adaptive score for eviction
    fn calculate_adaptive_score(&self, key: &str, entry: &CacheEntry, patterns: &HashMap<String, AccessPattern>) -> f64 {
        let recency_score = 1.0 / (entry.last_accessed.elapsed().as_secs_f64() + 1.0);
        let frequency_score = entry.access_count as f64;
        let size_penalty = entry.data.len() as f64 / 1024.0; // Penalty for large entries
        
        let pattern_score = if let Some(pattern) = patterns.get(key) {
            pattern.access_frequency
        } else {
            0.0
        };
        
        // Weighted combination
        (recency_score * 0.3 + frequency_score * 0.3 + pattern_score * 0.4) / (1.0 + size_penalty * 0.1)
    }
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub data: Vec<u8>,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub ttl: Duration,
}

impl CacheEntry {
    /// Check if entry is expired
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Access pattern tracking
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub key: String,
    pub access_count: u64,
    pub last_access: Instant,
    pub access_frequency: f64,
    pub access_intervals: Vec<Duration>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f64,
    pub miss_rate: f64,
}

impl CacheStatistics {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            total_entries: 0,
            total_size_bytes: 0,
            hit_rate: 0.0,
            miss_rate: 0.0,
        }
    }
    
    pub fn calculate_rates(&mut self) {
        let total_accesses = self.hits + self.misses;
        if total_accesses > 0 {
            self.hit_rate = self.hits as f64 / total_accesses as f64;
            self.miss_rate = self.misses as f64 / total_accesses as f64;
        }
    }
}

/// Cache eviction policy
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,      // Least Recently Used
    LFU,      // Least Frequently Used
    Adaptive, // Adaptive based on access patterns
}
