// High-performance memory pool implementation
//
// Provides efficient memory allocation and deallocation with
// intelligent pool management and optimization strategies.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

use crate::error::Result;
use super::PerformanceConfig;

/// High-performance memory pool
#[derive(Debug)]
pub struct MemoryPool {
    #[allow(dead_code)]
    config: PerformanceConfig,
    pools: Arc<RwLock<HashMap<usize, Pool>>>,
    allocation_stats: Arc<RwLock<AllocationStatistics>>,
    fragmentation_monitor: Arc<RwLock<FragmentationMonitor>>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        let mut pools = HashMap::new();
        
        // Create pools for common sizes
        let common_sizes = vec![64, 128, 256, 512, 1024, 2048, 4096, 8192];
        for size in common_sizes {
            pools.insert(size, Pool::new(size, 100)); // 100 chunks per pool initially
        }
        
        Ok(Self {
            config,
            pools: Arc::new(RwLock::new(pools)),
            allocation_stats: Arc::new(RwLock::new(AllocationStatistics::new())),
            fragmentation_monitor: Arc::new(RwLock::new(FragmentationMonitor::new())),
        })
    }
    
    /// Allocate memory from pool
    pub async fn allocate(&self, size: usize) -> Result<MemoryChunk> {
        let mut stats = self.allocation_stats.write().await;
        stats.total_allocations += 1;
        
        // Find appropriate pool size
        let pool_size = self.find_pool_size(size);
        
        let mut pools = self.pools.write().await;
        
        // Get or create pool
        let pool = pools.entry(pool_size).or_insert_with(|| Pool::new(pool_size, 50));
        
        // Try to get chunk from pool
        if let Some(chunk) = pool.allocate() {
            stats.pool_allocations += 1;
            Ok(chunk)
        } else {
            // Pool is empty, create new chunk
            stats.direct_allocations += 1;
            Ok(MemoryChunk::new(pool_size))
        }
    }
    
    /// Deallocate memory back to pool
    pub async fn deallocate(&self, chunk: MemoryChunk) -> Result<()> {
        let mut stats = self.allocation_stats.write().await;
        stats.total_deallocations += 1;
        
        let mut pools = self.pools.write().await;
        
        if let Some(pool) = pools.get_mut(&chunk.size) {
            if pool.can_accept_chunk() {
                pool.deallocate(chunk);
                stats.pool_deallocations += 1;
            } else {
                // Pool is full, chunk will be dropped
                stats.direct_deallocations += 1;
            }
        } else {
            stats.direct_deallocations += 1;
        }
        
        Ok(())
    }
    
    /// Get memory pool statistics
    pub async fn get_statistics(&self) -> Result<MemoryPoolStatistics> {
        let stats = self.allocation_stats.read().await.clone();
        let pools = self.pools.read().await;
        let fragmentation = self.fragmentation_monitor.read().await.clone();
        
        let mut pool_stats = Vec::new();
        for (size, pool) in pools.iter() {
            pool_stats.push(PoolStatistics {
                chunk_size: *size,
                total_chunks: pool.total_chunks,
                available_chunks: pool.available_chunks.len(),
                utilization: 1.0 - (pool.available_chunks.len() as f64 / pool.total_chunks as f64),
            });
        }
        
        Ok(MemoryPoolStatistics {
            allocation_stats: stats,
            pool_stats,
            fragmentation_info: fragmentation,
            total_memory_mb: self.calculate_total_memory(&pools).await / (1024 * 1024),
        })
    }
    
    /// Apply optimization parameters
    pub async fn apply_optimization(&self, parameters: &HashMap<String, String>) -> Result<()> {
        // Apply pool size optimization
        if let Some(size_str) = parameters.get("pool_size_mb") {
            if let Ok(size_mb) = size_str.parse::<usize>() {
                println!("Optimizing memory pool size to {}MB", size_mb);
                self.resize_pools(size_mb).await?;
            }
        }
        
        // Apply chunk size optimization
        if let Some(chunk_str) = parameters.get("chunk_size_kb") {
            if let Ok(chunk_kb) = chunk_str.parse::<usize>() {
                println!("Optimizing memory pool chunk size to {}KB", chunk_kb);
                self.optimize_chunk_sizes(chunk_kb * 1024).await?;
            }
        }
        
        Ok(())
    }
    
    /// Find appropriate pool size for allocation
    fn find_pool_size(&self, size: usize) -> usize {
        // Round up to next power of 2 or common size
        let common_sizes = vec![64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
        
        for &pool_size in &common_sizes {
            if size <= pool_size {
                return pool_size;
            }
        }
        
        // For very large allocations, round up to next multiple of 4KB
        ((size + 4095) / 4096) * 4096
    }
    
    /// Calculate total memory used by pools
    async fn calculate_total_memory(&self, pools: &HashMap<usize, Pool>) -> usize {
        pools.iter()
            .map(|(size, pool)| size * pool.total_chunks)
            .sum()
    }
    
    /// Resize pools based on optimization
    async fn resize_pools(&self, target_size_mb: usize) -> Result<()> {
        let target_bytes = target_size_mb * 1024 * 1024;
        let mut pools = self.pools.write().await;
        
        // Calculate current total size
        let current_size = self.calculate_total_memory(&pools).await;
        
        if target_bytes > current_size {
            // Expand pools
            let expansion_factor = target_bytes as f64 / current_size as f64;
            for pool in pools.values_mut() {
                let new_capacity = (pool.total_chunks as f64 * expansion_factor) as usize;
                pool.expand_capacity(new_capacity);
            }
        } else if target_bytes < current_size {
            // Shrink pools
            let shrink_factor = target_bytes as f64 / current_size as f64;
            for pool in pools.values_mut() {
                let new_capacity = (pool.total_chunks as f64 * shrink_factor) as usize;
                pool.shrink_capacity(new_capacity);
            }
        }
        
        Ok(())
    }
    
    /// Optimize chunk sizes based on allocation patterns
    async fn optimize_chunk_sizes(&self, base_chunk_size: usize) -> Result<()> {
        let stats = self.allocation_stats.read().await;
        
        // Analyze allocation patterns to determine optimal chunk sizes
        // This is a simplified implementation
        println!("Optimizing chunk sizes based on allocation patterns");
        println!("Base chunk size: {} bytes", base_chunk_size);
        println!("Total allocations: {}", stats.total_allocations);
        
        Ok(())
    }
}

/// Memory pool for specific chunk size
#[derive(Debug)]
pub struct Pool {
    pub chunk_size: usize,
    pub total_chunks: usize,
    pub available_chunks: VecDeque<MemoryChunk>,
    pub max_capacity: usize,
}

impl Pool {
    /// Create a new pool
    pub fn new(chunk_size: usize, initial_capacity: usize) -> Self {
        let mut available_chunks = VecDeque::new();
        
        // Pre-allocate chunks
        for _ in 0..initial_capacity {
            available_chunks.push_back(MemoryChunk::new(chunk_size));
        }
        
        Self {
            chunk_size,
            total_chunks: initial_capacity,
            available_chunks,
            max_capacity: initial_capacity * 2, // Allow growth up to 2x
        }
    }
    
    /// Allocate a chunk from the pool
    pub fn allocate(&mut self) -> Option<MemoryChunk> {
        self.available_chunks.pop_front()
    }
    
    /// Deallocate a chunk back to the pool
    pub fn deallocate(&mut self, chunk: MemoryChunk) {
        if self.available_chunks.len() < self.max_capacity {
            self.available_chunks.push_back(chunk);
        }
        // If pool is full, chunk is dropped
    }
    
    /// Check if pool can accept more chunks
    pub fn can_accept_chunk(&self) -> bool {
        self.available_chunks.len() < self.max_capacity
    }
    
    /// Expand pool capacity
    pub fn expand_capacity(&mut self, new_capacity: usize) {
        if new_capacity > self.total_chunks {
            let additional_chunks = new_capacity - self.total_chunks;
            
            for _ in 0..additional_chunks {
                self.available_chunks.push_back(MemoryChunk::new(self.chunk_size));
            }
            
            self.total_chunks = new_capacity;
            self.max_capacity = new_capacity;
        }
    }
    
    /// Shrink pool capacity
    pub fn shrink_capacity(&mut self, new_capacity: usize) {
        if new_capacity < self.total_chunks {
            let chunks_to_remove = self.total_chunks - new_capacity;
            
            for _ in 0..chunks_to_remove {
                if self.available_chunks.pop_back().is_none() {
                    break;
                }
            }
            
            self.total_chunks = new_capacity;
            self.max_capacity = new_capacity;
        }
    }
}

/// Memory chunk
#[derive(Debug)]
pub struct MemoryChunk {
    pub data: Vec<u8>,
    pub size: usize,
    pub allocated_at: Instant,
}

impl MemoryChunk {
    /// Create a new memory chunk
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
            size,
            allocated_at: Instant::now(),
        }
    }
    
    /// Get chunk data as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
    
    /// Get chunk data as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

/// Allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStatistics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub pool_allocations: u64,
    pub pool_deallocations: u64,
    pub direct_allocations: u64,
    pub direct_deallocations: u64,
    pub pool_hit_rate: f64,
}

impl AllocationStatistics {
    pub fn new() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            pool_allocations: 0,
            pool_deallocations: 0,
            direct_allocations: 0,
            direct_deallocations: 0,
            pool_hit_rate: 0.0,
        }
    }
    
    pub fn calculate_hit_rate(&mut self) {
        if self.total_allocations > 0 {
            self.pool_hit_rate = self.pool_allocations as f64 / self.total_allocations as f64;
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatistics {
    pub chunk_size: usize,
    pub total_chunks: usize,
    pub available_chunks: usize,
    pub utilization: f64,
}

/// Fragmentation monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationMonitor {
    pub internal_fragmentation: f64,
    pub external_fragmentation: f64,
    pub fragmentation_score: f64,
}

impl FragmentationMonitor {
    pub fn new() -> Self {
        Self {
            internal_fragmentation: 0.0,
            external_fragmentation: 0.0,
            fragmentation_score: 0.0,
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolStatistics {
    pub allocation_stats: AllocationStatistics,
    pub pool_stats: Vec<PoolStatistics>,
    pub fragmentation_info: FragmentationMonitor,
    pub total_memory_mb: usize,
}
