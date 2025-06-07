//! Distributed graph sharding for scalable memory storage
//! 
//! This module provides consistent hashing and sharding capabilities
//! for distributing memory nodes across multiple storage nodes.

use crate::error::{MemoryError, Result};
use crate::distributed::{NodeId, ShardId};
// use crate::memory::knowledge_graph::{MemoryNode, MemoryEdge};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use parking_lot::RwLock;
use uuid::Uuid;
use sha2::{Sha256, Digest};

/// Memory node for distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: Uuid,
    pub memory_key: String,
    pub content_hash: String,
    pub relationships: Vec<Uuid>,
    pub metadata: HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Memory edge for distributed storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: Uuid,
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub relationship_type: String,
    pub strength: f64,
}

/// Consistent hash ring for shard distribution
pub struct ConsistentHashRing {
    /// Virtual nodes on the ring (node_id -> hash_value)
    ring: BTreeMap<u64, NodeId>,
    /// Number of virtual nodes per physical node
    virtual_nodes: usize,
    /// Replication factor
    replication_factor: usize,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(virtual_nodes: usize, replication_factor: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            replication_factor,
        }
    }
    
    /// Add a node to the ring
    pub fn add_node(&mut self, node_id: NodeId) {
        for i in 0..self.virtual_nodes {
            let hash = self.hash_node(node_id, i);
            self.ring.insert(hash, node_id);
        }
    }
    
    /// Remove a node from the ring
    pub fn remove_node(&mut self, node_id: NodeId) {
        for i in 0..self.virtual_nodes {
            let hash = self.hash_node(node_id, i);
            self.ring.remove(&hash);
        }
    }
    
    /// Get the nodes responsible for a given key
    pub fn get_nodes(&self, key: &str) -> Vec<NodeId> {
        if self.ring.is_empty() {
            return Vec::new();
        }
        
        let key_hash = self.hash_key(key);
        let mut nodes = Vec::new();
        let mut seen_nodes = std::collections::HashSet::new();
        
        // Find the first node clockwise from the key hash
        let mut iter = self.ring.range(key_hash..).chain(self.ring.iter());
        
        for (_, &node_id) in iter {
            if !seen_nodes.contains(&node_id) {
                nodes.push(node_id);
                seen_nodes.insert(node_id);
                
                if nodes.len() >= self.replication_factor {
                    break;
                }
            }
        }
        
        nodes
    }
    
    /// Get all nodes in the ring
    pub fn get_all_nodes(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self.ring.values().cloned().collect();
        nodes.sort_by_key(|n| n.as_uuid());
        nodes.dedup();
        nodes
    }
    
    /// Hash a node with virtual node index
    fn hash_node(&self, node_id: NodeId, virtual_index: usize) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(node_id.as_uuid().as_bytes());
        hasher.update(&virtual_index.to_be_bytes());
        let result = hasher.finalize();
        
        // Take first 8 bytes as u64
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        u64::from_be_bytes(bytes)
    }
    
    /// Hash a key
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();
        
        // Take first 8 bytes as u64
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        u64::from_be_bytes(bytes)
    }
}

/// Distributed graph shard
pub struct GraphShard {
    /// Shard identifier
    pub shard_id: ShardId,
    /// Local memory nodes in this shard
    local_nodes: Arc<RwLock<HashMap<Uuid, MemoryNode>>>,
    /// Local edges in this shard
    local_edges: Arc<RwLock<HashMap<Uuid, MemoryEdge>>>,
    /// Cross-shard edge references
    cross_shard_edges: Arc<RwLock<HashMap<Uuid, RemoteEdgeRef>>>,
    /// Shard statistics
    stats: Arc<RwLock<ShardStats>>,
}

impl GraphShard {
    /// Create a new graph shard
    pub fn new(shard_id: ShardId) -> Self {
        Self {
            shard_id,
            local_nodes: Arc::new(RwLock::new(HashMap::new())),
            local_edges: Arc::new(RwLock::new(HashMap::new())),
            cross_shard_edges: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ShardStats::default())),
        }
    }
    
    /// Add a memory node to this shard
    pub fn add_node(&self, node: MemoryNode) -> Result<()> {
        let node_id = node.id;
        self.local_nodes.write().insert(node_id, node);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.node_count += 1;
            stats.last_modified = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Remove a memory node from this shard
    pub fn remove_node(&self, node_id: Uuid) -> Result<Option<MemoryNode>> {
        let node = self.local_nodes.write().remove(&node_id);
        
        if node.is_some() {
            // Update statistics
            let mut stats = self.stats.write();
            stats.node_count = stats.node_count.saturating_sub(1);
            stats.last_modified = chrono::Utc::now();
        }
        
        Ok(node)
    }
    
    /// Get a memory node from this shard
    pub fn get_node(&self, node_id: Uuid) -> Option<MemoryNode> {
        self.local_nodes.read().get(&node_id).cloned()
    }
    
    /// Add an edge to this shard
    pub fn add_edge(&self, edge: MemoryEdge) -> Result<()> {
        let edge_id = edge.id;
        self.local_edges.write().insert(edge_id, edge);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.edge_count += 1;
            stats.last_modified = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Remove an edge from this shard
    pub fn remove_edge(&self, edge_id: Uuid) -> Result<Option<MemoryEdge>> {
        let edge = self.local_edges.write().remove(&edge_id);
        
        if edge.is_some() {
            // Update statistics
            let mut stats = self.stats.write();
            stats.edge_count = stats.edge_count.saturating_sub(1);
            stats.last_modified = chrono::Utc::now();
        }
        
        Ok(edge)
    }
    
    /// Get all nodes in this shard
    pub fn get_all_nodes(&self) -> Vec<MemoryNode> {
        self.local_nodes.read().values().cloned().collect()
    }
    
    /// Get all edges in this shard
    pub fn get_all_edges(&self) -> Vec<MemoryEdge> {
        self.local_edges.read().values().cloned().collect()
    }
    
    /// Add a cross-shard edge reference
    pub fn add_cross_shard_edge(&self, edge_id: Uuid, remote_ref: RemoteEdgeRef) {
        self.cross_shard_edges.write().insert(edge_id, remote_ref);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.cross_shard_edges += 1;
            stats.last_modified = chrono::Utc::now();
        }
    }
    
    /// Get cross-shard edge references
    pub fn get_cross_shard_edges(&self) -> HashMap<Uuid, RemoteEdgeRef> {
        self.cross_shard_edges.read().clone()
    }
    
    /// Get shard statistics
    pub fn get_stats(&self) -> ShardStats {
        self.stats.read().clone()
    }
    
    /// Get shard size in bytes (approximate)
    pub fn get_size_bytes(&self) -> usize {
        let nodes = self.local_nodes.read();
        let edges = self.local_edges.read();
        
        // Rough estimation
        let node_size = nodes.len() * std::mem::size_of::<MemoryNode>();
        let edge_size = edges.len() * std::mem::size_of::<MemoryEdge>();
        
        node_size + edge_size
    }
}

/// Reference to an edge in a remote shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteEdgeRef {
    /// Edge identifier
    pub edge_id: Uuid,
    /// Shard containing the edge
    pub shard_id: ShardId,
    /// Node containing the shard
    pub node_id: NodeId,
    /// Edge metadata for quick access
    pub metadata: EdgeMetadata,
}

/// Lightweight edge metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub relationship_type: String,
    pub strength: f64,
}

/// Distributed graph coordinator
pub struct DistributedGraph {
    /// This node's ID
    node_id: NodeId,
    /// Consistent hash ring for shard distribution
    hash_ring: Arc<RwLock<ConsistentHashRing>>,
    /// Local shards managed by this node
    local_shards: Arc<RwLock<HashMap<ShardId, GraphShard>>>,
    /// Remote shard locations
    shard_locations: Arc<RwLock<HashMap<ShardId, NodeId>>>,
    /// Replication factor
    replication_factor: usize,
}

impl DistributedGraph {
    /// Create a new distributed graph
    pub fn new(node_id: NodeId, replication_factor: usize) -> Self {
        let hash_ring = ConsistentHashRing::new(100, replication_factor); // 100 virtual nodes
        
        Self {
            node_id,
            hash_ring: Arc::new(RwLock::new(hash_ring)),
            local_shards: Arc::new(RwLock::new(HashMap::new())),
            shard_locations: Arc::new(RwLock::new(HashMap::new())),
            replication_factor,
        }
    }
    
    /// Add a node to the cluster
    pub fn add_node(&self, node_id: NodeId) {
        self.hash_ring.write().add_node(node_id);
    }
    
    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: NodeId) {
        self.hash_ring.write().remove_node(node_id);
    }
    
    /// Get the shard ID for a memory node
    pub fn get_shard_id(&self, memory_id: Uuid) -> ShardId {
        let key = memory_id.to_string();
        let hash = self.hash_memory_id(&key);
        ShardId::new(hash % 1000) // Limit to 1000 shards
    }
    
    /// Get the nodes responsible for a shard
    pub fn get_shard_nodes(&self, shard_id: ShardId) -> Vec<NodeId> {
        let key = format!("shard-{}", shard_id.as_u64());
        self.hash_ring.read().get_nodes(&key)
    }
    
    /// Add a memory node to the distributed graph
    pub fn add_memory_node(&self, node: MemoryNode) -> Result<()> {
        let shard_id = self.get_shard_id(node.id);
        let responsible_nodes = self.get_shard_nodes(shard_id);
        
        // If this node is responsible for the shard, store locally
        if responsible_nodes.contains(&self.node_id) {
            let mut shards = self.local_shards.write();
            let shard = shards.entry(shard_id).or_insert_with(|| GraphShard::new(shard_id));
            shard.add_node(node)?;
        }
        
        Ok(())
    }
    
    /// Get a memory node from the distributed graph
    pub fn get_memory_node(&self, memory_id: Uuid) -> Option<MemoryNode> {
        let shard_id = self.get_shard_id(memory_id);
        
        // Check if we have the shard locally
        if let Some(shard) = self.local_shards.read().get(&shard_id) {
            return shard.get_node(memory_id);
        }
        
        // In a real implementation, we would make a remote call here
        None
    }
    
    /// Get statistics for all local shards
    pub fn get_local_stats(&self) -> Vec<(ShardId, ShardStats)> {
        self.local_shards
            .read()
            .iter()
            .map(|(&shard_id, shard)| (shard_id, shard.get_stats()))
            .collect()
    }
    
    /// Get total number of local nodes
    pub fn get_local_node_count(&self) -> usize {
        self.local_shards
            .read()
            .values()
            .map(|shard| shard.get_stats().node_count)
            .sum()
    }
    
    /// Hash a memory ID to determine shard placement
    fn hash_memory_id(&self, key: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let result = hasher.finalize();
        
        // Take first 8 bytes as u64
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        u64::from_be_bytes(bytes)
    }
}

/// Statistics for a graph shard
#[derive(Debug, Clone, Default)]
pub struct ShardStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub cross_shard_edges: usize,
    pub size_bytes: usize,
    pub last_modified: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::knowledge_graph::RelationshipType;

    #[test]
    fn test_consistent_hash_ring() {
        let mut ring = ConsistentHashRing::new(3, 2);
        
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        ring.add_node(node1);
        ring.add_node(node2);
        
        let nodes = ring.get_nodes("test-key");
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&node1) || nodes.contains(&node2));
    }

    #[test]
    fn test_graph_shard() {
        let shard = GraphShard::new(ShardId::new(1));
        
        let node = MemoryNode {
            id: Uuid::new_v4(),
            memory_key: "test".to_string(),
            content_hash: "hash".to_string(),
            relationships: Vec::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        
        let node_id = node.id;
        shard.add_node(node).unwrap();
        
        let retrieved = shard.get_node(node_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, node_id);
        
        let stats = shard.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_distributed_graph() {
        let node_id = NodeId::new();
        let graph = DistributedGraph::new(node_id, 2);
        
        graph.add_node(node_id);
        
        let memory_node = MemoryNode {
            id: Uuid::new_v4(),
            memory_key: "test".to_string(),
            content_hash: "hash".to_string(),
            relationships: Vec::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
        };
        
        let memory_id = memory_node.id;
        graph.add_memory_node(memory_node).unwrap();
        
        // Should be able to retrieve the node
        let retrieved = graph.get_memory_node(memory_id);
        assert!(retrieved.is_some());
    }
}
