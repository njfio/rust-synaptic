//! Distributed architecture components for Synaptic
//! 
//! This module provides the foundation for distributed memory systems,
//! including event-driven architecture, consensus, and real-time synchronization.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub mod events;
pub mod consensus;
pub mod sharding;
// pub mod realtime; // Temporarily disabled due to WebSocket dependencies
pub mod coordination;

/// Node identifier in the distributed system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node-{}", self.0.to_string()[..8].to_lowercase())
    }
}

/// Shard identifier for distributed graph storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u64);

impl ShardId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard-{}", self.0)
    }
}

/// Configuration for distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// This node's identifier
    pub node_id: NodeId,
    /// List of known peer nodes
    pub peers: Vec<NodeAddress>,
    /// Number of replicas for each shard
    pub replication_factor: usize,
    /// Number of shards to distribute data across
    pub shard_count: u64,
    /// Consensus configuration
    pub consensus: ConsensusConfig,
    /// Event system configuration
    pub events: EventConfig,
    /// Real-time sync configuration
    pub realtime: RealtimeConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId::new(),
            peers: Vec::new(),
            replication_factor: 3,
            shard_count: 16,
            consensus: ConsensusConfig::default(),
            events: EventConfig::default(),
            realtime: RealtimeConfig::default(),
        }
    }
}

/// Network address of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAddress {
    pub node_id: NodeId,
    pub host: String,
    pub port: u16,
    pub is_leader: bool,
}

impl NodeAddress {
    pub fn new(node_id: NodeId, host: String, port: u16) -> Self {
        Self {
            node_id,
            host,
            port,
            is_leader: false,
        }
    }
    
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Consensus algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Election timeout in milliseconds
    pub election_timeout_ms: u64,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Maximum log entries per append
    pub max_log_entries: usize,
    /// Snapshot threshold (log entries before snapshot)
    pub snapshot_threshold: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            election_timeout_ms: 150,
            heartbeat_interval_ms: 50,
            max_log_entries: 100,
            snapshot_threshold: 1000,
        }
    }
}

/// Event system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventConfig {
    /// Kafka broker addresses
    pub kafka_brokers: Vec<String>,
    /// Event topic name
    pub event_topic: String,
    /// Consumer group ID
    pub consumer_group: String,
    /// Batch size for event processing
    pub batch_size: usize,
    /// Event retention time in hours
    pub retention_hours: u64,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            kafka_brokers: vec!["localhost:9092".to_string()],
            event_topic: "synaptic-events".to_string(),
            consumer_group: "synaptic-consumers".to_string(),
            batch_size: 100,
            retention_hours: 168, // 1 week
        }
    }
}

/// Real-time synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// WebSocket server port
    pub websocket_port: u16,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Heartbeat interval for connections
    pub heartbeat_interval_ms: u64,
    /// Message buffer size per connection
    pub message_buffer_size: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            websocket_port: 8080,
            max_connections: 10000,
            heartbeat_interval_ms: 30000,
            message_buffer_size: 1000,
        }
    }
}

/// Consistency level for distributed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Eventually consistent - fastest, may be temporarily inconsistent
    Eventual,
    /// Strongly consistent - slower, always consistent
    Strong,
    /// Causally consistent - maintains causal ordering
    Causal,
}

impl Default for ConsistencyLevel {
    fn default() -> Self {
        ConsistencyLevel::Eventual
    }
}

/// Operation metadata for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetadata {
    /// Unique operation identifier
    pub operation_id: Uuid,
    /// Node that initiated the operation
    pub source_node: NodeId,
    /// Timestamp when operation was created
    pub timestamp: DateTime<Utc>,
    /// Required consistency level
    pub consistency: ConsistencyLevel,
    /// Operation timeout in milliseconds
    pub timeout_ms: u64,
}

impl OperationMetadata {
    pub fn new(source_node: NodeId, consistency: ConsistencyLevel) -> Self {
        Self {
            operation_id: Uuid::new_v4(),
            source_node,
            timestamp: Utc::now(),
            consistency,
            timeout_ms: 5000, // 5 seconds default
        }
    }
}

/// Statistics about the distributed system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedStats {
    /// Current node information
    pub current_node: NodeId,
    /// Number of active peer nodes
    pub active_peers: usize,
    /// Total number of shards
    pub total_shards: u64,
    /// Shards owned by this node
    pub owned_shards: Vec<ShardId>,
    /// Current leader node (if known)
    pub leader_node: Option<NodeId>,
    /// Consensus state
    pub consensus_state: String,
    /// Event processing statistics
    pub events_processed: u64,
    /// Real-time connections
    pub realtime_connections: usize,
    /// System uptime in seconds
    pub uptime_seconds: u64,
}

/// Distributed system health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Partitioned,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: HealthStatus,
    pub node_id: NodeId,
    pub timestamp: DateTime<Utc>,
    pub details: HashMap<String, String>,
    pub metrics: HashMap<String, f64>,
}

impl HealthCheck {
    pub fn healthy(node_id: NodeId) -> Self {
        Self {
            status: HealthStatus::Healthy,
            node_id,
            timestamp: Utc::now(),
            details: HashMap::new(),
            metrics: HashMap::new(),
        }
    }
    
    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn with_metric(mut self, key: &str, value: f64) -> Self {
        self.metrics.insert(key.to_string(), value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_creation() {
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        
        assert_ne!(node1, node2);
        assert_ne!(node1.to_string(), node2.to_string());
    }

    #[test]
    fn test_shard_id() {
        let shard = ShardId::new(42);
        assert_eq!(shard.as_u64(), 42);
        assert_eq!(shard.to_string(), "shard-42");
    }

    #[test]
    fn test_node_address() {
        let node_id = NodeId::new();
        let addr = NodeAddress::new(node_id, "localhost".to_string(), 8080);
        
        assert_eq!(addr.address(), "localhost:8080");
        assert!(!addr.is_leader);
    }

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        
        assert_eq!(config.replication_factor, 3);
        assert_eq!(config.shard_count, 16);
        assert!(config.peers.is_empty());
    }

    #[test]
    fn test_health_check() {
        let node_id = NodeId::new();
        let health = HealthCheck::healthy(node_id)
            .with_detail("version", "1.0.0")
            .with_metric("cpu_usage", 0.25);
        
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.details.get("version"), Some(&"1.0.0".to_string()));
        assert_eq!(health.metrics.get("cpu_usage"), Some(&0.25));
    }
}
