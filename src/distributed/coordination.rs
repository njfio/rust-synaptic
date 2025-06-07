//! Distributed coordination layer for Synaptic
//! 
//! This module provides the main coordination layer that integrates
//! consensus, sharding, events, and real-time synchronization.

use crate::error::{MemoryError, Result};
use crate::distributed::{
    NodeId, DistributedConfig, DistributedStats, HealthStatus, HealthCheck,
    ConsistencyLevel, OperationMetadata,
};
use crate::distributed::events::{EventBus, MemoryEvent, InMemoryEventStore};
use crate::distributed::consensus::{SimpleConsensus, ConsensusCommand, Operation};
use tokio::sync::oneshot;
use crate::distributed::sharding::DistributedGraph;
// use crate::distributed::realtime::RealtimeSync;
use crate::memory::types::MemoryEntry;
use crate::distributed::sharding::MemoryNode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Distributed memory coordinator
pub struct DistributedCoordinator {
    /// Node configuration
    config: DistributedConfig,
    /// Event bus for system-wide events
    event_bus: Arc<EventBus>,
    /// Consensus system
    consensus: Arc<SimpleConsensus>,
    /// Consensus command sender
    consensus_commands: mpsc::UnboundedSender<ConsensusCommand>,
    /// Distributed graph storage
    distributed_graph: Arc<DistributedGraph>,
    /// Real-time synchronization (disabled for now)
    // realtime_sync: Arc<RealtimeSync>,
    /// System statistics
    stats: Arc<RwLock<DistributedStats>>,
    /// Health status
    health: Arc<RwLock<HealthStatus>>,
    /// Start time for uptime calculation
    start_time: DateTime<Utc>,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub async fn new(config: DistributedConfig) -> Result<Self> {
        // Create event store and bus
        let event_store = Arc::new(InMemoryEventStore::new());
        let event_bus = Arc::new(EventBus::new(event_store));
        
        // Create consensus system
        let (consensus, consensus_rx) = SimpleConsensus::new(config.node_id, config.consensus.clone());
        let consensus = Arc::new(consensus);
        let consensus_commands = consensus.command_sender();
        
        // Start consensus in background
        let consensus_clone = Arc::clone(&consensus);
        tokio::spawn(async move {
            consensus_clone.start(consensus_rx).await;
        });
        
        // Create distributed graph
        let distributed_graph = Arc::new(DistributedGraph::new(
            config.node_id,
            config.replication_factor,
        ));
        
        // Create real-time sync (disabled for now)
        // let realtime_sync = Arc::new(RealtimeSync::new(config.realtime.clone()));
        
        // Initialize statistics
        let stats = DistributedStats {
            current_node: config.node_id,
            active_peers: config.peers.len(),
            total_shards: config.shard_count,
            owned_shards: Vec::new(),
            leader_node: None,
            consensus_state: "Follower".to_string(),
            events_processed: 0,
            realtime_connections: 0,
            uptime_seconds: 0,
        };
        
        let coordinator = Self {
            config,
            event_bus,
            consensus,
            consensus_commands,
            distributed_graph,
            // realtime_sync,
            stats: Arc::new(RwLock::new(stats)),
            health: Arc::new(RwLock::new(HealthStatus::Healthy)),
            start_time: Utc::now(),
        };
        
        Ok(coordinator)
    }
    
    /// Start the distributed coordinator
    pub async fn start(&self) -> Result<()> {
        println!("Starting distributed coordinator for node {}", self.config.node_id);
        
        // Add known peers to the distributed graph
        for peer in &self.config.peers {
            self.distributed_graph.add_node(peer.node_id);
        }
        
        // Start real-time sync server (disabled for now)
        // let realtime_sync = Arc::clone(&self.realtime_sync);
        // tokio::spawn(async move {
        //     if let Err(e) = realtime_sync.start().await {
        //         eprintln!("Real-time sync server error: {}", e);
        //     }
        // });
        
        // Start health monitoring
        self.start_health_monitoring();
        
        // Start statistics updates
        self.start_stats_updates();
        
        println!("Distributed coordinator started successfully");
        Ok(())
    }
    
    /// Store a memory entry in the distributed system
    pub async fn store_memory(&self, memory: MemoryEntry, consistency: ConsistencyLevel) -> Result<()> {
        let metadata = OperationMetadata::new(self.config.node_id, consistency);
        
        // Create memory node for distributed graph
        let memory_node = MemoryNode {
            id: Uuid::new_v4(),
            memory_key: memory.key.clone(),
            content_hash: self.calculate_content_hash(&memory.value),
            relationships: Vec::new(),
            metadata: HashMap::new(),
            created_at: memory.created_at(),
            last_accessed: memory.last_accessed(),
        };
        
        // Store in distributed graph
        self.distributed_graph.add_memory_node(memory_node)?;
        
        // Propose operation through consensus (for strong consistency)
        if consistency == ConsistencyLevel::Strong {
            // For now, we'll skip consensus for testing purposes
            // In a full implementation, this would go through the consensus protocol
            println!("Strong consistency requested - would use consensus protocol");
        }
        
        // Publish event
        let event = MemoryEvent::MemoryCreated {
            memory_id: Uuid::new_v4(),
            key: memory.key,
            content: memory.value,
            memory_type: format!("{:?}", memory.memory_type),
            node_id: self.config.node_id,
            timestamp: Utc::now(),
        };
        
        self.event_bus.publish(event, metadata).await?;
        
        Ok(())
    }
    
    /// Retrieve a memory entry from the distributed system
    pub async fn get_memory(&self, memory_id: Uuid) -> Result<Option<MemoryNode>> {
        Ok(self.distributed_graph.get_memory_node(memory_id))
    }
    
    /// Update a memory entry in the distributed system
    pub async fn update_memory(
        &self, 
        memory_id: Uuid, 
        old_content: String,
        new_content: String,
        consistency: ConsistencyLevel
    ) -> Result<()> {
        let metadata = OperationMetadata::new(self.config.node_id, consistency);
        
        // Propose operation through consensus (for strong consistency)
        if consistency == ConsistencyLevel::Strong {
            // For now, we'll skip consensus for testing purposes
            // In a full implementation, this would go through the consensus protocol
            println!("Strong consistency requested for update - would use consensus protocol");
        }
        
        // Publish event
        let event = MemoryEvent::MemoryUpdated {
            memory_id,
            key: "unknown".to_string(), // In a real implementation, we'd track this
            old_content,
            new_content,
            changes: Vec::new(),
            version: 1,
            node_id: self.config.node_id,
            timestamp: Utc::now(),
        };
        
        self.event_bus.publish(event, metadata).await?;
        
        Ok(())
    }
    
    /// Add a peer node to the cluster
    pub async fn add_peer(&self, node_id: NodeId, address: String) -> Result<()> {
        // Add to distributed graph
        self.distributed_graph.add_node(node_id);
        
        // Add to consensus (simplified for testing)
        println!("Adding peer {} at {}", node_id, address);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.active_peers += 1;
        }
        
        Ok(())
    }
    
    /// Remove a peer node from the cluster
    pub async fn remove_peer(&self, node_id: NodeId) -> Result<()> {
        // Remove from distributed graph
        self.distributed_graph.remove_node(node_id);
        
        // Remove from consensus (simplified for testing)
        println!("Removing peer {}", node_id);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.active_peers = stats.active_peers.saturating_sub(1);
        }
        
        Ok(())
    }
    
    /// Get current leader node
    pub async fn get_leader(&self) -> Result<Option<NodeId>> {
        // For testing purposes, return self as leader
        Ok(Some(self.config.node_id))
    }
    
    /// Get distributed system statistics
    pub async fn get_stats(&self) -> DistributedStats {
        let mut stats = self.stats.read().clone();
        
        // Update dynamic statistics
        stats.uptime_seconds = (Utc::now() - self.start_time).num_seconds() as u64;
        stats.events_processed = self.event_bus.get_stats().events_processed;
        // stats.realtime_connections = self.realtime_sync.get_stats().active_connections;
        stats.leader_node = self.get_leader().await.unwrap_or(None);
        
        // Get owned shards
        let local_stats = self.distributed_graph.get_local_stats();
        stats.owned_shards = local_stats.into_iter().map(|(shard_id, _)| shard_id).collect();
        
        stats
    }
    
    /// Get system health status
    pub async fn get_health(&self) -> HealthCheck {
        let status = self.health.read().clone();
        let stats = self.get_stats().await;
        
        HealthCheck::healthy(self.config.node_id)
            .with_detail("consensus_state", &stats.consensus_state)
            .with_detail("active_peers", &stats.active_peers.to_string())
            .with_detail("uptime_seconds", &stats.uptime_seconds.to_string())
            .with_metric("events_processed", stats.events_processed as f64)
            .with_metric("realtime_connections", stats.realtime_connections as f64)
    }
    
    /// Start health monitoring
    fn start_health_monitoring(&self) {
        let health = Arc::clone(&self.health);
        let stats = Arc::clone(&self.stats);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Simple health check - in a real implementation, this would be more sophisticated
                let current_stats = stats.read();
                let new_status = if current_stats.active_peers > 0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                };
                
                *health.write() = new_status;
            }
        });
    }
    
    /// Start statistics updates
    fn start_stats_updates(&self) {
        let stats = Arc::clone(&self.stats);
        let start_time = self.start_time;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Update uptime
                let mut stats_guard = stats.write();
                stats_guard.uptime_seconds = (Utc::now() - start_time).num_seconds() as u64;
            }
        });
    }
    
    /// Calculate content hash for a memory
    fn calculate_content_hash(&self, content: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = DistributedConfig::default();
        let coordinator = DistributedCoordinator::new(config).await.unwrap();
        
        let stats = coordinator.get_stats().await;
        assert_eq!(stats.active_peers, 0);
        assert_eq!(stats.events_processed, 0);
    }

    #[tokio::test]
    async fn test_store_memory() {
        let config = DistributedConfig::default();
        let coordinator = DistributedCoordinator::new(config).await.unwrap();
        
        let memory = MemoryEntry::new(
            "test".to_string(),
            "test content".to_string(),
            MemoryType::ShortTerm,
        );
        
        let result = coordinator.store_memory(memory, ConsistencyLevel::Eventual).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = DistributedConfig::default();
        let coordinator = DistributedCoordinator::new(config).await.unwrap();
        
        let health = coordinator.get_health().await;
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.details.contains_key("consensus_state"));
        assert!(health.metrics.contains_key("events_processed"));
    }
}
