//! EXPERIMENTAL distributed coordination layer for Synaptic.
//!
//! Integrates the single-node consensus scaffolding, local sharding, and the
//! in-memory event bus. Real-time synchronization is not implemented, and
//! strong consistency (which would require consensus replication) returns an
//! error rather than pretending to replicate. See the `distributed` module
//! docs for the full list of limitations.

use crate::distributed::consensus::{ConsensusCommand, Operation, SimpleConsensus};
use crate::distributed::events::{EventBus, InMemoryEventStore, MemoryEvent};
use crate::distributed::sharding::DistributedGraph;
use crate::distributed::sharding::MemoryNode;
use crate::distributed::{
    ConsistencyLevel, DistributedConfig, DistributedStats, HealthCheck, HealthStatus, NodeId,
    OperationMetadata,
};
use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

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
        let (consensus, consensus_rx) =
            SimpleConsensus::new(config.node_id, config.consensus.clone());
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
            stats: Arc::new(RwLock::new(stats)),
            health: Arc::new(RwLock::new(HealthStatus::Healthy)),
            start_time: Utc::now(),
        };

        Ok(coordinator)
    }

    /// Start the distributed coordinator
    pub async fn start(&self) -> Result<()> {
        tracing::info!(
            component = "distributed_coordinator",
            operation = "start",
            node_id = %self.config.node_id,
            peer_count = %self.config.peers.len(),
            "Starting distributed coordinator"
        );

        // Add known peers to the distributed graph
        for peer in &self.config.peers {
            self.distributed_graph.add_node(peer.node_id);
        }

        // Note: real-time synchronization is not implemented in this
        // experimental module; no sync server is started.

        // Start health monitoring
        self.start_health_monitoring();

        // Start statistics updates
        self.start_stats_updates();

        tracing::info!(
            component = "distributed_coordinator",
            operation = "start_complete",
            node_id = %self.config.node_id,
            "Distributed coordinator started successfully"
        );
        Ok(())
    }

    /// Store a memory entry in the distributed system
    pub async fn store_memory(
        &self,
        memory: MemoryEntry,
        consistency: ConsistencyLevel,
    ) -> Result<()> {
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

        // Strong consistency would require consensus replication, which is not
        // implemented. Refuse honestly instead of silently degrading.
        if consistency == ConsistencyLevel::Strong {
            return Err(MemoryError::feature_disabled(
                "distributed-experimental",
                "store_memory with strong consistency (consensus replication)",
            ));
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
        consistency: ConsistencyLevel,
    ) -> Result<()> {
        let metadata = OperationMetadata::new(self.config.node_id, consistency);

        // Strong consistency would require consensus replication, which is not
        // implemented. Refuse honestly instead of silently degrading.
        if consistency == ConsistencyLevel::Strong {
            return Err(MemoryError::feature_disabled(
                "distributed-experimental",
                "update_memory with strong consistency (consensus replication)",
            ));
        }

        // Resolve the memory key from the local shard, if present.
        let key = self
            .distributed_graph
            .get_memory_node(memory_id)
            .map(|node| node.memory_key)
            .unwrap_or_default();

        // Publish event
        let event = MemoryEvent::MemoryUpdated {
            memory_id,
            key,
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
        // Parse the peer address ("host:port")
        let (host, port) = address.rsplit_once(':').ok_or_else(|| {
            MemoryError::configuration(format!(
                "invalid peer address '{}': expected 'host:port'",
                address
            ))
        })?;
        let port: u16 = port.parse().map_err(|_| {
            MemoryError::configuration(format!("invalid peer port in address '{}'", address))
        })?;

        // Add to distributed graph
        self.distributed_graph.add_node(node_id);

        // Register the peer with the consensus loop
        self.consensus_commands
            .send(ConsensusCommand::AddPeer {
                node_id,
                address: crate::distributed::NodeAddress::new(node_id, host.to_string(), port),
            })
            .map_err(|_| MemoryError::ConsensusError {
                message: "consensus command channel closed".to_string(),
            })?;

        tracing::info!(
            component = "distributed_coordinator",
            operation = "add_peer",
            node_id = %node_id,
            address = %address,
            "Adding peer to distributed system"
        );

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

        // Deregister the peer from the consensus loop
        self.consensus_commands
            .send(ConsensusCommand::RemovePeer { node_id })
            .map_err(|_| MemoryError::ConsensusError {
                message: "consensus command channel closed".to_string(),
            })?;

        tracing::info!(
            component = "distributed_coordinator",
            operation = "remove_peer",
            node_id = %node_id,
            "Removing peer from distributed system"
        );

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.active_peers = stats.active_peers.saturating_sub(1);
        }

        Ok(())
    }

    /// Get current leader node, as reported by the consensus loop.
    ///
    /// Returns `None` when this node is not the leader; leader tracking
    /// across peers is not implemented in this experimental module.
    pub async fn get_leader(&self) -> Result<Option<NodeId>> {
        let (response_tx, response_rx) = oneshot::channel();
        self.consensus_commands
            .send(ConsensusCommand::GetLeader { response_tx })
            .map_err(|_| MemoryError::ConsensusError {
                message: "consensus command channel closed".to_string(),
            })?;
        response_rx.await.map_err(|_| MemoryError::ConsensusError {
            message: "consensus loop dropped leader query".to_string(),
        })
    }

    /// Get distributed system statistics
    pub async fn get_stats(&self) -> DistributedStats {
        let mut stats = self.stats.read().clone();

        // Update dynamic statistics
        stats.uptime_seconds = (Utc::now() - self.start_time).num_seconds() as u64;
        stats.events_processed = self.event_bus.get_stats().events_processed;
        // Real-time sync is not implemented; connection count is always 0.
        stats.realtime_connections = 0;
        stats.leader_node = self.get_leader().await.unwrap_or(None);

        // Get owned shards
        let local_stats = self.distributed_graph.get_local_stats();
        stats.owned_shards = local_stats
            .into_iter()
            .map(|(shard_id, _)| shard_id)
            .collect();

        stats
    }

    /// Get system health status
    ///
    /// The status is a coarse heuristic maintained by the background monitor
    /// (peer-count based); it does not probe peers or perform real checks.
    pub async fn get_health(&self) -> HealthCheck {
        let status = *self.health.read();
        let stats = self.get_stats().await;

        HealthCheck {
            status,
            node_id: self.config.node_id,
            timestamp: Utc::now(),
            details: HashMap::new(),
            metrics: HashMap::new(),
        }
        .with_detail("consensus_state", &stats.consensus_state)
        .with_detail("active_peers", &stats.active_peers.to_string())
        .with_detail("uptime_seconds", &stats.uptime_seconds.to_string())
        .with_metric("events_processed", stats.events_processed as f64)
        .with_metric("realtime_connections", stats.realtime_connections as f64)
    }

    /// Start health monitoring.
    ///
    /// This is a coarse heuristic only: a node with at least one peer is
    /// reported `Healthy`, otherwise `Degraded`. It does not ping peers,
    /// check consensus liveness, or detect partitions.
    fn start_health_monitoring(&self) {
        let health = Arc::clone(&self.health);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Coarse peer-count heuristic (see start_health_monitoring docs)
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
        use sha2::{Digest, Sha256};
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
        let coordinator = DistributedCoordinator::new(config)
            .await
            .expect("Failed to create coordinator in test");

        let stats = coordinator.get_stats().await;
        assert_eq!(stats.active_peers, 0);
        assert_eq!(stats.events_processed, 0);
    }

    #[tokio::test]
    async fn test_store_memory() {
        let config = DistributedConfig::default();
        let coordinator = DistributedCoordinator::new(config)
            .await
            .expect("Failed to create coordinator in test");

        let memory = MemoryEntry::new(
            "test".to_string(),
            "test content".to_string(),
            MemoryType::ShortTerm,
        );

        let result = coordinator
            .store_memory(memory, ConsistencyLevel::Eventual)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = DistributedConfig::default();
        let coordinator = DistributedCoordinator::new(config)
            .await
            .expect("Failed to create coordinator in test");

        let health = coordinator.get_health().await;
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.details.contains_key("consensus_state"));
        assert!(health.metrics.contains_key("events_processed"));
    }
}
