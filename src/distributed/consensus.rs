//! Simple consensus implementation for distributed coordination
//!
//! This module provides a simplified consensus algorithm implementation
//! for coordinating distributed memory operations across nodes.

use crate::error::{MemoryError, Result};
use crate::distributed::{NodeId, NodeAddress, ConsensusConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, Instant, interval};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Simple consensus node state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
}

/// Consensus log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Log entry index
    pub index: u64,
    /// Term when entry was created
    pub term: u64,
    /// The operation to apply
    pub operation: Operation,
    /// Timestamp when entry was created
    pub timestamp: DateTime<Utc>,
}

/// Operations that can be replicated via consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Memory operation
    Memory {
        operation_id: Uuid,
        operation_type: String,
        memory_id: Uuid,
        data: Vec<u8>,
    },
    /// Configuration change
    ConfigChange {
        change_type: ConfigChangeType,
        node_id: NodeId,
        address: Option<String>,
    },
    /// No-op operation (for heartbeats)
    NoOp,
}

/// Configuration change types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigChangeType {
    AddNode,
    RemoveNode,
    UpdateNode,
}

/// Simple consensus implementation
pub struct SimpleConsensus {
    /// This node's ID
    node_id: NodeId,
    /// Current state
    state: Arc<RwLock<NodeState>>,
    /// Current term
    current_term: Arc<RwLock<u64>>,
    /// Node voted for in current term
    voted_for: Arc<RwLock<Option<NodeId>>>,
    /// Log entries
    log: Arc<RwLock<Vec<LogEntry>>>,
    /// Index of highest log entry known to be committed
    commit_index: Arc<RwLock<u64>>,
    /// Index of highest log entry applied to state machine
    last_applied: Arc<RwLock<u64>>,
    /// Known peer nodes
    peers: Arc<RwLock<HashMap<NodeId, NodeAddress>>>,
    /// Configuration
    config: ConsensusConfig,
    /// Command channel for external operations
    command_tx: mpsc::UnboundedSender<ConsensusCommand>,
    /// Statistics
    stats: Arc<RwLock<ConsensusStats>>,
}

/// Commands sent to the consensus system
#[derive(Debug)]
pub enum ConsensusCommand {
    /// Propose a new operation
    Propose {
        operation: Operation,
        response_tx: oneshot::Sender<Result<u64>>,
    },
    /// Add a peer node
    AddPeer {
        node_id: NodeId,
        address: NodeAddress,
    },
    /// Remove a peer node
    RemovePeer {
        node_id: NodeId,
    },
    /// Get current leader
    GetLeader {
        response_tx: oneshot::Sender<Option<NodeId>>,
    },
    /// Get consensus statistics
    GetStats {
        response_tx: oneshot::Sender<ConsensusStats>,
    },
}

impl SimpleConsensus {
    /// Create a new simple consensus instance
    pub fn new(node_id: NodeId, config: ConsensusConfig) -> (Self, mpsc::UnboundedReceiver<ConsensusCommand>) {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        
        let consensus = Self {
            node_id,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            current_term: Arc::new(RwLock::new(0)),
            voted_for: Arc::new(RwLock::new(None)),
            log: Arc::new(RwLock::new(Vec::new())),
            commit_index: Arc::new(RwLock::new(0)),
            last_applied: Arc::new(RwLock::new(0)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            config,
            command_tx,
            stats: Arc::new(RwLock::new(ConsensusStats::default())),
        };
        
        (consensus, command_rx)
    }
    
    /// Start the consensus algorithm
    pub async fn start(&self, mut command_rx: mpsc::UnboundedReceiver<ConsensusCommand>) {
        // Start election timer
        let mut election_timer = self.start_election_timer();

        // Start heartbeat timer (for leaders)
        let mut heartbeat_timer = self.start_heartbeat_timer();
        
        // Main consensus loop
        loop {
            tokio::select! {
                // Handle external commands
                Some(command) = command_rx.recv() => {
                    self.handle_command(command).await;
                }
                
                // Election timeout
                _ = election_timer.tick() => {
                    if *self.state.read() != NodeState::Leader {
                        self.start_election().await;
                    }
                }
                
                // Heartbeat timeout (for leaders)
                _ = heartbeat_timer.tick() => {
                    if *self.state.read() == NodeState::Leader {
                        self.send_heartbeats().await;
                    }
                }
            }
        }
    }
    
    /// Handle external commands
    async fn handle_command(&self, command: ConsensusCommand) {
        match command {
            ConsensusCommand::Propose { operation, response_tx } => {
                let result = self.propose_operation(operation).await;
                let _ = response_tx.send(result);
            }
            
            ConsensusCommand::AddPeer { node_id, address } => {
                self.peers.write().insert(node_id, address);
            }
            
            ConsensusCommand::RemovePeer { node_id } => {
                self.peers.write().remove(&node_id);
            }
            
            ConsensusCommand::GetLeader { response_tx } => {
                let leader = if *self.state.read() == NodeState::Leader {
                    Some(self.node_id)
                } else {
                    // In a real implementation, we'd track the current leader
                    None
                };
                let _ = response_tx.send(leader);
            }
            
            ConsensusCommand::GetStats { response_tx } => {
                let stats = self.stats.read().clone();
                let _ = response_tx.send(stats);
            }
        }
    }
    
    /// Propose a new operation to be replicated
    async fn propose_operation(&self, operation: Operation) -> Result<u64> {
        // Only leaders can propose operations
        if *self.state.read() != NodeState::Leader {
            return Err(MemoryError::ConsensusError {
                message: "Only leaders can propose operations".to_string(),
            });
        }
        
        let term = *self.current_term.read();
        let index = {
            let mut log = self.log.write();
            let index = log.len() as u64 + 1;
            log.push(LogEntry {
                index,
                term,
                operation,
                timestamp: Utc::now(),
            });
            index
        };
        
        // In a real implementation, we would replicate to followers here
        // For now, we'll just commit immediately
        *self.commit_index.write() = index;
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.operations_proposed += 1;
            stats.last_operation_time = Some(Utc::now());
        }
        
        Ok(index)
    }
    
    /// Start an election to become leader
    async fn start_election(&self) {
        {
            let mut state = self.state.write();
            *state = NodeState::Candidate;
        }
        
        {
            let mut term = self.current_term.write();
            *term += 1;
        }
        
        {
            let mut voted_for = self.voted_for.write();
            *voted_for = Some(self.node_id);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.elections_started += 1;
        }
        
        // In a real implementation, we would send vote requests to peers
        // For now, we'll just become leader if we have no peers
        if self.peers.read().is_empty() {
            self.become_leader().await;
        }
    }
    
    /// Become the leader
    async fn become_leader(&self) {
        {
            let mut state = self.state.write();
            *state = NodeState::Leader;
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.times_became_leader += 1;
            stats.leader_since = Some(Utc::now());
        }
        
        // Send initial heartbeats
        self.send_heartbeats().await;
    }
    
    /// Send heartbeats to followers
    async fn send_heartbeats(&self) {
        // In a real implementation, we would send append entries RPCs
        // For now, we'll just update statistics
        let mut stats = self.stats.write();
        stats.heartbeats_sent += 1;
        stats.last_heartbeat_time = Some(Utc::now());
    }
    
    /// Create election timer
    fn start_election_timer(&self) -> tokio::time::Interval {
        let timeout = Duration::from_millis(self.config.election_timeout_ms);
        interval(timeout)
    }
    
    /// Create heartbeat timer
    fn start_heartbeat_timer(&self) -> tokio::time::Interval {
        let interval_duration = Duration::from_millis(self.config.heartbeat_interval_ms);
        interval(interval_duration)
    }
    
    /// Get current consensus state
    pub fn get_state(&self) -> ConsensusState {
        ConsensusState {
            node_id: self.node_id,
            state: self.state.read().clone(),
            current_term: *self.current_term.read(),
            commit_index: *self.commit_index.read(),
            last_applied: *self.last_applied.read(),
            log_length: self.log.read().len(),
            peer_count: self.peers.read().len(),
        }
    }
    
    /// Get command sender for external communication
    pub fn command_sender(&self) -> mpsc::UnboundedSender<ConsensusCommand> {
        self.command_tx.clone()
    }
}

/// Current state of the consensus system
#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub node_id: NodeId,
    pub state: NodeState,
    pub current_term: u64,
    pub commit_index: u64,
    pub last_applied: u64,
    pub log_length: usize,
    pub peer_count: usize,
}

/// Consensus statistics
#[derive(Debug, Clone, Default)]
pub struct ConsensusStats {
    pub operations_proposed: u64,
    pub elections_started: u64,
    pub times_became_leader: u64,
    pub heartbeats_sent: u64,
    pub last_operation_time: Option<DateTime<Utc>>,
    pub last_heartbeat_time: Option<DateTime<Utc>>,
    pub leader_since: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_creation() {
        let node_id = NodeId::new();
        let config = ConsensusConfig::default();
        let (consensus, _command_rx) = SimpleConsensus::new(node_id, config);
        
        let state = consensus.get_state();
        assert_eq!(state.node_id, node_id);
        assert_eq!(state.state, NodeState::Follower);
        assert_eq!(state.current_term, 0);
        assert_eq!(state.log_length, 0);
    }

    #[tokio::test]
    async fn test_propose_operation_as_follower() {
        let node_id = NodeId::new();
        let config = ConsensusConfig::default();
        let (consensus, _command_rx) = SimpleConsensus::new(node_id, config);
        
        let operation = Operation::NoOp;
        let result = consensus.propose_operation(operation).await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_become_leader() {
        let node_id = NodeId::new();
        let config = ConsensusConfig::default();
        let (consensus, _command_rx) = SimpleConsensus::new(node_id, config);
        
        consensus.become_leader().await;
        
        let state = consensus.get_state();
        assert_eq!(state.state, NodeState::Leader);
        
        let stats = consensus.stats.read().clone();
        assert_eq!(stats.times_became_leader, 1);
        assert!(stats.leader_since.is_some());
    }

    #[tokio::test]
    async fn test_propose_operation_as_leader() {
        let node_id = NodeId::new();
        let config = ConsensusConfig::default();
        let (consensus, _command_rx) = SimpleConsensus::new(node_id, config);
        
        // Become leader first
        consensus.become_leader().await;
        
        let operation = Operation::Memory {
            operation_id: Uuid::new_v4(),
            operation_type: "create".to_string(),
            memory_id: Uuid::new_v4(),
            data: b"test data".to_vec(),
        };
        
        let result = consensus.propose_operation(operation).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        let state = consensus.get_state();
        assert_eq!(state.log_length, 1);
        assert_eq!(state.commit_index, 1);
    }
}
