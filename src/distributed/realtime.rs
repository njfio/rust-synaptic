//! Real-time synchronization for distributed memory systems
//! 
//! This module provides WebSocket-based real-time updates and
//! live synchronization across distributed memory nodes.

use crate::error::{MemoryError, Result};
use crate::distributed::{NodeId, RealtimeConfig};
use crate::distributed::events::{MemoryEvent, EventEnvelope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::{accept_async, WebSocketStream};
use tokio_tungstenite::tungstenite::Message;
use futures_util::{SinkExt, StreamExt};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::net::SocketAddr;

/// Client connection identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientId(pub Uuid);

impl ClientId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for ClientId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "client-{}", self.0.to_string()[..8].to_lowercase())
    }
}

/// Real-time update message sent to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveUpdate {
    /// Update identifier
    pub update_id: Uuid,
    /// Type of update
    pub update_type: UpdateType,
    /// Affected memory IDs
    pub affected_memories: Vec<Uuid>,
    /// Change details
    pub changes: ChangeSet,
    /// Source node that generated the update
    pub source_node: NodeId,
    /// Timestamp when update occurred
    pub timestamp: DateTime<Utc>,
    /// Sequence number for ordering
    pub sequence: u64,
}

/// Types of real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    MemoryCreated,
    MemoryUpdated,
    MemoryDeleted,
    RelationshipAdded,
    RelationshipRemoved,
    PatternDetected,
    GraphRestructured,
    NodeStatusChanged,
}

/// Set of changes in an update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSet {
    /// Individual changes
    pub changes: Vec<Change>,
    /// Summary of the changeset
    pub summary: String,
    /// Total number of affected items
    pub affected_count: usize,
}

/// Individual change within an update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    /// Type of change
    pub change_type: String,
    /// Path to the changed field
    pub path: String,
    /// Old value (if applicable)
    pub old_value: Option<serde_json::Value>,
    /// New value (if applicable)
    pub new_value: Option<serde_json::Value>,
}

/// Client subscription preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// Client identifier
    pub client_id: ClientId,
    /// Types of updates to receive
    pub update_types: Vec<UpdateType>,
    /// Memory IDs to watch (empty = all)
    pub memory_filters: Vec<Uuid>,
    /// Node IDs to watch (empty = all)
    pub node_filters: Vec<NodeId>,
    /// Minimum update sequence to receive
    pub min_sequence: u64,
}

/// Active client connection
pub struct ClientConnection {
    /// Client identifier
    pub client_id: ClientId,
    /// WebSocket stream
    pub websocket: Arc<RwLock<WebSocketStream<tokio::net::TcpStream>>>,
    /// Client subscription
    pub subscription: Arc<RwLock<Subscription>>,
    /// Connection metadata
    pub metadata: ConnectionMetadata,
    /// Message queue for this client
    pub message_queue: mpsc::UnboundedSender<LiveUpdate>,
}

/// Connection metadata
#[derive(Debug, Clone)]
pub struct ConnectionMetadata {
    pub client_id: ClientId,
    pub remote_addr: SocketAddr,
    pub connected_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub user_agent: Option<String>,
    pub protocol_version: String,
}

/// Real-time synchronization server
pub struct RealtimeSync {
    /// Server configuration
    config: RealtimeConfig,
    /// Active client connections
    connections: Arc<RwLock<HashMap<ClientId, ClientConnection>>>,
    /// Broadcast channel for updates
    update_sender: broadcast::Sender<LiveUpdate>,
    /// Statistics
    stats: Arc<RwLock<RealtimeStats>>,
    /// Sequence counter for updates
    sequence_counter: Arc<RwLock<u64>>,
}

impl RealtimeSync {
    /// Create a new real-time sync server
    pub fn new(config: RealtimeConfig) -> Self {
        let (update_sender, _) = broadcast::channel(10000);
        
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            update_sender,
            stats: Arc::new(RwLock::new(RealtimeStats::default())),
            sequence_counter: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Start the real-time sync server
    pub async fn start(&self) -> Result<()> {
        let addr = format!("127.0.0.1:{}", self.config.websocket_port);
        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| MemoryError::NetworkError { 
                message: format!("Failed to bind to {}: {}", addr, e) 
            })?;
        
        tracing::info!(
            component = "realtime_sync",
            operation = "start_server",
            address = %addr,
            "Real-time sync server listening"
        );
        
        // Start heartbeat task
        self.start_heartbeat_task();
        
        // Accept connections
        while let Ok((stream, addr)) = listener.accept().await {
            let connections = Arc::clone(&self.connections);
            let stats = Arc::clone(&self.stats);
            let update_sender = self.update_sender.clone();
            
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, addr, connections, stats, update_sender).await {
                    tracing::error!(
                        component = "realtime_sync",
                        operation = "handle_connection",
                        client_addr = %addr,
                        error = %e,
                        "Error handling client connection"
                    );
                }
            });
        }
        
        Ok(())
    }
    
    /// Handle a new WebSocket connection
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        addr: SocketAddr,
        connections: Arc<RwLock<HashMap<ClientId, ClientConnection>>>,
        stats: Arc<RwLock<RealtimeStats>>,
        update_sender: broadcast::Sender<LiveUpdate>,
    ) -> Result<()> {
        let websocket = accept_async(stream).await
            .map_err(|e| MemoryError::NetworkError { 
                message: format!("WebSocket handshake failed: {}", e) 
            })?;
        
        let client_id = ClientId::new();
        let (message_tx, mut message_rx) = mpsc::unbounded_channel();
        
        let metadata = ConnectionMetadata {
            client_id,
            remote_addr: addr,
            connected_at: Utc::now(),
            last_activity: Utc::now(),
            user_agent: None,
            protocol_version: "1.0".to_string(),
        };
        
        let subscription = Subscription {
            client_id,
            update_types: vec![
                UpdateType::MemoryCreated,
                UpdateType::MemoryUpdated,
                UpdateType::MemoryDeleted,
            ],
            memory_filters: Vec::new(),
            node_filters: Vec::new(),
            min_sequence: 0,
        };
        
        let connection = ClientConnection {
            client_id,
            websocket: Arc::new(RwLock::new(websocket)),
            subscription: Arc::new(RwLock::new(subscription)),
            metadata,
            message_queue: message_tx,
        };
        
        // Register connection
        {
            let mut conns = connections.write();
            conns.insert(client_id, connection);
        }
        
        // Update statistics
        {
            let mut stats_guard = stats.write();
            stats_guard.active_connections += 1;
            stats_guard.total_connections += 1;
        }
        
        tracing::info!(
            component = "realtime_sync",
            operation = "client_connected",
            client_id = %client_id,
            client_addr = %addr,
            "New client connected"
        );
        
        // Subscribe to updates
        let mut update_receiver = update_sender.subscribe();
        
        // Handle messages
        tokio::select! {
            // Send updates to client
            _ = async {
                while let Ok(update) = update_receiver.recv().await {
                    if let Err(_) = message_tx.send(update) {
                        break; // Client disconnected
                    }
                }
            } => {},
            
            // Process outgoing messages
            _ = async {
                while let Some(update) = message_rx.recv().await {
                    match serde_json::to_string(&update) {
                        Ok(message) => {
                            // Send to client (in a real implementation, we'd handle this properly)
                            // For now, we'll just log it
                            tracing::debug!(
                                component = "realtime_sync",
                                operation = "send_update",
                                client_id = %client_id,
                                message_size = %message.len(),
                                "Sending update to client"
                            );
                        }
                        Err(e) => {
                            tracing::error!("Failed to serialize update for client {}: {}", client_id, e);
                            break; // Disconnect client on serialization error
                        }
                    }
                }
            } => {},
        }
        
        // Clean up connection
        {
            let mut conns = connections.write();
            conns.remove(&client_id);
        }
        
        {
            let mut stats_guard = stats.write();
            stats_guard.active_connections = stats_guard.active_connections.saturating_sub(1);
        }
        
        tracing::info!(
            component = "realtime_sync",
            operation = "client_disconnected",
            client_id = %client_id,
            "Client disconnected"
        );
        
        Ok(())
    }
    
    /// Broadcast an update to all connected clients
    pub async fn broadcast_update(&self, event: &EventEnvelope) -> Result<()> {
        let update = self.event_to_update(event)?;
        
        // Send to all subscribers
        if let Err(_) = self.update_sender.send(update.clone()) {
            // No subscribers, that's okay
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.updates_sent += 1;
            stats.last_update_time = Some(Utc::now());
        }
        
        Ok(())
    }
    
    /// Convert an event to a live update
    fn event_to_update(&self, envelope: &EventEnvelope) -> Result<LiveUpdate> {
        let sequence = {
            let mut counter = self.sequence_counter.write();
            *counter += 1;
            *counter
        };
        
        let (update_type, affected_memories, changes) = match &envelope.event {
            MemoryEvent::MemoryCreated { memory_id, key, content, .. } => {
                let changes = ChangeSet {
                    changes: vec![Change {
                        change_type: "created".to_string(),
                        path: "memory".to_string(),
                        old_value: None,
                        new_value: Some(serde_json::json!({
                            "key": key,
                            "content": content
                        })),
                    }],
                    summary: format!("Memory '{}' created", key),
                    affected_count: 1,
                };
                (UpdateType::MemoryCreated, vec![*memory_id], changes)
            },
            
            MemoryEvent::MemoryUpdated { memory_id, key, old_content, new_content, .. } => {
                let changes = ChangeSet {
                    changes: vec![Change {
                        change_type: "updated".to_string(),
                        path: "content".to_string(),
                        old_value: Some(serde_json::Value::String(old_content.clone())),
                        new_value: Some(serde_json::Value::String(new_content.clone())),
                    }],
                    summary: format!("Memory '{}' updated", key),
                    affected_count: 1,
                };
                (UpdateType::MemoryUpdated, vec![*memory_id], changes)
            },
            
            MemoryEvent::MemoryDeleted { memory_id, key, .. } => {
                let changes = ChangeSet {
                    changes: vec![Change {
                        change_type: "deleted".to_string(),
                        path: "memory".to_string(),
                        old_value: Some(serde_json::Value::String(key.clone())),
                        new_value: None,
                    }],
                    summary: format!("Memory '{}' deleted", key),
                    affected_count: 1,
                };
                (UpdateType::MemoryDeleted, vec![*memory_id], changes)
            },
            
            _ => {
                // Handle other event types
                let changes = ChangeSet {
                    changes: Vec::new(),
                    summary: "Other event".to_string(),
                    affected_count: 0,
                };
                (UpdateType::NodeStatusChanged, Vec::new(), changes)
            },
        };
        
        Ok(LiveUpdate {
            update_id: Uuid::new_v4(),
            update_type,
            affected_memories,
            changes,
            source_node: envelope.metadata.source_node,
            timestamp: envelope.metadata.timestamp,
            sequence,
        })
    }
    
    /// Start heartbeat task to keep connections alive
    fn start_heartbeat_task(&self) {
        let connections = Arc::clone(&self.connections);
        let heartbeat_interval = self.config.heartbeat_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(heartbeat_interval)
            );
            
            loop {
                interval.tick().await;
                
                // Send heartbeat to all connections
                let conns = connections.read();
                for (client_id, _connection) in conns.iter() {
                    // In a real implementation, we'd send a ping frame
                    tracing::trace!(
                        component = "realtime_sync",
                        operation = "heartbeat",
                        client_id = %client_id,
                        "Sending heartbeat to client"
                    );
                }
            }
        });
    }
    
    /// Get real-time sync statistics
    pub fn get_stats(&self) -> RealtimeStats {
        self.stats.read().clone()
    }
    
    /// Get list of connected clients
    pub fn get_connected_clients(&self) -> Vec<ClientId> {
        self.connections.read().keys().cloned().collect()
    }
}

/// Real-time synchronization statistics
#[derive(Debug, Clone, Default)]
pub struct RealtimeStats {
    pub active_connections: usize,
    pub total_connections: u64,
    pub updates_sent: u64,
    pub messages_received: u64,
    pub last_update_time: Option<DateTime<Utc>>,
    pub server_start_time: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::OperationMetadata;
    use crate::distributed::ConsistencyLevel;

    #[test]
    fn test_client_id_creation() {
        let client1 = ClientId::new();
        let client2 = ClientId::new();
        
        assert_ne!(client1, client2);
        assert_ne!(client1.to_string(), client2.to_string());
    }

    #[test]
    fn test_realtime_sync_creation() {
        let config = RealtimeConfig::default();
        let sync = RealtimeSync::new(config);
        
        let stats = sync.get_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_connections, 0);
    }

    #[test]
    fn test_event_to_update_conversion() {
        let config = RealtimeConfig::default();
        let sync = RealtimeSync::new(config);
        
        let event = MemoryEvent::MemoryCreated {
            memory_id: Uuid::new_v4(),
            key: "test".to_string(),
            content: "content".to_string(),
            memory_type: "ShortTerm".to_string(),
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        };
        
        let metadata = OperationMetadata::new(NodeId::new(), ConsistencyLevel::Eventual);
        let envelope = crate::distributed::events::EventEnvelope::new(event, metadata);
        
        let update = sync.event_to_update(&envelope).expect("Failed to convert event to update in test");

        assert!(matches!(update.update_type, UpdateType::MemoryCreated));
        assert_eq!(update.affected_memories.len(), 1);
        assert_eq!(update.changes.affected_count, 1);
    }
}
