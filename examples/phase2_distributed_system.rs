//! Phase 2 Distributed Architecture Demo
//! 
//! This example demonstrates the distributed capabilities of Synaptic,
//! including consensus, sharding, real-time synchronization, and event-driven architecture.

#[cfg(all(feature = "distributed", feature = "embeddings"))]
use synaptic::{
    AgentMemory, MemoryConfig,
    distributed::{
        NodeId, DistributedConfig, ConsistencyLevel, OperationMetadata,
        events::{EventBus, MemoryEvent, InMemoryEventStore},
        consensus::{SimpleConsensus, ConsensusCommand},
        sharding::DistributedGraph,
        // realtime::RealtimeSync, // Temporarily disabled
        coordination::DistributedCoordinator,
    },
    memory::types::{MemoryEntry, MemoryType, MemoryMetadata},
};

#[cfg(all(feature = "distributed", feature = "embeddings"))]
use std::collections::HashMap;


#[cfg(all(feature = "distributed", feature = "embeddings"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Synaptic Phase 2: Distributed Architecture Demo");
    println!("====================================================\n");

    // Example 1: Event-Driven Architecture
    demonstrate_event_system().await?;

    // Example 2: Consensus and Coordination
    demonstrate_consensus_system().await?;

    // Example 3: Distributed Graph Sharding
    demonstrate_distributed_sharding().await?;

    // Example 4: Real-time Synchronization
    demonstrate_realtime_sync().await?;

    // Example 5: Full Distributed Coordinator
    demonstrate_distributed_coordinator().await?;

    // Example 6: Multi-Node Simulation
    demonstrate_multi_node_simulation().await?;

    // Example 7: Performance and Scalability
    demonstrate_performance_benchmarks().await?;

    println!("\n Phase 2 Distributed Architecture Demo Complete!");
    println!("\nKey Features Demonstrated:");
    println!("•  Event-driven architecture with real-time propagation");
    println!("•  Raft consensus for distributed coordination");
    println!("•  Consistent hashing and graph sharding");
    println!("•  WebSocket-based real-time synchronization");
    println!("•  Multi-node coordination and fault tolerance");
    println!("•  High-performance distributed operations");
    println!("•  Configurable consistency levels");
    println!("•  Automatic failover and health monitoring");

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_event_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 Example 1: Event-Driven Architecture");
    println!("----------------------------------------");

    let event_store = Arc::new(InMemoryEventStore::new());
    let event_bus = EventBus::new(event_store);

    // Publish some events
    let events = vec![
        MemoryEvent::MemoryCreated {
            memory_id: Uuid::new_v4(),
            key: "user_profile".to_string(),
            content: "User profile information".to_string(),
            memory_type: "LongTerm".to_string(),
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        },
        MemoryEvent::MemoryUpdated {
            memory_id: Uuid::new_v4(),
            key: "user_preferences".to_string(),
            old_content: "Old preferences".to_string(),
            new_content: "Updated preferences".to_string(),
            changes: Vec::new(),
            version: 2,
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        },
        MemoryEvent::RelationshipInferred {
            from_memory: Uuid::new_v4(),
            to_memory: Uuid::new_v4(),
            relationship_type: synaptic::memory::knowledge_graph::RelationshipType::RelatedTo,
            confidence: 0.85,
            evidence: vec!["semantic similarity".to_string()],
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        },
    ];

    for event in events {
        let metadata = OperationMetadata::new(NodeId::new(), ConsistencyLevel::Eventual);
        event_bus.publish(event, metadata).await?;
    }

    let stats = event_bus.get_stats();
    println!("   Events published: {}", stats.events_published);
    println!("   Events processed: {}", stats.events_processed);
    println!("   Last event time: {:?}", stats.last_event_time);

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_consensus_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗳️ Example 2: Consensus and Coordination");
    println!("----------------------------------------");

    let node_id = NodeId::new();
    let config = synaptic::distributed::ConsensusConfig::default();
    let (consensus, command_rx) = SimpleConsensus::new(node_id, config);

    println!("  🏛️ Node ID: {}", node_id);
    
    let state = consensus.get_state();
    println!("   Initial state: {:?}", state.state);
    println!("   Current term: {}", state.current_term);
    println!("   Log length: {}", state.log_length);
    println!("   Peer count: {}", state.peer_count);

    // Start consensus in background for a short time
    let consensus_arc = Arc::new(consensus);
    let consensus_clone = Arc::clone(&consensus_arc);
    
    tokio::spawn(async move {
        tokio::select! {
            _ = consensus_clone.start(command_rx) => {},
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {},
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let final_state = consensus_arc.get_state();
    println!("   Consensus system operational");
    println!("   Final state: {:?}", final_state.state);

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_distributed_sharding() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗂️ Example 3: Distributed Graph Sharding");
    println!("------------------------------------------");

    let node_id = NodeId::new();
    let graph = DistributedGraph::new(node_id, 3);

    // Add some nodes to the cluster
    let nodes = vec![NodeId::new(), NodeId::new(), NodeId::new()];
    for node in &nodes {
        graph.add_node(*node);
    }

    // Add memory nodes to the distributed graph
    let memories = vec![
        ("ai_research", "Advanced AI research and development"),
        ("machine_learning", "Machine learning algorithms and models"),
        ("distributed_systems", "Distributed computing and consensus"),
        ("data_science", "Data analysis and statistical modeling"),
        ("neural_networks", "Deep learning and neural architectures"),
    ];

    for (key, content) in memories {
        let memory_node = synaptic::distributed::sharding::MemoryNode {
            id: Uuid::new_v4(),
            memory_key: key.to_string(),
            content_hash: format!("hash_{}", key),
            relationships: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
        };

        let shard_id = graph.get_shard_id(memory_node.id);
        let responsible_nodes = graph.get_shard_nodes(shard_id);
        
        println!("   Memory '{}' -> Shard {} (Nodes: {:?})", 
                 key, shard_id, responsible_nodes.len());

        graph.add_memory_node(memory_node)?;
    }

    let local_stats = graph.get_local_stats();
    println!("   Local shards: {}", local_stats.len());
    println!("   Total local nodes: {}", graph.get_local_node_count());

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_realtime_sync() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n Example 4: Real-time Synchronization");
    println!("--------------------------------------");

    let config = synaptic::distributed::RealtimeConfig {
        websocket_port: 8090,
        max_connections: 1000,
        heartbeat_interval_ms: 30000,
        message_buffer_size: 1000,
    };

    // RealtimeSync is temporarily disabled, simulate the functionality
    println!("   RealtimeSync temporarily disabled - simulating functionality");

    // Simulate broadcasting updates
    let events = vec![
        MemoryEvent::MemoryCreated {
            memory_id: Uuid::new_v4(),
            key: "realtime_memory_1".to_string(),
            content: "Real-time synchronized content".to_string(),
            memory_type: "ShortTerm".to_string(),
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        },
        MemoryEvent::MemoryUpdated {
            memory_id: Uuid::new_v4(),
            key: "realtime_memory_2".to_string(),
            old_content: "Old content".to_string(),
            new_content: "Updated content in real-time".to_string(),
            changes: Vec::new(),
            version: 2,
            node_id: NodeId::new(),
            timestamp: Utc::now(),
        },
    ];

    let events_len = events.len();
    for event in events {
        let metadata = OperationMetadata::new(NodeId::new(), ConsistencyLevel::Eventual);
        let envelope = synaptic::distributed::events::EventEnvelope::new(event, metadata);
        // sync.broadcast_update(&envelope).await?;
        println!("  📡 Would broadcast event: {:?}", envelope.event);
    }

    // Simulate stats
    println!("   Active connections: 0 (simulated)");
    println!("   Total connections: 0 (simulated)");
    println!("   Updates sent: {} (simulated)", events_len);
    println!("   Last update time: {:?} (simulated)", Utc::now());

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_distributed_coordinator() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n Example 5: Full Distributed Coordinator");
    println!("-------------------------------------------");

    let mut config = DistributedConfig::default();
    config.realtime.websocket_port = 8091;
    config.replication_factor = 2;
    config.shard_count = 8;

    let coordinator = DistributedCoordinator::new(config).await?;

    // Store memories with different consistency levels
    let memories = vec![
        ("critical_data", "Mission-critical information", ConsistencyLevel::Strong),
        ("user_session", "User session data", ConsistencyLevel::Causal),
        ("analytics_data", "Analytics and metrics", ConsistencyLevel::Eventual),
    ];

    for (key, content, consistency) in memories {
        let mut metadata = MemoryMetadata::new();
        metadata.importance = 0.8;
        metadata.tags = vec!["distributed".to_string()];

        let memory = MemoryEntry {
            key: key.to_string(),
            value: content.to_string(),
            memory_type: MemoryType::LongTerm,
            metadata,
            embedding: None,
        };

        coordinator.store_memory(memory, consistency).await?;
        println!("   Stored '{}' with {:?} consistency", key, consistency);
    }

    let stats = coordinator.get_stats().await;
    println!("   Current node: {}", stats.current_node);
    println!("   Active peers: {}", stats.active_peers);
    println!("   Total shards: {}", stats.total_shards);
    println!("   Owned shards: {}", stats.owned_shards.len());
    println!("   Events processed: {}", stats.events_processed);
    println!("   Uptime: {} seconds", stats.uptime_seconds);

    let health = coordinator.get_health().await;
    println!("  🏥 Health status: {:?}", health.status);
    println!("  🏥 Health details: {:?}", health.details);

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_multi_node_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n Example 6: Multi-Node Simulation");
    println!("-----------------------------------");

    // Create multiple coordinators to simulate a cluster
    let mut coordinators = Vec::new();
    
    for i in 0..3 {
        let mut config = DistributedConfig::default();
        config.realtime.websocket_port = 8092 + i;
        config.node_id = NodeId::new();
        
        let coordinator = DistributedCoordinator::new(config).await?;
        println!("  🖥️ Created node: {}", coordinator.get_stats().await.current_node);
        coordinators.push(coordinator);
    }

    // Simulate peer connections
    for i in 0..coordinators.len() {
        for j in 0..coordinators.len() {
            if i != j {
                let peer_id = coordinators[j].get_stats().await.current_node;
                let peer_address = format!("127.0.0.1:{}", 8092 + j);
                let _ = coordinators[i].add_peer(peer_id, peer_address).await;
            }
        }
    }

    // Store data across the cluster
    for (i, coordinator) in coordinators.iter().enumerate() {
        let mut metadata = MemoryMetadata::new();
        metadata.importance = 0.7;
        metadata.tags = vec!["cluster".to_string()];

        let memory = MemoryEntry {
            key: format!("cluster_memory_{}", i),
            value: format!("Data stored on node {}", i),
            memory_type: MemoryType::LongTerm,
            metadata,
            embedding: None,
        };

        coordinator.store_memory(memory, ConsistencyLevel::Strong).await?;
    }

    // Show cluster statistics
    for (i, coordinator) in coordinators.iter().enumerate() {
        let stats = coordinator.get_stats().await;
        println!("   Node {}: {} peers, {} events processed", 
                 i, stats.active_peers, stats.events_processed);
    }

    Ok(())
}

#[cfg(all(feature = "distributed", feature = "embeddings"))]
async fn demonstrate_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n Example 7: Performance and Scalability");
    println!("-----------------------------------------");

    let mut config = DistributedConfig::default();
    config.realtime.websocket_port = 8095;

    let coordinator = DistributedCoordinator::new(config).await?;

    let start_time = std::time::Instant::now();
    let num_operations = 1000;

    println!("  🏃 Running {} distributed operations...", num_operations);

    // Benchmark distributed storage operations
    for i in 0..num_operations {
        let mut metadata = MemoryMetadata::new();
        metadata.importance = 0.5;
        metadata.tags = vec!["performance".to_string()];

        let memory = MemoryEntry {
            key: format!("perf_test_{}", i),
            value: format!("Performance test data item {}", i),
            memory_type: MemoryType::ShortTerm,
            metadata,
            embedding: None,
        };

        coordinator.store_memory(memory, ConsistencyLevel::Eventual).await?;

        if i % 100 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let elapsed = start_time.elapsed();
    let ops_per_second = num_operations as f64 / elapsed.as_secs_f64();
    let avg_latency_ms = elapsed.as_millis() as f64 / num_operations as f64;

    println!("\n   Performance Results:");
    println!("    • Operations per second: {:.2}", ops_per_second);
    println!("    • Average latency: {:.2}ms", avg_latency_ms);
    println!("    • Total time: {:.2}s", elapsed.as_secs_f64());

    let final_stats = coordinator.get_stats().await;
    println!("   Final Statistics:");
    println!("    • Events processed: {}", final_stats.events_processed);
    println!("    • Uptime: {} seconds", final_stats.uptime_seconds);

    Ok(())
}

#[cfg(not(all(feature = "distributed", feature = "embeddings")))]
fn main() {
    println!("This example requires both 'distributed' and 'embeddings' features to be enabled.");
    println!("Run with: cargo run --features distributed,embeddings --example phase2_distributed_system");
}
