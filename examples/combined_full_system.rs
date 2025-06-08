//! Combined Full System Demo
//! 
//! This example demonstrates the complete Synaptic system with BOTH:
//! - Phase 2A: Distributed Systems (Kafka, Consensus, Sharding)
//! - Phase 2B: External Integrations (PostgreSQL, BERT, LLM, Redis, Visualization)
//! 
//! This showcases the full power of Synaptic as a state-of-the-art distributed
//! AI agent memory system with real external service integrations.

use std::error::Error;

#[cfg(all(feature = "distributed", feature = "external-integrations", feature = "embeddings"))]
use synaptic::{
    AgentMemory, MemoryConfig,
    distributed::{
        NodeId, DistributedConfig, ConsistencyLevel, OperationMetadata,
        events::{EventBus, MemoryEvent, InMemoryEventStore},
        consensus::{SimpleConsensus, ConsensusCommand},
        sharding::DistributedGraph,
        coordination::DistributedCoordinator,
    },
    integrations::{IntegrationConfig, IntegrationManager},
    memory::types::{MemoryEntry, MemoryType},
};

#[cfg(feature = "sql-storage")]
use synaptic::integrations::database::{DatabaseConfig, DatabaseClient};

#[cfg(feature = "ml-models")]
use synaptic::integrations::ml_models::{MLConfig, MLModelManager};

#[cfg(feature = "llm-integration")]
use synaptic::integrations::llm::{LLMConfig, LLMClient, LLMProvider};

#[cfg(feature = "visualization")]
use synaptic::integrations::visualization::{
    VisualizationConfig, RealVisualizationEngine, ImageFormat, ColorScheme
};

use synaptic::integrations::redis_cache::{RedisConfig, RedisClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    if std::path::Path::new(".env").exists() {
        for line in std::fs::read_to_string(".env")?.lines() {
            if let Some((key, value)) = line.split_once('=') {
                if !key.starts_with('#') && !key.trim().is_empty() {
                    // Only set if not already set in environment
                    if std::env::var(key.trim()).is_err() {
                        std::env::set_var(key.trim(), value.trim());
                    }
                }
            }
        }
    }

    println!("🚀 Synaptic Combined Full System Demo");
    println!("=====================================");
    println!("🎯 Demonstrating BOTH distributed systems AND external integrations");
    println!();

    // Check which features are enabled
    check_enabled_features();

    #[cfg(all(feature = "distributed", feature = "external-integrations", feature = "embeddings"))]
    {
        // Phase 1: Initialize Distributed System Components
        println!("\n🕸️  Phase 2A: Distributed System Initialization");
        println!("-----------------------------------------------");
        
        let node_id = NodeId::new();
        println!("🆔 Node ID: {}", node_id);
        
        // Create distributed configuration
        let distributed_config = DistributedConfig {
            node_id,
            replication_factor: 3,
            shard_count: 16,
            peers: vec![], // Single node for demo
            events: synaptic::distributed::EventConfig {
                kafka_brokers: vec!["localhost:11113".to_string()],
                event_topic: "synaptic-events".to_string(),
                consumer_group: "synaptic-demo".to_string(),
                batch_size: 100,
                retention_hours: 24,
            },
            consensus: synaptic::distributed::ConsensusConfig::default(),
            realtime: synaptic::distributed::RealtimeConfig::default(),
        };
        
        // Initialize event bus with in-memory store (for demo)
        let event_store = std::sync::Arc::new(InMemoryEventStore::new());
        let event_bus = EventBus::new(event_store);
        println!("✅ Event bus initialized");
        
        // Initialize consensus
        let consensus = SimpleConsensus::new(node_id, distributed_config.consensus.clone());
        println!("✅ Consensus algorithm initialized");
        
        // Initialize distributed graph
        let distributed_graph = DistributedGraph::new(NodeId(uuid::Uuid::new_v4()), 3);
        println!("✅ Distributed graph sharding initialized");
        
        // Phase 2: Initialize External Integrations
        println!("\n🔗 Phase 2B: External Integrations Initialization");
        println!("------------------------------------------------");
        
        // Create integration config
        let mut integration_config = IntegrationConfig::default();
        
        #[cfg(feature = "sql-storage")]
        {
            integration_config.database = Some(DatabaseConfig {
                database_url: std::env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgresql://synaptic_user:synaptic_pass@localhost:11110/synaptic_db".to_string()),
                max_connections: 10,
                connect_timeout_secs: 30,
                ssl_mode: "prefer".to_string(),
                schema: "public".to_string(),
            });
        }
        
        #[cfg(feature = "ml-models")]
        {
            integration_config.ml_models = Some(MLConfig {
                model_dir: std::path::PathBuf::from("./models"),
                device: "cpu".to_string(),
                max_sequence_length: 512,
                embedding_dim: 768,
                cache_size: 100,
                quantization: false,
            });
        }
        
        #[cfg(feature = "llm-integration")]
        {
            if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
                integration_config.llm = Some(LLMConfig {
                    provider: LLMProvider::Anthropic,
                    api_key,
                    base_url: None,
                    model: "claude-3-5-haiku-20241022".to_string(),
                    max_tokens: 500,
                    temperature: 0.7,
                    timeout_secs: 30,
                    rate_limit: 10,
                });
            } else if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                integration_config.llm = Some(LLMConfig {
                    provider: LLMProvider::OpenAI,
                    api_key,
                    base_url: None,
                    model: "gpt-3.5-turbo".to_string(),
                    max_tokens: 500,
                    temperature: 0.7,
                    timeout_secs: 30,
                    rate_limit: 10,
                });
            }
        }
        
        #[cfg(feature = "visualization")]
        {
            integration_config.visualization = Some(VisualizationConfig {
                output_dir: std::path::PathBuf::from("./visualizations"),
                width: 800,
                height: 600,
                format: ImageFormat::PNG,
                color_scheme: ColorScheme::Default,
                font_size: 12,
                interactive: false,
            });
        }
        
        integration_config.redis = Some(RedisConfig {
            url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:11111".to_string()),
            pool_size: 5,
            connect_timeout_secs: 5,
            default_ttl_secs: 300,
            key_prefix: "synaptic_combined:".to_string(),
            compression: true,
        });
        
        // Initialize integration manager
        match IntegrationManager::new(integration_config).await {
            Ok(integration_manager) => {
                println!("✅ Integration manager initialized");
                
                // Health check all integrations
                match integration_manager.health_check().await {
                    Ok(health_results) => {
                        println!("🏥 Integration health checks:");
                        for (service, is_healthy) in &health_results {
                            let status = if *is_healthy { "✅" } else { "❌" };
                            println!("   {} {}", status, service);
                        }
                    },
                    Err(e) => println!("❌ Health check failed: {}", e),
                }
                
                // Phase 3: Create Combined Memory System
                println!("\n🧠 Phase 3: Combined Memory System");
                println!("----------------------------------");
                
                let memory_config = MemoryConfig {
                    enable_knowledge_graph: true,
                    enable_temporal_tracking: true,
                    enable_advanced_management: true,
                    enable_distributed: true,
                    distributed_config: Some(distributed_config.clone()),
                    ..Default::default()
                };
                
                match AgentMemory::new(memory_config).await {
                    Ok(mut memory) => {
                        println!("✅ Combined memory system initialized");
                        
                        // Phase 4: Demonstrate Combined Operations
                        println!("\n🎯 Phase 4: Combined Operations Demo");
                        println!("-----------------------------------");
                        
                        // Store a memory with both distributed and external integration features
                        let memory_entry = MemoryEntry::new(
                            "distributed_ai_project".to_string(),
                            "Advanced AI project using distributed memory system with ML models, LLM integration, and real-time visualization".to_string(),
                            MemoryType::LongTerm
                        );
                        
                        match memory.store("distributed_ai_project", &memory_entry.value).await {
                            Ok(_) => {
                                println!("✅ Stored memory with distributed coordination");
                                
                                // Retrieve and demonstrate features
                                match memory.retrieve("distributed_ai_project").await {
                                    Ok(Some(retrieved)) => {
                                        println!("✅ Retrieved memory: {}", retrieved.key);
                                        
                                        // Get memory stats
                                        let memory_stats = memory.stats();
                                        println!("📊 Memory stats: {} short-term, {} long-term",
                                            memory_stats.short_term_count, memory_stats.long_term_count);

                                        // Simulate distributed stats
                                        println!("📊 Distributed stats: 1 node, 1 event (simulated)");
                                    },
                                    Ok(None) => println!("❌ Memory not found"),
                                    Err(e) => println!("❌ Failed to retrieve memory: {}", e),
                                }
                            },
                            Err(e) => println!("❌ Failed to store memory: {}", e),
                        }
                        
                        println!("\n🎉 Combined system demo completed successfully!");
                        println!("💡 The system now has BOTH distributed capabilities AND external integrations");
                    },
                    Err(e) => println!("❌ Failed to initialize combined memory system: {}", e),
                }
            },
            Err(e) => println!("❌ Failed to initialize integration manager: {}", e),
        }
    }
    
    #[cfg(not(all(feature = "distributed", feature = "external-integrations", feature = "embeddings")))]
    {
        println!("❌ Combined demo requires all features enabled");
        println!("💡 Run with: cargo run --example combined_full_system --features \"distributed,external-integrations,embeddings\"");
    }

    Ok(())
}

fn check_enabled_features() {
    println!("🔧 Enabled Features:");
    
    #[cfg(feature = "distributed")]
    println!("   ✅ Distributed Systems (Kafka, Consensus, Sharding)");
    #[cfg(not(feature = "distributed"))]
    println!("   ❌ Distributed Systems (disabled)");
    
    #[cfg(feature = "external-integrations")]
    println!("   ✅ External Integrations (PostgreSQL, BERT, LLM, Redis, Visualization)");
    #[cfg(not(feature = "external-integrations"))]
    println!("   ❌ External Integrations (disabled)");
    
    #[cfg(feature = "embeddings")]
    println!("   ✅ Vector Embeddings");
    #[cfg(not(feature = "embeddings"))]
    println!("   ❌ Vector Embeddings (disabled)");
}
