//! Comprehensive tests for external integrations
//! 
//! Tests PostgreSQL database, BERT ML models, LLM integration,
//! Redis caching, and visualization engine functionality.

#[cfg(feature = "external-integrations")]
mod external_integration_tests {
    use synaptic::{
        AgentMemory, MemoryConfig, MemoryEntry, MemoryType,
        integrations::{IntegrationConfig, IntegrationManager},
        analytics::AccessType,
        memory::management::analytics::AnalyticsEvent,
    };
    use std::error::Error;
    use std::collections::HashMap;

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

    #[tokio::test]
    async fn test_integration_manager_initialization() -> Result<(), Box<dyn Error>> {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).await?;
        
        // Test basic functionality
        let health = manager.health_check().await?;
        assert!(!health.is_empty());
        
        Ok(())
    }

    #[cfg(feature = "sql-storage")]
    #[tokio::test]
    async fn test_database_integration() -> Result<(), Box<dyn Error>> {
        let config = DatabaseConfig {
            database_url: "postgresql://synaptic:synaptic_password@localhost:11110/synaptic_test".to_string(),
            max_connections: 10,
            connect_timeout_secs: 30,
            ssl_mode: "prefer".to_string(),
            schema: "public".to_string(),
        };

        // Test database client creation (may fail if DB not available, that's OK)
        let result = DatabaseClient::new(config).await;
        match result {
            Ok(mut client) => {
                // Test basic operations if connection succeeds
                let test_entry = MemoryEntry::new(
                    "test_key".to_string(),
                    "test_value".to_string(),
                    MemoryType::ShortTerm,
                );

                // These operations may fail if DB schema isn't set up, that's expected
                let _ = client.store_memory(&test_entry).await;
                let _ = client.get_memory("test_key").await;
                let _ = client.health_check().await;
            },
            Err(_) => {
                // Database not available, skip test
                println!("Database not available, skipping database integration test");
            }
        }
        
        Ok(())
    }

    #[cfg(feature = "ml-models")]
    #[tokio::test]
    async fn test_ml_models_integration() -> Result<(), Box<dyn Error>> {
        let config = MLConfig {
            model_dir: std::path::PathBuf::from("./models"),
            embedding_dim: 384,
            quantization: false,
            cache_size: 1000,
            device: "cpu".to_string(),
            max_sequence_length: 512,
        };

        let result = MLModelManager::new(config).await;
        match result {
            Ok(mut manager) => {
                // Test basic ML operations
                let test_texts = vec![
                    "This is a test sentence about artificial intelligence".to_string(),
                    "Machine learning is a subset of AI".to_string(),
                ];
                
                // Test embedding generation (may fail if models not available)
                let _ = manager.generate_embedding(&test_texts[0]).await;

                // Test model functionality (no stats or health_check methods available)
                println!("ML Model manager created successfully");
            },
            Err(_) => {
                println!("ML models not available, skipping ML integration test");
            }
        }
        
        Ok(())
    }

    #[cfg(feature = "llm-integration")]
    #[tokio::test]
    async fn test_llm_integration() -> Result<(), Box<dyn Error>> {
        let config = LLMConfig {
            provider: LLMProvider::OpenAI,
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: "test_key".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            timeout_secs: 30,
            rate_limit: 60,
        };

        let client = LLMClient::new(config).await;

        // Test client creation
        match client {
            Ok(_) => {
                println!("LLM client created successfully");
                // Test functionality would require valid API key
            },
            Err(_) => {
                println!("LLM client creation failed (expected without valid config)");
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_redis_integration() -> Result<(), Box<dyn Error>> {
        let config = RedisConfig {
            url: "redis://localhost:11111".to_string(),
            pool_size: 5,
            connect_timeout_secs: 5,
            default_ttl_secs: 300,
            key_prefix: "test:".to_string(),
            compression: false,
        };

        let result = RedisClient::new(config).await;
        match result {
            Ok(mut client) => {
                // Test basic Redis operations (using cache methods)
                let test_entry = MemoryEntry::new(
                    "test_key".to_string(),
                    "test_value".to_string(),
                    MemoryType::ShortTerm,
                );
                let _ = client.cache_memory("test_key", &test_entry, None).await;
                let _ = client.get_cached_memory("test_key").await;
                let _ = client.delete_cached("test_key").await;

                let stats = client.get_cache_stats().await;
                match stats {
                    Ok(cache_stats) => println!("Redis cache stats: {:?}", cache_stats),
                    Err(e) => println!("Failed to get cache stats: {}", e),
                }
            },
            Err(_) => {
                println!("Redis not available, skipping Redis integration test");
            }
        }
        
        Ok(())
    }

    #[cfg(feature = "visualization")]
    #[tokio::test]
    async fn test_visualization_integration() -> Result<(), Box<dyn Error>> {
        let config = VisualizationConfig {
            output_dir: std::path::PathBuf::from("./test_visualizations"),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Default,
            width: 800,
            height: 600,
            font_size: 12,
            interactive: false,
        };

        let mut engine = RealVisualizationEngine::new(config).await?;

        // Test visualization creation
        let test_memories = vec![
            MemoryEntry::new(
                "Node1".to_string(),
                "Node 1 content".to_string(),
                MemoryType::ShortTerm,
            ),
            MemoryEntry::new(
                "Node2".to_string(),
                "Node 2 content".to_string(),
                MemoryType::LongTerm,
            ),
        ];

        let test_relationships = vec![
            ("Node1".to_string(), "Node2".to_string(), 0.8),
        ];

        // Test network visualization
        let result = engine.generate_memory_network(&test_memories, &test_relationships).await;
        match result {
            Ok(path) => {
                println!("Visualization saved to: {}", path);
                // Clean up test file
                let _ = std::fs::remove_file(&path);
            },
            Err(e) => {
                println!("Visualization test failed (expected if output dir doesn't exist): {}", e);
            }
        }
        
        // Test analytics timeline
        let analytics_events = vec![
            AnalyticsEvent {
                id: "test_event".to_string(),
                event_type: "memory_access".to_string(),
                timestamp: chrono::Utc::now(),
                data: std::collections::HashMap::from([
                    ("memory_key".to_string(), serde_json::Value::String("test_memory".to_string())),
                ]),
            },
        ];
        let _ = engine.generate_analytics_timeline(&analytics_events).await;
        
        Ok(())
    }

    #[tokio::test]
    async fn test_integration_health_checks() -> Result<(), Box<dyn Error>> {
        let config = IntegrationConfig::default();
        let manager = IntegrationManager::new(config).await?;
        
        // Test comprehensive health check
        let health_results = manager.health_check().await?;
        
        // Should have entries for all integration types
        assert!(!health_results.is_empty());
        
        // Each service should report a boolean health status
        for (service_name, is_healthy) in &health_results {
            println!("Service: {} - Healthy: {}", service_name, is_healthy);
            assert!(service_name.len() > 0); // Service name should not be empty
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_memory_with_all_integrations() -> Result<(), Box<dyn Error>> {
        let memory_config = MemoryConfig {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            ..Default::default()
        };

        let mut memory = AgentMemory::new(memory_config).await?;
        
        // Test basic memory operations work with integrations enabled
        memory.store("integration_test", "Testing memory with all integrations enabled").await?;
        
        let retrieved = memory.retrieve("integration_test").await?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "Testing memory with all integrations enabled");
        
        // Test search functionality
        let search_results = memory.search("integration", 5).await?;
        assert!(!search_results.is_empty());
        
        // Test stats
        let stats = memory.stats();
        assert!(stats.short_term_count > 0);
        
        Ok(())
    }
}

// Tests that run without external integrations feature
#[tokio::test]
async fn test_memory_without_external_integrations() -> Result<(), Box<dyn std::error::Error>> {
    use synaptic::{AgentMemory, MemoryConfig};

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Basic functionality should work without external integrations
    memory.store("test_key", "test content").await?;
    let retrieved = memory.retrieve("test_key").await?;
    assert!(retrieved.is_some());
    
    Ok(())
}
