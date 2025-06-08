// Real External Integrations Example
// Demonstrates actual database, ML models, LLM, and visualization integrations

use synaptic::{AgentMemory, MemoryConfig};
use synaptic::integrations::{IntegrationConfig, IntegrationManager};
use std::error::Error;
use std::collections::HashMap;

#[cfg(feature = "sql-storage")]
use synaptic::integrations::database::{DatabaseConfig, DatabaseClient};

#[cfg(feature = "ml-models")]
use synaptic::integrations::ml_models::{MLConfig, MLModelManager};

#[cfg(feature = "llm-integration")]
use synaptic::integrations::llm::{LLMConfig, LLMClient, LLMProvider};

#[cfg(feature = "visualization")]
use synaptic::integrations::visualization::{VisualizationConfig, RealVisualizationEngine, ImageFormat, ColorScheme};

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

    println!("🚀 Synaptic Real External Integrations Demo");
    println!("===========================================");

    // Check which features are enabled
    check_enabled_features();

    // Demonstrate each integration
    database_integration_demo().await?;
    ml_models_integration_demo().await?;
    llm_integration_demo().await?;
    visualization_integration_demo().await?;
    redis_cache_integration_demo().await?;
    integrated_system_demo().await?;

    Ok(())
}

fn check_enabled_features() {
    println!("\n🔧 Enabled Features:");
    
    #[cfg(feature = "sql-storage")]
    println!("   ✅ SQL Database Storage");
    #[cfg(not(feature = "sql-storage"))]
    println!("   ❌ SQL Database Storage (disabled)");

    #[cfg(feature = "ml-models")]
    println!("   ✅ ML Models (Candle)");
    #[cfg(not(feature = "ml-models"))]
    println!("   ❌ ML Models (disabled)");

    #[cfg(feature = "llm-integration")]
    println!("   ✅ LLM Integration");
    #[cfg(not(feature = "llm-integration"))]
    println!("   ❌ LLM Integration (disabled)");

    #[cfg(feature = "visualization")]
    println!("   ✅ Real Visualization (Plotters)");
    #[cfg(not(feature = "visualization"))]
    println!("   ❌ Real Visualization (disabled)");

    #[cfg(feature = "external-integrations")]
    println!("   ✅ Redis Cache");
    #[cfg(not(feature = "external-integrations"))]
    println!("   ❌ Redis Cache (disabled)");
}

async fn database_integration_demo() -> Result<(), Box<dyn Error>> {
    println!("\n🗄️  Database Integration Demo");
    println!("----------------------------");

    #[cfg(feature = "sql-storage")]
    {
        // Check if PostgreSQL is available
        let db_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://synaptic_user:synaptic_pass@localhost:11110/synaptic_db".to_string());

        println!("📡 Connecting to PostgreSQL: {}", db_url);

        let config = DatabaseConfig {
            database_url: db_url,
            max_connections: 5,
            connect_timeout_secs: 30,
            ssl_mode: "prefer".to_string(),
            schema: "public".to_string(),
        };

        match DatabaseClient::new(config).await {
            Ok(mut client) => {
                println!("✅ Connected to PostgreSQL successfully");
                
                // Test health check
                match client.health_check().await {
                    Ok(_) => println!("✅ Database health check passed"),
                    Err(e) => println!("❌ Database health check failed: {}", e),
                }

                // Test storing a memory entry
                let memory_entry = synaptic::memory::types::MemoryEntry::new(
                    "test_db_key".to_string(),
                    "Test database integration content".to_string(),
                    synaptic::memory::types::MemoryType::LongTerm
                );

                match client.store_memory(&memory_entry).await {
                    Ok(_) => {
                        println!("✅ Stored memory entry in database");
                        
                        // Test retrieving the memory entry
                        match client.get_memory("test_db_key").await {
                            Ok(Some(retrieved)) => {
                                println!("✅ Retrieved memory entry: {}", retrieved.key);
                            },
                            Ok(None) => println!("❌ Memory entry not found"),
                            Err(e) => println!("❌ Failed to retrieve memory: {}", e),
                        }
                    },
                    Err(e) => println!("❌ Failed to store memory: {}", e),
                }

                let metrics = client.get_metrics();
                println!("📊 Database metrics: {} queries executed", metrics.queries_executed);
            },
            Err(e) => {
                println!("❌ Failed to connect to PostgreSQL: {}", e);
                println!("💡 Make sure PostgreSQL is running and DATABASE_URL is set");
            }
        }
    }

    #[cfg(not(feature = "sql-storage"))]
    {
        println!("❌ SQL storage feature not enabled");
        println!("💡 Run with: cargo run --example real_integrations --features sql-storage");
    }

    Ok(())
}

async fn ml_models_integration_demo() -> Result<(), Box<dyn Error>> {
    println!("\n🤖 ML Models Integration Demo");
    println!("-----------------------------");

    #[cfg(feature = "ml-models")]
    {
        let config = MLConfig {
            model_dir: std::path::PathBuf::from("./models"),
            device: "cpu".to_string(),
            max_sequence_length: 512,
            embedding_dim: 768,
            cache_size: 100,
            quantization: false,
        };

        println!("🔄 Initializing ML models...");
        
        match MLModelManager::new(config).await {
            Ok(mut manager) => {
                println!("✅ ML model manager initialized");

                // Test health check
                match manager.health_check().await {
                    Ok(_) => println!("✅ ML models health check passed"),
                    Err(e) => println!("❌ ML models health check failed: {}", e),
                }

                // Test embedding generation
                let memory_entry = synaptic::memory::types::MemoryEntry::new(
                    "ml_test_key".to_string(),
                    "This is a test for machine learning embedding generation".to_string(),
                    synaptic::memory::types::MemoryType::ShortTerm
                );

                match manager.generate_memory_embedding(&memory_entry).await {
                    Ok(embedding) => {
                        println!("✅ Generated embedding with {} dimensions", embedding.len());
                        
                        // Test similarity calculation
                        let embedding2 = manager.generate_memory_embedding(&memory_entry).await?;
                        let similarity = manager.calculate_similarity(&embedding, &embedding2);
                        println!("✅ Similarity calculation: {:.3}", similarity);
                    },
                    Err(e) => println!("❌ Failed to generate embedding: {}", e),
                }

                // Test access pattern prediction
                let memory_keys = vec!["key1".to_string(), "key2".to_string()];
                let historical_data = vec![
                    ("key1".to_string(), chrono::Utc::now() - chrono::Duration::hours(2)),
                    ("key1".to_string(), chrono::Utc::now() - chrono::Duration::hours(1)),
                    ("key2".to_string(), chrono::Utc::now() - chrono::Duration::minutes(30)),
                ];

                match manager.predict_access_pattern(&memory_keys, &historical_data).await {
                    Ok(predictions) => {
                        println!("✅ Generated {} access predictions", predictions.len());
                        for prediction in predictions.iter().take(2) {
                            println!("   • {}: {:.1}% confidence", 
                                prediction.memory_key, 
                                prediction.confidence * 100.0
                            );
                        }
                    },
                    Err(e) => println!("❌ Failed to generate predictions: {}", e),
                }

                let metrics = manager.get_metrics();
                println!("📊 ML metrics: {} embeddings, {} predictions", 
                    metrics.embeddings_generated, 
                    metrics.predictions_made
                );
            },
            Err(e) => {
                println!("❌ Failed to initialize ML models: {}", e);
                println!("💡 Make sure BERT model is available in ./models/bert-base-uncased/");
            }
        }
    }

    #[cfg(not(feature = "ml-models"))]
    {
        println!("❌ ML models feature not enabled");
        println!("💡 Run with: cargo run --example real_integrations --features ml-models");
    }

    Ok(())
}

async fn llm_integration_demo() -> Result<(), Box<dyn Error>> {
    println!("\n🧠 LLM Integration Demo");
    println!("----------------------");

    #[cfg(feature = "llm-integration")]
    {
        // Check for API key and determine provider
        let (api_key, provider, model) = if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            (key, LLMProvider::Anthropic, "claude-3-5-haiku-20241022".to_string())
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            (key, LLMProvider::OpenAI, "gpt-3.5-turbo".to_string())
        } else {
            println!("❌ No API key found");
            println!("💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable");
            return Ok(());
        };

        let config = LLMConfig {
            provider,
            api_key: api_key.clone(),
            base_url: None,
            model,
            max_tokens: 500,
            temperature: 0.7,
            timeout_secs: 30,
            rate_limit: 10,
        };

        println!("🔄 Initializing LLM client...");

        match LLMClient::new(config).await {
            Ok(mut client) => {
                println!("✅ LLM client initialized");

                // Test health check
                match client.health_check().await {
                    Ok(_) => println!("✅ LLM health check passed"),
                    Err(e) => println!("❌ LLM health check failed: {}", e),
                }

                // Test insight generation
                let memories = vec![
                    synaptic::memory::types::MemoryEntry::new(
                        "project_status".to_string(),
                        "Project is 80% complete with some performance issues".to_string(),
                        synaptic::memory::types::MemoryType::ShortTerm
                    ),
                    synaptic::memory::types::MemoryEntry::new(
                        "user_feedback".to_string(),
                        "Users report slow response times during peak hours".to_string(),
                        synaptic::memory::types::MemoryType::ShortTerm
                    ),
                ];

                match client.generate_insights(&memories, "Software development project").await {
                    Ok(insights) => {
                        println!("✅ Generated {} insights from LLM", insights.len());
                        for insight in insights.iter().take(2) {
                            println!("   • {}: {}", insight.title, insight.description.chars().take(50).collect::<String>());
                        }
                    },
                    Err(e) => println!("❌ Failed to generate insights: {}", e),
                }

                // Test memory summarization
                let memory_to_summarize = &memories[0];
                match client.summarize_memory(memory_to_summarize).await {
                    Ok(summary) => {
                        println!("✅ Generated summary: {}", summary.chars().take(100).collect::<String>());
                    },
                    Err(e) => println!("❌ Failed to generate summary: {}", e),
                }

                let metrics = client.get_metrics();
                println!("📊 LLM metrics: {} requests, {} tokens consumed, ${:.4} cost", 
                    metrics.requests_made, 
                    metrics.tokens_consumed,
                    metrics.total_cost_usd
                );
            },
            Err(e) => {
                println!("❌ Failed to initialize LLM client: {}", e);
            }
        }
    }

    #[cfg(not(feature = "llm-integration"))]
    {
        println!("❌ LLM integration feature not enabled");
        println!("💡 Run with: cargo run --example real_integrations --features llm-integration");
    }

    Ok(())
}

async fn visualization_integration_demo() -> Result<(), Box<dyn Error>> {
    println!("\n📊 Visualization Integration Demo");
    println!("--------------------------------");

    #[cfg(feature = "visualization")]
    {
        let config = VisualizationConfig {
            output_dir: std::path::PathBuf::from("./visualizations"),
            width: 800,
            height: 600,
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Default,
            font_size: 12,
            interactive: false,
        };

        println!("🔄 Initializing visualization engine...");

        match RealVisualizationEngine::new(config).await {
            Ok(mut engine) => {
                println!("✅ Visualization engine initialized");

                // Test health check
                match engine.health_check().await {
                    Ok(_) => println!("✅ Visualization health check passed"),
                    Err(e) => println!("❌ Visualization health check failed: {}", e),
                }

                // Test memory network generation
                let memories = vec![
                    synaptic::memory::types::MemoryEntry::new(
                        "node1".to_string(),
                        "First memory node".to_string(),
                        synaptic::memory::types::MemoryType::ShortTerm
                    ),
                    synaptic::memory::types::MemoryEntry::new(
                        "node2".to_string(),
                        "Second memory node".to_string(),
                        synaptic::memory::types::MemoryType::LongTerm
                    ),
                ];

                let relationships = vec![
                    ("node1".to_string(), "node2".to_string(), 0.8),
                ];

                match engine.generate_memory_network(&memories, &relationships).await {
                    Ok(filename) => {
                        println!("✅ Generated memory network: {}", filename);
                    },
                    Err(e) => println!("❌ Failed to generate memory network: {}", e),
                }

                // Test analytics timeline
                let events = vec![
                    synaptic::memory::management::analytics::AnalyticsEvent {
                        id: "test_event".to_string(),
                        event_type: "memory_access".to_string(),
                        timestamp: chrono::Utc::now(),
                        data: std::collections::HashMap::new(),
                    },
                ];

                match engine.generate_analytics_timeline(&events).await {
                    Ok(filename) => {
                        println!("✅ Generated analytics timeline: {}", filename);
                    },
                    Err(e) => println!("❌ Failed to generate timeline: {}", e),
                }

                let metrics = engine.get_metrics();
                println!("📊 Visualization metrics: {} charts generated, {} images exported", 
                    metrics.charts_generated, 
                    metrics.images_exported
                );
            },
            Err(e) => {
                println!("❌ Failed to initialize visualization engine: {}", e);
            }
        }
    }

    #[cfg(not(feature = "visualization"))]
    {
        println!("❌ Visualization feature not enabled");
        println!("💡 Run with: cargo run --example real_integrations --features visualization");
    }

    Ok(())
}

async fn redis_cache_integration_demo() -> Result<(), Box<dyn Error>> {
    println!("\n🔄 Redis Cache Integration Demo");
    println!("------------------------------");

    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://localhost:11111".to_string());

    println!("📡 Connecting to Redis: {}", redis_url);

    let config = RedisConfig {
        url: redis_url,
        pool_size: 5,
        connect_timeout_secs: 5,
        default_ttl_secs: 300,
        key_prefix: "synaptic_demo:".to_string(),
        compression: true,
    };

    match RedisClient::new(config).await {
        Ok(mut client) => {
            println!("✅ Connected to Redis successfully");

            // Test health check
            match client.health_check().await {
                Ok(_) => println!("✅ Redis health check passed"),
                Err(e) => println!("❌ Redis health check failed: {}", e),
            }

            // Test caching
            let memory_entry = synaptic::memory::types::MemoryEntry::new(
                "cache_test_key".to_string(),
                "Test cache integration content".to_string(),
                synaptic::memory::types::MemoryType::ShortTerm
            );

            #[cfg(feature = "distributed")]
            {
                match client.cache_memory("test_key", &memory_entry, Some(60)).await {
                    Ok(_) => {
                        println!("✅ Cached memory entry");
                        
                        // Test retrieval
                        match client.get_cached_memory("test_key").await {
                            Ok(Some(cached)) => {
                                println!("✅ Retrieved cached memory: {}", cached.key);
                            },
                            Ok(None) => println!("❌ Cached memory not found"),
                            Err(e) => println!("❌ Failed to retrieve cached memory: {}", e),
                        }
                    },
                    Err(e) => println!("❌ Failed to cache memory: {}", e),
                }

                // Test cache statistics
                match client.get_cache_stats().await {
                    Ok(stats) => {
                        println!("📊 Cache stats: {:.1}% hit rate, {} total keys", 
                            stats.hit_rate * 100.0, 
                            stats.total_keys
                        );
                    },
                    Err(e) => println!("❌ Failed to get cache stats: {}", e),
                }
            }

            let metrics = client.get_metrics();
            println!("📊 Redis metrics: {} hits, {} misses, {} operations", 
                metrics.cache_hits, 
                metrics.cache_misses,
                metrics.total_operations
            );
        },
        Err(e) => {
            println!("❌ Failed to connect to Redis: {}", e);
            println!("💡 Make sure Redis is running and REDIS_URL is set");
        }
    }

    Ok(())
}

async fn integrated_system_demo() -> Result<(), Box<dyn Error>> {
    println!("\n🔗 Integrated System Demo");
    println!("-------------------------");

    // Create integration config with correct ports
    let mut integration_config = IntegrationConfig::default();

    // Override database config with correct port
    #[cfg(feature = "sql-storage")]
    {
        integration_config.database = Some(DatabaseConfig {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://synaptic_user:synaptic_pass@localhost:11110/synaptic_db".to_string()),
            max_connections: 5,
            connect_timeout_secs: 30,
            ssl_mode: "prefer".to_string(),
            schema: "public".to_string(),
        });
    }

    // Override Redis config with correct port
    integration_config.redis = Some(RedisConfig {
        url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:11111".to_string()),
        pool_size: 5,
        connect_timeout_secs: 5,
        default_ttl_secs: 300,
        key_prefix: "synaptic_demo:".to_string(),
        compression: false,
    });

    println!("🔄 Initializing integrated system...");

    match IntegrationManager::new(integration_config.clone()).await {
        Ok(manager) => {
            println!("✅ Integration manager initialized");

            // Test health check for all integrations
            match manager.health_check().await {
                Ok(health_status) => {
                    println!("🏥 Health check results:");
                    for (service, healthy) in health_status {
                        let status = if healthy { "✅" } else { "❌" };
                        println!("   {} {}", status, service);
                    }
                },
                Err(e) => println!("❌ Health check failed: {}", e),
            }

            // Create memory system with integrations
            let mut memory_config = MemoryConfig::default();
            memory_config.enable_integrations = true;
            memory_config.integrations_config = Some(integration_config.clone());

            match AgentMemory::new(memory_config).await {
                Ok(mut memory) => {
                    println!("✅ Memory system with integrations initialized");

                    // Test storing and retrieving with all integrations
                    memory.store("integrated_test", "This is a test of the integrated system with real external services").await?;
                    
                    if let Some(retrieved) = memory.retrieve("integrated_test").await? {
                        println!("✅ Stored and retrieved memory with integrations: {}", retrieved.key);
                    }

                    println!("📊 Memory stats: {:?}", memory.stats());
                },
                Err(e) => println!("❌ Failed to create integrated memory system: {}", e),
            }
        },
        Err(e) => {
            println!("❌ Failed to initialize integration manager: {}", e);
            println!("💡 Some external services may not be available");
        }
    }

    println!("\n🎉 Real integrations demo completed!");
    println!("💡 To enable all features, run with:");
    println!("   cargo run --example real_integrations --features external-integrations");

    Ok(())
}
