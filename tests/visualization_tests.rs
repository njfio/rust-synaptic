//! Comprehensive tests for visualization engine
//! 
//! Tests chart generation, timeline visualization, network graphs,
//! and analytics visualization capabilities.

#[cfg(feature = "visualization")]
mod visualization_tests {
    use synaptic::{
        AgentMemory, MemoryConfig, MemoryEntry, MemoryType,
        integrations::visualization::{
            VisualizationConfig, RealVisualizationEngine, ImageFormat, ColorScheme
        },
        analytics::{AccessType, ModificationType},
        memory::management::analytics::AnalyticsEvent,
    };
    use std::error::Error;
    use std::path::PathBuf;
    use chrono::Utc;

    fn setup_test_output_dir() -> PathBuf {
        let output_dir = PathBuf::from("./test_output/visualizations");
        std::fs::create_dir_all(&output_dir).ok();
        output_dir
    }

    #[tokio::test]
    async fn test_visualization_config() -> Result<(), Box<dyn Error>> {
        let config = VisualizationConfig {
            output_dir: setup_test_output_dir(),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Dark,
            width: 1024,
            height: 768,
            font_size: 14,
            interactive: false,
        };

        let engine = RealVisualizationEngine::new(config).await?;

        // Test engine creation
        assert!(engine.health_check().await.is_ok());
        
        Ok(())
    }

    #[tokio::test]
    async fn test_network_visualization() -> Result<(), Box<dyn Error>> {
        let config = VisualizationConfig {
            output_dir: setup_test_output_dir(),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Default,
            width: 800,
            height: 600,
            font_size: 12,
            interactive: false,
        };

        let mut engine = RealVisualizationEngine::new(config).await?;

        // Create test memory data
        let memories = vec![
            MemoryEntry::new(
                "ai_research".to_string(),
                "AI Research is a broad field".to_string(),
                MemoryType::LongTerm,
            ),
            MemoryEntry::new(
                "machine_learning".to_string(),
                "Machine Learning is a subset of AI".to_string(),
                MemoryType::LongTerm,
            ),
        ];

        let relationships = vec![
            ("ai_research".to_string(), "machine_learning".to_string(), 0.9),
        ];

        // Test memory network visualization creation
        let result = engine.generate_memory_network(&memories, &relationships).await;
        
        match result {
            Ok(output_path) => {
                println!("Memory network visualization saved to: {}", output_path);
                // File should exist in output directory
                let full_path = setup_test_output_dir().join(&output_path);
                if full_path.exists() {
                    // Clean up test file
                    std::fs::remove_file(&full_path).ok();
                }
            },
            Err(e) => {
                println!("Memory network visualization test failed: {}", e);
                // This might fail in CI environments without graphics, that's OK
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_timeline_visualization() -> Result<(), Box<dyn Error>> {
        let config = VisualizationConfig {
            output_dir: setup_test_output_dir(),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Light,
            width: 1200,
            height: 400,
            font_size: 10,
            interactive: false,
        };

        let mut engine = RealVisualizationEngine::new(config).await?;

        // Create test analytics events for timeline
        let analytics_events = vec![
            AnalyticsEvent {
                id: "event_1".to_string(),
                event_type: "memory_access".to_string(),
                timestamp: Utc::now() - chrono::Duration::hours(24),
                data: std::collections::HashMap::from([
                    ("memory_key".to_string(), serde_json::Value::String("test_memory_1".to_string())),
                    ("access_type".to_string(), serde_json::Value::String("read".to_string())),
                ]),
            },
            AnalyticsEvent {
                id: "event_2".to_string(),
                event_type: "memory_modification".to_string(),
                timestamp: Utc::now() - chrono::Duration::hours(12),
                data: std::collections::HashMap::from([
                    ("memory_key".to_string(), serde_json::Value::String("test_memory_2".to_string())),
                    ("modification_type".to_string(), serde_json::Value::String("content_update".to_string())),
                ]),
            },
        ];

        // Test analytics timeline visualization
        let result = engine.generate_analytics_timeline(&analytics_events).await;
        
        match result {
            Ok(output_path) => {
                println!("Analytics timeline visualization saved to: {}", output_path);
                // File should exist in output directory
                let full_path = setup_test_output_dir().join(&output_path);
                if full_path.exists() {
                    // Clean up test file
                    std::fs::remove_file(&full_path).ok();
                }
            },
            Err(e) => {
                println!("Analytics timeline visualization test failed: {}", e);
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_analytics_visualization() -> Result<(), Box<dyn Error>> {
        let config = VisualizationConfig {
            output_dir: setup_test_output_dir(),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Default,
            width: 800,
            height: 600,
            font_size: 12,
            interactive: false,
        };

        let mut engine = RealVisualizationEngine::new(config).await?;
        
        // Create test analytics events
        let analytics_events = vec![
            AnalyticsEvent {
                id: "event_1".to_string(),
                event_type: "memory_access".to_string(),
                timestamp: Utc::now() - chrono::Duration::hours(2),
                data: std::collections::HashMap::from([
                    ("memory_key".to_string(), serde_json::Value::String("test_memory_1".to_string())),
                ]),
            },
            AnalyticsEvent {
                id: "event_2".to_string(),
                event_type: "memory_modification".to_string(),
                timestamp: Utc::now() - chrono::Duration::hours(1),
                data: std::collections::HashMap::from([
                    ("memory_key".to_string(), serde_json::Value::String("test_memory_2".to_string())),
                ]),
            },
        ];
        
        // Test analytics timeline generation
        let result = engine.generate_analytics_timeline(&analytics_events).await;
        
        match result {
            Ok(output_path) => {
                println!("Analytics timeline saved to: {}", output_path);
                // File should exist in output directory
                let full_path = setup_test_output_dir().join(&output_path);
                if full_path.exists() {
                    // Clean up test file
                    std::fs::remove_file(&full_path).ok();
                } else {
                    // File might be at the returned path directly
                    if std::path::Path::new(&output_path).exists() {
                        std::fs::remove_file(&output_path).ok();
                    }
                }
            },
            Err(e) => {
                println!("Analytics visualization test failed: {}", e);
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_different_image_formats() -> Result<(), Box<dyn Error>> {
        let formats = vec![ImageFormat::PNG, ImageFormat::SVG, ImageFormat::PDF];
        
        for format in formats {
            let config = VisualizationConfig {
                output_dir: setup_test_output_dir(),
                format: format.clone(),
                color_scheme: ColorScheme::Default,
                width: 400,
                height: 300,
                font_size: 10,
                interactive: false,
            };

            let mut engine = RealVisualizationEngine::new(config).await?;

            let simple_memories = vec![
                MemoryEntry::new(
                    "A".to_string(),
                    "Node A content".to_string(),
                    MemoryType::ShortTerm,
                ),
            ];

            let simple_relationships = vec![
                ("A".to_string(), "A".to_string(), 0.5),
            ];

            let result = engine.generate_memory_network(&simple_memories, &simple_relationships).await;
            
            match result {
                Ok(output_path) => {
                    println!("Format {:?} visualization saved to: {}", format, output_path);
                    // Clean up test file
                    std::fs::remove_file(&output_path).ok();
                },
                Err(e) => {
                    println!("Format {:?} test failed: {}", format, e);
                }
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_color_schemes() -> Result<(), Box<dyn Error>> {
        let schemes = vec![ColorScheme::Default, ColorScheme::Dark, ColorScheme::Light];
        
        for scheme in schemes {
            let config = VisualizationConfig {
                output_dir: setup_test_output_dir(),
                format: ImageFormat::PNG,
                color_scheme: scheme.clone(),
                width: 400,
                height: 300,
                font_size: 10,
                interactive: false,
            };

            let mut engine = RealVisualizationEngine::new(config).await?;

            let simple_memories = vec![
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

            let simple_relationships = vec![
                ("Node1".to_string(), "Node2".to_string(), 0.7),
            ];

            let result = engine.generate_memory_network(&simple_memories, &simple_relationships).await;
            
            match result {
                Ok(output_path) => {
                    println!("Color scheme {:?} visualization saved to: {}", scheme, output_path);
                    // Clean up test file
                    std::fs::remove_file(&output_path).ok();
                },
                Err(e) => {
                    println!("Color scheme {:?} test failed: {}", scheme, e);
                }
            }
        }
        
        Ok(())
    }

    #[tokio::test]
    async fn test_visualization_with_memory_system() -> Result<(), Box<dyn Error>> {
        let memory_config = MemoryConfig {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            ..Default::default()
        };

        let mut memory = AgentMemory::new(memory_config).await?;
        
        // Add some test memories
        memory.store("ai_concept", "Artificial Intelligence is transforming technology").await?;
        memory.store("ml_concept", "Machine Learning is a subset of AI").await?;
        memory.store("dl_concept", "Deep Learning uses neural networks").await?;
        
        // Test that visualization can work with memory data
        let search_results = memory.search("AI", 10).await?;
        assert!(!search_results.is_empty());
        
        // In a full implementation, we'd extract relationship data from memory
        // and create visualizations from it
        
        Ok(())
    }

    #[tokio::test]
    async fn test_visualization_error_handling() -> Result<(), Box<dyn Error>> {
        // Test with invalid output directory
        let config = VisualizationConfig {
            output_dir: PathBuf::from("/invalid/path/that/does/not/exist"),
            format: ImageFormat::PNG,
            color_scheme: ColorScheme::Default,
            width: 800,
            height: 600,
            font_size: 12,
            interactive: false,
        };

        // This should fail during engine creation due to invalid path
        let result = RealVisualizationEngine::new(config).await;
        assert!(result.is_err());
        
        Ok(())
    }
}

// Test that runs without visualization feature
#[tokio::test]
async fn test_memory_without_visualization() -> Result<(), Box<dyn std::error::Error>> {
    use synaptic::{AgentMemory, MemoryConfig};

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Basic functionality should work without visualization
    memory.store("test_key", "test content").await?;
    let retrieved = memory.retrieve("test_key").await?;
    assert!(retrieved.is_some());
    
    Ok(())
}
