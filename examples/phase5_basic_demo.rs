//! # Phase 5: Basic Multi-Modal & Cross-Platform Demo
//!
//! Demonstrates the basic Phase 5 capabilities without heavy external dependencies.
//! Shows multi-modal content handling, cross-platform adaptation, and relationship detection.

use synaptic::{
    AgentMemory, MemoryConfig,
    phase5_basic::{
        BasicMultiModalManager, BasicMemoryAdapter, BasicContentDetector,
        BasicContentType, BasicMetadata, BasicCrossPlatformAdapter,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Phase 5: Basic Multi-Modal & Cross-Platform Demo");
    println!("===================================================");

    // Initialize the basic multi-modal manager
    let adapter = Box::new(BasicMemoryAdapter::new());
    let platform_info = adapter.get_platform_info();
    
    println!("\n Platform Detection");
    println!("---------------------");
    println!(" Platform: {}", platform_info.platform_name);
    println!(" File System Support: {}", platform_info.supports_file_system);
    println!(" Network Support: {}", platform_info.supports_network);
    println!(" Max Memory: {} MB", platform_info.max_memory_mb);
    println!(" Max Storage: {} MB", platform_info.max_storage_mb);

    let mut manager = BasicMultiModalManager::new(adapter);

    // Demo 1: Multi-Modal Content Detection and Storage
    println!("\n Multi-Modal Content Detection");
    println!("=================================");

    // Test image content
    let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]; // PNG header
    let image_type = BasicContentDetector::detect_content_type(&png_data);
    println!(" PNG image detected: {:?}", image_type);

    let image_features = BasicContentDetector::extract_features(&image_type, &png_data);
    let image_metadata = BasicMetadata {
        title: Some("Dashboard Screenshot".to_string()),
        description: Some("Screenshot of the main dashboard interface".to_string()),
        tags: vec!["ui".to_string(), "dashboard".to_string(), "screenshot".to_string()],
        quality_score: 0.9,
        extracted_features: image_features,
    };

    let image_id = manager.store_multimodal(
        "dashboard_screenshot",
        png_data,
        image_type,
        image_metadata,
    )?;
    println!(" Stored image memory: {}", image_id);

    // Test audio content
    let wav_data = b"RIFF\x00\x00\x00\x00WAVE".to_vec(); // WAV header
    let audio_type = BasicContentDetector::detect_content_type(&wav_data);
    println!(" WAV audio detected: {:?}", audio_type);

    let audio_features = BasicContentDetector::extract_features(&audio_type, &wav_data);
    let audio_metadata = BasicMetadata {
        title: Some("Meeting Recording".to_string()),
        description: Some("Weekly team meeting discussion".to_string()),
        tags: vec!["meeting".to_string(), "audio".to_string(), "team".to_string()],
        quality_score: 0.8,
        extracted_features: audio_features,
    };

    let audio_id = manager.store_multimodal(
        "team_meeting_audio",
        wav_data,
        audio_type,
        audio_metadata,
    )?;
    println!(" Stored audio memory: {}", audio_id);

    // Test code content
    let rust_code = r#"
        fn analyze_user_behavior(data: &[UserAction]) -> BehaviorInsights {
            let mut insights = BehaviorInsights::new();
            
            for action in data {
                match action.action_type {
                    ActionType::Click => insights.click_count += 1,
                    ActionType::Scroll => insights.scroll_count += 1,
                    ActionType::Search => {
                        insights.search_count += 1;
                        insights.search_terms.push(action.data.clone());
                    }
                }
            }
            
            insights.calculate_patterns();
            insights
        }
    "#;

    let code_type = BasicContentDetector::detect_content_type(rust_code.as_bytes());
    println!(" Rust code detected: {:?}", code_type);

    let code_features = BasicContentDetector::extract_features(&code_type, rust_code.as_bytes());
    let code_metadata = BasicMetadata {
        title: Some("User Behavior Analysis".to_string()),
        description: Some("Function to analyze user interaction patterns".to_string()),
        tags: vec!["rust".to_string(), "analytics".to_string(), "behavior".to_string()],
        quality_score: 0.95,
        extracted_features: code_features,
    };

    let code_id = manager.store_multimodal(
        "behavior_analysis_function",
        rust_code.as_bytes().to_vec(),
        code_type,
        code_metadata,
    )?;
    println!(" Stored code memory: {}", code_id);

    // Test text content
    let text_content = "User behavior analysis reveals interesting patterns in dashboard usage. \
                       Most users spend 60% of their time on the main dashboard, with frequent \
                       interactions with the search functionality and data visualization components.";

    let text_type = BasicContentDetector::detect_content_type(text_content.as_bytes());
    println!(" Text content detected: {:?}", text_type);

    let text_features = BasicContentDetector::extract_features(&text_type, text_content.as_bytes());
    let text_metadata = BasicMetadata {
        title: Some("Behavior Analysis Report".to_string()),
        description: Some("Summary of user behavior analysis findings".to_string()),
        tags: vec!["analysis".to_string(), "report".to_string(), "behavior".to_string()],
        quality_score: 0.85,
        extracted_features: text_features,
    };

    let text_id = manager.store_multimodal(
        "behavior_analysis_report",
        text_content.as_bytes().to_vec(),
        text_type,
        text_metadata,
    )?;
    println!(" Stored text memory: {}", text_id);

    // Demo 2: Cross-Modal Relationship Detection
    println!("\n🔗 Cross-Modal Relationship Detection");
    println!("=====================================");

    // Detect relationships for each memory
    manager.detect_relationships(&image_id)?;
    manager.detect_relationships(&audio_id)?;
    manager.detect_relationships(&code_id)?;
    manager.detect_relationships(&text_id)?;

    println!(" Analyzed cross-modal relationships");

    // Demo 3: Multi-Modal Search
    println!("\n Multi-Modal Similarity Search");
    println!("================================");

    // Search for content related to "behavior analysis"
    let search_features = vec![1.0, 2.0, 3.0, 4.0]; // Simulated query features
    let search_results = manager.search_multimodal(&search_features, 0.3);

    println!(" Found {} similar memories:", search_results.len());
    for (i, (memory_id, similarity)) in search_results.iter().enumerate() {
        println!("  {}. {} (similarity: {:.3})", i + 1, memory_id, similarity);
    }

    // Demo 4: Content Retrieval and Analysis
    println!("\n Content Retrieval and Analysis");
    println!("=================================");

    // Retrieve and analyze stored memories
    if let Some(retrieved_image) = manager.retrieve_multimodal("dashboard_screenshot")? {
        println!(" Retrieved image memory:");
        println!("   - Title: {:?}", retrieved_image.metadata.title);
        println!("   - Content Type: {:?}", retrieved_image.content_type);
        println!("   - Features: {} dimensions", retrieved_image.metadata.extracted_features.len());
        println!("   - Relationships: {}", retrieved_image.relationships.len());
        
        for relationship in &retrieved_image.relationships {
            println!("     → {} ({}): {:.2}", 
                     relationship.target_id, 
                     relationship.relationship_type, 
                     relationship.confidence);
        }
    }

    if let Some(retrieved_code) = manager.retrieve_multimodal("behavior_analysis_function")? {
        println!(" Retrieved code memory:");
        println!("   - Title: {:?}", retrieved_code.metadata.title);
        println!("   - Content Type: {:?}", retrieved_code.content_type);
        println!("   - Quality Score: {:.2}", retrieved_code.metadata.quality_score);
        println!("   - Tags: {:?}", retrieved_code.metadata.tags);
    }

    // Demo 5: Statistics and Performance
    println!("\n System Statistics");
    println!("===================");

    let stats = manager.get_statistics();
    println!(" Multi-Modal Memory Statistics:");
    println!("   - Total memories: {}", stats.total_memories);
    println!("   - Total size: {} bytes", stats.total_size);
    println!("   - Total relationships: {}", stats.total_relationships);
    println!("   - Memories by type:");
    
    for (content_type, count) in &stats.memories_by_type {
        println!("     • {}: {} memories", content_type, count);
    }

    let platform_info = manager.get_platform_info();
    println!(" Platform Capabilities:");
    println!("   - Platform: {}", platform_info.platform_name);
    println!("   - File System: {}", if platform_info.supports_file_system { "✓" } else { "✗" });
    println!("   - Network: {}", if platform_info.supports_network { "✓" } else { "✗" });
    println!("   - Memory Limit: {} MB", platform_info.max_memory_mb);
    println!("   - Storage Limit: {} MB", platform_info.max_storage_mb);

    // Demo 6: Feature Comparison
    println!("\n🧮 Feature Vector Analysis");
    println!("==========================");

    // Compare feature vectors between different content types
    let image_features = vec![1920.0, 1080.0, 8.0]; // width, height, size
    let code_features = vec![15.0, 500.0, 3.0]; // lines, size, complexity
    let text_features = vec![200.0, 30.0, 25.0, 5.0]; // length, words, alpha, numeric

    println!(" Feature Vector Dimensions:");
    println!("   - Image features: {} dimensions", image_features.len());
    println!("   - Code features: {} dimensions", code_features.len());
    println!("   - Text features: {} dimensions", text_features.len());

    // Demo 7: Content Type Distribution
    println!("\n Content Type Analysis");
    println!("========================");

    let total_memories = stats.total_memories as f32;
    for (content_type, count) in &stats.memories_by_type {
        let percentage = (*count as f32 / total_memories) * 100.0;
        println!(" {}: {:.1}% ({} memories)", content_type, percentage, count);
    }

    println!("\n Phase 5 Basic Demo Complete!");
    println!("===============================");
    println!(" Multi-modal content detection and storage");
    println!(" Cross-platform adapter functionality");
    println!(" Cross-modal relationship detection");
    println!(" Similarity-based search across modalities");
    println!(" Feature extraction and analysis");
    println!(" Platform capability detection");
    println!(" Content type classification");
    println!(" Memory statistics and analytics");

    println!("\n Key Phase 5 Achievements:");
    println!("• Unified multi-modal memory system");
    println!("• Cross-platform compatibility layer");
    println!("• Intelligent content type detection");
    println!("• Relationship-aware memory storage");
    println!("• Feature-based similarity search");
    println!("• Platform-optimized performance");

    Ok(())
}
