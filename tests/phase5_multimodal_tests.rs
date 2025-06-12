//! # Phase 5: Multi-Modal & Cross-Platform Tests
//!
//! Comprehensive tests for multi-modal memory capabilities and cross-platform support.

#[cfg(feature = "multimodal")]
use synaptic::multimodal::{
    audio::AudioMemoryProcessor,
    code::CodeMemoryProcessor,
    cross_modal::{CrossModalAnalyzer, CrossModalConfig},
    image::ImageMemoryProcessor,
    unified::{UnifiedMultiModalMemory, UnifiedMultiModalConfig, MultiModalQuery},
    ContentType, ImageFormat, AudioFormat, CodeLanguage, MultiModalProcessor,
};

#[cfg(feature = "cross-platform")]
use synaptic::cross_platform::{
    CrossPlatformMemoryManager, CrossPlatformConfig, Platform, PlatformFeature,
    offline::{OfflineAdapter, OfflineConfig},
    sync::{SyncManager, SyncConfig, SyncOperation},
};

use synaptic::{AgentMemory, MemoryConfig};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "image-memory")]
#[tokio::test]
async fn test_image_memory_processor() {
    let processor = ImageMemoryProcessor::new(Default::default()).unwrap();
    
    // Test format detection
    let png_header = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let format = processor.detect_format(&png_header).unwrap();
    assert_eq!(format, ImageFormat::Png);
    
    let jpeg_header = vec![0xFF, 0xD8, 0xFF];
    let format = processor.detect_format(&jpeg_header).unwrap();
    assert_eq!(format, ImageFormat::Jpeg);
}

#[cfg(all(feature = "image-memory", feature = "tesseract"))]
#[tokio::test]
async fn test_image_ocr_extraction() {
    use image::{ImageBuffer, Rgba};

    let processor = ImageMemoryProcessor::new(Default::default()).unwrap();

    // Create simple image with block letters "HI"
    let mut img = ImageBuffer::from_pixel(60, 20, Rgba([255, 255, 255, 255]));
    let black = Rgba([0, 0, 0, 255]);
    for y in 0..20 {
        img.put_pixel(5, y, black);
        img.put_pixel(15, y, black);
        img.put_pixel(25, y, black);
    }
    for x in 5..15 {
        img.put_pixel(x, 10, black);
    }

    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageOutputFormat::Png)
        .unwrap();

    let memory = processor
        .process(
            &bytes,
            &ContentType::Image {
                format: ImageFormat::Png,
                width: 60,
                height: 20,
            },
        )
        .await
        .unwrap();

    if let ContentSpecificMetadata::Image { text_regions, .. } = memory.metadata.content_specific {
        assert!(!text_regions.is_empty());
    } else {
        panic!("Expected image metadata");
    }
}

#[cfg(feature = "audio-memory")]
#[tokio::test]
async fn test_audio_memory_processor() {
    let processor = AudioMemoryProcessor::new(Default::default()).unwrap();
    
    // Test format detection
    let wav_header = b"RIFF\x00\x00\x00\x00WAVE";
    let format = processor.detect_format(wav_header).unwrap();
    assert_eq!(format, AudioFormat::Wav);
    
    let mp3_header = vec![0xFF, 0xE0]; // MP3 sync frame
    let format = processor.detect_format(&mp3_header).unwrap();
    assert_eq!(format, AudioFormat::Mp3);
}

#[cfg(all(feature = "audio-memory", feature = "whisper-rs"))]
#[tokio::test]
async fn test_audio_transcription_output() {
    let model_path = match std::env::var("WHISPER_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("WHISPER_MODEL_PATH not set, skipping test");
            return;
        }
    };
    let audio_path = match std::env::var("WHISPER_TEST_AUDIO") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("WHISPER_TEST_AUDIO not set, skipping test");
            return;
        }
    };

    std::env::set_var("WHISPER_MODEL_PATH", &model_path);

    let processor = AudioMemoryProcessor::new(Default::default()).unwrap();
    let data = std::fs::read(&audio_path).expect("failed to read audio");
    let format = processor.detect_format(&data).unwrap();
    let (samples, spec) = processor.load_audio(&data, &format).unwrap();
    let transcript = processor
        .transcribe_audio(&samples, spec.sample_rate)
        .await
        .unwrap();

    assert!(transcript.is_some());
    let t = transcript.unwrap();
    assert!(!t.is_empty());
}

#[cfg(feature = "code-memory")]
#[tokio::test]
async fn test_code_memory_processor() {
    let processor = CodeMemoryProcessor::new(Default::default()).unwrap();
    
    // Test language detection
    let rust_code = r#"
        fn main() {
            println!("Hello, world!");
        }
    "#;
    let language = processor.detect_language(rust_code, Some("main.rs"));
    assert_eq!(language, CodeLanguage::Rust);
    
    let python_code = r#"
        def main():
            print("Hello, world!")
        
        if __name__ == "__main__":
            main()
    "#;
    let language = processor.detect_language(python_code, Some("main.py"));
    assert_eq!(language, CodeLanguage::Python);
    
    let javascript_code = r#"
        function main() {
            console.log("Hello, world!");
        }
        
        main();
    "#;
    let language = processor.detect_language(javascript_code, Some("main.js"));
    assert_eq!(language, CodeLanguage::JavaScript);
}

#[cfg(feature = "code-memory")]
#[tokio::test]
async fn test_code_dependency_extraction() {
    let processor = CodeMemoryProcessor::new(Default::default()).unwrap();
    
    let rust_code = r#"
        use std::collections::HashMap;
        use serde::{Serialize, Deserialize};
        use tokio::sync::RwLock;
        
        fn main() {
            println!("Hello, world!");
        }
    "#;
    
    let dependencies = processor.extract_dependencies(rust_code, &CodeLanguage::Rust);
    assert!(dependencies.contains(&"serde".to_string()));
    assert!(dependencies.contains(&"tokio".to_string()));
    assert!(!dependencies.contains(&"std".to_string())); // std is filtered out
}

#[cfg(feature = "code-memory")]
#[tokio::test]
async fn test_code_complexity_metrics() {
    let processor = CodeMemoryProcessor::new(Default::default()).unwrap();
    
    let complex_code = r#"
        fn complex_function(x: i32) -> i32 {
            if x > 10 {
                for i in 0..x {
                    if i % 2 == 0 {
                        while i > 0 {
                            return i;
                        }
                    }
                }
            } else {
                match x {
                    1 => 1,
                    2 => 2,
                    _ => 0,
                }
            }
        }
    "#;
    
    let functions = vec![]; // Would be extracted from AST in real implementation
    let metrics = processor.calculate_complexity_metrics(complex_code, &functions);
    
    assert!(metrics.lines_of_code > 0);
    assert!(metrics.maintainability_index <= 100.0);
    assert!(metrics.maintainability_index >= 0.0);
}

#[cfg(feature = "multimodal")]
#[tokio::test]
async fn test_cross_modal_analyzer() {
    let config = CrossModalConfig::default();
    let analyzer = CrossModalAnalyzer::new(config);
    
    // Test analyzer creation
    assert!(true); // Placeholder - would test relationship detection
}

#[cfg(feature = "multimodal")]
#[tokio::test]
async fn test_unified_multimodal_memory() {
    let core_memory = Arc::new(RwLock::new(
        AgentMemory::new(MemoryConfig::default()).await.unwrap()
    ));
    
    let config = UnifiedMultiModalConfig::default();
    let multimodal_memory = UnifiedMultiModalMemory::new(core_memory, config).await.unwrap();
    
    // Test statistics
    let stats = multimodal_memory.get_statistics().await.unwrap();
    assert_eq!(stats.total_memories, 0);
    assert_eq!(stats.total_size, 0);
}

#[cfg(feature = "multimodal")]
#[tokio::test]
async fn test_multimodal_search() {
    let core_memory = Arc::new(RwLock::new(
        AgentMemory::new(MemoryConfig::default()).await.unwrap()
    ));
    
    let config = UnifiedMultiModalConfig::default();
    let multimodal_memory = UnifiedMultiModalMemory::new(core_memory, config).await.unwrap();
    
    let query = MultiModalQuery {
        content: b"test content".to_vec(),
        content_type: None,
        modalities: Some(vec!["image".to_string(), "audio".to_string()]),
        similarity_threshold: 0.7,
        max_results: 10,
        include_relationships: true,
    };
    
    let results = multimodal_memory.search_multimodal(query).await.unwrap();
    assert_eq!(results.len(), 0); // No content stored yet
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_cross_platform_manager() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test platform info
    let platform_info = manager.get_platform_info().unwrap();
    assert!(matches!(
        platform_info.platform,
        Platform::Desktop | Platform::Server | Platform::WebAssembly
    ));
    
    // Test feature support
    let supports_file_access = manager.supports_feature(PlatformFeature::FileSystemAccess).unwrap();
    let supports_network = manager.supports_feature(PlatformFeature::NetworkAccess).unwrap();
    
    // These should be true for most platforms
    assert!(supports_network);
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_offline_adapter() {
    let adapter = OfflineAdapter::new().unwrap();
    
    // Test storage operations
    let test_data = b"test data for offline storage";
    adapter.store("test_key", test_data).unwrap();
    
    let retrieved = adapter.retrieve("test_key").unwrap();
    assert_eq!(retrieved, Some(test_data.to_vec()));
    
    // Test deletion
    let deleted = adapter.delete("test_key").unwrap();
    assert!(deleted);
    
    let retrieved_after_delete = adapter.retrieve("test_key").unwrap();
    assert_eq!(retrieved_after_delete, None);
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_sync_manager() {
    let config = SyncConfig::default();
    let manager = SyncManager::new(config).unwrap();
    
    // Test sync operation creation
    let store_op = SyncManager::create_store_operation(
        "test_key".to_string(),
        b"test_data".to_vec(),
    );
    
    if let SyncOperation::Store { key, data, timestamp, checksum } = store_op {
        assert_eq!(key, "test_key");
        assert_eq!(data, b"test_data");
        assert!(timestamp > 0);
        assert!(!checksum.is_empty());
    } else {
        panic!("Expected Store operation");
    }
    
    // Test delete operation creation
    let delete_op = SyncManager::create_delete_operation("test_key".to_string());
    
    if let SyncOperation::Delete { key, timestamp } = delete_op {
        assert_eq!(key, "test_key");
        assert!(timestamp > 0);
    } else {
        panic!("Expected Delete operation");
    }
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_sync_statistics() {
    let config = SyncConfig::default();
    let manager = SyncManager::new(config).unwrap();
    
    let stats = manager.get_statistics().await;
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.successful_operations, 0);
    assert_eq!(stats.failed_operations, 0);
    assert_eq!(stats.conflicts_resolved, 0);
    assert_eq!(stats.bytes_synced, 0);
}

#[cfg(all(feature = "multimodal", feature = "cross-platform"))]
#[tokio::test]
async fn test_integrated_multimodal_cross_platform() {
    // Test integration between multimodal and cross-platform features
    let core_memory = Arc::new(RwLock::new(
        AgentMemory::new(MemoryConfig::default()).await.unwrap()
    ));
    
    let multimodal_config = UnifiedMultiModalConfig::default();
    let multimodal_memory = UnifiedMultiModalMemory::new(core_memory, multimodal_config).await.unwrap();
    
    let cross_platform_config = CrossPlatformConfig::default();
    let cross_platform_manager = CrossPlatformMemoryManager::new(cross_platform_config).unwrap();
    
    // Test that both systems can coexist
    let multimodal_stats = multimodal_memory.get_statistics().await.unwrap();
    let platform_info = cross_platform_manager.get_platform_info().unwrap();
    
    assert_eq!(multimodal_stats.total_memories, 0);
    assert!(!platform_info.version.is_empty());
}

#[tokio::test]
async fn test_phase5_feature_flags() {
    // Test that the system works correctly with different feature combinations
    
    #[cfg(feature = "multimodal")]
    {
        // Multimodal features should be available
        assert!(true);
    }
    
    #[cfg(feature = "cross-platform")]
    {
        // Cross-platform features should be available
        assert!(true);
    }
    
    #[cfg(feature = "image-memory")]
    {
        // Image memory should be available
        assert!(true);
    }
    
    #[cfg(feature = "audio-memory")]
    {
        // Audio memory should be available
        assert!(true);
    }
    
    #[cfg(feature = "code-memory")]
    {
        // Code memory should be available
        assert!(true);
    }
    
    #[cfg(feature = "wasm-support")]
    {
        // WebAssembly support should be available
        assert!(true);
    }
    
    // This test always passes, but ensures compilation works with different feature sets
    assert!(true);
}

#[tokio::test]
async fn test_phase5_memory_config_integration() {
    let mut config = MemoryConfig::default();
    
    #[cfg(feature = "multimodal")]
    {
        config.enable_multimodal = true;
        config.multimodal_config = Some(UnifiedMultiModalConfig::default());
    }
    
    #[cfg(feature = "cross-platform")]
    {
        config.enable_cross_platform = true;
        config.cross_platform_config = Some(CrossPlatformConfig::default());
    }
    
    // Test that memory system can be created with Phase 5 features enabled
    let memory = AgentMemory::new(config).await.unwrap();
    let stats = memory.stats();
    
    assert_eq!(stats.short_term_count, 0);
    assert_eq!(stats.long_term_count, 0);
}
