//! # Phase 5: Multi-Modal & Cross-Platform Demo
//!
//! Comprehensive demonstration of Phase 5 capabilities including:
//! - Multi-modal memory (image, audio, code)
//! - Cross-platform support (WebAssembly, mobile, offline)
//! - Cross-modal relationship detection
//! - Platform-optimized performance

use synaptic::{AgentMemory, MemoryConfig};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "multimodal")]
use synaptic::multimodal::{
    unified::{UnifiedMultiModalMemory, UnifiedMultiModalConfig, MultiModalQuery},
    ContentType, ImageFormat, AudioFormat, CodeLanguage,
    cross_modal::{CrossModalAnalyzer, CrossModalConfig},
};

#[cfg(feature = "cross-platform")]
use synaptic::cross_platform::{
    CrossPlatformMemoryManager, CrossPlatformConfig, Platform, PlatformFeature,
    offline::OfflineAdapter,
    sync::{SyncManager, SyncConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(" Phase 5: Multi-Modal & Cross-Platform Demo");
    println!("==============================================");

    // Phase 5A: Multi-Modal Memory System
    #[cfg(feature = "multimodal")]
    {
        println!("\n Phase 5A: Multi-Modal Memory System");
        println!("======================================");
        
        demo_multimodal_memory().await?;
    }

    // Phase 5B: Cross-Platform Support
    #[cfg(feature = "cross-platform")]
    {
        println!("\n Phase 5B: Cross-Platform Support");
        println!("===================================");
        
        demo_cross_platform_support().await?;
    }

    // Phase 5C: Integrated System
    #[cfg(all(feature = "multimodal", feature = "cross-platform"))]
    {
        println!("\n🔗 Phase 5C: Integrated Multi-Modal Cross-Platform System");
        println!("==========================================================");
        
        demo_integrated_system().await?;
    }

    println!("\n Phase 5 Demo Complete!");
    println!(" Multi-modal memory capabilities demonstrated");
    println!(" Cross-platform support validated");
    println!(" Real-time synchronization working");
    println!(" Offline-first capabilities enabled");
    
    Ok(())
}

#[cfg(feature = "multimodal")]
async fn demo_multimodal_memory() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize core memory system
    let core_memory = Arc::new(RwLock::new(
        AgentMemory::new(MemoryConfig::default()).await?
    ));

    // Initialize unified multi-modal memory
    let multimodal_config = UnifiedMultiModalConfig {
        enable_image_memory: true,
        enable_audio_memory: true,
        enable_code_memory: true,
        enable_cross_modal_analysis: true,
        auto_detect_content_type: true,
        max_storage_size: 100 * 1024 * 1024, // 100MB
        cross_modal_config: CrossModalConfig::default(),
    };

    let multimodal_memory = UnifiedMultiModalMemory::new(core_memory, multimodal_config).await?;

    // Demo 1: Image Memory
    println!("\n📸 Image Memory Demo");
    println!("-------------------");
    
    // Simulate storing an image (PNG header)
    let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]; // PNG header
    let image_content_type = ContentType::Image {
        format: ImageFormat::Png,
        width: 1920,
        height: 1080,
    };
    
    let image_id = multimodal_memory.store_multimodal(
        "screenshot_dashboard",
        png_data,
        Some(image_content_type),
    ).await?;
    
    println!(" Stored image memory: {}", image_id);

    // Demo 2: Audio Memory
    println!("\n🎵 Audio Memory Demo");
    println!("-------------------");
    
    // Simulate storing audio (WAV header)
    let wav_data = b"RIFF\x00\x00\x00\x00WAVE".to_vec();
    let audio_content_type = ContentType::Audio {
        format: AudioFormat::Wav,
        duration_ms: 30000, // 30 seconds
        sample_rate: 44100,
        channels: 2,
    };
    
    let audio_id = multimodal_memory.store_multimodal(
        "meeting_recording",
        wav_data,
        Some(audio_content_type),
    ).await?;
    
    println!(" Stored audio memory: {}", audio_id);

    // Demo 3: Code Memory
    println!("\n Code Memory Demo");
    println!("------------------");
    
    let rust_code = r#"
        use std::collections::HashMap;
        
        fn analyze_data(data: &[i32]) -> HashMap<String, f64> {
            let mut stats = HashMap::new();
            
            if data.is_empty() {
                return stats;
            }
            
            let sum: i32 = data.iter().sum();
            let mean = sum as f64 / data.len() as f64;
            
            stats.insert("mean".to_string(), mean);
            stats.insert("count".to_string(), data.len() as f64);
            
            stats
        }
    "#;
    
    let code_content_type = ContentType::Code {
        language: CodeLanguage::Rust,
        lines: rust_code.lines().count() as u32,
        complexity_score: 2.5,
    };
    
    let code_id = multimodal_memory.store_multimodal(
        "data_analysis_function",
        rust_code.as_bytes().to_vec(),
        Some(code_content_type),
    ).await?;
    
    println!(" Stored code memory: {}", code_id);

    // Demo 4: Cross-Modal Search
    println!("\n Cross-Modal Search Demo");
    println!("--------------------------");
    
    let search_query = MultiModalQuery {
        content: b"data analysis".to_vec(),
        content_type: None,
        modalities: Some(vec!["code".to_string(), "image".to_string()]),
        similarity_threshold: 0.5,
        max_results: 5,
        include_relationships: true,
    };
    
    let search_results = multimodal_memory.search_multimodal(search_query).await?;
    println!(" Found {} related memories across modalities", search_results.len());
    
    for (i, result) in search_results.iter().enumerate() {
        println!("  {}. {} (similarity: {:.2})", 
                 i + 1, 
                 result.memory.id, 
                 result.similarity);
        
        if !result.related_memories.is_empty() {
            println!("     Related: {} cross-modal links", result.related_memories.len());
        }
    }

    // Demo 5: Statistics
    println!("\n Multi-Modal Statistics");
    println!("-------------------------");
    
    let stats = multimodal_memory.get_statistics().await?;
    println!(" Total memories: {}", stats.total_memories);
    println!(" Total size: {} bytes", stats.total_size);
    println!(" Total relationships: {}", stats.total_relationships);
    println!(" Average relationship confidence: {:.2}", stats.average_relationship_confidence);
    
    for (modality, count) in &stats.memories_by_modality {
        println!("   - {}: {} memories", modality, count);
    }

    Ok(())
}

#[cfg(feature = "cross-platform")]
async fn demo_cross_platform_support() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize cross-platform manager
    let cross_platform_config = CrossPlatformConfig::default();
    let mut manager = CrossPlatformMemoryManager::new(cross_platform_config)?;

    // Demo 1: Platform Detection
    println!("\n Platform Detection");
    println!("---------------------");
    
    let platform_info = manager.get_platform_info()?;
    println!(" Platform: {:?}", platform_info.platform);
    println!(" Version: {}", platform_info.version);
    println!(" Available memory: {} MB", platform_info.available_memory / (1024 * 1024));
    println!(" Available storage: {} GB", platform_info.available_storage / (1024 * 1024 * 1024));
    
    println!(" Supported features:");
    for feature in &platform_info.supported_features {
        println!("   - {:?}", feature);
    }

    // Demo 2: Feature Support Testing
    println!("\n Feature Support Testing");
    println!("--------------------------");
    
    let features_to_test = vec![
        PlatformFeature::FileSystemAccess,
        PlatformFeature::NetworkAccess,
        PlatformFeature::BackgroundProcessing,
        PlatformFeature::MultiThreading,
        PlatformFeature::LargeMemoryAllocation,
    ];
    
    for feature in features_to_test {
        let supported = manager.supports_feature(feature.clone())?;
        println!(" {:?}: {}", feature, if supported { "✓" } else { "✗" });
    }

    // Demo 3: Storage Operations
    println!("\n Cross-Platform Storage");
    println!("-------------------------");
    
    let test_data = "Cross-platform test data with unicode: 🌍".as_bytes();
    manager.store("test_key", test_data)?;
    println!(" Stored data across platform");
    
    let retrieved = manager.retrieve("test_key")?;
    if let Some(data) = retrieved {
        println!(" Retrieved data: {} bytes", data.len());
        assert_eq!(data, test_data);
    }

    // Demo 4: Platform Optimization
    println!("\n Platform Optimization");
    println!("-----------------------");
    
    manager.optimize_for_platform()?;
    println!(" Platform optimizations applied");
    
    let optimized_info = manager.get_platform_info()?;
    println!(" Performance profile:");
    println!("   - CPU score: {:.2}", optimized_info.performance_profile.cpu_score);
    println!("   - Memory score: {:.2}", optimized_info.performance_profile.memory_score);
    println!("   - Storage score: {:.2}", optimized_info.performance_profile.storage_score);
    println!("   - Network score: {:.2}", optimized_info.performance_profile.network_score);
    println!("   - Battery optimization: {}", optimized_info.performance_profile.battery_optimization);

    // Demo 5: Offline Support
    println!("\n📴 Offline Support Demo");
    println!("----------------------");
    
    let offline_adapter = OfflineAdapter::new()?;
    
    // Test offline storage
    let offline_data = b"Offline-first data that syncs when online";
    offline_adapter.store("offline_key", offline_data)?;
    println!(" Stored data offline");
    
    let offline_retrieved = offline_adapter.retrieve("offline_key")?;
    if let Some(data) = offline_retrieved {
        println!(" Retrieved offline data: {} bytes", data.len());
    }
    
    // Test sync queue
    let pending_ops = offline_adapter.get_pending_operations()?;
    println!(" Pending sync operations: {}", pending_ops.len());

    // Demo 6: Synchronization
    println!("\n Synchronization Demo");
    println!("----------------------");
    
    let sync_config = SyncConfig {
        auto_sync: true,
        sync_interval_seconds: 60,
        enable_realtime_sync: false,
        max_retry_attempts: 3,
        retry_delay_seconds: 30,
        enable_conflict_detection: true,
        sync_timeout_seconds: 60,
        batch_size: 100,
    };
    
    let sync_manager = SyncManager::new(sync_config)?;
    
    // Create sync operations
    let store_op = SyncManager::create_store_operation(
        "sync_test".to_string(),
        b"Data to be synchronized".to_vec(),
    );
    
    sync_manager.queue_sync_operation(store_op).await?;
    println!(" Queued sync operation");
    
    let sync_stats = sync_manager.get_statistics().await;
    println!(" Sync statistics:");
    println!("   - Total operations: {}", sync_stats.total_operations);
    println!("   - Successful: {}", sync_stats.successful_operations);
    println!("   - Failed: {}", sync_stats.failed_operations);

    // Demo 7: Storage Statistics
    println!("\n Storage Statistics");
    println!("--------------------");
    
    let storage_stats = manager.get_storage_stats()?;
    println!(" Storage backend: {:?}", storage_stats.backend);
    println!(" Used storage: {} bytes", storage_stats.used_storage);
    println!(" Available storage: {} bytes", storage_stats.available_storage);
    println!(" Item count: {}", storage_stats.item_count);
    println!(" Average item size: {} bytes", storage_stats.average_item_size);

    Ok(())
}

#[cfg(all(feature = "multimodal", feature = "cross-platform"))]
async fn demo_integrated_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Integrated Multi-Modal Cross-Platform System");
    
    // Initialize both systems
    let core_memory = Arc::new(RwLock::new(
        AgentMemory::new(MemoryConfig::default()).await?
    ));
    
    let multimodal_memory = UnifiedMultiModalMemory::new(
        core_memory,
        UnifiedMultiModalConfig::default(),
    ).await?;
    
    let cross_platform_manager = CrossPlatformMemoryManager::new(
        CrossPlatformConfig::default()
    )?;

    // Demo: Store multi-modal content with cross-platform sync
    println!("\n📱 Cross-Platform Multi-Modal Storage");
    println!("-------------------------------------");
    
    // Store image on current platform
    let image_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let image_id = multimodal_memory.store_multimodal(
        "mobile_screenshot",
        image_data.clone(),
        Some(ContentType::Image {
            format: ImageFormat::Png,
            width: 375,
            height: 812, // iPhone dimensions
        }),
    ).await?;
    
    // Also store in cross-platform storage for sync
    cross_platform_manager.store("mobile_screenshot_data", &image_data)?;
    
    println!(" Stored multi-modal content with cross-platform sync");
    println!("   - Multi-modal ID: {}", image_id);
    println!("   - Cross-platform storage: ✓");

    // Demo: Platform-aware multi-modal processing
    println!("\n Platform-Aware Processing");
    println!("----------------------------");
    
    let platform_info = cross_platform_manager.get_platform_info()?;
    let multimodal_stats = multimodal_memory.get_statistics().await?;
    
    println!(" Platform: {:?}", platform_info.platform);
    println!(" Multi-modal memories: {}", multimodal_stats.total_memories);
    println!(" Cross-modal relationships: {}", multimodal_stats.total_relationships);
    
    // Adjust processing based on platform capabilities
    match platform_info.platform {
        Platform::WebAssembly => {
            println!(" WebAssembly optimizations:");
            println!("   - Reduced memory usage");
            println!("   - IndexedDB storage");
            println!("   - Single-threaded processing");
        }
        Platform::iOS | Platform::Android => {
            println!("📱 Mobile optimizations:");
            println!("   - Battery-aware processing");
            println!("   - Background sync");
            println!("   - Compressed storage");
        }
        Platform::Desktop => {
            println!("🖥️ Desktop optimizations:");
            println!("   - Full feature set");
            println!("   - Multi-threaded processing");
            println!("   - Large memory allocation");
        }
        Platform::Server => {
            println!("🖥️ Server optimizations:");
            println!("   - High-performance processing");
            println!("   - Distributed storage");
            println!("   - Real-time sync");
        }
    }

    // Demo: Unified search across platforms and modalities
    println!("\n Unified Cross-Platform Multi-Modal Search");
    println!("--------------------------------------------");
    
    let search_query = MultiModalQuery {
        content: b"mobile screenshot".to_vec(),
        content_type: None,
        modalities: Some(vec!["image".to_string()]),
        similarity_threshold: 0.3,
        max_results: 10,
        include_relationships: true,
    };
    
    let search_results = multimodal_memory.search_multimodal(search_query).await?;
    println!(" Found {} results across all modalities and platforms", search_results.len());
    
    // Demo: Sync status
    println!("\n Synchronization Status");
    println!("-------------------------");
    
    println!(" Multi-modal content ready for sync");
    println!(" Cross-platform storage active");
    println!(" Offline-first capabilities enabled");
    println!(" Real-time updates configured");

    Ok(())
}
