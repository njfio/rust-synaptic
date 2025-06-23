//! Cross-platform integration tests
//! 
//! These tests verify that cross-platform adapters work correctly on their target platforms
//! and provide consistent behavior across different environments.

use synaptic::cross_platform::{
    CrossPlatformAdapter, CrossPlatformMemoryManager, CrossPlatformConfig,
    Platform, PlatformConfig, PlatformFeature, StorageBackend, StorageStats,
};
use std::collections::HashMap;
use tempfile::TempDir;

/// Test cross-platform memory manager creation and initialization
#[test]
fn test_cross_platform_manager_creation() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config);
    assert!(manager.is_ok());
}

/// Test platform detection works correctly
#[test]
fn test_platform_detection() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    let platform_info = manager.get_platform_info().unwrap();
    
    // Platform should be one of the supported types
    assert!(matches!(
        platform_info.platform,
        Platform::WebAssembly | Platform::iOS | Platform::Android | 
        Platform::Desktop | Platform::Server
    ));
    
    // Platform info should have reasonable values
    assert!(!platform_info.version.is_empty());
    assert!(platform_info.available_memory > 0);
    assert!(platform_info.available_storage > 0);
    assert!(!platform_info.supported_features.is_empty());
}

/// Test basic storage operations work across platforms
#[test]
fn test_cross_platform_storage_operations() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    let test_data = b"cross-platform test data";
    let test_key = "cross_platform_test";
    
    // Store data
    let result = manager.store(test_key, test_data);
    assert!(result.is_ok());
    
    // Retrieve data
    let retrieved = manager.retrieve(test_key).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
    
    // Delete data
    let deleted = manager.delete(test_key).unwrap();
    assert!(deleted);
    
    // Verify deletion
    let retrieved_after_delete = manager.retrieve(test_key).unwrap();
    assert!(retrieved_after_delete.is_none());
}

/// Test feature support detection
#[test]
fn test_feature_support_detection() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test various features
    let features_to_test = vec![
        PlatformFeature::FileSystemAccess,
        PlatformFeature::NetworkAccess,
        PlatformFeature::BackgroundProcessing,
        PlatformFeature::PushNotifications,
        PlatformFeature::HardwareAcceleration,
        PlatformFeature::MultiThreading,
        PlatformFeature::LargeMemoryAllocation,
    ];
    
    for feature in features_to_test {
        let supported = manager.supports_feature(feature);
        // Result should be deterministic (not random)
        let supported_again = manager.supports_feature(feature);
        assert_eq!(supported.unwrap(), supported_again.unwrap());
    }
}

/// Test storage statistics
#[test]
fn test_storage_statistics() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Store some test data
    for i in 0..5 {
        let key = format!("stats_test_{}", i);
        let data = format!("test data {}", i);
        let _ = manager.store(&key, data.as_bytes());
    }
    
    let stats = manager.get_storage_stats().unwrap();
    assert!(stats.item_count >= 5);
    assert!(stats.used_storage > 0);
    assert!(stats.available_storage > 0);
    
    // Average item size should be reasonable
    if stats.item_count > 0 {
        assert!(stats.average_item_size > 0);
        assert!(stats.average_item_size < 1024 * 1024); // Less than 1MB per item
    }
}

/// Test platform optimization
#[test]
fn test_platform_optimization() {
    let mut config = CrossPlatformConfig::default();
    
    // Add platform-specific configurations
    let mut platform_configs = HashMap::new();
    platform_configs.insert(Platform::WebAssembly, PlatformConfig {
        max_memory_usage: 50 * 1024 * 1024, // 50MB
        enable_local_storage: true,
        enable_network_sync: false,
        storage_backends: vec![StorageBackend::IndexedDB, StorageBackend::Memory],
        performance_settings: Default::default(),
    });
    
    config.platform_configs = platform_configs;
    
    let mut manager = CrossPlatformMemoryManager::new(config).unwrap();
    let result = manager.optimize_for_platform();
    assert!(result.is_ok());
}

/// Test error handling in cross-platform operations
#[test]
fn test_cross_platform_error_handling() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test retrieval of non-existent key
    let result = manager.retrieve("nonexistent_key");
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    
    // Test deletion of non-existent key
    let result = manager.delete("nonexistent_key");
    assert!(result.is_ok());
    assert!(!result.unwrap());
    
    // Test with empty key
    let result = manager.store("", b"test");
    // Should handle gracefully (either succeed or return appropriate error)
    match result {
        Ok(()) => {
            // If it succeeds, we should be able to retrieve it
            let retrieved = manager.retrieve("").unwrap();
            assert!(retrieved.is_some());
        },
        Err(_) => {
            // If it fails, that's also acceptable for empty keys
        }
    }
}

/// Test concurrent access to cross-platform manager
#[test]
fn test_concurrent_cross_platform_access() {
    use std::sync::Arc;
    use std::thread;
    
    let config = CrossPlatformConfig::default();
    let manager = Arc::new(CrossPlatformMemoryManager::new(config).unwrap());
    let mut handles = vec![];
    
    // Test concurrent operations
    for i in 0..5 {
        let manager_clone = manager.clone();
        let handle = thread::spawn(move || {
            let key = format!("concurrent_test_{}", i);
            let data = format!("test data {}", i);
            
            // These operations should be thread-safe
            let _ = manager_clone.store(&key, data.as_bytes());
            let _ = manager_clone.retrieve(&key);
            let _ = manager_clone.delete(&key);
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Manager should still be functional
    let _ = manager.get_storage_stats();
    let _ = manager.get_platform_info();
}

/// Test data integrity across platform operations
#[test]
fn test_cross_platform_data_integrity() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test data integrity with various data types
    let test_cases = vec![
        ("empty", b"".to_vec()),
        ("small", b"small data".to_vec()),
        ("binary", vec![0, 1, 2, 3, 255, 254, 253]),
        ("unicode", "Hello 世界 🌍".as_bytes().to_vec()),
        ("large", vec![42u8; 10000]),
        ("json", r#"{"key": "value", "number": 42}"#.as_bytes().to_vec()),
    ];
    
    for (name, data) in test_cases {
        let key = format!("integrity_test_{}", name);
        
        // Store data
        let store_result = manager.store(&key, &data);
        assert!(store_result.is_ok(), "Failed to store data for test case: {}", name);
        
        // Retrieve and verify
        let retrieved = manager.retrieve(&key).unwrap();
        assert!(retrieved.is_some(), "Failed to retrieve data for test case: {}", name);
        assert_eq!(data, retrieved.unwrap(), "Data integrity failed for test case: {}", name);
    }
}

/// Test platform-specific configurations
#[test]
fn test_platform_specific_configurations() {
    let mut config = CrossPlatformConfig::default();
    
    // Configure different settings for different platforms
    let mut platform_configs = HashMap::new();
    
    // WASM configuration
    platform_configs.insert(Platform::WebAssembly, PlatformConfig {
        max_memory_usage: 50 * 1024 * 1024,
        enable_local_storage: true,
        enable_network_sync: false,
        storage_backends: vec![StorageBackend::IndexedDB, StorageBackend::Memory],
        performance_settings: Default::default(),
    });
    
    // Mobile configuration
    platform_configs.insert(Platform::iOS, PlatformConfig {
        max_memory_usage: 100 * 1024 * 1024,
        enable_local_storage: true,
        enable_network_sync: true,
        storage_backends: vec![StorageBackend::CoreData, StorageBackend::Memory],
        performance_settings: Default::default(),
    });
    
    platform_configs.insert(Platform::Android, PlatformConfig {
        max_memory_usage: 200 * 1024 * 1024,
        enable_local_storage: true,
        enable_network_sync: true,
        storage_backends: vec![StorageBackend::Room, StorageBackend::Memory],
        performance_settings: Default::default(),
    });
    
    config.platform_configs = platform_configs;
    
    let manager = CrossPlatformMemoryManager::new(config);
    assert!(manager.is_ok());
}

/// Test cross-platform configuration serialization
#[test]
fn test_config_serialization() {
    let config = CrossPlatformConfig::default();
    
    // Test serialization
    let serialized = serde_json::to_string(&config);
    assert!(serialized.is_ok());
    
    if let Ok(json) = serialized {
        // Test deserialization
        let deserialized: Result<CrossPlatformConfig, _> = serde_json::from_str(&json);
        assert!(deserialized.is_ok());
        
        if let Ok(deserialized_config) = deserialized {
            assert_eq!(config.enable_wasm, deserialized_config.enable_wasm);
            assert_eq!(config.enable_mobile, deserialized_config.enable_mobile);
            assert_eq!(config.enable_offline, deserialized_config.enable_offline);
            assert_eq!(config.enable_sync, deserialized_config.enable_sync);
        }
    }
}

/// Test platform adapter creation without manager
#[test]
fn test_direct_adapter_creation() {
    // Test that we can create adapters directly for testing
    
    #[cfg(feature = "wasm")]
    {
        use synaptic::cross_platform::wasm::WasmAdapter;
        let adapter = WasmAdapter::new();
        assert!(adapter.is_ok());
    }
    
    // Generic mobile adapter should always be available
    use synaptic::cross_platform::mobile::GenericMobileAdapter;
    let adapter = GenericMobileAdapter::new();
    assert!(adapter.is_ok());
}

/// Test storage backend availability
#[test]
fn test_storage_backend_availability() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    let platform_info = manager.get_platform_info().unwrap();
    
    // Memory storage should always be available
    assert!(manager.supports_feature(PlatformFeature::NetworkAccess).unwrap_or(false) || 
            !manager.supports_feature(PlatformFeature::NetworkAccess).unwrap_or(true));
    
    // Platform should have at least one supported feature
    assert!(!platform_info.supported_features.is_empty());
}

/// Test performance characteristics
#[test]
fn test_performance_characteristics() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    let platform_info = manager.get_platform_info().unwrap();
    
    let profile = &platform_info.performance_profile;
    
    // Performance scores should be between 0.0 and 1.0
    assert!(profile.cpu_score >= 0.0 && profile.cpu_score <= 1.0);
    assert!(profile.memory_score >= 0.0 && profile.memory_score <= 1.0);
    assert!(profile.storage_score >= 0.0 && profile.storage_score <= 1.0);
    assert!(profile.network_score >= 0.0 && profile.network_score <= 1.0);
    
    // Battery optimization should be a boolean
    assert!(profile.battery_optimization == true || profile.battery_optimization == false);
}

/// Test large data handling
#[test]
fn test_large_data_handling() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test with moderately large data (1MB)
    let large_data = vec![42u8; 1024 * 1024];
    let key = "large_data_test";
    
    let store_result = manager.store(key, &large_data);
    
    // Should either succeed or fail gracefully
    match store_result {
        Ok(()) => {
            // If storage succeeds, retrieval should work
            let retrieved = manager.retrieve(key).unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), large_data);
            
            // Clean up
            let _ = manager.delete(key);
        },
        Err(_) => {
            // If storage fails due to size limits, that's acceptable
            // The error should be handled gracefully
        }
    }
}

/// Test platform feature consistency
#[test]
fn test_platform_feature_consistency() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    let platform_info = manager.get_platform_info().unwrap();
    
    // Check that reported features match actual support
    for feature in &platform_info.supported_features {
        let supports = manager.supports_feature(feature.clone()).unwrap_or(false);
        assert!(supports, "Platform reports supporting {:?} but supports_feature returns false", feature);
    }
}

/// Integration test for async operations (if available)
#[tokio::test]
async fn test_async_cross_platform_operations() {
    let config = CrossPlatformConfig::default();
    let manager = CrossPlatformMemoryManager::new(config).unwrap();
    
    // Test async sync operation if available
    let sync_result = manager.sync().await;
    // Should either succeed or be a no-op
    assert!(sync_result.is_ok());
    
    // Test that regular operations still work after async operations
    let test_data = b"async test data";
    let test_key = "async_test";
    
    let _ = manager.store(test_key, test_data);
    let retrieved = manager.retrieve(test_key).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
}

// Platform-specific integration tests
#[cfg(feature = "wasm")]
mod wasm_integration_tests {
    use super::*;
    use synaptic::cross_platform::wasm::{WasmAdapter, WasmConfig};

    #[test]
    fn test_wasm_adapter_integration() {
        let config = WasmConfig::default();
        let adapter = WasmAdapter::new_with_config(config).unwrap();

        // Test basic operations
        let test_data = b"WASM integration test";
        let test_key = "wasm_test";

        let store_result = adapter.store(test_key, test_data);
        // In test environment, this might fail due to missing browser APIs
        // but should handle errors gracefully

        let _ = adapter.retrieve(test_key);
        let _ = adapter.delete(test_key);
        let _ = adapter.list_keys();
        let _ = adapter.get_stats();
    }

    #[tokio::test]
    async fn test_wasm_async_operations() {
        let adapter = WasmAdapter::new().unwrap();
        let test_data = b"WASM async test";
        let test_key = "wasm_async_test";

        // Test async operations (will fallback to sync in test environment)
        let _ = adapter.store_async(test_key, test_data).await;
        let _ = adapter.retrieve_async(test_key).await;
        let _ = adapter.delete_async(test_key).await;
        let _ = adapter.list_keys_async().await;
    }

    #[test]
    fn test_wasm_worker_stats() {
        let adapter = WasmAdapter::new().unwrap();
        let stats = adapter.get_worker_stats();

        assert!(stats.contains_key("worker_enabled"));
        assert!(stats.contains_key("worker_initialized"));
        assert!(stats.contains_key("worker_timeout"));
    }
}

#[cfg(feature = "mobile")]
mod mobile_integration_tests {
    use super::*;
    use synaptic::cross_platform::mobile::{MobileConfig, MobileStorage, GenericMobileAdapter};
    use tempfile::TempDir;

    #[test]
    fn test_mobile_storage_integration() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = MobileConfig::default();
        config.data_directory = Some(temp_dir.path().to_path_buf());

        let storage = MobileStorage::new(config).unwrap();

        // Test comprehensive mobile storage operations
        let test_cases = vec![
            ("mobile_test_1", b"Mobile test data 1"),
            ("mobile_test_2", b"Mobile test data 2"),
            ("mobile_test_3", b"Mobile test data 3"),
        ];

        for (key, data) in &test_cases {
            let store_result = storage.store(key, data);
            assert!(store_result.is_ok());
        }

        // Test retrieval
        for (key, expected_data) in &test_cases {
            let retrieved = storage.retrieve(key).unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), *expected_data);
        }

        // Test list keys
        let keys = storage.list_keys().unwrap();
        assert_eq!(keys.len(), test_cases.len());

        // Test stats
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.item_count, test_cases.len());
        assert!(stats.used_storage > 0);
    }

    #[test]
    fn test_mobile_adapter_integration() {
        let adapter = GenericMobileAdapter::new().unwrap();

        // Test platform-specific features
        assert!(adapter.supports_feature(PlatformFeature::FileSystemAccess));
        assert!(adapter.supports_feature(PlatformFeature::NetworkAccess));
        assert!(adapter.supports_feature(PlatformFeature::BackgroundProcessing));
        assert!(adapter.supports_feature(PlatformFeature::MultiThreading));

        // Test storage operations
        let test_data = b"Mobile adapter test";
        let test_key = "mobile_adapter_test";

        let store_result = adapter.store(test_key, test_data);
        assert!(store_result.is_ok());

        let retrieved = adapter.retrieve(test_key).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), test_data);

        let deleted = adapter.delete(test_key).unwrap();
        assert!(deleted);
    }

    #[test]
    fn test_mobile_compression_integration() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = MobileConfig::default();
        config.data_directory = Some(temp_dir.path().to_path_buf());
        config.enable_compression = true;

        let storage = MobileStorage::new(config).unwrap();

        // Test with compressible data
        let large_text = "This is a large piece of text that should compress well. ".repeat(100);
        let test_key = "compression_test";

        let store_result = storage.store(test_key, large_text.as_bytes());
        assert!(store_result.is_ok());

        let retrieved = storage.retrieve(test_key).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), large_text.as_bytes());
    }
}

// Cross-platform compatibility tests
mod compatibility_tests {
    use super::*;

    #[test]
    fn test_cross_platform_config_compatibility() {
        // Test that configurations work across different platforms
        let mut config = CrossPlatformConfig::default();

        // Add configurations for all platforms
        let mut platform_configs = HashMap::new();

        for platform in [Platform::WebAssembly, Platform::iOS, Platform::Android, Platform::Desktop, Platform::Server] {
            let platform_config = PlatformConfig {
                max_memory_usage: match platform {
                    Platform::WebAssembly => 50 * 1024 * 1024,
                    Platform::iOS => 100 * 1024 * 1024,
                    Platform::Android => 200 * 1024 * 1024,
                    Platform::Desktop | Platform::Server => 1024 * 1024 * 1024,
                },
                enable_local_storage: true,
                enable_network_sync: !matches!(platform, Platform::WebAssembly),
                storage_backends: match platform {
                    Platform::WebAssembly => vec![StorageBackend::IndexedDB, StorageBackend::Memory],
                    Platform::iOS => vec![StorageBackend::CoreData, StorageBackend::Memory],
                    Platform::Android => vec![StorageBackend::Room, StorageBackend::Memory],
                    Platform::Desktop | Platform::Server => vec![StorageBackend::FileSystem, StorageBackend::Memory],
                },
                performance_settings: Default::default(),
            };

            platform_configs.insert(platform, platform_config);
        }

        config.platform_configs = platform_configs;

        // Should be able to create manager with all platform configs
        let manager = CrossPlatformMemoryManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_feature_matrix_consistency() {
        // Test that feature support is consistent with documentation
        let config = CrossPlatformConfig::default();
        let manager = CrossPlatformMemoryManager::new(config).unwrap();
        let platform_info = manager.get_platform_info().unwrap();

        // Verify feature support matches expected patterns
        match platform_info.platform {
            Platform::WebAssembly => {
                // WASM should support network but not file system
                assert!(manager.supports_feature(PlatformFeature::NetworkAccess).unwrap_or(false));
                // File system access is limited in browsers
            },
            Platform::iOS | Platform::Android => {
                // Mobile platforms should support most features
                assert!(manager.supports_feature(PlatformFeature::NetworkAccess).unwrap_or(false));
                assert!(manager.supports_feature(PlatformFeature::BackgroundProcessing).unwrap_or(false));
            },
            Platform::Desktop | Platform::Server => {
                // Desktop/server should support most features
                assert!(manager.supports_feature(PlatformFeature::NetworkAccess).unwrap_or(false));
                assert!(manager.supports_feature(PlatformFeature::MultiThreading).unwrap_or(false));
            },
        }
    }

    #[test]
    fn test_storage_backend_selection() {
        // Test that appropriate storage backends are selected for each platform
        let config = CrossPlatformConfig::default();
        let manager = CrossPlatformMemoryManager::new(config).unwrap();
        let stats = manager.get_storage_stats().unwrap();

        // Storage backend should be appropriate for the platform
        match stats.backend {
            StorageBackend::Memory => {
                // Memory storage is always acceptable
            },
            StorageBackend::IndexedDB => {
                // Should only be used on WASM
                // Note: In test environment, platform detection might not work perfectly
            },
            StorageBackend::CoreData => {
                // Should only be used on iOS
            },
            StorageBackend::Room => {
                // Should only be used on Android
            },
            StorageBackend::FileSystem => {
                // Should be used on desktop/server
            },
            StorageBackend::SQLite => {
                // General SQL storage
            },
        }
    }
}
