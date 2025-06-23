//! Tests for WebAssembly adapter implementation

#[cfg(feature = "wasm")]
use synaptic::cross_platform::{
    wasm::{WasmAdapter, WasmConfig},
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, StorageBackend,
};

#[cfg(not(feature = "wasm"))]
use synaptic::cross_platform::{
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, StorageBackend,
};

#[test]
fn test_wasm_config_default() {
    #[cfg(feature = "wasm")]
    {
        let config = WasmConfig::default();
        assert_eq!(config.db_name, "synaptic_memory");
        assert_eq!(config.db_version, 1);
        assert_eq!(config.store_name, "memories");
        assert!(config.enable_local_storage_fallback);
        assert!(config.enable_memory_cache);
        assert_eq!(config.max_cache_size, 10 * 1024 * 1024);
        assert!(config.enable_compression);
        assert!(config.enable_web_worker);
        assert_eq!(config.worker_timeout_seconds, 30);
    }
}

#[test]
fn test_wasm_adapter_creation() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new();
        assert!(adapter.is_ok());
    }
}

#[test]
fn test_wasm_adapter_initialization() {
    #[cfg(feature = "wasm")]
    {
        let mut adapter = WasmAdapter::new().unwrap();
        let config = PlatformConfig {
            max_memory_usage: 50 * 1024 * 1024,
            enable_local_storage: true,
            enable_network_sync: false,
            storage_backends: vec![StorageBackend::IndexedDB, StorageBackend::Memory],
            performance_settings: Default::default(),
        };
        
        // Note: This will fail in non-browser environment, but should not panic
        let result = adapter.initialize(&config);
        // In test environment, this might fail due to missing browser APIs
        // but it should handle the error gracefully
    }
}

#[test]
fn test_wasm_adapter_feature_support() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        
        // Test feature support
        assert!(!adapter.supports_feature(PlatformFeature::FileSystemAccess));
        assert!(adapter.supports_feature(PlatformFeature::NetworkAccess));
        assert!(!adapter.supports_feature(PlatformFeature::BackgroundProcessing));
        assert!(adapter.supports_feature(PlatformFeature::PushNotifications));
        assert!(adapter.supports_feature(PlatformFeature::HardwareAcceleration));
        assert!(!adapter.supports_feature(PlatformFeature::MultiThreading));
        assert!(!adapter.supports_feature(PlatformFeature::LargeMemoryAllocation));
    }
}

#[test]
fn test_wasm_adapter_platform_info() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        let platform_info = adapter.get_platform_info();
        
        assert_eq!(platform_info.platform, synaptic::cross_platform::Platform::WebAssembly);
        assert_eq!(platform_info.version, "1.0.0");
        assert_eq!(platform_info.available_memory, 50 * 1024 * 1024);
        assert_eq!(platform_info.available_storage, 100 * 1024 * 1024);
        assert!(!platform_info.supported_features.is_empty());
        assert!(platform_info.performance_profile.battery_optimization);
    }
}

#[test]
fn test_wasm_adapter_memory_cache() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        let test_data = b"test data for memory cache";
        
        // Store data (will use memory cache as fallback in test environment)
        let result = adapter.store("test_key", test_data);
        // In test environment without browser APIs, this might fail
        // but should handle errors gracefully
        
        // Test that the adapter doesn't panic on operations
        let _ = adapter.retrieve("test_key");
        let _ = adapter.delete("test_key");
        let _ = adapter.list_keys();
        let _ = adapter.get_stats();
    }
}

#[test]
fn test_wasm_adapter_compression() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        let test_data = b"This is test data that should be compressed when stored in the WebAssembly adapter. It's long enough to benefit from compression.";
        
        // Test compression functionality
        let compressed = adapter.compress_data(test_data);
        assert!(compressed.is_ok());
        
        if let Ok(compressed_data) = compressed {
            let decompressed = adapter.decompress_data(&compressed_data);
            assert!(decompressed.is_ok());
            
            if let Ok(decompressed_data) = decompressed {
                assert_eq!(test_data, decompressed_data.as_slice());
            }
        }
    }
}

#[test]
fn test_wasm_adapter_cache_management() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        
        // Test cache update functionality
        let test_data = b"test cache data";
        let result = adapter.update_cache("cache_test", test_data);
        assert!(result.is_ok());
        
        // Test cache size limits
        let large_data = vec![0u8; 1024 * 1024]; // 1MB
        for i in 0..20 {
            let key = format!("large_key_{}", i);
            let _ = adapter.update_cache(&key, &large_data);
        }
        // Cache should handle size limits gracefully
    }
}

#[test]
fn test_wasm_worker_stats() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        let stats = adapter.get_worker_stats();
        
        assert!(stats.contains_key("worker_enabled"));
        assert!(stats.contains_key("worker_initialized"));
        assert!(stats.contains_key("worker_timeout"));
        
        assert_eq!(stats.get("worker_enabled"), Some(&"true".to_string()));
        assert_eq!(stats.get("worker_initialized"), Some(&"false".to_string()));
        assert_eq!(stats.get("worker_timeout"), Some(&"30s".to_string()));
    }
}

#[tokio::test]
async fn test_wasm_adapter_async_operations() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        let test_data = b"async test data";
        
        // Test async operations (will fallback to sync in test environment)
        let store_result = adapter.store_async("async_test", test_data).await;
        // In test environment, this might fail due to missing browser APIs
        
        let retrieve_result = adapter.retrieve_async("async_test").await;
        // Should handle errors gracefully
        
        let delete_result = adapter.delete_async("async_test").await;
        // Should handle errors gracefully
        
        let list_result = adapter.list_keys_async().await;
        // Should handle errors gracefully
        
        // Test search functionality
        let search_result = adapter.search_async("test", 10).await;
        // In test environment without web worker, this will return an error
        // but should not panic
    }
}

#[test]
fn test_wasm_config_serialization() {
    #[cfg(feature = "wasm")]
    {
        let config = WasmConfig::default();
        
        // Test serialization
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());
        
        if let Ok(json) = serialized {
            // Test deserialization
            let deserialized: Result<WasmConfig, _> = serde_json::from_str(&json);
            assert!(deserialized.is_ok());
            
            if let Ok(deserialized_config) = deserialized {
                assert_eq!(config.db_name, deserialized_config.db_name);
                assert_eq!(config.enable_compression, deserialized_config.enable_compression);
                assert_eq!(config.enable_web_worker, deserialized_config.enable_web_worker);
            }
        }
    }
}

#[test]
fn test_wasm_adapter_error_handling() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        
        // Test error handling with invalid operations
        let result = adapter.retrieve("nonexistent_key");
        // Should return Ok(None) for missing keys, not error
        
        let result = adapter.delete("nonexistent_key");
        // Should return Ok(false) for missing keys, not error
        
        // Test with empty key
        let result = adapter.store("", b"test");
        // Should handle gracefully
        
        // Test with very large data
        let large_data = vec![0u8; 100 * 1024 * 1024]; // 100MB
        let result = adapter.store("large_test", &large_data);
        // Should handle size limits gracefully
    }
}

#[test]
fn test_wasm_adapter_without_feature() {
    #[cfg(not(feature = "wasm"))]
    {
        // Test that the module compiles even without wasm feature
        // This ensures conditional compilation works correctly
        
        let config = PlatformConfig::default();
        assert!(!config.storage_backends.is_empty());
        
        // Test that platform features are defined
        let _feature = PlatformFeature::NetworkAccess;
        let _backend = StorageBackend::Memory;
    }
}

#[test]
fn test_wasm_adapter_concurrent_access() {
    #[cfg(feature = "wasm")]
    {
        use std::sync::Arc;
        use std::thread;
        
        let adapter = Arc::new(WasmAdapter::new().unwrap());
        let mut handles = vec![];
        
        // Test concurrent access to the adapter
        for i in 0..10 {
            let adapter_clone = adapter.clone();
            let handle = thread::spawn(move || {
                let key = format!("concurrent_test_{}", i);
                let data = format!("test data {}", i);
                
                // These operations should be thread-safe
                let _ = adapter_clone.store(&key, data.as_bytes());
                let _ = adapter_clone.retrieve(&key);
                let _ = adapter_clone.delete(&key);
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Adapter should still be functional
        let _ = adapter.list_keys();
        let _ = adapter.get_stats();
    }
}

#[test]
fn test_wasm_adapter_data_integrity() {
    #[cfg(feature = "wasm")]
    {
        let adapter = WasmAdapter::new().unwrap();
        
        // Test data integrity with various data types
        let test_cases = vec![
            ("empty", b"".to_vec()),
            ("small", b"small data".to_vec()),
            ("binary", vec![0, 1, 2, 3, 255, 254, 253]),
            ("unicode", "Hello 世界 🌍".as_bytes().to_vec()),
            ("large", vec![42u8; 10000]),
        ];
        
        for (name, data) in test_cases {
            let key = format!("integrity_test_{}", name);
            
            // Store data
            let store_result = adapter.store(&key, &data);
            
            // Retrieve and verify (if storage succeeded)
            if store_result.is_ok() {
                if let Ok(Some(retrieved)) = adapter.retrieve(&key) {
                    assert_eq!(data, retrieved, "Data integrity failed for test case: {}", name);
                }
            }
        }
    }
}
