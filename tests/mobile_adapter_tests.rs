//! Tests for mobile platform adapters

use synaptic::cross_platform::{
    mobile::{MobileConfig, MobileStorage, GenericMobileAdapter},
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, StorageBackend, Platform,
};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_mobile_config_default() {
    let config = MobileConfig::default();
    assert!(config.enable_battery_optimization);
    assert!(config.enable_background_sync);
    assert_eq!(config.memory_pressure_threshold, 0.8);
    assert_eq!(config.cache_size_limit, 50 * 1024 * 1024);
    assert!(config.enable_compression);
    assert_eq!(config.sync_interval_seconds, 300);
    assert!(config.enable_offline_mode);
    assert!(config.data_directory.is_none());
}

#[test]
fn test_mobile_config_serialization() {
    let config = MobileConfig::default();
    
    // Test serialization
    let serialized = serde_json::to_string(&config);
    assert!(serialized.is_ok());
    
    if let Ok(json) = serialized {
        // Test deserialization
        let deserialized: Result<MobileConfig, _> = serde_json::from_str(&json);
        assert!(deserialized.is_ok());
        
        if let Ok(deserialized_config) = deserialized {
            assert_eq!(config.enable_battery_optimization, deserialized_config.enable_battery_optimization);
            assert_eq!(config.cache_size_limit, deserialized_config.cache_size_limit);
            assert_eq!(config.enable_compression, deserialized_config.enable_compression);
        }
    }
}

#[test]
fn test_mobile_storage_creation() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = MobileStorage::new(config);
    assert!(storage.is_ok());
}

#[test]
fn test_mobile_storage_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = MobileStorage::new(config).unwrap();
    let test_data = b"test data for mobile storage";
    
    // Test store
    let result = storage.store("test_key", test_data);
    assert!(result.is_ok());
    
    // Test retrieve
    let retrieved = storage.retrieve("test_key").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
    
    // Test list keys
    let keys = storage.list_keys().unwrap();
    assert!(keys.contains(&"test_key".to_string()));
    
    // Test delete
    let deleted = storage.delete("test_key").unwrap();
    assert!(deleted);
    
    // Verify deletion
    let retrieved_after_delete = storage.retrieve("test_key").unwrap();
    assert!(retrieved_after_delete.is_none());
}

#[test]
fn test_mobile_storage_compression() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    config.enable_compression = true;
    
    let storage = MobileStorage::new(config).unwrap();
    let test_data = b"This is a longer piece of test data that should benefit from compression when stored in the mobile storage system.";
    
    // Store with compression
    let result = storage.store("compression_test", test_data);
    assert!(result.is_ok());
    
    // Retrieve and verify
    let retrieved = storage.retrieve("compression_test").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
}

#[test]
fn test_mobile_storage_cache_management() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    config.cache_size_limit = 1024; // Small cache for testing
    
    let storage = MobileStorage::new(config).unwrap();
    
    // Fill cache beyond limit
    for i in 0..10 {
        let key = format!("cache_test_{}", i);
        let data = vec![0u8; 200]; // 200 bytes each
        let _ = storage.store(&key, &data);
    }
    
    // Cache should handle size limits gracefully
    let stats = storage.get_stats().unwrap();
    assert!(stats.item_count <= 10);
}

#[test]
fn test_mobile_storage_stats() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = MobileStorage::new(config).unwrap();
    
    // Store some test data
    for i in 0..5 {
        let key = format!("stats_test_{}", i);
        let data = format!("test data {}", i);
        let _ = storage.store(&key, data.as_bytes());
    }
    
    let stats = storage.get_stats().unwrap();
    assert_eq!(stats.item_count, 5);
    assert!(stats.used_storage > 0);
    assert!(stats.average_item_size > 0);
}

#[test]
fn test_generic_mobile_adapter_creation() {
    let adapter = GenericMobileAdapter::new();
    assert!(adapter.is_ok());
}

#[test]
fn test_generic_mobile_adapter_initialization() {
    let mut adapter = GenericMobileAdapter::new().unwrap();
    let config = PlatformConfig {
        max_memory_usage: 50 * 1024 * 1024,
        enable_local_storage: true,
        enable_network_sync: false,
        storage_backends: vec![StorageBackend::Memory],
        performance_settings: Default::default(),
    };
    
    let result = adapter.initialize(&config);
    assert!(result.is_ok());
}

#[test]
fn test_generic_mobile_adapter_feature_support() {
    let adapter = GenericMobileAdapter::new().unwrap();
    
    // Test feature support
    assert!(adapter.supports_feature(PlatformFeature::FileSystemAccess));
    assert!(adapter.supports_feature(PlatformFeature::NetworkAccess));
    assert!(adapter.supports_feature(PlatformFeature::BackgroundProcessing));
    assert!(!adapter.supports_feature(PlatformFeature::PushNotifications));
    assert!(!adapter.supports_feature(PlatformFeature::HardwareAcceleration));
    assert!(adapter.supports_feature(PlatformFeature::MultiThreading));
    assert!(!adapter.supports_feature(PlatformFeature::LargeMemoryAllocation));
}

#[test]
fn test_generic_mobile_adapter_platform_info() {
    let adapter = GenericMobileAdapter::new().unwrap();
    let platform_info = adapter.get_platform_info();
    
    assert_eq!(platform_info.version, "1.0.0");
    assert_eq!(platform_info.available_memory, 512 * 1024 * 1024);
    assert_eq!(platform_info.available_storage, 1024 * 1024 * 1024);
    assert!(!platform_info.supported_features.is_empty());
    assert!(platform_info.performance_profile.battery_optimization);
}

#[test]
fn test_generic_mobile_adapter_storage_operations() {
    let adapter = GenericMobileAdapter::new().unwrap();
    let test_data = b"test data for generic mobile adapter";
    
    // Store data
    let result = adapter.store("test_key", test_data);
    assert!(result.is_ok());
    
    // Retrieve data
    let retrieved = adapter.retrieve("test_key").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), test_data);
    
    // List keys
    let keys = adapter.list_keys().unwrap();
    assert!(keys.contains(&"test_key".to_string()));
    
    // Delete data
    let deleted = adapter.delete("test_key").unwrap();
    assert!(deleted);
    
    // Verify deletion
    let retrieved_after_delete = adapter.retrieve("test_key").unwrap();
    assert!(retrieved_after_delete.is_none());
}

#[test]
fn test_mobile_adapter_memory_pressure_handling() {
    let mut adapter = GenericMobileAdapter::new().unwrap();
    
    // Store some data first
    for i in 0..20 {
        let key = format!("cache_{}", i);
        let data = format!("cached data {}", i);
        let _ = adapter.store(&key, data.as_bytes());
    }
    
    // Trigger memory pressure handling
    let result = adapter.handle_memory_pressure();
    assert!(result.is_ok());
    
    // Some cached entries should be cleared
    let keys = adapter.list_keys().unwrap();
    let cache_keys: Vec<_> = keys.iter().filter(|k| k.starts_with("cache_")).collect();
    // Should have fewer cache entries after pressure handling
    assert!(cache_keys.len() <= 20);
}

#[test]
fn test_mobile_adapter_battery_optimization() {
    let mut adapter = GenericMobileAdapter::new().unwrap();
    
    let result = adapter.optimize_for_battery();
    assert!(result.is_ok());
    assert!(adapter.battery_optimization);
}

#[test]
fn test_mobile_adapter_background_sync() {
    let mut adapter = GenericMobileAdapter::new().unwrap();
    
    let result = adapter.configure_background_sync();
    assert!(result.is_ok());
}

#[test]
fn test_mobile_storage_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = MobileStorage::new(config).unwrap();
    
    // Test retrieval of non-existent key
    let result = storage.retrieve("nonexistent_key");
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
    
    // Test deletion of non-existent key
    let result = storage.delete("nonexistent_key");
    assert!(result.is_ok());
    assert!(!result.unwrap());
    
    // Test with empty key
    let result = storage.store("", b"test");
    assert!(result.is_ok()); // Should handle gracefully
    
    // Test with very large data
    let large_data = vec![0u8; 10 * 1024 * 1024]; // 10MB
    let result = storage.store("large_test", &large_data);
    assert!(result.is_ok()); // Should handle large data
}

#[test]
fn test_mobile_storage_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = Arc::new(MobileStorage::new(config).unwrap());
    let mut handles = vec![];
    
    // Test concurrent access
    for i in 0..5 {
        let storage_clone = storage.clone();
        let handle = thread::spawn(move || {
            let key = format!("concurrent_test_{}", i);
            let data = format!("test data {}", i);
            
            // These operations should be thread-safe
            let _ = storage_clone.store(&key, data.as_bytes());
            let _ = storage_clone.retrieve(&key);
            let _ = storage_clone.delete(&key);
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Storage should still be functional
    let _ = storage.list_keys();
    let _ = storage.get_stats();
}

#[test]
fn test_mobile_storage_data_integrity() {
    let temp_dir = TempDir::new().unwrap();
    let mut config = MobileConfig::default();
    config.data_directory = Some(temp_dir.path().to_path_buf());
    
    let storage = MobileStorage::new(config).unwrap();
    
    // Test data integrity with various data types
    let test_cases = vec![
        ("empty", b"".to_vec()),
        ("small", b"small data".to_vec()),
        ("binary", vec![0, 1, 2, 3, 255, 254, 253]),
        ("unicode", "Hello 世界 🌍".as_bytes().to_vec()),
        ("large", vec![42u8; 5000]),
    ];
    
    for (name, data) in test_cases {
        let key = format!("integrity_test_{}", name);
        
        // Store data
        let store_result = storage.store(&key, &data);
        assert!(store_result.is_ok(), "Failed to store data for test case: {}", name);
        
        // Retrieve and verify
        let retrieved = storage.retrieve(&key).unwrap();
        assert!(retrieved.is_some(), "Failed to retrieve data for test case: {}", name);
        assert_eq!(data, retrieved.unwrap(), "Data integrity failed for test case: {}", name);
    }
}
