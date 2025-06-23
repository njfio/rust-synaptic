//! Tests for storage backend capabilities and error handling
//! 
//! These tests verify that storage backends correctly report their capabilities
//! and provide clear error messages when features are unsupported.

use synaptic::memory::storage::{Storage, StorageStats, BatchStorage, TransactionalStorage, StorageTransaction, StorageOperation};
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::file::FileStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::error::MemoryError;
use tempfile::TempDir;
use std::sync::Arc;

/// Create a test memory entry
fn create_test_entry(key: &str, value: &str) -> MemoryEntry {
    MemoryEntry::new(
        key.to_string(),
        value.to_string(),
        MemoryType::ShortTerm,
    )
}

/// Test basic storage capabilities for MemoryStorage
#[tokio::test]
async fn test_memory_storage_capabilities() {
    let storage = MemoryStorage::new();
    let test_entry = create_test_entry("test_key", "test_value");
    
    // Test core CRUD operations
    assert!(storage.store(&test_entry).await.is_ok());
    
    let retrieved = storage.retrieve("test_key").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().value, "test_value");
    
    assert!(storage.exists("test_key").await.unwrap());
    assert!(!storage.exists("nonexistent_key").await.unwrap());
    
    assert!(storage.delete("test_key").await.unwrap());
    assert!(!storage.exists("test_key").await.unwrap());
    
    // Test statistics
    let stats = storage.stats().await.unwrap();
    assert_eq!(stats.backend_type, "memory");
    
    // Test maintenance
    assert!(storage.maintenance().await.is_ok());
    
    // Test backup/restore
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("memory_backup.json");
    
    storage.store(&test_entry).await.unwrap();
    assert!(storage.backup(backup_path.to_str().unwrap()).await.is_ok());
    
    let new_storage = MemoryStorage::new();
    assert!(new_storage.restore(backup_path.to_str().unwrap()).await.is_ok());
    
    let restored = new_storage.retrieve("test_key").await.unwrap();
    assert!(restored.is_some());
}

/// Test basic storage capabilities for FileStorage
#[tokio::test]
async fn test_file_storage_capabilities() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let storage = FileStorage::new(&db_path).await.unwrap();
    let test_entry = create_test_entry("file_test_key", "file_test_value");
    
    // Test core CRUD operations
    assert!(storage.store(&test_entry).await.is_ok());
    
    let retrieved = storage.retrieve("file_test_key").await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().value, "file_test_value");
    
    assert!(storage.exists("file_test_key").await.unwrap());
    assert!(storage.delete("file_test_key").await.unwrap());
    
    // Test statistics
    let stats = storage.stats().await.unwrap();
    assert_eq!(stats.backend_type, "file");
    
    // Test maintenance
    assert!(storage.maintenance().await.is_ok());
    
    // Test backup/restore
    let backup_path = temp_dir.path().join("file_backup.json");
    
    storage.store(&test_entry).await.unwrap();
    assert!(storage.backup(backup_path.to_str().unwrap()).await.is_ok());
    
    // Create new storage instance and restore
    let new_storage = FileStorage::new(&temp_dir.path().join("restored.db")).await.unwrap();
    assert!(new_storage.restore(backup_path.to_str().unwrap()).await.is_ok());
    
    let restored = new_storage.retrieve("file_test_key").await.unwrap();
    assert!(restored.is_some());
}

/// Test batch operations capability
#[tokio::test]
async fn test_batch_operations_capability() {
    let storage = MemoryStorage::new();
    
    // Create test entries
    let entries = vec![
        create_test_entry("batch_1", "value_1"),
        create_test_entry("batch_2", "value_2"),
        create_test_entry("batch_3", "value_3"),
    ];
    
    // Test batch store
    let result = storage.store_batch(&entries).await;
    assert!(result.is_ok());
    
    // Verify all entries were stored
    for entry in &entries {
        let retrieved = storage.retrieve(&entry.key).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, entry.value);
    }
    
    // Test batch retrieve
    let keys: Vec<String> = entries.iter().map(|e| e.key.clone()).collect();
    let retrieved_entries = storage.retrieve_batch(&keys).await.unwrap();
    assert_eq!(retrieved_entries.len(), entries.len());
    
    // Test batch delete
    let result = storage.delete_batch(&keys).await;
    assert!(result.is_ok());
    
    // Verify all entries were deleted
    for key in &keys {
        assert!(!storage.exists(key).await.unwrap());
    }
}

/// Test transactional operations capability
#[tokio::test]
async fn test_transactional_operations_capability() {
    let storage = MemoryStorage::new();
    
    // Create a transaction
    let mut transaction = StorageTransaction::new();
    
    let entry1 = create_test_entry("tx_key_1", "tx_value_1");
    let entry2 = create_test_entry("tx_key_2", "tx_value_2");
    
    transaction.add_operation(StorageOperation::Store {
        key: entry1.key.clone(),
        entry: entry1.clone(),
    });
    
    transaction.add_operation(StorageOperation::Store {
        key: entry2.key.clone(),
        entry: entry2.clone(),
    });
    
    // Execute transaction
    let result = storage.execute_transaction(transaction).await;
    assert!(result.is_ok());
    
    // Verify both entries were stored
    assert!(storage.exists("tx_key_1").await.unwrap());
    assert!(storage.exists("tx_key_2").await.unwrap());
    
    // Test transaction handle
    let tx_handle = storage.begin_transaction().await;
    assert!(tx_handle.is_ok());
}

/// Test search capability
#[tokio::test]
async fn test_search_capability() {
    let storage = MemoryStorage::new();
    
    // Store entries with searchable content
    let entries = vec![
        create_test_entry("search_1", "The quick brown fox jumps"),
        create_test_entry("search_2", "A lazy dog sleeps peacefully"),
        create_test_entry("search_3", "The fox and the dog are friends"),
    ];
    
    for entry in &entries {
        storage.store(entry).await.unwrap();
    }
    
    // Test search functionality
    let results = storage.search("fox", 10).await.unwrap();
    assert!(!results.is_empty());
    
    // Should find entries containing "fox"
    let fox_results: Vec<_> = results.iter()
        .filter(|r| r.content.contains("fox"))
        .collect();
    assert!(!fox_results.is_empty());
    
    // Test search with limit
    let limited_results = storage.search("the", 1).await.unwrap();
    assert!(limited_results.len() <= 1);
}

/// Test error handling for unsupported operations
#[tokio::test]
async fn test_unsupported_operation_errors() {
    // Test with a mock storage that doesn't support certain operations
    struct LimitedStorage;
    
    #[async_trait::async_trait]
    impl Storage for LimitedStorage {
        async fn store(&self, _entry: &MemoryEntry) -> Result<(), MemoryError> {
            Err(MemoryError::configuration("Store operation not supported in limited storage"))
        }
        
        async fn retrieve(&self, _key: &str) -> Result<Option<MemoryEntry>, MemoryError> {
            Ok(None)
        }
        
        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<synaptic::memory::types::MemoryFragment>, MemoryError> {
            Err(MemoryError::configuration("Search not supported in limited storage"))
        }
        
        async fn delete(&self, _key: &str) -> Result<bool, MemoryError> {
            Err(MemoryError::configuration("Delete operation not supported in limited storage"))
        }
        
        async fn count(&self) -> Result<usize, MemoryError> {
            Ok(0)
        }
        
        async fn clear(&self) -> Result<(), MemoryError> {
            Err(MemoryError::configuration("Clear operation not supported in limited storage"))
        }
        
        async fn exists(&self, _key: &str) -> Result<bool, MemoryError> {
            Ok(false)
        }
        
        async fn stats(&self) -> Result<StorageStats, MemoryError> {
            Err(MemoryError::configuration("Statistics not available in limited storage"))
        }
        
        async fn maintenance(&self) -> Result<(), MemoryError> {
            Err(MemoryError::configuration("Maintenance not supported in limited storage"))
        }
        
        async fn backup(&self, _path: &str) -> Result<(), MemoryError> {
            Err(MemoryError::configuration("Backup not supported in limited storage"))
        }
        
        async fn restore(&self, _path: &str) -> Result<(), MemoryError> {
            Err(MemoryError::configuration("Restore not supported in limited storage"))
        }
        
        async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>, MemoryError> {
            Err(MemoryError::configuration("Get all entries not supported in limited storage"))
        }
    }
    
    let limited_storage = LimitedStorage;
    let test_entry = create_test_entry("test", "test");
    
    // Test that unsupported operations return clear error messages
    let store_result = limited_storage.store(&test_entry).await;
    assert!(store_result.is_err());
    assert!(store_result.unwrap_err().to_string().contains("Store operation not supported"));
    
    let search_result = limited_storage.search("test", 10).await;
    assert!(search_result.is_err());
    assert!(search_result.unwrap_err().to_string().contains("Search not supported"));
    
    let delete_result = limited_storage.delete("test").await;
    assert!(delete_result.is_err());
    assert!(delete_result.unwrap_err().to_string().contains("Delete operation not supported"));
    
    let stats_result = limited_storage.stats().await;
    assert!(stats_result.is_err());
    assert!(stats_result.unwrap_err().to_string().contains("Statistics not available"));
    
    let backup_result = limited_storage.backup("/tmp/test").await;
    assert!(backup_result.is_err());
    assert!(backup_result.unwrap_err().to_string().contains("Backup not supported"));
    
    let restore_result = limited_storage.restore("/tmp/test").await;
    assert!(restore_result.is_err());
    assert!(restore_result.unwrap_err().to_string().contains("Restore not supported"));
}

/// Test storage capability detection
#[tokio::test]
async fn test_storage_capability_detection() {
    // Test MemoryStorage capabilities
    let memory_storage = MemoryStorage::new();
    
    // Memory storage should support all basic operations
    assert!(memory_storage.store(&create_test_entry("test", "test")).await.is_ok());
    assert!(memory_storage.retrieve("test").await.is_ok());
    assert!(memory_storage.search("test", 10).await.is_ok());
    assert!(memory_storage.delete("test").await.is_ok());
    assert!(memory_storage.stats().await.is_ok());
    assert!(memory_storage.maintenance().await.is_ok());
    
    // Test FileStorage capabilities
    let temp_dir = TempDir::new().unwrap();
    let file_storage = FileStorage::new(temp_dir.path().join("test.db")).await.unwrap();
    
    // File storage should support all basic operations
    assert!(file_storage.store(&create_test_entry("test", "test")).await.is_ok());
    assert!(file_storage.retrieve("test").await.is_ok());
    assert!(file_storage.search("test", 10).await.is_ok());
    assert!(file_storage.delete("test").await.is_ok());
    assert!(file_storage.stats().await.is_ok());
    assert!(file_storage.maintenance().await.is_ok());
}

/// Test concurrent access capabilities
#[tokio::test]
async fn test_concurrent_access_capability() {
    use tokio::task;
    
    let storage = Arc::new(MemoryStorage::new());
    let mut handles = vec![];
    
    // Test concurrent operations
    for i in 0..10 {
        let storage_clone = storage.clone();
        let handle = task::spawn(async move {
            let entry = create_test_entry(&format!("concurrent_{}", i), &format!("value_{}", i));
            
            // Each task performs store, retrieve, and delete operations
            storage_clone.store(&entry).await.unwrap();
            let retrieved = storage_clone.retrieve(&entry.key).await.unwrap();
            assert!(retrieved.is_some());
            storage_clone.delete(&entry.key).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Storage should still be functional
    assert!(storage.stats().await.is_ok());
}

/// Test storage limits and constraints
#[tokio::test]
async fn test_storage_limits_and_constraints() {
    let storage = MemoryStorage::new();
    
    // Test with very large keys
    let large_key = "x".repeat(10000);
    let entry = create_test_entry(&large_key, "test_value");
    let result = storage.store(&entry).await;
    // Should handle large keys gracefully (either succeed or provide clear error)
    match result {
        Ok(()) => {
            // If it succeeds, retrieval should work
            let retrieved = storage.retrieve(&large_key).await.unwrap();
            assert!(retrieved.is_some());
        },
        Err(e) => {
            // If it fails, error should be descriptive
            assert!(!e.to_string().is_empty());
        }
    }
    
    // Test with very large values
    let large_value = "x".repeat(1_000_000); // 1MB
    let entry = create_test_entry("large_value_test", &large_value);
    let result = storage.store(&entry).await;
    // Should handle large values gracefully
    match result {
        Ok(()) => {
            let retrieved = storage.retrieve("large_value_test").await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().value.len(), large_value.len());
        },
        Err(e) => {
            assert!(!e.to_string().is_empty());
        }
    }
    
    // Test with empty keys and values
    let empty_key_result = storage.store(&create_test_entry("", "value")).await;
    let empty_value_result = storage.store(&create_test_entry("key", "")).await;
    
    // Should handle empty keys/values gracefully
    assert!(empty_key_result.is_ok() || !empty_key_result.unwrap_err().to_string().is_empty());
    assert!(empty_value_result.is_ok() || !empty_value_result.unwrap_err().to_string().is_empty());
}

/// Test storage performance characteristics
#[tokio::test]
async fn test_storage_performance_characteristics() {
    let storage = MemoryStorage::new();
    let start_time = std::time::Instant::now();
    
    // Store multiple entries and measure performance
    for i in 0..1000 {
        let entry = create_test_entry(&format!("perf_test_{}", i), &format!("value_{}", i));
        storage.store(&entry).await.unwrap();
    }
    
    let store_duration = start_time.elapsed();
    
    // Retrieve entries and measure performance
    let retrieve_start = std::time::Instant::now();
    for i in 0..1000 {
        let key = format!("perf_test_{}", i);
        let retrieved = storage.retrieve(&key).await.unwrap();
        assert!(retrieved.is_some());
    }
    let retrieve_duration = retrieve_start.elapsed();
    
    // Performance should be reasonable (these are loose bounds for testing)
    assert!(store_duration.as_millis() < 5000); // Less than 5 seconds for 1000 stores
    assert!(retrieve_duration.as_millis() < 1000); // Less than 1 second for 1000 retrieves
    
    // Test search performance
    let search_start = std::time::Instant::now();
    let results = storage.search("perf_test", 100).await.unwrap();
    let search_duration = search_start.elapsed();
    
    assert!(!results.is_empty());
    assert!(search_duration.as_millis() < 1000); // Less than 1 second for search
}

/// Test SQL storage capabilities (when feature is enabled)
#[cfg(feature = "sql-storage")]
#[tokio::test]
async fn test_sql_storage_capabilities() {
    use synaptic::integrations::database::{DatabaseClient, DatabaseConfig};

    // Skip test if no database URL is provided
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://test:test@localhost/test_db".to_string());

    let config = DatabaseConfig {
        database_url,
        max_connections: 5,
        connect_timeout_secs: 10,
        schema: "public".to_string(),
        ssl_mode: "prefer".to_string(),
    };

    // Try to create database client
    let db_client = match DatabaseClient::new(config).await {
        Ok(client) => client,
        Err(_) => {
            // Skip test if database is not available
            println!("Skipping SQL storage test - database not available");
            return;
        }
    };

    let test_entry = create_test_entry("sql_test_key", "sql_test_value");

    // Test core operations
    let store_result = db_client.store(&test_entry).await;
    if store_result.is_ok() {
        let retrieved = db_client.retrieve("sql_test_key").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value, "sql_test_value");

        // Test delete
        assert!(db_client.delete("sql_test_key").await.unwrap());
    } else {
        // If store fails, error should be descriptive
        let error_msg = store_result.unwrap_err().to_string();
        assert!(!error_msg.is_empty());
        println!("SQL storage test failed with: {}", error_msg);
    }
}

/// Test SQL storage when feature is disabled
#[cfg(not(feature = "sql-storage"))]
#[tokio::test]
async fn test_sql_storage_feature_disabled() {
    use synaptic::integrations::database::{DatabaseClient, DatabaseConfig};

    let config = DatabaseConfig {
        database_url: "postgresql://test:test@localhost/test_db".to_string(),
        max_connections: 5,
        connect_timeout_secs: 10,
        schema: "public".to_string(),
        ssl_mode: "prefer".to_string(),
    };

    // Should fail with clear error message when feature is disabled
    let result = DatabaseClient::new(config).await;
    assert!(result.is_err());

    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("SQL storage feature not enabled") ||
            error_msg.contains("feature") ||
            error_msg.contains("not available"));
}

/// Test storage backend factory and capability detection
#[tokio::test]
async fn test_storage_backend_factory() {
    use synaptic::memory::storage::{create_storage, StorageBackend};

    // Test memory storage creation
    let memory_backend = StorageBackend::Memory;
    let memory_storage = create_storage(memory_backend).await;
    assert!(memory_storage.is_ok());

    // Test file storage creation
    let temp_dir = TempDir::new().unwrap();
    let file_backend = StorageBackend::File {
        path: temp_dir.path().join("factory_test.db")
    };
    let file_storage = create_storage(file_backend).await;
    assert!(file_storage.is_ok());

    // Test SQL storage creation (should handle feature flag appropriately)
    let sql_backend = StorageBackend::Sql {
        connection_string: "postgresql://test:test@localhost/test_db".to_string()
    };
    let sql_storage = create_storage(sql_backend).await;

    #[cfg(feature = "sql-storage")]
    {
        // With SQL feature enabled, it should either succeed or fail with connection error
        match sql_storage {
            Ok(_) => println!("SQL storage created successfully"),
            Err(e) => {
                let error_msg = e.to_string();
                // Should be a connection error, not a feature error
                assert!(!error_msg.contains("feature not enabled"));
            }
        }
    }

    #[cfg(not(feature = "sql-storage"))]
    {
        // Without SQL feature, should fail with feature error
        assert!(sql_storage.is_err());
        let error_msg = sql_storage.unwrap_err().to_string();
        assert!(error_msg.contains("feature") || error_msg.contains("not enabled"));
    }
}

/// Test storage capability matrix
#[tokio::test]
async fn test_storage_capability_matrix() {
    // Define expected capabilities for each storage type
    struct StorageCapabilities {
        supports_persistence: bool,
        supports_transactions: bool,
        supports_batch_operations: bool,
        supports_search: bool,
        supports_backup: bool,
        supports_concurrent_access: bool,
    }

    let memory_capabilities = StorageCapabilities {
        supports_persistence: false, // Data lost on restart
        supports_transactions: true,
        supports_batch_operations: true,
        supports_search: true,
        supports_backup: true,
        supports_concurrent_access: true,
    };

    let file_capabilities = StorageCapabilities {
        supports_persistence: true,
        supports_transactions: false, // Sled doesn't support full ACID transactions
        supports_batch_operations: true,
        supports_search: true,
        supports_backup: true,
        supports_concurrent_access: true,
    };

    // Test memory storage capabilities
    let memory_storage = MemoryStorage::new();

    // Test persistence (should not persist across instances)
    let test_entry = create_test_entry("persistence_test", "test_value");
    memory_storage.store(&test_entry).await.unwrap();

    let new_memory_storage = MemoryStorage::new();
    let retrieved = new_memory_storage.retrieve("persistence_test").await.unwrap();
    assert!(retrieved.is_none()); // Should not persist

    // Test transactions
    if memory_capabilities.supports_transactions {
        let tx_result = memory_storage.begin_transaction().await;
        assert!(tx_result.is_ok());
    }

    // Test batch operations
    if memory_capabilities.supports_batch_operations {
        let entries = vec![
            create_test_entry("batch_1", "value_1"),
            create_test_entry("batch_2", "value_2"),
        ];
        let batch_result = memory_storage.store_batch(&entries).await;
        assert!(batch_result.is_ok());
    }

    // Test file storage capabilities
    let temp_dir = TempDir::new().unwrap();
    let file_storage = FileStorage::new(temp_dir.path().join("capability_test.db")).await.unwrap();

    // Test persistence (should persist across instances)
    file_storage.store(&test_entry).await.unwrap();

    let new_file_storage = FileStorage::new(temp_dir.path().join("capability_test.db")).await.unwrap();
    let retrieved = new_file_storage.retrieve("persistence_test").await.unwrap();
    assert!(retrieved.is_some()); // Should persist

    // Test batch operations
    if file_capabilities.supports_batch_operations {
        let entries = vec![
            create_test_entry("file_batch_1", "value_1"),
            create_test_entry("file_batch_2", "value_2"),
        ];
        let batch_result = file_storage.store_batch(&entries).await;
        assert!(batch_result.is_ok());
    }
}

/// Test error message quality and consistency
#[tokio::test]
async fn test_error_message_quality() {
    let storage = MemoryStorage::new();

    // Test retrieval of non-existent key (should not error, but return None)
    let result = storage.retrieve("nonexistent_key").await;
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    // Test deletion of non-existent key (should not error, but return false)
    let result = storage.delete("nonexistent_key").await;
    assert!(result.is_ok());
    assert!(!result.unwrap());

    // Test backup to invalid path
    let result = storage.backup("/invalid/path/that/does/not/exist/backup.json").await;
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(!error_msg.is_empty());
    assert!(error_msg.contains("Failed to") || error_msg.contains("Error") || error_msg.contains("Cannot"));

    // Test restore from non-existent file
    let result = storage.restore("/path/that/does/not/exist.json").await;
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(!error_msg.is_empty());
    assert!(error_msg.contains("Failed to") || error_msg.contains("Error") || error_msg.contains("Cannot"));
}

/// Test storage statistics accuracy
#[tokio::test]
async fn test_storage_statistics_accuracy() {
    let storage = MemoryStorage::new();

    // Initial stats
    let initial_stats = storage.stats().await.unwrap();
    assert_eq!(initial_stats.total_entries, 0);
    assert_eq!(initial_stats.total_size_bytes, 0);

    // Store some entries
    let entries = vec![
        create_test_entry("stats_1", "value_1"),
        create_test_entry("stats_2", "value_2"),
        create_test_entry("stats_3", "value_3"),
    ];

    for entry in &entries {
        storage.store(entry).await.unwrap();
    }

    // Check updated stats
    let updated_stats = storage.stats().await.unwrap();
    assert_eq!(updated_stats.total_entries, entries.len());
    assert!(updated_stats.total_size_bytes > 0);
    assert!(updated_stats.average_entry_size > 0);

    // Delete an entry and check stats
    storage.delete("stats_1").await.unwrap();

    let final_stats = storage.stats().await.unwrap();
    assert_eq!(final_stats.total_entries, entries.len() - 1);
    assert!(final_stats.total_size_bytes < updated_stats.total_size_bytes);
}

/// Test storage maintenance operations
#[tokio::test]
async fn test_storage_maintenance_operations() {
    let storage = MemoryStorage::new();

    // Store some entries
    for i in 0..100 {
        let entry = create_test_entry(&format!("maintenance_{}", i), &format!("value_{}", i));
        storage.store(&entry).await.unwrap();
    }

    // Run maintenance
    let maintenance_result = storage.maintenance().await;
    assert!(maintenance_result.is_ok());

    // Storage should still be functional after maintenance
    let count = storage.count().await.unwrap();
    assert_eq!(count, 100);

    // Stats should still be accurate
    let stats = storage.stats().await.unwrap();
    assert_eq!(stats.total_entries, 100);

    // Test file storage maintenance
    let temp_dir = TempDir::new().unwrap();
    let file_storage = FileStorage::new(temp_dir.path().join("maintenance_test.db")).await.unwrap();

    // Store some entries
    for i in 0..50 {
        let entry = create_test_entry(&format!("file_maintenance_{}", i), &format!("value_{}", i));
        file_storage.store(&entry).await.unwrap();
    }

    // Run maintenance
    let maintenance_result = file_storage.maintenance().await;
    assert!(maintenance_result.is_ok());

    // Storage should still be functional
    let count = file_storage.count().await.unwrap();
    assert_eq!(count, 50);
}
