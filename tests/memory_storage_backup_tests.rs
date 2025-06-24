//! Tests for MemoryStorage backup and restore functionality

use synaptic::memory::storage::memory::{MemoryStorage, BackupOptions, RestoreOptions};
use synaptic::memory::storage::Storage;
use synaptic::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
use tempfile::TempDir;
use std::collections::HashMap;

/// Create test memory entries
fn create_test_entries() -> Vec<MemoryEntry> {
    vec![
        MemoryEntry::new(
            "test_key_1".to_string(),
            "Test value 1".to_string(),
            MemoryType::ShortTerm,
        ),
        MemoryEntry::new(
            "test_key_2".to_string(),
            "Test value 2 with more content".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "test_key_3".to_string(),
            "Test value 3 with unicode: 世界 🌍".to_string(),
            MemoryType::ShortTerm,
        ),
    ]
}

#[tokio::test]
async fn test_basic_backup_and_restore() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("test_backup.json");
    
    let storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    
    // Store test entries
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Create backup
    let result = storage.backup(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    assert!(backup_path.exists());
    
    // Create new storage and restore
    let new_storage = MemoryStorage::new();
    let result = new_storage.restore(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Verify all entries were restored
    for entry in &test_entries {
        let retrieved = new_storage.retrieve(&entry.key).await.unwrap();
        assert!(retrieved.is_some());
        let retrieved_entry = retrieved.unwrap();
        assert_eq!(retrieved_entry.key, entry.key);
        assert_eq!(retrieved_entry.value, entry.value);
        assert_eq!(retrieved_entry.memory_type, entry.memory_type);
    }
    
    // Verify count
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, test_entries.len());
}

#[tokio::test]
async fn test_backup_with_compression() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("compressed_backup.json.gz");
    
    let storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    
    // Store test entries
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Create backup with compression
    let options = BackupOptions {
        compression: Some("gzip".to_string()),
        creation_method: "test_compression".to_string(),
        pretty_print: false,
        custom_fields: {
            let mut fields = HashMap::new();
            fields.insert("test_field".to_string(), "test_value".to_string());
            fields
        },
    };
    
    let result = storage.backup_with_options(backup_path.to_str().unwrap(), options).await;
    assert!(result.is_ok());
    assert!(backup_path.exists());
    
    // Restore from compressed backup
    let new_storage = MemoryStorage::new();
    let result = new_storage.restore(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Verify restoration
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, test_entries.len());
}

#[tokio::test]
async fn test_backup_info() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("info_test_backup.json");
    
    let storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    
    // Store test entries
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Get backup info
    let backup_info = storage.get_backup_info(backup_path.to_str().unwrap()).await.unwrap();
    
    assert_eq!(backup_info.entry_count, test_entries.len());
    assert!(backup_info.total_size > 0);
    assert_eq!(backup_info.version, "1.1");
    assert_eq!(backup_info.creation_method, "manual");
    assert!(!backup_info.is_compressed);
}

#[tokio::test]
async fn test_restore_with_merge_strategy() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("merge_test_backup.json");
    
    let storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    
    // Store test entries
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Create new storage with different data
    let new_storage = MemoryStorage::new();
    let additional_entry = MemoryEntry::new(
        "additional_key".to_string(),
        "Additional value".to_string(),
        MemoryType::ShortTerm,
    );
    new_storage.store(&additional_entry).await.unwrap();
    
    // Restore with merge strategy
    let restore_options = RestoreOptions {
        merge_strategy: true,
        overwrite_existing: false,
        verify_integrity: true,
    };
    
    let result = new_storage.restore_with_options(backup_path.to_str().unwrap(), restore_options).await;
    assert!(result.is_ok());
    
    // Verify both original and restored entries exist
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, test_entries.len() + 1); // Original entries + additional entry
    
    // Verify additional entry still exists
    let retrieved = new_storage.retrieve("additional_key").await.unwrap();
    assert!(retrieved.is_some());
}

#[tokio::test]
async fn test_restore_with_overwrite() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("overwrite_test_backup.json");
    
    let storage = MemoryStorage::new();
    
    // Store entry with original value
    let original_entry = MemoryEntry::new(
        "test_key".to_string(),
        "Original value".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&original_entry).await.unwrap();
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Create new storage and add entry with same key but different value
    let new_storage = MemoryStorage::new();
    let modified_entry = MemoryEntry::new(
        "test_key".to_string(),
        "Modified value".to_string(),
        MemoryType::LongTerm,
    );
    new_storage.store(&modified_entry).await.unwrap();
    
    // Restore with overwrite enabled
    let restore_options = RestoreOptions {
        merge_strategy: true,
        overwrite_existing: true,
        verify_integrity: true,
    };
    
    let result = new_storage.restore_with_options(backup_path.to_str().unwrap(), restore_options).await;
    assert!(result.is_ok());
    
    // Verify original value was restored (overwritten)
    let retrieved = new_storage.retrieve("test_key").await.unwrap();
    assert!(retrieved.is_some());
    let retrieved_entry = retrieved.unwrap();
    assert_eq!(retrieved_entry.value, "Original value");
    assert_eq!(retrieved_entry.memory_type, MemoryType::ShortTerm);
}

#[tokio::test]
async fn test_backup_integrity_verification() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("integrity_test_backup.json");
    
    let storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    
    // Store test entries
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Restore with integrity verification
    let new_storage = MemoryStorage::new();
    let restore_options = RestoreOptions {
        verify_integrity: true,
        ..Default::default()
    };
    
    let result = new_storage.restore_with_options(backup_path.to_str().unwrap(), restore_options).await;
    assert!(result.is_ok());
    
    // Verify restoration
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, test_entries.len());
}

#[tokio::test]
async fn test_empty_storage_backup_restore() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("empty_backup.json");
    
    let storage = MemoryStorage::new();
    
    // Create backup of empty storage
    let result = storage.backup(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Restore to new storage
    let new_storage = MemoryStorage::new();
    let result = new_storage.restore(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Verify empty state
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_large_data_backup_restore() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("large_data_backup.json");
    
    let storage = MemoryStorage::new();
    
    // Create entries with large data
    for i in 0..100 {
        let large_value = "x".repeat(10000); // 10KB per entry
        let entry = MemoryEntry::new(
            format!("large_key_{}", i),
            large_value,
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
    }
    
    // Create backup
    let result = storage.backup(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Restore
    let new_storage = MemoryStorage::new();
    let result = new_storage.restore(backup_path.to_str().unwrap()).await;
    assert!(result.is_ok());
    
    // Verify count
    let count = new_storage.count().await.unwrap();
    assert_eq!(count, 100);
    
    // Verify a few entries
    for i in [0, 50, 99] {
        let key = format!("large_key_{}", i);
        let retrieved = new_storage.retrieve(&key).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().value.len(), 10000);
    }
}

#[tokio::test]
async fn test_backup_error_handling() {
    let storage = MemoryStorage::new();
    
    // Test backup to invalid path
    let result = storage.backup("/invalid/path/backup.json").await;
    assert!(result.is_err());
    
    // Test restore from non-existent file
    let result = storage.restore("/non/existent/file.json").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_backup_with_custom_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("custom_metadata_backup.json");
    
    let storage = MemoryStorage::new();
    let test_entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test value".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&test_entry).await.unwrap();
    
    // Create backup with custom metadata
    let mut custom_fields = HashMap::new();
    custom_fields.insert("environment".to_string(), "test".to_string());
    custom_fields.insert("version".to_string(), "1.0.0".to_string());
    
    let options = BackupOptions {
        creation_method: "automated_test".to_string(),
        custom_fields,
        ..Default::default()
    };
    
    let result = storage.backup_with_options(backup_path.to_str().unwrap(), options).await;
    assert!(result.is_ok());
    
    // Get backup info to verify metadata
    let backup_info = storage.get_backup_info(backup_path.to_str().unwrap()).await.unwrap();
    assert_eq!(backup_info.creation_method, "automated_test");
}

#[tokio::test]
async fn test_concurrent_backup_operations() {
    use std::sync::Arc;
    use tokio::task;
    
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(MemoryStorage::new());
    
    // Store some test data
    for i in 0..10 {
        let entry = MemoryEntry::new(
            format!("concurrent_key_{}", i),
            format!("Concurrent value {}", i),
            MemoryType::ShortTerm,
        );
        storage.store(&entry).await.unwrap();
    }
    
    // Perform concurrent backup operations
    let mut handles = vec![];
    for i in 0..5 {
        let storage_clone = storage.clone();
        let backup_path = temp_dir.path().join(format!("concurrent_backup_{}.json", i));
        
        let handle = task::spawn(async move {
            storage_clone.backup(backup_path.to_str().unwrap()).await
        });
        handles.push(handle);
    }
    
    // Wait for all backups to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    // Verify all backup files were created
    for i in 0..5 {
        let backup_path = temp_dir.path().join(format!("concurrent_backup_{}.json", i));
        assert!(backup_path.exists());
    }
}
