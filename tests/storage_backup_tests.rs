//! Comprehensive tests for storage backup and restore functionality

use synaptic::memory::storage::{Storage, memory::MemoryStorage, file::FileStorage};
use synaptic::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
use tempfile::TempDir;
use tokio;

/// Create test memory entries
fn create_test_entries() -> Vec<MemoryEntry> {
    vec![
        MemoryEntry::new(
            "test_key_1".to_string(),
            "This is test content for memory entry 1".to_string(),
            MemoryType::ShortTerm,
        ).with_metadata(MemoryMetadata::new()),
        
        MemoryEntry::new(
            "test_key_2".to_string(),
            "This is test content for memory entry 2".to_string(),
            MemoryType::LongTerm,
        ).with_metadata(MemoryMetadata::new()),
        
        MemoryEntry::new(
            "test_key_3".to_string(),
            "This is test content for memory entry 3 with more complex data".to_string(),
            MemoryType::LongTerm,
        ).with_metadata(MemoryMetadata::new()),
    ]
}

#[tokio::test]
async fn test_memory_storage_backup_restore() {
    let storage = MemoryStorage::new();
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("memory_backup.json");
    
    // Store test entries
    let test_entries = create_test_entries();
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Verify entries are stored
    assert_eq!(storage.count().await.unwrap(), 3);
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify backup file exists
    assert!(backup_path.exists());
    
    // Clear storage
    storage.clear().await.unwrap();
    assert_eq!(storage.count().await.unwrap(), 0);
    
    // Restore from backup
    storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify all entries are restored
    assert_eq!(storage.count().await.unwrap(), 3);
    
    // Verify content integrity
    for entry in &test_entries {
        let restored = storage.retrieve(&entry.key).await.unwrap().unwrap();
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.value, entry.value);
        assert_eq!(restored.memory_type, entry.memory_type);
    }
}

#[tokio::test]
async fn test_file_storage_backup_restore() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let backup_path = temp_dir.path().join("file_backup.json");
    
    let storage = FileStorage::new(&db_path).await.unwrap();
    
    // Store test entries
    let test_entries = create_test_entries();
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Verify entries are stored
    assert_eq!(storage.count().await.unwrap(), 3);
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify backup file exists
    assert!(backup_path.exists());
    
    // Clear storage
    storage.clear().await.unwrap();
    assert_eq!(storage.count().await.unwrap(), 0);
    
    // Restore from backup
    storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify all entries are restored
    assert_eq!(storage.count().await.unwrap(), 3);
    
    // Verify content integrity
    for entry in &test_entries {
        let restored = storage.retrieve(&entry.key).await.unwrap().unwrap();
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.value, entry.value);
        assert_eq!(restored.memory_type, entry.memory_type);
    }
}

#[tokio::test]
async fn test_backup_format_validation() {
    let storage = MemoryStorage::new();
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("backup.json");
    
    // Store a test entry
    let entry = create_test_entries().into_iter().next().unwrap();
    storage.store(&entry).await.unwrap();
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Read and verify backup format
    let backup_content = tokio::fs::read_to_string(&backup_path).await.unwrap();
    let backup_data: serde_json::Value = serde_json::from_str(&backup_content).unwrap();
    
    // Verify required fields
    assert!(backup_data.get("entries").is_some());
    assert!(backup_data.get("created_at").is_some());
    assert!(backup_data.get("backup_timestamp").is_some());
    assert!(backup_data.get("version").is_some());
    assert!(backup_data.get("entry_count").is_some());
    
    // Verify version
    assert_eq!(backup_data["version"], "1.0");
    
    // Verify entry count
    assert_eq!(backup_data["entry_count"], 1);
}

#[tokio::test]
async fn test_backup_restore_empty_storage() {
    let storage = MemoryStorage::new();
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("empty_backup.json");
    
    // Create backup of empty storage
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Add some data
    let entry = create_test_entries().into_iter().next().unwrap();
    storage.store(&entry).await.unwrap();
    assert_eq!(storage.count().await.unwrap(), 1);
    
    // Restore empty backup
    storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify storage is empty
    assert_eq!(storage.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_backup_restore_large_dataset() {
    let storage = MemoryStorage::new();
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("large_backup.json");
    
    // Create a larger dataset
    let mut large_entries = Vec::new();
    for i in 0..100 {
        large_entries.push(MemoryEntry::new(
            format!("key_{}", i),
            format!("Content for entry number {} with some additional data to make it larger", i),
            MemoryType::ShortTerm,
        ).with_metadata(MemoryMetadata::new()));
    }
    
    // Store all entries
    for entry in &large_entries {
        storage.store(entry).await.unwrap();
    }
    
    assert_eq!(storage.count().await.unwrap(), 100);
    
    // Create backup
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Clear and restore
    storage.clear().await.unwrap();
    storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify all entries are restored
    assert_eq!(storage.count().await.unwrap(), 100);
    
    // Spot check a few entries
    let restored_0 = storage.retrieve("key_0").await.unwrap().unwrap();
    assert_eq!(restored_0.key, "key_0");
    
    let restored_50 = storage.retrieve("key_50").await.unwrap().unwrap();
    assert_eq!(restored_50.key, "key_50");
    
    let restored_99 = storage.retrieve("key_99").await.unwrap().unwrap();
    assert_eq!(restored_99.key, "key_99");
}

#[tokio::test]
async fn test_backup_error_handling() {
    let storage = MemoryStorage::new();
    
    // Test backup to invalid path
    let result = storage.backup("/invalid/path/backup.json").await;
    assert!(result.is_err());
    
    // Test restore from non-existent file
    let result = storage.restore("/non/existent/backup.json").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_cross_storage_migration() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("migration.db");
    let backup_path = temp_dir.path().join("migration.json");
    
    // Create data in memory storage
    let memory_storage = MemoryStorage::new();
    let test_entries = create_test_entries();
    for entry in &test_entries {
        memory_storage.store(entry).await.unwrap();
    }
    
    // Backup from memory storage
    memory_storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    
    // Restore to file storage
    let file_storage = FileStorage::new(&db_path).await.unwrap();
    file_storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Verify migration
    assert_eq!(file_storage.count().await.unwrap(), 3);
    
    for entry in &test_entries {
        let restored = file_storage.retrieve(&entry.key).await.unwrap().unwrap();
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.value, entry.value);
    }
}

#[tokio::test]
async fn test_backup_statistics_consistency() {
    let storage = MemoryStorage::new();
    let temp_dir = TempDir::new().unwrap();
    let backup_path = temp_dir.path().join("stats_backup.json");
    
    // Store test entries
    let test_entries = create_test_entries();
    for entry in &test_entries {
        storage.store(entry).await.unwrap();
    }
    
    // Get stats before backup
    let stats_before = storage.stats().await.unwrap();
    
    // Create backup and restore
    storage.backup(backup_path.to_str().unwrap()).await.unwrap();
    storage.clear().await.unwrap();
    storage.restore(backup_path.to_str().unwrap()).await.unwrap();
    
    // Get stats after restore
    let stats_after = storage.stats().await.unwrap();
    
    // Verify statistics consistency
    assert_eq!(stats_before.total_entries, stats_after.total_entries);
    // Note: total_size_bytes might differ slightly due to internal storage differences
    assert_eq!(stats_before.storage_type, stats_after.storage_type);
}
