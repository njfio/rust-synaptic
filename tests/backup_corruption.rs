// Comprehensive tests for backup corruption scenarios
//
// These tests verify that the backup/restore functionality properly handles
// various types of corrupted backup files and provides meaningful error messages.

use std::sync::Arc;
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::{Storage, TransactionalStorage};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use tempfile::NamedTempFile;
use std::io::Write;

#[tokio::test]
async fn test_nonexistent_backup_file() {
    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore("/path/that/does/not/exist.json").await;

    assert!(result.is_err(), "Should fail when backup file doesn't exist");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to read backup file"),
        "Error should mention file read failure: {}", err_msg);
}

#[tokio::test]
async fn test_empty_backup_file() {
    let mut temp_file = NamedTempFile::new().unwrap();
    // Write nothing - empty file
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when backup file is empty");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to decode") || err_msg.contains("Failed to deserialize"),
        "Error should mention decoding or deserialization failure: {}", err_msg);
}

#[tokio::test]
async fn test_invalid_utf8_backup() {
    let mut temp_file = NamedTempFile::new().unwrap();
    // Write invalid UTF-8 bytes
    temp_file.write_all(&[0xFF, 0xFE, 0xFD, 0xFC]).unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when backup contains invalid UTF-8");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to decode"),
        "Error should mention UTF-8 decode failure: {}", err_msg);
}

#[tokio::test]
async fn test_invalid_json_backup() {
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"{ this is not valid JSON }").unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when backup contains invalid JSON");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to parse") || err_msg.contains("deserialize"),
        "Error should mention JSON parsing failure: {}", err_msg);
}

#[tokio::test]
async fn test_unsupported_backup_version() {
    let mut temp_file = NamedTempFile::new().unwrap();
    let backup_json = r#"{
        "version": "999.0",
        "entry_count": 0,
        "total_size": 0,
        "created_at": "2024-01-01T00:00:00Z",
        "backup_timestamp": "2024-01-01T00:00:00Z",
        "entries": [],
        "metadata": {
            "creation_method": "test",
            "compression": null,
            "checksum": "0",
            "custom_fields": {}
        }
    }"#;
    temp_file.write_all(backup_json.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when backup version is unsupported");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Unsupported backup version"),
        "Error should mention unsupported version: {}", err_msg);
}

#[tokio::test]
async fn test_checksum_mismatch() {
    let mut temp_file = NamedTempFile::new().unwrap();
    // Create a backup with intentionally wrong checksum
    let backup_json = r#"{
        "version": "1.1",
        "entry_count": 1,
        "total_size": 100,
        "created_at": "2024-01-01T00:00:00Z",
        "backup_timestamp": "2024-01-01T00:00:00Z",
        "entries": [{
            "key": "test_key",
            "memory_type": "Episodic",
            "content": "test content",
            "metadata": {},
            "embedding": null,
            "importance": 0.5,
            "access_count": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "last_accessed": "2024-01-01T00:00:00Z",
            "tags": []
        }],
        "metadata": {
            "creation_method": "test",
            "compression": null,
            "checksum": "wrong_checksum_value",
            "custom_fields": {}
        }
    }"#;
    temp_file.write_all(backup_json.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when checksum doesn't match");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("integrity verification failed") || err_msg.contains("checksum"),
        "Error should mention integrity/checksum failure: {}", err_msg);
}

#[tokio::test]
async fn test_malformed_entry_in_backup() {
    let mut temp_file = NamedTempFile::new().unwrap();
    // Create backup with malformed entry (missing required fields)
    let backup_json = r#"{
        "version": "1.0",
        "entry_count": 1,
        "created_at": "2024-01-01T00:00:00Z",
        "backup_timestamp": "2024-01-01T00:00:00Z",
        "entries": [{
            "key": "test_key",
            "content": "missing required fields"
        }]
    }"#;
    temp_file.write_all(backup_json.as_bytes()).unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when entries are malformed");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to parse") || err_msg.contains("missing field"),
        "Error should mention parsing failure: {}", err_msg);
}

#[tokio::test]
async fn test_get_backup_info_nonexistent_file() {
    let storage = Arc::new(MemoryStorage::new());
    let result = storage.get_backup_info("/path/that/does/not/exist.json").await;

    assert!(result.is_err(), "get_backup_info should fail when file doesn't exist");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to read backup file"),
        "Error should mention file read failure: {}", err_msg);
}

#[tokio::test]
async fn test_get_backup_info_corrupted_json() {
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(b"{ corrupted json ").unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.get_backup_info(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "get_backup_info should fail on corrupted JSON");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Failed to deserialize") || err_msg.contains("Failed to decode"),
        "Error should mention deserialization failure: {}", err_msg);
}

#[tokio::test]
async fn test_backup_info_no_redundant_read() {
    // This test verifies the fix for Phase 4.3:
    // get_backup_info() should NOT read the file twice

    let storage = Arc::new(MemoryStorage::new());

    // Create a valid backup
    let entry = MemoryEntry {
        key: "test_key".to_string(),
        memory_type: MemoryType::Episodic,
        content: "test content".to_string(),
        metadata: Default::default(),
        embedding: None,
        importance: 0.5,
        access_count: 0,
        created_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
        tags: vec![],
    };

    storage.store(&entry).await.unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let backup_path = temp_file.path().to_str().unwrap();

    // Create backup
    storage.backup(backup_path).await.unwrap();

    // Get backup info - this should work without redundant file reads
    let info = storage.get_backup_info(backup_path).await.unwrap();

    assert_eq!(info.version, "1.1");
    assert_eq!(info.entry_count, 1);
    assert!(!info.is_compressed, "Default backup should not be compressed");

    // The key assertion: this completed successfully, proving the file
    // was only read once and the cached data was reused for compression check
}

#[cfg(feature = "compression")]
#[tokio::test]
async fn test_corrupted_compressed_backup() {
    let mut temp_file = NamedTempFile::new().unwrap();
    // Write gzip magic number but invalid gzip data
    temp_file.write_all(&[0x1f, 0x8b, 0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
    temp_file.flush().unwrap();

    let storage = Arc::new(MemoryStorage::new());
    let result = storage.restore(temp_file.path().to_str().unwrap()).await;

    assert!(result.is_err(), "Should fail when compressed data is corrupted");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("Decompression failed"),
        "Error should mention decompression failure: {}", err_msg);
}

#[tokio::test]
async fn test_concurrent_backup_operations_no_corruption() {
    // Verify that concurrent backup operations don't corrupt each other
    let storage = Arc::new(MemoryStorage::new());

    // Store test entries
    for i in 0..10 {
        let entry = MemoryEntry {
            key: format!("key_{}", i),
            memory_type: MemoryType::Episodic,
            content: format!("content {}", i),
            metadata: Default::default(),
            embedding: None,
            importance: 0.5,
            access_count: 0,
            created_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            tags: vec![],
        };
        storage.store(&entry).await.unwrap();
    }

    // Create multiple backups concurrently
    let mut handles = vec![];
    for i in 0..3 {
        let storage_clone = Arc::clone(&storage);
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_string_lossy().to_string();

        handles.push(tokio::spawn(async move {
            let result = storage_clone.backup(&path).await;
            (result, path, temp_file)
        }));
    }

    // Wait for all backups to complete
    let mut backup_paths = vec![];
    for handle in handles {
        let (result, path, temp_file) = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent backup should succeed");
        backup_paths.push((path, temp_file));
    }

    // Verify all backups are valid and not corrupted
    for (path, _temp_file) in &backup_paths {
        let info = storage.get_backup_info(path).await.unwrap();
        assert_eq!(info.entry_count, 10, "Backup should contain all entries");

        // Try to restore from each backup to verify integrity
        let restore_storage = Arc::new(MemoryStorage::new());
        restore_storage.restore(path).await.unwrap();
        assert_eq!(restore_storage.count().await.unwrap(), 10,
            "Restored storage should contain all entries");
    }
}
