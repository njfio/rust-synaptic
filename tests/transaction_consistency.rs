//! Transaction consistency tests for memory storage
//!
//! These tests verify that:
//! 1. Transactions commit to the live storage (not a copy)
//! 2. Rollback properly discards changes
//! 3. Concurrent transactions work correctly
//! 4. Transaction isolation is maintained

use synaptic::memory::storage::{Storage, TransactionalStorage};
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_transaction_commit_writes_to_live_storage() {
    // This is the critical test for the bug fix:
    // Transactions must write to the actual storage, not a copy

    let storage = Arc::new(MemoryStorage::new());

    // Store an initial entry directly
    let entry1 = MemoryEntry::new(
        "existing_key".to_string(),
        "existing_value".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry1).await.unwrap();

    // Verify it exists
    assert_eq!(storage.count().await.unwrap(), 1);

    // Begin a transaction and add another entry
    let mut transaction = storage.begin_transaction().await.unwrap();

    let entry2 = MemoryEntry::new(
        "transaction_key".to_string(),
        "transaction_value".to_string(),
        MemoryType::ShortTerm,
    );

    transaction.store("transaction_key", &entry2).await.unwrap();
    transaction.commit().await.unwrap();

    // CRITICAL: The transactional entry should now be in the live storage
    assert_eq!(storage.count().await.unwrap(), 2, "Transaction did not commit to live storage");

    let retrieved = storage.retrieve("transaction_key").await.unwrap();
    assert!(retrieved.is_some(), "Transaction entry not found in live storage");
    assert_eq!(retrieved.unwrap().value, "transaction_value");
}

#[tokio::test]
async fn test_transaction_rollback_discards_changes() {
    let storage = Arc::new(MemoryStorage::new());

    // Store an initial entry
    let entry1 = MemoryEntry::new(
        "original_key".to_string(),
        "original_value".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry1).await.unwrap();
    assert_eq!(storage.count().await.unwrap(), 1);

    // Begin a transaction and add entries
    let mut transaction = storage.begin_transaction().await.unwrap();

    let entry2 = MemoryEntry::new(
        "temp_key1".to_string(),
        "temp_value1".to_string(),
        MemoryType::ShortTerm,
    );
    let entry3 = MemoryEntry::new(
        "temp_key2".to_string(),
        "temp_value2".to_string(),
        MemoryType::ShortTerm,
    );

    transaction.store("temp_key1", &entry2).await.unwrap();
    transaction.store("temp_key2", &entry3).await.unwrap();

    // Rollback the transaction
    transaction.rollback().await.unwrap();

    // Verify changes were discarded
    assert_eq!(storage.count().await.unwrap(), 1, "Rollback did not discard changes");
    assert!(storage.retrieve("temp_key1").await.unwrap().is_none());
    assert!(storage.retrieve("temp_key2").await.unwrap().is_none());

    // Original entry should still exist
    assert!(storage.retrieve("original_key").await.unwrap().is_some());
}

#[tokio::test]
async fn test_transaction_update_existing_entry() {
    let storage = Arc::new(MemoryStorage::new());

    // Store an initial entry
    let entry1 = MemoryEntry::new(
        "update_key".to_string(),
        "original_value".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry1).await.unwrap();

    // Update via transaction
    let mut transaction = storage.begin_transaction().await.unwrap();

    let updated_entry = MemoryEntry::new(
        "update_key".to_string(),
        "updated_value".to_string(),
        MemoryType::LongTerm,
    );

    transaction.update("update_key", &updated_entry).await.unwrap();
    transaction.commit().await.unwrap();

    // Verify update was applied
    let retrieved = storage.retrieve("update_key").await.unwrap().unwrap();
    assert_eq!(retrieved.value, "updated_value");
    assert_eq!(retrieved.memory_type, MemoryType::LongTerm);
}

#[tokio::test]
async fn test_transaction_delete_entry() {
    let storage = Arc::new(MemoryStorage::new());

    // Store entries
    let entry1 = MemoryEntry::new(
        "delete_key".to_string(),
        "delete_value".to_string(),
        MemoryType::ShortTerm,
    );
    let entry2 = MemoryEntry::new(
        "keep_key".to_string(),
        "keep_value".to_string(),
        MemoryType::ShortTerm,
    );

    storage.store(&entry1).await.unwrap();
    storage.store(&entry2).await.unwrap();
    assert_eq!(storage.count().await.unwrap(), 2);

    // Delete via transaction
    let mut transaction = storage.begin_transaction().await.unwrap();
    transaction.delete("delete_key").await.unwrap();
    transaction.commit().await.unwrap();

    // Verify deletion
    assert_eq!(storage.count().await.unwrap(), 1);
    assert!(storage.retrieve("delete_key").await.unwrap().is_none());
    assert!(storage.retrieve("keep_key").await.unwrap().is_some());
}

#[tokio::test]
async fn test_transaction_multiple_operations() {
    let storage = Arc::new(MemoryStorage::new());

    // Setup initial state
    let entry1 = MemoryEntry::new(
        "existing".to_string(),
        "original".to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&entry1).await.unwrap();

    // Execute multiple operations in one transaction
    let mut transaction = storage.begin_transaction().await.unwrap();

    // Store new entry
    let new_entry = MemoryEntry::new(
        "new_key".to_string(),
        "new_value".to_string(),
        MemoryType::ShortTerm,
    );
    transaction.store("new_key", &new_entry).await.unwrap();

    // Update existing entry
    let updated_entry = MemoryEntry::new(
        "existing".to_string(),
        "updated".to_string(),
        MemoryType::LongTerm,
    );
    transaction.update("existing", &updated_entry).await.unwrap();

    // Store and delete another entry
    let temp_entry = MemoryEntry::new(
        "temp".to_string(),
        "temp_value".to_string(),
        MemoryType::ShortTerm,
    );
    transaction.store("temp", &temp_entry).await.unwrap();
    transaction.delete("temp").await.unwrap();

    transaction.commit().await.unwrap();

    // Verify all operations were applied
    assert_eq!(storage.count().await.unwrap(), 2); // existing + new_key (temp was deleted)

    let existing = storage.retrieve("existing").await.unwrap().unwrap();
    assert_eq!(existing.value, "updated");

    let new = storage.retrieve("new_key").await.unwrap().unwrap();
    assert_eq!(new.value, "new_value");

    assert!(storage.retrieve("temp").await.unwrap().is_none());
}

#[tokio::test]
async fn test_concurrent_transactions() {
    let storage = Arc::new(MemoryStorage::new());

    // Spawn multiple concurrent transactions
    let mut handles = vec![];

    for i in 0..10 {
        let storage_clone = Arc::clone(&storage);
        let handle = tokio::spawn(async move {
            let mut transaction = storage_clone.begin_transaction().await.unwrap();

            let entry = MemoryEntry::new(
                format!("concurrent_key_{}", i),
                format!("value_{}", i),
                MemoryType::ShortTerm,
            );

            transaction.store(&format!("concurrent_key_{}", i), &entry).await.unwrap();
            transaction.commit().await.unwrap();
        });
        handles.push(handle);
    }

    // Wait for all transactions to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all entries were stored
    assert_eq!(storage.count().await.unwrap(), 10);

    for i in 0..10 {
        let key = format!("concurrent_key_{}", i);
        let entry = storage.retrieve(&key).await.unwrap();
        assert!(entry.is_some(), "Entry {} not found", i);
        assert_eq!(entry.unwrap().value, format!("value_{}", i));
    }
}

#[tokio::test]
async fn test_transaction_isolation() {
    // Test that uncommitted transaction changes are not visible to other readers
    let storage = Arc::new(MemoryStorage::new());

    // Begin a transaction but don't commit yet
    let mut transaction = storage.begin_transaction().await.unwrap();

    let entry = MemoryEntry::new(
        "isolated_key".to_string(),
        "isolated_value".to_string(),
        MemoryType::ShortTerm,
    );

    transaction.store("isolated_key", &entry).await.unwrap();

    // Verify the entry is NOT yet visible in the storage
    assert_eq!(storage.count().await.unwrap(), 0);
    assert!(storage.retrieve("isolated_key").await.unwrap().is_none());

    // Now commit
    transaction.commit().await.unwrap();

    // Now it should be visible
    assert_eq!(storage.count().await.unwrap(), 1);
    assert!(storage.retrieve("isolated_key").await.unwrap().is_some());
}

#[tokio::test]
async fn test_empty_transaction() {
    let storage = Arc::new(MemoryStorage::new());

    // Create and commit an empty transaction
    let transaction = storage.begin_transaction().await.unwrap();
    transaction.commit().await.unwrap();

    // Storage should remain empty
    assert_eq!(storage.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_transaction_preserves_existing_data() {
    let storage = Arc::new(MemoryStorage::new());

    // Add some initial data
    for i in 0..5 {
        let entry = MemoryEntry::new(
            format!("key_{}", i),
            format!("value_{}", i),
            MemoryType::ShortTerm,
        );
        storage.store(&entry).await.unwrap();
    }

    assert_eq!(storage.count().await.unwrap(), 5);

    // Execute a transaction that adds more data
    let mut transaction = storage.begin_transaction().await.unwrap();

    for i in 5..10 {
        let entry = MemoryEntry::new(
            format!("key_{}", i),
            format!("value_{}", i),
            MemoryType::ShortTerm,
        );
        transaction.store(&format!("key_{}", i), &entry).await.unwrap();
    }

    transaction.commit().await.unwrap();

    // All 10 entries should now exist
    assert_eq!(storage.count().await.unwrap(), 10);

    // Verify original entries are still intact
    for i in 0..5 {
        let entry = storage.retrieve(&format!("key_{}", i)).await.unwrap().unwrap();
        assert_eq!(entry.value, format!("value_{}", i));
    }
}
