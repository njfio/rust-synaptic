//! Integration tests for real lifecycle archival/compression/deletion against Storage.

use synaptic::memory::management::lifecycle::{
    LifecycleAction, LifecycleCondition, LifecyclePolicy, MemoryLifecycleManager, MemoryStage,
};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryType};

fn policy(id: &str, tag: &str, action: LifecycleAction) -> LifecyclePolicy {
    LifecyclePolicy {
        id: id.to_string(),
        name: id.to_string(),
        conditions: vec![LifecycleCondition::HasTags {
            tags: vec![tag.to_string()],
        }],
        actions: vec![action],
        active: true,
        priority: 100,
    }
}

fn entry_with_tag(key: &str, value: &str, tag: &str) -> MemoryEntry {
    let mut entry = MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::LongTerm);
    entry.metadata.tags.push(tag.to_string());
    entry
}

#[tokio::test]
async fn archival_marks_stored_entry_archived_and_removes_from_active_set() {
    let storage = MemoryStorage::new();
    let mut manager = MemoryLifecycleManager::new();
    manager.add_policy(policy("archive_tagged", "old", LifecycleAction::Archive));

    let entry = entry_with_tag("mem_arch", "some content to archive", "old");
    storage.store(&entry).await.expect("store");
    manager
        .track_memory_creation(&storage, &entry)
        .await
        .expect("track");

    // Lifecycle state must be Archived and not in any active-ish stage
    let state = manager.get_memory_state("mem_arch").expect("state");
    assert_eq!(state.stage, MemoryStage::Archived);
    assert!(manager
        .get_memories_in_stage(&MemoryStage::Active)
        .is_empty());
    assert!(manager
        .get_memories_in_stage(&MemoryStage::Created)
        .is_empty());

    // The stored entry itself must be retrievable in archived form
    let stored = storage
        .retrieve("mem_arch")
        .await
        .expect("retrieve")
        .expect("entry still present");
    assert!(
        stored.metadata.tags.iter().any(|t| t == "archived"),
        "stored entry should carry the 'archived' tag, tags: {:?}",
        stored.metadata.tags
    );
}

#[tokio::test]
async fn compression_reduces_stored_size_for_large_entry() {
    let storage = MemoryStorage::new();
    let mut manager = MemoryLifecycleManager::new();
    manager.add_policy(policy(
        "compress_tagged",
        "bulky",
        LifecycleAction::Compress,
    ));

    let big_value = "synaptic memory content block ".repeat(500); // ~15KB, highly compressible
    let original_size = big_value.len();
    let entry = entry_with_tag("mem_comp", &big_value, "bulky");
    storage.store(&entry).await.expect("store");
    manager
        .track_memory_creation(&storage, &entry)
        .await
        .expect("track");

    let stored = storage
        .retrieve("mem_comp")
        .await
        .expect("retrieve")
        .expect("entry present");
    assert!(stored.value.starts_with("COMPRESSED:"));
    assert!(
        stored.value.len() < original_size,
        "stored size {} should be less than original {}",
        stored.value.len(),
        original_size
    );
    assert!(stored.metadata.tags.iter().any(|t| t == "compressed"));
}

#[tokio::test]
async fn deletion_removes_entry_from_storage() {
    let storage = MemoryStorage::new();
    let mut manager = MemoryLifecycleManager::new();
    manager.add_policy(policy("delete_tagged", "temp", LifecycleAction::Delete));

    let entry = entry_with_tag("mem_del", "ephemeral content", "temp");
    storage.store(&entry).await.expect("store");
    manager
        .track_memory_creation(&storage, &entry)
        .await
        .expect("track");

    let state = manager.get_memory_state("mem_del").expect("state");
    assert_eq!(state.stage, MemoryStage::Deleted);
    let stored = storage.retrieve("mem_del").await.expect("retrieve");
    assert!(stored.is_none(), "entry should be deleted from storage");
}

#[tokio::test]
async fn snapshot_custom_action_stores_snapshot_copy() {
    let storage = MemoryStorage::new();
    let mut manager = MemoryLifecycleManager::new();
    manager.add_policy(policy(
        "snapshot_tagged",
        "important",
        LifecycleAction::Custom {
            action: "create_snapshot".to_string(),
        },
    ));

    let entry = entry_with_tag("mem_snap", "content worth snapshotting", "important");
    storage.store(&entry).await.expect("store");
    manager
        .track_memory_creation(&storage, &entry)
        .await
        .expect("track");

    let keys = storage.list_keys().await.expect("keys");
    let snapshot_key = keys
        .iter()
        .find(|k| k.starts_with("snapshot:mem_snap:"))
        .expect("a snapshot key should exist");
    let snapshot = storage
        .retrieve(snapshot_key)
        .await
        .expect("retrieve")
        .expect("snapshot entry present");
    assert_eq!(snapshot.value, "content worth snapshotting");
}
