#![cfg(feature = "test-utils")]

use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::FlakyStorage;
use synaptic::memory::Storage;
use synaptic::{AgentMemory, MemoryConfig};

/// Happy path: restoring a checkpoint must remove keys written after the
/// checkpoint was taken and restore the checkpointed content, without
/// wiping and repopulating storage destructively.
#[tokio::test]
async fn restore_checkpoint_prunes_post_checkpoint_keys_and_restores_content() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("agent memory should initialize");

    memory.store("k1", "v1").await.expect("store k1");
    memory.store("k2", "v2").await.expect("store k2");
    memory.store("k3", "v3").await.expect("store k3");

    let checkpoint_id = memory.checkpoint().await.expect("checkpoint succeeds");

    memory.store("k4", "v4").await.expect("store k4");
    memory.store("k5", "v5").await.expect("store k5");

    memory
        .restore_checkpoint(checkpoint_id)
        .await
        .expect("restore succeeds");

    assert!(memory.retrieve("k1").await.expect("ok").is_some());
    assert!(memory.retrieve("k2").await.expect("ok").is_some());
    assert!(memory.retrieve("k3").await.expect("ok").is_some());
    assert!(
        memory.retrieve("k4").await.expect("ok").is_none(),
        "post-checkpoint key k4 must be pruned by restore"
    );
    assert!(
        memory.retrieve("k5").await.expect("ok").is_none(),
        "post-checkpoint key k5 must be pruned by restore"
    );
}

/// If a storage write fails partway through the restore's upsert phase, the
/// restore must return an error and must NOT have destroyed any data that
/// was durably persisted before the failure: the original keys must still
/// be retrievable afterward.
#[tokio::test]
async fn restore_checkpoint_failure_mid_restore_preserves_original_data() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("agent memory should initialize");

    memory.store("k1", "v1").await.expect("store k1");
    memory.store("k2", "v2").await.expect("store k2");
    memory.store("k3", "v3").await.expect("store k3");

    let checkpoint_id = memory.checkpoint().await.expect("checkpoint succeeds");

    memory.store("k4", "v4").await.expect("store k4");
    memory.store("k5", "v5").await.expect("store k5");

    // Seed a flaky storage backend with the current (post-checkpoint)
    // contents so reads still work, but only allow one more successful
    // `store` call before every subsequent write fails. The checkpoint
    // being restored has 3 entries, so this guarantees the restore's
    // upsert phase fails partway through.
    let mut seeded = MemoryStorage::new();
    for key in ["k1", "k2", "k3", "k4", "k5"] {
        let entry = memory
            .retrieve(key)
            .await
            .expect("retrieve ok")
            .expect("entry present");
        seeded.store(&entry).await.expect("seed store");
    }
    memory.set_storage_for_test(std::sync::Arc::new(FlakyStorage::new(seeded, 1)));

    let result = memory.restore_checkpoint(checkpoint_id).await;
    assert!(
        result.is_err(),
        "restore should fail when a mid-restore storage write fails"
    );

    // Original (pre-restore) data must still be intact and retrievable.
    for key in ["k1", "k2", "k3", "k4", "k5"] {
        assert!(
            memory.retrieve(key).await.expect("retrieve ok").is_some(),
            "key {key} should still be present after a failed restore"
        );
    }
}
