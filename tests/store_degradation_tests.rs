use synaptic::{AgentMemory, MemoryConfig};

#[tokio::test]
async fn store_with_report_returns_clean_report_on_happy_path() {
    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("default config constructs");
    let report = memory
        .store_with_report("k1", "v1")
        .await
        .expect("storage write succeeds");
    assert!(report.is_clean(), "no subsystem should degrade: {report:?}");
    // and the write is durable:
    let got = memory.retrieve("k1").await.expect("retrieve ok");
    assert!(got.is_some());
}

/// Verifies that `AgentMemory::store` treats storage as the source of truth:
/// if the storage write fails, the in-process state cache must not contain
/// the entry, so a subsequent `retrieve` correctly reports it as absent.
#[cfg(feature = "test-utils")]
#[tokio::test]
async fn store_failure_does_not_pollute_state_cache() {
    use synaptic::memory::storage::FailingStorage;

    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("agent memory should initialize");

    memory.set_storage_for_test(std::sync::Arc::new(FailingStorage::new()));

    let result = memory.store("key1", "value1").await;
    assert!(
        result.is_err(),
        "store should fail when the storage backend fails"
    );

    let retrieved = memory
        .retrieve("key1")
        .await
        .expect("retrieve should not error on a miss");
    assert!(
        retrieved.is_none(),
        "state cache must not contain an entry whose storage write failed"
    );
}
