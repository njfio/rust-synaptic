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
