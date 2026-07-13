//! Write-path scaling tests (Task P1, agent-memory-v2 follow-ups).
//!
//! These assert operation-count proxies rather than wall-clock time so they
//! stay deterministic: storage stats must not rescan all entries per store
//! (incremental size counter), and `neighbor_facts` must examine a bounded
//! candidate set regardless of store size.

// Test code: panic on unexpected state is the intended behaviour.
#![allow(clippy::panic)]

use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::{BatchStorage, Storage, TransactionalStorage};
use synaptic::memory::types::{MemoryEntry, MemoryType};

fn entry(key: &str, value: &str) -> MemoryEntry {
    MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::LongTerm)
}

/// Recompute the expected total size from the entries the test itself
/// tracks — independent of any storage-internal full scan.
fn expected_size(entries: &[MemoryEntry]) -> usize {
    entries.iter().map(|e| e.estimated_size()).sum()
}

#[tokio::test]
async fn total_size_is_correct_after_inserts_updates_and_deletes() {
    let storage = MemoryStorage::new();
    let mut live: Vec<MemoryEntry> = Vec::new();

    // Inserts.
    for i in 0..50 {
        let e = entry(&format!("k{i}"), &format!("value number {i} padding"));
        storage.store(&e).await.expect("store");
        live.push(e);
    }
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_entries, 50);
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Update (replaces an existing entry with a longer value).
    let updated = entry("k10", "a much longer replacement value with extra content");
    storage.update("k10", &updated).await.expect("update");
    live[10] = updated;
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Re-store an existing key (insert overwrite path).
    let restored = entry("k11", "short");
    storage.store(&restored).await.expect("re-store");
    live[11] = restored;
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Delete: the incremental counter must be decremented.
    assert!(storage.delete("k20").await.expect("delete"));
    live.retain(|e| e.key != "k20");
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_entries, 49);
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Deleting a missing key must not change anything.
    assert!(!storage.delete("missing").await.expect("delete missing"));
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));
}

#[tokio::test]
async fn total_size_is_correct_after_batch_and_clear() {
    let storage = MemoryStorage::new();
    let mut live: Vec<MemoryEntry> = Vec::new();

    let batch: Vec<MemoryEntry> = (0..30)
        .map(|i| entry(&format!("b{i}"), &format!("batch value {i}")))
        .collect();
    storage.store_batch(&batch).await.expect("store_batch");
    live.extend(batch.iter().cloned());
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Batch overwrite of existing keys must not double-count.
    let overwrite: Vec<MemoryEntry> = (0..10)
        .map(|i| {
            entry(
                &format!("b{i}"),
                &format!("rewritten longer batch value {i}"),
            )
        })
        .collect();
    storage.store_batch(&overwrite).await.expect("overwrite");
    for e in &overwrite {
        let slot = live
            .iter_mut()
            .find(|l| l.key == e.key)
            .expect("existing slot");
        *slot = e.clone();
    }
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    let deleted = storage
        .delete_batch(&["b0".to_string(), "b1".to_string(), "nope".to_string()])
        .await
        .expect("delete_batch");
    assert_eq!(deleted, 2);
    live.retain(|e| e.key != "b0" && e.key != "b1");
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    storage.clear().await.expect("clear");
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_entries, 0);
    assert_eq!(stats.total_size_bytes, 0);
}

#[tokio::test]
async fn total_size_is_correct_after_transaction_commit() {
    use synaptic::memory::storage::StorageTransaction;

    let storage = MemoryStorage::new();
    let seed = entry("seed", "seed value that will be deleted");
    storage.store(&seed).await.expect("seed");

    let mut tx = StorageTransaction::new();
    let t1 = entry("t1", "transactional value one");
    let t2 = entry("t2", "transactional value two, a little longer");
    tx.store("t1".to_string(), t1.clone());
    tx.store("t2".to_string(), t2.clone());
    tx.delete("seed".to_string());
    storage.execute_transaction(tx).await.expect("execute tx");

    let live = vec![t1, t2];
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_entries, 2);
    assert_eq!(stats.total_size_bytes, expected_size(&live));

    // Handle-based transaction path must also keep stats coherent.
    let mut handle = storage.begin_transaction().await.expect("begin");
    let t3 = entry("t3", "handle transactional value three");
    handle.store("t3", &t3).await.expect("tx store");
    handle.delete("t1").await.expect("tx delete");
    handle.commit().await.expect("commit");

    let live = vec![live[1].clone(), t3];
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.total_entries, 2);
    assert_eq!(stats.total_size_bytes, expected_size(&live));
}

/// Op-count proxy: computing stats must be O(1) — no per-call rescan of all
/// entries. The storage exposes (behind `test-utils`) a counter of entries
/// examined by stats computation; it must stay 0 no matter how many stores
/// and stats calls happen.
#[cfg(feature = "test-utils")]
#[tokio::test]
async fn stats_computation_never_rescans_entries() {
    let storage = MemoryStorage::new();
    for i in 0..500 {
        let e = entry(&format!("s{i}"), &format!("scaling entry number {i}"));
        storage.store(&e).await.expect("store");
    }
    for _ in 0..5 {
        storage.stats().await.expect("stats");
    }
    assert_eq!(
        storage.stats_entries_scanned(),
        0,
        "stats computation must be incremental (O(1)), not a full rescan"
    );
}

/// Op-count proxy: `neighbor_facts` must examine a bounded candidate set
/// (top-K from the tokenized storage search), not the whole store, no matter
/// how many memories exist.
#[cfg(all(feature = "test-utils", feature = "embeddings"))]
#[tokio::test]
async fn neighbor_facts_examines_bounded_candidates() {
    use synaptic::{AgentMemory, MemoryConfig};

    // Generous upper bound on the configured candidate cap: the keyword hit
    // set (<=21) unioned with a small constant number of per-subject lookups
    // (<=21 each). Independent of store size, which is the point.
    const K: usize = 96;

    let mut memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("create memory");

    // 300 fact-bearing stores: each triggers the reasoner and thus
    // neighbor_facts. With an unbounded scan the examined count grows with
    // store size (~299 by the end); bounded it must stay <= K.
    for i in 0..300 {
        memory
            .store(
                &format!("residence_{i}"),
                &format!("Person{i} lives in City{i}."),
            )
            .await
            .expect("store");
    }

    let examined = memory.max_neighbor_candidates_examined();
    assert!(
        examined > 0,
        "instrumentation must have observed neighbor_facts runs"
    );
    assert!(
        examined <= K,
        "neighbor_facts examined {examined} candidates; must be bounded by {K} regardless of store size"
    );
}
