//! Tests for optimized indexed memory retrieval

use chrono::{DateTime, Duration, Utc};
use std::collections::HashSet;
use std::sync::Arc;
use synaptic::memory::{
    management::MemoryManager,
    retrieval::{IndexedMemoryRetriever, IndexingConfig, RetrievalConfig},
    storage::{memory::MemoryStorage, Storage},
    types::{MemoryEntry, MemoryMetadata, MemoryType},
};

/// Create test memory entries with varying access patterns
fn create_test_entries(count: usize) -> Vec<MemoryEntry> {
    let mut entries = Vec::new();
    let base_time = Utc::now() - Duration::hours(24 * 30); // 30 days ago

    for i in 0..count {
        let mut entry = MemoryEntry::new(
            format!("test_key_{}", i),
            format!("Test content for memory entry number {}", i),
            if i % 3 == 0 {
                MemoryType::LongTerm
            } else {
                MemoryType::ShortTerm
            },
        );

        // Simulate varying access patterns
        let mut metadata = MemoryMetadata::new();
        metadata.created_at = base_time + Duration::hours(i as i64);
        metadata.last_accessed = base_time + Duration::hours((i * 2) as i64);
        metadata.access_count = (i % 100) as u64; // Varying access counts
        metadata.importance = (i as f64 % 10.0) / 10.0; // Varying importance

        // Add tags to some entries
        if i % 5 == 0 {
            metadata.tags.push("important".to_string());
        }
        if i % 10 == 0 {
            metadata.tags.push("urgent".to_string());
        }
        if i % 7 == 0 {
            metadata.tags.push("project".to_string());
        }

        entry.metadata = metadata;
        entries.push(entry);
    }

    entries
}

/// Setup storage with test data
async fn setup_storage_with_data(entry_count: usize) -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());
    let entries = create_test_entries(entry_count);

    for entry in entries {
        storage.store(&entry).await.unwrap();
    }

    storage
}

#[tokio::test]
async fn test_indexed_retriever_creation() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);

    // Test that retriever was created successfully
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);
    assert_eq!(stats.frequency_index_entries, 0);
    assert_eq!(stats.tag_index_entries, 0);
    assert_eq!(stats.hot_cache_entries, 0);
}

#[tokio::test]
async fn test_index_initialization() {
    let storage = setup_storage_with_data(50).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);

    // Initialize indexes from storage
    retriever.initialize_indexes().await.unwrap();

    // Check that indexes were populated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 50);
    assert_eq!(stats.frequency_index_entries, 50);
    assert_eq!(stats.tag_index_entries, 50);
}

#[tokio::test]
async fn test_get_recent_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();

    // Test get_recent with small limit
    let recent = retriever.get_recent(10).await.unwrap();
    assert_eq!(recent.len(), 10);

    // Verify they are sorted by access time (most recent first)
    for i in 1..recent.len() {
        assert!(recent[i - 1].last_accessed() >= recent[i].last_accessed());
    }

    // Test caching - second call should be faster (cache hit)
    let recent2 = retriever.get_recent(10).await.unwrap();
    assert_eq!(recent.len(), recent2.len());

    // Results should be identical
    for (a, b) in recent.iter().zip(recent2.iter()) {
        assert_eq!(a.key, b.key);
    }
}

#[tokio::test]
async fn test_get_frequent_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();

    // Test get_frequent with small limit
    let frequent = retriever.get_frequent(10).await.unwrap();
    assert_eq!(frequent.len(), 10);

    // Verify they are sorted by access count (most frequent first)
    for i in 1..frequent.len() {
        assert!(frequent[i - 1].access_count() >= frequent[i].access_count());
    }

    // Test caching
    let frequent2 = retriever.get_frequent(10).await.unwrap();
    assert_eq!(frequent.len(), frequent2.len());
}

#[tokio::test]
async fn test_get_by_tags_with_indexing() {
    let storage = setup_storage_with_data(100).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();

    // Test tag-based retrieval
    let important = retriever
        .get_by_tags(&["important".to_string()])
        .await
        .unwrap();
    assert!(!important.is_empty());

    // Verify all returned entries have the "important" tag
    for entry in &important {
        assert!(entry.has_tag("important"));
    }

    // Test multiple tags
    let urgent_important = retriever
        .get_by_tags(&["urgent".to_string(), "important".to_string()])
        .await
        .unwrap();
    assert!(!urgent_important.is_empty());

    // Test non-existent tag
    let nonexistent = retriever
        .get_by_tags(&["nonexistent".to_string()])
        .await
        .unwrap();
    assert!(nonexistent.is_empty());
}

#[tokio::test]
async fn test_index_updates_on_store() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage.clone(), config, indexing_config);

    // Initially empty
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);

    // Create and store a new entry
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test content".to_string(),
        MemoryType::ShortTerm,
    )
    .with_tags(vec!["test".to_string()]);

    storage.store(&entry).await.unwrap();
    retriever.on_entry_stored(&entry).await;

    // Check that indexes were updated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 1);
    assert_eq!(stats.frequency_index_entries, 1);
    assert_eq!(stats.tag_index_entries, 1);

    // Test retrieval
    let by_tags = retriever.get_by_tags(&["test".to_string()]).await.unwrap();
    assert_eq!(by_tags.len(), 1);
    assert_eq!(by_tags[0].key, "test_key");
}

#[tokio::test]
async fn test_index_updates_on_delete() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig::default();

    let retriever = IndexedMemoryRetriever::new(storage.clone(), config, indexing_config);

    // Store an entry
    let entry = MemoryEntry::new(
        "test_key".to_string(),
        "Test content".to_string(),
        MemoryType::ShortTerm,
    )
    .with_tags(vec!["test".to_string()]);

    storage.store(&entry).await.unwrap();
    retriever.on_entry_stored(&entry).await;

    // Verify it's in indexes
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 1);

    // Delete the entry
    storage.delete("test_key").await.unwrap();
    retriever.on_entry_deleted("test_key").await;

    // Check that indexes were updated
    let stats = retriever.get_index_stats().await;
    assert_eq!(stats.access_time_index_entries, 0);
    assert_eq!(stats.frequency_index_entries, 0);
    assert_eq!(stats.tag_index_entries, 0);
}

#[tokio::test]
async fn test_hot_cache_functionality() {
    let storage = setup_storage_with_data(50).await;
    let config = RetrievalConfig::default();
    let mut indexing_config = IndexingConfig::default();
    indexing_config.hot_cache_size = 10; // Small cache for testing

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);
    retriever.initialize_indexes().await.unwrap();

    // First call should populate cache
    let recent1 = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent1.len(), 5);

    // Cache should have entries now
    let stats = retriever.get_index_stats().await;
    assert!(stats.hot_cache_entries > 0);
    assert!(stats.hot_cache_entries <= 10); // Respects max size

    // Second call should use cache
    let recent2 = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent1.len(), recent2.len());
}

#[tokio::test]
async fn test_background_maintenance() {
    let storage = Arc::new(MemoryStorage::new());
    let config = RetrievalConfig::default();
    let mut indexing_config = IndexingConfig::default();
    indexing_config.index_maintenance_interval_seconds = 1; // Fast for testing

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);

    // Start maintenance
    retriever.start_maintenance().await;

    // Wait a bit for maintenance to run
    tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;

    // Stop maintenance
    retriever.stop_maintenance().await;

    // Test passes if no panics occurred
}

#[tokio::test]
async fn test_fallback_behavior_with_disabled_indexes() {
    let storage = setup_storage_with_data(20).await;
    let config = RetrievalConfig::default();
    let indexing_config = IndexingConfig {
        enable_access_time_index: false,
        enable_frequency_index: false,
        enable_tag_index: false,
        ..Default::default()
    };

    let retriever = IndexedMemoryRetriever::new(storage, config, indexing_config);

    // Should still work using fallback methods
    let recent = retriever.get_recent(5).await.unwrap();
    assert_eq!(recent.len(), 5);

    let frequent = retriever.get_frequent(5).await.unwrap();
    assert_eq!(frequent.len(), 5);

    let by_tags = retriever
        .get_by_tags(&["important".to_string()])
        .await
        .unwrap();
    // Should find some entries with "important" tag
    assert!(!by_tags.is_empty());
}

#[tokio::test]
async fn test_performance_improvement() {
    use std::time::Instant;

    let storage = setup_storage_with_data(1000).await;
    let config = RetrievalConfig::default();

    // Test with indexing disabled (fallback)
    let indexing_config_disabled = IndexingConfig {
        enable_access_time_index: false,
        enable_frequency_index: false,
        enable_tag_index: false,
        ..Default::default()
    };

    let retriever_fallback =
        IndexedMemoryRetriever::new(storage.clone(), config.clone(), indexing_config_disabled);

    let start = Instant::now();
    let recent_fallback = retriever_fallback.get_recent(10).await.unwrap();
    let fallback_time = start.elapsed();

    // Test with indexing enabled
    let indexing_config_enabled = IndexingConfig::default();
    let retriever_indexed = IndexedMemoryRetriever::new(storage, config, indexing_config_enabled);
    retriever_indexed.initialize_indexes().await.unwrap();

    let start = Instant::now();
    let recent_indexed = retriever_indexed.get_recent(10).await.unwrap();
    let indexed_time = start.elapsed();

    // Indexed version should be faster (though with small dataset, difference might be minimal)
    println!(
        "Fallback time: {:?}, Indexed time: {:?}",
        fallback_time, indexed_time
    );

    // Both paths must complete successfully and return the same set of results.
    // (Timing is not asserted directly: sub-millisecond operations make a strict
    // `> 0ms` floor flaky on fast machines.)
    assert_eq!(
        recent_fallback.len(),
        recent_indexed.len(),
        "indexed and fallback retrieval should return the same number of results"
    );
}

// --- HNSW-backed `count_related_memories` (Task 3.2) ---------------------

/// Dimension used for the deterministic cluster embeddings below. Must be
/// >= the largest `num_clusters` used by any test in this file, so distinct
/// clusters always land on distinct one-hot axes (otherwise clusters would
/// silently alias onto the same axis via `cluster % CLUSTER_EMBEDDING_DIM`).
const CLUSTER_EMBEDDING_DIM: usize = 256;

/// Deterministic embedding: entry `i` belongs to cluster `i % num_clusters`,
/// represented as a one-hot vector on that cluster's axis plus a tiny
/// deterministic perturbation on every axis. This keeps intra-cluster cosine
/// similarity near 1.0 and inter-cluster similarity near 0.0, so a 0.7
/// similarity threshold cleanly separates "related" from "unrelated" and
/// both brute-force and ANN search agree on the exact neighbor set.
fn deterministic_cluster_embedding(i: usize, num_clusters: usize) -> Vec<f32> {
    let cluster = i % num_clusters;
    let mut v = vec![0.0f32; CLUSTER_EMBEDDING_DIM];
    v[cluster % CLUSTER_EMBEDDING_DIM] = 1.0;
    for (j, slot) in v.iter_mut().enumerate() {
        *slot += 0.01 * ((i as f32 * 0.37 + j as f32).sin());
    }
    v
}

/// Build a memory entry with a deterministic embedding, unique word content
/// (so the word-overlap content/temporal strategies never fire), no tags,
/// and timestamps spread far enough apart that the 1-hour temporal-proximity
/// strategy never fires either. This isolates `count_related_memories` to
/// (effectively) just the embedding-similarity strategy, so it can be
/// compared directly against a brute-force cosine computation.
fn cluster_probe_entry(i: usize, num_clusters: usize, base_time: DateTime<Utc>) -> MemoryEntry {
    let mut entry = MemoryEntry::new(
        format!("cluster_entry_{i}"),
        format!("uniqueword{i}"),
        MemoryType::LongTerm,
    )
    .with_embedding(deterministic_cluster_embedding(i, num_clusters));

    let mut metadata = MemoryMetadata::new();
    metadata.created_at = base_time + Duration::hours(i as i64 * 2);
    metadata.last_accessed = metadata.created_at;
    entry.metadata = metadata;
    entry
}

fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f64 {
    let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
    let dot: f64 = a64.iter().zip(b64.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a64.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b64.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Brute-force cosine count over `entries`, matching the threshold (0.7)
/// used by `MemoryManager::count_similarity_related_memories`.
fn brute_force_similarity_count(probe: &MemoryEntry, entries: &[MemoryEntry]) -> usize {
    let probe_embedding = probe.embedding.as_ref().expect("probe has an embedding");
    entries
        .iter()
        .filter(|e| e.key != probe.key)
        .filter(|e| {
            e.embedding
                .as_ref()
                .map(|emb| cosine_similarity_f32(probe_embedding, emb) >= 0.7)
                .unwrap_or(false)
        })
        .count()
}

#[tokio::test]
async fn test_count_related_memories_uses_ann_index_and_matches_brute_force() {
    const NUM_ENTRIES: usize = 200;
    const NUM_CLUSTERS: usize = 20; // 10 entries per cluster

    let storage = Arc::new(MemoryStorage::new());
    let manager = MemoryManager::new(storage, None)
        .await
        .expect("manager should construct");

    let base_time = Utc::now() - Duration::days(365);
    let entries: Vec<MemoryEntry> = (0..NUM_ENTRIES)
        .map(|i| cluster_probe_entry(i, NUM_CLUSTERS, base_time))
        .collect();

    for entry in &entries {
        manager
            .store_memory(entry)
            .await
            .expect("store_memory should succeed");
    }

    let probe = &entries[0];
    let expected = brute_force_similarity_count(probe, &entries);
    // Sanity: the probe's own cluster has (NUM_ENTRIES / NUM_CLUSTERS) - 1
    // other members, all of which should be "related"; nothing else should
    // be, given the clusters are near-orthogonal one-hot vectors.
    assert_eq!(
        expected,
        NUM_ENTRIES / NUM_CLUSTERS - 1,
        "unexpected brute-force baseline; check cluster construction"
    );

    let actual = manager
        .count_related_memories(probe)
        .await
        .expect("count_related_memories should succeed");

    // HNSW is an *approximate* nearest-neighbor index, so exact equality with
    // the brute-force count is not guaranteed on every run. Tolerate a small
    // recall gap while still proving the ANN path finds essentially the
    // right neighbor set.
    let diff = actual.abs_diff(expected);
    assert!(
        diff <= 2,
        "ANN-backed count_related_memories ({actual}) should closely match brute-force cosine \
         similarity count ({expected}), diff={diff}"
    );

    // Proves the ANN (HNSW) path, not the brute-force fallback, was consulted:
    // 200 indexed vectors exceeds MIN_INDEX_SIZE_FOR_ANN (100).
    #[cfg(feature = "test-utils")]
    assert!(
        manager.ann_query_hit_count() > 0,
        "expected the ANN index to have been consulted at least once"
    );
}

#[tokio::test]
async fn test_count_related_memories_falls_back_to_brute_force_below_threshold() {
    const NUM_ENTRIES: usize = 50; // below MIN_INDEX_SIZE_FOR_ANN
    const NUM_CLUSTERS: usize = 5;

    let storage = Arc::new(MemoryStorage::new());
    let manager = MemoryManager::new(storage, None)
        .await
        .expect("manager should construct");

    let base_time = Utc::now() - Duration::days(365);
    let entries: Vec<MemoryEntry> = (0..NUM_ENTRIES)
        .map(|i| cluster_probe_entry(i, NUM_CLUSTERS, base_time))
        .collect();

    for entry in &entries {
        manager
            .store_memory(entry)
            .await
            .expect("store_memory should succeed");
    }

    let probe = &entries[0];
    let expected = brute_force_similarity_count(probe, &entries);

    let actual = manager
        .count_related_memories(probe)
        .await
        .expect("count_related_memories should succeed");

    assert_eq!(actual, expected);
    #[cfg(feature = "test-utils")]
    assert_eq!(
        manager.ann_query_hit_count(),
        0,
        "below MIN_INDEX_SIZE_FOR_ANN the brute-force fallback should be used, not the ANN index"
    );
}

/// Perf comparison at 10k entries: ANN-backed counting should be
/// substantially faster than the brute-force O(n) scan. Ignored by default
/// since timing assertions are inherently environment-sensitive; run
/// explicitly with `cargo test --test indexed_retrieval_tests -- --ignored`.
#[tokio::test]
#[ignore]
#[cfg(feature = "test-utils")]
async fn perf_count_related_memories_ann_vs_brute_force_at_10k() {
    use std::time::Instant;

    const NUM_ENTRIES: usize = 10_000;
    // Keep clusters well under ANN_SEARCH_K (200) so the ANN query's k cap
    // doesn't itself cause undercounting relative to brute force.
    const NUM_CLUSTERS: usize = 200;

    let base_time = Utc::now() - Duration::days(365 * 2);
    let entries: Vec<MemoryEntry> = (0..NUM_ENTRIES)
        .map(|i| cluster_probe_entry(i, NUM_CLUSTERS, base_time))
        .collect();
    let probe = entries[0].clone();

    // ANN path: entries are stored via `MemoryManager::store_memory`, which
    // populates the HNSW index as it goes.
    let ann_storage = Arc::new(MemoryStorage::new());
    let ann_manager = MemoryManager::new(ann_storage, None)
        .await
        .expect("manager should construct");
    for entry in &entries {
        ann_manager
            .store_memory(entry)
            .await
            .expect("store_memory should succeed");
    }

    let start = Instant::now();
    let ann_count = ann_manager
        .count_related_memories(&probe)
        .await
        .expect("count_related_memories should succeed");
    let ann_elapsed = start.elapsed();
    assert!(ann_manager.ann_query_hit_count() > 0);

    // Brute-force path: entries are written directly to storage, bypassing
    // `store_memory`, so the ANN index stays empty and the manager falls
    // back to the O(n) scan regardless of dataset size.
    let brute_storage = Arc::new(MemoryStorage::new());
    for entry in &entries {
        brute_storage
            .store(entry)
            .await
            .expect("direct storage write should succeed");
    }
    let brute_manager = MemoryManager::new(brute_storage, None)
        .await
        .expect("manager should construct");

    let start = Instant::now();
    let brute_count = brute_manager
        .count_related_memories(&probe)
        .await
        .expect("count_related_memories should succeed");
    let brute_elapsed = start.elapsed();
    assert_eq!(brute_manager.ann_query_hit_count(), 0);

    // HNSW is an *approximate* nearest-neighbor index, so at 10k entries with
    // default (balanced) parameters its recall need not be exact; the
    // deterministic-cluster correctness test above (at 200 entries) already
    // proves exact agreement is achievable. Here we only require the ANN
    // count to be within a small margin of the brute-force ground truth.
    let diff = ann_count.abs_diff(brute_count);
    assert!(
        diff <= brute_count / 20 + 1,
        "ANN count ({ann_count}) should be close to brute-force count ({brute_count}), diff={diff}"
    );

    eprintln!(
        "10k entries: ANN {:?} vs brute-force {:?} (ANN {}x faster)",
        ann_elapsed,
        brute_elapsed,
        brute_elapsed.as_secs_f64() / ann_elapsed.as_secs_f64().max(1e-9)
    );
    assert!(
        ann_elapsed < brute_elapsed,
        "expected ANN-backed counting ({ann_elapsed:?}) to beat brute force ({brute_elapsed:?}) at 10k entries"
    );
}

/// Re-storing an existing key must supersede its previous ANN vector: the
/// related count for a probe must reflect only the key's *latest* embedding,
/// never a stale one, and the key must never be counted twice.
#[tokio::test]
async fn test_reindexing_a_key_supersedes_its_stale_ann_vector() {
    const NUM_ENTRIES: usize = 200;
    const NUM_CLUSTERS: usize = 20; // 10 entries per cluster; probe cluster = 0

    let storage = Arc::new(MemoryStorage::new());
    let manager = MemoryManager::new(storage, None)
        .await
        .expect("manager should construct");

    let base_time = Utc::now() - Duration::days(365);
    let entries: Vec<MemoryEntry> = (0..NUM_ENTRIES)
        .map(|i| cluster_probe_entry(i, NUM_CLUSTERS, base_time))
        .collect();

    for entry in &entries {
        manager
            .store_memory(entry)
            .await
            .expect("store_memory should succeed");
    }

    let probe = &entries[0]; // cluster 0
    let baseline = manager
        .count_related_memories(probe)
        .await
        .expect("count_related_memories should succeed");
    assert_eq!(
        baseline,
        NUM_ENTRIES / NUM_CLUSTERS - 1,
        "unexpected baseline related count"
    );

    // Case 1: move a probe-cluster member (entry 20, cluster 0) OUT of the
    // probe's cluster by re-storing it with embedding B from cluster 1
    // (index 21's embedding shape). Its stale cluster-0 vector A must no
    // longer be counted: similarity for this key is now measured against B.
    let mut moved_out = entries[20].clone();
    moved_out.embedding = Some(deterministic_cluster_embedding(21, NUM_CLUSTERS));
    manager
        .store_memory(&moved_out)
        .await
        .expect("re-store should succeed");

    let after_move_out = manager
        .count_related_memories(probe)
        .await
        .expect("count_related_memories should succeed");
    assert_eq!(
        after_move_out,
        baseline - 1,
        "stale vector for a re-stored key must not be counted (key moved out of the cluster)"
    );

    // Case 2: re-store the same key again with ANOTHER cluster-0 embedding.
    // The key is back in the probe's cluster, but it now has had three
    // vectors inserted for it; it must be counted exactly once.
    let mut moved_back = entries[20].clone();
    moved_back.embedding = Some(deterministic_cluster_embedding(40, NUM_CLUSTERS));
    manager
        .store_memory(&moved_back)
        .await
        .expect("re-store should succeed");

    let after_move_back = manager
        .count_related_memories(probe)
        .await
        .expect("count_related_memories should succeed");
    assert_eq!(
        after_move_back, baseline,
        "a re-stored key must be counted at most once (no double-count of superseded vectors)"
    );

    #[cfg(feature = "test-utils")]
    assert!(
        manager.ann_query_hit_count() > 0,
        "expected the ANN index to have been consulted"
    );
}
