//! P4: inverted (n-gram) index correctness + scaling tests for
//! `MemoryStorage::search_entries`.
//!
//! Correctness: index-backed search must return EXACTLY the same keys in the
//! same order as a naive full-scan reference implementation.
//! Scaling: search must rank only index candidates, not all n entries.

#![cfg(feature = "test-utils")]
// Test code: panics on failure are the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::Storage;
use synaptic::memory::types::{MemoryEntry, MemoryType};

/// Full-scan reference: replicates the documented ranking contract of
/// `search_entries` (any-term substring match; distinct terms desc, total
/// frequency desc, key asc) independently of the index.
fn reference_search(corpus: &[(String, String)], query: &str, limit: usize) -> Vec<String> {
    let query_lower = query.to_lowercase();
    let terms: Vec<&str> = query_lower.split_whitespace().collect();
    if terms.is_empty() {
        return Vec::new();
    }
    let mut matches: Vec<(usize, usize, String)> = Vec::new();
    for (key, value) in corpus {
        let content = value.to_lowercase();
        let mut distinct = 0usize;
        let mut freq = 0usize;
        for term in &terms {
            let occurrences = content.matches(term).count();
            if occurrences > 0 {
                distinct += 1;
                freq += occurrences;
            }
        }
        if distinct > 0 {
            matches.push((distinct, freq, key.clone()));
        }
    }
    matches.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    matches.truncate(limit);
    matches.into_iter().map(|(_, _, k)| k).collect()
}

async fn seed(storage: &MemoryStorage, corpus: &[(String, String)]) {
    for (key, value) in corpus {
        let entry = MemoryEntry::new(key.clone(), value.clone(), MemoryType::LongTerm);
        storage.store(&entry).await.unwrap();
    }
}

async fn searched_keys(storage: &MemoryStorage, query: &str, limit: usize) -> Vec<String> {
    storage
        .search(query, limit)
        .await
        .unwrap()
        .into_iter()
        .map(|f| f.entry.key)
        .collect()
}

fn corpus() -> Vec<(String, String)> {
    vec![
        ("doc_cat".into(), "the cat sat on the mat".into()),
        ("doc_dog".into(), "a dog chased the cat around".into()),
        ("doc_bird".into(), "birds sing in the morning".into()),
        (
            "doc_concat".into(),
            "we concatenate strings carefully".into(),
        ),
        ("doc_pet".into(), "a cat is a fine pet and companion".into()),
        ("doc_freq".into(), "cat cat cat cat everywhere".into()),
        ("doc_tie_a".into(), "one cat only here".into()),
        ("doc_tie_b".into(), "one cat only here".into()),
        ("doc_upper".into(), "The CAT and The DOG shout".into()),
        ("doc_short".into(), "ox ax".into()),
        ("doc_unrelated".into(), "quantum flux capacitors hum".into()),
        ("doc_empty".into(), "".into()),
    ]
}

#[tokio::test]
async fn index_search_matches_full_scan_reference_exactly() {
    let corpus = corpus();
    let storage = MemoryStorage::new();
    seed(&storage, &corpus).await;

    let queries = [
        "cat",              // single term, several matches incl. substring "concatenate"
        "cat dog",          // multi-term union
        "cat animal pet",   // partial term overlap
        "zebra",            // no match
        "the",              // matches most docs (stopword-like, high fan-out)
        "e",                // 1-char term (sub-trigram)
        "ox",               // 2-char term
        "CAT DOG",          // case-insensitivity
        "cat cat",          // duplicate query terms
        "quantum flux hum", // all terms in one doc
    ];
    for query in queries {
        for limit in [1usize, 3, 100] {
            let expected = reference_search(&corpus, query, limit);
            let actual = searched_keys(&storage, query, limit).await;
            assert_eq!(
                actual, expected,
                "index search diverged from full-scan reference for query {query:?} limit {limit}"
            );
        }
    }
}

#[tokio::test]
async fn index_is_maintained_across_update_and_delete() {
    let mut corpus = corpus();
    let storage = MemoryStorage::new();
    seed(&storage, &corpus).await;

    // Delete: the key must no longer appear for its terms.
    storage.delete("doc_freq").await.unwrap();
    corpus.retain(|(k, _)| k != "doc_freq");
    assert_eq!(
        searched_keys(&storage, "cat", 100).await,
        reference_search(&corpus, "cat", 100)
    );
    assert!(!searched_keys(&storage, "cat", 100)
        .await
        .contains(&"doc_freq".to_string()));

    // Update: old tokens must stop matching, new tokens must match.
    let updated = MemoryEntry::new(
        "doc_bird".to_string(),
        "a stealthy cat stalks".to_string(),
        MemoryType::LongTerm,
    );
    storage.update("doc_bird", &updated).await.unwrap();
    for (k, v) in corpus.iter_mut() {
        if k == "doc_bird" {
            *v = "a stealthy cat stalks".to_string();
        }
    }
    assert_eq!(
        searched_keys(&storage, "cat", 100).await,
        reference_search(&corpus, "cat", 100)
    );
    assert!(searched_keys(&storage, "sing morning", 100)
        .await
        .is_empty());

    // Clear: nothing matches anymore.
    storage.clear().await.unwrap();
    assert!(searched_keys(&storage, "cat", 100).await.is_empty());
}

#[tokio::test]
async fn search_ranks_only_index_candidates_not_all_entries() {
    let storage = MemoryStorage::new();
    // Large store: n entries that share NO grams with the query term.
    let n = 5_000usize;
    for i in 0..n {
        let entry = MemoryEntry::new(
            format!("filler_{i:05}"),
            "www www www www".to_string(),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
    }
    // A handful of matching entries.
    for i in 0..5 {
        let entry = MemoryEntry::new(
            format!("hit_{i}"),
            format!("zebra sighting number {i}"),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await.unwrap();
    }

    let before = storage.entries_ranked();
    let results = storage.search("zebra", 10).await.unwrap();
    let ranked = storage.entries_ranked() - before;

    assert_eq!(results.len(), 5, "all matching entries found");
    assert!(
        ranked <= 5,
        "search must rank only index candidates (got {ranked}, store has {})",
        n + 5
    );
    assert!(ranked < n / 100, "ranking work must not scale with n");
}
