//! Candidate-widening storage wrapper for the hybrid retrieval pipeline
//!
//! `MemoryStorage::search` (the default storage backend) requires the
//! *entire* query string to appear verbatim in an entry's content before it
//! is even considered a candidate (see `memory::storage::memory::MemoryStorage::search_entries`).
//! That is fine as a cheap fallback, but it starves the retrieval pipeline's
//! ranking stages (`DenseVectorRetriever`, `KeywordRetriever`, ...) of
//! candidates for any multi-word query where the exact phrase is not present
//! verbatim in a relevant document — exactly the case ranking is supposed to
//! fix.
//!
//! This wrapper is used only when constructing the retrieval pipeline; it
//! leaves the storage backend's own `search` (used as the substring fallback
//! and by any other direct caller) untouched. It widens candidate retrieval
//! to an any-term match over the full entry set, and hands the actual
//! relevance scoring to the pipeline's ranking stages.

use crate::error::Result;
use crate::memory::storage::{Storage, StorageStats};
use crate::memory::types::{MemoryEntry, MemoryFragment};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

pub struct CandidateWideningStorage {
    inner: Arc<dyn Storage + Send + Sync>,
}

impl CandidateWideningStorage {
    pub fn new(inner: Arc<dyn Storage + Send + Sync>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Storage for CandidateWideningStorage {
    async fn store(&self, entry: &MemoryEntry) -> Result<()> {
        self.inner.store(entry).await
    }

    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>> {
        self.inner.retrieve(key).await
    }

    /// Any-term-match candidate widening: an entry is a candidate if its
    /// content contains at least one whitespace-delimited query term.
    /// Relevance scoring/ranking is delegated to the pipeline's own signal
    /// retrievers, not performed here.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryFragment>> {
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();
        if terms.is_empty() {
            return self.inner.search(query, limit).await;
        }

        let mut candidates: HashMap<String, MemoryFragment> = HashMap::new();
        for entry in self.inner.get_all_entries().await? {
            let content_lower = entry.value.to_lowercase();
            if terms.iter().any(|term| content_lower.contains(term)) {
                candidates
                    .entry(entry.key.clone())
                    .or_insert_with(|| MemoryFragment::new(entry, 0.0));
            }
            if candidates.len() >= limit {
                break;
            }
        }

        Ok(candidates.into_values().collect())
    }

    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()> {
        self.inner.update(key, entry).await
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        self.inner.delete(key).await
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        self.inner.list_keys().await
    }

    async fn count(&self) -> Result<usize> {
        self.inner.count().await
    }

    async fn clear(&self) -> Result<()> {
        self.inner.clear().await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.inner.exists(key).await
    }

    async fn stats(&self) -> Result<StorageStats> {
        self.inner.stats().await
    }

    async fn maintenance(&self) -> Result<()> {
        self.inner.maintenance().await
    }

    async fn backup(&self, path: &str) -> Result<()> {
        self.inner.backup(path).await
    }

    async fn restore(&self, path: &str) -> Result<()> {
        self.inner.restore(path).await
    }

    async fn get_all_entries(&self) -> Result<Vec<MemoryEntry>> {
        self.inner.get_all_entries().await
    }
}
