//! Index manager for automatic index maintenance and lifecycle management.
//!
//! This module provides hooks for keeping vector indexes synchronized with storage operations.

use super::vector::{IndexError, IndexResult, VectorIndex};
use crate::memory::embeddings::EmbeddingProvider;
use crate::memory::types::MemoryEntry;
use parking_lot::RwLock;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Manager for vector index lifecycle and synchronization
///
/// Provides hooks for automatic index updates when memories are:
/// - Added (index insertion)
/// - Updated (re-index with new content)
/// - Deleted (removal from index)
/// - Batch imported (bulk rebuild)
pub struct IndexManager {
    /// The vector index being managed
    index: Arc<RwLock<Box<dyn VectorIndex>>>,
    /// Embedding provider for generating vectors
    provider: Arc<dyn EmbeddingProvider>,
    /// Whether to auto-rebuild on certain thresholds
    auto_rebuild: bool,
    /// Number of deletions before triggering rebuild
    deletion_threshold: usize,
    /// Current deletion count since last rebuild
    deletion_count: Arc<RwLock<usize>>,
}

impl IndexManager {
    /// Create a new index manager
    ///
    /// # Arguments
    /// * `index` - The vector index to manage
    /// * `provider` - Embedding provider for generating vectors
    pub fn new(
        index: Box<dyn VectorIndex>,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            provider,
            auto_rebuild: true,
            deletion_threshold: 1000, // Rebuild after 1000 deletions
            deletion_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Configure auto-rebuild behavior
    pub fn with_auto_rebuild(mut self, enabled: bool) -> Self {
        self.auto_rebuild = enabled;
        self
    }

    /// Configure deletion threshold for auto-rebuild
    pub fn with_deletion_threshold(mut self, threshold: usize) -> Self {
        self.deletion_threshold = threshold;
        self
    }

    /// Get a reference to the managed index
    pub fn index(&self) -> Arc<RwLock<Box<dyn VectorIndex>>> {
        Arc::clone(&self.index)
    }

    /// Hook: Called when a memory is added
    ///
    /// Generates embedding and adds vector to index
    #[instrument(skip(self, entry), fields(key = %entry.key))]
    pub async fn on_memory_added(&self, entry: &MemoryEntry) -> IndexResult<()> {
        debug!("Index hook: memory added");

        // Generate embedding
        let embedding = self
            .provider
            .embed(&entry.value, None)
            .await
            .map_err(|e| IndexError::BuildFailed(format!("Embedding generation failed: {}", e)))?;

        // Add to index
        let mut index = self.index.write();
        index.add(embedding.vector(), entry.clone()).await?;
        drop(index);

        debug!("Successfully added memory to index");
        Ok(())
    }

    /// Hook: Called when a memory is updated
    ///
    /// Removes old vector and adds new one (re-indexing)
    #[instrument(skip(self, old_entry, new_entry), fields(key = %new_entry.key))]
    pub async fn on_memory_updated(
        &self,
        old_entry: &MemoryEntry,
        new_entry: &MemoryEntry,
    ) -> IndexResult<()> {
        debug!("Index hook: memory updated");

        // Remove old entry
        let mut index = self.index.write();
        index.remove(&old_entry.key).await?;
        drop(index);

        // Add new entry
        self.on_memory_added(new_entry).await?;

        debug!("Successfully updated memory in index");
        Ok(())
    }

    /// Hook: Called when a memory is deleted
    ///
    /// Removes vector from index and tracks deletion count
    #[instrument(skip(self, entry), fields(key = %entry.key))]
    pub async fn on_memory_deleted(&self, entry: &MemoryEntry) -> IndexResult<()> {
        debug!("Index hook: memory deleted");

        // Remove from index
        let mut index = self.index.write();
        let removed = index.remove(&entry.key).await?;
        drop(index);

        if removed {
            // Increment deletion counter
            let mut count = self.deletion_count.write();
            *count += 1;
            let current_count = *count;
            drop(count);

            debug!(
                deletions = current_count,
                threshold = self.deletion_threshold,
                "Deletion count updated"
            );

            // Check if rebuild is needed
            if self.auto_rebuild && current_count >= self.deletion_threshold {
                info!(
                    "Deletion threshold reached ({}), triggering index rebuild",
                    self.deletion_threshold
                );
                self.rebuild().await?;
            }
        }

        Ok(())
    }

    /// Hook: Called for batch memory addition
    ///
    /// Efficiently adds multiple memories at once
    #[instrument(skip(self, entries), fields(count = entries.len()))]
    pub async fn on_batch_added(&self, entries: &[MemoryEntry]) -> IndexResult<()> {
        info!("Index hook: batch addition of {} memories", entries.len());

        // Generate embeddings for all entries
        let mut vectors = Vec::new();
        for entry in entries {
            let embedding = self
                .provider
                .embed(&entry.value, None)
                .await
                .map_err(|e| IndexError::BuildFailed(format!("Embedding generation failed: {}", e)))?;
            vectors.push(embedding.vector().to_vec());
        }

        // Batch add to index
        let mut index = self.index.write();
        index.add_batch(&vectors, entries.to_vec()).await?;
        drop(index);

        info!("Successfully added {} memories to index", entries.len());
        Ok(())
    }

    /// Manually trigger index rebuild
    ///
    /// Rebuilds the index structure for optimization (removes fragmentation)
    #[instrument(skip(self))]
    pub async fn rebuild(&self) -> IndexResult<()> {
        info!("Rebuilding index");

        let mut index = self.index.write();
        index.rebuild().await?;
        drop(index);

        // Reset deletion counter
        *self.deletion_count.write() = 0;

        info!("Index rebuild completed");
        Ok(())
    }

    /// Save the index to disk
    #[instrument(skip(self, path))]
    pub async fn save<P: AsRef<std::path::Path> + Send>(&self, path: P) -> IndexResult<()> {
        info!("Saving index to disk");

        let index = self.index.read();
        index.save(path).await?;
        drop(index);

        info!("Index saved successfully");
        Ok(())
    }

    /// Load the index from disk
    #[instrument(skip(self, path))]
    pub async fn load<P: AsRef<std::path::Path> + Send>(&self, path: P) -> IndexResult<()> {
        info!("Loading index from disk");

        let mut index = self.index.write();
        index.load(path).await?;
        drop(index);

        // Reset deletion counter
        *self.deletion_count.write() = 0;

        info!("Index loaded successfully");
        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> super::vector::IndexStats {
        let index = self.index.read();
        index.stats()
    }

    /// Get current deletion count
    pub fn deletion_count(&self) -> usize {
        *self.deletion_count.read()
    }

    /// Clear the index
    #[instrument(skip(self))]
    pub async fn clear(&self) -> IndexResult<()> {
        info!("Clearing index");

        let mut index = self.index.write();
        index.clear().await?;
        drop(index);

        *self.deletion_count.write() = 0;

        info!("Index cleared");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::TfIdfProvider;
    use crate::memory::indexing::{HnswIndex, IndexConfig};
    use crate::memory::types::{MemoryType, Metadata};
    use chrono::Utc;

    fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            key: key.to_string(),
            content: content.to_string(),
            memory_type: MemoryType::ShortTerm,
            metadata: Metadata::default(),
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 0,
        }
    }

    #[tokio::test]
    async fn test_index_manager_creation() {
        let config = IndexConfig::new(128);
        let index = Box::new(HnswIndex::new(config));
        let provider = Arc::new(TfIdfProvider::default());
        let manager = IndexManager::new(index, provider);

        assert_eq!(manager.deletion_count(), 0);
    }

    #[tokio::test]
    async fn test_memory_added_hook() {
        let config = IndexConfig::new(128);
        let index = Box::new(HnswIndex::new(config));
        let provider = Arc::new(TfIdfProvider::default());
        let manager = IndexManager::new(index, provider);

        let entry = create_test_entry("key1", "test content");
        manager.on_memory_added(&entry).await.expect("Failed to add");

        let stats = manager.stats();
        assert_eq!(stats.num_vectors, 1);
    }

    #[tokio::test]
    async fn test_memory_deleted_hook() {
        let config = IndexConfig::new(128);
        let index = Box::new(HnswIndex::new(config));
        let provider = Arc::new(TfIdfProvider::default());
        let manager = IndexManager::new(index, provider).with_auto_rebuild(false);

        // Add a memory
        let entry = create_test_entry("key1", "test content");
        manager.on_memory_added(&entry).await.expect("Failed to add");

        assert_eq!(manager.stats().num_vectors, 1);

        // Delete it
        manager
            .on_memory_deleted(&entry)
            .await
            .expect("Failed to delete");

        assert_eq!(manager.deletion_count(), 1);
    }

    #[tokio::test]
    async fn test_batch_addition() {
        let config = IndexConfig::new(128);
        let index = Box::new(HnswIndex::new(config));
        let provider = Arc::new(TfIdfProvider::default());
        let manager = IndexManager::new(index, provider);

        let entries = vec![
            create_test_entry("key1", "content 1"),
            create_test_entry("key2", "content 2"),
            create_test_entry("key3", "content 3"),
        ];

        manager
            .on_batch_added(&entries)
            .await
            .expect("Failed to batch add");

        let stats = manager.stats();
        assert_eq!(stats.num_vectors, 3);
    }

    #[tokio::test]
    async fn test_auto_rebuild_threshold() {
        let config = IndexConfig::new(128);
        let index = Box::new(HnswIndex::new(config));
        let provider = Arc::new(TfIdfProvider::default());
        let manager = IndexManager::new(index, provider)
            .with_auto_rebuild(true)
            .with_deletion_threshold(2);

        // Add and delete to trigger rebuild
        for i in 0..3 {
            let entry = create_test_entry(&format!("key{}", i), &format!("content {}", i));
            manager.on_memory_added(&entry).await.expect("Failed to add");
            manager
                .on_memory_deleted(&entry)
                .await
                .expect("Failed to delete");
        }

        // After threshold, deletion count should be reset
        assert_eq!(manager.deletion_count(), 1); // 1 deletion after last rebuild
    }
}
