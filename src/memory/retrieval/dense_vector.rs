//! Dense vector retriever using embeddings
//!
//! This retriever uses embedding providers to perform semantic similarity search.

use super::pipeline::{RetrievalPipeline, RetrievalSignal, ScoredMemory, PipelineConfig};
use crate::error::{MemoryError, Result};
use crate::memory::embeddings::{EmbeddingProvider, Embedding, EmbedOptions};
use crate::memory::storage::Storage;
use crate::memory::types::MemoryFragment;
use async_trait::async_trait;
use std::sync::Arc;

/// Dense vector retriever using semantic embeddings
///
/// This retriever generates embeddings for the query and memories,
/// then ranks results by cosine similarity in the embedding space.
pub struct DenseVectorRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    provider: Arc<dyn EmbeddingProvider>,
    similarity_threshold: f64,
}

impl DenseVectorRetriever {
    /// Create a new dense vector retriever
    ///
    /// # Arguments
    /// * `storage` - Storage backend
    /// * `provider` - Embedding provider to use
    pub fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            storage,
            provider,
            similarity_threshold: 0.3, // Default threshold
        }
    }

    /// Set similarity threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Compute similarity score for a fragment against query embedding
    async fn compute_similarity_score(
        &self,
        fragment: &MemoryFragment,
        query_embedding: &Embedding,
    ) -> Result<f64> {
        // Generate embedding for the fragment
        let fragment_embedding = self
            .provider
            .embed(&fragment.content, None)
            .await?;

        // Compute cosine similarity
        let similarity = query_embedding.cosine_similarity(&fragment_embedding);

        Ok(similarity)
    }
}

#[async_trait]
impl RetrievalPipeline for DenseVectorRetriever {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        _config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        tracing::debug!(
            query = %query,
            limit = limit,
            provider = self.provider.name(),
            "DenseVectorRetriever: starting semantic search"
        );

        // Generate embedding for the query
        let query_embedding = self.provider.embed(query, None).await?;

        // Get candidate memories from storage
        // In a production system, this would use an ANN index
        // For now, we retrieve more candidates and filter
        let fragments = self.storage.search(query, limit * 3).await?;

        tracing::debug!(
            candidate_count = fragments.len(),
            "DenseVectorRetriever: retrieved candidates"
        );

        // Score each fragment by semantic similarity
        let mut scored_results = Vec::new();

        for fragment in fragments {
            let similarity = self.compute_similarity_score(&fragment, &query_embedding).await?;

            if similarity >= self.similarity_threshold {
                let scored = ScoredMemory::new(fragment, similarity, RetrievalSignal::DenseVector)
                    .with_explanation(format!(
                        "Semantic similarity: {:.3} (model: {})",
                        similarity,
                        self.provider.model_id()
                    ));

                scored_results.push(scored);
            }
        }

        // Sort by similarity score descending
        scored_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top limit
        scored_results.truncate(limit);

        tracing::info!(
            result_count = scored_results.len(),
            provider = self.provider.name(),
            "DenseVectorRetriever: semantic search completed"
        );

        Ok(scored_results)
    }

    fn name(&self) -> &'static str {
        "DenseVectorRetriever"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::DenseVector
    }

    fn is_available(&self) -> bool {
        self.provider.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::TfIdfProvider;
    use crate::memory::storage::memory::MemoryStorage;
    use crate::memory::types::{MemoryEntry, MemoryType};

    #[tokio::test]
    async fn test_dense_vector_retriever_creation() {
        let storage = Arc::new(MemoryStorage::new());
        let provider = Arc::new(TfIdfProvider::default());
        let retriever = DenseVectorRetriever::new(storage, provider);

        assert_eq!(retriever.name(), "DenseVectorRetriever");
        assert_eq!(retriever.signal_type(), RetrievalSignal::DenseVector);
        assert!(retriever.is_available());
    }

    #[tokio::test]
    async fn test_dense_vector_search() {
        let storage = Arc::new(MemoryStorage::new());
        let provider = Arc::new(TfIdfProvider::default());
        let retriever = DenseVectorRetriever::new(storage.clone(), provider);

        // Store some test memories
        let mem1 = MemoryEntry::new(
            "rust1".to_string(),
            "Rust programming language systems".to_string(),
            MemoryType::ShortTerm,
        );
        let mem2 = MemoryEntry::new(
            "rust2".to_string(),
            "Advanced Rust programming concepts".to_string(),
            MemoryType::ShortTerm,
        );
        let mem3 = MemoryEntry::new(
            "python1".to_string(),
            "Python scripting and data science".to_string(),
            MemoryType::ShortTerm,
        );

        storage.store(&mem1).await.unwrap();
        storage.store(&mem2).await.unwrap();
        storage.store(&mem3).await.unwrap();

        // Search for Rust-related content
        let results = retriever.search("Rust programming", 10, None).await.unwrap();

        // Should find Rust-related memories
        assert!(!results.is_empty());

        // Results should have similarity scores
        for result in &results {
            assert!(result.score >= retriever.similarity_threshold);
            assert_eq!(result.signal, RetrievalSignal::DenseVector);
        }
    }

    #[tokio::test]
    async fn test_similarity_threshold() {
        let storage = Arc::new(MemoryStorage::new());
        let provider = Arc::new(TfIdfProvider::default());
        let retriever = DenseVectorRetriever::new(storage.clone(), provider)
            .with_threshold(0.8); // High threshold

        // Store a memory
        let mem = MemoryEntry::new(
            "test".to_string(),
            "completely different unrelated content about cooking".to_string(),
            MemoryType::ShortTerm,
        );
        storage.store(&mem).await.unwrap();

        // Search with unrelated query
        let results = retriever.search("Rust programming language", 10, None).await.unwrap();

        // High threshold should filter out low-similarity results
        // Results should be empty or very few
        for result in results {
            assert!(result.score >= 0.8);
        }
    }

    #[tokio::test]
    async fn test_semantic_similarity_ranking() {
        let storage = Arc::new(MemoryStorage::new());
        let provider = Arc::new(TfIdfProvider::default());
        let retriever = DenseVectorRetriever::new(storage.clone(), provider)
            .with_threshold(0.1); // Low threshold to include all

        // Store memories with varying relevance
        let exact_match = MemoryEntry::new(
            "exact".to_string(),
            "machine learning artificial intelligence".to_string(),
            MemoryType::ShortTerm,
        );
        let related = MemoryEntry::new(
            "related".to_string(),
            "machine learning neural networks".to_string(),
            MemoryType::ShortTerm,
        );
        let somewhat_related = MemoryEntry::new(
            "somewhat".to_string(),
            "computer science algorithms".to_string(),
            MemoryType::ShortTerm,
        );

        storage.store(&exact_match).await.unwrap();
        storage.store(&related).await.unwrap();
        storage.store(&somewhat_related).await.unwrap();

        // Search
        let results = retriever
            .search("machine learning artificial intelligence", 10, None)
            .await
            .unwrap();

        // Results should be ranked by similarity
        assert!(!results.is_empty());

        // Verify descending order
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }
}
