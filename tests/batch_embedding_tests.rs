//! Batched candidate embedding: correctness and wiring proofs.
//!
//! (a) Identity: `embed_for_scoring_batch(&[a, b, c])` must return the SAME
//!     vectors as calling `embed_for_scoring` on each text individually —
//!     batching is a performance optimization, never a semantics change.
//! (b) Wiring: the dense retriever must embed its candidate set through ONE
//!     batch call per query, not N per-candidate calls (proved with a
//!     counting mock provider).
//! (c) FallbackEmbeddingProvider forwards the batch path to the primary and,
//!     after a primary failure, serves batches from the TF-IDF fallback
//!     (sticky, no panic).

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use async_trait::async_trait;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use synaptic::error::{MemoryError, Result};
use synaptic::memory::embeddings::provider::{EmbedOptions, Embedding, EmbeddingProvider};
use synaptic::memory::embeddings::providers::FallbackEmbeddingProvider;
use synaptic::memory::embeddings::TfIdfProvider;
use synaptic::memory::retrieval::dense_vector::DenseVectorRetriever;
use synaptic::memory::retrieval::pipeline::RetrievalPipeline;
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::Storage;
use synaptic::memory::types::{MemoryEntry, MemoryType};

/// Deterministic mock provider that counts single vs batch scoring calls.
struct CountingProvider {
    single_scoring_calls: AtomicUsize,
    batch_scoring_calls: AtomicUsize,
    batch_texts_embedded: AtomicUsize,
    fail: bool,
}

impl CountingProvider {
    fn new() -> Self {
        Self {
            single_scoring_calls: AtomicUsize::new(0),
            batch_scoring_calls: AtomicUsize::new(0),
            batch_texts_embedded: AtomicUsize::new(0),
            fail: false,
        }
    }

    fn failing() -> Self {
        Self {
            fail: true,
            ..Self::new()
        }
    }

    /// Deterministic per-text vector so identity checks are meaningful.
    fn vector_for(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; 8];
        for (i, b) in text.bytes().enumerate() {
            v[i % 8] += b as f32 / 255.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl EmbeddingProvider for CountingProvider {
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        if self.fail {
            return Err(MemoryError::External("mock primary down".to_string()));
        }
        Ok(Embedding::new(
            Self::vector_for(text),
            "counting-mock".to_string(),
        ))
    }

    async fn embed_for_scoring(
        &self,
        text: &str,
        options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        self.single_scoring_calls.fetch_add(1, Ordering::SeqCst);
        self.embed(text, options).await
    }

    async fn embed_for_scoring_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        self.batch_scoring_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(MemoryError::External("mock primary down".to_string()));
        }
        self.batch_texts_embedded
            .fetch_add(texts.len(), Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| Embedding::new(Self::vector_for(t), "counting-mock".to_string()))
            .collect())
    }

    fn embedding_dimension(&self) -> usize {
        8
    }

    fn name(&self) -> &'static str {
        "CountingProvider"
    }

    fn model_id(&self) -> String {
        "counting-mock".to_string()
    }
}

fn assert_vectors_identical(a: &[f32], b: &[f32], context: &str) {
    assert_eq!(a.len(), b.len(), "dimension mismatch: {}", context);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "{}: component {} differs: {} vs {}",
            context,
            i,
            x,
            y
        );
    }
}

/// (a) Default trait impl: batch == per-text singles, in order.
#[tokio::test]
async fn batch_scoring_matches_individual_scoring_default_impl() {
    // TF-IDF exercises the trait's default sequential implementation and the
    // provider's own content-hash scoring cache.
    let provider = TfIdfProvider::default();
    for doc in [
        "database connection pooling configuration",
        "postgres tuning guide",
        "summer volleyball picnic",
    ] {
        provider.embed(doc, None).await.unwrap();
    }

    let texts = [
        "database connection pooling configuration",
        "postgres tuning guide",
        "summer volleyball picnic",
    ];
    let individual = {
        let mut v = Vec::new();
        for t in &texts {
            v.push(provider.embed_for_scoring(t, None).await.unwrap());
        }
        v
    };
    let batched = provider.embed_for_scoring_batch(&texts).await.unwrap();

    assert_eq!(batched.len(), individual.len());
    for (i, (b, s)) in batched.iter().zip(individual.iter()).enumerate() {
        assert_vectors_identical(&b.vector, &s.vector, &format!("text {}", i));
    }
}

/// (a) Mock provider identity through the batch path.
#[tokio::test]
async fn batch_scoring_matches_individual_scoring_mock() {
    let provider = CountingProvider::new();
    let texts = ["alpha beta", "gamma", "delta epsilon zeta"];
    let batched = provider.embed_for_scoring_batch(&texts).await.unwrap();
    for (i, t) in texts.iter().enumerate() {
        let single = provider.embed_for_scoring(t, None).await.unwrap();
        assert_vectors_identical(&batched[i].vector, &single.vector, t);
    }
}

/// (b) The dense retriever embeds ALL candidates via ONE batch call per
/// query; the only single scoring call is the query itself.
#[tokio::test]
async fn dense_retriever_uses_one_batch_call_per_query() {
    let storage = Arc::new(MemoryStorage::new());
    let provider = Arc::new(CountingProvider::new());
    let retriever =
        DenseVectorRetriever::new(storage.clone(), provider.clone()).with_threshold(0.0);

    for i in 0..5 {
        let entry = MemoryEntry::new(
            format!("mem{}", i),
            format!("shared topic content variant {}", i),
            MemoryType::ShortTerm,
        );
        storage.store(&entry).await.unwrap();
    }

    let results = retriever
        .search("shared topic content", 10, None)
        .await
        .unwrap();
    assert!(!results.is_empty(), "retriever must return candidates");

    assert_eq!(
        provider.batch_scoring_calls.load(Ordering::SeqCst),
        1,
        "all candidates must be embedded in exactly ONE batch call"
    );
    assert!(
        provider.batch_texts_embedded.load(Ordering::SeqCst) >= results.len(),
        "the batch must cover every scored candidate"
    );
    assert_eq!(
        provider.single_scoring_calls.load(Ordering::SeqCst),
        1,
        "only the query may use the single scoring path"
    );
}

/// (c) FallbackEmbeddingProvider forwards batches to the primary while
/// healthy, and identity holds through the wrapper.
#[tokio::test]
async fn fallback_provider_forwards_batch_to_primary() {
    let primary = Arc::new(CountingProvider::new());
    let tfidf = Arc::new(TfIdfProvider::default());
    let wrapper = FallbackEmbeddingProvider::new(primary.clone(), tfidf);

    let texts = ["one", "two", "three"];
    let batched = wrapper.embed_for_scoring_batch(&texts).await.unwrap();

    assert_eq!(primary.batch_scoring_calls.load(Ordering::SeqCst), 1);
    assert!(wrapper.primary_active());
    for (i, t) in texts.iter().enumerate() {
        assert_vectors_identical(&batched[i].vector, &CountingProvider::vector_for(t), t);
    }
}

/// (c) On primary batch failure the wrapper fails over to TF-IDF (sticky),
/// returns real fallback embeddings, and never panics.
#[tokio::test]
async fn fallback_provider_batch_fails_over_to_tfidf() {
    let primary = Arc::new(CountingProvider::failing());
    let tfidf = Arc::new(TfIdfProvider::default());
    tfidf
        .embed("database connection pooling", None)
        .await
        .unwrap();
    let wrapper = FallbackEmbeddingProvider::new(primary.clone(), tfidf.clone());

    let texts = ["database connection", "pooling"];
    let batched = wrapper.embed_for_scoring_batch(&texts).await.unwrap();
    assert_eq!(batched.len(), 2);
    assert!(!wrapper.primary_active(), "primary failure must be sticky");

    // Identity against the fallback's own scoring path.
    for (i, t) in texts.iter().enumerate() {
        let single = tfidf.embed_for_scoring(t, None).await.unwrap();
        assert_vectors_identical(&batched[i].vector, &single.vector, t);
    }

    // Subsequent batches go straight to the fallback (primary not retried).
    let before = primary.batch_scoring_calls.load(Ordering::SeqCst);
    wrapper.embed_for_scoring_batch(&texts).await.unwrap();
    assert_eq!(primary.batch_scoring_calls.load(Ordering::SeqCst), before);
}
