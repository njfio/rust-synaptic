//! Tests for the pluggable semantic embedding provider (retrieval Task 1).
//!
//! Proves, with a MOCK semantic provider (no network), that:
//! (a) a real semantic embedding space lets the dense retriever rank a
//!     PARAPHRASE of the query above a lexically-overlapping-but-unrelated
//!     document,
//! (b) the TF-IDF default CANNOT do that on the same corpus (so the pluggable
//!     provider adds real capability, not a re-labelled lexical path), and
//! (c) a configured-but-broken semantic provider falls back to TF-IDF with no
//!     panic and no pipeline failure.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use async_trait::async_trait;
use std::sync::Arc;
use synaptic::error::{MemoryError, Result};
use synaptic::memory::embeddings::provider::{EmbedOptions, Embedding, EmbeddingProvider};
use synaptic::memory::embeddings::{
    build_retrieval_provider, FallbackEmbeddingProvider, RetrievalEmbeddingConfig, TfIdfProvider,
};
use synaptic::memory::retrieval::{DenseVectorRetriever, RetrievalPipeline};
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::storage::Storage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

const QUERY: &str = "automobile repair";
/// Paraphrase of the query: same meaning, minimal lexical overlap
/// (shares only "automobile" so it can be a lexical candidate).
const PARAPHRASE_DOC: &str = "the mechanic fixed my broken automobile yesterday";
/// Lexically loaded but semantically unrelated (a pricing catalog).
const DISTRACTOR_DOC: &str =
    "automobile repair automobile repair manual pricing catalog advertisement";

/// Mock semantic provider: returns controlled vectors so that the paraphrase
/// is CLOSE to the query in embedding space and the distractor is FAR —
/// exactly what a real semantic model (e.g. nomic-embed-text) produces and
/// TF-IDF cannot.
struct MockSemanticProvider;

impl MockSemanticProvider {
    fn vector_for(text: &str) -> Vec<f32> {
        if text == QUERY {
            vec![1.0, 0.0, 0.0]
        } else if text == PARAPHRASE_DOC {
            // Near the query direction: cosine ~0.98
            vec![0.98, 0.2, 0.0]
        } else if text == DISTRACTOR_DOC {
            // Nearly orthogonal to the query: cosine ~0.05
            vec![0.05, 0.0, 1.0]
        } else {
            vec![0.0, 0.0, 1.0]
        }
    }
}

#[async_trait]
impl EmbeddingProvider for MockSemanticProvider {
    async fn embed(&self, text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        Ok(Embedding::new(
            Self::vector_for(text),
            "mock-semantic".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        3
    }

    fn name(&self) -> &'static str {
        "MockSemanticProvider"
    }

    fn model_id(&self) -> String {
        "mock-semantic".to_string()
    }
}

/// A provider that always fails: stands in for a configured Ollama endpoint
/// that is down.
struct AlwaysFailingProvider;

#[async_trait]
impl EmbeddingProvider for AlwaysFailingProvider {
    async fn embed(&self, _text: &str, _options: Option<&EmbedOptions>) -> Result<Embedding> {
        Err(MemoryError::External(
            "unreachable semantic endpoint (test double)".to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        768
    }

    fn name(&self) -> &'static str {
        "AlwaysFailingProvider"
    }

    fn model_id(&self) -> String {
        "failing".to_string()
    }
}

async fn corpus_storage() -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());
    let paraphrase = MemoryEntry::new(
        "paraphrase".to_string(),
        PARAPHRASE_DOC.to_string(),
        MemoryType::ShortTerm,
    );
    let distractor = MemoryEntry::new(
        "distractor".to_string(),
        DISTRACTOR_DOC.to_string(),
        MemoryType::ShortTerm,
    );
    storage.store(&paraphrase).await.unwrap();
    storage.store(&distractor).await.unwrap();
    storage
}

/// (a) With a semantic provider the dense retriever ranks the PARAPHRASE
/// first, above the lexically-overlapping distractor.
#[tokio::test]
async fn semantic_provider_ranks_paraphrase_first() {
    let storage = corpus_storage().await;
    let provider: Arc<dyn EmbeddingProvider> = Arc::new(MockSemanticProvider);
    let retriever = DenseVectorRetriever::new(storage, provider).with_threshold(0.0);

    let results = retriever.search(QUERY, 10, None).await.unwrap();
    assert!(
        results.len() >= 2,
        "both docs must be candidates, got {}",
        results.len()
    );
    assert_eq!(
        results[0].memory.entry.key, "paraphrase",
        "semantic provider must rank the paraphrase first"
    );
}

/// (b) TF-IDF on the SAME corpus does NOT rank the paraphrase first — the
/// lexical distractor wins. This proves the semantic provider adds a real
/// capability rather than duplicating the lexical path.
#[tokio::test]
async fn tfidf_provider_cannot_rank_paraphrase_first() {
    let storage = corpus_storage().await;
    let tfidf = Arc::new(TfIdfProvider::default());
    // Feed the corpus through the document path, as the store path does.
    tfidf.embed(PARAPHRASE_DOC, None).await.unwrap();
    tfidf.embed(DISTRACTOR_DOC, None).await.unwrap();
    let provider: Arc<dyn EmbeddingProvider> = tfidf;
    let retriever = DenseVectorRetriever::new(storage, provider).with_threshold(0.0);

    let results = retriever.search(QUERY, 10, None).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(
        results[0].memory.entry.key, "distractor",
        "TF-IDF is lexical: the term-loaded distractor must outrank the paraphrase"
    );
}

/// (c) A configured-but-broken semantic provider falls back to TF-IDF:
/// embedding still succeeds (from the fallback), nothing panics, and the
/// wrapper reports the primary as no longer active.
#[tokio::test]
async fn broken_semantic_provider_falls_back_to_tfidf() {
    let tfidf = Arc::new(TfIdfProvider::default());
    let fallback =
        FallbackEmbeddingProvider::new(Arc::new(AlwaysFailingProvider), Arc::clone(&tfidf));
    assert!(fallback.primary_active(), "primary starts active");

    // Document path: primary fails -> TF-IDF fallback serves the embed.
    let emb = fallback.embed(PARAPHRASE_DOC, None).await.unwrap();
    assert!(!emb.vector.is_empty());
    assert!(
        !fallback.primary_active(),
        "primary must be marked failed after first error"
    );
    assert!(
        fallback.is_available(),
        "pipeline stays available via the TF-IDF fallback"
    );

    // Scoring path also served by the fallback; corpus feed above is visible.
    let q = fallback.embed_for_scoring(QUERY, None).await.unwrap();
    assert!(!q.vector.is_empty());
    #[cfg(feature = "test-utils")]
    assert!(
        tfidf.vocabulary_size() > 0,
        "fallback TF-IDF corpus is live"
    );
    #[cfg(not(feature = "test-utils"))]
    let _ = &tfidf;

    // And the dense retriever keeps working end-to-end with the wrapper.
    let storage = corpus_storage().await;
    let retriever =
        DenseVectorRetriever::new(storage, Arc::new(fallback) as Arc<dyn EmbeddingProvider>)
            .with_threshold(0.0);
    let results = retriever.search(QUERY, 10, None).await.unwrap();
    assert!(!results.is_empty(), "fallback path must return results");
}

/// While the primary is healthy, the wrapper still feeds the document path
/// into the TF-IDF fallback so a later fail-over scores against a live
/// corpus, not an empty vocabulary.
#[tokio::test]
async fn fallback_corpus_stays_warm_while_primary_healthy() {
    let tfidf = Arc::new(TfIdfProvider::default());
    let fallback =
        FallbackEmbeddingProvider::new(Arc::new(MockSemanticProvider), Arc::clone(&tfidf));

    let emb = fallback.embed(PARAPHRASE_DOC, None).await.unwrap();
    // Primary healthy: its embedding is returned...
    assert_eq!(emb.vector.len(), 3, "primary embedding served");
    assert!(fallback.primary_active());
    // ...but the TF-IDF fallback corpus was fed too.
    #[cfg(feature = "test-utils")]
    assert!(
        tfidf.vocabulary_size() > 0,
        "document path must warm the fallback corpus"
    );
    #[cfg(not(feature = "test-utils"))]
    {
        // Without test-utils the vocabulary counter is not exposed; prove the
        // warm corpus behaviorally: the fallback embeds the fed doc fine.
        let fed = tfidf.embed_for_scoring(PARAPHRASE_DOC, None).await.unwrap();
        assert!(!fed.vector.is_empty());
    }
}

/// Selecting the Ollama provider config with an unreachable endpoint must
/// never fail the pipeline: construction succeeds and embeds are served
/// (falling back to TF-IDF). Works with or without the `llm-integration`
/// feature; no external network is touched (loopback port 9).
#[tokio::test]
async fn ollama_config_unreachable_endpoint_never_fails_pipeline() {
    let config = RetrievalEmbeddingConfig::Ollama {
        endpoint: "http://127.0.0.1:9".to_string(),
        model: "nomic-embed-text".to_string(),
        dimension: 768,
    };
    let (provider, tfidf) = build_retrieval_provider(&config);
    let emb = provider.embed("hello fallback world", None).await.unwrap();
    assert!(!emb.vector.is_empty());
    assert!(provider.is_available());
    #[cfg(feature = "test-utils")]
    assert!(
        tfidf.vocabulary_size() > 0,
        "TF-IDF fallback corpus fed through the returned provider"
    );
    #[cfg(not(feature = "test-utils"))]
    let _ = &tfidf;
}

/// Default config selects TF-IDF (offline, no new deps): the returned dyn
/// provider IS the TF-IDF instance (same corpus statistics).
#[tokio::test]
async fn default_config_is_tfidf() {
    let config = RetrievalEmbeddingConfig::default();
    let (provider, tfidf) = build_retrieval_provider(&config);
    assert_eq!(provider.model_id(), tfidf.model_id());
    provider.embed("corpus doc one", None).await.unwrap();
    #[cfg(feature = "test-utils")]
    assert!(
        tfidf.vocabulary_size() > 0,
        "embedding through the dyn handle must update the shared TF-IDF corpus"
    );
    #[cfg(not(feature = "test-utils"))]
    let _ = &tfidf;
}

/// End-to-end wiring: `MemoryConfig.retrieval_embedding_provider` with a
/// custom (mock semantic) provider constructs an AgentMemory whose store and
/// search paths work — no panic, results returned.
#[tokio::test]
async fn agent_memory_accepts_custom_retrieval_provider() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_embeddings: true,
        retrieval_embedding_provider: RetrievalEmbeddingConfig::Custom(Arc::new(
            MockSemanticProvider,
        )),
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await.unwrap();
    memory.store("paraphrase", PARAPHRASE_DOC).await.unwrap();
    memory.store("distractor", DISTRACTOR_DOC).await.unwrap();

    let results = memory.search(QUERY, 10).await.unwrap();
    assert!(
        !results.is_empty(),
        "search must work with a custom semantic provider"
    );
}
