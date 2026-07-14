//! TF-IDF based embedding provider
//!
//! Provides a local, fast baseline for semantic embeddings using
//! Term Frequency-Inverse Document Frequency (TF-IDF) vectors.

use super::super::provider::{
    compute_content_hash, normalize_vector, EmbedOptions, Embedding, EmbeddingProvider,
    ProviderCapabilities,
};
use crate::error::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// TF-IDF based embedding provider
///
/// This provider uses Term Frequency-Inverse Document Frequency to create
/// semantic embeddings without requiring external APIs or models.
///
/// Two embedding paths exist:
/// - [`EmbeddingProvider::embed`] — the document (store-time) path: updates
///   the vocabulary incrementally (O(tokens), no full IDF-table rebuild) and
///   then computes the vector against the updated statistics.
/// - [`EmbeddingProvider::embed_for_scoring`] — the query-time path: computes
///   the vector against the EXISTING statistics without taking a write lock
///   or mutating any state, and caches the result by content hash so a given
///   content string is embedded at most once (shared across the dense
///   retriever and the reranker when they hold the same provider).
///
/// The scoring cache is invalidated whenever `embed` mutates the vocabulary
/// (so a cached vector never reflects stale IDF state), and its size is
/// capped ([`SCORING_CACHE_CAP`]) to bound memory.
///
/// In the shipped hybrid pipeline `AgentMemory` retains the provider instance
/// shared by the dense retriever and reranker and feeds every stored memory's
/// content into it via `embed`, so the corpus IDF statistics stay live and
/// query-time scoring performs real IDF weighting (previously the provider
/// was never fed and IDF degenerated to the uniform `1.0` fallback).
pub struct TfIdfProvider {
    config: TfIdfConfig,
    state: Arc<RwLock<TfIdfState>>,
    /// Content-hash-keyed cache of scoring vectors (query-time path only).
    /// Invalidated on vocabulary mutation; bounded by [`SCORING_CACHE_CAP`].
    scoring_cache: DashMap<String, Vec<f32>>,
    /// Test-only op counters for the scoring path.
    #[cfg(feature = "test-utils")]
    scoring_embeds_computed: std::sync::atomic::AtomicUsize,
    #[cfg(feature = "test-utils")]
    scoring_cache_hits: std::sync::atomic::AtomicUsize,
}

/// Upper bound on the number of entries retained in the scoring cache. When
/// an insertion would exceed this cap the cache is cleared wholesale (a crude
/// but O(1)-amortized eviction — correctness is unaffected, only cache hit
/// rate). Sized generously so a single search's candidate set always fits.
const SCORING_CACHE_CAP: usize = 8192;

/// Configuration for TF-IDF embeddings
#[derive(Debug, Clone)]
pub struct TfIdfConfig {
    /// Dimension of the embedding vectors
    pub embedding_dim: usize,
    /// Minimum word length to include
    pub min_word_length: usize,
    /// Maximum words to include in vocabulary
    pub max_vocabulary_size: usize,
    /// Whether to normalize vectors
    pub normalize: bool,
}

impl Default for TfIdfConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            min_word_length: 3,
            max_vocabulary_size: 10000,
            normalize: true,
        }
    }
}

/// Internal state for TF-IDF embeddings
///
/// IDF is derived lazily from `(vocabulary doc-frequency, document_count)`
/// at embed time — there is no materialized IDF table to rebuild, so
/// ingesting a document is O(tokens), not O(vocabulary).
struct TfIdfState {
    vocabulary: HashMap<String, usize>,
    document_count: usize,
}

impl TfIdfState {
    fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            document_count: 0,
        }
    }

    /// Incremental vocabulary update: O(tokens in `text`). No IDF-table
    /// rebuild — IDF is computed lazily via [`Self::idf`].
    fn update_vocabulary(&mut self, text: &str, min_word_length: usize) {
        let tokens = Self::tokenize(text, min_word_length);
        let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();

        self.document_count += 1;

        for token in unique_tokens {
            *self.vocabulary.entry(token).or_insert(0) += 1;
        }
    }

    /// Lazy IDF for a single term, identical in value to the previous
    /// materialized table: `ln(N / (df + 1))` for known terms, `1.0` for
    /// terms outside the vocabulary (the old table-miss default).
    fn idf(&self, token: &str) -> f64 {
        match self.vocabulary.get(token) {
            Some(doc_freq) if self.document_count > 0 => {
                ((self.document_count as f64) / (*doc_freq as f64 + 1.0)).ln()
            }
            _ => 1.0,
        }
    }

    fn tokenize(text: &str, min_length: usize) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() >= min_length)
            .map(|word| {
                // Remove punctuation
                word.chars()
                    .filter(|c| c.is_alphabetic() || c.is_numeric())
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn calculate_tf(tokens: &[String]) -> HashMap<String, f64> {
        let total = tokens.len() as f64;
        let mut tf_scores = HashMap::new();

        for token in tokens {
            *tf_scores.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize by total count
        for score in tf_scores.values_mut() {
            *score /= total;
        }

        tf_scores
    }

    fn hash_to_index(token: &str, embedding_dim: usize) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        (hasher.finish() % embedding_dim as u64) as usize
    }
}

impl Default for TfIdfProvider {
    fn default() -> Self {
        Self::new(TfIdfConfig::default())
    }
}

impl TfIdfProvider {
    /// Create a new TF-IDF provider
    pub fn new(config: TfIdfConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(TfIdfState::new())),
            scoring_cache: DashMap::new(),
            #[cfg(feature = "test-utils")]
            scoring_embeds_computed: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "test-utils")]
            scoring_cache_hits: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Embed text using TF-IDF (document path: updates vocabulary first)
    fn embed_text_sync(&self, text: &str) -> Result<Vec<f32>> {
        // Incremental vocabulary update (O(tokens), no IDF-table rebuild)
        {
            let mut state = self
                .state
                .write()
                .expect("TfIdfProvider state lock is never poisoned: no panics while held");
            state.update_vocabulary(text, self.config.min_word_length);
        }
        // Vocabulary/IDF state just changed: any cached scoring vector is now
        // computed against stale statistics, so drop the whole scoring cache.
        // (Under the shipped wiring this fires on every store — the store
        // path feeds each memory into the scoring provider via `embed` — so
        // scoring vectors always reflect the current corpus IDF.)
        self.scoring_cache.clear();
        self.embed_text_readonly(text)
    }

    /// Compute the TF-IDF vector for `text` against the EXISTING statistics.
    /// Read-only: takes only a read lock and never mutates vocabulary.
    fn embed_text_readonly(&self, text: &str) -> Result<Vec<f32>> {
        let state = self
            .state
            .read()
            .expect("TfIdfProvider state lock is never poisoned: no panics while held");
        let tokens = TfIdfState::tokenize(text, self.config.min_word_length);
        let tf_scores = TfIdfState::calculate_tf(&tokens);

        // Create embedding vector
        let mut embedding = vec![0.0f32; self.config.embedding_dim];

        for (token, tf) in tf_scores {
            let idf = state.idf(&token);
            let tfidf = (tf * idf) as f32;

            // Hash token to embedding dimension
            let index = TfIdfState::hash_to_index(&token, self.config.embedding_dim);
            embedding[index] += tfidf;
        }

        // Normalize if configured
        if self.config.normalize {
            normalize_vector(&mut embedding);
        }

        Ok(embedding)
    }

    /// Wrap a raw vector into an [`Embedding`] with standard metadata.
    fn build_embedding(
        &self,
        mut vector: Vec<f32>,
        content_hash: String,
        options: Option<&EmbedOptions>,
    ) -> Embedding {
        if let Some(opts) = options {
            if opts.normalize {
                normalize_vector(&mut vector);
            }
        }

        let mut embedding = Embedding::new(vector, self.model_id())
            .with_content_hash(content_hash)
            .with_version("1.0".to_string());

        if let Some(opts) = options {
            for (key, value) in &opts.metadata {
                embedding.metadata.insert(key.clone(), value.clone());
            }
        }

        embedding
    }

    /// Number of terms currently in the vocabulary (test-utils).
    #[cfg(feature = "test-utils")]
    pub fn vocabulary_size(&self) -> usize {
        self.state
            .read()
            .expect("TfIdfProvider state lock is never poisoned: no panics while held")
            .vocabulary
            .len()
    }

    /// Number of scoring embeddings actually computed (cache misses)
    /// since construction (test-utils).
    #[cfg(feature = "test-utils")]
    pub fn scoring_embeds_computed(&self) -> usize {
        self.scoring_embeds_computed
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of scoring embeddings served from the content-hash cache
    /// since construction (test-utils).
    #[cfg(feature = "test-utils")]
    pub fn scoring_cache_hits(&self) -> usize {
        self.scoring_cache_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl EmbeddingProvider for TfIdfProvider {
    async fn embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);
        let vector = self.embed_text_sync(text)?;
        Ok(self.build_embedding(vector, content_hash, options))
    }

    /// Query-time scoring path: read-only against the existing IDF
    /// statistics, cached by content hash so each distinct content is
    /// embedded at most once across the dense retriever and the reranker.
    async fn embed_for_scoring(
        &self,
        text: &str,
        options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);

        if let Some(cached) = self.scoring_cache.get(&content_hash) {
            #[cfg(feature = "test-utils")]
            self.scoring_cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let vector = cached.clone();
            return Ok(self.build_embedding(vector, content_hash, options));
        }

        let vector = self.embed_text_readonly(text)?;
        #[cfg(feature = "test-utils")]
        self.scoring_embeds_computed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Bound cache growth: clear wholesale before crossing the cap. Crude
        // but O(1)-amortized; only affects hit rate, never correctness.
        if self.scoring_cache.len() >= SCORING_CACHE_CAP {
            self.scoring_cache.clear();
        }
        self.scoring_cache
            .insert(content_hash.clone(), vector.clone());
        Ok(self.build_embedding(vector, content_hash, options))
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        // TF-IDF benefits from batch processing to update vocabulary
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            embeddings.push(self.embed(text, options).await?);
        }

        Ok(embeddings)
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dim
    }

    fn name(&self) -> &'static str {
        "TfIdfProvider"
    }

    fn model_id(&self) -> String {
        format!("tfidf-{}", self.config.embedding_dim)
    }

    fn is_available(&self) -> bool {
        true // Always available (local)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_batch: true,
            max_batch_size: Some(1000),
            max_input_length: None, // No hard limit
            supports_input_types: false,
            requires_api_key: false,
            is_local: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tfidf_provider_creation() {
        let provider = TfIdfProvider::default();
        assert_eq!(provider.embedding_dimension(), 384);
        assert_eq!(provider.name(), "TfIdfProvider");
        assert!(provider.is_available());
    }

    #[tokio::test]
    async fn scoring_cache_invalidated_on_vocab_mutation() {
        // A scoring vector cached against one vocabulary state must NOT be
        // served stale after `embed` mutates the vocabulary/IDF state.
        let provider = TfIdfProvider::default();
        let text = "alpha beta gamma delta";

        // Seed some vocabulary, then cache a scoring vector for `text`.
        provider
            .embed("alpha beta gamma delta", None)
            .await
            .expect("seed embed");
        let before = provider
            .embed_for_scoring(text, None)
            .await
            .expect("scoring embed")
            .vector;

        // Mutate the vocabulary via the document path so that `alpha`'s
        // document frequency rises while `beta`/`gamma`/`delta` stay at 1.
        // This changes the RELATIVE IDF weighting among the query's tokens
        // (a change that survives normalization), so the scoring vector must
        // differ from the cached one.
        for _ in 0..25 {
            provider
                .embed("alpha epsilon zeta eta", None)
                .await
                .expect("mutating embed");
        }

        let after = provider
            .embed_for_scoring(text, None)
            .await
            .expect("scoring embed after mutation")
            .vector;

        assert_ne!(
            before, after,
            "scoring cache must reflect the mutated vocabulary, not a stale hit"
        );
    }

    #[tokio::test]
    async fn test_tfidf_embedding() {
        let provider = TfIdfProvider::default();
        let text = "This is a test document for TF-IDF embeddings";

        let embedding = provider
            .embed(text, None)
            .await
            .expect("await should be present");

        assert_eq!(embedding.dimension(), 384);
        assert_eq!(embedding.model, "tfidf-384");
        assert!(!embedding.content_hash.is_empty());
    }

    #[tokio::test]
    async fn test_tfidf_similarity() {
        let provider = TfIdfProvider::default();

        let emb1 = provider
            .embed("machine learning artificial intelligence", None)
            .await
            .expect("await should be present");
        let emb2 = provider
            .embed("machine learning deep neural networks", None)
            .await
            .expect("await should be present");
        let emb3 = provider
            .embed("cooking recipes Italian pasta", None)
            .await
            .expect("await should be present");

        // Similar documents should have higher similarity
        let sim_related = emb1.cosine_similarity(&emb2);
        let sim_unrelated = emb1.cosine_similarity(&emb3);

        assert!(sim_related > sim_unrelated);
    }

    #[tokio::test]
    async fn test_tfidf_batch_embedding() {
        let provider = TfIdfProvider::default();

        let texts = vec![
            "first document".to_string(),
            "second document".to_string(),
            "third document".to_string(),
        ];

        let embeddings = provider
            .embed_batch(&texts, None)
            .await
            .expect("await should be present");

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.dimension(), 384);
        }
    }

    #[tokio::test]
    async fn test_tfidf_with_options() {
        let provider = TfIdfProvider::default();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let options = EmbedOptions {
            truncate: false,
            normalize: true,
            metadata,
        };

        let embedding = provider
            .embed("test text", Some(&options))
            .await
            .expect("await should be present");

        assert_eq!(embedding.metadata.get("source"), Some(&"test".to_string()));

        // Check normalization (vector should have unit length)
        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tokenization() {
        let tokens = TfIdfState::tokenize("Hello, World! This is a test.", 3);
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Short words should be filtered
        assert!(!tokens.iter().any(|t| t.len() < 3));
    }

    #[test]
    fn test_tf_calculation() {
        let tokens = vec!["test".to_string(), "test".to_string(), "word".to_string()];

        let tf = TfIdfState::calculate_tf(&tokens);

        assert!((tf["test"] - 2.0 / 3.0).abs() < 0.001);
        assert!((tf["word"] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_vocabulary_update() {
        let mut state = TfIdfState::new();

        state.update_vocabulary("first document", 3);
        assert_eq!(state.document_count, 1);

        state.update_vocabulary("second document", 3);
        assert_eq!(state.document_count, 2);

        // "document" should appear in both, so doc_freq = 2
        assert_eq!(state.vocabulary.get("document"), Some(&2));
    }
}
