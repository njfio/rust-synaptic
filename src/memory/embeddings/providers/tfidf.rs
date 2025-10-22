//! TF-IDF based embedding provider
//!
//! Provides a local, fast baseline for semantic embeddings using
//! Term Frequency-Inverse Document Frequency (TF-IDF) vectors.

use super::super::provider::{
    EmbeddingProvider, Embedding, EmbedOptions, ProviderCapabilities, compute_content_hash, normalize_vector,
};
use crate::error::{MemoryError, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// TF-IDF based embedding provider
///
/// This provider uses Term Frequency-Inverse Document Frequency to create
/// semantic embeddings without requiring external APIs or models.
pub struct TfIdfProvider {
    config: TfIdfConfig,
    state: Arc<RwLock<TfIdfState>>,
}

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
struct TfIdfState {
    vocabulary: HashMap<String, usize>,
    idf_scores: HashMap<String, f64>,
    document_count: usize,
}

impl TfIdfState {
    fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
            document_count: 0,
        }
    }

    fn update_vocabulary(&mut self, text: &str, min_word_length: usize) {
        let tokens = Self::tokenize(text, min_word_length);
        let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();

        self.document_count += 1;

        for token in unique_tokens {
            *self.vocabulary.entry(token.clone()).or_insert(0) += 1;
        }

        // Recalculate IDF scores
        self.calculate_idf_scores();
    }

    fn calculate_idf_scores(&mut self) {
        if self.document_count == 0 {
            return;
        }

        self.idf_scores.clear();
        for (term, doc_freq) in &self.vocabulary {
            let idf = ((self.document_count as f64) / (*doc_freq as f64 + 1.0)).ln();
            self.idf_scores.insert(term.clone(), idf);
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

impl TfIdfProvider {
    /// Create a new TF-IDF provider
    pub fn new(config: TfIdfConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(TfIdfState::new())),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(TfIdfConfig::default())
    }

    /// Embed text using TF-IDF
    fn embed_text_sync(&self, text: &str) -> Result<Vec<f32>> {
        // Update vocabulary (this modifies state)
        {
            let mut state = self.state.write().unwrap();
            state.update_vocabulary(text, self.config.min_word_length);
        }

        // Generate embedding (read-only)
        let state = self.state.read().unwrap();
        let tokens = TfIdfState::tokenize(text, self.config.min_word_length);
        let tf_scores = TfIdfState::calculate_tf(&tokens);

        // Create embedding vector
        let mut embedding = vec![0.0f32; self.config.embedding_dim];

        for (token, tf) in tf_scores {
            let idf = state.idf_scores.get(&token).unwrap_or(&1.0);
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
}

#[async_trait]
impl EmbeddingProvider for TfIdfProvider {
    async fn embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding> {
        let content_hash = compute_content_hash(text);

        // Generate embedding
        let mut vector = self.embed_text_sync(text)?;

        // Apply options if provided
        if let Some(opts) = options {
            if opts.normalize {
                normalize_vector(&mut vector);
            }
        }

        // Create embedding with metadata
        let mut embedding = Embedding::new(vector, self.model_id())
            .with_content_hash(content_hash)
            .with_version("1.0".to_string());

        // Add custom metadata if provided
        if let Some(opts) = options {
            for (key, value) in &opts.metadata {
                embedding.metadata.insert(key.clone(), value.clone());
            }
        }

        Ok(embedding)
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
    async fn test_tfidf_embedding() {
        let provider = TfIdfProvider::default();
        let text = "This is a test document for TF-IDF embeddings";

        let embedding = provider.embed(text, None).await.unwrap();

        assert_eq!(embedding.dimension(), 384);
        assert_eq!(embedding.model, "tfidf-384");
        assert!(!embedding.content_hash.is_empty());
    }

    #[tokio::test]
    async fn test_tfidf_similarity() {
        let provider = TfIdfProvider::default();

        let emb1 = provider.embed("machine learning artificial intelligence", None).await.unwrap();
        let emb2 = provider.embed("machine learning deep neural networks", None).await.unwrap();
        let emb3 = provider.embed("cooking recipes Italian pasta", None).await.unwrap();

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

        let embeddings = provider.embed_batch(&texts, None).await.unwrap();

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

        let embedding = provider.embed("test text", Some(&options)).await.unwrap();

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
        let tokens = vec![
            "test".to_string(),
            "test".to_string(),
            "word".to_string(),
        ];

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
