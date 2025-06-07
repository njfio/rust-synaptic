//! Simple TF-IDF based embeddings implementation
//! 
//! This provides a working baseline for semantic embeddings using
//! Term Frequency-Inverse Document Frequency (TF-IDF) vectors.

use crate::error::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Simple embedder using TF-IDF
pub struct SimpleEmbedder {
    embedding_dim: usize,
    vocabulary: HashMap<String, usize>,
    idf_scores: HashMap<String, f64>,
    document_count: usize,
}

impl SimpleEmbedder {
    /// Create a new simple embedder
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
            document_count: 0,
        }
    }

    /// Embed text into a vector
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f64>> {
        // Update vocabulary with this text
        self.update_vocabulary(text);

        let tokens = self.tokenize(text);
        let tf_scores = self.calculate_tf(&tokens);
        
        // Create embedding vector
        let mut embedding = vec![0.0; self.embedding_dim];
        
        for (token, tf) in tf_scores {
            let idf = self.idf_scores.get(&token).unwrap_or(&1.0);
            let tfidf = tf * idf;
            
            // Hash token to embedding dimension
            let index = self.hash_to_index(&token);
            embedding[index] += tfidf;
        }

        // Normalize the vector
        self.normalize_vector(&mut embedding);
        
        Ok(embedding)
    }

    /// Update vocabulary and IDF scores with new text
    pub fn update_vocabulary(&mut self, text: &str) {
        let tokens = self.tokenize(text);
        let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();
        
        self.document_count += 1;
        
        for token in unique_tokens {
            *self.vocabulary.entry(token.clone()).or_insert(0) += 1;
        }
        
        // Recalculate IDF scores
        self.calculate_idf_scores();
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2) // Filter out very short words
            .map(|word| {
                // Remove punctuation
                word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Calculate term frequency for tokens
    fn calculate_tf(&self, tokens: &[String]) -> HashMap<String, f64> {
        let mut tf_scores = HashMap::new();
        let total_tokens = tokens.len() as f64;
        
        for token in tokens {
            *tf_scores.entry(token.clone()).or_insert(0.0) += 1.0;
        }
        
        // Normalize by total token count
        for score in tf_scores.values_mut() {
            *score /= total_tokens;
        }
        
        tf_scores
    }

    /// Calculate IDF scores for all vocabulary
    fn calculate_idf_scores(&mut self) {
        self.idf_scores.clear();
        
        for (token, doc_freq) in &self.vocabulary {
            let idf = (self.document_count as f64 / *doc_freq as f64).ln();
            self.idf_scores.insert(token.clone(), idf);
        }
    }

    /// Hash a token to an embedding dimension index
    fn hash_to_index(&self, token: &str) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        (hasher.finish() as usize) % self.embedding_dim
    }

    /// Normalize a vector to unit length
    fn normalize_vector(&self, vector: &mut [f64]) {
        let magnitude: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if magnitude > 0.0 {
            for value in vector.iter_mut() {
                *value /= magnitude;
            }
        }
    }

    /// Get vocabulary statistics
    pub fn get_vocab_stats(&self) -> VocabularyStats {
        VocabularyStats {
            vocabulary_size: self.vocabulary.len(),
            document_count: self.document_count,
            embedding_dimension: self.embedding_dim,
            most_frequent_terms: self.get_most_frequent_terms(10),
        }
    }

    /// Get most frequent terms
    fn get_most_frequent_terms(&self, limit: usize) -> Vec<(String, usize)> {
        let mut terms: Vec<_> = self.vocabulary.iter().collect();
        terms.sort_by(|a, b| b.1.cmp(a.1));
        terms.into_iter()
            .take(limit)
            .map(|(term, freq)| (term.clone(), *freq))
            .collect()
    }
}

/// Statistics about the vocabulary
#[derive(Debug, Clone)]
pub struct VocabularyStats {
    pub vocabulary_size: usize,
    pub document_count: usize,
    pub embedding_dimension: usize,
    pub most_frequent_terms: Vec<(String, usize)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_embedder_creation() {
        let embedder = SimpleEmbedder::new(100);
        let stats = embedder.get_vocab_stats();
        
        assert_eq!(stats.vocabulary_size, 0);
        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.embedding_dimension, 100);
    }

    #[test]
    fn test_tokenization() {
        let embedder = SimpleEmbedder::new(100);
        let tokens = embedder.tokenize("Hello, world! This is a test.");
        
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Short words should be filtered out
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_vocabulary_update() {
        let mut embedder = SimpleEmbedder::new(100);
        
        embedder.update_vocabulary("artificial intelligence machine learning");
        embedder.update_vocabulary("machine learning algorithms");
        
        let stats = embedder.get_vocab_stats();
        assert_eq!(stats.document_count, 2);
        assert!(stats.vocabulary_size > 0);
        
        // "machine" and "learning" should appear in both documents
        assert_eq!(embedder.vocabulary.get("machine"), Some(&2));
        assert_eq!(embedder.vocabulary.get("learning"), Some(&2));
    }

    #[test]
    fn test_embedding_generation() {
        let mut embedder = SimpleEmbedder::new(100);
        
        // Update vocabulary first
        embedder.update_vocabulary("artificial intelligence");
        embedder.update_vocabulary("machine learning");
        
        let embedding = embedder.embed_text("artificial intelligence").unwrap();
        
        assert_eq!(embedding.len(), 100);
        
        // Check that the vector is normalized (magnitude should be close to 1)
        let magnitude: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_similar_texts_have_similar_embeddings() {
        let mut embedder = SimpleEmbedder::new(100);
        
        // Build vocabulary
        embedder.update_vocabulary("artificial intelligence machine learning");
        embedder.update_vocabulary("deep learning neural networks");
        embedder.update_vocabulary("cooking recipes food");
        
        let ai_embedding1 = embedder.embed_text("artificial intelligence").unwrap();
        let ai_embedding2 = embedder.embed_text("machine learning").unwrap();
        let cooking_embedding = embedder.embed_text("cooking recipes").unwrap();
        
        // Calculate cosine similarity
        let ai_similarity = cosine_similarity(&ai_embedding1, &ai_embedding2);
        let ai_cooking_similarity = cosine_similarity(&ai_embedding1, &cooking_embedding);
        
        // AI-related texts should be more similar to each other than to cooking
        assert!(ai_similarity > ai_cooking_similarity);
    }
}

/// Helper function for cosine similarity (used in tests)
#[cfg(test)]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}
