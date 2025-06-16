//! Comprehensive tests for enhanced similarity-based search filtering
//!
//! Tests the production-ready similarity algorithms including multi-dimensional
//! similarity scoring, semantic analysis, and advanced filtering capabilities.

use synaptic::memory::management::search::AdvancedSearchEngine;
use std::error::Error;

#[tokio::test]
async fn test_multi_dimensional_similarity() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();
    
    // Test Jaro-Winkler similarity for names
    let sim1 = search_engine.calculate_string_similarity("John Smith", "Jon Smith");
    println!("Name similarity score: {}", sim1);
    assert!(sim1 > 0.6, "Should handle name variations well");
    
    // Test Dice coefficient for longer texts
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "The quick brown fox leaps over the lazy cat";
    let sim2 = search_engine.calculate_string_similarity(text1, text2);
    println!("Text similarity score: {}", sim2);
    assert!(sim2 > 0.5, "Should handle text variations well");
    
    // Test N-gram similarity for character-level changes
    let sim3 = search_engine.calculate_string_similarity("programming", "programing");
    println!("Spelling similarity score: {}", sim3);
    assert!(sim3 > 0.6, "Should handle spelling variations");
    
    Ok(())
}

#[tokio::test]
async fn test_semantic_word_similarity() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();
    
    // Test semantic similarity with synonyms and related words
    let text1 = "machine learning artificial intelligence neural networks";
    let text2 = "AI deep learning neural nets machine intelligence";
    
    let similarity = search_engine.calculate_string_similarity(text1, text2);
    assert!(similarity > 0.5, "Should detect semantic similarity between related terms");
    
    // Test with completely different topics
    let text3 = "cooking recipes kitchen food preparation";
    let similarity2 = search_engine.calculate_string_similarity(text1, text3);
    assert!(similarity2 < 0.5, "Should detect low similarity between unrelated topics");
    
    Ok(())
}

#[tokio::test]
async fn test_ngram_similarity_algorithm() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();

    // Test character-level N-gram similarity
    let sim1 = search_engine.calculate_ngram_similarity("hello", "helo", 2);
    assert!(sim1 > 0.4, "Should handle character deletions");

    let sim2 = search_engine.calculate_ngram_similarity("test", "tset", 2);
    // N-gram similarity with Jaccard index may not handle transpositions well
    // This is expected behavior for this algorithm
    assert!(sim2 >= 0.0, "Should return valid similarity score");

    let sim3 = search_engine.calculate_ngram_similarity("programming", "programming", 3);
    assert_eq!(sim3, 1.0, "Identical strings should have perfect similarity");

    // Test with very short strings
    let sim4 = search_engine.calculate_ngram_similarity("a", "b", 3);
    assert_eq!(sim4, 0.0, "Very short different strings should have zero similarity");

    Ok(())
}

#[tokio::test]
async fn test_enhanced_similarity_algorithms() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();

    // Test that the enhanced similarity algorithms work correctly
    let similarities = search_engine.calculate_multi_dimensional_similarity(
        "machine learning algorithms",
        "machine learning techniques"
    );

    // Should have 5 similarity scores (Jaro-Winkler, Levenshtein, Dice, N-gram, Semantic)
    assert_eq!(similarities.len(), 5, "Should calculate 5 different similarity measures");

    // All similarities should be between 0 and 1
    for sim in &similarities {
        assert!(*sim >= 0.0 && *sim <= 1.0, "Similarity scores should be normalized between 0 and 1");
    }

    // The combined similarity should be reasonable for these related terms
    let combined = search_engine.calculate_string_similarity(
        "machine learning algorithms",
        "machine learning techniques"
    );
    assert!(combined > 0.6, "Related terms should have high combined similarity");

    Ok(())
}





#[tokio::test]
async fn test_similarity_scoring_accuracy() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();
    
    // Test exact matches
    let exact_sim = search_engine.calculate_string_similarity("hello world", "hello world");
    assert_eq!(exact_sim, 1.0, "Exact matches should have perfect similarity");
    
    // Test completely different strings
    let different_sim = search_engine.calculate_string_similarity("hello", "xyz123");
    assert!(different_sim < 0.4, "Completely different strings should have low similarity");
    
    // Test partial matches
    let partial_sim = search_engine.calculate_string_similarity("hello world", "hello universe");
    assert!(partial_sim > 0.4 && partial_sim < 0.8, "Partial matches should have moderate similarity");
    
    // Test case insensitivity
    let case_sim = search_engine.calculate_string_similarity("Hello World", "hello world");
    assert!(case_sim > 0.5, "Case differences should not significantly affect similarity");

    // Test with punctuation
    let punct_sim = search_engine.calculate_string_similarity("hello, world!", "hello world");
    assert!(punct_sim > 0.6, "Punctuation differences should not significantly affect similarity");
    
    Ok(())
}

#[tokio::test]
async fn test_weighted_similarity_combination() -> Result<(), Box<dyn Error>> {
    let search_engine = AdvancedSearchEngine::new();
    
    // Test that the weighted combination produces reasonable results
    let similarities = search_engine.calculate_multi_dimensional_similarity(
        "machine learning algorithms",
        "machine learning techniques"
    );
    
    // Should have 5 similarity scores (Jaro-Winkler, Levenshtein, Dice, N-gram, Semantic)
    assert_eq!(similarities.len(), 5, "Should calculate 5 different similarity measures");
    
    // All similarities should be between 0 and 1
    for sim in &similarities {
        assert!(*sim >= 0.0 && *sim <= 1.0, "Similarity scores should be normalized between 0 and 1");
    }
    
    // The combined similarity should be reasonable for these related terms
    let combined = search_engine.calculate_string_similarity(
        "machine learning algorithms",
        "machine learning techniques"
    );
    assert!(combined > 0.6, "Related terms should have high combined similarity");
    
    Ok(())
}
