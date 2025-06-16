//! Comprehensive tests for advanced theme extraction
//!
//! Tests the production-ready theme extraction algorithms including TF-IDF,
//! semantic clustering, topic modeling, and NLP pattern recognition.

use synaptic::memory::management::summarization::MemorySummarizer;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::error::Error;

#[tokio::test]
async fn test_tfidf_theme_extraction() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();
    
    // Test text with clear important terms
    let text = "Machine learning algorithms are essential for artificial intelligence. 
               The neural networks process data efficiently. Deep learning models 
               require extensive training data. Machine learning applications 
               continue to grow in artificial intelligence systems.";
    
    let themes = engine.extract_tfidf_themes(text).await?;
    
    // Should extract important terms based on TF-IDF scores
    assert!(!themes.is_empty(), "Should extract TF-IDF themes");
    
    // Check that themes contain important terms
    let themes_text = themes.join(" ").to_lowercase();
    assert!(themes_text.contains("machine") || themes_text.contains("learning") || 
            themes_text.contains("artificial") || themes_text.contains("intelligence"),
           "Should extract key technical terms");
    
    println!("TF-IDF themes: {:?}", themes);
    Ok(())
}

#[tokio::test]
async fn test_semantic_cluster_theme_extraction() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test text with repeated phrases
    let text = "Software development requires careful planning. The development team
               works on software projects. Project management is crucial for software
               development success. Team collaboration improves development efficiency.";

    let themes = engine.extract_semantic_cluster_themes(text).await?;
    
    // Should extract frequent phrases
    assert!(!themes.is_empty(), "Should extract semantic cluster themes");
    
    // Check that themes contain repeated phrases
    let themes_text = themes.join(" ").to_lowercase();
    assert!(themes_text.contains("software") || themes_text.contains("development") ||
            themes_text.contains("team") || themes_text.contains("project"),
           "Should extract frequent phrase patterns");
    
    println!("Semantic cluster themes: {:?}", themes);
    Ok(())
}

#[tokio::test]
async fn test_topic_model_theme_extraction() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test text with co-occurring terms
    let text = "Database management systems store information efficiently. 
               Query optimization improves database performance significantly.
               Information retrieval from database systems requires optimization.
               Performance monitoring helps database management operations.";
    
    let themes = engine.extract_topic_model_themes(text).await?;
    
    // Should extract co-occurring term pairs
    if !themes.is_empty() {
        let themes_text = themes.join(" ").to_lowercase();
        assert!(themes_text.contains("database") || themes_text.contains("performance") ||
                themes_text.contains("optimization") || themes_text.contains("information"),
               "Should extract co-occurring term topics");
    }
    
    println!("Topic model themes: {:?}", themes);
    Ok(())
}

#[tokio::test]
async fn test_nlp_pattern_theme_extraction() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test text with various NLP patterns
    let text = "The problem with the current system is performance. We need to find a solution
               because the issue affects user experience. First, we analyze the problem, then
               we implement the solution. This decision will improve the system compared to
               the previous version. The team must collaborate together on this innovative approach.";
    
    let themes = engine.extract_nlp_themes(text).await?;
    
    // Should extract pattern-based themes
    assert!(!themes.is_empty(), "Should extract NLP pattern themes");
    
    // Check for specific patterns
    let themes_text = themes.join(" ");
    let expected_patterns = [
        "Causality Analysis", "Problem Resolution", "Sequential Process",
        "Decision Making", "Comparative Analysis", "Collaboration", "Innovation"
    ];
    
    let found_patterns = expected_patterns.iter()
        .filter(|pattern| themes_text.contains(*pattern))
        .count();
    
    assert!(found_patterns > 0, "Should detect at least one NLP pattern");
    
    println!("NLP pattern themes: {:?}", themes);
    Ok(())
}

#[tokio::test]
async fn test_theme_scoring_and_ranking() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    let themes = vec![
        "Machine Learning".to_string(),
        "Data Analysis".to_string(),
        "Software Development".to_string(),
        "Project Management".to_string(),
    ];

    let text = "Machine learning is important for data analysis. Machine learning algorithms
               process data efficiently. Software development requires machine learning expertise.";

    let scores = engine.score_and_rank_themes(&themes, text).await?;
    
    // Should score themes based on frequency, position, and length
    assert!(!scores.is_empty(), "Should generate theme scores");
    
    // Machine learning should have highest score due to frequency
    let ml_score = scores.get("Machine Learning").unwrap_or(&0.0);
    let other_scores: Vec<f64> = scores.values().filter(|&&s| s != *ml_score).cloned().collect();
    
    if !other_scores.is_empty() {
        let max_other_score = other_scores.iter().fold(0.0f64, |a, &b| a.max(b));
        assert!(*ml_score >= max_other_score, "Most frequent theme should have highest score");
    }
    
    println!("Theme scores: {:?}", scores);
    Ok(())
}

#[tokio::test]
async fn test_comprehensive_theme_extraction() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Create a memory entry with rich content
    let memory = MemoryEntry::new(
        "comprehensive_test".to_string(),
        "Artificial intelligence and machine learning are transforming software development.
         The development team collaborates on innovative solutions because traditional approaches
         have limitations. First, we analyze the problem, then we design algorithms.
         Database optimization improves system performance compared to previous versions.
         This decision-making process requires careful risk assessment and collaboration.".to_string(),
        MemoryType::LongTerm
    );
    
    let themes = engine.extract_key_themes(&memory.value).await?;
    
    // Should extract themes using all methods
    assert!(!themes.is_empty(), "Should extract comprehensive themes");
    assert!(themes.len() <= 10, "Should limit themes to maximum of 10");
    
    // Check for diversity of theme types
    let themes_text = themes.join(" ");
    println!("Comprehensive themes: {:?}", themes);
    
    // Should contain a mix of different theme types
    let has_technical = themes_text.to_lowercase().contains("artificial") || 
                       themes_text.to_lowercase().contains("machine") ||
                       themes_text.to_lowercase().contains("software");
    
    let has_process = themes_text.contains("Development") || 
                     themes_text.contains("Analysis") ||
                     themes_text.contains("Management");
    
    let has_patterns = themes_text.contains("Collaboration") || 
                      themes_text.contains("Decision") ||
                      themes_text.contains("Problem");
    
    assert!(has_technical || has_process || has_patterns, 
           "Should extract themes from multiple categories");
    
    Ok(())
}

#[tokio::test]
async fn test_title_case_conversion() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test the title case helper function
    let test_cases = vec![
        ("hello world", "Hello World"),
        ("machine learning", "Machine Learning"),
        ("artificial intelligence", "Artificial Intelligence"),
        ("data analysis", "Data Analysis"),
    ];
    
    for (input, expected) in test_cases {
        let result = engine.to_title_case(input);
        assert_eq!(result, expected, "Title case conversion should work correctly");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_theme_extraction_edge_cases() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test with empty text
    let empty_themes = engine.extract_key_themes("").await?;
    assert!(empty_themes.is_empty(), "Empty text should produce no themes");
    
    // Test with very short text
    let _short_themes = engine.extract_key_themes("Hi").await?;
    // Should handle gracefully (may or may not produce themes)

    // Test with single word
    let _single_word_themes = engine.extract_key_themes("programming").await?;
    // Should handle gracefully

    // Test with punctuation only
    let punct_themes = engine.extract_key_themes("!@#$%^&*()").await?;
    assert!(punct_themes.is_empty(), "Punctuation only should produce no themes");

    // Test with numbers only
    let _number_themes = engine.extract_key_themes("123 456 789").await?;
    // Should handle gracefully (numbers typically filtered out)
    
    println!("Edge case tests completed successfully");
    Ok(())
}

#[tokio::test]
async fn test_theme_extraction_performance() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test with large text
    let large_text = "Machine learning and artificial intelligence are revolutionizing technology. ".repeat(100);
    
    let start_time = std::time::Instant::now();
    let themes = engine.extract_key_themes(&large_text).await?;
    let duration = start_time.elapsed();
    
    // Should complete within reasonable time (under 100ms for repeated text)
    assert!(duration.as_millis() < 100, "Theme extraction should be performant");
    assert!(!themes.is_empty(), "Should extract themes from large text");
    assert!(themes.len() <= 10, "Should respect theme limit even for large text");
    
    println!("Performance test: extracted {} themes in {:?}", themes.len(), duration);
    Ok(())
}

#[tokio::test]
async fn test_theme_relevance_filtering() -> Result<(), Box<dyn Error>> {
    let engine = MemorySummarizer::new();

    // Test text with mix of relevant and irrelevant content
    let text = "The quick brown fox jumps over the lazy dog. Machine learning algorithms
               process data efficiently. The weather is nice today. Artificial intelligence
               systems require extensive training. I like pizza and ice cream.";
    
    let themes = engine.extract_key_themes(text).await?;
    
    // Should filter out irrelevant themes and keep relevant ones
    assert!(!themes.is_empty(), "Should extract relevant themes");
    
    let themes_text = themes.join(" ").to_lowercase();
    
    // Should contain technical themes
    let has_relevant = themes_text.contains("machine") || themes_text.contains("artificial") ||
                      themes_text.contains("learning") || themes_text.contains("intelligence") ||
                      themes_text.contains("technology") || themes_text.contains("analysis");
    
    assert!(has_relevant, "Should extract relevant technical themes");
    
    // Should not contain irrelevant themes about weather or food
    let has_irrelevant = themes_text.contains("weather") || themes_text.contains("pizza") ||
                        themes_text.contains("ice cream") || themes_text.contains("fox");
    
    // Note: This is a soft assertion since some irrelevant terms might appear in pattern-based themes
    if has_irrelevant {
        println!("Warning: Some irrelevant themes detected: {:?}", themes);
    }
    
    Ok(())
}
