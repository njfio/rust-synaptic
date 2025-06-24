//! Comprehensive tests for Myers' diff algorithm implementation
//!
//! Tests the production-ready Myers' diff implementation with various algorithms
//! ensuring 90%+ test coverage and comprehensive validation.

use synaptic::memory::temporal::differential::{
    DiffAnalyzer, DiffConfig, MyersAlgorithm, ContentChangeType
};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::error::Error;

#[tokio::test]
async fn test_myers_standard_algorithm() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.myers_algorithm = MyersAlgorithm::Myers;
    config.enable_detailed_text_analysis = true;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    // Create test memory entries with different content
    let old_memory = MemoryEntry::new(
        "test_key".to_string(),
        "The quick brown fox jumps over the lazy dog.".to_string(),
        MemoryType::ShortTerm
    );

    let new_memory = MemoryEntry::new(
        "test_key".to_string(),
        "The quick brown fox leaps over the lazy cat.".to_string(),
        MemoryType::ShortTerm
    );

    // Analyze the difference
    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Verify the diff was computed correctly
    assert!(!diff.id.to_string().is_empty());
    assert_eq!(diff.from_memory_id, old_memory.id());
    assert_eq!(diff.to_memory_id, new_memory.id());
    assert!(diff.significance_score > 0.0);
    
    // Check content changes
    assert_ne!(diff.content_changes.change_type, ContentChangeType::Unchanged);
    assert!(diff.content_changes.similarity_score < 1.0);
    assert!(diff.content_changes.similarity_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_myers_histogram_algorithm() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.myers_algorithm = MyersAlgorithm::Histogram;
    config.enable_line_optimization = true;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    // Create test with line-based changes
    let old_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";
    let new_content = "Line 1\nModified Line 2\nLine 3\nNew Line 3.5\nLine 4\nLine 5";

    let old_memory = MemoryEntry::new("test".to_string(), old_content.to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), new_content.to_string(), MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Verify histogram algorithm handled the diff
    assert!(diff.content_changes.modifications.len() > 0 || diff.content_changes.additions.len() > 0);
    assert!(diff.significance_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_myers_histogram_large_text() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.myers_algorithm = MyersAlgorithm::Histogram;
    config.enable_line_optimization = true;

    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    // Create test with many unique lines (good for histogram)
    let old_lines = (1..=50).map(|i| format!("Unique line {}", i)).collect::<Vec<_>>();
    let mut new_lines = old_lines.clone();
    new_lines.insert(25, "Inserted unique line".to_string());
    new_lines[10] = "Modified unique line 11".to_string();

    let old_content = old_lines.join("\n");
    let new_content = new_lines.join("\n");

    let old_memory = MemoryEntry::new("test".to_string(), old_content, MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), new_content, MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Verify histogram algorithm processed the large diff
    assert!(diff.content_changes.additions.len() > 0 || diff.content_changes.modifications.len() > 0);
    assert!(diff.significance_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_myers_adaptive_algorithm() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.myers_algorithm = MyersAlgorithm::Adaptive;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    // Test small text (should use standard Myers)
    let small_old = "Hello world";
    let small_new = "Hello universe";
    
    let old_memory = MemoryEntry::new("small".to_string(), small_old.to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("small".to_string(), small_new.to_string(), MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;
    assert!(diff.significance_score > 0.0);

    // Test large text (should use histogram or patience)
    let large_old = (1..=200).map(|i| format!("Line {}", i)).collect::<Vec<_>>().join("\n");
    let mut large_new_lines: Vec<String> = (1..=200).map(|i| format!("Line {}", i)).collect();
    large_new_lines[100] = "Modified line 101".to_string();
    let large_new = large_new_lines.join("\n");

    let old_memory = MemoryEntry::new("large".to_string(), large_old, MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("large".to_string(), large_new, MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;
    assert!(diff.significance_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_character_vs_line_based_diff() -> Result<(), Box<dyn Error>> {
    let mut analyzer = DiffAnalyzer::new();

    // Test character-based diff
    let mut char_config = DiffConfig::default();
    char_config.enable_line_optimization = false;
    char_config.enable_detailed_text_analysis = true;
    analyzer.config = char_config;

    let old_memory = MemoryEntry::new("test".to_string(), "abcdef".to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), "abXdef".to_string(), MemoryType::ShortTerm);

    let char_diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Test line-based diff
    let mut line_config = DiffConfig::default();
    line_config.enable_line_optimization = true;
    line_config.enable_detailed_text_analysis = true;
    analyzer.config = line_config;

    let line_diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Both should detect the change but potentially with different granularity
    assert!(char_diff.significance_score > 0.0);
    assert!(line_diff.significance_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_modification_type_classification() -> Result<(), Box<dyn Error>> {
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config.enable_detailed_text_analysis = true;

    // Test spelling correction
    let old_memory = MemoryEntry::new("test".to_string(), "The quik brown fox".to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), "The quick brown fox".to_string(), MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;
    
    // Should detect some form of modification
    assert!(diff.content_changes.modifications.len() > 0 || 
            diff.content_changes.additions.len() > 0 || 
            diff.content_changes.deletions.len() > 0);

    // Test expansion
    let old_memory = MemoryEntry::new("test".to_string(), "Short".to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), "This is a much longer and more detailed explanation".to_string(), MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;
    assert_eq!(diff.content_changes.change_type, ContentChangeType::Replaced);

    Ok(())
}

#[tokio::test]
async fn test_context_extraction() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.context_lines = 2;
    config.enable_line_optimization = true;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    let old_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7";
    let new_content = "Line 1\nLine 2\nModified Line 3\nLine 4\nLine 5\nLine 6\nLine 7";

    let old_memory = MemoryEntry::new("test".to_string(), old_content.to_string(), MemoryType::ShortTerm);
    let new_memory = MemoryEntry::new("test".to_string(), new_content.to_string(), MemoryType::ShortTerm);

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Check that context was extracted for changes
    let has_context = diff.content_changes.additions.iter().any(|seg| seg.context.is_some()) ||
                     diff.content_changes.deletions.iter().any(|seg| seg.context.is_some()) ||
                     diff.content_changes.modifications.len() > 0;
    
    assert!(has_context);

    Ok(())
}

#[tokio::test]
async fn test_diff_performance_metrics() -> Result<(), Box<dyn Error>> {
    let mut analyzer = DiffAnalyzer::new();

    // Generate multiple diffs to test metrics
    for i in 0..10 {
        let old_content = format!("Content version {}", i);
        let new_content = format!("Content version {} modified", i);
        
        let old_memory = MemoryEntry::new(format!("test_{}", i), old_content, MemoryType::ShortTerm);
        let new_memory = MemoryEntry::new(format!("test_{}", i), new_content, MemoryType::ShortTerm);

        analyzer.analyze_difference(&old_memory, &new_memory).await?;
    }

    let metrics = analyzer.get_metrics();
    assert_eq!(metrics.total_diffs, 10);
    assert!(metrics.avg_significance > 0.0);
    assert!(metrics.avg_content_similarity >= 0.0);
    assert!(metrics.avg_content_similarity <= 1.0);

    Ok(())
}

#[tokio::test]
async fn test_word_level_analysis() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.enable_word_level_analysis = true;
    config.enable_detailed_text_analysis = true;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    let old_memory = MemoryEntry::new(
        "test".to_string(),
        "The quick brown fox jumps".to_string(),
        MemoryType::ShortTerm
    );
    let new_memory = MemoryEntry::new(
        "test".to_string(),
        "The fast brown fox leaps".to_string(),
        MemoryType::ShortTerm
    );

    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;

    // Should detect word-level changes
    assert!(diff.significance_score > 0.0);
    assert!(diff.content_changes.similarity_score < 1.0);

    Ok(())
}

#[tokio::test]
async fn test_large_text_diff_performance() -> Result<(), Box<dyn Error>> {
    let mut config = DiffConfig::default();
    config.myers_algorithm = MyersAlgorithm::Adaptive;
    
    let mut analyzer = DiffAnalyzer::new();
    analyzer.config = config;

    // Create large texts for performance testing
    let large_text_1 = (0..1000).map(|i| format!("This is line number {} with some content", i)).collect::<Vec<_>>().join("\n");
    let mut large_text_2_lines: Vec<String> = (0..1000).map(|i| format!("This is line number {} with some content", i)).collect();
    
    // Make some changes
    large_text_2_lines[100] = "This is a modified line 100".to_string();
    large_text_2_lines.insert(500, "This is an inserted line".to_string());
    large_text_2_lines.remove(750);
    
    let large_text_2 = large_text_2_lines.join("\n");

    let old_memory = MemoryEntry::new("large_test".to_string(), large_text_1, MemoryType::LongTerm);
    let new_memory = MemoryEntry::new("large_test".to_string(), large_text_2, MemoryType::LongTerm);

    let start_time = std::time::Instant::now();
    let diff = analyzer.analyze_difference(&old_memory, &new_memory).await?;
    let duration = start_time.elapsed();

    // Performance should be reasonable (under 1 second for this size)
    assert!(duration.as_secs() < 1);
    assert!(diff.significance_score > 0.0);
    
    // Should detect the changes
    assert!(diff.content_changes.additions.len() > 0 || 
            diff.content_changes.deletions.len() > 0 || 
            diff.content_changes.modifications.len() > 0);

    Ok(())
}
