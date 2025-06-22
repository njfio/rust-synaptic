use synaptic::{
    memory::management::{MemorySummarizer, SummaryStrategy},
    memory::storage::memory::MemoryStorage,
    memory::types::{MemoryEntry, MemoryType, MemoryMetadata},
    memory::Storage,
};
use std::sync::Arc;
use chrono::{Utc, Duration};

#[tokio::test]
async fn test_key_points_summary() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    let e1 = MemoryEntry::new("k1".into(), "Learn Rust".into(), MemoryType::ShortTerm);
    let e2 = MemoryEntry::new("k2".into(), "Write tests".into(), MemoryType::ShortTerm);
    storage.store(&e1).await?;
    storage.store(&e2).await?;

    let result = summarizer
        .summarize_memories(&*storage, vec!["k1".into(), "k2".into()], SummaryStrategy::KeyPoints)
        .await?;

    assert!(result.summary_content.contains("Learn Rust"));
    assert!(result.summary_content.contains("Write tests"));
    Ok(())
}

#[tokio::test]
async fn test_chronological_summary() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    let mut e1 = MemoryEntry::new("old".into(), "Old event".into(), MemoryType::ShortTerm);
    let mut meta1 = MemoryMetadata::new();
    meta1.created_at = Utc::now() - Duration::hours(1);
    e1.metadata = meta1;

    let e2 = MemoryEntry::new("new".into(), "New event".into(), MemoryType::ShortTerm);

    storage.store(&e1).await?;
    storage.store(&e2).await?;

    let result = summarizer
        .summarize_memories(&*storage, vec!["old".into(), "new".into()], SummaryStrategy::Chronological)
        .await?;

    let lines: Vec<&str> = result.summary_content.lines().collect();
    // first line is header
    assert!(lines[1].contains("Old event"));
    assert!(lines[2].contains("New event"));
    Ok(())
}

#[tokio::test]
async fn test_storage_backend_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test with memory storage
    let memory_storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Create test memories
    let e1 = MemoryEntry::new("mem1".into(), "First memory content".into(), MemoryType::ShortTerm);
    let e2 = MemoryEntry::new("mem2".into(), "Second memory content".into(), MemoryType::LongTerm);
    let e3 = MemoryEntry::new("mem3".into(), "Third memory content".into(), MemoryType::ShortTerm);

    // Store memories
    memory_storage.store(&e1).await?;
    memory_storage.store(&e2).await?;
    memory_storage.store(&e3).await?;

    // Verify memories are stored
    assert!(memory_storage.retrieve("mem1").await?.is_some());
    assert!(memory_storage.retrieve("mem2").await?.is_some());
    assert!(memory_storage.retrieve("mem3").await?.is_some());

    // Test summarization with stored memories
    let result = summarizer
        .summarize_memories(
            &*memory_storage,
            vec!["mem1".into(), "mem2".into(), "mem3".into()],
            SummaryStrategy::KeyPoints,
        )
        .await?;

    // Verify summary contains content from all memories
    assert!(result.summary_content.contains("First"));
    assert!(result.summary_content.contains("Second"));
    assert!(result.summary_content.contains("Third"));
    assert_eq!(result.source_memory_keys.len(), 3);
    assert!(result.confidence_score >= 0.0);
    assert!(result.compression_ratio > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_data_integrity_across_operations() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Create memories with specific metadata
    let mut e1 = MemoryEntry::new("data1".into(), "Important data entry".into(), MemoryType::LongTerm);
    e1.metadata.importance = 0.9;
    e1.metadata.tags = vec!["important".to_string(), "data".to_string()];

    let mut e2 = MemoryEntry::new("data2".into(), "Related data entry".into(), MemoryType::LongTerm);
    e2.metadata.importance = 0.8;
    e2.metadata.tags = vec!["related".to_string(), "data".to_string()];

    // Store memories
    storage.store(&e1).await?;
    storage.store(&e2).await?;

    // Verify original data integrity
    let retrieved_e1 = storage.retrieve("data1").await?.unwrap();
    let retrieved_e2 = storage.retrieve("data2").await?.unwrap();

    assert_eq!(retrieved_e1.value, "Important data entry");
    assert_eq!(retrieved_e2.value, "Related data entry");
    assert_eq!(retrieved_e1.metadata.importance, 0.9);
    assert_eq!(retrieved_e2.metadata.importance, 0.8);

    // Perform summarization
    let summary_result = summarizer
        .summarize_memories(
            &*storage,
            vec!["data1".into(), "data2".into()],
            SummaryStrategy::ImportanceBased,
        )
        .await?;

    // Verify original memories are unchanged after summarization
    let post_summary_e1 = storage.retrieve("data1").await?.unwrap();
    let post_summary_e2 = storage.retrieve("data2").await?.unwrap();

    assert_eq!(post_summary_e1.value, "Important data entry");
    assert_eq!(post_summary_e2.value, "Related data entry");
    assert_eq!(post_summary_e1.metadata.importance, 0.9);
    assert_eq!(post_summary_e2.metadata.importance, 0.8);

    // Verify summary quality
    assert!(summary_result.confidence_score > 0.0);
    assert!(!summary_result.summary_content.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_empty_memory_list_handling() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Test with empty memory list
    let result = summarizer
        .summarize_memories(&*storage, vec![], SummaryStrategy::KeyPoints)
        .await;

    // Should return an error for empty memory list
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_nonexistent_memory_keys() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Test with non-existent memory keys
    let result = summarizer
        .summarize_memories(
            &*storage,
            vec!["nonexistent1".into(), "nonexistent2".into()],
            SummaryStrategy::KeyPoints,
        )
        .await;

    // Should return an error when no memories are found
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_mixed_existing_nonexistent_keys() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Store one memory
    let e1 = MemoryEntry::new("exists".into(), "This memory exists".into(), MemoryType::ShortTerm);
    storage.store(&e1).await?;

    // Test with mix of existing and non-existing keys
    let result = summarizer
        .summarize_memories(
            &*storage,
            vec!["exists".into(), "nonexistent".into()],
            SummaryStrategy::KeyPoints,
        )
        .await?;

    // Should succeed with the existing memory
    assert!(!result.summary_content.is_empty());
    assert_eq!(result.source_memory_keys.len(), 2); // Original request had 2 keys
    assert!(result.confidence_score >= 0.0);

    Ok(())
}

#[tokio::test]
async fn test_all_summary_strategies() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Create diverse test memories
    let memories = vec![
        MemoryEntry::new("tech1".into(), "Learn Rust programming language".into(), MemoryType::LongTerm),
        MemoryEntry::new("tech2".into(), "Write unit tests for code".into(), MemoryType::ShortTerm),
        MemoryEntry::new("tech3".into(), "Deploy application to production".into(), MemoryType::LongTerm),
    ];

    // Store all memories
    for memory in &memories {
        storage.store(memory).await?;
    }

    let memory_keys = vec!["tech1".into(), "tech2".into(), "tech3".into()];

    // Test all summary strategies
    let strategies = vec![
        SummaryStrategy::KeyPoints,
        SummaryStrategy::Chronological,
        SummaryStrategy::ImportanceBased,
        SummaryStrategy::Consolidation,
        SummaryStrategy::Hierarchical,
        SummaryStrategy::Conceptual,
    ];

    for strategy in strategies {
        let result = summarizer
            .summarize_memories(&*storage, memory_keys.clone(), strategy.clone())
            .await?;

        // Verify each strategy produces valid results
        assert!(!result.summary_content.is_empty(), "Strategy {:?} produced empty summary", strategy);
        assert_eq!(result.strategy, strategy);
        assert_eq!(result.source_memory_keys, memory_keys);
        assert!(result.confidence_score >= 0.0);
        assert!(result.compression_ratio > 0.0);
    }

    Ok(())
}

#[tokio::test]
async fn test_large_memory_set_performance() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Create a large set of memories
    let mut memory_keys = Vec::new();
    for i in 0..100 {
        let memory = MemoryEntry::new(
            format!("large_mem_{}", i),
            format!("This is memory number {} with some content to summarize", i),
            MemoryType::ShortTerm,
        );
        storage.store(&memory).await?;
        memory_keys.push(format!("large_mem_{}", i));
    }

    let start_time = std::time::Instant::now();

    // Test summarization performance
    let result = summarizer
        .summarize_memories(&*storage, memory_keys.clone(), SummaryStrategy::KeyPoints)
        .await?;

    let duration = start_time.elapsed();

    // Verify performance is reasonable (should complete within 5 seconds)
    assert!(duration.as_secs() < 5, "Summarization took too long: {:?}", duration);

    // Verify result quality
    assert!(!result.summary_content.is_empty());
    assert_eq!(result.source_memory_keys.len(), 100);
    assert!(result.compression_ratio > 1.0); // Should compress the content

    Ok(())
}

#[tokio::test]
async fn test_concurrent_summarization() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());

    // Create test memories
    for i in 0..20 {
        let memory = MemoryEntry::new(
            format!("concurrent_mem_{}", i),
            format!("Concurrent memory content {}", i),
            MemoryType::ShortTerm,
        );
        storage.store(&memory).await?;
    }

    // Create multiple summarization tasks
    let mut tasks = Vec::new();
    for batch in 0..4 {
        let storage_clone = storage.clone();
        let memory_keys: Vec<String> = (batch * 5..(batch + 1) * 5)
            .map(|i| format!("concurrent_mem_{}", i))
            .collect();

        let task = tokio::spawn(async move {
            let mut summarizer = MemorySummarizer::new();
            summarizer
                .summarize_memories(&*storage_clone, memory_keys, SummaryStrategy::KeyPoints)
                .await
        });
        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::try_join_all(tasks).await?;

    // Verify all summarizations succeeded
    for result in results {
        let summary = result?;
        assert!(!summary.summary_content.is_empty());
        assert_eq!(summary.source_memory_keys.len(), 5);
        assert!(summary.confidence_score >= 0.0);
    }

    Ok(())
}

#[tokio::test]
async fn test_storage_consistency_during_summarization() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new());
    let mut summarizer = MemorySummarizer::new();

    // Create initial memories
    let initial_memories = vec![
        MemoryEntry::new("consistency1".into(), "Initial content 1".into(), MemoryType::ShortTerm),
        MemoryEntry::new("consistency2".into(), "Initial content 2".into(), MemoryType::ShortTerm),
    ];

    for memory in &initial_memories {
        storage.store(memory).await?;
    }

    // Start summarization
    let memory_keys = vec!["consistency1".into(), "consistency2".into()];
    let summary_result = summarizer
        .summarize_memories(&*storage, memory_keys.clone(), SummaryStrategy::KeyPoints)
        .await?;

    // Verify storage state is consistent after summarization
    let retrieved_mem1 = storage.retrieve("consistency1").await?.unwrap();
    let retrieved_mem2 = storage.retrieve("consistency2").await?.unwrap();

    assert_eq!(retrieved_mem1.value, "Initial content 1");
    assert_eq!(retrieved_mem2.value, "Initial content 2");

    // Verify summary was created successfully
    assert!(!summary_result.summary_content.is_empty());
    assert!(summary_result.summary_content.contains("Initial content"));

    Ok(())
}

