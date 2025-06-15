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

