//! Comprehensive tests for context assembly functionality
//!
//! Tests cover:
//! - ContextBuilder usage
//! - AgentContext creation
//! - LLM formatting (OpenAI, Anthropic, Markdown, etc.)
//! - Token counting and truncation
//! - Temporal filtering
//! - Graph-based context retrieval
//! - Deduplication

use synaptic::memory::context::{
    AgentContext, ContextBuilder, LlmFormat, MemorySection, RelationshipFilter, TemporalRange,
};
use synaptic::memory::knowledge_graph::{MemoryKnowledgeGraph, RelationshipType};
use synaptic::memory::storage::memory::MemoryStorage;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::AgentMemory;
use chrono::{Duration, Utc};
use std::sync::Arc;

// Helper to create test memory entries
fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
    MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::ShortTerm)
}

// Helper to create test memory with specific timestamp
fn create_test_entry_with_time(key: &str, content: &str, hours_ago: i64) -> MemoryEntry {
    let mut entry = MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::ShortTerm);
    entry.created_at = Utc::now() - Duration::hours(hours_ago);
    entry.accessed_at = entry.created_at;
    entry
}

#[tokio::test]
async fn test_context_builder_basic() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Add test memories
    let entry1 = create_test_entry("rust1", "Rust programming language basics");
    let entry2 = create_test_entry("rust2", "Advanced Rust async programming");
    let entry3 = create_test_entry("python1", "Python scripting tutorials");

    memory.store(entry1).await.expect("Failed to store");
    memory.store(entry2).await.expect("Failed to store");
    memory.store(entry3).await.expect("Failed to store");

    // Build context
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("Rust programming", 2)
        .expect("Failed to set query")
        .build()
        .await
        .expect("Failed to build context");

    // Should have core memories
    assert!(!context.core_memories.is_empty());
    assert!(context.core_memories.len() <= 2);
}

#[tokio::test]
async fn test_context_with_graph_neighbors() {
    let storage = Arc::new(MemoryStorage::new());
    let graph = Arc::new(MemoryKnowledgeGraph::new());
    let mut memory = AgentMemory::new(storage.clone());
    memory.knowledge_graph = Some(graph.clone());

    // Add memories
    let entry1 = create_test_entry("concept1", "Main concept");
    let entry2 = create_test_entry("concept2", "Related concept");

    memory.store(entry1).await.expect("Failed to store");
    memory.store(entry2).await.expect("Failed to store");

    // Add graph relationship
    graph
        .add_relationship("concept1", "concept2", RelationshipType::Similar, 0.8)
        .expect("Failed to add relationship");

    // Build context with graph neighbors
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("Main concept", 1)
        .expect("Failed to set query")
        .with_graph_neighbors(1, None)
        .expect("Failed to set graph neighbors")
        .build()
        .await
        .expect("Failed to build context");

    // Should have both core and related memories
    assert!(!context.core_memories.is_empty());
    // Graph relationships should be included
    assert!(context.core_memories.len() + context.related_memories.len() >= 1);
}

#[tokio::test]
async fn test_context_with_temporal_slice() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Add memories at different times
    let entry1 = create_test_entry_with_time("recent", "Recent memory", 1);
    let entry2 = create_test_entry_with_time("old", "Old memory", 25);

    memory.store(entry1).await.expect("Failed to store");
    memory.store(entry2).await.expect("Failed to store");

    // Build context with temporal filter (last 24 hours)
    let context = ContextBuilder::new(&memory)
        .with_temporal_slice(TemporalRange::Hours(24))
        .expect("Failed to set temporal range")
        .build()
        .await
        .expect("Failed to build context");

    // Should only have recent memory
    assert!(!context.temporal_memories.is_empty());

    // Check that old memory is not included
    let has_old = context
        .temporal_memories
        .iter()
        .any(|m| m.key == "old");
    assert!(!has_old);
}

#[tokio::test]
async fn test_context_with_summaries() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Add memories
    let entry = create_test_entry("test", "Test content");
    memory.store(entry).await.expect("Failed to store");

    // Build context with summaries
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("test", 5)
        .expect("Failed to set query")
        .with_summaries()
        .expect("Failed to enable summaries")
        .build()
        .await
        .expect("Failed to build context");

    // Should have summary
    assert!(!context.summary.is_empty());
    assert!(context.summary.contains("memories"));
}

#[tokio::test]
async fn test_context_deduplication() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Add a memory
    let entry = create_test_entry("duplicate", "Duplicate content");
    memory.store(entry).await.expect("Failed to store");

    // Build context (the same memory might appear in both core and temporal)
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("Duplicate", 5)
        .expect("Failed to set query")
        .with_temporal_slice(TemporalRange::Hours(24))
        .expect("Failed to set temporal range")
        .with_deduplication(true)
        .build()
        .await
        .expect("Failed to build context");

    // Count total unique keys
    let mut all_keys = Vec::new();
    all_keys.extend(context.core_memories.iter().map(|m| &m.key));
    all_keys.extend(context.related_memories.iter().map(|m| &m.key));
    all_keys.extend(context.temporal_memories.iter().map(|m| &m.key));

    // Should have no duplicates
    let mut unique_keys = all_keys.clone();
    unique_keys.sort();
    unique_keys.dedup();

    assert_eq!(all_keys.len(), unique_keys.len());
}

#[tokio::test]
async fn test_context_format_openai() {
    let mut context = AgentContext::new();
    context.core_memories.push(create_test_entry("key1", "Memory content 1"));
    context.core_memories.push(create_test_entry("key2", "Memory content 2"));
    context.summary = "Test summary".to_string();

    let formatted = context
        .format(LlmFormat::OpenAI, 1000)
        .expect("Failed to format");

    assert!(formatted.contains("Memory Context"));
    assert!(formatted.contains("Test summary"));
    assert!(formatted.contains("Memory content 1"));
    assert!(formatted.contains("Core Memories"));
}

#[tokio::test]
async fn test_context_format_anthropic() {
    let mut context = AgentContext::new();
    context.core_memories.push(create_test_entry("key1", "Memory content"));

    let formatted = context
        .format(LlmFormat::Anthropic, 1000)
        .expect("Failed to format");

    assert!(formatted.contains("<memory>"));
    assert!(formatted.contains("Memory content"));
    assert!(formatted.contains("</memory>"));
}

#[tokio::test]
async fn test_context_format_markdown() {
    let mut context = AgentContext::new();
    context.core_memories.push(create_test_entry("key1", "Memory content"));
    context.summary = "Test summary".to_string();

    let formatted = context
        .format(LlmFormat::Markdown, 1000)
        .expect("Failed to format");

    assert!(formatted.contains("# Memory Context"));
    assert!(formatted.contains("## Summary"));
    assert!(formatted.contains("## Core Memories"));
    assert!(formatted.contains("Memory content"));
}

#[tokio::test]
async fn test_context_format_json() {
    let mut context = AgentContext::new();
    context.core_memories.push(create_test_entry("key1", "Memory content"));

    let formatted = context
        .format(LlmFormat::Json, 1000)
        .expect("Failed to format");

    // Should be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&formatted).expect("Failed to parse JSON");
    assert!(parsed["core_memories"].is_array());
}

#[tokio::test]
async fn test_context_format_plain_text() {
    let mut context = AgentContext::new();
    context.core_memories.push(create_test_entry("key1", "Memory content"));

    let formatted = context
        .format(LlmFormat::PlainText, 1000)
        .expect("Failed to format");

    assert!(formatted.contains("Memory content"));
    // Should not have markdown or XML formatting
    assert!(!formatted.contains("#"));
    assert!(!formatted.contains("<"));
}

#[tokio::test]
async fn test_token_estimation() {
    let mut context = AgentContext::new();

    // Add memories with known content
    context.core_memories.push(create_test_entry("key1", "a".repeat(100)));
    context.core_memories.push(create_test_entry("key2", "b".repeat(200)));

    context.update_token_estimate();

    // Token estimate should be reasonable (roughly chars/4)
    assert!(context.estimated_tokens > 0);
    assert!(context.estimated_tokens < 1000); // Should be less than 1000 for 300 chars
}

#[tokio::test]
async fn test_token_truncation() {
    let mut context = AgentContext::new();
    context
        .core_memories
        .push(create_test_entry("key1", &"a".repeat(10000)));

    // Format with small token limit
    let formatted = context
        .format(LlmFormat::PlainText, 100)
        .expect("Failed to format");

    // Should be truncated
    assert!(formatted.contains("truncated"));
    // Should be much shorter than original
    assert!(formatted.len() < 10000);
}

#[tokio::test]
async fn test_memory_section() {
    let mut section = MemorySection::new("Test Section".to_string(), "Description".to_string());
    section.add_memory(create_test_entry("key1", "Content 1"));
    section.add_memory(create_test_entry("key2", "Content 2"));
    section = section.with_importance(0.8);

    assert_eq!(section.memories.len(), 2);
    assert_eq!(section.importance, 0.8);
    assert!(section.estimate_tokens() > 0);
}

#[tokio::test]
async fn test_temporal_range_custom() {
    let start = Utc::now() - Duration::days(7);
    let end = Utc::now();
    let range = TemporalRange::Range { start, end };

    let within = Utc::now() - Duration::days(3);
    let before = Utc::now() - Duration::days(10);

    assert!(range.contains(within));
    assert!(!range.contains(before));
}

#[tokio::test]
async fn test_temporal_range_since() {
    let since = Utc::now() - Duration::days(7);
    let range = TemporalRange::Since(since);

    let after = Utc::now() - Duration::days(3);
    let before = Utc::now() - Duration::days(10);

    assert!(range.contains(after));
    assert!(!range.contains(before));
}

#[tokio::test]
async fn test_relationship_filter_all() {
    let filter = RelationshipFilter::All;
    assert!(filter.matches(&RelationshipType::Similar));
    assert!(filter.matches(&RelationshipType::Causal));
    assert!(filter.matches(&RelationshipType::Prerequisite));
}

#[tokio::test]
async fn test_relationship_filter_include() {
    let filter = RelationshipFilter::Include(vec![
        RelationshipType::Similar,
        RelationshipType::Causal,
    ]);

    assert!(filter.matches(&RelationshipType::Similar));
    assert!(filter.matches(&RelationshipType::Causal));
    assert!(!filter.matches(&RelationshipType::Prerequisite));
}

#[tokio::test]
async fn test_relationship_filter_exclude() {
    let filter = RelationshipFilter::Exclude(vec![RelationshipType::Temporal]);

    assert!(filter.matches(&RelationshipType::Similar));
    assert!(!filter.matches(&RelationshipType::Temporal));
}

#[tokio::test]
async fn test_context_metadata() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    let entry = create_test_entry("test", "Test content");
    memory.store(entry).await.expect("Failed to store");

    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("test", 5)
        .expect("Failed to set query")
        .build()
        .await
        .expect("Failed to build context");

    // Should have metadata
    assert!(context.metadata.contains_key("generated_at"));
    assert!(context.metadata.contains_key("query"));
    assert_eq!(context.metadata.get("query").unwrap(), "test");
}

#[tokio::test]
async fn test_complex_context_assembly() {
    let storage = Arc::new(MemoryStorage::new());
    let graph = Arc::new(MemoryKnowledgeGraph::new());
    let mut memory = AgentMemory::new(storage.clone());
    memory.knowledge_graph = Some(graph.clone());

    // Add multiple memories at different times
    let entries = vec![
        create_test_entry_with_time("rust1", "Rust programming basics", 1),
        create_test_entry_with_time("rust2", "Advanced Rust patterns", 2),
        create_test_entry_with_time("rust3", "Rust async programming", 3),
        create_test_entry_with_time("python1", "Python basics", 1),
        create_test_entry_with_time("old", "Old content", 48),
    ];

    for entry in entries {
        memory.store(entry).await.expect("Failed to store");
    }

    // Add graph relationships
    graph
        .add_relationship("rust1", "rust2", RelationshipType::Prerequisite, 0.9)
        .expect("Failed to add relationship");
    graph
        .add_relationship("rust2", "rust3", RelationshipType::Similar, 0.7)
        .expect("Failed to add relationship");

    // Build comprehensive context
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("Rust programming", 3)
        .expect("Failed to set query")
        .with_graph_neighbors(1, Some(RelationshipFilter::All))
        .expect("Failed to set graph neighbors")
        .with_temporal_slice(TemporalRange::Hours(24))
        .expect("Failed to set temporal range")
        .with_summaries()
        .expect("Failed to enable summaries")
        .with_deduplication(true)
        .build()
        .await
        .expect("Failed to build context");

    // Should have all components
    assert!(!context.core_memories.is_empty());
    assert!(!context.summary.is_empty());
    assert!(context.estimated_tokens > 0);

    // Old memory should not be in temporal memories
    let has_old = context.temporal_memories.iter().any(|m| m.key == "old");
    assert!(!has_old);

    // Format for different providers
    let openai = context.format(LlmFormat::OpenAI, 2000).expect("Failed to format OpenAI");
    let anthropic = context.format(LlmFormat::Anthropic, 2000).expect("Failed to format Anthropic");
    let markdown = context.format(LlmFormat::Markdown, 2000).expect("Failed to format Markdown");

    assert!(!openai.is_empty());
    assert!(!anthropic.is_empty());
    assert!(!markdown.is_empty());
}

#[tokio::test]
async fn test_empty_context() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Build context with no memories
    let context = ContextBuilder::new(&memory)
        .with_summaries()
        .expect("Failed to enable summaries")
        .build()
        .await
        .expect("Failed to build context");

    assert!(context.core_memories.is_empty());
    assert!(context.related_memories.is_empty());
    assert!(context.temporal_memories.is_empty());
}

#[tokio::test]
async fn test_large_context_assembly() {
    let storage = Arc::new(MemoryStorage::new());
    let memory = AgentMemory::new(storage.clone());

    // Add many memories
    for i in 0..100 {
        let entry = create_test_entry(&format!("key{}", i), &format!("Content number {}", i));
        memory.store(entry).await.expect("Failed to store");
    }

    // Build context with limits
    let context = ContextBuilder::new(&memory)
        .with_relevant_memories("Content", 10)
        .expect("Failed to set query")
        .with_temporal_slice(TemporalRange::Hours(24))
        .expect("Failed to set temporal range")
        .build()
        .await
        .expect("Failed to build context");

    // Should respect limits
    assert!(context.core_memories.len() <= 10);
    assert!(context.temporal_memories.len() <= 20); // Default limit in implementation

    // Should estimate reasonable token count
    context.update_token_estimate();
    assert!(context.estimated_tokens > 0);
}
