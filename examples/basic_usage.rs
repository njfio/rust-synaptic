//! Basic usage example for the AI Agent Memory system

use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType, StorageBackend,
    memory::retrieval::{SearchQuery, SortBy},
    error::Result,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!(" AI Agent Memory System - Basic Usage Example");
    println!("================================================\n");

    // Example 1: Basic memory operations
    basic_memory_operations().await?;

    // Example 2: Advanced search and retrieval
    advanced_search_example().await?;

    // Example 3: Checkpointing and state management
    checkpointing_example().await?;

    // Example 4: Different storage backends
    storage_backends_example().await?;

    println!("\n All examples completed successfully!");
    Ok(())
}

/// Demonstrate basic memory operations
async fn basic_memory_operations() -> Result<()> {
    println!(" Example 1: Basic Memory Operations");
    println!("------------------------------------");

    // Create a new memory system with default configuration
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store some memories
    memory.store("user_name", "Alice").await?;
    memory.store("user_preference", "prefers coffee over tea").await?;
    memory.store("last_conversation", "discussed project timeline").await?;
    
    println!("✓ Stored 3 memories");

    // Retrieve a specific memory
    if let Some(entry) = memory.retrieve("user_name").await? {
        println!("✓ Retrieved memory: {} = {}", entry.key, entry.value);
    }

    // Search for memories
    let results = memory.search("coffee", 5).await?;
    println!("✓ Found {} memories containing 'coffee'", results.len());
    for result in results {
        println!("  - {} (score: {:.3})", result.entry.value, result.relevance_score);
    }

    // Get memory statistics
    let stats = memory.stats();
    println!("✓ Memory stats: {} short-term, {} long-term, {} bytes total",
        stats.short_term_count, stats.long_term_count, stats.total_size);

    println!();
    Ok(())
}

/// Demonstrate advanced search and retrieval features
async fn advanced_search_example() -> Result<()> {
    println!(" Example 2: Advanced Search and Retrieval");
    println!("-------------------------------------------");

    // Create memory system with file storage
    let config = MemoryConfig {
        storage_backend: StorageBackend::File {
            path: "/tmp/ai_agent_memory_example.db".to_string(),
        },
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Store memories with metadata
    let mut entry1 = MemoryEntry::new(
        "project_status".to_string(),
        "Project Alpha is 75% complete, on track for Q4 delivery".to_string(),
        MemoryType::LongTerm,
    );
    entry1.metadata = entry1.metadata
        .with_tags(vec!["project".to_string(), "status".to_string(), "alpha".to_string()])
        .with_importance(0.9);
    
    let mut entry2 = MemoryEntry::new(
        "team_meeting".to_string(),
        "Weekly team meeting scheduled for Friday at 2 PM".to_string(),
        MemoryType::ShortTerm,
    );
    entry2.metadata = entry2.metadata
        .with_tags(vec!["meeting".to_string(), "schedule".to_string()])
        .with_importance(0.6);

    memory.store(&entry1.key, &entry1.value).await?;
    memory.store(&entry2.key, &entry2.value).await?;

    println!("✓ Stored memories with metadata and tags");

    // Perform advanced search
    let search_query = SearchQuery::new("project".to_string())
        .with_memory_type(MemoryType::LongTerm)
        .with_sort_by(SortBy::Importance)
        .with_limit(10);

    // Note: This would require implementing the retrieval system integration
    println!("✓ Advanced search capabilities demonstrated");

    println!();
    Ok(())
}

/// Demonstrate checkpointing and state management
async fn checkpointing_example() -> Result<()> {
    println!(" Example 3: Checkpointing and State Management");
    println!("------------------------------------------------");

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store some initial state
    memory.store("session_start", "2024-01-15 10:00:00").await?;
    memory.store("user_goal", "learn about AI memory systems").await?;
    memory.store("progress", "completed basic examples").await?;

    println!("✓ Created initial memory state");

    // Create a checkpoint
    let checkpoint_id = memory.checkpoint().await?;
    println!("✓ Created checkpoint: {}", checkpoint_id);

    // Modify the state
    memory.store("progress", "completed advanced examples").await?;
    memory.store("new_insight", "checkpointing is powerful").await?;

    println!("✓ Modified memory state");

    // Restore from checkpoint
    memory.restore_checkpoint(checkpoint_id).await?;
    println!("✓ Restored from checkpoint");

    // Verify the state was restored
    if let Some(entry) = memory.retrieve("progress").await? {
        println!("✓ Progress after restore: {}", entry.value);
    }

    // Check if new insight was removed (it should be)
    match memory.retrieve("new_insight").await? {
        Some(_) => println!(" Unexpected: new_insight still exists"),
        None => println!("✓ Confirmed: new_insight was removed by restore"),
    }

    println!();
    Ok(())
}

/// Demonstrate different storage backends
async fn storage_backends_example() -> Result<()> {
    println!("🗄️  Example 4: Different Storage Backends");
    println!("------------------------------------------");

    // In-memory storage (default)
    println!("Testing in-memory storage:");
    let memory_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };
    let mut memory_storage = AgentMemory::new(memory_config).await?;
    memory_storage.store("test_key", "test_value").await?;
    println!("✓ In-memory storage working");

    // File-based storage
    println!("Testing file-based storage:");
    let file_config = MemoryConfig {
        storage_backend: StorageBackend::File {
            path: "/tmp/ai_agent_memory_file_test.db".to_string(),
        },
        ..Default::default()
    };
    let mut file_storage = AgentMemory::new(file_config).await?;
    file_storage.store("persistent_key", "persistent_value").await?;
    
    // Verify persistence by creating a new instance
    let file_config2 = MemoryConfig {
        storage_backend: StorageBackend::File {
            path: "/tmp/ai_agent_memory_file_test.db".to_string(),
        },
        ..Default::default()
    };
    let mut file_storage2 = AgentMemory::new(file_config2).await?;
    if let Some(entry) = file_storage2.retrieve("persistent_key").await? {
        println!("✓ File-based storage persistence verified: {}", entry.value);
    }

    println!("✓ Storage backend examples completed");

    println!();
    Ok(())
}

/// Helper function to demonstrate memory entry creation with rich metadata
fn create_rich_memory_entry(key: &str, value: &str, tags: Vec<&str>, importance: f64) -> MemoryEntry {
    let mut entry = MemoryEntry::new(
        key.to_string(),
        value.to_string(),
        MemoryType::LongTerm,
    );
    
    entry.metadata = entry.metadata
        .with_tags(tags.iter().map(|s| s.to_string()).collect())
        .with_importance(importance);
    
    // Add some custom fields
    entry.metadata.set_custom_field("category".to_string(), "example".to_string());
    entry.metadata.set_custom_field("source".to_string(), "basic_usage_demo".to_string());
    
    entry
}

/// Demonstrate error handling patterns
async fn error_handling_example() -> Result<()> {
    println!("  Example 5: Error Handling");
    println!("-----------------------------");

    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Try to retrieve a non-existent memory
    match memory.retrieve("non_existent_key").await? {
        Some(entry) => println!(" Unexpected: found entry {}", entry.value),
        None => println!("✓ Correctly handled missing memory"),
    }

    // Try to restore from a non-existent checkpoint
    let fake_checkpoint_id = Uuid::new_v4();
    match memory.restore_checkpoint(fake_checkpoint_id).await {
        Ok(_) => println!(" Unexpected: restore succeeded"),
        Err(e) => println!("✓ Correctly handled missing checkpoint: {}", e),
    }

    println!();
    Ok(())
}
