//! Comprehensive example demonstrating the SynapticMemory implementation
//! of the MemoryOperations trait.
//!
//! This example shows:
//! - Basic store, retrieve, update, and search operations
//! - Builder pattern for configuration
//! - Integration with knowledge graphs and analytics
//! - Different memory types (short-term vs long-term)
//! - Concurrent operations
//! - Best practices for production use

use synaptic::memory::operations::{SynapticMemory, SynapticMemoryBuilder};
use synaptic::memory::{MemoryOperations, MemoryEntry, MemoryType};
use synaptic::memory::storage::StorageBackend;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for observability
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Synaptic Memory Operations Example ===\n");

    // Example 1: Basic usage with defaults
    println!("1. Basic Usage with Defaults");
    basic_usage().await?;

    // Example 2: Using the builder pattern
    println!("\n2. Using Builder Pattern for Configuration");
    builder_pattern_usage().await?;

    // Example 3: Store and retrieve operations
    println!("\n3. Store and Retrieve Operations");
    store_and_retrieve().await?;

    // Example 4: Update operations
    println!("\n4. Update Operations");
    update_operations().await?;

    // Example 5: Search operations
    println!("\n5. Search Operations");
    search_operations().await?;

    // Example 6: Working with different memory types
    println!("\n6. Different Memory Types");
    memory_types_example().await?;

    // Example 7: Concurrent operations
    println!("\n7. Concurrent Operations");
    concurrent_operations().await?;

    // Example 8: Statistics and monitoring
    println!("\n8. Statistics and Monitoring");
    statistics_example().await?;

    println!("\n=== All Examples Completed Successfully ===");

    Ok(())
}

/// Example 1: Basic usage with default configuration
async fn basic_usage() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new SynapticMemory instance with defaults
    let mut memory = SynapticMemory::new().await?;

    println!("  ✓ Created SynapticMemory with default configuration");
    println!("  Session ID: {}", memory.session_id());

    // Store a simple memory
    let entry = MemoryEntry::new(
        "greeting".to_string(),
        "Hello, World!".to_string(),
        MemoryType::ShortTerm,
    );

    memory.store_memory(entry).await?;
    println!("  ✓ Stored memory with key 'greeting'");

    // Retrieve it
    if let Some(retrieved) = memory.get_memory("greeting").await? {
        println!("  ✓ Retrieved memory: {}", retrieved.content);
    }

    Ok(())
}

/// Example 2: Using the builder pattern for custom configuration
async fn builder_pattern_usage() -> Result<(), Box<dyn std::error::Error>> {
    let custom_session_id = uuid::Uuid::new_v4();

    let memory = SynapticMemoryBuilder::new()
        .with_storage(StorageBackend::Memory)
        .with_knowledge_graph(true)
        .with_temporal_tracking(true)
        .with_checkpoint_interval(Duration::from_secs(300))
        .with_analytics(true)
        .with_session_id(custom_session_id)
        .build()
        .await?;

    println!("  ✓ Built SynapticMemory with custom configuration:");
    println!("    - Storage: In-Memory");
    println!("    - Knowledge Graph: Enabled");
    println!("    - Temporal Tracking: Enabled");
    println!("    - Checkpoint Interval: 5 minutes");
    println!("    - Analytics: Enabled");
    println!("    - Custom Session ID: {}", custom_session_id);

    Ok(())
}

/// Example 3: Store and retrieve operations
async fn store_and_retrieve() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = SynapticMemory::new().await?;

    // Store multiple memories
    let memories = vec![
        ("user_name", "Alice", MemoryType::LongTerm),
        ("current_task", "Writing documentation", MemoryType::ShortTerm),
        ("preference_theme", "dark", MemoryType::LongTerm),
        ("last_action", "saved file", MemoryType::ShortTerm),
    ];

    for (key, value, mem_type) in memories {
        let entry = MemoryEntry::new(
            key.to_string(),
            value.to_string(),
            mem_type,
        );
        memory.store_memory(entry).await?;
        println!("  ✓ Stored: {} = {} ({:?})", key, value, mem_type);
    }

    // Retrieve all memories
    println!("\n  Retrieving stored memories:");
    for (key, _, _) in [
        ("user_name", "", MemoryType::LongTerm),
        ("current_task", "", MemoryType::ShortTerm),
        ("preference_theme", "", MemoryType::LongTerm),
        ("last_action", "", MemoryType::ShortTerm),
    ] {
        if let Some(entry) = memory.get_memory(key).await? {
            println!("  ✓ {} = {} (accessed {} times)",
                key, entry.content, entry.access_count);
        }
    }

    // Try retrieving non-existent memory
    if memory.get_memory("nonexistent").await?.is_none() {
        println!("  ✓ Non-existent key returned None (as expected)");
    }

    Ok(())
}

/// Example 4: Update operations
async fn update_operations() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = SynapticMemory::new().await?;

    // Store initial value
    let entry = MemoryEntry::new(
        "status".to_string(),
        "initializing".to_string(),
        MemoryType::ShortTerm,
    );
    memory.store_memory(entry).await?;
    println!("  ✓ Initial status: initializing");

    // Update through different stages
    let stages = vec!["loading", "processing", "ready", "completed"];

    for stage in stages {
        memory.update_memory("status", stage).await?;
        println!("  ✓ Updated status: {}", stage);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Verify final state
    if let Some(final_entry) = memory.get_memory("status").await? {
        println!("  ✓ Final status: {} (accessed {} times)",
            final_entry.content, final_entry.access_count);
    }

    // Try to update non-existent memory (should fail)
    match memory.update_memory("nonexistent", "value").await {
        Err(_) => println!("  ✓ Update of non-existent memory failed (as expected)"),
        Ok(_) => println!("  ✗ Update should have failed!"),
    }

    Ok(())
}

/// Example 5: Search operations
async fn search_operations() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = SynapticMemory::new().await?;

    // Store a corpus of documents
    let documents = vec![
        ("doc1", "Rust is a systems programming language focused on safety and performance"),
        ("doc2", "Python is popular for data science and machine learning applications"),
        ("doc3", "Rust provides memory safety without garbage collection"),
        ("doc4", "JavaScript is the language of the web, running in all browsers"),
        ("doc5", "Rust has excellent concurrency features with async/await"),
        ("doc6", "TypeScript adds type safety to JavaScript development"),
        ("doc7", "Rust's ownership system prevents data races at compile time"),
    ];

    println!("  Storing document corpus...");
    for (key, content) in documents {
        let entry = MemoryEntry::new(
            key.to_string(),
            content.to_string(),
            MemoryType::LongTerm,
        );
        memory.store_memory(entry).await?;
    }
    println!("  ✓ Stored {} documents", 7);

    // Search for Rust-related content
    println!("\n  Searching for 'Rust':");
    let rust_results = memory.search_memories("Rust", 10).await?;
    for (i, fragment) in rust_results.iter().enumerate() {
        println!("    {}. [Score: {:.2}] {}",
            i + 1,
            fragment.relevance_score,
            fragment.entry.content.chars().take(60).collect::<String>()
        );
    }

    // Search for programming languages
    println!("\n  Searching for 'language':");
    let lang_results = memory.search_memories("language", 5).await?;
    println!("  ✓ Found {} results (limit: 5)", lang_results.len());

    // Search for safety features
    println!("\n  Searching for 'safety':");
    let safety_results = memory.search_memories("safety", 10).await?;
    println!("  ✓ Found {} results related to safety", safety_results.len());

    Ok(())
}

/// Example 6: Working with different memory types
async fn memory_types_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = SynapticMemory::new().await?;

    // Short-term memories (temporary, working memory)
    println!("  Storing short-term memories:");
    for i in 1..=3 {
        let entry = MemoryEntry::new(
            format!("temp_{}", i),
            format!("Temporary item {}", i),
            MemoryType::ShortTerm,
        );
        memory.store_memory(entry).await?;
    }
    println!("  ✓ Stored 3 short-term memories");

    // Long-term memories (persistent, important information)
    println!("\n  Storing long-term memories:");
    for i in 1..=3 {
        let entry = MemoryEntry::new(
            format!("permanent_{}", i),
            format!("Important fact {}", i),
            MemoryType::LongTerm,
        );
        memory.store_memory(entry).await?;
    }
    println!("  ✓ Stored 3 long-term memories");

    // Retrieve and display memory types
    println!("\n  Memory types verification:");
    for key in ["temp_1", "permanent_1"] {
        if let Some(entry) = memory.get_memory(key).await? {
            println!("  ✓ {} is {:?}", key, entry.memory_type);
        }
    }

    Ok(())
}

/// Example 7: Concurrent operations
async fn concurrent_operations() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let memory = Arc::new(Mutex::new(SynapticMemory::new().await?));

    println!("  Spawning 10 concurrent write operations...");

    let mut handles = vec![];
    for i in 0..10 {
        let mem = Arc::clone(&memory);
        let handle = tokio::spawn(async move {
            let mut m = mem.lock().await;
            let entry = MemoryEntry::new(
                format!("concurrent_{}", i),
                format!("Value from thread {}", i),
                MemoryType::ShortTerm,
            );
            m.store_memory(entry).await
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await??;
    }
    println!("  ✓ All concurrent writes completed successfully");

    // Verify all writes succeeded
    println!("\n  Verifying concurrent writes:");
    for i in 0..10 {
        let retrieved = memory.lock().await
            .get_memory(&format!("concurrent_{}", i))
            .await?;
        if retrieved.is_some() {
            println!("  ✓ concurrent_{} exists", i);
        }
    }

    Ok(())
}

/// Example 8: Statistics and monitoring
async fn statistics_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = SynapticMemory::new().await?;

    // Store various memories
    for i in 0..20 {
        let mem_type = if i % 2 == 0 {
            MemoryType::ShortTerm
        } else {
            MemoryType::LongTerm
        };

        let entry = MemoryEntry::new(
            format!("item_{}", i),
            format!("Content for item {}", i),
            mem_type,
        );
        memory.store_memory(entry).await?;
    }

    // Get statistics
    let stats = memory.get_stats();

    println!("  Memory System Statistics:");
    println!("  -------------------------");
    println!("  Session ID: {}", stats.session_id);
    println!("  Total Entries: {}", stats.total_entries);
    println!("  Short-term: {}", stats.short_term_entries);
    println!("  Long-term: {}", stats.long_term_entries);
    println!("  Total Size: {} bytes", stats.total_size_bytes);
    println!("  Average Size: {:.2} bytes", stats.average_entry_size);

    if let Some(oldest) = stats.oldest_entry {
        println!("  Oldest Entry: {}", oldest.format("%Y-%m-%d %H:%M:%S"));
    }

    if let Some(newest) = stats.newest_entry {
        println!("  Newest Entry: {}", newest.format("%Y-%m-%d %H:%M:%S"));
    }

    Ok(())
}
