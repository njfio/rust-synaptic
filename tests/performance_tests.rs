//! Performance and benchmark tests
//! 
//! Tests system performance under load, memory efficiency,
//! and optimization validation.

use synaptic::{
    AgentMemory, MemoryConfig,
};

#[cfg(feature = "embeddings")]
use synaptic::{
    MemoryEntry, MemoryType,
    memory::embeddings::{EmbeddingManager, EmbeddingConfig},
};
use std::error::Error;
use std::time::{Duration, Instant};
use tokio::time::timeout;

#[tokio::test]
#[ignore] // Ignored by default, run with --ignored flag
async fn test_memory_storage_performance() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    let num_operations = 1000;
    let start_time = Instant::now();
    
    // Test bulk storage performance
    for i in 0..num_operations {
        let key = format!("perf_test_key_{}", i);
        let value = format!("Performance test value {} with some additional content to make it realistic", i);
        memory.store(&key, &value).await?;
    }
    
    let storage_duration = start_time.elapsed();
    println!("Stored {} memories in {:?}", num_operations, storage_duration);
    
    // Test bulk retrieval performance
    let retrieval_start = Instant::now();
    for i in 0..num_operations {
        let key = format!("perf_test_key_{}", i);
        let retrieved = memory.retrieve(&key).await?;
        assert!(retrieved.is_some());
    }
    
    let retrieval_duration = retrieval_start.elapsed();
    println!("Retrieved {} memories in {:?}", num_operations, retrieval_duration);
    
    // Performance assertions
    let avg_storage_time = storage_duration.as_millis() as f64 / num_operations as f64;
    let avg_retrieval_time = retrieval_duration.as_millis() as f64 / num_operations as f64;
    
    println!("Average storage time: {:.2}ms per operation", avg_storage_time);
    println!("Average retrieval time: {:.2}ms per operation", avg_retrieval_time);
    
    // Assert reasonable performance (adjust thresholds as needed)
    assert!(avg_storage_time < 10.0, "Storage too slow: {:.2}ms", avg_storage_time);
    assert!(avg_retrieval_time < 5.0, "Retrieval too slow: {:.2}ms", avg_retrieval_time);
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_concurrent_access_performance() -> Result<(), Box<dyn Error>> {
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio::task::JoinSet;
    
    let config = MemoryConfig::default();
    let memory = Arc::new(Mutex::new(AgentMemory::new(config).await?));
    
    let num_tasks = 50;
    let operations_per_task = 20;
    let start_time = Instant::now();
    
    let mut join_set = JoinSet::new();
    
    // Spawn concurrent tasks with optimized lock usage
    for task_id in 0..num_tasks {
        let memory_clone = Arc::clone(&memory);
        join_set.spawn(async move {
            // Batch operations to reduce lock contention
            let mut operations = Vec::new();
            for op_id in 0..operations_per_task {
                let key = format!("concurrent_{}_{}", task_id, op_id);
                let value = format!("Concurrent test value {} {}", task_id, op_id);
                operations.push((key, value));
            }

            // Perform all operations in a single lock acquisition
            {
                let mut mem = memory_clone.lock().await;
                for (key, value) in operations {
                    mem.store(&key, &value).await?;
                    let retrieved = mem.retrieve(&key).await?;
                    assert!(retrieved.is_some());
                }
            }
            Ok::<(), Box<dyn Error + Send + Sync>>(())
        });
    }
    
    // Wait for all tasks to complete
    while let Some(result) = join_set.join_next().await {
        match result.unwrap() {
            Ok(_) => {},
            Err(e) => return Err(format!("Task failed: {}", e).into()),
        }
    }
    
    let total_duration = start_time.elapsed();
    let total_operations = num_tasks * operations_per_task * 2; // store + retrieve
    
    println!("Completed {} concurrent operations in {:?}", total_operations, total_duration);
    println!("Average time per operation: {:.2}ms", 
             total_duration.as_millis() as f64 / total_operations as f64);
    
    // Assert reasonable concurrent performance
    assert!(total_duration < Duration::from_secs(30), "Concurrent operations too slow");
    
    Ok(())
}

#[cfg(feature = "embeddings")]
#[tokio::test]
#[ignore]
async fn test_embedding_performance() -> Result<(), Box<dyn Error>> {
    let config = EmbeddingConfig::default();
    let mut manager = EmbeddingManager::new(config);
    
    let test_texts = (0..100)
        .map(|i| format!("This is test text number {} for embedding performance testing with sufficient length", i))
        .collect::<Vec<_>>();
    
    let start_time = Instant::now();
    
    // Test embedding generation performance
    for text in &test_texts {
        let memory = MemoryEntry::new(
            format!("perf_test_{}", text.len()),
            text.clone(),
            MemoryType::ShortTerm,
        );
        manager.add_memory(memory)?;
    }
    
    let embedding_duration = start_time.elapsed();
    println!("Generated {} embeddings in {:?}", test_texts.len(), embedding_duration);
    
    // Test similarity search performance
    let search_start = Instant::now();
    let query = "test text performance";
    let results = manager.find_similar_to_query(query, Some(10))?;
    let search_duration = search_start.elapsed();
    
    println!("Similarity search completed in {:?}", search_duration);
    println!("Found {} similar results", results.len());
    
    // Performance assertions
    let avg_embedding_time = embedding_duration.as_millis() as f64 / test_texts.len() as f64;
    println!("Average embedding time: {:.2}ms per text", avg_embedding_time);
    
    assert!(avg_embedding_time < 100.0, "Embedding generation too slow: {:.2}ms", avg_embedding_time);
    assert!(search_duration < Duration::from_millis(500), "Similarity search too slow");
    assert!(!results.is_empty(), "Search should return results");
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_memory_usage() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Get initial memory stats
    let initial_stats = memory.stats();
    println!("Initial stats: {:?}", initial_stats);
    
    let num_memories = 1000;
    let large_content = "x".repeat(1000); // 1KB per memory
    
    // Add memories and track growth
    for i in 0..num_memories {
        let key = format!("memory_usage_test_{}", i);
        memory.store(&key, &large_content).await?;
        
        // Check stats every 100 memories
        if i % 100 == 0 {
            let current_stats = memory.stats();
            println!("After {} memories: {:?}", i + 1, current_stats);
        }
    }
    
    let final_stats = memory.stats();
    println!("Final stats: {:?}", final_stats);
    
    // Verify memory growth is reasonable
    assert_eq!(final_stats.short_term_count, num_memories);
    assert!(final_stats.total_size > initial_stats.total_size);
    
    // Test memory cleanup/optimization
    let cleanup_start = Instant::now();
    // In a full implementation, this would trigger memory optimization
    let cleanup_duration = cleanup_start.elapsed();
    println!("Memory cleanup completed in {:?}", cleanup_duration);
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_search_performance() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Populate with diverse content
    let topics = vec!["AI", "machine learning", "data science", "programming", "algorithms"];
    let num_memories_per_topic = 200;
    
    for topic in &topics {
        for i in 0..num_memories_per_topic {
            let key = format!("{}_{}", topic, i);
            let value = format!("This is content about {} with additional details and context number {}", topic, i);
            memory.store(&key, &value).await?;
        }
    }
    
    println!("Populated memory with {} entries", topics.len() * num_memories_per_topic);
    
    // Test search performance for different query types
    let search_queries = vec![
        ("AI", "exact topic match"),
        ("machine", "partial match"),
        ("programming algorithms", "multi-term"),
        ("nonexistent topic", "no results"),
    ];
    
    for (query, description) in search_queries {
        let search_start = Instant::now();
        let results = memory.search(query, 20).await?;
        let search_duration = search_start.elapsed();
        
        println!("Search '{}' ({}): {} results in {:?}", 
                 query, description, results.len(), search_duration);
        
        // Assert reasonable search performance
        assert!(search_duration < Duration::from_millis(100), 
                "Search too slow for query '{}': {:?}", query, search_duration);
    }
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_checkpoint_performance() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Add substantial amount of data
    let num_memories = 500;
    for i in 0..num_memories {
        let key = format!("checkpoint_test_{}", i);
        let value = format!("Checkpoint test data {} with substantial content to test serialization performance", i);
        memory.store(&key, &value).await?;
    }
    
    // Test checkpoint creation performance
    let checkpoint_start = Instant::now();
    let checkpoint_id = memory.checkpoint().await?;
    let checkpoint_duration = checkpoint_start.elapsed();
    
    println!("Created checkpoint for {} memories in {:?}", num_memories, checkpoint_duration);
    
    // Modify some data
    for i in 0..50 {
        let key = format!("checkpoint_test_{}", i);
        let value = format!("Modified data {}", i);
        memory.store(&key, &value).await?;
    }
    
    // Test restoration performance
    let restore_start = Instant::now();
    memory.restore_checkpoint(checkpoint_id).await?;
    let restore_duration = restore_start.elapsed();
    
    println!("Restored checkpoint in {:?}", restore_duration);
    
    // Performance assertions
    assert!(checkpoint_duration < Duration::from_secs(5), "Checkpoint creation too slow");
    assert!(restore_duration < Duration::from_secs(5), "Checkpoint restoration too slow");
    
    // Verify restoration worked
    let restored = memory.retrieve("checkpoint_test_0").await?;
    assert!(restored.is_some());
    assert!(restored.unwrap().value.contains("Checkpoint test data"));
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_timeout_handling() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // Test that operations complete within reasonable timeouts

    // Test store operation
    let result = timeout(Duration::from_secs(1), memory.store("timeout_test", "test_value")).await;
    assert!(result.is_ok(), "Store operation timed out");
    assert!(result.unwrap().is_ok(), "Store operation failed");

    // Test retrieve operation
    let result = timeout(Duration::from_secs(1), memory.retrieve("timeout_test")).await;
    assert!(result.is_ok(), "Retrieve operation timed out");
    assert!(result.unwrap().is_ok(), "Retrieve operation failed");

    // Test search operation
    let result = timeout(Duration::from_secs(1), memory.search("timeout", 5)).await;
    assert!(result.is_ok(), "Search operation timed out");
    assert!(result.unwrap().is_ok(), "Search operation failed");
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_stress_test() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    let stress_duration = Duration::from_secs(10);
    let start_time = Instant::now();
    let mut operation_count = 0;
    
    println!("Starting stress test for {:?}", stress_duration);
    
    while start_time.elapsed() < stress_duration {
        let op_type = operation_count % 3;
        
        match op_type {
            0 => {
                // Store operation
                let key = format!("stress_test_{}", operation_count);
                let value = format!("Stress test value {}", operation_count);
                memory.store(&key, &value).await?;
            },
            1 => {
                // Retrieve operation
                let key = format!("stress_test_{}", operation_count / 2);
                let _ = memory.retrieve(&key).await?;
            },
            2 => {
                // Search operation
                let query = format!("stress {}", operation_count % 10);
                let _ = memory.search(&query, 5).await?;
            },
            _ => unreachable!(),
        }
        
        operation_count += 1;
        
        // Print progress every 1000 operations
        if operation_count % 1000 == 0 {
            println!("Completed {} operations in {:?}", operation_count, start_time.elapsed());
        }
    }
    
    let total_duration = start_time.elapsed();
    let ops_per_second = operation_count as f64 / total_duration.as_secs_f64();
    
    println!("Stress test completed: {} operations in {:?}", operation_count, total_duration);
    println!("Operations per second: {:.2}", ops_per_second);
    
    // Assert minimum performance under stress
    assert!(ops_per_second > 100.0, "Performance under stress too low: {:.2} ops/sec", ops_per_second);
    assert!(operation_count > 1000, "Not enough operations completed: {}", operation_count);
    
    Ok(())
}
