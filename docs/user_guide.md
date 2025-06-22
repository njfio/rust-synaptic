# Synaptic User Guide

This comprehensive guide will help you get started with Synaptic and make the most of its features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Examples](#examples)

## Getting Started

### Installation

Add Synaptic to your Rust project:

```toml
[dependencies]
synaptic = "0.1.0"

# Or with specific features
synaptic = { version = "0.1.0", features = ["analytics", "security", "multimodal"] }
```

### Quick Start

```rust
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a memory instance
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store a memory
    memory.store("greeting", "Hello, World!").await?;

    // Retrieve the memory
    if let Some(entry) = memory.retrieve("greeting").await? {
        println!("Retrieved: {}", entry.value);
    }

    Ok(())
}
```

## Basic Usage

### Memory Operations

#### Storing Information

```rust
// Store simple text
memory.store("user_name", "Alice").await?;

// Store structured data (JSON)
let user_data = serde_json::json!({
    "name": "Alice",
    "age": 30,
    "preferences": ["dark_mode", "notifications"]
});
memory.store("user_profile", &user_data.to_string()).await?;

// Store with specific memory type
use synaptic::memory::types::{MemoryEntry, MemoryType};
let entry = MemoryEntry::new(
    "important_fact".to_string(),
    "The capital of France is Paris".to_string(),
    MemoryType::LongTerm,
);
memory.store_entry(&entry).await?;
```

#### Retrieving Information

```rust
// Retrieve by key
if let Some(entry) = memory.retrieve("user_name").await? {
    println!("User name: {}", entry.value);
    println!("Created at: {}", entry.created_at);
    println!("Access count: {}", entry.access_count);
}

// Retrieve multiple keys
let keys = vec!["user_name", "user_profile"];
let entries = memory.retrieve_batch(&keys).await?;
```

#### Searching Memories

```rust
// Basic text search
let results = memory.search("capital France", 5).await?;
for result in results {
    println!("Found: {} (score: {})", result.key, result.score);
}

// Advanced search with filters
use synaptic::memory::management::search::SearchOptions;
let options = SearchOptions {
    memory_types: Some(vec![MemoryType::LongTerm]),
    date_range: None,
    min_score: Some(0.5),
    include_metadata: true,
};
let results = memory.search_with_options("artificial intelligence", 10, options).await?;
```

#### Updating and Deleting

```rust
// Update existing memory
memory.update("user_name", "Alice Smith").await?;

// Delete memory
memory.delete("old_data").await?;

// Bulk delete
let keys_to_delete = vec!["temp1", "temp2", "temp3"];
memory.delete_batch(&keys_to_delete).await?;
```

### Memory Statistics

```rust
// Get basic statistics
let stats = memory.stats();
println!("Total memories: {}", stats.total_count);
println!("Short-term: {}", stats.short_term_count);
println!("Long-term: {}", stats.long_term_count);
println!("Total size: {} bytes", stats.total_size);

// Get detailed analytics (requires "analytics" feature)
if let Some(analytics) = memory.analytics() {
    let insights = analytics.generate_insights().await?;
    for insight in insights {
        println!("Insight: {} (priority: {:?})", insight.description, insight.priority);
    }
}
```

## Advanced Features

### Knowledge Graph

```rust
use synaptic::memory::knowledge_graph::KnowledgeGraph;

// Enable knowledge graph
let mut config = MemoryConfig::default();
config.enable_knowledge_graph = true;
let mut memory = AgentMemory::new(config).await?;

// Store related information
memory.store("ai", "Artificial Intelligence is a field of computer science").await?;
memory.store("ml", "Machine Learning is a subset of AI").await?;
memory.store("dl", "Deep Learning is a subset of Machine Learning").await?;

// The knowledge graph will automatically detect relationships
if let Some(graph) = memory.knowledge_graph() {
    let related = graph.find_related("ai", 2).await?;
    println!("Related to AI: {:?}", related);
}
```

### Memory Consolidation

```rust
// Enable automatic consolidation
let mut config = MemoryConfig::default();
config.enable_consolidation = true;
config.consolidation_interval = std::time::Duration::from_hours(1);

let mut memory = AgentMemory::new(config).await?;

// Store similar memories - they will be automatically consolidated
memory.store("fact1", "Paris is the capital of France").await?;
memory.store("fact2", "The capital city of France is Paris").await?;
memory.store("fact3", "France's capital is Paris").await?;

// Trigger manual consolidation
memory.consolidate_memories().await?;
```

### Security Features

```rust
use synaptic::security::{SecurityConfig, SecurityManager};

// Enable security features
let mut config = MemoryConfig::default();
config.enable_security = true;

let security_config = SecurityConfig {
    enable_encryption: true,
    enable_access_control: true,
    enable_audit_logging: true,
    ..Default::default()
};
config.security_config = Some(security_config);

let mut memory = AgentMemory::new(config).await?;

// Memories will be automatically encrypted
memory.store("sensitive_data", "This is confidential information").await?;
```

### Multi-modal Processing

```rust
use synaptic::multimodal::{DocumentProcessor, ImageProcessor};

// Enable multi-modal processing
let mut config = MemoryConfig::default();
config.enable_multimodal = true;

let mut memory = AgentMemory::new(config).await?;

// Process documents
let doc_processor = DocumentProcessor::new().await?;
let content = doc_processor.process_pdf("document.pdf").await?;
memory.store("document_content", &content.text).await?;

// Process images
let img_processor = ImageProcessor::new().await?;
let features = img_processor.extract_features("image.jpg").await?;
memory.store("image_features", &serde_json::to_string(&features)?).await?;
```

## Configuration

### Memory Configuration

```rust
use synaptic::{MemoryConfig, StorageBackend};
use std::time::Duration;

let config = MemoryConfig {
    // Storage configuration
    storage_backend: StorageBackend::File,
    storage_path: Some("./data/memories".to_string()),
    
    // Performance settings
    max_memory_size: Some(1024 * 1024 * 1024), // 1GB
    cache_size: 10000,
    enable_compression: true,
    
    // Feature flags
    enable_knowledge_graph: true,
    enable_consolidation: true,
    enable_analytics: true,
    enable_security: false,
    enable_multimodal: false,
    
    // Timing settings
    consolidation_interval: Duration::from_hours(6),
    cleanup_interval: Duration::from_days(1),
    
    // Search settings
    default_search_limit: 10,
    enable_semantic_search: true,
    
    // Session settings
    session_id: Some(uuid::Uuid::new_v4()),
    user_id: Some("user123".to_string()),
};
```

### Storage Backend Configuration

```rust
use synaptic::memory::storage::{StorageConfig, StorageBackend};

// File storage configuration
let storage_config = StorageConfig {
    backend: StorageBackend::File,
    file_path: Some("./data/synaptic.db".to_string()),
    max_connections: 10,
    enable_compression: true,
    enable_encryption: false,
    connection_timeout: Duration::from_secs(30),
    ..Default::default()
};

// SQL storage configuration (requires "sql-storage" feature)
let sql_config = StorageConfig {
    backend: StorageBackend::Sql,
    connection_string: Some("postgresql://user:pass@localhost/synaptic".to_string()),
    max_connections: 20,
    enable_connection_pooling: true,
    ..Default::default()
};
```

### Analytics Configuration

```rust
use synaptic::analytics::AnalyticsConfig;

let analytics_config = AnalyticsConfig {
    enable_behavioral_analysis: true,
    enable_performance_monitoring: true,
    enable_predictive_analytics: true,
    
    // Data retention
    metrics_retention_days: 90,
    detailed_logs_retention_days: 30,
    
    // Sampling
    sampling_rate: 0.1, // 10% sampling
    enable_real_time_processing: true,
    
    // Thresholds
    performance_alert_threshold: Duration::from_millis(1000),
    memory_usage_alert_threshold: 0.8, // 80%
};
```

## Best Practices

### Memory Organization

1. **Use Appropriate Memory Types**
   ```rust
   // For temporary data
   let temp_entry = MemoryEntry::new(key, value, MemoryType::ShortTerm);
   
   // For important facts
   let fact_entry = MemoryEntry::new(key, value, MemoryType::LongTerm);
   
   // For current context
   let context_entry = MemoryEntry::new(key, value, MemoryType::Working);
   ```

2. **Implement Proper Key Naming**
   ```rust
   // Use hierarchical keys
   memory.store("user:123:preferences", preferences).await?;
   memory.store("session:abc:context", context).await?;
   memory.store("system:config:theme", theme).await?;
   ```

3. **Regular Cleanup**
   ```rust
   // Implement periodic cleanup
   tokio::spawn(async move {
       let mut interval = tokio::time::interval(Duration::from_hours(24));
       loop {
           interval.tick().await;
           if let Err(e) = memory.cleanup_expired_memories().await {
               eprintln!("Cleanup error: {}", e);
           }
       }
   });
   ```

### Performance Optimization

1. **Batch Operations**
   ```rust
   // Instead of multiple individual stores
   let entries = vec![
       ("key1", "value1"),
       ("key2", "value2"),
       ("key3", "value3"),
   ];
   memory.store_batch(&entries).await?;
   ```

2. **Use Appropriate Search Limits**
   ```rust
   // Don't retrieve more than needed
   let results = memory.search("query", 5).await?; // Not 1000
   ```

3. **Enable Compression for Large Data**
   ```rust
   let mut config = MemoryConfig::default();
   config.enable_compression = true; // Reduces storage size
   ```

### Error Handling

```rust
use synaptic::error::{Result, SynapticError};

async fn robust_memory_operation(memory: &mut AgentMemory) -> Result<()> {
    match memory.store("key", "value").await {
        Ok(()) => {
            println!("Successfully stored memory");
            Ok(())
        }
        Err(SynapticError::StorageError(msg)) => {
            eprintln!("Storage error: {}", msg);
            // Implement retry logic or fallback
            Err(SynapticError::StorageError(msg))
        }
        Err(SynapticError::ValidationError(msg)) => {
            eprintln!("Validation error: {}", msg);
            // Fix the input and retry
            Err(SynapticError::ValidationError(msg))
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            Err(e)
        }
    }
}
```

## Troubleshooting

### Common Issues

#### Memory Not Found
```rust
// Always check if memory exists
match memory.retrieve("key").await? {
    Some(entry) => println!("Found: {}", entry.value),
    None => println!("Memory not found - check the key"),
}
```

#### Storage Errors
```rust
// Check storage backend status
if let Err(e) = memory.health_check().await {
    eprintln!("Storage health check failed: {}", e);
    // Implement recovery logic
}
```

#### Performance Issues
```rust
// Monitor performance
let start = std::time::Instant::now();
memory.search("query", 10).await?;
let duration = start.elapsed();
if duration > Duration::from_millis(100) {
    println!("Slow search detected: {:?}", duration);
}
```

### Debugging

Enable detailed logging:

```rust
use tracing::{info, debug, error};
use tracing_subscriber;

// Initialize logging
tracing_subscriber::fmt::init();

// Use throughout your code
debug!("Storing memory with key: {}", key);
info!("Memory operation completed successfully");
error!("Failed to retrieve memory: {}", error);
```

## Examples

### Complete Application Example

```rust
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the memory system
    let config = MemoryConfig {
        storage_backend: StorageBackend::File,
        storage_path: Some("./app_data/memories.db".to_string()),
        enable_knowledge_graph: true,
        enable_consolidation: true,
        consolidation_interval: Duration::from_hours(1),
        ..Default::default()
    };

    // Create memory instance
    let mut memory = AgentMemory::new(config).await?;

    // Store user preferences
    memory.store("user:theme", "dark").await?;
    memory.store("user:language", "en").await?;
    memory.store("user:notifications", "enabled").await?;

    // Store application state
    let app_state = serde_json::json!({
        "last_opened": "2024-01-15T10:30:00Z",
        "active_projects": ["project1", "project2"],
        "recent_files": ["/path/to/file1.txt", "/path/to/file2.txt"]
    });
    memory.store("app:state", &app_state.to_string()).await?;

    // Search for user-related memories
    let user_memories = memory.search("user:", 10).await?;
    println!("Found {} user memories", user_memories.len());

    // Get statistics
    let stats = memory.stats();
    println!("Total memories: {}", stats.total_count);
    println!("Storage size: {} bytes", stats.total_size);

    // Cleanup old memories
    memory.cleanup_expired_memories().await?;

    Ok(())
}
```

This user guide provides comprehensive coverage of Synaptic's features and best practices. For more detailed information, see:

- [API Guide](api_guide.md) - Detailed API documentation
- [Architecture Guide](architecture.md) - System architecture overview
- [Deployment Guide](deployment.md) - Production deployment instructions
- [Error Handling Guide](error_handling_guide.md) - Comprehensive error handling
