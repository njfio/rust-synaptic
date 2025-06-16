# Synaptic API Documentation

This document provides comprehensive API documentation for the Synaptic intelligent memory system.

## Table of Contents

1. [Core API](#core-api)
2. [Memory Management](#memory-management)
3. [Knowledge Graph](#knowledge-graph)
4. [Temporal Operations](#temporal-operations)
5. [Security & Privacy](#security--privacy)
6. [External Integrations](#external-integrations)
7. [Multimodal Processing](#multimodal-processing)
8. [Error Handling](#error-handling)
9. [Configuration](#configuration)
10. [Examples](#examples)

## Core API

### AgentMemory

The main entry point for the Synaptic memory system.

```rust
use synaptic::AgentMemory;

// Create a new memory instance
let memory = AgentMemory::new().await?;

// Store a memory entry
let entry_id = memory.store("key", "content", None).await?;

// Retrieve a memory entry
let entry = memory.retrieve("key").await?;

// Search memories
let results = memory.search("query", 10).await?;
```

#### Methods

##### `new() -> Result<Self>`
Creates a new AgentMemory instance with default configuration.

##### `with_config(config: MemoryConfig) -> Result<Self>`
Creates a new AgentMemory instance with custom configuration.

##### `store(key: &str, content: &str, metadata: Option<MemoryMetadata>) -> Result<String>`
Stores a new memory entry.

**Parameters:**
- `key`: Unique identifier for the memory
- `content`: The content to store
- `metadata`: Optional metadata for the memory

**Returns:** The ID of the stored memory entry

##### `retrieve(key: &str) -> Result<Option<MemoryEntry>>`
Retrieves a memory entry by key.

**Parameters:**
- `key`: The key of the memory to retrieve

**Returns:** The memory entry if found

##### `search(query: &str, limit: usize) -> Result<Vec<SearchResult>>`
Searches for memories matching the query.

**Parameters:**
- `query`: Search query string
- `limit`: Maximum number of results to return

**Returns:** Vector of search results

##### `update(key: &str, content: &str) -> Result<()>`
Updates an existing memory entry.

##### `delete(key: &str) -> Result<bool>`
Deletes a memory entry.

##### `list_keys() -> Result<Vec<String>>`
Lists all memory keys.

## Memory Management

### MemoryManager

Handles advanced memory operations including lifecycle management, optimization, and analytics.

```rust
use synaptic::memory::management::MemoryManager;

let manager = MemoryManager::new(storage).await?;

// Optimize memory storage
manager.optimize().await?;

// Get memory analytics
let analytics = manager.get_analytics().await?;

// Manage memory lifecycle
manager.cleanup_expired().await?;
```

#### Key Features

- **Lifecycle Management**: Automatic archiving and cleanup of old memories
- **Optimization**: Compression, deduplication, and indexing optimization
- **Analytics**: Usage patterns, performance metrics, and insights
- **Search Enhancement**: Advanced similarity search with multiple algorithms

### MemoryEntry

Represents a single memory entry in the system.

```rust
pub struct MemoryEntry {
    pub id: String,
    pub key: String,
    pub content: String,
    pub metadata: MemoryMetadata,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub embedding: Option<Vec<f32>>,
}
```

### MemoryMetadata

Contains metadata for memory entries.

```rust
pub struct MemoryMetadata {
    pub memory_type: MemoryType,
    pub tags: Vec<String>,
    pub importance: f64,
    pub source: Option<String>,
    pub context: HashMap<String, String>,
    pub relationships: Vec<String>,
}
```

## Knowledge Graph

### MemoryKnowledgeGraph

Manages relationships between memories and enables graph-based operations.

```rust
use synaptic::memory::knowledge_graph::MemoryKnowledgeGraph;

let graph = MemoryKnowledgeGraph::new();

// Add a relationship
graph.add_relationship("memory1", "memory2", RelationshipType::Related, 0.8).await?;

// Find related memories
let related = graph.find_related("memory1", 2).await?;

// Get graph statistics
let stats = graph.get_statistics().await?;
```

#### Relationship Types

- `Related`: General relationship between memories
- `Causal`: One memory caused or led to another
- `Temporal`: Memories are related by time
- `Semantic`: Memories share semantic meaning
- `Hierarchical`: Parent-child relationship

## Temporal Operations

### Temporal Patterns

Analyze and detect patterns in memory access and creation over time.

```rust
use synaptic::memory::temporal::patterns::PatternDetector;

let detector = PatternDetector::new();

// Detect access patterns
let patterns = detector.detect_patterns(&access_history).await?;

// Predict future access
let prediction = detector.predict_access("memory_key").await?;
```

### Differential Analysis

Track changes in memories over time.

```rust
use synaptic::memory::temporal::differential::DiffAnalyzer;

let analyzer = DiffAnalyzer::new();

// Analyze changes between versions
let diff = analyzer.analyze_changes(&old_content, &new_content).await?;

// Get change summary
let summary = analyzer.summarize_changes(&diffs).await?;
```

## Security & Privacy

### Encryption

All sensitive data is encrypted using AES-256-GCM encryption.

```rust
use synaptic::security::encryption::EncryptionManager;

let encryption = EncryptionManager::new()?;

// Encrypt data
let encrypted = encryption.encrypt("sensitive data").await?;

// Decrypt data
let decrypted = encryption.decrypt(&encrypted).await?;
```

### Zero-Knowledge Proofs

Support for zero-knowledge proofs for privacy-preserving operations.

```rust
use synaptic::security::zero_knowledge::ZeroKnowledgeManager;

let zk = ZeroKnowledgeManager::new()?;

// Generate proof
let proof = zk.generate_proof(&data, &witness).await?;

// Verify proof
let is_valid = zk.verify_proof(&proof, &public_inputs).await?;
```

### Access Control

Role-based access control for memory operations.

```rust
use synaptic::security::access_control::AccessControlManager;

let acl = AccessControlManager::new()?;

// Check permissions
let can_access = acl.check_permission(&user_id, &resource, &action).await?;

// Grant permission
acl.grant_permission(&user_id, &resource, &action).await?;
```

## External Integrations

### Redis Cache

High-performance caching layer using Redis.

```rust
use synaptic::integrations::redis_cache::RedisClient;

let redis = RedisClient::new(&config).await?;

// Cache data
redis.set("key", "value", Some(3600)).await?;

// Retrieve cached data
let value = redis.get("key").await?;
```

### Database Integration

Support for PostgreSQL and SQLite databases.

```rust
use synaptic::integrations::database::DatabaseManager;

let db = DatabaseManager::new(&database_url).await?;

// Execute query
let results = db.query("SELECT * FROM memories WHERE key = ?", &[key]).await?;
```

## Multimodal Processing

### Document Processing

Process various document formats including PDF, DOCX, Markdown, and CSV.

```rust
use synaptic::multimodal::document::DocumentProcessor;

let processor = DocumentProcessor::new();

// Process PDF document
let content = processor.process_pdf(&pdf_data).await?;

// Extract text from DOCX
let text = processor.extract_docx_text(&docx_data).await?;
```

### Image Processing

Extract text and features from images.

```rust
use synaptic::multimodal::image::ImageProcessor;

let processor = ImageProcessor::new();

// Extract text from image (OCR)
let text = processor.extract_text(&image_data).await?;

// Extract visual features
let features = processor.extract_features(&image_data).await?;
```

## Error Handling

All API methods return `Result<T, MemoryError>` for proper error handling.

```rust
use synaptic::error::MemoryError;

match memory.store("key", "content", None).await {
    Ok(id) => println!("Stored with ID: {}", id),
    Err(MemoryError::DuplicateKey) => println!("Key already exists"),
    Err(MemoryError::StorageError(e)) => println!("Storage error: {}", e),
    Err(e) => println!("Other error: {}", e),
}
```

### Error Types

- `DuplicateKey`: Attempting to store with an existing key
- `NotFound`: Requested memory not found
- `StorageError`: Underlying storage system error
- `EncryptionError`: Encryption/decryption failure
- `ValidationError`: Invalid input data
- `NetworkError`: Network-related errors
- `ConfigurationError`: Configuration issues

## Configuration

### MemoryConfig

Main configuration structure for the memory system.

```rust
use synaptic::config::MemoryConfig;

let config = MemoryConfig {
    storage_type: StorageType::File,
    storage_path: "./memory_data".to_string(),
    encryption_enabled: true,
    compression_enabled: true,
    max_memory_size: 1024 * 1024 * 1024, // 1GB
    cleanup_interval: Duration::from_secs(3600), // 1 hour
    indexing_enabled: true,
    analytics_enabled: true,
};

let memory = AgentMemory::with_config(config).await?;
```

## Examples

### Basic Usage

```rust
use synaptic::AgentMemory;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize memory system
    let memory = AgentMemory::new().await?;
    
    // Store some memories
    memory.store("user_preference", "dark_mode", None).await?;
    memory.store("last_action", "file_saved", None).await?;
    
    // Search for memories
    let results = memory.search("dark", 5).await?;
    
    for result in results {
        println!("Found: {} -> {}", result.key, result.content);
    }
    
    Ok(())
}
```

### Advanced Usage with Knowledge Graph

```rust
use synaptic::{AgentMemory, memory::knowledge_graph::RelationshipType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = AgentMemory::new().await?;
    
    // Store related memories
    memory.store("project_a", "AI research project", None).await?;
    memory.store("paper_1", "Neural networks paper", None).await?;
    memory.store("paper_2", "Deep learning paper", None).await?;
    
    // Create relationships
    let graph = memory.knowledge_graph();
    graph.add_relationship("project_a", "paper_1", RelationshipType::Related, 0.9).await?;
    graph.add_relationship("project_a", "paper_2", RelationshipType::Related, 0.8).await?;
    graph.add_relationship("paper_1", "paper_2", RelationshipType::Semantic, 0.7).await?;
    
    // Find related memories
    let related = graph.find_related("project_a", 2).await?;
    println!("Related to project_a: {:?}", related);
    
    Ok(())
}
```
