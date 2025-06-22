# Synaptic API Guide

This guide provides comprehensive documentation for the Synaptic AI agent memory system API.

## Table of Contents

1. [Core Memory Operations](#core-memory-operations)
2. [Storage Backends](#storage-backends)
3. [Knowledge Graph](#knowledge-graph)
4. [Analytics](#analytics)
5. [Security](#security)
6. [Multi-modal Processing](#multi-modal-processing)
7. [Error Handling](#error-handling)

## Core Memory Operations

### AgentMemory

The main entry point for the Synaptic memory system.

```rust
use synaptic::{AgentMemory, MemoryConfig};

// Create a new memory instance
let config = MemoryConfig::default();
let mut memory = AgentMemory::new(config).await?;
```

### Basic Operations

#### Store Memory

```rust
use synaptic::memory::types::{MemoryEntry, MemoryType};

// Create a memory entry
let entry = MemoryEntry::new(
    "user_preference".to_string(),
    "Dark mode enabled".to_string(),
    MemoryType::ShortTerm,
);

// Store the memory
memory.store("user_preference", "Dark mode enabled").await?;
```

#### Retrieve Memory

```rust
// Retrieve a specific memory
if let Some(entry) = memory.retrieve("user_preference").await? {
    println!("Value: {}", entry.value);
    println!("Type: {:?}", entry.memory_type);
    println!("Created: {}", entry.created_at);
}
```

#### Update Memory

```rust
// Update existing memory
memory.update("user_preference", "Light mode enabled").await?;
```

#### Delete Memory

```rust
// Delete a memory
memory.delete("user_preference").await?;
```

#### Search Memories

```rust
// Search for memories
let results = memory.search("dark mode", 10).await?;
for result in results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

### Memory Types

```rust
use synaptic::memory::types::MemoryType;

// Different memory types
let short_term = MemoryType::ShortTerm;    // Temporary memories
let long_term = MemoryType::LongTerm;      // Persistent memories
let working = MemoryType::Working;         // Active working memories
let episodic = MemoryType::Episodic;       // Event-based memories
let semantic = MemoryType::Semantic;       // Factual knowledge
```

## Storage Backends

### Memory Storage (Default)

```rust
use synaptic::memory::storage::create_storage;
use synaptic::StorageBackend;

// In-memory storage (default)
let storage = create_storage(StorageBackend::Memory).await?;
```

### File Storage

```rust
// File-based storage with Sled database
let storage = create_storage(StorageBackend::File).await?;
```

### SQL Storage (Optional Feature)

```rust
// PostgreSQL storage (requires "sql-storage" feature)
let storage = create_storage(StorageBackend::Sql).await?;
```

### Custom Storage Configuration

```rust
use synaptic::memory::storage::StorageConfig;

let config = StorageConfig {
    backend: StorageBackend::File,
    file_path: Some("./data/memories.db".to_string()),
    connection_string: None,
    max_connections: 10,
    enable_compression: true,
    enable_encryption: false,
};
```

## Knowledge Graph

### Basic Graph Operations

```rust
use synaptic::memory::knowledge_graph::KnowledgeGraph;

// Create a knowledge graph
let mut graph = KnowledgeGraph::new().await?;

// Add nodes
graph.add_node("concept1", "Artificial Intelligence").await?;
graph.add_node("concept2", "Machine Learning").await?;

// Add relationships
graph.add_edge("concept1", "concept2", 0.8, "includes").await?;
```

### Graph Queries

```rust
// Find related concepts
let related = graph.find_related("concept1", 2).await?;

// Get shortest path
let path = graph.shortest_path("concept1", "concept2").await?;

// Traverse graph
let traversal = graph.traverse_from("concept1", 3).await?;
```

### Reasoning Engine

```rust
use synaptic::memory::knowledge_graph::reasoning::InferenceEngine;

let mut engine = InferenceEngine::new().await?;

// Perform inference
let inferences = engine.infer_relationships(&graph).await?;

// Apply reasoning rules
let conclusions = engine.apply_rules(&graph, &rules).await?;
```

## Analytics

### Memory Analytics

```rust
use synaptic::analytics::AnalyticsEngine;

// Create analytics engine (requires "analytics" feature)
let mut analytics = AnalyticsEngine::new().await?;

// Generate insights
let insights = analytics.generate_insights(&memory).await?;

// Get memory statistics
let stats = analytics.get_memory_statistics().await?;
println!("Total memories: {}", stats.total_count);
println!("Average access frequency: {}", stats.avg_access_frequency);
```

### Performance Monitoring

```rust
use synaptic::analytics::performance::PerformanceAnalyzer;

let mut analyzer = PerformanceAnalyzer::new().await?;

// Monitor operation performance
let metrics = analyzer.analyze_operation_performance().await?;

// Get performance trends
let trends = analyzer.get_performance_trends().await?;
```

### Behavioral Analysis

```rust
use synaptic::analytics::behavioral::BehavioralAnalyzer;

let mut behavioral = BehavioralAnalyzer::new().await?;

// Track user interactions
behavioral.track_interaction("user123", "search", "AI concepts").await?;

// Analyze patterns
let patterns = behavioral.analyze_user_patterns("user123").await?;
```

## Security

### Access Control

```rust
use synaptic::security::{SecurityManager, SecurityConfig};
use synaptic::security::access_control::{AccessControlManager, Permission};

// Create security manager (requires "security" feature)
let config = SecurityConfig::default();
let mut security = SecurityManager::new(config).await?;

// Add roles and permissions
security.access_control.add_role(
    "admin".to_string(),
    vec![Permission::ReadMemory, Permission::WriteMemory, Permission::DeleteMemory]
).await?;
```

### Authentication

```rust
use synaptic::security::access_control::{AuthenticationCredentials, AuthenticationType};

// Authenticate user
let creds = AuthenticationCredentials {
    auth_type: AuthenticationType::Password,
    password: Some("secure_password".to_string()),
    api_key: None,
    certificate: None,
    mfa_token: None,
    ip_address: Some("127.0.0.1".to_string()),
    user_agent: Some("MyApp/1.0".to_string()),
};

let context = security.access_control.authenticate("user123".to_string(), creds).await?;
```

### Encryption

```rust
use synaptic::memory::types::MemoryEntry;

// Encrypt memory entry
let entry = MemoryEntry::new("secret".to_string(), "sensitive data".to_string(), MemoryType::LongTerm);
let encrypted = security.encrypt_memory(&entry, &context).await?;

// Decrypt memory entry
let decrypted = security.decrypt_memory(&encrypted, &context).await?;
```

## Multi-modal Processing

### Document Processing

```rust
use synaptic::multimodal::document::DocumentProcessor;

// Process documents (requires "multimodal" feature)
let mut processor = DocumentProcessor::new().await?;

// Process PDF
let pdf_content = processor.process_pdf("document.pdf").await?;

// Process Markdown
let md_content = processor.process_markdown("README.md").await?;
```

### Image Processing

```rust
use synaptic::multimodal::image::ImageProcessor;

let mut image_processor = ImageProcessor::new().await?;

// Extract features from image
let features = image_processor.extract_features("image.jpg").await?;

// Perform OCR
let text = image_processor.extract_text("document.png").await?;
```

### Audio Processing

```rust
use synaptic::multimodal::audio::AudioProcessor;

let mut audio_processor = AudioProcessor::new().await?;

// Process audio file
let audio_features = audio_processor.process_audio("speech.wav").await?;

// Extract speech
let transcript = audio_processor.speech_to_text("speech.wav").await?;
```

## Error Handling

### Result Types

All Synaptic operations return `Result<T, SynapticError>`:

```rust
use synaptic::error::{Result, SynapticError};

// Handle results
match memory.store("key", "value").await {
    Ok(()) => println!("Stored successfully"),
    Err(SynapticError::StorageError(msg)) => eprintln!("Storage error: {}", msg),
    Err(SynapticError::ValidationError(msg)) => eprintln!("Validation error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Error Types

```rust
use synaptic::error::SynapticError;

// Common error types
SynapticError::StorageError(String)      // Storage backend errors
SynapticError::ValidationError(String)   // Input validation errors
SynapticError::SecurityError(String)     // Security-related errors
SynapticError::NetworkError(String)      // Network/connectivity errors
SynapticError::SerializationError(String) // Serialization errors
```

### Error Recovery

```rust
// Implement retry logic
use tokio::time::{sleep, Duration};

async fn store_with_retry(memory: &mut AgentMemory, key: &str, value: &str) -> Result<()> {
    for attempt in 1..=3 {
        match memory.store(key, value).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < 3 => {
                eprintln!("Attempt {} failed: {}", attempt, e);
                sleep(Duration::from_millis(100 * attempt)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
```
