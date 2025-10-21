# Quick Start Guide

Get up and running with Synaptic in 5 minutes!

## Prerequisites

- Rust 1.79 or later
- (Optional) Docker and Docker Compose for services

## Installation

### Option 1: Using as a Library

Add Synaptic to your `Cargo.toml`:

```toml
[dependencies]
synaptic = { git = "https://github.com/njfio/rust-synaptic.git" }

# With specific features
synaptic = { git = "https://github.com/njfio/rust-synaptic.git", features = ["analytics"] }
```

### Option 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic

# Build the project
cargo build

# Run tests to verify
cargo test
```

## Your First Synaptic Application

Create a new Rust project and add this simple example:

```rust
use synaptic::{AgentMemory, MemoryConfig, MemoryType, MemoryEntry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a memory configuration
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        enable_advanced_management: false,
        ..Default::default()
    };

    // 2. Initialize the memory system
    let mut memory = AgentMemory::new(config).await?;

    // 3. Store some memories
    memory.store("user_name", "Alice Johnson").await?;
    memory.store("user_preference", "Prefers dark mode").await?;
    memory.store("project_alpha", "A web application using React").await?;

    // 4. Retrieve a memory
    if let Some(entry) = memory.retrieve("user_name").await? {
        println!("Retrieved: {}", entry.value);
    }

    // 5. Search for related memories
    let results = memory.search("user", 10).await?;
    println!("Found {} memories related to 'user'", results.len());

    for result in results {
        println!("- {}: {}", result.key, result.value);
    }

    // 6. Find related memories using knowledge graph
    let related = memory.find_related_memories("project_alpha", 5).await?;
    println!("Found {} related memories", related.len());

    Ok(())
}
```

## Running the Example

```bash
cargo run
```

## Next Steps

### Explore Features

#### 1. Storage Backends

```rust
use synaptic::{AgentMemory, MemoryConfig, StorageBackend};

// In-memory storage (fastest, for development)
let config = MemoryConfig {
    storage_backend: StorageBackend::Memory,
    ..Default::default()
};

// File-based storage (persistent)
let config = MemoryConfig {
    storage_backend: StorageBackend::File("./data/memories.db"),
    ..Default::default()
};

// SQL storage (PostgreSQL, requires feature)
#[cfg(feature = "sql-storage")]
let config = MemoryConfig {
    storage_backend: StorageBackend::Sql("postgresql://user:pass@localhost/db"),
    ..Default::default()
};
```

#### 2. Knowledge Graph

```rust
// Enable knowledge graph for relationship detection
let config = MemoryConfig {
    enable_knowledge_graph: true,
    ..Default::default()
};

let mut memory = AgentMemory::new(config).await?;

// Store related information
memory.store("person:alice", "Software engineer at TechCorp").await?;
memory.store("company:techcorp", "A technology company").await?;
memory.store("project:x", "Led by Alice at TechCorp").await?;

// Find relationships
let graph = memory.get_knowledge_graph().await?;
let connections = graph.find_connections("person:alice", "company:techcorp")?;
```

#### 3. Memory Types

```rust
use synaptic::MemoryType;

// Short-term memory (recent, transient)
let entry = MemoryEntry::new(
    "current_task".to_string(),
    "Writing documentation".to_string(),
    MemoryType::ShortTerm,
);

// Long-term memory (persistent, important)
let entry = MemoryEntry::new(
    "core_values".to_string(),
    "Quality, reliability, performance".to_string(),
    MemoryType::LongTerm,
);

// Working memory (active processing)
let entry = MemoryEntry::new(
    "active_conversation".to_string(),
    "Discussing Rust best practices".to_string(),
    MemoryType::Working,
);
```

#### 4. Analytics (Requires `analytics` feature)

```rust
#[cfg(feature = "analytics")]
{
    let config = MemoryConfig {
        enable_analytics: true,
        ..Default::default()
    };

    let memory = AgentMemory::new(config).await?;

    // Get memory statistics
    let stats = memory.get_analytics().await?;
    println!("Total memories: {}", stats.total_count);
    println!("Memory usage: {} bytes", stats.total_size);
}
```

### Try the Examples

The repository includes comprehensive examples:

```bash
# Basic usage
cargo run --example basic_usage

# Knowledge graph
cargo run --example knowledge_graph_usage

# Advanced features
cargo run --example phase3_analytics --features analytics

# Full system demo
cargo run --example combined_full_system --features "distributed,analytics,embeddings"
```

### Read the Docs

- [User Guide](user_guide.md) - Comprehensive usage guide
- [API Guide](api_guide.md) - Detailed API reference
- [Architecture](architecture.md) - System design and architecture
- [Deployment](deployment.md) - Production deployment guide

### Start Development Services (Optional)

If you want to use advanced features like distributed coordination or SQL storage:

```bash
# Start all services (PostgreSQL, Redis, Kafka, etc.)
docker-compose up -d

# Or use the Makefile
make docker-up

# Check service health
docker-compose ps
```

### Run Tests

```bash
# Run all tests
cargo test

# Run specific test categories
make test-critical    # Core, integration, security
make test-high        # Performance, lifecycle
make test-medium      # Temporal, analytics, search

# Run with specific features
cargo test --features analytics
cargo test --features "distributed,sql-storage"
```

## Common Patterns

### Pattern 1: Simple Key-Value Storage

```rust
let mut memory = AgentMemory::new(Default::default()).await?;

// Store
memory.store("key", "value").await?;

// Retrieve
let value = memory.retrieve("key").await?;

// Update
memory.update("key", "new value").await?;

// Delete
memory.delete("key").await?;
```

### Pattern 2: Semantic Search

```rust
// Store various memories
memory.store("doc1", "Rust programming language features").await?;
memory.store("doc2", "Python web development tutorial").await?;
memory.store("doc3", "Rust async programming guide").await?;

// Search semantically
let results = memory.search("rust", 10).await?;
// Returns doc1 and doc3 (Rust-related)
```

### Pattern 3: Temporal Tracking

```rust
let config = MemoryConfig {
    enable_temporal_tracking: true,
    ..Default::default()
};

let mut memory = AgentMemory::new(config).await?;

// Store and update
memory.store("project_status", "In progress").await?;
memory.update("project_status", "Completed").await?;

// Get version history
let history = memory.get_temporal_history("project_status").await?;
println!("Found {} versions", history.len());
```

## Troubleshooting

### Build Errors

If you encounter build errors:

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build
```

### Feature Not Available

Some features require specific feature flags:

```bash
# Analytics features
cargo build --features analytics

# Distributed features
cargo build --features distributed

# All features
cargo build --all-features
```

### Service Connection Issues

If using Docker services:

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs postgres
docker-compose logs redis

# Restart services
docker-compose restart
```

## Getting Help

- [Documentation](../docs/) - Full documentation
- [Examples](../examples/) - Code examples
- [GitHub Issues](https://github.com/njfio/rust-synaptic/issues) - Report bugs
- [Discussions](https://github.com/njfio/rust-synaptic/discussions) - Ask questions

## Next: Deep Dive

Ready to learn more? Check out:

1. [User Guide](user_guide.md) - Complete feature tour
2. [Architecture](architecture.md) - How Synaptic works
3. [API Reference](api_guide.md) - Full API documentation
4. [Contributing](../CONTRIBUTING.md) - Join development

---

**Happy coding with Synaptic! 🧠**
