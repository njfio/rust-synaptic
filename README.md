# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent memory system implemented in Rust with memory management, knowledge graphs, temporal tracking, and CLI interface.

## Overview

- Memory operations (store, retrieve, update, delete)
- Knowledge graph with relationship discovery
- Advanced search with similarity algorithms
- Temporal memory evolution tracking
- AES-256-GCM encryption and access control
- CLI with SyQL query language
- Multiple storage backends (memory, file, SQL)
- Multi-modal content processing (documents, images, audio)

## Installation

```toml
[dependencies]
synaptic = "0.1.0"

# With optional features
synaptic = { version = "0.1.0", features = ["analytics", "security", "sql-storage"] }
```

## Usage

```rust
use synaptic::memory::{MemoryEntry, MemoryType};
use synaptic::memory::storage::create_storage;
use synaptic::StorageBackend;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create storage backend
    let storage = create_storage(StorageBackend::Memory).await?;

    // Create a memory entry
    let entry = MemoryEntry::new(
        "user_preference".to_string(),
        "Dark mode enabled".to_string(),
        MemoryType::ShortTerm,
    );

    // Store the memory
    storage.store(&entry).await?;

    // Retrieve the memory
    if let Some(retrieved) = storage.retrieve("user_preference").await? {
        println!("Retrieved: {}", retrieved.value);
    }

    // Search for memories
    let results = storage.search("dark mode", 10).await?;
    println!("Found {} related memories", results.len());

    Ok(())
}
```

### CLI Usage

```bash
# Start interactive shell
synaptic shell

# Execute SyQL queries
synaptic query "SELECT * FROM memories WHERE type = 'long_term' LIMIT 5"

# Memory operations
synaptic memory list --limit 10
synaptic memory show --id "user_preference"
```

## Testing

```bash
# Run all tests
cargo test

# Run with features
cargo test --features "test-utils analytics security"
```

## Features

### Core (Default)
- Memory operations (store, retrieve, update, delete)
- In-memory and file-based storage
- Basic search and retrieval
- Memory metadata and tagging

### Optional (Feature Flags)
- `analytics`: Memory analytics and performance monitoring
- `security`: AES-256-GCM encryption and access control
- `sql-storage`: PostgreSQL database backend
- `multimodal`: Document, image, and audio processing
- `distributed`: Redis caching and coordination
- `ml-models`: Machine learning model integration

## Examples

```bash
# Basic functionality
cargo run --example basic_usage
cargo run --example knowledge_graph_usage

# With features
cargo run --example real_integrations --features "sql-storage"
cargo run --example enhanced_memory_statistics --features "analytics"
```

## Development

```bash
# Clone and build
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic
cargo build

# Run tests
cargo test

# Generate documentation
cargo doc --all-features --no-deps --open
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
