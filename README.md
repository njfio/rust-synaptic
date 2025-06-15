# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Synaptic is an AI agent memory system built in Rust. It provides persistent storage, knowledge graph relationships, and memory optimization features for AI applications.

## Features

### Core Memory Management
- Persistent memory storage with multiple backend options (in-memory, file-based, SQL)
- Memory deduplication with content similarity detection
- Memory compression using LZ4, ZSTD, and Brotli algorithms
- Intelligent cleanup with LRU, LFU, age-based, and hybrid strategies
- Memory optimization and analytics

### Knowledge Graph System
- Node and edge management for storing relationships between memories
- Graph traversal and querying capabilities
- Basic relationship discovery and analytics
- Graph statistics and insights

### Temporal Memory Features
- Memory versioning and history tracking
- Access pattern detection and analysis
- Memory evolution tracking over time
- Differential change detection

### Security Features
- Basic encryption for data at rest
- Access control mechanisms
- Audit logging for security events
- Key management utilities

## Architecture

### Storage Backends
- In-memory storage for development and testing
- File-based storage for single-node deployments
- SQL database integration (PostgreSQL support)

### External Integrations
- Vector embeddings using ML models
- LLM integration for content analysis
- Redis caching for performance optimization
- Basic visualization capabilities

### Analytics
- Memory usage pattern analysis
- Performance monitoring and metrics
- Basic behavioral analysis features

## Quick Start

### Installation

Add Synaptic to your `Cargo.toml`:

```toml
[dependencies]
synaptic = "0.1.0"
```

### Basic Usage

```rust
use synaptic::{AgentMemory, MemoryConfig, MemoryEntry, MemoryType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory system with basic configuration
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;

    // Store a memory entry
    let entry = MemoryEntry::new(
        "project_alpha".to_string(),
        "AI project using machine learning for data analysis".to_string(),
        MemoryType::LongTerm
    );
    memory.store("project_alpha", entry).await?;

    // Retrieve the memory
    let retrieved = memory.retrieve("project_alpha").await?;
    println!("Retrieved: {:?}", retrieved);

    // Search for related memories
    let results = memory.search("machine learning", 5).await?;
    println!("Found {} related memories", results.len());

    Ok(())
}
```

## Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific test modules
cargo test --test knowledge_graph_tests
cargo test --test integration_tests
```

## Examples

The repository includes examples demonstrating core functionality:

- **[Basic Usage](examples/basic_usage.rs)**: Getting started with memory operations
- **[Knowledge Graph](examples/knowledge_graph_usage.rs)**: Working with memory relationships
- **[Interactive Demo](examples/interactive_demo.rs)**: Command-line interface demo

Run examples with:

```bash
# Basic functionality
cargo run --example basic_usage

# Knowledge graph operations
cargo run --example knowledge_graph_usage

# Interactive demo
cargo run --example interactive_demo
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic

# Build the project
cargo build

# Run tests
cargo test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
