# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen.svg)](https://github.com/njfio/rust-synaptic)

**Synaptic** is an intelligent AI agent memory system implemented in Rust. It provides sophisticated memory management, advanced search capabilities, knowledge graph functionality, temporal tracking, and multi-modal content processing for AI applications.

## 🚀 Key Features

- **115 Passing Tests** with comprehensive library and integration test coverage
- **Core Memory Operations** with intelligent storage, retrieval, search, and lifecycle management
- **Knowledge Graph Integration** with relationship discovery and graph-based queries
- **Advanced Search Engine** with multiple similarity algorithms and ranking strategies
- **Temporal Intelligence** with memory evolution tracking and pattern detection
- **Security Features** including AES-256-GCM encryption and access control
- **CLI Interface** with SyQL (Synaptic Query Language) support
- **Multi-Modal Processing** for documents, images, audio, and code analysis
- **External Integrations** with PostgreSQL, Redis, and ML model support

## Core Features

### Memory Operations

- **Storage & Retrieval**: Store, retrieve, update, and delete memory entries with full async support
- **Advanced Search**: Text-based search with similarity scoring and relevance ranking
- **Memory Types**: Support for short-term and long-term memory classification
- **Metadata Management**: Rich metadata including tags, importance scores, and access tracking

### Knowledge Graph

- **Relationship Discovery**: Automatic detection of relationships between memory entries
- **Graph Traversal**: Find related memories through graph connections
- **Centrality Analysis**: Calculate importance based on graph position and connectivity
- **Relationship Types**: Support for various relationship types (semantic, temporal, causal)

### Temporal Intelligence

- **Memory Evolution**: Track changes to memory entries over time using Myers' diff algorithm
- **Pattern Detection**: Identify temporal patterns in memory access and creation
- **Decay Models**: Implement forgetting curves and memory decay algorithms
- **Temporal Queries**: Search memories based on time-based criteria

### Advanced Management

- **Memory Summarization**: Intelligent consolidation of related memories
- **Lifecycle Management**: Automated archiving and cleanup policies
- **Performance Optimization**: Dynamic optimization of storage and retrieval operations
- **Analytics Engine**: Comprehensive analysis of memory usage patterns and performance

### Security & Privacy

- **AES-256-GCM Encryption**: Strong encryption for sensitive memory data
- **Access Control**: Role-based access control with security contexts
- **Audit Logging**: Comprehensive logging of all memory operations
- **Data Privacy**: Privacy-preserving operations with differential privacy support

### Multi-Modal Processing

- **Document Processing**: Extract content from PDF, Markdown, CSV, and other document formats
- **Image Analysis**: Basic image processing and feature extraction capabilities
- **Audio Processing**: Audio content analysis and transcription support
- **Code Analysis**: Parse and analyze source code using tree-sitter

## Architecture

### Storage Backends

- **In-Memory Storage**: Fast, volatile storage for development and testing
- **File-Based Storage**: Persistent storage using Sled embedded database
- **SQL Database Support**: PostgreSQL integration with connection pooling (optional feature)
- **Storage Middleware**: Compression and encryption layers for enhanced storage

### CLI Interface

- **Interactive Shell**: Full-featured shell with command history and auto-completion
- **SyQL Query Language**: SQL-like query language for memory operations
- **Memory Commands**: Direct memory management through CLI commands
- **Performance Profiling**: Built-in profiling and benchmarking tools

### External Integrations

- **Database Integration**: PostgreSQL support with async connection pooling
- **Redis Caching**: Optional Redis integration for distributed caching
- **ML Model Support**: Framework for integrating machine learning models
- **Visualization**: Basic plotting and visualization capabilities

## Implementation Status

### Test Coverage

- **115 Library Tests**: Comprehensive unit tests covering core functionality
- **1 Integration Test**: Basic integration testing for key workflows
- **15 Documentation Tests**: All code examples in documentation are tested
- **Clean Compilation**: All code compiles without errors

### Core Features Status

- **✅ Memory Operations**: Store, retrieve, update, delete operations fully implemented
- **✅ Search Engine**: Text-based search with similarity scoring working
- **✅ Knowledge Graph**: Relationship discovery and graph operations implemented
- **✅ Temporal Tracking**: Memory evolution and pattern detection functional
- **✅ Security**: AES-256-GCM encryption and access control implemented
- **✅ CLI Interface**: Interactive shell and SyQL query language working
- **✅ Storage Backends**: In-memory, file-based, and SQL storage options available

### Optional Features

- **🔧 Multi-Modal Processing**: Document, image, and audio processing (requires feature flags)
- **🔧 External Integrations**: PostgreSQL, Redis, ML models (requires feature flags)
- **🔧 Advanced Security**: Homomorphic encryption, zero-knowledge proofs (requires feature flags)
- **🔧 Distributed Features**: Kafka, clustering support (requires feature flags)

## Installation

Add Synaptic to your `Cargo.toml`:

```toml
[dependencies]
synaptic = "0.1.0"

# For additional features
synaptic = { version = "0.1.0", features = ["sql-storage", "analytics", "security"] }
```

## Quick Start

### Basic Memory Operations

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

### Using the CLI

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

### Running Tests

```bash
# Run all tests
cargo test

# Run library tests only
cargo test --lib

# Run with specific features
cargo test --features "analytics security"

# Run documentation tests
cargo test --doc

# Run integration tests
cargo test --test integration_tests
```

### Test Coverage

- **115 Library Tests**: Core functionality and unit tests
- **1 Integration Test**: End-to-end workflow testing
- **15 Documentation Tests**: Code examples in documentation
- **Clean Compilation**: All tests compile and run successfully

## Examples

### Available Examples

- **[Basic Usage](examples/basic_usage.rs)**: Core memory operations and basic functionality
- **[Knowledge Graph](examples/knowledge_graph_usage.rs)**: Relationship management and graph operations
- **[Intelligent Updates](examples/intelligent_updates.rs)**: Smart memory updates and deduplication
- **[Security Demo](examples/simple_security_demo.rs)**: Encryption and security features
- **[Real Integrations](examples/real_integrations.rs)**: External service integrations
- **[Enhanced Statistics](examples/enhanced_memory_statistics.rs)**: Memory analytics and monitoring

### Running Examples

```bash
# Basic functionality
cargo run --example basic_usage
cargo run --example knowledge_graph_usage

# Advanced features
cargo run --example intelligent_updates
cargo run --example simple_security_demo

# With optional features
cargo run --example real_integrations --features "sql-storage"
cargo run --example enhanced_memory_statistics --features "analytics"
```

## Features by Category

### Core Features (Default)

- Memory operations (store, retrieve, update, delete)
- In-memory and file-based storage
- Basic search and retrieval
- Memory metadata and tagging

### Optional Features

Enable additional functionality with feature flags:

```toml
[dependencies]
synaptic = { version = "0.1.0", features = ["analytics", "security", "sql-storage"] }
```

- **`analytics`**: Advanced memory analytics and performance monitoring
- **`security`**: AES-256-GCM encryption and access control
- **`sql-storage`**: PostgreSQL database backend support
- **`multimodal`**: Document, image, and audio processing
- **`distributed`**: Redis caching and distributed coordination
- **`ml-models`**: Machine learning model integration
- **`visualization`**: Plotting and visualization capabilities

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

### Development Commands

```bash
# Code formatting
cargo fmt

# Linting
cargo clippy --all-targets --all-features

# Generate documentation
cargo doc --all-features --no-deps --open

# Run with specific features
cargo run --features "analytics security"

# Build for release
cargo build --release
```

## API Reference

Generate and view the API documentation:

```bash
cargo doc --all-features --no-deps --open
```

Key modules:

- `synaptic::memory` - Core memory operations and types
- `synaptic::memory::storage` - Storage backend implementations
- `synaptic::memory::knowledge_graph` - Knowledge graph functionality
- `synaptic::memory::temporal` - Temporal tracking and analysis
- `synaptic::cli` - Command-line interface
- `synaptic::security` - Security and encryption features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `cargo test`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
