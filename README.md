# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent memory system implemented in Rust with intelligent memory management, knowledge graphs, and temporal tracking.

## Project Status

- **Library**: Rust library with 179 passing tests
- **Version**: 0.1.0 (development)
- **Architecture**: Modular design with optional feature flags
- **Storage**: In-memory, file-based, and SQL backends
- **Testing**: Comprehensive test suite covering all modules
- **Build Status**: ✅ All tests passing, compilation successful

## Recent Improvements

- ✅ Fixed compilation errors in memory management tests
- ✅ Enhanced automatic summarization with fallback handling
- ✅ Improved test coverage for advanced memory operations
- ✅ Resolved all critical compilation issues
- ✅ Comprehensive test suite with 179 passing tests

## Core Features

### Memory Management

- Memory operations (store, retrieve, update, delete)
- Intelligent memory consolidation and summarization
- Automatic summarization with multiple strategies
- Lifecycle management with archival and cleanup
- Memory optimization and compression
- Advanced analytics and performance monitoring

### Knowledge Graph

- Dynamic relationship detection and reasoning
- Node merging for similar content
- Graph traversal and pattern analysis
- Temporal relationship tracking
- Cross-modal relationship discovery

### Storage Backends

- **Memory**: In-memory storage for development and testing
- **File**: Persistent file-based storage with Sled database
- **SQL**: PostgreSQL integration (optional feature)
- **Distributed**: Redis caching and coordination (optional)

### Security & Privacy

- AES-256-GCM encryption for sensitive data
- Access control and audit logging
- Zero-knowledge proofs (experimental)
- Homomorphic encryption support (experimental)

### Multi-Modal Processing

- **Documents**: PDF, Markdown, CSV, text files
- **Images**: Feature extraction, OCR, visual analysis
- **Audio**: Speech processing and feature extraction
- **Code**: Syntax analysis and semantic understanding

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
synaptic = "0.1.0"

# With optional features
synaptic = { version = "0.1.0", features = ["analytics", "security", "multimodal"] }
```

## Quick Start

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

## Feature Flags

### Core Features (Default)

- `core`: Basic memory operations
- `storage`: File and memory storage backends
- `bincode`: Binary serialization support

### Optional Features

- `analytics`: Memory analytics and performance monitoring
- `security`: Encryption, access control, and privacy features
- `sql-storage`: PostgreSQL database backend
- `distributed`: Redis caching and distributed coordination
- `multimodal`: Multi-modal content processing
- `external-integrations`: ML models, LLM, and visualization
- `cross-platform`: WebAssembly and mobile support

### Convenience Groups

- `full`: All features enabled
- `minimal`: Core features only

## Examples

The project includes comprehensive examples:

```bash
# Basic functionality
cargo run --example basic_usage
cargo run --example knowledge_graph_usage

# Advanced features
cargo run --example phase3_analytics --features "analytics"
cargo run --example phase4_security_privacy --features "security"
cargo run --example real_integrations --features "external-integrations"

# Multi-modal processing
cargo run --example phase5_multimodal_crossplatform --features "multimodal"
cargo run --example phase5b_document_demo --features "document-processing"
```

## Testing

```bash
# Run all tests (179 tests)
cargo test

# Run with specific features
cargo test --features "analytics security"

# Run specific test suites
cargo test --test security_suite --features "security"
cargo test --test multimodal_suite --features "multimodal"
```

## Development

### Setup

```bash
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic
cargo build
```

### Using Justfile

The project includes a Justfile for common tasks:

```bash
# Show available commands
just

# Development
just build          # Build the project
just test           # Run all tests
just test-features  # Run tests with all features
just clippy         # Run lints

# Infrastructure (requires Docker)
just setup          # Start all services
just services-up    # Start Docker services
just services-down  # Stop Docker services
```

### Project Structure

```text
src/
├── lib.rs                    # Library entry point
├── memory/                   # Core memory system
│   ├── storage/             # Storage backends
│   ├── knowledge_graph/     # Graph operations
│   ├── management/          # Memory management
│   ├── temporal/            # Temporal tracking
│   └── embeddings/          # Vector embeddings
├── analytics/               # Analytics and monitoring
├── security/                # Security and privacy
├── integrations/            # External integrations
├── multimodal/              # Multi-modal processing
├── distributed/             # Distributed systems
└── cross_platform/          # Cross-platform support
```

## Documentation

Generate and view documentation:

```bash
cargo doc --all-features --no-deps --open
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
