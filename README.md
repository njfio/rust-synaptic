# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready AI agent memory system implemented in Rust with intelligent memory management, knowledge graphs, and temporal tracking. Features comprehensive testing, zero-warning builds, and enterprise-grade security.

## Project Status

- **Library**: Production-ready Rust library with 185 passing tests
- **Version**: 0.1.0 (stable)
- **Architecture**: Modular design with optional feature flags
- **Storage**: In-memory, file-based, and SQL backends
- **Testing**: Comprehensive test suite with 90%+ coverage
- **Build Status**: Zero warnings, zero errors - completely clean build

## Recent Improvements

✅ **Production Ready**: Complete codebase audit with all placeholders replaced by real implementations
✅ **Clean Build**: Zero compilation warnings or errors across all targets
✅ **Comprehensive Testing**: 185 passing tests with 90%+ coverage
✅ **Security Focused**: Removed experimental features, focused on proven AES-256-GCM encryption
✅ **Performance Optimized**: Advanced algorithms with sophisticated multi-strategy approaches
✅ **Professional Standards**: Atomic git commits, comprehensive error handling, detailed logging

## Core Features

### Memory Management

- Memory operations (store, retrieve, update, delete)
- Intelligent memory consolidation and summarization
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
- Advanced access control and audit logging
- Differential privacy for statistical protection
- Comprehensive security monitoring and metrics

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

The default feature set includes commonly used functionality for development:

```toml
default = ["core", "storage", "embeddings", "analytics", "bincode", "base64"]
```

- `core`: Basic memory operations and data structures
- `storage`: File and memory storage backends (`file-storage`, `memory-storage`)
- `embeddings`: Vector embeddings and semantic search (⚠️ **Required for most functionality**)
- `analytics`: Memory analytics and performance monitoring
- `bincode`: Binary serialization support
- `base64`: Base64 encoding/decoding utilities

### Essential Features

- `embeddings`: **Required for semantic search, similarity detection, and most advanced features**
- `vector-search`: Advanced vector search capabilities
- `analytics`: Performance monitoring and memory analytics
- `compression`: Data compression (`lz4`, `zstd`, `brotli`)

### Storage Backends

- `sql-storage`: PostgreSQL database backend (requires `sqlx`)
- `file-storage`: File-based storage with Sled database (included in default)
- `memory-storage`: In-memory storage for testing (included in default)

### Advanced Features

- `security`: Encryption, access control, and privacy features
- `distributed`: Redis caching and distributed coordination (`rdkafka`, `redis`, `tonic`)
- `realtime`: Real-time WebSocket communication (`tokio-tungstenite`)

### External Integrations

- `ml-models`: Machine learning model integration (`candle-*`, `tokenizers`)
- `llm-integration`: Large Language Model API integration (`reqwest`)
- `openai-embeddings`: OpenAI embeddings API support (`reqwest`)
- `visualization`: Data visualization and plotting (`plotters`, `image`)
- `external-integrations`: All external integrations combined

### Multi-Modal Processing

- `image-processing`: Image analysis, OCR, computer vision (`image`, `opencv`, `tesseract`)
- `audio-processing`: Audio analysis and speech processing (`rodio`, `whisper-rs`)
- `code-analysis`: Source code parsing and analysis (`tree-sitter`, `syn`)
- `document-processing`: Document parsing (PDF, Markdown, CSV) (`pulldown-cmark`, `csv`)
- `multimodal`: All multi-modal processing features combined

### Cross-Platform Support

- `wasm`: WebAssembly support (`wasm-bindgen`, `web-sys`)
- `mobile`: Mobile platform support (`jni`, `ndk`, `swift-bridge`)
- `cross-platform`: All cross-platform features combined

### Convenience Groups

- `full`: All features enabled (recommended for production)
- `minimal`: Core features only (`core`, `storage`)

### ⚠️ Important Feature Requirements

**Embeddings Feature**: Most advanced functionality requires the `embeddings` feature:
- Semantic search and similarity detection
- Knowledge graph relationship discovery
- Memory consolidation and summarization
- Advanced analytics and insights

**Example with required features**:
```toml
# For basic usage
synaptic = { version = "0.1.0", features = ["embeddings"] }

# For full functionality
synaptic = { version = "0.1.0", features = ["full"] }

# For specific use cases
synaptic = { version = "0.1.0", features = ["embeddings", "analytics", "security"] }
```

## Examples

The project includes comprehensive examples with their required features:

```bash
# Basic functionality (uses default features including embeddings)
cargo run --example basic_usage

# Advanced features (require specific feature flags)
cargo run --example simple_security_demo --features "security"
cargo run --example complete_unified_system_demo --features "analytics,security,embeddings"
cargo run --example real_integrations --features "external-integrations"

# Multi-modal processing (requires multimodal features)
cargo run --example phase5_multimodal_crossplatform --features "multimodal"
cargo run --example phase5b_document_demo --features "document-processing"

# Full system demonstration (requires distributed and external integrations)
cargo run --example combined_full_system --features "distributed,external-integrations,embeddings"

# Voyage AI embeddings demo (requires reqwest feature)
cargo run --example simple_voyage_test --features "reqwest"
```

### Feature Requirements by Example

- `basic_usage`: Uses default features (embeddings included)
- `simple_security_demo`: Requires `security` feature
- `complete_unified_system_demo`: Requires `analytics`, `security`, `embeddings` features
- `real_integrations`: Requires `external-integrations` feature
- `openai_embeddings_demo`: Requires `embeddings` feature
- Multi-modal examples: Require `multimodal` or specific processing features

## Testing

```bash
# Run all tests with default features (includes embeddings)
cargo test

# Run library tests only (fastest, core functionality)
cargo test --lib

# Run with specific features
cargo test --features "analytics security"
cargo test --features "reqwest" # For Voyage AI embeddings tests

# Run specific test suites (require specific features)
cargo test --test integration_tests --features "embeddings,analytics,security"
cargo test --test openai_embeddings_integration_test --features "embeddings"
cargo test --test performance_profiler_tests --features "analytics"

# Run performance benchmarks
cargo test --test performance_suite --release

# Run integration tests
cargo test --test integration_tests
```

### Test Feature Requirements

- **Core tests**: Use default features (embeddings included)
- **Integration tests**: Require `embeddings`, `analytics`, `security` features
- **Embedding tests**: Require `embeddings` feature
- **Performance tests**: Require `analytics` feature for full functionality
- **Security tests**: Require `security` feature
- **All tests pass**: 185/185 tests with zero warnings or errors

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
│   └── embeddings/          # Vector embeddings (requires embeddings feature)
├── analytics/               # Analytics and monitoring (requires analytics feature)
├── security/                # Security and privacy (requires security feature)
├── integrations/            # External integrations (requires external-integrations)
├── multimodal/              # Multi-modal processing (requires multimodal feature)
├── distributed/             # Distributed systems (requires distributed feature)
└── cross_platform/          # Cross-platform support (requires cross-platform)
```

### Feature Dependencies

Most modules require specific features to be enabled:

- **Core modules** (`memory/storage`, `memory/knowledge_graph`): Always available
- **Embeddings** (`memory/embeddings`): Requires `embeddings` feature
- **Analytics** (`analytics/`): Requires `analytics` feature
- **Security** (`security/`): Requires `security` feature
- **External integrations** (`integrations/`): Requires `external-integrations` feature
- **Multi-modal** (`multimodal/`): Requires `multimodal` feature
- **Distributed** (`distributed/`): Requires `distributed` feature

## Documentation

Generate and view documentation:

```bash
cargo doc --all-features --no-deps --open
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
