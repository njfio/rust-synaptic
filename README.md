# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)](https://github.com/njfio/rust-synaptic)

**Synaptic** is an AI agent memory system implemented in Rust. It provides memory management, search capabilities, lifecycle automation, and knowledge graph functionality for AI applications.

## Features

### Memory Management

- **Related Memory Counting**: Multiple algorithms including knowledge graph traversal, similarity matching, tag-based relationships, temporal proximity, and content similarity
- **Memory Deduplication**: Detection methods using content hashing, normalized content comparison, semantic similarity, fuzzy string matching, and clustering
- **Lifecycle Management**: Automated policies with configurable conditions and actions for archival, compression, and retention
- **Memory Optimization**: LRU, LFU, age-based, and hybrid cleanup strategies with performance monitoring

### Search Capabilities

- **Multi-Modal Search**: Text search, semantic search, and graph-based search with filtering options
- **Ranking System**: Multiple ranking strategies including relevance, importance, recency, frequency, and combined scoring
- **Similarity Algorithms**: Cosine similarity, Jaccard similarity, Levenshtein distance, and multi-factor memory similarity
- **Search Analytics**: Performance metrics, search history, and optimization insights

### Summarization System

- **Trigger Strategies**: Six trigger strategies including related count, age threshold, similarity clustering, importance accumulation, tag-based rules, and temporal patterns
- **Content Consolidation**: Automated summarization with configurable thresholds and event tracking
- **Content Analysis**: Entity extraction, theme identification, and quality metrics calculation

### Knowledge Graph

- **Relationship Discovery**: Multi-strategy relationship detection with strength scoring and type classification
- **Graph Analysis**: Centrality measures and connectivity analysis within the knowledge graph
- **Graph Traversal**: Algorithms for finding related memories with configurable distance limits
- **Graph Analytics**: Statistics, insights, and performance monitoring

### Temporal Features

- **Pattern Detection**: Nine temporal pattern types including daily, weekly, monthly, seasonal, cyclical, burst, gradual, irregular, and custom patterns
- **Evolution Tracking**: Change detection with differential analysis and compression
- **Access Pattern Analysis**: Behavioral analysis with trend detection
- **Temporal Proximity**: Time-decay based similarity for temporal relationships

### Security Features

- **Encryption**: AES-256, ChaCha20-Poly1305, and homomorphic encryption support
- **Zero-Knowledge Proofs**: Privacy-preserving access control with cryptographic verification
- **Differential Privacy**: Statistical privacy protection with configurable epsilon values
- **Audit Logging**: Security event tracking with tamper-evident logs

## Architecture

### Storage Infrastructure

- **Multi-Backend Support**: In-memory, file-based, and SQL database storage options
- **Indexing**: Search indices with inverted word indexing and metadata optimization
- **Compression**: LZ4, ZSTD, and Brotli compression with storage middleware
- **Distributed Coordination**: Redis-based caching and coordination

### External Integrations

- **ML Model Integration**: BERT embeddings, vector similarity, and semantic search
- **LLM Integration**: Content analysis, summarization, and processing
- **Database Connectivity**: PostgreSQL integration with connection pooling and transaction management
- **Visualization**: Memory visualization with graphs and analytics dashboards

### Analytics

- **Memory Analytics**: Usage patterns, access frequency, and behavioral analysis
- **Performance Monitoring**: Metrics, search performance, and optimization insights
- **Trend Analysis**: Pattern recognition and forecasting
- **Reporting**: Analytics reports with insights and recommendations

## Implementation Status

### Completed Systems

- **29 Passing Tests**: Test coverage with functional verification
- **Real Implementations**: No mocks or placeholder code in core systems
- **Error Handling**: Result types and error management throughout
- **Multi-Strategy Approaches**: Multiple algorithms for core systems

### Core Features

- **Memory Management**: Related memory counting, summarization triggering, memory deduplication, lifecycle management, search capabilities
- **Multi-Strategy Implementations**: Each system implements multiple strategies for robustness
- **Performance Optimization**: Algorithms with performance monitoring and analytics
- **Security**: Encryption, zero-knowledge proofs, and differential privacy

### Technical Quality

- **Clean Compilation**: Codebase compiles with minimal warnings
- **Documentation**: Inline documentation and examples
- **Modular Architecture**: Structured, maintainable, and extensible design
- **Production Ready**: Suitable for deployment with scalable infrastructure

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

The repository includes examples demonstrating functionality:

### Core Functionality

- **[Basic Usage](examples/basic_usage.rs)**: Getting started with memory operations and basic features
- **[Knowledge Graph](examples/knowledge_graph_usage.rs)**: Relationship management and graph operations
- **[Intelligent Updates](examples/intelligent_updates.rs)**: Memory updates with deduplication and optimization

### Advanced Features

- **[Real Integrations](examples/real_integrations.rs)**: External service integration with PostgreSQL, Redis, and ML models
- **[Security Demo](examples/simple_security_demo.rs)**: Security features and encryption
- **[Complex Visualizations](examples/complex_visualizations.rs)**: Analytics and visualization capabilities

### Specialized Systems

- **[Phase 5 Multimodal](examples/phase5_multimodal_crossplatform.rs)**: Cross-platform multimodal memory management
- **[Document Processing](examples/phase5_basic_demo.rs)**: Document processing and content extraction
- **[Distributed Systems](examples/phase2_distributed_system.rs)**: Distributed memory coordination and scaling

Run examples with:

```bash
# Core functionality
cargo run --example basic_usage
cargo run --example knowledge_graph_usage
cargo run --example intelligent_updates

# Advanced features
cargo run --example real_integrations
cargo run --example simple_security_demo
cargo run --example complex_visualizations

# Specialized systems
cargo run --example phase5_multimodal_crossplatform
cargo run --example phase5_basic_demo
cargo run --example phase2_distributed_system
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
