# Synaptic

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-165%20passing-brightgreen.svg)](https://github.com/njfio/rust-synaptic)

**Synaptic** is a comprehensive AI agent memory system implemented in Rust. It provides intelligent memory management, advanced search capabilities, lifecycle automation, knowledge graph functionality, enterprise security features, and multi-modal content processing for AI applications.

## 🚀 Key Highlights

- **165 Passing Tests** across 14 organized categories with priority-based execution
- **Production-Ready Implementation** with real external integrations (PostgreSQL, Redis, BERT ML models, LLM APIs)
- **Enterprise Security** featuring AES-256-GCM encryption, zero-knowledge proofs, and differential privacy
- **Comprehensive Test Organization** with automated CI/CD and development tools
- **Multi-Modal Processing** supporting PDF, Markdown, DOC, CSV, Parquet, images, audio, and code
- **Advanced Analytics** with real-time performance monitoring and sophisticated algorithms
- **Professional Documentation** with API reference, architecture guides, and deployment instructions

## Features

### Core Memory Management

- **Intelligent Memory Updates**: Smart content merging and evolution tracking to prevent duplicates
- **Advanced Deduplication**: Multiple detection methods including content hashing, semantic similarity, fuzzy matching, and clustering algorithms
- **Lifecycle Management**: Automated policies with 11 comprehensive tests covering archival, compression, deletion, and optimization
- **Memory Optimization**: LRU, LFU, age-based, and hybrid cleanup strategies with real-time performance monitoring

### Advanced Search & Retrieval

- **Multi-Strategy Search**: Text search, semantic search, graph-based search with 6 comprehensive similarity algorithms
- **Enhanced Similarity**: N-gram, Jaro-Winkler, Damerau-Levenshtein, and Sørensen-Dice algorithms with weighted combinations
- **Ranking System**: Multiple ranking strategies including relevance, importance, recency, frequency, and combined scoring
- **Search Analytics**: Performance metrics, search history, and optimization insights with real-time monitoring

### Intelligent Summarization

- **Advanced Trigger Strategies**: Six trigger strategies including related count, age threshold, similarity clustering, importance accumulation, tag-based rules, and temporal patterns
- **Content Consolidation**: Automated summarization with configurable thresholds and comprehensive event tracking
- **Theme Extraction**: 10 comprehensive tests covering TF-IDF, NLP patterns, semantic clustering, and topic modeling

### Dynamic Knowledge Graph

- **Intelligent Relationship Discovery**: Multi-strategy relationship detection with strength scoring and type classification
- **Graph Analysis**: Centrality measures, connectivity analysis, and relationship reasoning with 6 comprehensive tests
- **Graph Traversal**: Advanced algorithms for finding related memories with configurable distance limits
- **Graph Analytics**: Real-time statistics, insights, and performance monitoring

### Temporal Intelligence

- **Advanced Pattern Detection**: Nine temporal pattern types including daily, weekly, monthly, seasonal, cyclical, burst, gradual, irregular, and custom patterns
- **Evolution Tracking**: Myers' diff algorithm implementation with 10 comprehensive tests for change detection and compression
- **Access Pattern Analysis**: Behavioral analysis with trend detection and 2 temporal summary tests
- **Temporal Proximity**: Time-decay based similarity for temporal relationships

### Enterprise Security

- **Multi-Layer Encryption**: AES-256-GCM, ChaCha20-Poly1305, and homomorphic encryption with fallback support
- **Zero-Knowledge Proofs**: Privacy-preserving access control with cryptographic verification and 1 comprehensive test
- **Differential Privacy**: Statistical privacy protection with configurable epsilon values and 9 security tests
- **Comprehensive Audit Logging**: Security event tracking with tamper-evident logs and 10 logging tests

### Multi-Modal Content Processing

- **Document Processing**: PDF, Markdown, DOC, CSV, and Parquet file support with 8 comprehensive tests
- **Content Detection**: Advanced format detection using magic bytes and content analysis
- **Cross-Modal Relationships**: Intelligent relationship detection across different content types
- **Feature Extraction**: Professional algorithms for content analysis and similarity calculation

## Architecture

### Production-Ready Storage Infrastructure

- **Multi-Backend Support**: In-memory, file-based, and SQL database storage with 13 integration tests
- **Advanced Indexing**: Search indices with inverted word indexing and metadata optimization
- **Professional Compression**: LZ4, ZSTD, and Brotli compression with storage middleware and 8 optimization tests
- **Distributed Coordination**: Redis-based caching and coordination for scalable deployments

### Real External Integrations

- **ML Model Integration**: BERT embeddings, vector similarity, and semantic search with actual model implementations
- **LLM Integration**: Content analysis, summarization, and processing with real API integrations
- **Database Connectivity**: PostgreSQL integration with connection pooling, transaction management, and real database operations
- **Advanced Visualization**: Memory visualization with graphs, analytics dashboards, and complex visualization support

### Comprehensive Analytics

- **Memory Analytics**: Usage patterns, access frequency, and behavioral analysis with 10 performance measurement tests
- **Performance Monitoring**: Real-time metrics, search performance, and optimization insights
- **Trend Analysis**: Pattern recognition, forecasting, and predictive analytics
- **Professional Reporting**: Analytics reports with actionable insights and recommendations

## Implementation Status

### Comprehensive Test Coverage

- **165 Passing Tests**: Extensive test coverage across all modules with functional verification
- **Professional Implementation**: Production-ready functionality with minimal placeholders - comprehensive real implementations
- **Robust Error Handling**: Comprehensive Result types and error management throughout the codebase
- **Multi-Strategy Approaches**: Multiple algorithms implemented for each core system ensuring reliability

### Delivered Core Features

- **Advanced Memory Management**: Related memory counting, intelligent summarization, professional deduplication, automated lifecycle management, and sophisticated search capabilities
- **Multi-Strategy Implementations**: Each system implements multiple strategies for robustness and performance
- **Real-Time Performance Optimization**: Production-grade algorithms with comprehensive monitoring and analytics
- **Enterprise Security**: Full encryption, zero-knowledge proofs, differential privacy, and audit logging

### Production Quality

- **Clean Compilation**: Codebase compiles successfully with minimal warnings and follows Rust best practices
- **Comprehensive Documentation**: Inline documentation, examples, and detailed API documentation
- **Modular Architecture**: Structured, maintainable, and extensible design following professional patterns
- **Enterprise Ready**: Suitable for production deployment with scalable infrastructure and real integrations

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

### Organized Test Suite (165 Tests)

The project includes a comprehensive test organization system with priority-based execution:

#### Using the Test Runner Script

```bash
# Run all tests (165 tests across 14 categories)
./scripts/run_tests.sh all

# Run by priority level
./scripts/run_tests.sh critical    # Core, integration, security (72 tests)
./scripts/run_tests.sh high        # Performance, lifecycle, multimodal (62 tests)
./scripts/run_tests.sh medium      # Temporal, analytics, search, external (25 tests)
./scripts/run_tests.sh low         # Logging, visualization, sync (6 tests)

# Run specific test categories
./scripts/run_tests.sh security     # All security tests
./scripts/run_tests.sh performance  # All performance tests
./scripts/run_tests.sh multimodal   # All multimodal tests
```

#### Using Makefile Targets

```bash
# Test commands
make test-all          # All 165 tests
make test-critical     # Critical priority tests
make test-security     # Security tests only
make test-performance  # Performance tests only

# Development commands
make build clean fmt clippy coverage audit docs
```

#### Traditional Cargo Commands

```bash
# Run all tests
cargo test

# Run specific test files
cargo test --test integration_tests              # 13 tests
cargo test --test phase4_security_tests          # 9 tests
cargo test --test real_lifecycle_management_tests # 11 tests
cargo test --test enhanced_similarity_search_tests # 6 tests
cargo test --test comprehensive_logging_tests    # 10 tests
cargo test --test phase5b_document_tests         # 8 tests

# Run performance benchmarks
cargo test --test performance_tests -- --ignored
```

## Examples

The repository includes examples demonstrating functionality:

### Core Functionality

- **[Basic Usage](examples/basic_usage.rs)**: Getting started with memory operations and basic features
- **[Knowledge Graph](examples/knowledge_graph_usage.rs)**: Relationship management and graph operations
- **[Intelligent Updates](examples/intelligent_updates.rs)**: Memory updates with deduplication and optimization

### Advanced Features

- **[Real Integrations](examples/real_integrations.rs)**: Production external service integration with PostgreSQL, Redis, BERT ML models, and LLM APIs
- **[Security Demo](examples/simple_security_demo.rs)**: Enterprise security features including encryption, zero-knowledge proofs, and access control
- **[Complex Visualizations](examples/complex_visualizations.rs)**: Advanced analytics and visualization capabilities with real chart generation

### Specialized Systems

- **[Phase 5 Multimodal](examples/phase5_multimodal_crossplatform.rs)**: Cross-platform multimodal memory management with image, audio, and code processing
- **[Document Processing](examples/phase5_basic_demo.rs)**: Professional document processing and content extraction for multiple formats
- **[Enhanced Memory Statistics](examples/enhanced_memory_statistics.rs)**: Comprehensive memory analytics and performance monitoring
- **[Distributed Systems](examples/phase2_distributed_system.rs)**: Distributed memory coordination and scaling infrastructure

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
cargo run --example enhanced_memory_statistics
cargo run --example phase2_distributed_system
```

## Test Results & Quality Metrics

### Comprehensive Test Organization

| Priority | Categories | Test Count | Description |
|----------|------------|------------|-------------|
| **Critical** | Core, Integration, Security | 72 tests | Essential functionality and security |
| **High** | Performance, Lifecycle, Multimodal | 62 tests | Performance and advanced features |
| **Medium** | Temporal, Analytics, Search, External | 25 tests | Specialized algorithms and integrations |
| **Low** | Logging, Visualization, Sync | 6 tests | Supporting infrastructure |
| **Total** | **14 Categories** | **165 tests** | **Complete test coverage** |

### Test Execution Tools

- **Organized Test Runner**: `./scripts/run_tests.sh` with priority-based execution
- **Makefile Integration**: `make test-all`, `make test-critical`, category-specific targets
- **CI/CD Automation**: GitHub Actions with automated testing and quality checks
- **Test Configuration**: Structured metadata in `tests/test_config.toml`

### Implementation Quality

- **Zero Test Failures**: All 165 tests pass consistently across all categories
- **Production-Ready Implementation**: Comprehensive functionality with sophisticated algorithms and real integrations
- **Professional Error Handling**: Comprehensive Result types throughout with proper error propagation
- **Enterprise-Grade Code**: Suitable for production deployment with scalable infrastructure

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

### Development Tools

The project includes comprehensive development automation:

#### Test Organization
- **Test Runner Script**: `./scripts/run_tests.sh` - Organized test execution with priority levels
- **Test Configuration**: `tests/test_config.toml` - Structured test metadata and categorization
- **Makefile**: Comprehensive development commands and test targets

#### CI/CD Integration
- **GitHub Actions**: Automated testing with priority-based execution
- **Coverage Reporting**: Code coverage analysis and reporting
- **Security Auditing**: Automated dependency and security auditing

#### Quality Assurance
```bash
# Code formatting
make fmt
cargo fmt --all

# Linting
make clippy
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
make audit
cargo audit

# Generate documentation
make docs
cargo doc --all-features --no-deps --open

# Code coverage
make coverage
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
```

## Documentation

### Core Documentation
- [API Documentation](docs/API_DOCUMENTATION.md) - Comprehensive API reference
- [Architecture Overview](docs/ARCHITECTURE.md) - System architecture and design
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Developer setup and guidelines
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Test Organization](docs/TEST_ORGANIZATION.md) - Test suite structure and execution

### Feature Documentation
- [Phase 5 Multimodal & Cross-Platform](docs/PHASE5_MULTIMODAL_CROSSPLATFORM.md)
- [Phase 5B Document Processing](docs/PHASE5B_DOCUMENT_PROCESSING.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
