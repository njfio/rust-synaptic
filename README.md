# Synaptic 🧠

[![Rust](https://img.shields.io/badge/rust-1.79+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/njfio/rust-synaptic)

**Synaptic** is an intelligent AI agent memory system built in Rust that creates and manages dynamic knowledge graphs with smart content updates. Unlike traditional memory systems that create duplicate entries, Synaptic intelligently merges similar content and evolves relationships over time.

## ✨ Key Features

### 🧠 **Intelligent Memory Updates**
- **Smart Node Merging**: Automatically detects and merges similar content instead of creating duplicates
- **Content Evolution Tracking**: Monitors how memories develop and change over time
- **Relationship Dynamics**: Updates connections between memories as content evolves
- **Semantic Understanding**: Uses similarity algorithms to understand content relationships

### 🕸️ **Advanced Knowledge Graph**
- **Dynamic Relationship Detection**: Automatically discovers connections between memories
- **Graph Reasoning Engine**: Infers new relationships using transitive, symmetric, and similarity-based rules
- **Real-time Analytics**: Comprehensive graph statistics including density, connectivity, and centrality
- **Efficient Traversal**: Optimized algorithms for path finding and graph exploration

### ⏰ **Temporal Intelligence**
- **Version History**: Complete tracking of memory changes with differential analysis
- **Pattern Detection**: Identifies temporal patterns in memory access and modification
- **Change Impact Assessment**: Calculates significance of changes for relationship updates
- **Evolution Metrics**: Tracks how the knowledge graph grows and adapts

### 🔍 **Advanced Search & Retrieval**
- **Multi-criteria Search**: Content, tags, importance, temporal, and semantic search
- **Relevance Ranking**: Sophisticated scoring algorithms for result ordering
- **Fuzzy Matching**: Finds relevant memories even with partial or inexact queries
- **Context-aware Results**: Considers relationship strength and graph position

### 🎯 **Memory Management**
- **Intelligent Summarization**: Consolidates related memories while preserving information
- **Lifecycle Policies**: Automated archival, cleanup, and optimization strategies
- **Performance Optimization**: Deduplication, compression, and index optimization
- **Analytics & Insights**: Comprehensive reporting on memory usage patterns

## 🚀 Quick Start

### Installation

Add Synaptic to your `Cargo.toml`:

```toml
[dependencies]
synaptic = "0.1.0"
```

### Basic Usage

```rust
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create memory system with knowledge graph enabled
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        enable_advanced_management: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Store memories - similar content will be intelligently merged
    memory.store("project_alpha", "A web application using React and Node.js").await?;
    memory.store("team_alice", "Alice is the lead developer working on frontend").await?;
    
    // Update existing memory - will merge with existing content
    memory.store("project_alpha", "A web application using React and Node.js with real-time features").await?;
    
    // Find related memories
    let related = memory.find_related_memories("project_alpha", 5).await?;
    
    // Get knowledge graph statistics
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("Graph: {} nodes, {} edges", stats.node_count, stats.edge_count);
    }

    Ok(())
}
```

## 🏗️ Architecture

Synaptic is built with a modular architecture that separates concerns while maintaining high performance:

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentMemory (Core API)                   │
├─────────────────────────────────────────────────────────────┤
│  Memory Types  │  Knowledge Graph  │  Temporal Tracking    │
│  - Short-term  │  - Nodes & Edges  │  - Versioning         │
│  - Long-term   │  - Relationships  │  - Change Detection   │
│  - Working     │  - Reasoning      │  - Pattern Analysis   │
├─────────────────────────────────────────────────────────────┤
│  Advanced Management                                        │
│  - Search Engine  │  Summarization  │  Analytics           │
│  - Lifecycle      │  Optimization   │  Insights            │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                              │
│  - Memory Storage │  File Storage   │  Middleware          │
│  - Caching        │  Serialization  │  Compression         │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Intelligent Updates in Action

Here's how Synaptic handles memory updates intelligently:

### Traditional Approach (Creates Duplicates)
```
Memory 1: "Project Alpha - React app"
Memory 2: "Project Alpha - React app with authentication"  // Duplicate!
Memory 3: "Project Alpha - React app with auth and chat"   // Another duplicate!
```

### Synaptic Approach (Intelligent Merging)
```
Memory 1: "Project Alpha - React app"
         ↓ (Update detected, content merged)
Memory 1: "Project Alpha - React app with authentication and real-time chat"
         ↓ (Relationships updated based on new content)
    Connected to: [Authentication System, Chat Module, React Framework]
```

## 🧪 Examples

The repository includes comprehensive examples:

- **[Basic Usage](examples/basic_usage.rs)**: Getting started with core functionality
- **[Knowledge Graph](examples/knowledge_graph_usage.rs)**: Advanced graph operations
- **[Intelligent Updates](examples/intelligent_updates.rs)**: Demonstrating smart merging
- **[Simple Demo](examples/simple_intelligent_updates.rs)**: Quick demonstration

Run examples with:
```bash
cargo run --example basic_usage
cargo run --example intelligent_updates
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test module
cargo test knowledge_graph

# Run benchmarks
cargo bench
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic

# Install dependencies
cargo build

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ in Rust
- Inspired by neuroscience and cognitive science research
- Thanks to the Rust community for excellent crates and tools

## 📞 Support

- 📖 [Documentation](https://docs.rs/synaptic)
- 🐛 [Issue Tracker](https://github.com/njfio/rust-synaptic/issues)
- 💬 [Discussions](https://github.com/njfio/rust-synaptic/discussions)

---

**Synaptic** - *Intelligent Memory for AI Agents* 🧠✨
