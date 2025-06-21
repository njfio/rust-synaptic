# 🚀 Embedding Providers Guide (Late 2024)

## Overview

The embedding landscape has evolved significantly in 2024. While OpenAI was previously the go-to choice, specialized embedding companies now lead the MTEB (Massive Text Embedding Benchmark) leaderboard with superior performance.

## 🏆 Current Top Performers

### 1. 🥇 Voyage AI (Recommended)
- **Models**: `voyage-large-2-instruct`, `voyage-3-large`
- **MTEB Score**: ~69.5-70.0 (Top performer)
- **Dimensions**: 1024-2048
- **Strengths**: 
  - Highest MTEB benchmark scores
  - Instruction-tuned for better retrieval
  - Specialized embedding company focus
- **Cost**: ~$0.12-0.15 per 1K tokens
- **Setup**: `export VOYAGE_API_KEY="your-key"`

### 2. 🥈 Cohere (Best Value)
- **Models**: `embed-english-v3.0`, `embed-english-light-v3.0`
- **MTEB Score**: ~68.0
- **Dimensions**: 1024
- **Strengths**:
  - Excellent cost/performance ratio
  - Strong retrieval performance
  - Good multilingual support
- **Cost**: ~$0.10 per 1K tokens
- **Setup**: `export COHERE_API_KEY="your-key"`

### 3. 🥉 OpenAI (Widely Supported)
- **Models**: `text-embedding-3-large` (updated from 3-small)
- **MTEB Score**: ~64.6
- **Dimensions**: 3072 (can be reduced)
- **Strengths**:
  - Widely supported ecosystem
  - High dimensional embeddings
  - Reliable infrastructure
- **Cost**: ~$0.13 per 1K tokens
- **Setup**: `export OPENAI_API_KEY="your-key"`

## 📊 Performance Comparison

| Provider | Model | MTEB Score | Dimensions | Cost/1K | Best For |
|----------|-------|------------|------------|---------|----------|
| Voyage AI | voyage-large-2-instruct | 69.5 | 1024 | $0.12 | Best performance |
| Voyage AI | voyage-3-large | 70.0 | 2048 | $0.15 | Highest quality |
| Cohere | embed-english-v3.0 | 68.0 | 1024 | $0.10 | Best value |
| OpenAI | text-embedding-3-large | 64.6 | 3072 | $0.13 | Ecosystem support |
| OpenAI | text-embedding-3-small | 62.3 | 1536 | $0.02 | High volume |

## 🔧 Configuration

### Automatic Provider Selection

The system automatically selects the best available provider:

```rust
// Priority: Voyage AI > Cohere > OpenAI > Simple TF-IDF
let config = EmbeddingConfig::default();
let manager = EmbeddingManager::new(config)?;
```

### Manual Provider Configuration

```rust
// Voyage AI (Best Performance)
let voyage_config = VoyageAIConfig {
    api_key: env::var("VOYAGE_API_KEY")?,
    model: "voyage-large-2-instruct".to_string(),
    embedding_dim: 1024,
    ..Default::default()
};

let config = EmbeddingConfig {
    provider: EmbeddingProvider::VoyageAI,
    voyage_config: Some(voyage_config),
    ..Default::default()
};

// Cohere (Best Value)
let cohere_config = CohereConfig {
    api_key: env::var("COHERE_API_KEY")?,
    model: "embed-english-v3.0".to_string(),
    embedding_dim: 1024,
    ..Default::default()
};

let config = EmbeddingConfig {
    provider: EmbeddingProvider::Cohere,
    cohere_config: Some(cohere_config),
    ..Default::default()
};

// OpenAI (Updated to 3-large)
let openai_config = OpenAIEmbeddingConfig {
    api_key: env::var("OPENAI_API_KEY")?,
    model: "text-embedding-3-large".to_string(), // Updated!
    embedding_dim: 3072, // Can be reduced to 1536 if needed
    ..Default::default()
};
```

## 🎯 Recommendations by Use Case

### Maximum Performance
- **Provider**: Voyage AI
- **Model**: `voyage-3-large`
- **Why**: Highest MTEB scores, latest technology

### Best Value
- **Provider**: Cohere
- **Model**: `embed-english-v3.0`
- **Why**: Excellent performance at lower cost

### High Volume / Cost Sensitive
- **Provider**: OpenAI
- **Model**: `text-embedding-3-small`
- **Why**: Very low cost at $0.02/1K tokens

### Enterprise / Ecosystem
- **Provider**: OpenAI
- **Model**: `text-embedding-3-large`
- **Why**: Widely supported, reliable infrastructure

## 🔄 Migration Guide

### From OpenAI 3-small to Better Models

```rust
// Old configuration (suboptimal)
let old_config = OpenAIEmbeddingConfig {
    model: "text-embedding-3-small".to_string(),
    embedding_dim: 1536,
    ..Default::default()
};

// New recommended configurations
// Option 1: Upgrade to OpenAI 3-large
let openai_large = OpenAIEmbeddingConfig {
    model: "text-embedding-3-large".to_string(),
    embedding_dim: 3072, // Better performance
    ..Default::default()
};

// Option 2: Switch to Voyage AI (best performance)
let voyage_config = VoyageAIConfig {
    model: "voyage-large-2-instruct".to_string(),
    embedding_dim: 1024,
    ..Default::default()
};

// Option 3: Switch to Cohere (best value)
let cohere_config = CohereConfig {
    model: "embed-english-v3.0".to_string(),
    embedding_dim: 1024,
    ..Default::default()
};
```

## 📈 Performance Impact

Upgrading from `text-embedding-3-small` to top performers:

- **Voyage AI**: +7-8 MTEB points (~12% improvement)
- **Cohere**: +5-6 MTEB points (~9% improvement)  
- **OpenAI 3-large**: +2-3 MTEB points (~4% improvement)

## 🚀 Getting Started

1. **Choose your provider** based on needs and budget
2. **Set environment variable** for your chosen provider
3. **Update configuration** or use automatic selection
4. **Test performance** with your specific use case

```bash
# For best performance
export VOYAGE_API_KEY="your-voyage-key"

# For best value
export COHERE_API_KEY="your-cohere-key"

# For ecosystem compatibility (updated model)
export OPENAI_API_KEY="your-openai-key"
```

## 📚 References

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Cohere Embeddings Guide](https://docs.cohere.com/docs/embeddings)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

*Last updated: December 2024*
*Based on MTEB leaderboard and community benchmarks*
