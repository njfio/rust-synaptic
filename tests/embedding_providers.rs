//! Integration tests for embedding providers and dense vector retrieval
//!
//! These tests verify that embedding providers work correctly with the
//! retrieval pipeline and can be used for semantic search.

use rust_synaptic::{
    AgentMemory, MemoryConfig, StorageBackend,
    memory::{
        embeddings::{
            EmbeddingProvider, TfIdfProvider, TfIdfConfig, Embedding,
            EmbedOptions, EmbeddingCache, CacheStats, compute_content_hash,
            normalize_vector,
        },
        retrieval::{
            RetrievalPipeline, HybridRetriever, PipelineConfig, FusionStrategy,
            KeywordRetriever, DenseVectorRetriever, RetrievalSignal,
        },
    },
};
use std::sync::Arc;

#[tokio::test]
async fn test_tfidf_provider_basic() {
    let provider = TfIdfProvider::default();

    assert_eq!(provider.name(), "TfIdfProvider");
    assert!(provider.is_available());
    assert_eq!(provider.embedding_dimension(), 384);

    let text = "rust programming language";
    let embedding = provider.embed(text, None).await.unwrap();

    assert_eq!(embedding.vector.len(), 384);
    assert_eq!(embedding.model, provider.model_id());
}

#[tokio::test]
async fn test_tfidf_provider_with_options() {
    let provider = TfIdfProvider::default();

    let text = "machine learning artificial intelligence";
    let options = EmbedOptions {
        normalize: true,
        ..Default::default()
    };

    let embedding = provider.embed(text, Some(&options)).await.unwrap();

    // Check normalization (L2 norm should be ~1.0)
    let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Normalized vector should have L2 norm ≈ 1.0");
}

#[tokio::test]
async fn test_tfidf_provider_custom_config() {
    let config = TfIdfConfig {
        embedding_dim: 512,
        ..Default::default()
    };
    let provider = TfIdfProvider::with_config(config);

    let embedding = provider.embed("test", None).await.unwrap();
    assert_eq!(embedding.vector.len(), 512);
}

#[tokio::test]
async fn test_embedding_cosine_similarity() {
    let provider = TfIdfProvider::default();

    let text1 = "machine learning neural networks";
    let text2 = "machine learning deep learning";
    let text3 = "cooking recipes food";

    let emb1 = provider.embed(text1, None).await.unwrap();
    let emb2 = provider.embed(text2, None).await.unwrap();
    let emb3 = provider.embed(text3, None).await.unwrap();

    // Similar texts should have higher similarity
    let sim_12 = emb1.cosine_similarity(&emb2);
    let sim_13 = emb1.cosine_similarity(&emb3);

    assert!(sim_12 > sim_13, "Similar texts should have higher similarity score");
    assert!(sim_12 > 0.5, "Related texts should have positive similarity");
}

#[tokio::test]
async fn test_embedding_identical_content() {
    let provider = TfIdfProvider::default();

    let text = "rust programming systems language";
    let emb1 = provider.embed(text, None).await.unwrap();
    let emb2 = provider.embed(text, None).await.unwrap();

    let similarity = emb1.cosine_similarity(&emb2);
    assert!((similarity - 1.0).abs() < 0.01, "Identical content should have similarity ≈ 1.0");
}

#[tokio::test]
async fn test_embedding_content_hash() {
    let provider = TfIdfProvider::default();

    let text = "test content";
    let embedding = provider.embed(text, None).await.unwrap();

    let expected_hash = compute_content_hash(text);
    assert_eq!(embedding.content_hash, expected_hash);
}

#[tokio::test]
async fn test_embedding_metadata() {
    let provider = TfIdfProvider::default();

    let embedding = provider.embed("test", None).await.unwrap();

    assert!(!embedding.model.is_empty());
    assert!(embedding.version.is_some());
    assert!(embedding.created_at <= chrono::Utc::now());
}

#[tokio::test]
async fn test_batch_embedding() {
    let provider = TfIdfProvider::default();

    let texts = vec![
        "rust programming".to_string(),
        "python scripting".to_string(),
        "javascript web development".to_string(),
    ];

    let embeddings = provider.embed_batch(&texts, None).await.unwrap();

    assert_eq!(embeddings.len(), texts.len());
    for embedding in embeddings {
        assert_eq!(embedding.vector.len(), 384);
    }
}

#[tokio::test]
async fn test_batch_embedding_normalized() {
    let provider = TfIdfProvider::default();

    let texts = vec![
        "machine learning".to_string(),
        "deep learning".to_string(),
    ];

    let options = EmbedOptions {
        normalize: true,
        ..Default::default()
    };

    let embeddings = provider.embed_batch(&texts, Some(&options)).await.unwrap();

    for embedding in embeddings {
        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}

#[tokio::test]
async fn test_embedding_cache_basic() {
    let mut cache = EmbeddingCache::new(10, 3600);

    let provider = TfIdfProvider::default();
    let text = "cached content";
    let embedding = provider.embed(text, None).await.unwrap();

    cache.put(text.to_string(), embedding.clone());

    let retrieved = cache.get(text);
    assert!(retrieved.is_some());

    let stats = cache.stats();
    assert_eq!(stats.entry_count, 1);
}

#[tokio::test]
async fn test_embedding_cache_eviction() {
    let mut cache = EmbeddingCache::new(2, 3600);

    let provider = TfIdfProvider::default();

    for i in 0..3 {
        let text = format!("content {}", i);
        let embedding = provider.embed(&text, None).await.unwrap();
        cache.put(text, embedding);
    }

    let stats = cache.stats();
    assert!(stats.entry_count <= 2, "Cache should respect max size");
    assert_eq!(stats.evictions, 1, "Should have evicted one entry");
}

#[tokio::test]
async fn test_embedding_cache_ttl_expiry() {
    use tokio::time::{sleep, Duration};

    let mut cache = EmbeddingCache::new(10, 1); // 1 second TTL

    let provider = TfIdfProvider::default();
    let text = "expiring content";
    let embedding = provider.embed(text, None).await.unwrap();

    cache.put(text.to_string(), embedding);
    assert!(cache.get(text).is_some());

    // Wait for expiry
    sleep(Duration::from_secs(2)).await;

    assert!(cache.get(text).is_none(), "Entry should have expired");
}

#[tokio::test]
async fn test_embedding_cache_clear() {
    let mut cache = EmbeddingCache::new(10, 3600);

    let provider = TfIdfProvider::default();
    let embedding = provider.embed("test", None).await.unwrap();

    cache.put("test".to_string(), embedding);
    assert_eq!(cache.stats().entry_count, 1);

    cache.clear();
    assert_eq!(cache.stats().entry_count, 0);
}

#[tokio::test]
async fn test_dense_vector_retriever_basic() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let agent = AgentMemory::new(config).await.unwrap();
    let provider = Arc::new(TfIdfProvider::default());
    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider);

    assert_eq!(retriever.name(), "DenseVectorRetriever");
    assert_eq!(retriever.signal_type(), RetrievalSignal::DenseVector);
    assert!(retriever.is_available());
}

#[tokio::test]
async fn test_dense_vector_retriever_threshold() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let agent = AgentMemory::new(config).await.unwrap();
    let provider = Arc::new(TfIdfProvider::default());

    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider)
        .with_threshold(0.5);

    // Threshold is private, but we can test that it was created successfully
    assert!(retriever.is_available());
}

#[tokio::test]
async fn test_dense_vector_search() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store test memories
    agent.store("ml1", "machine learning neural networks").await.unwrap();
    agent.store("ml2", "deep learning artificial intelligence").await.unwrap();
    agent.store("cook1", "cooking recipes food preparation").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());
    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider)
        .with_threshold(0.1);

    // Search for ML-related content
    let results = retriever.search("machine learning AI", 10, None).await.unwrap();

    // Should find ML-related memories
    assert!(!results.is_empty());

    // Top results should be ML-related
    let top_keys: Vec<&str> = results.iter().take(2).map(|r| r.fragment.key.as_str()).collect();
    assert!(top_keys.contains(&"ml1") || top_keys.contains(&"ml2"));
}

#[tokio::test]
async fn test_dense_vector_search_ranking() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memories with varying relevance
    agent.store("exact", "rust programming language").await.unwrap();
    agent.store("related", "rust systems programming").await.unwrap();
    agent.store("distant", "python scripting language").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());
    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider)
        .with_threshold(0.1);

    let results = retriever.search("rust programming", 10, None).await.unwrap();

    // Results should be ranked by similarity
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[tokio::test]
async fn test_dense_vector_high_threshold() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    agent.store("unrelated", "cooking recipes food").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());
    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider)
        .with_threshold(0.9); // Very high threshold

    // Search with unrelated query
    let results = retriever.search("machine learning AI", 10, None).await.unwrap();

    // High threshold should filter out low-similarity results
    for result in results {
        assert!(result.score >= 0.9);
    }
}

#[tokio::test]
async fn test_hybrid_with_dense_vector() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store diverse memories
    agent.store("rust_lang", "rust programming language").await.unwrap();
    agent.store("rust_sys", "rust systems programming").await.unwrap();
    agent.store("python", "python scripting").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());

    let pipeline_config = PipelineConfig::semantic_focus();
    let retriever = HybridRetriever::new(pipeline_config)
        .add_pipeline(Arc::new(DenseVectorRetriever::new(
            agent.storage().clone(),
            provider.clone()
        ).with_threshold(0.1)))
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("rust", 10).await.unwrap();

    // Should find rust-related memories
    assert!(!results.is_empty());
    let rust_count = results.iter()
        .filter(|r| r.key.contains("rust"))
        .count();
    assert!(rust_count >= 1, "Should find rust-related memories");
}

#[tokio::test]
async fn test_hybrid_dense_vs_keyword() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    // Store memories
    agent.store("ml", "machine learning deep learning neural networks").await.unwrap();
    agent.store("ai", "artificial intelligence ML algorithms").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());

    // Test with semantic focus
    let semantic_config = PipelineConfig::semantic_focus();
    let semantic_retriever = HybridRetriever::new(semantic_config)
        .add_pipeline(Arc::new(DenseVectorRetriever::new(
            agent.storage().clone(),
            provider.clone()
        ).with_threshold(0.1)));

    let semantic_results = semantic_retriever.search("deep neural nets", 10).await.unwrap();

    // Should find related content even without exact keyword matches
    assert!(!semantic_results.is_empty());

    // Test with keyword focus
    let keyword_config = PipelineConfig::keyword_focus();
    let keyword_retriever = HybridRetriever::new(keyword_config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let keyword_results = keyword_retriever.search("ML", 10).await.unwrap();
    assert!(!keyword_results.is_empty());
}

#[tokio::test]
async fn test_fusion_strategies_with_dense_vector() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();

    agent.store("test", "rust programming systems").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());

    // Test ReciprocRankFusion
    let rrf_config = PipelineConfig::default()
        .with_fusion_strategy(FusionStrategy::ReciprocRankFusion);
    let rrf_retriever = HybridRetriever::new(rrf_config)
        .add_pipeline(Arc::new(DenseVectorRetriever::new(
            agent.storage().clone(),
            provider.clone()
        ).with_threshold(0.1)))
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let rrf_results = rrf_retriever.search("rust programming", 5).await.unwrap();
    assert!(!rrf_results.is_empty());

    // Test WeightedAverage
    let wa_config = PipelineConfig::default()
        .with_fusion_strategy(FusionStrategy::WeightedAverage);
    let wa_retriever = HybridRetriever::new(wa_config)
        .add_pipeline(Arc::new(DenseVectorRetriever::new(
            agent.storage().clone(),
            provider.clone()
        ).with_threshold(0.1)))
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let wa_results = wa_retriever.search("rust programming", 5).await.unwrap();
    assert!(!wa_results.is_empty());
}

#[tokio::test]
async fn test_embedding_provider_batch_efficiency() {
    let provider = TfIdfProvider::default();

    let texts: Vec<String> = (0..100)
        .map(|i| format!("test content {}", i))
        .collect();

    let start = std::time::Instant::now();
    let _embeddings = provider.embed_batch(&texts, None).await.unwrap();
    let batch_duration = start.elapsed();

    // Batch should be faster than individual embeds
    // (This is a basic smoke test - real perf test would be in benches/)
    assert!(batch_duration.as_secs() < 10, "Batch embedding should complete reasonably quickly");
}

#[tokio::test]
async fn test_normalize_vector_helper() {
    let mut vector = vec![3.0, 4.0, 0.0];
    normalize_vector(&mut vector);

    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Normalized vector should have L2 norm = 1.0");
}

#[tokio::test]
async fn test_compute_content_hash_consistency() {
    let text = "consistent content";

    let hash1 = compute_content_hash(text);
    let hash2 = compute_content_hash(text);

    assert_eq!(hash1, hash2, "Same content should produce same hash");
}

#[tokio::test]
async fn test_compute_content_hash_uniqueness() {
    let text1 = "content one";
    let text2 = "content two";

    let hash1 = compute_content_hash(text1);
    let hash2 = compute_content_hash(text2);

    assert_ne!(hash1, hash2, "Different content should produce different hashes");
}

#[tokio::test]
async fn test_embedding_with_empty_text() {
    let provider = TfIdfProvider::default();

    let result = provider.embed("", None).await;

    // Should handle empty text gracefully
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_dense_vector_empty_query() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(config).await.unwrap();
    agent.store("test", "content").await.unwrap();

    let provider = Arc::new(TfIdfProvider::default());
    let retriever = DenseVectorRetriever::new(agent.storage().clone(), provider);

    let results = retriever.search("", 10, None).await.unwrap();

    // Should handle empty query gracefully
    assert!(results.len() <= 1);
}

#[tokio::test]
async fn test_provider_capabilities() {
    let provider = TfIdfProvider::default();
    let caps = provider.capabilities();

    assert!(caps.supports_batch);
    assert_eq!(caps.max_batch_size, 100);
    assert_eq!(caps.max_text_length, 8192);
}
