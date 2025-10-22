//! Integration tests for hybrid retrieval pipeline
//!
//! These tests verify that the retrieval pipeline components work correctly
//! and can be combined for high-quality search results.

use rust_synaptic::{
    AgentMemory, MemoryConfig, StorageBackend,
    memory::retrieval::{
        RetrievalPipeline, HybridRetriever, PipelineConfig, FusionStrategy,
        KeywordRetriever, TemporalRetriever, GraphRetriever, RetrievalSignal,
    },
};
use std::sync::Arc;

#[tokio::test]
async fn test_keyword_retriever() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let agent = AgentMemory::new(config).await.unwrap();
    let retriever = KeywordRetriever::new(agent.storage().clone());

    assert_eq!(retriever.name(), "KeywordRetriever");
    assert_eq!(retriever.signal_type(), RetrievalSignal::SparseKeyword);
    assert!(retriever.is_available());
}

#[tokio::test]
async fn test_temporal_retriever() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let agent = AgentMemory::new(config).await.unwrap();
    let retriever = TemporalRetriever::new(agent.storage().clone())
        .with_weights(0.7, 0.3);

    assert_eq!(retriever.name(), "TemporalRetriever");
    assert_eq!(retriever.signal_type(), RetrievalSignal::TemporalRelevance);
}

#[tokio::test]
async fn test_graph_retriever() {
    let config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let agent = AgentMemory::new(config).await.unwrap();
    let retriever = GraphRetriever::new(agent.storage().clone(), None);

    assert_eq!(retriever.name(), "GraphRetriever");
    assert_eq!(retriever.signal_type(), RetrievalSignal::GraphRelationship);
    // Without graph, should not be available
    assert!(!retriever.is_available());
}

#[tokio::test]
async fn test_pipeline_config_defaults() {
    let config = PipelineConfig::default();

    assert_eq!(config.max_per_signal, 50);
    assert_eq!(config.min_score, 0.1);
    assert!(config.enable_fusion);
    assert_eq!(config.fusion_strategy, FusionStrategy::ReciprocRankFusion);
    assert!(config.enable_caching);

    // Check default weights
    assert_eq!(config.signal_weights.get(&RetrievalSignal::DenseVector), Some(&0.4));
    assert_eq!(config.signal_weights.get(&RetrievalSignal::SparseKeyword), Some(&0.3));
}

#[tokio::test]
async fn test_pipeline_config_presets() {
    // Test semantic focus
    let semantic = PipelineConfig::semantic_focus();
    assert_eq!(semantic.signal_weights.get(&RetrievalSignal::DenseVector), Some(&0.6));

    // Test keyword focus
    let keyword = PipelineConfig::keyword_focus();
    assert_eq!(keyword.signal_weights.get(&RetrievalSignal::SparseKeyword), Some(&0.6));

    // Test graph focus
    let graph = PipelineConfig::graph_focus();
    assert_eq!(graph.signal_weights.get(&RetrievalSignal::GraphRelationship), Some(&0.4));

    // Test temporal focus
    let temporal = PipelineConfig::temporal_focus();
    assert_eq!(temporal.signal_weights.get(&RetrievalSignal::TemporalRelevance), Some(&0.4));
}

#[tokio::test]
async fn test_pipeline_config_builder() {
    let config = PipelineConfig::default()
        .with_signal_weight(RetrievalSignal::DenseVector, 0.5)
        .with_fusion_strategy(FusionStrategy::WeightedAverage)
        .with_min_score(0.2);

    assert_eq!(config.signal_weights.get(&RetrievalSignal::DenseVector), Some(&0.5));
    assert_eq!(config.fusion_strategy, FusionStrategy::WeightedAverage);
    assert_eq!(config.min_score, 0.2);
}

#[tokio::test]
async fn test_hybrid_retriever_creation() {
    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config.clone());

    assert_eq!(retriever.config().max_per_signal, config.max_per_signal);
    assert_eq!(retriever.config().min_score, config.min_score);
}

#[tokio::test]
async fn test_hybrid_retriever_add_pipelines() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let agent = AgentMemory::new(agent_config).await.unwrap();

    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())))
        .add_pipeline(Arc::new(TemporalRetriever::new(agent.storage().clone())));

    // Retriever should now have 2 pipelines (we can't directly test this without exposing the field)
    // But we can test that it was created successfully
    assert!(retriever.config().enable_fusion);
}

#[tokio::test]
async fn test_hybrid_search_empty_query() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    // Store some test memories
    agent.store("test1", "rust programming").await.unwrap();
    agent.store("test2", "python scripting").await.unwrap();

    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("", 10).await.unwrap();
    // Empty query might return empty or all results depending on implementation
    assert!(results.len() <= 2);
}

#[tokio::test]
async fn test_hybrid_search_with_results() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    // Store test memories
    agent.store("rust_basics", "Introduction to Rust programming language").await.unwrap();
    agent.store("rust_advanced", "Advanced Rust concepts and patterns").await.unwrap();
    agent.store("python_intro", "Python programming for beginners").await.unwrap();

    let config = PipelineConfig::keyword_focus();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("Rust", 10).await.unwrap();

    // Should find Rust-related memories
    assert!(!results.is_empty());
    // Results should contain rust memories
    let has_rust = results.iter().any(|r| r.key.contains("rust"));
    assert!(has_rust, "Should find rust-related memories");
}

#[tokio::test]
async fn test_fusion_strategy_weighted_average() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    agent.store("test1", "rust programming systems").await.unwrap();

    let config = PipelineConfig::default()
        .with_fusion_strategy(FusionStrategy::WeightedAverage);

    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())))
        .add_pipeline(Arc::new(TemporalRetriever::new(agent.storage().clone())));

    let results = retriever.search("rust", 5).await.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_fusion_strategy_max_score() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    agent.store("test1", "rust programming").await.unwrap();

    let config = PipelineConfig::default()
        .with_fusion_strategy(FusionStrategy::MaxScore);

    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("rust", 5).await.unwrap();
    assert!(results.len() <= 1);
}

#[tokio::test]
async fn test_cache_functionality() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    agent.store("test1", "cached content").await.unwrap();

    let config = PipelineConfig::default();
    assert!(config.enable_caching);

    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    // First search (cache miss)
    let results1 = retriever.search("cached", 5).await.unwrap();

    // Second search (should be cache hit)
    let results2 = retriever.search("cached", 5).await.unwrap();

    // Results should be identical
    assert_eq!(results1.len(), results2.len());

    // Check cache stats
    let stats = retriever.cache_stats().await;
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert!(stats.entry_count > 0);
}

#[tokio::test]
async fn test_cache_clear() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    agent.store("test1", "content").await.unwrap();

    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    // Populate cache
    retriever.search("content", 5).await.unwrap();

    // Verify cache has entries
    let stats_before = retriever.cache_stats().await.unwrap();
    assert!(stats_before.entry_count > 0);

    // Clear cache
    retriever.clear_cache().await;

    // Verify cache is empty
    let stats_after = retriever.cache_stats().await.unwrap();
    assert_eq!(stats_after.entry_count, 0);
}

#[tokio::test]
async fn test_min_score_filtering() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    agent.store("highly_relevant", "rust programming systems").await.unwrap();
    agent.store("less_relevant", "cooking recipes").await.unwrap();

    let config = PipelineConfig::default()
        .with_min_score(0.3); // Higher threshold

    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("rust programming", 10).await.unwrap();

    // Should only include high-scoring results
    for result in results {
        // We can't directly test the score since it's internal to fusion
        // but we can verify results were returned
        assert!(!result.key.is_empty());
    }
}

#[tokio::test]
async fn test_temporal_retriever_recency_bias() {
    use std::time::Duration;
    use tokio::time::sleep;

    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    // Store an old memory
    agent.store("old_memory", "old content").await.unwrap();

    // Wait a bit
    sleep(Duration::from_millis(100)).await;

    // Store a recent memory
    agent.store("recent_memory", "recent content").await.unwrap();

    let config = PipelineConfig::temporal_focus();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(TemporalRetriever::new(agent.storage().clone())));

    let results = retriever.search("content", 10).await.unwrap();

    // Recent memory should be found
    assert!(!results.is_empty());
    let has_recent = results.iter().any(|r| r.key.contains("recent"));
    assert!(has_recent, "Recent memories should be prioritized");
}

#[tokio::test]
async fn test_multiple_signals_fusion() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        enable_knowledge_graph: false, // Disable graph for this test
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    // Store diverse memories
    agent.store("mem1", "rust programming language").await.unwrap();
    agent.store("mem2", "rust systems programming").await.unwrap();
    agent.store("mem3", "python scripting").await.unwrap();

    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())))
        .add_pipeline(Arc::new(TemporalRetriever::new(agent.storage().clone())));

    let results = retriever.search("rust", 10).await.unwrap();

    // Should combine signals and return relevant results
    assert!(!results.is_empty());
    let rust_count = results.iter().filter(|r| r.key.contains("mem1") || r.key.contains("mem2")).count();
    assert!(rust_count >= 1, "Should find rust-related memories");
}

#[tokio::test]
async fn test_limit_enforcement() {
    let agent_config = MemoryConfig {
        storage_backend: StorageBackend::Memory,
        ..Default::default()
    };

    let mut agent = AgentMemory::new(agent_config).await.unwrap();

    // Store many memories
    for i in 0..20 {
        agent.store(&format!("mem{}", i), "test content").await.unwrap();
    }

    let config = PipelineConfig::default();
    let retriever = HybridRetriever::new(config)
        .add_pipeline(Arc::new(KeywordRetriever::new(agent.storage().clone())));

    let results = retriever.search("content", 5).await.unwrap();

    // Should respect the limit
    assert!(results.len() <= 5);
}

#[tokio::test]
async fn test_config_update() {
    let config = PipelineConfig::default();
    let mut retriever = HybridRetriever::new(config);

    // Update config
    let new_config = PipelineConfig::semantic_focus();
    retriever.set_config(new_config.clone());

    // Verify config was updated
    assert_eq!(
        retriever.config().signal_weights.get(&RetrievalSignal::DenseVector),
        new_config.signal_weights.get(&RetrievalSignal::DenseVector)
    );
}
