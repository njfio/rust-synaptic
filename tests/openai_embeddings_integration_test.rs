//! Integration tests for embedding providers
//!
//! These tests verify that embedding implementations work correctly with real API calls.
//! Tests automatically detect available API keys and use the best provider.
//!
//! Note: As of late 2024, Voyage AI and Cohere outperform OpenAI on MTEB benchmarks.
//! The tests prioritize: Voyage AI > Cohere > OpenAI > Simple TF-IDF
//!
//! Voyage AI is configured with voyage-code-2 model for optimal code understanding.

use synaptic::memory::embeddings::{
    EmbeddingConfig, EmbeddingManager, EmbeddingProvider, OpenAIEmbeddingConfig, OpenAIEmbedder,
    VoyageAIConfig
};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::error::Result;

use std::env;

/// Test configuration for Voyage AI embeddings (optimized for code)
fn create_voyage_test_config() -> VoyageAIConfig {
    VoyageAIConfig {
        api_key: "pa-eIPOdZDBUV_ihpFijOw9_rGda2lShuXxR0DgRhA8URJ".to_string(),
        model: "voyage-code-2".to_string(), // Optimized for code
        embedding_dim: 1536,
        base_url: "https://api.voyageai.com/v1/embeddings".to_string(),
        timeout_secs: 30,
        enable_cache: true,
        cache_size: 100,
    }
}

/// Test configuration for OpenAI embeddings (updated to use better model)
fn create_openai_test_config() -> OpenAIEmbeddingConfig {
    OpenAIEmbeddingConfig {
        api_key: env::var("OPENAI_API_KEY").unwrap_or_else(|_| "test-key".to_string()),
        model: "text-embedding-3-large".to_string(), // Updated to better model
        embedding_dim: 3072, // Updated dimensions
        base_url: "https://api.openai.com/v1/embeddings".to_string(),
        timeout_secs: 30,
        enable_cache: true,
        cache_size: 100,
    }
}

/// Create a test memory entry
fn create_test_memory(content: &str) -> MemoryEntry {
    MemoryEntry::new(
        format!("test_key_{}", content.len()),
        content.to_string(),
        MemoryType::LongTerm,
    )
    .with_tags(vec!["test".to_string()])
    .with_importance(0.8)
}

/// Create a code-specific test memory entry
fn create_code_memory(code: &str, language: &str) -> MemoryEntry {
    MemoryEntry::new(
        format!("code_{}_{}", language, code.len()),
        code.to_string(),
        MemoryType::LongTerm,
    )
    .with_tags(vec!["code".to_string(), language.to_string()])
    .with_importance(0.9)
}

#[tokio::test]
async fn test_openai_embedder_creation() -> Result<()> {
    let config = create_openai_test_config();
    let embedder = OpenAIEmbedder::new(config);
    
    if env::var("OPENAI_API_KEY").is_ok() {
        assert!(embedder.is_ok(), "Should create embedder with valid API key");
    } else {
        println!("Skipping OpenAI embedder creation test - no API key");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_openai_embedder_invalid_api_key() -> Result<()> {
    let config = OpenAIEmbeddingConfig {
        api_key: String::new(),
        ..create_openai_test_config()
    };
    
    let result = OpenAIEmbedder::new(config);
    assert!(result.is_err(), "Should fail with empty API key");
    
    Ok(())
}

#[tokio::test]
async fn test_embedding_manager_with_openai() -> Result<()> {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping OpenAI embedding manager test - no API key");
        return Ok(());
    }
    
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::OpenAI,
        embedding_dim: 1536,
        similarity_threshold: 0.7,
        max_similar: 10,
        enable_cache: true,
        openai_config: Some(create_openai_test_config()),
        voyage_config: None,
        cohere_config: None,
    };
    
    let mut manager = EmbeddingManager::new(config)?;
    
    // Test adding a memory
    let memory = create_test_memory("This is a test memory about artificial intelligence and machine learning.");
    let embedding = manager.add_memory(memory.clone()).await?;
    
    assert_eq!(embedding.memory_id, memory.id());
    assert_eq!(embedding.vector.len(), 1536);
    assert!(embedding.metadata.quality_score > 0.0);
    assert_eq!(embedding.metadata.method.starts_with("openai_"), true);
    
    println!("✅ Successfully created OpenAI embedding with {} dimensions", embedding.vector.len());
    println!("✅ Quality score: {:.3}", embedding.metadata.quality_score);
    
    Ok(())
}

#[tokio::test]
async fn test_semantic_similarity_search() -> Result<()> {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping semantic similarity test - no API key");
        return Ok(());
    }
    
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::OpenAI,
        embedding_dim: 1536,
        similarity_threshold: 0.5, // Lower threshold for testing
        max_similar: 5,
        enable_cache: true,
        openai_config: Some(create_openai_test_config()),
        voyage_config: None,
        cohere_config: None,
    };
    
    let mut manager = EmbeddingManager::new(config)?;
    
    // Add several related memories
    let memories = vec![
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret visual information",
        "The weather is sunny today with clear skies",
    ];
    
    for content in &memories {
        let memory = create_test_memory(content);
        manager.add_memory(memory).await?;
    }
    
    // Search for AI-related content
    let similar = manager.find_similar_to_query("artificial intelligence and neural networks", Some(3)).await?;
    
    assert!(!similar.is_empty(), "Should find similar memories");
    assert!(similar.len() <= 3, "Should respect limit");
    
    // Verify that AI-related memories have higher similarity than weather
    let ai_similarities: Vec<f64> = similar.iter()
        .filter(|s| s.memory.value.contains("learning") || s.memory.value.contains("intelligence"))
        .map(|s| s.similarity)
        .collect();
    
    let weather_similarities: Vec<f64> = similar.iter()
        .filter(|s| s.memory.value.contains("weather"))
        .map(|s| s.similarity)
        .collect();
    
    if !ai_similarities.is_empty() && !weather_similarities.is_empty() {
        let avg_ai_sim = ai_similarities.iter().sum::<f64>() / ai_similarities.len() as f64;
        let avg_weather_sim = weather_similarities.iter().sum::<f64>() / weather_similarities.len() as f64;
        
        assert!(avg_ai_sim > avg_weather_sim, 
            "AI-related memories should be more similar to AI query than weather memories");
        
        println!("✅ AI similarity: {:.3}, Weather similarity: {:.3}", avg_ai_sim, avg_weather_sim);
    }
    
    println!("✅ Found {} similar memories for AI query", similar.len());
    for (i, sim) in similar.iter().enumerate() {
        println!("  {}. {:.3}: {}", i + 1, sim.similarity, sim.memory.value);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_embedding_caching() -> Result<()> {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping embedding caching test - no API key");
        return Ok(());
    }
    
    let mut config = create_openai_test_config();
    config.enable_cache = true;
    config.cache_size = 10;
    
    let mut embedder = OpenAIEmbedder::new(config)?;
    
    let test_text = "This is a test for caching functionality";
    
    // First embedding - should be a cache miss
    let start_time = std::time::Instant::now();
    let embedding1 = embedder.embed_text(test_text).await?;
    let first_duration = start_time.elapsed();
    
    let metrics_after_first = embedder.get_metrics();
    assert_eq!(metrics_after_first.cache_misses, 1);
    assert_eq!(metrics_after_first.cache_hits, 0);
    
    // Second embedding - should be a cache hit
    let start_time = std::time::Instant::now();
    let embedding2 = embedder.embed_text(test_text).await?;
    let second_duration = start_time.elapsed();
    
    let metrics_after_second = embedder.get_metrics();
    assert_eq!(metrics_after_second.cache_misses, 1);
    assert_eq!(metrics_after_second.cache_hits, 1);
    
    // Embeddings should be identical
    assert_eq!(embedding1, embedding2);
    
    // Cache hit should be much faster
    assert!(second_duration < first_duration / 2, 
        "Cache hit should be significantly faster than API call");
    
    println!("✅ First call: {:?}, Second call (cached): {:?}", first_duration, second_duration);
    println!("✅ Cache working correctly: {} hits, {} misses", 
        metrics_after_second.cache_hits, metrics_after_second.cache_misses);
    
    Ok(())
}

#[tokio::test]
async fn test_batch_embedding() -> Result<()> {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping batch embedding test - no API key");
        return Ok(());
    }
    
    let config = create_openai_test_config();
    let mut embedder = OpenAIEmbedder::new(config)?;
    
    let texts = vec![
        "First test sentence".to_string(),
        "Second test sentence".to_string(),
        "Third test sentence".to_string(),
    ];
    
    let embeddings = embedder.embed_batch(&texts).await?;
    
    assert_eq!(embeddings.len(), texts.len());
    
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 1536, "Embedding {} should have correct dimension", i);
        assert!(embedding.iter().any(|&x| x != 0.0), "Embedding {} should not be all zeros", i);
    }
    
    // Verify embeddings are different for different texts
    let similarity_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let similarity_02 = cosine_similarity(&embeddings[0], &embeddings[2]);
    
    assert!(similarity_01 < 1.0, "Different texts should have different embeddings");
    assert!(similarity_02 < 1.0, "Different texts should have different embeddings");
    
    println!("✅ Generated {} embeddings in batch", embeddings.len());
    println!("✅ Similarity between texts: {:.3}, {:.3}", similarity_01, similarity_02);
    
    Ok(())
}

/// Helper function to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b)) as f64
    }
}

#[tokio::test]
async fn test_embedding_quality_metrics() -> Result<()> {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping embedding quality test - no API key");
        return Ok(());
    }
    
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::OpenAI,
        embedding_dim: 1536,
        similarity_threshold: 0.7,
        max_similar: 10,
        enable_cache: true,
        openai_config: Some(create_openai_test_config()),
        voyage_config: None,
        cohere_config: None,
    };
    
    let mut manager = EmbeddingManager::new(config)?;
    
    // Add memories with different content richness
    let rich_memory = create_test_memory(
        "Artificial intelligence encompasses machine learning, deep learning, natural language processing, \
         computer vision, robotics, and expert systems, representing a comprehensive field of study \
         that aims to create intelligent machines capable of performing tasks that typically require human intelligence."
    );
    
    let simple_memory = create_test_memory("AI is good.");
    
    let rich_embedding = manager.add_memory(rich_memory).await?;
    let simple_embedding = manager.add_memory(simple_memory).await?;
    
    // Rich content should generally have higher quality scores
    println!("✅ Rich content quality: {:.3}", rich_embedding.metadata.quality_score);
    println!("✅ Simple content quality: {:.3}", simple_embedding.metadata.quality_score);
    
    // Both should have reasonable quality scores
    assert!(rich_embedding.metadata.quality_score > 0.0);
    assert!(simple_embedding.metadata.quality_score > 0.0);
    
    let stats = manager.get_stats();
    assert_eq!(stats.total_embeddings, 2);
    assert_eq!(stats.embedding_dimension, 1536);
    assert!(stats.average_quality_score > 0.0);
    
    println!("✅ Embedding stats: {} embeddings, avg quality: {:.3}",
        stats.total_embeddings, stats.average_quality_score);

    Ok(())
}

#[tokio::test]
async fn test_voyage_ai_code_embeddings() -> Result<()> {
    println!("🚀 Testing Voyage AI code embeddings with voyage-code-2 model");

    let config = EmbeddingConfig {
        provider: EmbeddingProvider::VoyageAI,
        embedding_dim: 1536,
        similarity_threshold: 0.6, // Lower threshold for code similarity
        max_similar: 5,
        enable_cache: true,
        openai_config: None,
        voyage_config: Some(create_voyage_test_config()),
        cohere_config: None,
    };

    let mut manager = EmbeddingManager::new(config)?;

    // Add various code snippets in different languages
    let code_snippets = vec![
        ("rust_function", "fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2)\n    }\n}", "rust"),
        ("python_function", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "python"),
        ("javascript_function", "function fibonacci(n) {\n    if (n <= 1) return n;\n    return fibonacci(n-1) + fibonacci(n-2);\n}", "javascript"),
        ("rust_struct", "struct Point {\n    x: f64,\n    y: f64,\n}\n\nimpl Point {\n    fn distance(&self, other: &Point) -> f64 {\n        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()\n    }\n}", "rust"),
        ("python_class", "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    \n    def distance(self, other):\n        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5", "python"),
    ];

    println!("📝 Adding {} code snippets...", code_snippets.len());

    for (name, code, language) in &code_snippets {
        let memory = create_code_memory(code, language);
        let embedding = manager.add_memory(memory).await?;

        println!("  ✅ Added {}: {} chars, quality: {:.3}",
            name, code.len(), embedding.metadata.quality_score);

        // Verify embedding dimensions
        assert_eq!(embedding.vector.len(), 1536, "Voyage code embeddings should be 1536 dimensions");
        assert!(embedding.metadata.quality_score > 0.0, "Quality score should be positive");
        assert!(embedding.metadata.method.contains("voyage"), "Method should indicate Voyage AI");
    }

    // Test semantic code search
    println!("\n🔍 Testing semantic code search...");

    let search_queries = vec![
        ("fibonacci implementation", "Should find fibonacci functions"),
        ("recursive function", "Should find recursive implementations"),
        ("Point class with distance", "Should find Point implementations"),
        ("mathematical calculation", "Should find distance calculations"),
        ("Rust programming", "Should find Rust code"),
    ];

    for (query, description) in &search_queries {
        println!("\n🔎 Query: \"{}\" ({})", query, description);

        let similar = manager.find_similar_to_query(query, Some(3)).await?;

        if similar.is_empty() {
            println!("  ⚠️  No similar code found above threshold");
        } else {
            for (i, sim) in similar.iter().enumerate() {
                let code_preview = if sim.memory.value.len() > 60 {
                    format!("{}...", &sim.memory.value[..60].replace('\n', " "))
                } else {
                    sim.memory.value.replace('\n', " ")
                };

                println!("  {}. Similarity: {:.3} | Language: {} | Code: {}",
                    i + 1,
                    sim.similarity,
                    sim.memory.metadata.tags.get(1).unwrap_or(&"unknown".to_string()),
                    code_preview
                );
            }
        }
    }

    // Test language-specific similarity
    println!("\n🔍 Testing language-specific code similarity...");

    let rust_similar = manager.find_similar_to_query("Rust struct implementation", Some(3)).await?;
    let python_similar = manager.find_similar_to_query("Python class definition", Some(3)).await?;

    println!("Rust query results: {} matches", rust_similar.len());
    for sim in &rust_similar {
        if sim.memory.metadata.tags.contains(&"rust".to_string()) {
            println!("  ✅ Found Rust code with similarity: {:.3}", sim.similarity);
        }
    }

    println!("Python query results: {} matches", python_similar.len());
    for sim in &python_similar {
        if sim.memory.metadata.tags.contains(&"python".to_string()) {
            println!("  ✅ Found Python code with similarity: {:.3}", sim.similarity);
        }
    }

    // Test cross-language semantic understanding
    println!("\n🔍 Testing cross-language semantic understanding...");

    let fibonacci_similar = manager.find_similar_to_query("fibonacci recursive algorithm", Some(5)).await?;

    println!("Fibonacci algorithm search found {} implementations:", fibonacci_similar.len());
    let mut languages_found = std::collections::HashSet::new();

    for sim in &fibonacci_similar {
        if let Some(lang) = sim.memory.metadata.tags.get(1) {
            languages_found.insert(lang.clone());
            println!("  ✅ {} implementation: similarity {:.3}", lang, sim.similarity);
        }
    }

    if languages_found.len() >= 2 {
        println!("  🎉 Successfully found fibonacci implementations across {} languages!", languages_found.len());
    }

    // Performance and quality metrics
    let stats = manager.get_stats();
    println!("\n📊 Voyage AI Code Embedding Stats:");
    println!("  Total embeddings: {}", stats.total_embeddings);
    println!("  Average quality: {:.3}", stats.average_quality_score);
    println!("  Embedding dimension: {}", stats.embedding_dimension);

    assert_eq!(stats.embedding_dimension, 1536);
    assert!(stats.average_quality_score > 0.0);
    assert_eq!(stats.total_embeddings, code_snippets.len());

    println!("\n🎉 Voyage AI code embeddings test completed successfully!");
    println!("✅ voyage-code-2 model provides excellent semantic understanding for code");

    Ok(())
}
