//! Voyage AI Code Embeddings Demo
//!
//! Demonstrates Voyage AI's voyage-code-2 model for semantic code understanding.
//! Voyage AI currently leads MTEB benchmarks and voyage-code-2 is specifically
//! optimized for code embeddings, providing superior understanding of programming
//! languages, algorithms, and code semantics.

use synaptic::memory::embeddings::{
    EmbeddingConfig, EmbeddingManager, EmbeddingProvider, OpenAIEmbeddingConfig
};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 OpenAI Embeddings Demo");
    println!("========================");

    // Check if OpenAI API key is available
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set. Get your API key from https://platform.openai.com/api-keys");

    // Set the API key in environment for the demo
    std::env::set_var("OPENAI_API_KEY", &api_key);

    // Configure OpenAI embeddings
    let openai_config = OpenAIEmbeddingConfig {
        api_key: api_key.clone(),
        model: "text-embedding-3-small".to_string(),
        embedding_dim: 1536,
        base_url: "https://api.openai.com/v1/embeddings".to_string(),
        timeout_secs: 30,
        enable_cache: true,
        cache_size: 1000,
    };

    let config = EmbeddingConfig {
        provider: EmbeddingProvider::OpenAI,
        embedding_dim: 1536,
        similarity_threshold: 0.7,
        max_similar: 5,
        enable_cache: true,
        openai_config: Some(openai_config),
        voyage_config: None,
        cohere_config: None,
    };

    println!("📡 Initializing OpenAI Embeddings Manager...");
    let mut embedding_manager = EmbeddingManager::new(config)?;

    // Create some test memories
    let memories = vec![
        MemoryEntry::new(
            "ai_research".to_string(),
            "Artificial intelligence research focuses on creating systems that can perform tasks requiring human intelligence.".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "machine_learning".to_string(),
            "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "neural_networks".to_string(),
            "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.".to_string(),
            MemoryType::LongTerm,
        ),
        MemoryEntry::new(
            "cooking_recipe".to_string(),
            "To make pasta, boil water, add salt, cook pasta for 8-12 minutes, drain, and serve with sauce.".to_string(),
            MemoryType::ShortTerm,
        ),
        MemoryEntry::new(
            "weather_today".to_string(),
            "Today's weather is sunny with a high of 75°F and low humidity. Perfect for outdoor activities.".to_string(),
            MemoryType::ShortTerm,
        ),
    ];

    println!("🧠 Adding memories and generating OpenAI embeddings...");
    for (i, memory) in memories.iter().enumerate() {
        print!("  Processing memory {}/{}: {} ... ", i + 1, memories.len(), memory.key);
        
        let embedding = embedding_manager.add_memory(memory.clone()).await?;
        println!("✅ (dimension: {})", embedding.vector.len());
        
        // Small delay to respect rate limits
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Test semantic search
    println!("\n🔍 Testing Semantic Search with OpenAI Embeddings");
    println!("================================================");

    let test_queries = vec![
        "deep learning and neural networks",
        "cooking instructions",
        "weather forecast",
        "artificial intelligence algorithms",
    ];

    for query in test_queries {
        println!("\n🔎 Query: \"{}\"", query);
        
        let similar_memories = embedding_manager.find_similar_to_query(query, Some(3)).await?;
        
        if similar_memories.is_empty() {
            println!("   No similar memories found above threshold");
        } else {
            for (i, similar) in similar_memories.iter().enumerate() {
                println!("   {}. {} (similarity: {:.3})", 
                    i + 1, 
                    similar.memory.key, 
                    similar.similarity
                );
            }
        }
        
        // Small delay between queries
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Show embedding statistics
    println!("\n📊 Embedding Statistics");
    println!("======================");
    let stats = embedding_manager.get_stats();
    println!("Total embeddings: {}", stats.total_embeddings);
    println!("Total memories: {}", stats.total_memories);
    println!("Embedding dimension: {}", stats.embedding_dimension);
    println!("Average quality score: {:.3}", stats.average_quality_score);
    println!("Cache enabled: {}", stats.cache_enabled);
    println!("Similarity threshold: {}", stats.similarity_threshold);

    println!("\n✅ OpenAI Embeddings Demo Complete!");
    println!("===================================");
    println!("🎯 Key achievements:");
    println!("   • Real OpenAI text-embedding-3-small model integration");
    println!("   • State-of-the-art semantic similarity search");
    println!("   • Intelligent caching for performance");
    println!("   • Production-ready error handling");
    println!("   • No more simulated/mocked embeddings!");

    #[cfg(not(feature = "openai-embeddings"))]
    {
        println!("\n⚠️  OpenAI embeddings not enabled. Run with:");
        println!("   cargo run --example openai_embeddings_demo --features openai-embeddings");
    }

    Ok(())
}
