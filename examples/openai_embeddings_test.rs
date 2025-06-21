//! OpenAI Embeddings Test
//! 
//! Tests the real OpenAI embeddings integration with the provided API key.

use synaptic::memory::embeddings::openai_embeddings::{OpenAIEmbedder, OpenAIEmbeddingConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing OpenAI Embeddings Integration");
    
    // Create OpenAI embedder with default config (includes provided API key)
    let config = OpenAIEmbeddingConfig::default();
    println!("📝 Using model: {}", config.model);
    println!("📏 Embedding dimensions: {}", config.embedding_dim);
    
    let mut embedder = OpenAIEmbedder::new(config)?;
    
    // Test embedding generation
    println!("📝 Testing embedding generation...");
    let test_text = "This is a test of OpenAI embeddings for the Synaptic memory system.";
    
    match embedder.embed_text(test_text).await {
        Ok(embedding) => {
            println!("✅ Successfully generated embedding!");
            println!("  Dimensions: {}", embedding.len());
            println!("  First 5 values: {:?}", &embedding[0..5.min(embedding.len())]);
            
            // Get metrics
            let metrics = embedder.get_metrics().clone();
            println!("📊 Metrics:");
            println!("  API calls: {}", metrics.api_calls);
            println!("  Total tokens: {}", metrics.total_tokens);
            println!("  Cache hits: {}", metrics.cache_hits);
            println!("  Cache misses: {}", metrics.cache_misses);
            println!("  Errors: {}", metrics.errors);

            // Test caching by calling again
            println!("📝 Testing caching with same text...");
            let _cached_embedding = embedder.embed_text(test_text).await?;
            let updated_metrics = embedder.get_metrics();
            println!("📊 Updated Metrics:");
            println!("  Cache hits: {}", updated_metrics.cache_hits);
            println!("  Cache misses: {}", updated_metrics.cache_misses);

            if updated_metrics.cache_hits > metrics.cache_hits {
                println!("✅ Caching is working correctly!");
            }
        }
        Err(e) => {
            println!("❌ Failed to generate embedding: {}", e);
            return Err(e.into());
        }
    }
    
    println!("🎉 OpenAI embeddings test completed successfully!");
    Ok(())
}
