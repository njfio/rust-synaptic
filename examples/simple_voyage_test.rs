//! Simple Voyage AI Test
//!
//! Basic test to verify Voyage AI integration works with your API key

#[cfg(feature = "reqwest")]
use synaptic::memory::embeddings::{VoyageAIConfig, VoyageAIEmbedder};
#[cfg(feature = "reqwest")]
use synaptic::error::Result;

#[cfg(feature = "reqwest")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Testing Voyage AI Integration");
    
    let config = VoyageAIConfig {
        api_key: "pa-eIPOdZDBUV_ihpFijOw9_rGda2lShuXxR0DgRhA8URJ".to_string(),
        model: "voyage-code-2".to_string(),
        embedding_dim: 1536,
        base_url: "https://api.voyageai.com/v1/embeddings".to_string(),
        timeout_secs: 30,
        enable_cache: true,
        cache_size: 100,
    };
    
    let mut embedder = VoyageAIEmbedder::new(config)?;
    
    println!("📝 Testing code embedding...");
    
    let code = "fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2)\n    }\n}";
    
    let embedding = embedder.embed_text(code).await?;
    
    println!("✅ Successfully generated embedding!");
    println!("  Dimensions: {}", embedding.len());
    println!("  First 5 values: {:?}", &embedding[..5]);
    
    let quality = embedder.calculate_quality_score(&embedding);
    println!("  Quality score: {:.3}", quality);
    
    let metrics = embedder.get_metrics();
    println!("📊 Metrics:");
    println!("  Total requests: {}", metrics.total_requests);
    println!("  Cache hits: {}", metrics.cache_hits);
    println!("  Cache misses: {}", metrics.cache_misses);
    
    let model_info = embedder.get_model_info();
    println!("🔧 Model Info:");
    for (key, value) in &model_info {
        println!("  {}: {}", key, value);
    }
    
    println!("\n🎉 Voyage AI integration test completed successfully!");
    
    Ok(())
}

#[cfg(not(feature = "reqwest"))]
fn main() {
    println!("This example requires the 'reqwest' feature to be enabled.");
    println!("Run with: cargo run --example simple_voyage_test --features reqwest");
}
