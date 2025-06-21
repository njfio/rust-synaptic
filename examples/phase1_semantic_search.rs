//! Phase 1 Implementation: Advanced AI Integration
//! 
//! This example demonstrates the semantic search capabilities using
//! vector embeddings for intelligent memory retrieval.

use synaptic::{
    AgentMemory, MemoryConfig,
};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!(" Synaptic Phase 1: Advanced AI Integration Demo");
    println!("==================================================");
    
    // Create memory system with embeddings enabled
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        enable_advanced_management: true,
        #[cfg(feature = "embeddings")]
        enable_embeddings: true,
        ..Default::default()
    };
    
    let mut memory = AgentMemory::new(config).await?;
    
    println!("\n Example 1: Building Knowledge Base");
    println!("------------------------------------");
    
    // Add diverse memories to build a knowledge base
    let memories = vec![
        ("ai_research", "Artificial intelligence research focuses on machine learning algorithms and neural networks"),
        ("cooking_pasta", "To cook perfect pasta, boil water with salt and cook al dente for 8-10 minutes"),
        ("rust_programming", "Rust is a systems programming language focused on safety and performance"),
        ("machine_learning", "Machine learning uses statistical models to enable computers to learn from data"),
        ("italian_cuisine", "Italian cuisine features pasta, pizza, risotto, and fresh ingredients like basil and tomatoes"),
        ("deep_learning", "Deep learning uses neural networks with multiple layers to process complex data"),
        ("web_development", "Web development involves creating websites using HTML, CSS, JavaScript, and frameworks"),
        ("data_science", "Data science combines statistics, programming, and domain expertise to extract insights"),
        ("recipe_collection", "My favorite recipes include carbonara, margherita pizza, and tiramisu dessert"),
        ("python_programming", "Python is a versatile programming language popular for data science and web development"),
    ];
    
    for (key, content) in &memories {
        memory.store(key, content).await?;
        println!("✓ Stored memory: {}", key);
    }
    
    // Get basic statistics
    let stats = memory.stats();
    println!("\n Memory System Statistics:");
    println!("  • Total memories: {}", stats.short_term_count + stats.long_term_count);
    println!("  • Short-term: {}", stats.short_term_count);
    println!("  • Total size: {} bytes", stats.total_size);
    
    // Get embedding statistics if available
    #[cfg(feature = "embeddings")]
    if let Some(embedding_stats) = memory.embedding_stats() {
        println!("\n Embedding Statistics:");
        println!("  • Total embeddings: {}", embedding_stats.total_embeddings);
        println!("  • Embedding dimension: {}", embedding_stats.embedding_dimension);
        println!("  • Average quality score: {:.3}", embedding_stats.average_quality_score);
        println!("  • Similarity threshold: {:.2}", embedding_stats.similarity_threshold);
    }
    
    println!("\n Example 2: Semantic Search Queries");
    println!("------------------------------------");
    
    // Perform semantic searches
    let search_queries = vec![
        "artificial intelligence and neural networks",
        "cooking and food recipes",
        "programming languages and software development",
        "statistical analysis and data processing",
    ];
    
    for query in &search_queries {
        println!("\n🔎 Query: \"{}\"", query);
        
        // Traditional keyword search
        let keyword_results = memory.search(query, 3).await?;
        println!("   Keyword search found {} results:", keyword_results.len());
        for result in &keyword_results {
            println!("    • {} (score: {:.3})", result.entry.key, result.relevance_score);
        }
        
        // Semantic search using embeddings
        #[cfg(feature = "embeddings")]
        {
            let semantic_results = memory.semantic_search(query, Some(3)).await?;
            println!("   Semantic search found {} results:", semantic_results.len());
            for result in &semantic_results {
                println!("    • {} (similarity: {:.3}, distance: {:.3})", 
                    result.memory.key, result.similarity, result.distance);
            }
        }
        
        #[cfg(not(feature = "embeddings"))]
        {
            println!("   Semantic search: Not available (embeddings feature disabled)");
        }
    }
    
    println!("\n🔗 Example 3: Knowledge Graph Integration");
    println!("----------------------------------------");
    
    // Find related memories using knowledge graph
    let related_memories = memory.find_related_memories("ai_research", 5).await?;
    println!("✓ Found {} memories related to 'ai_research':", related_memories.len());
    for related in &related_memories {
        println!("  • {} (strength: {:.2})", related.memory_key, related.relationship_strength);
    }
    
    // Get knowledge graph statistics
    if let Some(kg_stats) = memory.knowledge_graph_stats() {
        println!("\n Knowledge Graph Statistics:");
        println!("  • Total nodes: {}", kg_stats.node_count);
        println!("  • Total edges: {}", kg_stats.edge_count);
        println!("  • Graph density: {:.4}", kg_stats.density);
        println!("  • Average connections per node: {:.2}", kg_stats.average_degree);
        if let Some(most_connected) = &kg_stats.most_connected_node {
            println!("  • Most connected node: {}", most_connected);
        }
    }
    
    println!("\n Example 4: Advanced Memory Operations");
    println!("---------------------------------------");
    
    // Demonstrate intelligent memory updates
    println!(" Updating existing memory with additional information...");
    memory.store("ai_research", 
        "Artificial intelligence research focuses on machine learning algorithms, neural networks, and deep learning models for computer vision and natural language processing").await?;
    
    // Show that the system intelligently merged the content
    let updated_memory = memory.retrieve("ai_research").await?;
    if let Some(mem) = updated_memory {
        println!("✓ Updated memory content length: {} characters", mem.value.len());
    }
    
    // Perform inference to discover new relationships
    let inference_results = memory.infer_relationships().await?;
    println!("✓ Discovered {} new relationships through inference:", inference_results.len());
    for result in &inference_results {
        println!("  • {} (confidence: {:.3}): {}", 
            result.relationship_type, result.confidence, result.explanation);
    }
    
    println!("\n Example 5: Performance Comparison");
    println!("-----------------------------------");
    
    // Compare different search methods
    let test_query = "machine learning algorithms";
    
    let start_time = std::time::Instant::now();
    let keyword_results = memory.search(test_query, 5).await?;
    let keyword_time = start_time.elapsed();
    
    #[cfg(feature = "embeddings")]
    let (semantic_results, semantic_time) = {
        let start_time = std::time::Instant::now();
        let results = memory.semantic_search(test_query, Some(5)).await?;
        let time = start_time.elapsed();
        (results, time)
    };
    
    println!(" Performance Results for query: \"{}\"", test_query);
    println!("   Keyword search: {} results in {:?}", keyword_results.len(), keyword_time);
    
    #[cfg(feature = "embeddings")]
    println!("   Semantic search: {} results in {:?}", semantic_results.len(), semantic_time);
    
    #[cfg(not(feature = "embeddings"))]
    println!("   Semantic search: Not available (embeddings feature disabled)");
    
    let graph_start = std::time::Instant::now();
    let graph_results = memory.find_related_memories("machine_learning", 5).await?;
    let graph_time = graph_start.elapsed();
    println!("   Graph traversal: {} results in {:?}", graph_results.len(), graph_time);
    
    println!("\n Phase 1 Advanced AI Integration Demo Complete!");
    println!("\nKey Features Demonstrated:");
    println!("•  Vector embeddings for semantic understanding");
    println!("•  Intelligent similarity search beyond keywords");
    println!("•  Integration with knowledge graph relationships");
    println!("•  Performance comparison of search methods");
    println!("•  Automatic relationship inference");
    println!("•  Smart memory updates and content merging");
    
    #[cfg(feature = "embeddings")]
    println!("•  Real-time embedding generation and caching");
    
    #[cfg(not(feature = "embeddings"))]
    println!("•   Embeddings feature disabled - enable with --features embeddings");
    
    Ok(())
}


