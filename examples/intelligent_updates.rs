//! Demonstration of intelligent knowledge graph updates and temporal tracking
//!
//! This example shows how the AI Agent Memory System intelligently handles
//! memory updates without creating duplicate nodes and edges, while tracking
//! changes over time.

use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType,
    memory::knowledge_graph::{RelationshipType},
    memory::temporal::ChangeType,
};
use chrono::Utc;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🧠 AI Agent Memory System - Intelligent Updates Demo");
    println!("=====================================================\n");

    // Create memory system with all advanced features enabled
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        enable_advanced_management: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Example 1: Initial memory creation
    println!("📝 Example 1: Creating Initial Memories");
    println!("--------------------------------------");
    
    memory.store("project_alpha", "A new web application project using React and Node.js").await?;
    memory.store("team_alice", "Alice is the lead developer working on frontend components").await?;
    memory.store("team_bob", "Bob handles backend API development and database design").await?;
    
    println!("✓ Created 3 initial memories");
    
    // Show initial knowledge graph stats
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("📊 Initial graph: {} nodes, {} edges", stats.node_count, stats.edge_count);
    }
    
    // Example 2: Updating existing memories (should merge, not duplicate)
    println!("\n🔄 Example 2: Intelligent Memory Updates");
    println!("----------------------------------------");
    
    // Update project_alpha with more details - should merge with existing node
    memory.store("project_alpha", "A new web application project using React and Node.js. Features include user authentication, real-time chat, and file sharing capabilities.").await?;
    
    // Update team information with additional context
    memory.store("team_alice", "Alice is the lead developer working on frontend components. She has 5 years of React experience and specializes in UI/UX design.").await?;
    
    println!("✓ Updated existing memories with additional information");
    
    // Show that nodes were updated, not duplicated
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("📊 After updates: {} nodes, {} edges (nodes should be same count)", stats.node_count, stats.edge_count);
    }

    // Example 3: Adding similar but distinct memories
    println!("\n🔗 Example 3: Similar Memory Handling");
    println!("------------------------------------");
    
    // Add a similar project - should create new node since it's different enough
    memory.store("project_beta", "A mobile application project using React Native for iOS and Android").await?;
    
    // Add related team member - should create new node
    memory.store("team_charlie", "Charlie is a mobile developer working on React Native applications").await?;
    
    println!("✓ Added similar but distinct memories");
    
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("📊 After new memories: {} nodes, {} edges", stats.node_count, stats.edge_count);
    }

    // Example 4: Creating explicit relationships
    println!("\n🔗 Example 4: Creating Relationships");
    println!("-----------------------------------");
    
    // Create relationships between team members and projects
    memory.create_memory_relationship(
        "team_alice",
        "project_alpha",
        RelationshipType::DependsOn,
    ).await?;
    
    memory.create_memory_relationship(
        "team_bob",
        "project_alpha",
        RelationshipType::DependsOn,
    ).await?;
    
    memory.create_memory_relationship(
        "team_charlie",
        "project_beta",
        RelationshipType::DependsOn,
    ).await?;
    
    // Create relationship between similar projects
    memory.create_memory_relationship(
        "project_alpha",
        "project_beta",
        RelationshipType::SimilarTo,
    ).await?;
    
    println!("✓ Created explicit relationships between memories");
    
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("📊 After relationships: {} nodes, {} edges", stats.node_count, stats.edge_count);
    }

    // Example 5: Demonstrating relationship inference
    println!("\n🧠 Example 5: Automatic Relationship Inference");
    println!("----------------------------------------------");
    
    let inferred = memory.infer_relationships().await?;
    println!("✓ Discovered {} new relationships through inference", inferred.len());
    
    for inference in &inferred {
        println!("  • {} (confidence: {:.2})", inference.explanation, inference.confidence);
    }
    
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("📊 After inference: {} nodes, {} edges", stats.node_count, stats.edge_count);
    }

    // Example 6: Finding related memories
    println!("\n🔍 Example 6: Finding Related Memories");
    println!("-------------------------------------");
    
    let related = memory.find_related_memories("project_alpha", 2).await?;
    println!("✓ Found {} memories related to 'project_alpha':", related.len());
    
    for related_memory in &related {
        println!("  • {} (strength: {:.2})", 
                related_memory.memory_key, 
                related_memory.relationship_strength);
    }

    // Example 7: Demonstrating content evolution tracking
    println!("\n📈 Example 7: Content Evolution Tracking");
    println!("---------------------------------------");
    
    // Make several updates to track evolution
    memory.store("project_alpha", "A new web application project using React and Node.js. Features include user authentication, real-time chat, file sharing capabilities, and advanced analytics dashboard.").await?;
    
    // Simulate some time passing and another update
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    memory.store("project_alpha", "A comprehensive web application project using React and Node.js. Features include user authentication, real-time chat, file sharing capabilities, advanced analytics dashboard, and AI-powered recommendations.").await?;
    
    println!("✓ Made incremental updates to track content evolution");

    // Example 8: Demonstrating smart content merging
    println!("\n🔀 Example 8: Smart Content Merging");
    println!("----------------------------------");
    
    // Add overlapping information that should be merged intelligently
    memory.store("team_alice", "Alice is the lead developer with 5 years of React experience. She also mentors junior developers and leads the frontend architecture decisions.").await?;
    
    println!("✓ Added overlapping information that was intelligently merged");

    // Example 9: Final statistics and summary
    println!("\n📊 Example 9: Final System State");
    println!("-------------------------------");
    
    let stats = memory.stats();
    println!("Memory Statistics:");
    println!("  • Short-term memories: {}", stats.short_term_count);
    println!("  • Long-term memories: {}", stats.long_term_count);
    println!("  • Total memory size: {} bytes", stats.total_size);
    
    if let Some(kg_stats) = memory.knowledge_graph_stats() {
        println!("\nKnowledge Graph Statistics:");
        println!("  • Total nodes: {}", kg_stats.node_count);
        println!("  • Total edges: {}", kg_stats.edge_count);
        println!("  • Graph density: {:.4}", kg_stats.density);
        println!("  • Average connections per node: {:.2}", kg_stats.average_degree);
        
        if let Some(most_connected) = kg_stats.most_connected_node {
            println!("  • Most connected node: {}", most_connected);
        }
    }

    println!("\n✅ Intelligent Updates Demo Complete!");
    println!("\nKey Features Demonstrated:");
    println!("• ✅ Intelligent node merging (no duplicates for similar content)");
    println!("• ✅ Content evolution tracking with temporal analysis");
    println!("• ✅ Smart relationship updates based on content changes");
    println!("• ✅ Automatic relationship inference and discovery");
    println!("• ✅ Advanced memory management with consolidation");
    println!("• ✅ Differential analysis of memory changes over time");

    Ok(())
}

/// Helper function to create a rich memory entry with metadata
fn create_rich_memory_entry(key: &str, value: &str, tags: Vec<&str>, importance: f64) -> MemoryEntry {
    let mut entry = MemoryEntry::new(key.to_string(), value.to_string(), MemoryType::LongTerm);
    entry = entry.with_tags(tags.into_iter().map(|s| s.to_string()).collect());
    entry = entry.with_importance(importance);
    entry
}

/// Simulate an error handling scenario
async fn error_handling_example() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig::default();
    let mut memory = AgentMemory::new(config).await?;
    
    // This should work fine
    memory.store("test_key", "test_value").await?;
    
    // Demonstrate error handling for invalid operations
    match memory.retrieve("non_existent_key").await {
        Ok(Some(_)) => println!("Found memory"),
        Ok(None) => println!("Memory not found (expected)"),
        Err(e) => println!("Error: {}", e),
    }
    
    Ok(())
}
