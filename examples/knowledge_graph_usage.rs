//! Knowledge graph usage example for the AI Agent Memory system

use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType, StorageBackend,
    memory::knowledge_graph::{RelationshipType, NodeType},
    error::Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("  AI Agent Memory System - Knowledge Graph Example");
    println!("====================================================\n");

    // Example 1: Basic knowledge graph operations
    basic_knowledge_graph_operations().await?;

    // Example 2: Relationship creation and traversal
    relationship_creation_and_traversal().await?;

    // Example 3: Inference and reasoning
    inference_and_reasoning().await?;

    // Example 4: Complex graph queries
    complex_graph_queries().await?;

    println!("\n All knowledge graph examples completed successfully!");
    Ok(())
}

/// Demonstrate basic knowledge graph operations
async fn basic_knowledge_graph_operations() -> Result<()> {
    println!(" Example 1: Basic Knowledge Graph Operations");
    println!("----------------------------------------------");

    // Create memory system with knowledge graph enabled
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Store some related memories
    memory.store("user_alice", "Alice is a software engineer").await?;
    memory.store("project_alpha", "Project Alpha is a web application").await?;
    memory.store("technology_rust", "Rust is a systems programming language").await?;
    memory.store("meeting_standup", "Daily standup meeting at 9 AM").await?;

    println!("✓ Stored 4 memories in the system");

    // Get knowledge graph statistics
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("✓ Knowledge graph stats:");
        println!("  - Nodes: {}", stats.node_count);
        println!("  - Edges: {}", stats.edge_count);
        println!("  - Average degree: {:.2}", stats.average_degree);
    }

    println!();
    Ok(())
}

/// Demonstrate relationship creation and traversal
async fn relationship_creation_and_traversal() -> Result<()> {
    println!("🔗 Example 2: Relationship Creation and Traversal");
    println!("------------------------------------------------");

    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Store memories about a project
    memory.store("alice", "Alice is the lead developer").await?;
    memory.store("bob", "Bob is a frontend developer").await?;
    memory.store("project_web_app", "Building a web application for e-commerce").await?;
    memory.store("technology_react", "Using React for the frontend").await?;
    memory.store("technology_rust", "Using Rust for the backend API").await?;
    memory.store("deadline", "Project deadline is December 2024").await?;

    println!("✓ Stored project-related memories");

    // Create explicit relationships
    memory.create_memory_relationship(
        "alice",
        "project_web_app",
        RelationshipType::PartOf,
    ).await?;

    memory.create_memory_relationship(
        "bob",
        "project_web_app",
        RelationshipType::PartOf,
    ).await?;

    memory.create_memory_relationship(
        "technology_react",
        "project_web_app",
        RelationshipType::PartOf,
    ).await?;

    memory.create_memory_relationship(
        "technology_rust",
        "project_web_app",
        RelationshipType::PartOf,
    ).await?;

    memory.create_memory_relationship(
        "project_web_app",
        "deadline",
        RelationshipType::RelatedTo,
    ).await?;

    println!("✓ Created explicit relationships between memories");

    // Find related memories
    let related = memory.find_related_memories("project_web_app", 2).await?;
    println!("✓ Found {} memories related to 'project_web_app':", related.len());
    for related_memory in related {
        println!("  - {} (strength: {:.3})", 
            related_memory.memory_key, 
            related_memory.relationship_strength
        );
    }

    println!();
    Ok(())
}

/// Demonstrate inference and reasoning
async fn inference_and_reasoning() -> Result<()> {
    println!(" Example 3: Inference and Reasoning");
    println!("------------------------------------");

    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Store memories that can lead to inferences
    memory.store("coffee_shop", "Local coffee shop serves excellent espresso").await?;
    memory.store("alice_likes_coffee", "Alice loves drinking coffee").await?;
    memory.store("alice_works_nearby", "Alice works in the downtown office").await?;
    memory.store("coffee_shop_location", "Coffee shop is located downtown").await?;

    println!("✓ Stored memories for inference testing");

    // Create some relationships
    memory.create_memory_relationship(
        "alice_likes_coffee",
        "coffee_shop",
        RelationshipType::RelatedTo,
    ).await?;

    memory.create_memory_relationship(
        "alice_works_nearby",
        "coffee_shop_location",
        RelationshipType::SemanticallyRelated,
    ).await?;

    println!("✓ Created initial relationships");

    // Perform inference
    let inferences = memory.infer_relationships().await?;
    println!("✓ Discovered {} new relationships through inference:", inferences.len());
    
    for inference in inferences.iter().take(5) { // Show first 5
        println!("  - {} (confidence: {:.3}): {}", 
            inference.rule_id,
            inference.confidence,
            inference.explanation
        );
    }

    println!();
    Ok(())
}

/// Demonstrate complex graph queries
async fn complex_graph_queries() -> Result<()> {
    println!(" Example 4: Complex Graph Queries");
    println!("----------------------------------");

    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };
    let mut memory = AgentMemory::new(config).await?;

    // Create a more complex knowledge structure
    let memories = vec![
        ("person_alice", "Alice is a senior software engineer"),
        ("person_bob", "Bob is a product manager"),
        ("person_charlie", "Charlie is a UX designer"),
        ("project_mobile_app", "Mobile app for task management"),
        ("project_web_platform", "Web platform for team collaboration"),
        ("technology_flutter", "Flutter for cross-platform mobile development"),
        ("technology_typescript", "TypeScript for type-safe web development"),
        ("skill_leadership", "Leadership and team management skills"),
        ("skill_design", "User experience and interface design"),
        ("deadline_q1", "Q1 2024 deadline for mobile app"),
        ("deadline_q2", "Q2 2024 deadline for web platform"),
    ];

    for (key, value) in memories {
        memory.store(key, value).await?;
    }

    println!("✓ Created complex knowledge structure with {} memories", 11);

    // Create a web of relationships
    let relationships = vec![
        ("person_alice", "project_mobile_app", RelationshipType::PartOf),
        ("person_alice", "skill_leadership", RelationshipType::RelatedTo),
        ("person_bob", "project_mobile_app", RelationshipType::PartOf),
        ("person_bob", "project_web_platform", RelationshipType::PartOf),
        ("person_charlie", "project_web_platform", RelationshipType::PartOf),
        ("person_charlie", "skill_design", RelationshipType::RelatedTo),
        ("technology_flutter", "project_mobile_app", RelationshipType::PartOf),
        ("technology_typescript", "project_web_platform", RelationshipType::PartOf),
        ("project_mobile_app", "deadline_q1", RelationshipType::RelatedTo),
        ("project_web_platform", "deadline_q2", RelationshipType::RelatedTo),
    ];

    for (from, to, rel_type) in relationships {
        memory.create_memory_relationship(from, to, rel_type).await?;
    }

    println!("✓ Created {} explicit relationships", 10);

    // Perform complex queries
    println!("\n Complex Query Results:");

    // Find all memories related to Alice
    let alice_related = memory.find_related_memories("person_alice", 3).await?;
    println!("✓ Memories related to Alice ({} found):", alice_related.len());
    for related in alice_related.iter().take(5) {
        println!("  - {} (strength: {:.3})", 
            related.memory_key, 
            related.relationship_strength
        );
    }

    // Find all memories related to projects
    let project_memories = vec!["project_mobile_app", "project_web_platform"];
    for project in project_memories {
        let related = memory.find_related_memories(project, 2).await?;
        println!("✓ Memories related to {} ({} found)", project, related.len());
    }

    // Get final knowledge graph statistics
    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("\n Final Knowledge Graph Statistics:");
        println!("  - Total nodes: {}", stats.node_count);
        println!("  - Total edges: {}", stats.edge_count);
        println!("  - Graph density: {:.4}", stats.density);
        println!("  - Average connections per node: {:.2}", stats.average_degree);
        if let Some(most_connected) = stats.most_connected_node {
            println!("  - Most connected node: {}", most_connected);
        }
    }

    println!();
    Ok(())
}

/// Helper function to demonstrate memory entry creation with rich metadata
fn create_rich_memory_entry(key: &str, value: &str, tags: Vec<&str>) -> MemoryEntry {
    let mut entry = MemoryEntry::new(
        key.to_string(),
        value.to_string(),
        MemoryType::LongTerm,
    );
    
    entry.metadata = entry.metadata
        .with_tags(tags.iter().map(|s| s.to_string()).collect())
        .with_importance(0.8);
    
    // Add some custom fields for knowledge graph
    entry.metadata.set_custom_field("category".to_string(), "knowledge_graph_demo".to_string());
    entry.metadata.set_custom_field("source".to_string(), "example_usage".to_string());
    
    entry
}
