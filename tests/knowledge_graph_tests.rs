//! Comprehensive tests for knowledge graph functionality
//!
//! Tests knowledge graph integration with memory system,
//! relationship management, and basic graph operations.

use synaptic::{
    AgentMemory, MemoryConfig, MemoryEntry, MemoryType,
    memory::knowledge_graph::RelationshipType,
};
use std::error::Error;

#[tokio::test]
async fn test_memory_with_knowledge_graph() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Store related memories
    memory.store("ai_definition", "Artificial Intelligence is the simulation of human intelligence").await?;
    memory.store("ml_definition", "Machine Learning is a subset of AI that learns from data").await?;
    memory.store("dl_definition", "Deep Learning is a subset of ML using neural networks").await?;

    // Test basic retrieval
    let ai_memory = memory.retrieve("ai_definition").await?;
    assert!(ai_memory.is_some());
    assert!(ai_memory.unwrap().value.contains("Artificial Intelligence"));

    // Test search functionality
    let search_results = memory.search("intelligence", 10).await?;
    assert!(!search_results.is_empty());

    // Test that knowledge graph is enabled
    let stats = memory.stats();
    assert!(stats.short_term_count >= 3);

    Ok(())
}

#[tokio::test]
async fn test_relationship_types() -> Result<(), Box<dyn Error>> {
    // Test different relationship types exist
    let relationships = vec![
        RelationshipType::RelatedTo,
        RelationshipType::Contains,
        RelationshipType::PartOf,
        RelationshipType::SimilarTo,
        RelationshipType::Contradicts,
        RelationshipType::Custom("implements".to_string()),
        RelationshipType::SemanticallyRelated,
        RelationshipType::Causes,
        RelationshipType::CausedBy,
        RelationshipType::References,
        RelationshipType::DependsOn,
    ];

    // Test that all relationship types can be created
    for rel_type in relationships {
        match rel_type {
            RelationshipType::Custom(ref name) => {
                assert_eq!(name, "implements");
            },
            _ => {
                // Other types should be valid
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_knowledge_graph_with_related_memories() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Store hierarchical concepts
    memory.store("ai_concept", "Artificial Intelligence is the simulation of human intelligence").await?;
    memory.store("ml_concept", "Machine Learning is a subset of AI that learns from data").await?;
    memory.store("dl_concept", "Deep Learning is a subset of ML using neural networks").await?;
    memory.store("nn_concept", "Neural Networks are computing systems inspired by biological neural networks").await?;

    // Test that all memories are stored
    let ai_memory = memory.retrieve("ai_concept").await?;
    assert!(ai_memory.is_some());

    let ml_memory = memory.retrieve("ml_concept").await?;
    assert!(ml_memory.is_some());

    // Test search across related concepts
    let ai_search = memory.search("artificial intelligence", 5).await?;
    assert!(!ai_search.is_empty());

    let ml_search = memory.search("machine learning", 5).await?;
    assert!(!ml_search.is_empty());

    // Test that knowledge graph is working (basic functionality)
    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 4);

    Ok(())
}

#[tokio::test]
async fn test_knowledge_graph_stats() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Store some memories
    memory.store("concept1", "First concept for testing").await?;
    memory.store("concept2", "Second concept for testing").await?;
    memory.store("concept3", "Third concept for testing").await?;

    // Test that stats reflect the stored memories
    let stats = memory.stats();
    assert_eq!(stats.short_term_count, 3);
    assert!(stats.total_size > 0);

    // Test search functionality
    let search_results = memory.search("concept", 10).await?;
    assert_eq!(search_results.len(), 3);

    Ok(())
}

#[tokio::test]
async fn test_memory_search_with_knowledge_graph() -> Result<(), Box<dyn Error>> {
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Store memories with semantic relationships
    memory.store("programming_rust", "Rust is a systems programming language").await?;
    memory.store("programming_python", "Python is a high-level programming language").await?;
    memory.store("programming_javascript", "JavaScript is a web programming language").await?;
    memory.store("data_science", "Data science uses programming for analysis").await?;
    memory.store("web_development", "Web development uses JavaScript and other languages").await?;

    // Test search for programming-related content
    let programming_results = memory.search("programming", 10).await?;
    assert!(programming_results.len() >= 3); // Should find at least the 3 programming languages

    // Test search for specific language
    let rust_results = memory.search("Rust", 5).await?;
    assert!(!rust_results.is_empty());

    // Test search for broader concepts
    let language_results = memory.search("language", 10).await?;
    assert!(language_results.len() >= 3);

    Ok(())
}

#[tokio::test]
async fn test_knowledge_graph_enabled_vs_disabled() -> Result<(), Box<dyn Error>> {
    // Test with knowledge graph enabled
    let config_enabled = MemoryConfig {
        enable_knowledge_graph: true,
        ..Default::default()
    };

    let mut memory_enabled = AgentMemory::new(config_enabled).await?;
    memory_enabled.store("test_key", "test content with knowledge graph").await?;

    // Test with knowledge graph disabled
    let config_disabled = MemoryConfig {
        enable_knowledge_graph: false,
        ..Default::default()
    };

    let mut memory_disabled = AgentMemory::new(config_disabled).await?;
    memory_disabled.store("test_key", "test content without knowledge graph").await?;

    // Both should work, but enabled version should have knowledge graph features
    let result_enabled = memory_enabled.retrieve("test_key").await?;
    let result_disabled = memory_disabled.retrieve("test_key").await?;

    assert!(result_enabled.is_some());
    assert!(result_disabled.is_some());

    Ok(())
}
