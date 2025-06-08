// Complex Visualizations Example
// Demonstrates sophisticated visualization capabilities with large-scale data

use synaptic::{AgentMemory, MemoryConfig, MemoryEntry, MemoryType};
use synaptic::memory::knowledge_graph::RelationshipType;
use std::error::Error;
use std::collections::HashMap;

#[cfg(feature = "visualization")]
use synaptic::integrations::visualization::{VisualizationConfig, RealVisualizationEngine, ImageFormat, ColorScheme};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🎨 Synaptic Complex Visualizations Demo");
    println!("=======================================\n");

    // Create memory system with knowledge graph enabled
    let config = MemoryConfig {
        enable_knowledge_graph: true,
        enable_temporal_tracking: true,
        enable_advanced_management: true,
        ..Default::default()
    };

    let mut memory = AgentMemory::new(config).await?;

    // Example 1: Large-scale enterprise knowledge graph
    create_enterprise_knowledge_graph(&mut memory).await?;

    // Example 2: Complex temporal patterns
    create_temporal_patterns(&mut memory).await?;

    // Example 3: Multi-layered relationship networks
    create_multilayer_networks(&mut memory).await?;

    // Example 4: Generate sophisticated visualizations
    #[cfg(feature = "visualization")]
    generate_complex_visualizations(&memory).await?;

    println!("\n✅ Complex visualizations demo completed!");
    println!("📊 Check the visualizations/ directory for generated charts");

    Ok(())
}

/// Create a large-scale enterprise knowledge graph
async fn create_enterprise_knowledge_graph(memory: &mut AgentMemory) -> Result<(), Box<dyn Error>> {
    println!("🏢 Creating Enterprise Knowledge Graph");
    println!("-------------------------------------");

    // Create organizational structure
    let departments = vec![
        ("engineering", "Engineering Department - Core product development"),
        ("marketing", "Marketing Department - Brand and customer acquisition"),
        ("sales", "Sales Department - Revenue generation and client relations"),
        ("hr", "Human Resources - Talent management and culture"),
        ("finance", "Finance Department - Financial planning and analysis"),
        ("operations", "Operations Department - Business process optimization"),
    ];

    let projects = vec![
        ("project_ai_platform", "AI Platform - Next-generation machine learning infrastructure"),
        ("project_mobile_app", "Mobile Application - Customer-facing mobile experience"),
        ("project_data_pipeline", "Data Pipeline - Real-time analytics and processing"),
        ("project_security_audit", "Security Audit - Comprehensive security assessment"),
        ("project_cloud_migration", "Cloud Migration - Infrastructure modernization"),
        ("project_api_gateway", "API Gateway - Microservices communication layer"),
    ];

    let technologies = vec![
        ("tech_rust", "Rust Programming Language - Systems programming"),
        ("tech_python", "Python - Data science and automation"),
        ("tech_kubernetes", "Kubernetes - Container orchestration"),
        ("tech_postgresql", "PostgreSQL - Relational database"),
        ("tech_redis", "Redis - In-memory data structure store"),
        ("tech_kafka", "Apache Kafka - Distributed streaming platform"),
        ("tech_react", "React - Frontend user interface library"),
        ("tech_tensorflow", "TensorFlow - Machine learning framework"),
    ];

    let employees = vec![
        ("alice_smith", "Alice Smith - Senior Software Engineer"),
        ("bob_johnson", "Bob Johnson - Product Manager"),
        ("carol_davis", "Carol Davis - Data Scientist"),
        ("david_wilson", "David Wilson - DevOps Engineer"),
        ("eve_brown", "Eve Brown - UX Designer"),
        ("frank_miller", "Frank Miller - Security Specialist"),
        ("grace_taylor", "Grace Taylor - Marketing Director"),
        ("henry_anderson", "Henry Anderson - Sales Manager"),
    ];

    // Store all entities
    for (key, description) in &departments {
        memory.store(key, description).await?;
    }

    for (key, description) in &projects {
        memory.store(key, description).await?;
    }

    for (key, description) in &technologies {
        memory.store(key, description).await?;
    }

    for (key, description) in &employees {
        memory.store(key, description).await?;
    }

    // Create complex relationships
    if let Some(kg) = memory.knowledge_graph.as_mut() {
        // Department-Project relationships
        kg.add_relationship("engineering", "project_ai_platform", RelationshipType::Manages, 0.9).await?;
        kg.add_relationship("engineering", "project_mobile_app", RelationshipType::Manages, 0.8).await?;
        kg.add_relationship("engineering", "project_data_pipeline", RelationshipType::Manages, 0.9).await?;
        kg.add_relationship("operations", "project_security_audit", RelationshipType::Manages, 0.7).await?;
        kg.add_relationship("operations", "project_cloud_migration", RelationshipType::Manages, 0.8).await?;

        // Employee-Department relationships
        kg.add_relationship("alice_smith", "engineering", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("bob_johnson", "engineering", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("carol_davis", "engineering", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("david_wilson", "operations", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("eve_brown", "marketing", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("frank_miller", "operations", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("grace_taylor", "marketing", RelationshipType::BelongsTo, 1.0).await?;
        kg.add_relationship("henry_anderson", "sales", RelationshipType::BelongsTo, 1.0).await?;

        // Employee-Project relationships
        kg.add_relationship("alice_smith", "project_ai_platform", RelationshipType::WorksOn, 0.9).await?;
        kg.add_relationship("alice_smith", "project_data_pipeline", RelationshipType::WorksOn, 0.7).await?;
        kg.add_relationship("bob_johnson", "project_mobile_app", RelationshipType::WorksOn, 0.8).await?;
        kg.add_relationship("carol_davis", "project_ai_platform", RelationshipType::WorksOn, 0.9).await?;
        kg.add_relationship("carol_davis", "project_data_pipeline", RelationshipType::WorksOn, 0.8).await?;
        kg.add_relationship("david_wilson", "project_cloud_migration", RelationshipType::WorksOn, 0.9).await?;
        kg.add_relationship("frank_miller", "project_security_audit", RelationshipType::WorksOn, 0.9).await?;

        // Technology-Project relationships
        kg.add_relationship("project_ai_platform", "tech_rust", RelationshipType::Uses, 0.9).await?;
        kg.add_relationship("project_ai_platform", "tech_python", RelationshipType::Uses, 0.8).await?;
        kg.add_relationship("project_ai_platform", "tech_tensorflow", RelationshipType::Uses, 0.9).await?;
        kg.add_relationship("project_mobile_app", "tech_react", RelationshipType::Uses, 0.9).await?;
        kg.add_relationship("project_data_pipeline", "tech_kafka", RelationshipType::Uses, 0.9).await?;
        kg.add_relationship("project_data_pipeline", "tech_postgresql", RelationshipType::Uses, 0.8).await?;
        kg.add_relationship("project_cloud_migration", "tech_kubernetes", RelationshipType::Uses, 0.9).await?;

        // Employee-Technology expertise relationships
        kg.add_relationship("alice_smith", "tech_rust", RelationshipType::Knows, 0.9).await?;
        kg.add_relationship("alice_smith", "tech_python", RelationshipType::Knows, 0.7).await?;
        kg.add_relationship("carol_davis", "tech_python", RelationshipType::Knows, 0.9).await?;
        kg.add_relationship("carol_davis", "tech_tensorflow", RelationshipType::Knows, 0.8).await?;
        kg.add_relationship("bob_johnson", "tech_react", RelationshipType::Knows, 0.6).await?;
        kg.add_relationship("david_wilson", "tech_kubernetes", RelationshipType::Knows, 0.9).await?;
        kg.add_relationship("david_wilson", "tech_redis", RelationshipType::Knows, 0.8).await?;

        let stats = kg.get_statistics().await?;
        println!("✓ Created enterprise knowledge graph:");
        println!("  - {} nodes", stats.node_count);
        println!("  - {} edges", stats.edge_count);
        println!("  - {:.2} average degree", stats.average_degree);
    }

    Ok(())
}

/// Create complex temporal patterns
async fn create_temporal_patterns(memory: &mut AgentMemory) -> Result<(), Box<dyn Error>> {
    println!("\n⏰ Creating Temporal Patterns");
    println!("-----------------------------");

    // Simulate daily standup meetings
    for day in 1..=30 {
        let key = format!("standup_day_{}", day);
        let content = format!("Daily standup meeting - Day {}: Team sync, blockers discussion, sprint progress", day);
        memory.store(&key, &content).await?;
    }

    // Simulate weekly sprint reviews
    for week in 1..=4 {
        let key = format!("sprint_review_week_{}", week);
        let content = format!("Sprint review week {}: Demo completed features, retrospective, planning", week);
        memory.store(&key, &content).await?;
    }

    // Simulate monthly all-hands meetings
    for month in 1..=3 {
        let key = format!("allhands_month_{}", month);
        let content = format!("All-hands meeting month {}: Company updates, quarterly goals, team highlights", month);
        memory.store(&key, &content).await?;
    }

    println!("✓ Created temporal patterns:");
    println!("  - 30 daily standup meetings");
    println!("  - 4 weekly sprint reviews");
    println!("  - 3 monthly all-hands meetings");

    Ok(())
}

/// Create multi-layered relationship networks
async fn create_multilayer_networks(memory: &mut AgentMemory) -> Result<(), Box<dyn Error>> {
    println!("\n🕸️ Creating Multi-layered Networks");
    println!("----------------------------------");

    // Create skill networks
    let skills = vec![
        ("skill_machine_learning", "Machine Learning - AI and data science expertise"),
        ("skill_system_design", "System Design - Large-scale architecture planning"),
        ("skill_frontend_dev", "Frontend Development - User interface creation"),
        ("skill_backend_dev", "Backend Development - Server-side programming"),
        ("skill_devops", "DevOps - Infrastructure and deployment automation"),
        ("skill_data_analysis", "Data Analysis - Statistical analysis and insights"),
        ("skill_project_management", "Project Management - Team coordination and planning"),
        ("skill_security", "Security - Cybersecurity and threat assessment"),
    ];

    for (key, description) in &skills {
        memory.store(key, description).await?;
    }

    // Create certification networks
    let certifications = vec![
        ("cert_aws_architect", "AWS Solutions Architect - Cloud infrastructure design"),
        ("cert_kubernetes_admin", "Kubernetes Administrator - Container orchestration"),
        ("cert_data_scientist", "Certified Data Scientist - Advanced analytics"),
        ("cert_security_plus", "Security+ - Cybersecurity fundamentals"),
        ("cert_scrum_master", "Scrum Master - Agile project management"),
    ];

    for (key, description) in &certifications {
        memory.store(key, description).await?;
    }

    // Create complex multi-layer relationships
    if let Some(kg) = memory.knowledge_graph.as_mut() {
        // Skill-Employee relationships
        kg.add_relationship("alice_smith", "skill_machine_learning", RelationshipType::Has, 0.9).await?;
        kg.add_relationship("alice_smith", "skill_backend_dev", RelationshipType::Has, 0.8).await?;
        kg.add_relationship("carol_davis", "skill_machine_learning", RelationshipType::Has, 0.9).await?;
        kg.add_relationship("carol_davis", "skill_data_analysis", RelationshipType::Has, 0.9).await?;
        kg.add_relationship("bob_johnson", "skill_project_management", RelationshipType::Has, 0.8).await?;
        kg.add_relationship("bob_johnson", "skill_frontend_dev", RelationshipType::Has, 0.6).await?;
        kg.add_relationship("david_wilson", "skill_devops", RelationshipType::Has, 0.9).await?;
        kg.add_relationship("david_wilson", "skill_system_design", RelationshipType::Has, 0.7).await?;
        kg.add_relationship("frank_miller", "skill_security", RelationshipType::Has, 0.9).await?;

        // Certification-Employee relationships
        kg.add_relationship("alice_smith", "cert_data_scientist", RelationshipType::Holds, 1.0).await?;
        kg.add_relationship("carol_davis", "cert_data_scientist", RelationshipType::Holds, 1.0).await?;
        kg.add_relationship("david_wilson", "cert_aws_architect", RelationshipType::Holds, 1.0).await?;
        kg.add_relationship("david_wilson", "cert_kubernetes_admin", RelationshipType::Holds, 1.0).await?;
        kg.add_relationship("frank_miller", "cert_security_plus", RelationshipType::Holds, 1.0).await?;
        kg.add_relationship("bob_johnson", "cert_scrum_master", RelationshipType::Holds, 1.0).await?;

        // Skill-Technology relationships
        kg.add_relationship("skill_machine_learning", "tech_tensorflow", RelationshipType::RequiresKnowledgeOf, 0.9).await?;
        kg.add_relationship("skill_machine_learning", "tech_python", RelationshipType::RequiresKnowledgeOf, 0.8).await?;
        kg.add_relationship("skill_backend_dev", "tech_rust", RelationshipType::RequiresKnowledgeOf, 0.7).await?;
        kg.add_relationship("skill_devops", "tech_kubernetes", RelationshipType::RequiresKnowledgeOf, 0.9).await?;
        kg.add_relationship("skill_devops", "tech_redis", RelationshipType::RequiresKnowledgeOf, 0.6).await?;
        kg.add_relationship("skill_frontend_dev", "tech_react", RelationshipType::RequiresKnowledgeOf, 0.8).await?;

        let stats = kg.get_statistics().await?;
        println!("✓ Created multi-layered networks:");
        println!("  - {} total nodes", stats.node_count);
        println!("  - {} total edges", stats.edge_count);
        println!("  - {:.2} graph density", stats.density);
    }

    Ok(())
}

/// Generate sophisticated visualizations
#[cfg(feature = "visualization")]
async fn generate_complex_visualizations(memory: &AgentMemory) -> Result<(), Box<dyn Error>> {
    println!("\n🎨 Generating Complex Visualizations");
    println!("------------------------------------");

    let viz_config = VisualizationConfig {
        output_directory: "./visualizations".to_string(),
        image_format: ImageFormat::PNG,
        color_scheme: ColorScheme::Professional,
        width: 1920,
        height: 1080,
        enable_labels: true,
        enable_clustering: true,
        enable_animations: false,
    };

    let viz_engine = RealVisualizationEngine::new(viz_config)?;

    // Generate network visualization
    if let Some(kg) = &memory.knowledge_graph {
        let nodes = kg.get_all_nodes().await?;
        let edges = kg.get_all_edges().await?;

        println!("✓ Generating enterprise network visualization...");
        let network_path = viz_engine.create_network_visualization(
            &nodes,
            &edges,
            "Enterprise Knowledge Network"
        ).await?;
        println!("  📊 Saved: {}", network_path);

        // Generate hierarchical visualization
        println!("✓ Generating hierarchical organization chart...");
        let hierarchy_path = viz_engine.create_hierarchical_visualization(
            &nodes,
            &edges,
            "Organizational Hierarchy"
        ).await?;
        println!("  📊 Saved: {}", hierarchy_path);
    }

    // Generate temporal analytics
    let stats = memory.stats();
    let temporal_data = vec![
        (chrono::Utc::now() - chrono::Duration::days(30), stats.short_term_count as f64),
        (chrono::Utc::now() - chrono::Duration::days(20), (stats.short_term_count as f64) * 1.2),
        (chrono::Utc::now() - chrono::Duration::days(10), (stats.short_term_count as f64) * 1.5),
        (chrono::Utc::now(), (stats.short_term_count as f64) * 1.8),
    ];

    println!("✓ Generating temporal analytics timeline...");
    let timeline_path = viz_engine.create_timeline_visualization(
        &temporal_data,
        "Memory Growth Over Time",
        "Time",
        "Memory Count"
    ).await?;
    println!("  📊 Saved: {}", timeline_path);

    println!("✅ Complex visualizations generated successfully!");

    Ok(())
}
