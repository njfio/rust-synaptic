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

    // Create complex relationships using the public API
    // Department-Project relationships
    memory.create_memory_relationship("engineering", "project_ai_platform", RelationshipType::Custom("manages".to_string())).await?;
    memory.create_memory_relationship("engineering", "project_mobile_app", RelationshipType::Custom("manages".to_string())).await?;
    memory.create_memory_relationship("engineering", "project_data_pipeline", RelationshipType::Custom("manages".to_string())).await?;
    memory.create_memory_relationship("operations", "project_security_audit", RelationshipType::Custom("manages".to_string())).await?;
    memory.create_memory_relationship("operations", "project_cloud_migration", RelationshipType::Custom("manages".to_string())).await?;

    // Employee-Department relationships
    memory.create_memory_relationship("alice_smith", "engineering", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("bob_johnson", "engineering", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "engineering", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "operations", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("eve_brown", "marketing", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("frank_miller", "operations", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("grace_taylor", "marketing", RelationshipType::Custom("belongs_to".to_string())).await?;
    memory.create_memory_relationship("henry_anderson", "sales", RelationshipType::Custom("belongs_to".to_string())).await?;

    // Employee-Project relationships
    memory.create_memory_relationship("alice_smith", "project_ai_platform", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("alice_smith", "project_data_pipeline", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("bob_johnson", "project_mobile_app", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "project_ai_platform", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "project_data_pipeline", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "project_cloud_migration", RelationshipType::Custom("works_on".to_string())).await?;
    memory.create_memory_relationship("frank_miller", "project_security_audit", RelationshipType::Custom("works_on".to_string())).await?;

        // Technology-Project relationships
        memory.create_memory_relationship("project_ai_platform", "tech_rust", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_ai_platform", "tech_python", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_ai_platform", "tech_tensorflow", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_mobile_app", "tech_react", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_data_pipeline", "tech_kafka", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_data_pipeline", "tech_postgresql", RelationshipType::Custom("uses".to_string())).await?;
        memory.create_memory_relationship("project_cloud_migration", "tech_kubernetes", RelationshipType::Custom("uses".to_string())).await?;

        // Employee-Technology expertise relationships
        memory.create_memory_relationship("alice_smith", "tech_rust", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("alice_smith", "tech_python", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("carol_davis", "tech_python", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("carol_davis", "tech_tensorflow", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("bob_johnson", "tech_react", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("david_wilson", "tech_kubernetes", RelationshipType::Custom("knows".to_string())).await?;
        memory.create_memory_relationship("david_wilson", "tech_redis", RelationshipType::Custom("knows".to_string())).await?;

    if let Some(stats) = memory.knowledge_graph_stats() {
        println!("✓ Created enterprise knowledge graph:");
        println!("  - {} nodes", stats.node_count);
        println!("  - {} edges", stats.edge_count);
        println!("  - {:.2} graph density", stats.density);
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

    // Create complex multi-layer relationships using the public API
    // Skill-Employee relationships
    memory.create_memory_relationship("alice_smith", "skill_machine_learning", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("alice_smith", "skill_backend_dev", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "skill_machine_learning", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "skill_data_analysis", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("bob_johnson", "skill_project_management", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("bob_johnson", "skill_frontend_dev", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "skill_devops", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "skill_system_design", RelationshipType::Custom("has".to_string())).await?;
    memory.create_memory_relationship("frank_miller", "skill_security", RelationshipType::Custom("has".to_string())).await?;

    // Certification-Employee relationships
    memory.create_memory_relationship("alice_smith", "cert_data_scientist", RelationshipType::Custom("holds".to_string())).await?;
    memory.create_memory_relationship("carol_davis", "cert_data_scientist", RelationshipType::Custom("holds".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "cert_aws_architect", RelationshipType::Custom("holds".to_string())).await?;
    memory.create_memory_relationship("david_wilson", "cert_kubernetes_admin", RelationshipType::Custom("holds".to_string())).await?;
    memory.create_memory_relationship("frank_miller", "cert_security_plus", RelationshipType::Custom("holds".to_string())).await?;
    memory.create_memory_relationship("bob_johnson", "cert_scrum_master", RelationshipType::Custom("holds".to_string())).await?;

    // Skill-Technology relationships
    memory.create_memory_relationship("skill_machine_learning", "tech_tensorflow", RelationshipType::DependsOn).await?;
    memory.create_memory_relationship("skill_machine_learning", "tech_python", RelationshipType::DependsOn).await?;
    memory.create_memory_relationship("skill_backend_dev", "tech_rust", RelationshipType::DependsOn).await?;
    memory.create_memory_relationship("skill_devops", "tech_kubernetes", RelationshipType::DependsOn).await?;
    memory.create_memory_relationship("skill_devops", "tech_redis", RelationshipType::DependsOn).await?;
    memory.create_memory_relationship("skill_frontend_dev", "tech_react", RelationshipType::DependsOn).await?;

    if let Some(stats) = memory.knowledge_graph_stats() {
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
        output_dir: std::path::PathBuf::from("./visualizations"),
        format: ImageFormat::PNG,
        color_scheme: ColorScheme::Default,
        width: 1920,
        height: 1080,
        font_size: 14,
        interactive: false,
    };

    let mut viz_engine = RealVisualizationEngine::new(viz_config).await?;

    // Generate network visualization using memory entries
    let all_memories = vec![
        MemoryEntry::new("alice_smith".to_string(), "Alice Smith - Senior Engineer".to_string(), MemoryType::LongTerm),
        MemoryEntry::new("bob_johnson".to_string(), "Bob Johnson - Project Manager".to_string(), MemoryType::LongTerm),
        MemoryEntry::new("carol_davis".to_string(), "Carol Davis - Data Scientist".to_string(), MemoryType::LongTerm),
        MemoryEntry::new("david_wilson".to_string(), "David Wilson - DevOps Engineer".to_string(), MemoryType::LongTerm),
        MemoryEntry::new("frank_miller".to_string(), "Frank Miller - Security Specialist".to_string(), MemoryType::LongTerm),
    ];

    let relationships = vec![
        ("alice_smith".to_string(), "skill_machine_learning".to_string(), 0.9),
        ("carol_davis".to_string(), "skill_data_analysis".to_string(), 0.9),
        ("david_wilson".to_string(), "skill_devops".to_string(), 0.9),
        ("bob_johnson".to_string(), "skill_project_management".to_string(), 0.8),
        ("frank_miller".to_string(), "skill_security".to_string(), 0.9),
    ];

    println!("✓ Generating enterprise network visualization...");
    let network_path = viz_engine.generate_memory_network(&all_memories, &relationships).await?;
    println!("  📊 Saved: {}", network_path);

    // Generate temporal analytics
    let stats = memory.stats();
    let temporal_data = vec![
        (chrono::Utc::now() - chrono::Duration::days(30), stats.short_term_count as f64),
        (chrono::Utc::now() - chrono::Duration::days(20), (stats.short_term_count as f64) * 1.2),
        (chrono::Utc::now() - chrono::Duration::days(10), (stats.short_term_count as f64) * 1.5),
        (chrono::Utc::now(), (stats.short_term_count as f64) * 1.8),
    ];

    println!("✓ Generating temporal analytics timeline...");
    // Generate analytics timeline instead
    let analytics_events = vec![]; // Empty for demo
    let timeline_path = viz_engine.generate_analytics_timeline(&analytics_events).await?;
    println!("  📊 Saved: {}", timeline_path);

    println!("✅ Complex visualizations generated successfully!");

    Ok(())
}
