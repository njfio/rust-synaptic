//! Comprehensive tests for Domain Adaptation Engine
//!
//! Tests sophisticated domain adaptation algorithms including adversarial training,
//! feature alignment, and domain-invariant representations.

use synaptic::memory::meta_learning::{
    DomainAdaptationEngine, DomainAdaptationConfig, DomainAdaptationStrategy,
    Domain, domain_adaptation::DomainStatistics
};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use std::collections::HashMap;

/// Create test memory entries for different domains
fn create_domain_memories(count: usize, domain_prefix: &str, content_pattern: &str) -> Vec<MemoryEntry> {
    let mut memories = Vec::new();
    
    for i in 0..count {
        let content = format!("{} {} content item {}", domain_prefix, content_pattern, i);
        let key = format!("{}_{}", domain_prefix, i);
        
        let memory = MemoryEntry::new(
            key,
            content,
            MemoryType::LongTerm,
        );
        
        memories.push(memory);
    }
    
    memories
}

/// Create test domain with statistics
fn create_test_domain(id: &str, name: &str, sample_count: usize) -> Domain {
    let mut characteristics = HashMap::new();
    characteristics.insert("complexity".to_string(), 0.7);
    characteristics.insert("diversity".to_string(), 0.8);
    characteristics.insert("noise_level".to_string(), 0.2);
    
    let feature_stats = DomainStatistics {
        means: vec![0.5; 128],
        variances: vec![0.1; 128],
        correlations: vec![0.0; 128 * 128], // Flattened identity matrix
        correlation_dims: (128, 128),
        distribution_params: HashMap::new(),
    };
    
    Domain {
        id: id.to_string(),
        name: name.to_string(),
        characteristics,
        sample_count,
        feature_stats,
    }
}

#[tokio::test]
async fn test_domain_adaptation_engine_creation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig::default();
    let engine = DomainAdaptationEngine::new(config)?;
    
    assert_eq!(engine.get_domains().len(), 0);
    assert_eq!(engine.get_adaptation_history().len(), 0);
    assert_eq!(engine.get_metrics().total_adaptations, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_domain_registration() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig::default();
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    let source_domain = create_test_domain("source", "Source Domain", 100);
    let target_domain = create_test_domain("target", "Target Domain", 80);
    
    engine.register_domain(source_domain.clone()).await?;
    engine.register_domain(target_domain.clone()).await?;
    
    assert_eq!(engine.get_domains().len(), 2);
    assert!(engine.get_domains().contains_key("source"));
    assert!(engine.get_domains().contains_key("target"));
    
    Ok(())
}

#[tokio::test]
async fn test_adversarial_domain_adaptation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 100,
        convergence_threshold: 0.01,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    // Register domains
    let source_domain = create_test_domain("source", "Technical Domain", 50);
    let target_domain = create_test_domain("target", "Medical Domain", 50);
    
    engine.register_domain(source_domain).await?;
    engine.register_domain(target_domain).await?;
    
    // Create domain-specific data
    let source_data = create_domain_memories(20, "tech", "programming algorithm database");
    let target_data = create_domain_memories(20, "med", "patient diagnosis treatment");
    
    // Test adversarial adaptation
    let strategy = DomainAdaptationStrategy::Adversarial {
        discriminator_layers: vec![256, 128, 64, 1],
        gradient_reversal_lambda: 0.1,
    };
    
    let result = engine.adapt_domains(
        "source",
        "target", 
        &source_data,
        &target_data,
        Some(strategy)
    ).await?;
    
    assert_eq!(result.source_domain, "source");
    assert_eq!(result.target_domain, "target");
    assert!(result.adaptation_loss >= 0.0);
    assert!(result.domain_discrepancy >= 0.0);
    assert!(result.alignment_score >= 0.0 && result.alignment_score <= 1.0);
    assert!(result.adaptation_time_ms > 0);
    assert!(result.metrics.contains_key("initial_discrepancy"));
    assert!(result.metrics.contains_key("final_discrepancy"));
    
    // Check metrics update
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_adaptations, 1);
    assert!(metrics.avg_adaptation_time_ms > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_mmd_domain_adaptation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 50,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    // Register domains
    let source_domain = create_test_domain("source", "Text Domain", 30);
    let target_domain = create_test_domain("target", "Code Domain", 30);
    
    engine.register_domain(source_domain).await?;
    engine.register_domain(target_domain).await?;
    
    let source_data = create_domain_memories(15, "text", "document article paragraph");
    let target_data = create_domain_memories(15, "code", "function class method variable");
    
    let strategy = DomainAdaptationStrategy::MMD {
        kernel_type: "rbf".to_string(),
        bandwidth: 1.0,
    };
    
    let result = engine.adapt_domains(
        "source",
        "target",
        &source_data,
        &target_data,
        Some(strategy)
    ).await?;
    
    assert!(matches!(result.strategy, DomainAdaptationStrategy::MMD { .. }));
    assert!(result.metrics.contains_key("initial_mmd"));
    assert!(result.metrics.contains_key("final_mmd"));
    assert!(result.metrics.contains_key("mmd_reduction"));
    
    Ok(())
}

#[tokio::test]
async fn test_coral_domain_adaptation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 50,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    let source_domain = create_test_domain("source", "Formal Domain", 25);
    let target_domain = create_test_domain("target", "Informal Domain", 25);
    
    engine.register_domain(source_domain).await?;
    engine.register_domain(target_domain).await?;
    
    let source_data = create_domain_memories(12, "formal", "official document report");
    let target_data = create_domain_memories(12, "informal", "chat message conversation");
    
    let strategy = DomainAdaptationStrategy::CORAL {
        lambda: 0.5,
    };
    
    let result = engine.adapt_domains(
        "source",
        "target",
        &source_data,
        &target_data,
        Some(strategy)
    ).await?;
    
    assert!(matches!(result.strategy, DomainAdaptationStrategy::CORAL { .. }));
    assert!(result.metrics.contains_key("initial_coral"));
    assert!(result.metrics.contains_key("final_coral"));
    
    Ok(())
}

#[tokio::test]
async fn test_dan_domain_adaptation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 30,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    let source_domain = create_test_domain("source", "Academic Domain", 20);
    let target_domain = create_test_domain("target", "Business Domain", 20);
    
    engine.register_domain(source_domain).await?;
    engine.register_domain(target_domain).await?;
    
    let source_data = create_domain_memories(10, "academic", "research paper study");
    let target_data = create_domain_memories(10, "business", "meeting proposal strategy");
    
    let strategy = DomainAdaptationStrategy::DAN {
        adaptation_layers: vec!["layer1".to_string(), "layer2".to_string()],
        mmd_kernels: vec![0.5, 1.0, 2.0],
    };
    
    let result = engine.adapt_domains(
        "source",
        "target",
        &source_data,
        &target_data,
        Some(strategy)
    ).await?;
    
    assert!(matches!(result.strategy, DomainAdaptationStrategy::DAN { .. }));
    assert!(result.metrics.contains_key("multi_kernel_mmd"));
    assert!(result.metrics.contains_key("num_kernels"));
    assert_eq!(result.metrics["num_kernels"], 3.0);
    
    Ok(())
}

#[tokio::test]
async fn test_cdan_domain_adaptation() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 30,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    let source_domain = create_test_domain("source", "News Domain", 20);
    let target_domain = create_test_domain("target", "Social Domain", 20);
    
    engine.register_domain(source_domain).await?;
    engine.register_domain(target_domain).await?;
    
    let source_data = create_domain_memories(10, "news", "headline article breaking");
    let target_data = create_domain_memories(10, "social", "post comment like share");
    
    let strategy = DomainAdaptationStrategy::CDAN {
        entropy_conditioning: true,
        random_layer_dim: 64,
    };
    
    let result = engine.adapt_domains(
        "source",
        "target",
        &source_data,
        &target_data,
        Some(strategy)
    ).await?;
    
    assert!(matches!(result.strategy, DomainAdaptationStrategy::CDAN { .. }));
    assert!(result.metrics.contains_key("final_discrepancy"));
    assert!(result.metrics.contains_key("entropy_conditioning"));
    assert_eq!(result.metrics["entropy_conditioning"], 1.0);
    
    Ok(())
}

// Note: Feature extraction is tested indirectly through domain adaptation tests

#[tokio::test]
async fn test_multiple_adaptations_metrics() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig {
        max_iterations: 20,
        ..Default::default()
    };
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    // Register multiple domains
    for i in 0..3 {
        let domain = create_test_domain(&format!("domain_{}", i), &format!("Domain {}", i), 20);
        engine.register_domain(domain).await?;
    }
    
    // Perform multiple adaptations
    for i in 0..2 {
        let source_data = create_domain_memories(8, &format!("src_{}", i), "source content");
        let target_data = create_domain_memories(8, &format!("tgt_{}", i), "target content");
        
        let strategy = DomainAdaptationStrategy::Adversarial {
            discriminator_layers: vec![128, 64, 1],
            gradient_reversal_lambda: 0.1,
        };
        
        let _result = engine.adapt_domains(
            "domain_0",
            &format!("domain_{}", i + 1),
            &source_data,
            &target_data,
            Some(strategy)
        ).await?;
    }
    
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_adaptations, 2);
    assert!(metrics.avg_adaptation_time_ms > 0.0);
    assert!(metrics.strategy_performance.len() > 0);
    
    let history = engine.get_adaptation_history();
    assert_eq!(history.len(), 2);
    
    Ok(())
}

#[tokio::test]
async fn test_strategy_switching() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig::default();
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    // Test initial strategy
    assert!(matches!(engine.current_strategy, DomainAdaptationStrategy::Adversarial { .. }));
    
    // Switch to MMD strategy
    let new_strategy = DomainAdaptationStrategy::MMD {
        kernel_type: "rbf".to_string(),
        bandwidth: 2.0,
    };
    
    engine.set_strategy(new_strategy.clone());
    assert!(matches!(engine.current_strategy, DomainAdaptationStrategy::MMD { .. }));
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> synaptic::error::Result<()> {
    let config = DomainAdaptationConfig::default();
    let mut engine = DomainAdaptationEngine::new(config)?;
    
    // Test adaptation with unregistered domains
    let source_data = create_domain_memories(5, "test", "content");
    let target_data = create_domain_memories(5, "test", "content");
    
    let result = engine.adapt_domains(
        "nonexistent_source",
        "nonexistent_target",
        &source_data,
        &target_data,
        None
    ).await;
    
    assert!(result.is_err());
    
    // Test empty data
    let domain = create_test_domain("test", "Test Domain", 0);
    let register_result = engine.register_domain(domain).await;
    assert!(register_result.is_err());
    
    Ok(())
}
