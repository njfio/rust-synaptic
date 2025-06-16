use synaptic::memory::management::{AdvancedMemoryManager, MemoryManagementConfig};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryMetadata, MemoryType};
use chrono::Utc;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🧠 Enhanced Memory Statistics Demo");
    println!("==================================\n");

    // Create storage and memory manager
    let storage = MemoryStorage::new();
    let config = MemoryManagementConfig {
        enable_analytics: true,
        enable_lifecycle_management: true,
        enable_auto_optimization: true,
        enable_auto_summarization: true,
        ..Default::default()
    };
    let mut manager = AdvancedMemoryManager::new(config);

    // Create sample memories with diverse characteristics
    println!("📝 Creating sample memories...");
    
    let memories = vec![
        create_memory("project_alpha", "Project Alpha planning document with detailed requirements and specifications", vec!["project", "planning", "alpha"]),
        create_memory("meeting_notes_1", "Weekly team meeting notes discussing project progress and blockers", vec!["meeting", "notes", "team"]),
        create_memory("task_backend", "Backend API development task for user authentication system", vec!["task", "backend", "api", "auth"]),
        create_memory("research_ai", "Research findings on AI integration possibilities for the platform", vec!["research", "ai", "integration"]),
        create_memory("bug_report_1", "Critical bug in payment processing system affecting user transactions", vec!["bug", "critical", "payment"]),
        create_memory("design_mockups", "UI/UX design mockups for the new dashboard interface", vec!["design", "ui", "ux", "dashboard"]),
        create_memory("performance_metrics", "System performance metrics and optimization recommendations", vec!["performance", "metrics", "optimization"]),
        create_memory("user_feedback", "Compilation of user feedback from recent surveys and support tickets", vec!["feedback", "users", "survey"]),
        create_memory("security_audit", "Security audit report with vulnerability assessments and recommendations", vec!["security", "audit", "vulnerability"]),
        create_memory("deployment_guide", "Step-by-step deployment guide for production environment setup", vec!["deployment", "production", "guide"]),
    ];

    // Add memories to storage and manager
    for memory in memories {
        storage.store(&memory).await?;
        manager.add_memory(&storage, memory, None).await?;
    }

    // Simulate some access patterns
    println!("🔄 Simulating access patterns...");
    for i in 0..5 {
        if let Some(memory) = storage.retrieve("project_alpha").await? {
            let mut updated_memory = memory.clone();
            updated_memory.metadata.access_count += 1;
            updated_memory.metadata.last_accessed = Utc::now();
            storage.store(&updated_memory).await?;
        }
    }

    // Generate comprehensive statistics
    println!("📊 Generating comprehensive memory statistics...\n");
    let stats = manager.get_management_stats(&storage).await?;

    // Display basic statistics
    println!("📈 BASIC STATISTICS");
    println!("==================");
    println!("Total Memories: {}", stats.basic_stats.total_memories);
    println!("Active Memories: {}", stats.basic_stats.active_memories);
    println!("Archived Memories: {}", stats.basic_stats.archived_memories);
    println!("Deleted Memories: {}", stats.basic_stats.deleted_memories);
    println!("Average Age (days): {:.1}", stats.basic_stats.avg_memory_age_days);
    println!("Utilization Efficiency: {:.1}%", stats.basic_stats.utilization_efficiency * 100.0);
    println!("Total Summarizations: {}", stats.basic_stats.total_summarizations);
    println!("Total Optimizations: {}", stats.basic_stats.total_optimizations);

    // Display advanced analytics
    println!("\n🔍 ADVANCED ANALYTICS");
    println!("=====================");
    println!("Size Distribution:");
    println!("  Min Size: {} bytes", stats.analytics.size_distribution.min_size);
    println!("  Max Size: {} bytes", stats.analytics.size_distribution.max_size);
    println!("  Median Size: {} bytes", stats.analytics.size_distribution.median_size);
    println!("  95th Percentile: {} bytes", stats.analytics.size_distribution.percentile_95);
    
    println!("\nContent Types:");
    for (content_type, count) in &stats.analytics.content_types.types {
        println!("  {}: {} memories", content_type, count);
    }
    println!("  Dominant Type: {}", stats.analytics.content_types.dominant_type);
    println!("  Diversity Index: {:.3}", stats.analytics.content_types.diversity_index);

    println!("\nTag Statistics:");
    println!("  Total Unique Tags: {}", stats.analytics.tag_statistics.total_unique_tags);
    println!("  Average Tags per Memory: {:.1}", stats.analytics.tag_statistics.avg_tags_per_memory);
    println!("  Most Popular Tags:");
    for (tag, count) in stats.analytics.tag_statistics.most_popular_tags.iter().take(5) {
        println!("    {}: {} uses", tag, count);
    }

    // Display trend analysis
    println!("\n📈 TREND ANALYSIS");
    println!("=================");
    println!("Growth Trend:");
    println!("  Direction: {:?}", stats.trends.growth_trend.trend_direction);
    println!("  Strength: {:.3}", stats.trends.growth_trend.trend_strength);
    println!("  7-day Prediction: {:.1}", stats.trends.growth_trend.prediction_7d);
    println!("  30-day Prediction: {:.1}", stats.trends.growth_trend.prediction_30d);

    println!("\nAccess Trend:");
    println!("  Direction: {:?}", stats.trends.access_trend.trend_direction);
    println!("  Strength: {:.3}", stats.trends.access_trend.trend_strength);

    // Display predictive metrics
    println!("\n🔮 PREDICTIVE METRICS");
    println!("=====================");
    println!("30-day Predictions:");
    println!("  Memory Count: {:.0}", stats.predictions.predicted_memory_count_30d);
    println!("  Storage Usage: {:.1} MB", stats.predictions.predicted_storage_mb_30d);
    
    println!("\nOptimization Forecast:");
    println!("  Next Optimization: {}", stats.predictions.optimization_forecast.next_optimization_recommended.format("%Y-%m-%d"));
    println!("  Urgency: {:?}", stats.predictions.optimization_forecast.optimization_urgency);
    println!("  Expected Gain: {:.1}%", stats.predictions.optimization_forecast.expected_performance_gain * 100.0);

    println!("\nRisk Assessment:");
    println!("  Overall Risk: {:?}", stats.predictions.risk_assessment.overall_risk_level);
    println!("  Capacity Risk: {:.1}%", stats.predictions.risk_assessment.capacity_risk * 100.0);
    println!("  Performance Risk: {:.1}%", stats.predictions.risk_assessment.performance_risk * 100.0);

    // Display performance metrics
    println!("\n⚡ PERFORMANCE METRICS");
    println!("=====================");
    println!("Average Latency: {:.1} ms", stats.performance.avg_operation_latency_ms);
    println!("Operations/Second: {:.0}", stats.performance.operations_per_second);
    println!("Cache Hit Rate: {:.1}%", stats.performance.cache_hit_rate * 100.0);
    println!("Index Efficiency: {:.1}%", stats.performance.index_efficiency * 100.0);
    println!("Compression Effectiveness: {:.1}%", stats.performance.compression_effectiveness * 100.0);

    // Display content analysis
    println!("\n📝 CONTENT ANALYSIS");
    println!("===================");
    println!("Average Content Length: {:.0} characters", stats.content_analysis.avg_content_length);
    println!("Complexity Score: {:.3}", stats.content_analysis.complexity_score);
    println!("Semantic Diversity: {:.3}", stats.content_analysis.semantic_diversity);
    println!("Duplicate Content: {:.1}%", stats.content_analysis.duplicate_content_percentage * 100.0);
    
    println!("\nQuality Metrics:");
    println!("  Readability: {:.3}", stats.content_analysis.quality_metrics.readability_score);
    println!("  Information Density: {:.3}", stats.content_analysis.quality_metrics.information_density);
    println!("  Structural Consistency: {:.3}", stats.content_analysis.quality_metrics.structural_consistency);
    println!("  Metadata Completeness: {:.3}", stats.content_analysis.quality_metrics.metadata_completeness);

    // Display health indicators
    println!("\n🏥 SYSTEM HEALTH");
    println!("================");
    println!("Overall Health: {:.1}%", stats.health_indicators.overall_health_score * 100.0);
    println!("Data Integrity: {:.1}%", stats.health_indicators.data_integrity_score * 100.0);
    println!("Performance Health: {:.1}%", stats.health_indicators.performance_health_score * 100.0);
    println!("Storage Health: {:.1}%", stats.health_indicators.storage_health_score * 100.0);
    println!("Active Issues: {}", stats.health_indicators.active_issues_count);
    
    println!("\nRecommendations:");
    for (i, recommendation) in stats.health_indicators.improvement_recommendations.iter().enumerate() {
        println!("  {}. {}", i + 1, recommendation);
    }

    println!("\n✅ Enhanced memory statistics demonstration completed!");
    println!("The system now provides comprehensive analytics including:");
    println!("  • Basic statistics and utilization metrics");
    println!("  • Advanced analytics with size and content distribution");
    println!("  • Trend analysis with predictive modeling");
    println!("  • Performance metrics and optimization insights");
    println!("  • Content quality analysis and health indicators");
    println!("  • Actionable recommendations for system improvement");

    Ok(())
}

fn create_memory(key: &str, content: &str, tags: Vec<&str>) -> MemoryEntry {
    let metadata = MemoryMetadata::new()
        .with_tags(tags.into_iter().map(|s| s.to_string()).collect())
        .with_importance(0.7)
        .with_confidence(0.8);

    MemoryEntry {
        key: key.to_string(),
        value: content.to_string(),
        memory_type: MemoryType::LongTerm,
        metadata,
        embedding: None,
    }
}
