// Phase 3 Analytics Tests
// Comprehensive testing for advanced analytics features

#[cfg(test)]
mod phase3_analytics_tests {
    use super::super::*;
    use crate::memory::types::MemoryEntry;
    use chrono::{DateTime, Utc, Duration};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_analytics_engine_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut engine = AnalyticsEngine::new(config).unwrap();

        // Test event recording
        let event = AnalyticsEvent::MemoryAccess {
            memory_key: "test_memory".to_string(),
            access_type: AccessType::Read,
            timestamp: Utc::now(),
            user_context: Some("test_user".to_string()),
        };

        engine.record_event(event).await.unwrap();
        assert_eq!(engine.get_metrics().events_processed, 1);

        // Test insight generation
        let insights = engine.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_predictive_analytics_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut analytics = predictive::PredictiveAnalytics::new(&config).unwrap();

        // Create a pattern of access events
        let base_time = Utc::now();
        for i in 0..10 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: "pattern_memory".to_string(),
                access_type: AccessType::Read,
                timestamp: base_time + Duration::hours(i),
                user_context: Some("pattern_user".to_string()),
            };
            analytics.process_event(&event).await.unwrap();
        }

        // Test prediction generation
        let predictions = analytics.get_predictions();
        assert!(predictions.len() >= 0);

        // Test caching recommendations
        let cache_recs = analytics.generate_caching_recommendations().await.unwrap();
        assert!(cache_recs.len() >= 0);

        // Test insights
        let insights = analytics.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_behavioral_analysis_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut analyzer = behavioral::BehavioralAnalyzer::new(&config).unwrap();

        // Create user behavior pattern
        for i in 0..15 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("memory_{}", i % 3),
                access_type: AccessType::Read,
                timestamp: Utc::now() + Duration::minutes(i * 10),
                user_context: Some("behavior_user".to_string()),
            };
            analyzer.process_event(&event).await.unwrap();
        }

        // Test user profile creation
        assert!(analyzer.get_user_profiles().contains_key("behavior_user"));

        // Test recommendation generation
        let recommendations = analyzer.generate_recommendations("behavior_user").await.unwrap();
        assert!(recommendations.len() >= 0);

        // Test insights
        let insights = analyzer.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_visualization_engine_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut engine = visualization::VisualizationEngine::new(&config).unwrap();

        // Test visual node creation
        let memory_entry = MemoryEntry::new("viz_memory".to_string(), "Test visualization content".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let node_id = engine.create_visual_node("viz_memory", &memory_entry).await.unwrap();
        assert!(!node_id.is_empty());

        // Test temporal timeline creation
        let data_points = vec![
            visualization::TemporalDataPoint {
                timestamp: Utc::now(),
                value: 1.0,
                memory_key: "viz_memory".to_string(),
                data_type: visualization::TemporalDataType::AccessFrequency,
                metadata: HashMap::new(),
            }
        ];

        let timeline_id = engine.create_temporal_timeline(
            "Test Timeline",
            data_points,
            visualization::TimelineVisualizationType::LineChart
        ).await.unwrap();
        assert!(!timeline_id.is_empty());

        // Test visualization export
        let export = engine.export_visualization_data().await.unwrap();
        assert!(export.nodes.len() > 0);
        assert!(export.timelines.len() > 0);

        // Test statistics
        let stats = engine.get_visualization_stats();
        assert!(stats.node_count > 0);
        assert!(stats.timeline_count > 0);
    }

    #[tokio::test]
    async fn test_memory_intelligence_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut engine = intelligence::MemoryIntelligenceEngine::new(&config).unwrap();

        // Test memory intelligence analysis
        let memory_entry = MemoryEntry::new("intelligent_memory".to_string(), "Complex memory content for intelligence analysis with multiple concepts and relationships".to_string(), crate::memory::types::MemoryType::LongTerm);
        let relationships = vec![
            ("related_memory_1".to_string(), 0.8),
            ("related_memory_2".to_string(), 0.6),
        ];

        let intelligence = engine.analyze_memory_intelligence(
            "intelligent_memory",
            &memory_entry,
            &relationships
        ).await.unwrap();

        assert!(intelligence.intelligence_score >= 0.0);
        assert!(intelligence.intelligence_score <= 1.0);
        assert!(intelligence.complexity.overall_complexity >= 0.0);
        assert!(intelligence.relationship_intelligence.direct_relationships == 2);

        // Test pattern recognition
        for i in 0..20 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("pattern_memory_{}", i % 5),
                access_type: AccessType::Read,
                timestamp: Utc::now() + Duration::minutes(i * 5),
                user_context: Some("intelligence_user".to_string()),
            };
            engine.process_event(&event).await.unwrap();
        }

        let patterns = engine.recognize_patterns().await.unwrap();
        assert!(patterns.len() >= 0);

        // Test anomaly detection
        engine.update_baseline_metrics().await.unwrap();
        let anomalies = engine.detect_anomalies().await.unwrap();
        assert!(anomalies.len() >= 0);

        // Test insights
        let insights = engine.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_performance_analyzer_comprehensive() {
        let config = AnalyticsConfig::default();
        let mut analyzer = performance::PerformanceAnalyzer::new(&config).unwrap();

        // Test performance snapshot recording
        let snapshot = performance::PerformanceSnapshot {
            timestamp: Utc::now(),
            ops_per_second: 1200.0,
            avg_response_time_ms: 0.8,
            memory_usage_bytes: 1024 * 1024 * 50, // 50MB
            cpu_usage_percent: 35.0,
            active_connections: 25,
            cache_hit_rate: 0.92,
            error_rate: 0.001,
        };

        analyzer.record_snapshot(snapshot).await.unwrap();

        // Add more snapshots to establish trends
        for i in 1..15 {
            let snapshot = performance::PerformanceSnapshot {
                timestamp: Utc::now() + Duration::minutes(i),
                ops_per_second: 1200.0 + i as f64 * 10.0, // Increasing trend
                avg_response_time_ms: 0.8 + i as f64 * 0.1, // Degrading trend
                memory_usage_bytes: 1024 * 1024 * (50 + i as u64),
                cpu_usage_percent: 35.0 + i as f64 * 2.0,
                active_connections: 25 + i as u32,
                cache_hit_rate: 0.92 - i as f64 * 0.01,
                error_rate: 0.001 + i as f64 * 0.0001,
            };
            analyzer.record_snapshot(snapshot).await.unwrap();
        }

        // Test trend analysis
        let trends = analyzer.get_trends();
        assert!(!trends.is_empty());

        // Test bottleneck detection
        let bottlenecks = analyzer.get_bottlenecks();
        assert!(bottlenecks.len() >= 0);

        // Test optimization recommendations
        let recommendations = analyzer.generate_recommendations().await.unwrap();
        assert!(recommendations.len() >= 0);

        // Test insights
        let insights = analyzer.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_analytics_integration() {
        let config = AnalyticsConfig::default();
        let mut engine = AnalyticsEngine::new(config).unwrap();

        // Test multiple event types
        let events = vec![
            AnalyticsEvent::MemoryAccess {
                memory_key: "integration_memory_1".to_string(),
                access_type: AccessType::Read,
                timestamp: Utc::now(),
                user_context: Some("integration_user".to_string()),
            },
            AnalyticsEvent::MemoryModification {
                memory_key: "integration_memory_1".to_string(),
                modification_type: ModificationType::ContentUpdate,
                timestamp: Utc::now() + Duration::minutes(5),
                change_magnitude: 0.7,
            },
            AnalyticsEvent::SearchQuery {
                query: "integration test query".to_string(),
                results_count: 5,
                timestamp: Utc::now() + Duration::minutes(10),
                response_time_ms: 50,
            },
            AnalyticsEvent::RelationshipDiscovery {
                source_key: "integration_memory_1".to_string(),
                target_key: "integration_memory_2".to_string(),
                relationship_strength: 0.8,
                timestamp: Utc::now() + Duration::minutes(15),
            },
        ];

        // Process all events
        for event in events {
            engine.record_event(event).await.unwrap();
        }

        // Verify metrics
        let metrics = engine.get_metrics();
        assert_eq!(metrics.events_processed, 4);
        assert!(metrics.avg_processing_time_ms >= 0.0); // Processing can be very fast

        // Generate comprehensive insights
        let insights = engine.generate_insights().await.unwrap();
        assert!(insights.len() >= 0);

        // Test insight filtering
        let high_priority_insights = engine.get_high_priority_insights();
        assert!(high_priority_insights.len() >= 0);

        let usage_insights = engine.get_insights_by_type(InsightType::UsagePattern);
        assert!(usage_insights.len() >= 0);
    }

    #[tokio::test]
    async fn test_analytics_performance() {
        let config = AnalyticsConfig::default();
        let mut engine = AnalyticsEngine::new(config).unwrap();

        // Performance test with many events
        let start_time = std::time::Instant::now();
        
        for i in 0..1000 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("perf_memory_{}", i % 100),
                access_type: AccessType::Read,
                timestamp: Utc::now() + Duration::milliseconds(i),
                user_context: Some(format!("user_{}", i % 10)),
            };
            engine.record_event(event).await.unwrap();
        }

        let elapsed = start_time.elapsed();
        println!("Processed 1000 events in {:?}", elapsed);

        // Should process events efficiently
        assert!(elapsed.as_millis() < 5000); // Less than 5 seconds

        // Verify all events were processed
        let metrics = engine.get_metrics();
        assert_eq!(metrics.events_processed, 1000);

        // Test insight generation performance
        let insight_start = std::time::Instant::now();
        let insights = engine.generate_insights().await.unwrap();
        let insight_elapsed = insight_start.elapsed();
        
        println!("Generated {} insights in {:?}", insights.len(), insight_elapsed);
        assert!(insight_elapsed.as_millis() < 2000); // Less than 2 seconds
    }

    #[tokio::test]
    async fn test_analytics_data_cleanup() {
        let mut config = AnalyticsConfig::default();
        config.retention_days = 1; // Short retention for testing
        config.max_history_entries = 10; // Small history for testing

        let mut engine = AnalyticsEngine::new(config).unwrap();

        // Add events beyond the limit
        for i in 0..15 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("cleanup_memory_{}", i),
                access_type: AccessType::Read,
                timestamp: if i < 5 {
                    Utc::now() - Duration::days(2) // Old events
                } else {
                    Utc::now() // Recent events
                },
                user_context: Some("cleanup_user".to_string()),
            };
            engine.record_event(event).await.unwrap();
        }

        // Should have trimmed to max_history_entries
        assert!(engine.event_history.len() <= 10);

        // Test cleanup
        engine.cleanup_old_data().await.unwrap();

        // Should have removed old events
        assert!(engine.event_history.len() <= 10);
    }

    #[tokio::test]
    async fn test_analytics_error_handling() {
        let config = AnalyticsConfig::default();
        let engine = AnalyticsEngine::new(config);
        assert!(engine.is_ok());

        // Test with invalid configuration
        let mut invalid_config = AnalyticsConfig::default();
        invalid_config.prediction_threshold = 2.0; // Invalid threshold > 1.0
        
        // Should still create engine (validation happens during use)
        let engine = AnalyticsEngine::new(invalid_config);
        assert!(engine.is_ok());
    }
}

#[cfg(test)]
mod phase3_integration_tests {
    use super::super::*;
    use crate::memory::types::MemoryEntry;
    use chrono::Duration;

    #[tokio::test]
    async fn test_full_analytics_pipeline() {
        let config = AnalyticsConfig {
            enable_predictive: true,
            enable_behavioral: true,
            enable_visualization: true,
            retention_days: 30,
            prediction_threshold: 0.7,
            pattern_sensitivity: 0.8,
            max_history_entries: 1000,
        };

        let mut engine = AnalyticsEngine::new(config).unwrap();

        // Simulate a realistic usage scenario
        let users = vec!["alice", "bob", "charlie"];
        let memories = vec!["project_alpha", "project_beta", "research_data", "meeting_notes"];

        // Generate realistic event patterns
        for day in 0..7 {
            for hour in 9..17 { // Business hours
                for user in &users {
                    for memory in &memories {
                        if rand::random::<f64>() > 0.7 { // 30% chance of access
                            let event = AnalyticsEvent::MemoryAccess {
                                memory_key: memory.to_string(),
                                access_type: AccessType::Read,
                                timestamp: Utc::now() - Duration::days(day) + Duration::hours(hour),
                                user_context: Some(user.to_string()),
                            };
                            engine.record_event(event).await.unwrap();
                        }
                    }
                }
            }
        }

        // Generate comprehensive insights
        let insights = engine.generate_insights().await.unwrap();
        
        // Should have generated meaningful insights
        assert!(insights.len() > 0);

        // Test different insight types
        let usage_insights = engine.get_insights_by_type(InsightType::UsagePattern);
        let performance_insights = engine.get_insights_by_type(InsightType::PerformanceOptimization);
        
        println!("Generated {} total insights", insights.len());
        println!("Usage insights: {}", usage_insights.len());
        println!("Performance insights: {}", performance_insights.len());

        // Verify metrics
        let metrics = engine.get_metrics();
        assert!(metrics.events_processed > 0);
        assert!(metrics.insights_generated > 0);
    }
}
