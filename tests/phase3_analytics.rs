#[cfg(feature = "analytics")]
use synaptic::analytics::{behavioral::BehavioralAnalyzer, AnalyticsConfig, AnalyticsEvent, AccessType};
use synaptic::analytics::behavioral::RecommendationType;
#[cfg(feature = "analytics")]
use chrono::Utc;

#[cfg(feature = "analytics")]
#[tokio::test]
async fn test_collaboration_recommendation() {
    let config = AnalyticsConfig::default();
    let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();

    let now = Utc::now();

    // user_a accesses a unique memory and some common memories
    for key in ["project_design", "topic1", "topic2"] {
        let event = AnalyticsEvent::MemoryAccess {
            memory_key: key.to_string(),
            access_type: AccessType::Read,
            timestamp: now,
            user_context: Some("user_a".to_string()),
        };
        analyzer.process_event(&event).await.unwrap();
    }

    // user_b accesses related memories and shares common topics
    for key in ["project_design_notes", "topic1", "topic2"] {
        let event = AnalyticsEvent::MemoryAccess {
            memory_key: key.to_string(),
            access_type: AccessType::Read,
            timestamp: now,
            user_context: Some("user_b".to_string()),
        };
        analyzer.process_event(&event).await.unwrap();
    }

    // Generate recommendations for user_a
    let recs = analyzer.generate_recommendations("user_a").await.unwrap();
    let found = recs.iter().any(|r| r.recommendation_type == RecommendationType::CollaborationSuggestion && r.target == "project_design");
    assert!(found, "expected collaboration suggestion for project_design");
}

#[cfg(feature = "analytics")]
#[tokio::test]
async fn test_no_collaboration_suggestion() {
    let config = AnalyticsConfig::default();
    let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();
    let now = Utc::now();

    // Only user_a accesses this memory
    let event = AnalyticsEvent::MemoryAccess {
        memory_key: "solo_project".to_string(),
        access_type: AccessType::Read,
        timestamp: now,
        user_context: Some("user_a".to_string()),
    };
    analyzer.process_event(&event).await.unwrap();

    // unrelated user_c accesses other memory
    let event2 = AnalyticsEvent::MemoryAccess {
        memory_key: "unrelated".to_string(),
        access_type: AccessType::Read,
        timestamp: now,
        user_context: Some("user_c".to_string()),
    };
    analyzer.process_event(&event2).await.unwrap();

    let recs = analyzer.generate_recommendations("user_a").await.unwrap();
    let collab = recs.iter().any(|r| r.recommendation_type == RecommendationType::CollaborationSuggestion);
    assert!(!collab, "no collaboration suggestions expected");
}
