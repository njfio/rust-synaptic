use synaptic::memory::temporal::{DiffAnalyzer, ChangeSet, ChangeType, TimeRange};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use chrono::{Utc, Duration};
use uuid::Uuid;

#[tokio::test]
async fn test_diff_analysis_with_compression() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = DiffAnalyzer::new();
    let m1 = MemoryEntry::new("m1".into(), "Hello world".into(), MemoryType::ShortTerm);
    let m2 = MemoryEntry::new("m1".into(), "Hello brave new world".into(), MemoryType::ShortTerm);
    let diff = analyzer.analyze_difference(&m1, &m2).await?;
    assert!(diff.compression_ratio.is_some());
    assert!(!diff.content_changes.additions.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_analyze_changes_in_range_filter() -> Result<(), Box<dyn std::error::Error>> {
    let mut analyzer = DiffAnalyzer::new();
    let now = Utc::now();
    let cs_old = ChangeSet {
        id: Uuid::new_v4(),
        timestamp: now - Duration::days(2),
        memory_key: "a".into(),
        change_type: ChangeType::Created,
        diffs: Vec::new(),
        impact_score: 0.5,
        description: "old".into(),
    };
    let cs_new = ChangeSet {
        id: Uuid::new_v4(),
        timestamp: now - Duration::hours(1),
        memory_key: "b".into(),
        change_type: ChangeType::Updated,
        diffs: Vec::new(),
        impact_score: 0.5,
        description: "new".into(),
    };
    analyzer.add_change_set(cs_old.clone());
    analyzer.add_change_set(cs_new.clone());
    let range = TimeRange::last_hours(24);
    let filtered = analyzer.analyze_changes_in_range(&range).await?;
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, cs_new.id);
    Ok(())
}
