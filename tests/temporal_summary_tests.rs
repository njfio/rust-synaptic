use synaptic::{memory::temporal::{TemporalMemoryManager, TemporalConfig, TimeRange, ChangeType, MemoryVersion}, MemoryEntry, MemoryType};
use chrono::{Utc, Duration, Timelike};

#[tokio::test]
async fn test_most_active_period_daily() -> Result<(), Box<dyn std::error::Error>> {
    let manager = TemporalMemoryManager::new(TemporalConfig::default());
    let now = Utc::now() - Duration::days(3);
    let start_naive = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
    let start = start_naive.and_utc();
    let mut versions = Vec::new();

    // day 0
    for i in 0..2 {
        let mut mem = MemoryEntry::new(format!("d0_{}", i), "v".to_string(), MemoryType::ShortTerm);
        mem.metadata.created_at = start + Duration::hours(i);
        let mut v = MemoryVersion::new(mem, ChangeType::Created, i as u64);
        v.created_at = start + Duration::hours(i);
        versions.push(v);
    }
    // day 1 (most active)
    for i in 0..5 {
        let mut mem = MemoryEntry::new(format!("d1_{}", i), "v".to_string(), MemoryType::ShortTerm);
        mem.metadata.created_at = start + Duration::days(1) + Duration::hours(i);
        let mut v = MemoryVersion::new(mem, ChangeType::Updated, (10 + i) as u64);
        v.created_at = start + Duration::days(1) + Duration::hours(i);
        versions.push(v);
    }
    // day 2
    let mut mem = MemoryEntry::new("d2".to_string(), "v".to_string(), MemoryType::ShortTerm);
    mem.metadata.created_at = start + Duration::days(2);
    let mut v = MemoryVersion::new(mem, ChangeType::Updated, 20);
    v.created_at = start + Duration::days(2);
    versions.push(v);

    let range = TimeRange::new(start, start + Duration::days(3));
    let summary = manager.calculate_temporal_summary(&versions, &[], &range);
    let active = summary.most_active_period.expect("period");

    let expected_start_naive = (start + Duration::days(1)).date_naive().and_hms_opt(0,0,0).unwrap();
    let expected_start = expected_start_naive.and_utc();
    assert_eq!(active.start, expected_start);
    assert_eq!(active.end, expected_start + Duration::days(1));
    Ok(())
}

#[tokio::test]
async fn test_most_active_period_hourly() -> Result<(), Box<dyn std::error::Error>> {
    let manager = TemporalMemoryManager::new(TemporalConfig::default());
    let now = Utc::now() - Duration::hours(6);
    let start_naive = now.date_naive().and_hms_opt(now.hour(), 0, 0).unwrap();
    let start = start_naive.and_utc();
    let mut versions = Vec::new();

    // hour 1
    for i in 0..2 {
        let mut mem = MemoryEntry::new(format!("h1_{}", i), "v".to_string(), MemoryType::ShortTerm);
        mem.metadata.created_at = start + Duration::hours(1) + Duration::minutes(i as i64);
        let mut v = MemoryVersion::new(mem, ChangeType::Created, i as u64);
        v.created_at = start + Duration::hours(1) + Duration::minutes(i as i64);
        versions.push(v);
    }
    // hour 3 (most active)
    for i in 0..4 {
        let mut mem = MemoryEntry::new(format!("h3_{}", i), "v".to_string(), MemoryType::ShortTerm);
        mem.metadata.created_at = start + Duration::hours(3) + Duration::minutes(i as i64);
        let mut v = MemoryVersion::new(mem, ChangeType::Updated, (10 + i) as u64);
        v.created_at = start + Duration::hours(3) + Duration::minutes(i as i64);
        versions.push(v);
    }

    let range = TimeRange::new(start, start + Duration::hours(6));
    let summary = manager.calculate_temporal_summary(&versions, &[], &range);
    let active = summary.most_active_period.expect("period");

    let target = start + Duration::hours(3);
    let expected_start_naive = target.date_naive().and_hms_opt(target.hour(), 0, 0).unwrap();
    let expected_start = expected_start_naive.and_utc();
    assert_eq!(active.start, expected_start);
    assert_eq!(active.end, expected_start + Duration::hours(1));
    Ok(())
}
