use synaptic::memory::temporal::{TemporalMemoryManager, TemporalConfig, TemporalQuery, ChangeType, EvolutionData};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::Result;

#[tokio::test]
async fn test_per_memory_evolution_metrics() -> Result<()> {
    let config = TemporalConfig::default();
    let mut manager = TemporalMemoryManager::new(config);

    let mut entry = MemoryEntry::new("test".to_string(), "v1".to_string(), MemoryType::ShortTerm);
    manager.track_memory_change(&entry, ChangeType::Created).await?;
    entry.update_value("v2".to_string());
    manager.track_memory_change(&entry, ChangeType::Updated).await?;

    let query = TemporalQuery {
        memory_key: Some("test".to_string()),
        time_range: None,
        change_types: vec![],
        min_significance: None,
        include_patterns: false,
        include_evolution: true,
    };

    let analysis = manager.analyze_temporal_patterns(query).await?;
    match analysis.evolution_metrics {
        Some(EvolutionData::PerMemory(m)) => assert!(m.total_changes >= 2),
        other => panic!("Unexpected metrics: {:?}", other),
    }
    Ok(())
}

#[tokio::test]
async fn test_global_evolution_metrics() -> Result<()> {
    let config = TemporalConfig::default();
    let mut manager = TemporalMemoryManager::new(config);

    let mut entry1 = MemoryEntry::new("k1".to_string(), "a".to_string(), MemoryType::ShortTerm);
    manager.track_memory_change(&entry1, ChangeType::Created).await?;
    entry1.update_value("b".to_string());
    manager.track_memory_change(&entry1, ChangeType::Updated).await?;

    let entry2 = MemoryEntry::new("k2".to_string(), "x".to_string(), MemoryType::ShortTerm);
    manager.track_memory_change(&entry2, ChangeType::Created).await?;

    let query = TemporalQuery {
        memory_key: None,
        time_range: None,
        change_types: vec![],
        min_significance: None,
        include_patterns: false,
        include_evolution: true,
    };

    let analysis = manager.analyze_temporal_patterns(query).await?;
    match analysis.evolution_metrics {
        Some(EvolutionData::Global(g)) => assert!(g.total_memories >= 2),
        other => panic!("Unexpected metrics: {:?}", other),
    }
    Ok(())
}
