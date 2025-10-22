//! Comprehensive tests for memory promotion policies and hierarchical memory management.
//!
//! These tests verify that memories are automatically promoted from short-term to
//! long-term storage based on various criteria.

use synaptic::{AgentMemory, MemoryConfig};
use synaptic::memory::promotion::{
    PromotionConfig, PromotionPolicyType, AccessFrequencyPolicy, TimeBasedPolicy,
    ImportancePolicy, HybridPolicy, MemoryPromotionManager, MemoryPromotionPolicy,
};
use synaptic::memory::types::MemoryType;
use chrono::Duration;

#[tokio::test]
async fn test_access_frequency_promotion() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::AccessFrequency;
    promo_config.access_threshold = 3;
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory (short-term by default)
    memory.store("freq_test", "test content").await.unwrap();

    // Access it multiple times to trigger promotion
    for _ in 0..3 {
        let entry = memory.retrieve("freq_test").await.unwrap();
        assert!(entry.is_some(), "Memory should exist");
    }

    // On the next access, it should be promoted
    let entry = memory.retrieve("freq_test").await.unwrap().unwrap();
    assert_eq!(entry.memory_type, MemoryType::LongTerm,
        "Memory should be promoted to long-term after {} accesses", 3);
}

#[tokio::test]
async fn test_importance_promotion() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::Importance;
    promo_config.importance_threshold = 0.8;
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("importance_test", "important content").await.unwrap();

    // Manually set high importance (in real usage, consolidation would set this)
    // For this test, we'll need to retrieve, modify importance, and store again
    let mut entry = memory.retrieve("importance_test").await.unwrap().unwrap();
    entry.importance = 0.9;
    memory.storage().store(&entry).await.unwrap();

    // Retrieve again - should trigger promotion check
    let entry = memory.retrieve("importance_test").await.unwrap().unwrap();
    assert_eq!(entry.memory_type, MemoryType::LongTerm,
        "Memory with high importance should be promoted");
}

#[tokio::test]
async fn test_hybrid_policy_promotion() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::Hybrid;
    promo_config.access_threshold = 5;
    promo_config.importance_threshold = 0.8;
    promo_config.hybrid_promotion_threshold = 0.5;
    promo_config.hybrid_access_weight = 0.6;
    promo_config.hybrid_importance_weight = 0.4;
    promo_config.hybrid_time_weight = 0.0; // Disable time-based for this test
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store memory
    memory.store("hybrid_test", "content").await.unwrap();

    // Access several times (not quite enough for pure access frequency)
    for _ in 0..4 {
        memory.retrieve("hybrid_test").await.unwrap();
    }

    // Set moderate importance
    let mut entry = memory.retrieve("hybrid_test").await.unwrap().unwrap();
    entry.importance = 0.7;
    memory.storage().store(&entry).await.unwrap();

    // Hybrid policy should promote based on combined score
    let entry = memory.retrieve("hybrid_test").await.unwrap().unwrap();
    // With 4 accesses (80% of threshold) and 0.7 importance (87.5% of threshold)
    // Weighted: 0.8 * 0.6 + 0.875 * 0.4 = 0.48 + 0.35 = 0.83 > 0.5 threshold
    assert_eq!(entry.memory_type, MemoryType::LongTerm,
        "Hybrid policy should promote based on combined metrics");
}

#[tokio::test]
async fn test_promotion_disabled() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = false; // Disabled

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store and access many times
    memory.store("no_promote", "content").await.unwrap();
    for _ in 0..20 {
        memory.retrieve("no_promote").await.unwrap();
    }

    // Should still be short-term
    let entry = memory.retrieve("no_promote").await.unwrap().unwrap();
    assert_eq!(entry.memory_type, MemoryType::ShortTerm,
        "Memory should not be promoted when promotion is disabled");
}

#[tokio::test]
async fn test_long_term_not_re_promoted() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    config.promotion_config = Some(PromotionConfig::default());

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("already_long", "content").await.unwrap();

    // Manually promote to long-term
    let mut entry = memory.retrieve("already_long").await.unwrap().unwrap();
    entry.memory_type = MemoryType::LongTerm;
    memory.storage().store(&entry).await.unwrap();

    // Access many more times
    for _ in 0..20 {
        let e = memory.retrieve("already_long").await.unwrap().unwrap();
        assert_eq!(e.memory_type, MemoryType::LongTerm,
            "Should remain long-term");
    }
}

#[tokio::test]
async fn test_multiple_memories_promoted() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::AccessFrequency;
    promo_config.access_threshold = 3;
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store multiple memories
    for i in 0..5 {
        memory.store(&format!("mem_{}", i), &format!("content {}", i)).await.unwrap();
    }

    // Access them varying amounts
    for i in 0..5 {
        for _ in 0..=i {
            memory.retrieve(&format!("mem_{}", i)).await.unwrap();
        }
    }

    // Check promotion status
    for i in 0..5 {
        let entry = memory.retrieve(&format!("mem_{}", i)).await.unwrap().unwrap();
        if i >= 3 {
            assert_eq!(entry.memory_type, MemoryType::LongTerm,
                "Memory {} should be promoted", i);
        } else {
            assert_eq!(entry.memory_type, MemoryType::ShortTerm,
                "Memory {} should not be promoted yet", i);
        }
    }
}

#[tokio::test]
async fn test_promotion_updates_storage() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::AccessFrequency;
    promo_config.access_threshold = 2;
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("storage_test", "content").await.unwrap();

    // Access enough times to promote
    for _ in 0..3 {
        memory.retrieve("storage_test").await.unwrap();
    }

    // Directly check storage (bypassing state)
    let entry_from_storage = memory.storage().retrieve("storage_test").await.unwrap().unwrap();
    assert_eq!(entry_from_storage.memory_type, MemoryType::LongTerm,
        "Promoted memory should be updated in storage");
}

#[tokio::test]
async fn test_access_frequency_policy_unit() {
    let policy = AccessFrequencyPolicy::new(5);

    let mut entry = synaptic::memory::types::MemoryEntry::new(
        "test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );

    // Below threshold
    entry.access_count = 3;
    assert!(!policy.should_promote(&entry));
    assert!(policy.promotion_score(&entry) < 1.0);

    // At threshold
    entry.access_count = 5;
    assert!(policy.should_promote(&entry));
    assert_eq!(policy.promotion_score(&entry), 1.0);

    // Above threshold
    entry.access_count = 10;
    assert!(policy.should_promote(&entry));
    assert_eq!(policy.promotion_score(&entry), 1.0);
}

#[tokio::test]
async fn test_time_based_policy_unit() {
    let policy = TimeBasedPolicy::new(Duration::days(7));

    let mut entry = synaptic::memory::types::MemoryEntry::new(
        "test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );

    // Recent memory
    entry.created_at = chrono::Utc::now() - Duration::days(2);
    assert!(!policy.should_promote(&entry));

    // Old memory
    entry.created_at = chrono::Utc::now() - Duration::days(10);
    assert!(policy.should_promote(&entry));
}

#[tokio::test]
async fn test_importance_policy_unit() {
    let policy = ImportancePolicy::new(0.8);

    let mut entry = synaptic::memory::types::MemoryEntry::new(
        "test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );

    // Low importance
    entry.importance = 0.5;
    assert!(!policy.should_promote(&entry));

    // High importance
    entry.importance = 0.9;
    assert!(policy.should_promote(&entry));
}

#[tokio::test]
async fn test_hybrid_policy_unit() {
    let mut policy = HybridPolicy::new(0.5);
    policy.add_policy(Box::new(AccessFrequencyPolicy::new(5)), 0.5);
    policy.add_policy(Box::new(ImportancePolicy::new(0.8)), 0.5);

    let mut entry = synaptic::memory::types::MemoryEntry::new(
        "test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );

    // High access, low importance
    entry.access_count = 10;
    entry.importance = 0.3;
    assert!(policy.should_promote(&entry)); // Access alone pushes over threshold

    // Low access, high importance
    entry.access_count = 1;
    entry.importance = 0.9;
    assert!(policy.should_promote(&entry)); // Importance alone pushes over threshold

    // Both low
    entry.access_count = 1;
    entry.importance = 0.3;
    assert!(!policy.should_promote(&entry));
}

#[tokio::test]
async fn test_promotion_manager_unit() {
    let policy = Box::new(AccessFrequencyPolicy::new(3));
    let manager = MemoryPromotionManager::new(policy);

    let mut entry = synaptic::memory::types::MemoryEntry::new(
        "test".to_string(),
        "content".to_string(),
        MemoryType::ShortTerm,
    );
    entry.access_count = 5;

    assert!(manager.should_promote(&entry));

    let promoted = manager.promote_memory(entry).unwrap();
    assert_eq!(promoted.memory_type, MemoryType::LongTerm);
}

#[tokio::test]
async fn test_promotion_config_creates_correct_manager() {
    // Access frequency
    let mut config = PromotionConfig::default();
    config.policy_type = PromotionPolicyType::AccessFrequency;
    config.access_threshold = 10;
    let manager = config.create_manager();
    let (name, _) = manager.policy_info();
    assert_eq!(name, "AccessFrequency");

    // Time-based
    config.policy_type = PromotionPolicyType::TimeBased;
    let manager = config.create_manager();
    let (name, _) = manager.policy_info();
    assert_eq!(name, "TimeBased");

    // Importance
    config.policy_type = PromotionPolicyType::Importance;
    let manager = config.create_manager();
    let (name, _) = manager.policy_info();
    assert_eq!(name, "Importance");

    // Hybrid
    config.policy_type = PromotionPolicyType::Hybrid;
    let manager = config.create_manager();
    let (name, _) = manager.policy_info();
    assert_eq!(name, "Hybrid");
}

#[tokio::test]
async fn test_promotion_preserves_metadata() {
    let mut config = MemoryConfig::default();
    config.enable_memory_promotion = true;
    let mut promo_config = PromotionConfig::default();
    promo_config.policy_type = PromotionPolicyType::AccessFrequency;
    promo_config.access_threshold = 2;
    config.promotion_config = Some(promo_config);

    let mut memory = AgentMemory::new(config).await.unwrap();

    // Store a memory
    memory.store("metadata_test", "content").await.unwrap();

    // Set metadata
    let mut entry = memory.retrieve("metadata_test").await.unwrap().unwrap();
    entry.importance = 0.75;
    entry.tags = vec!["important".to_string(), "urgent".to_string()];
    let original_created_at = entry.created_at;
    memory.storage().store(&entry).await.unwrap();

    // Access to trigger promotion
    for _ in 0..3 {
        memory.retrieve("metadata_test").await.unwrap();
    }

    // Verify metadata preserved
    let promoted = memory.retrieve("metadata_test").await.unwrap().unwrap();
    assert_eq!(promoted.memory_type, MemoryType::LongTerm);
    assert_eq!(promoted.importance, 0.75, "Importance should be preserved");
    assert_eq!(promoted.tags, vec!["important", "urgent"], "Tags should be preserved");
    assert_eq!(promoted.created_at, original_created_at, "Created timestamp should be preserved");
}

#[tokio::test]
async fn test_concurrent_promotion_safe() {
    let config = MemoryConfig::default();
    let memory = std::sync::Arc::new(tokio::sync::Mutex::new(
        AgentMemory::new(config).await.unwrap()
    ));

    // Store memories
    for i in 0..10 {
        let mut m = memory.lock().await;
        m.store(&format!("concurrent_{}", i), &format!("content_{}", i)).await.unwrap();
    }

    // Access concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let mem = std::sync::Arc::clone(&memory);
        let handle = tokio::spawn(async move {
            for _ in 0..5 {
                let mut m = mem.lock().await;
                m.retrieve(&format!("concurrent_{}", i)).await.unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify no corruption
    for i in 0..10 {
        let entry = memory.lock().await
            .retrieve(&format!("concurrent_{}", i))
            .await
            .unwrap()
            .unwrap();
        assert!(entry.key.contains(&i.to_string()));
    }
}
