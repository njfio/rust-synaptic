//! Query understanding / decomposition tests.
//!
//! Covers the deterministic query-preprocessing stage:
//! (a) temporal-constraint extraction ("in 2021", "before 2020", "last year")
//!     that boosts temporally-matching memories in the hybrid pipeline;
//! (b) multi-part question splitting ("... and where does Bob work?") whose
//!     sub-query results are unioned so both facts surface.

// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use chrono::{Datelike, TimeZone, Utc};
use std::sync::Arc;
use synaptic::memory::retrieval::{
    HybridRetriever, KeywordRetriever, PipelineConfig, QueryPlan, TemporalConstraint,
};
use synaptic::memory::storage::{memory::MemoryStorage, Storage};
use synaptic::memory::types::{MemoryEntry, MemoryType};

// ---------------------------------------------------------------------------
// Unit: temporal-constraint extraction
// ---------------------------------------------------------------------------

#[test]
fn extracts_explicit_year_constraint() {
    let plan = QueryPlan::analyze("what did Alice do in 2021?");
    let temporal = plan
        .temporal
        .as_ref()
        .expect("query with 'in 2021' must yield a temporal constraint");
    assert_eq!(temporal.start.year(), 2021);
    assert!(temporal.start <= Utc.with_ymd_and_hms(2021, 6, 1, 0, 0, 0).unwrap());
    assert!(temporal.end >= Utc.with_ymd_and_hms(2021, 12, 31, 23, 59, 59).unwrap());
    // The whole question is still a single retrieval unit.
    assert_eq!(plan.sub_queries.len(), 1);
}

#[test]
fn extracts_before_year_as_open_start_range() {
    let plan = QueryPlan::analyze("what happened before 2020?");
    let temporal = plan.temporal.as_ref().expect("'before 2020' must parse");
    assert!(temporal.end <= Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap());
    assert!(
        temporal.start.year() < 2000,
        "start must be open/very early"
    );
}

#[test]
fn extracts_after_year_as_open_end_range() {
    let plan = QueryPlan::analyze("trips taken after 2022");
    let temporal = plan.temporal.as_ref().expect("'after 2022' must parse");
    assert!(temporal.start >= Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap());
    assert!(temporal.end.year() > 2100, "end must be open/far future");
}

#[test]
fn extracts_year_span_between_two_years() {
    let plan = QueryPlan::analyze("projects between 2019 and 2021");
    let temporal = plan.temporal.as_ref().expect("year span must parse");
    assert_eq!(temporal.start.year(), 2019);
    assert!(temporal.end >= Utc.with_ymd_and_hms(2021, 12, 31, 0, 0, 0).unwrap());
}

#[test]
fn extracts_relative_last_year_deterministically() {
    let now = Utc.with_ymd_and_hms(2026, 7, 14, 12, 0, 0).unwrap();
    let plan = QueryPlan::analyze_at("what did Alice cook last year?", now);
    let temporal = plan.temporal.as_ref().expect("'last year' must parse");
    assert_eq!(temporal.start.year(), 2025);
    assert!(temporal.end <= Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap());
}

#[test]
fn extracts_relative_last_summer_deterministically() {
    let now = Utc.with_ymd_and_hms(2026, 3, 10, 12, 0, 0).unwrap();
    let plan = QueryPlan::analyze_at("where did Bob travel last summer?", now);
    let temporal = plan.temporal.as_ref().expect("'last summer' must parse");
    // Most recent completed summer before March 2026 is summer 2025.
    assert_eq!(temporal.start.year(), 2025);
    assert_eq!(temporal.start.month(), 6);
    assert_eq!(temporal.end.year(), 2025);
}

#[test]
fn temporal_constraint_matches_created_at_and_content_year() {
    let plan = QueryPlan::analyze("what did Alice do in 2021?");
    let temporal = plan.temporal.as_ref().unwrap();

    let mut in_range = MemoryEntry::new(
        "k1".into(),
        "alice painted the fence".into(),
        MemoryType::LongTerm,
    );
    in_range.metadata.created_at = Utc.with_ymd_and_hms(2021, 5, 1, 0, 0, 0).unwrap();
    assert!(temporal.matches(&in_range));

    let mut out_of_range = MemoryEntry::new(
        "k2".into(),
        "alice painted the shed".into(),
        MemoryType::LongTerm,
    );
    out_of_range.metadata.created_at = Utc.with_ymd_and_hms(2019, 5, 1, 0, 0, 0).unwrap();
    assert!(!temporal.matches(&out_of_range));

    // Content mentioning the constrained year matches even if created later.
    let mut content_dated = MemoryEntry::new(
        "k3".into(),
        "in 2021 alice painted the barn".into(),
        MemoryType::LongTerm,
    );
    content_dated.metadata.created_at = Utc.with_ymd_and_hms(2024, 5, 1, 0, 0, 0).unwrap();
    assert!(temporal.matches(&content_dated));
}

// ---------------------------------------------------------------------------
// Unit: multi-part splitting
// ---------------------------------------------------------------------------

#[test]
fn splits_two_part_conjunction_question() {
    let plan = QueryPlan::analyze("where does Alice live and where does Bob work?");
    assert_eq!(
        plan.sub_queries.len(),
        2,
        "two-part question must split into 2 sub-queries: {:?}",
        plan.sub_queries
    );
    assert!(plan.sub_queries[0].to_lowercase().contains("alice"));
    assert!(plan.sub_queries[1].to_lowercase().contains("bob"));
}

#[test]
fn splits_on_multiple_question_marks() {
    let plan = QueryPlan::analyze("who is Alice? who is Bob?");
    assert_eq!(plan.sub_queries.len(), 2, "{:?}", plan.sub_queries);
}

#[test]
fn does_not_split_noun_phrase_conjunction() {
    // "and" joining a noun phrase is NOT a multi-part question.
    let plan = QueryPlan::analyze("recipe with salt and pepper");
    assert_eq!(plan.sub_queries.len(), 1, "{:?}", plan.sub_queries);
}

#[test]
fn simple_query_is_passthrough() {
    let plan = QueryPlan::analyze("harbor bridge maintenance notes");
    assert!(plan.is_passthrough());
    assert_eq!(plan.sub_queries, vec!["harbor bridge maintenance notes"]);
    assert!(plan.temporal.is_none());
}

#[test]
fn temporal_query_is_not_passthrough() {
    let plan = QueryPlan::analyze("what did Alice do in 2021?");
    assert!(!plan.is_passthrough());
}

// ---------------------------------------------------------------------------
// Integration: temporal boost in the hybrid pipeline
// ---------------------------------------------------------------------------

/// Two otherwise-similar memories about Alice, dated 2019 and 2021. Without
/// the temporal boost the composite recency term deterministically favors the
/// NEWER (2021) memory; with query understanding a "2019" question must rank
/// the 2019 memory first, and a "2021" question the 2021 one.
async fn alice_dated_fixture() -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());

    let mut alice_2019 = MemoryEntry::new(
        "alice_2019".into(),
        "alice worked on the bridge project".into(),
        MemoryType::LongTerm,
    );
    alice_2019.metadata.created_at = Utc.with_ymd_and_hms(2019, 6, 1, 0, 0, 0).unwrap();

    // Slight lexical edge (matches the extra query term "what") so the
    // no-boost control deterministically ranks the 2021 memory first.
    let mut alice_2021 = MemoryEntry::new(
        "alice_2021".into(),
        "what alice worked on the tunnel project".into(),
        MemoryType::LongTerm,
    );
    alice_2021.metadata.created_at = Utc.with_ymd_and_hms(2021, 6, 1, 0, 0, 0).unwrap();

    storage.store(&alice_2019).await.unwrap();
    storage.store(&alice_2021).await.unwrap();
    storage
}

fn keyword_hybrid(storage: Arc<MemoryStorage>, config: PipelineConfig) -> HybridRetriever {
    // min_score 0: long natural-language questions spread BM25 mass across
    // many terms, so raw scores are small with a keyword-only pipeline.
    HybridRetriever::new(config.with_min_score(0.0))
        .add_pipeline(Arc::new(KeywordRetriever::new(storage)))
}

#[tokio::test]
async fn temporal_question_boosts_matching_year_memory() {
    let storage = alice_dated_fixture().await;

    // Control: with query understanding disabled, the lexically-stronger
    // 2021 memory ranks first even for a question about 2019.
    let mut off = PipelineConfig::default();
    off.enable_query_understanding = false;
    let results = keyword_hybrid(storage.clone(), off)
        .search("what did alice work on in 2019", 5)
        .await
        .unwrap();
    assert_eq!(
        results[0].entry.key, "alice_2021",
        "without query understanding, the lexically-stronger newer memory must rank first"
    );

    // With query understanding ON (default), the 2019 constraint boosts the
    // 2019-dated memory above the newer one.
    let results = keyword_hybrid(storage.clone(), PipelineConfig::default())
        .search("what did alice work on in 2019", 5)
        .await
        .unwrap();
    assert_eq!(
        results[0].entry.key,
        "alice_2019",
        "temporal constraint must boost the 2019-dated memory: {:?}",
        results.iter().map(|m| &m.entry.key).collect::<Vec<_>>()
    );

    // And the required direction from the plan: a 2021 question ranks the
    // 2021-dated memory above the otherwise-similar 2019 memory.
    let results = keyword_hybrid(storage, PipelineConfig::default())
        .search("what did alice work on in 2021", 5)
        .await
        .unwrap();
    assert_eq!(results[0].entry.key, "alice_2021");
}

// ---------------------------------------------------------------------------
// Integration: multi-part split + union
// ---------------------------------------------------------------------------

/// Fixture where the combined two-part question is dominated by distractors
/// that soak up its many function words, so the un-split query misses at
/// least one of the two answer memories, while the split sub-queries each
/// rank their own answer first.
async fn two_facts_fixture() -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::new());

    let mut alice = MemoryEntry::new(
        "alice_home".into(),
        "alice does live in lisbon near the sea".into(),
        MemoryType::LongTerm,
    );
    alice.metadata.created_at = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();

    let mut bob = MemoryEntry::new(
        "bob_work".into(),
        "bob does work at the observatory".into(),
        MemoryType::LongTerm,
    );
    bob.metadata.created_at = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();

    // Distractors that match the combined query's repeated function words
    // (where x2, does x2, and, live, work) far better than either answer.
    let mut d1 = MemoryEntry::new(
        "distractor_1".into(),
        "where and where does one live and work and does it matter where".into(),
        MemoryType::LongTerm,
    );
    d1.metadata.created_at = Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();
    let mut d2 = MemoryEntry::new(
        "distractor_2".into(),
        "guide on where to live and where to work and where to visit".into(),
        MemoryType::LongTerm,
    );
    d2.metadata.created_at = Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap();

    for e in [&alice, &bob, &d1, &d2] {
        storage.store(e).await.unwrap();
    }
    storage
}

#[tokio::test]
async fn two_part_question_union_surfaces_both_facts() {
    let storage = two_facts_fixture().await;
    let query = "where does alice live and where does bob work";
    let limit = 3;

    // Control: un-split single query (query understanding OFF) misses at
    // least one of the two answers within the limit.
    let mut off = PipelineConfig::default();
    off.enable_query_understanding = false;
    let unsplit = keyword_hybrid(storage.clone(), off)
        .search(query, limit)
        .await
        .unwrap();
    let unsplit_keys: Vec<&str> = unsplit.iter().map(|m| m.entry.key.as_str()).collect();
    let unsplit_hits = ["alice_home", "bob_work"]
        .iter()
        .filter(|k| unsplit_keys.contains(k))
        .count();
    assert!(
        unsplit_hits < 2,
        "control: the un-split query must miss at least one answer, got {unsplit_keys:?}"
    );

    // With query understanding ON, the split sub-queries' union surfaces BOTH.
    let union = keyword_hybrid(storage, PipelineConfig::default())
        .search(query, limit)
        .await
        .unwrap();
    let union_keys: Vec<&str> = union.iter().map(|m| m.entry.key.as_str()).collect();
    assert!(
        union_keys.contains(&"alice_home") && union_keys.contains(&"bob_work"),
        "split + union must surface both answers, got {union_keys:?}"
    );
}

/// Determinism: the same multi-part query yields identical result orderings
/// across repeated fresh retrievers.
#[tokio::test]
async fn query_understanding_is_deterministic() {
    let storage = two_facts_fixture().await;
    let query = "where does alice live and where does bob work";
    let mut config = PipelineConfig::default();
    config.enable_caching = false;

    let mut orderings = Vec::new();
    for _ in 0..3 {
        let results = keyword_hybrid(storage.clone(), config.clone())
            .search(query, 4)
            .await
            .unwrap();
        orderings.push(
            results
                .iter()
                .map(|m| m.entry.key.clone())
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(orderings[0], orderings[1]);
    assert_eq!(orderings[1], orderings[2]);
}

/// Simple queries are a no-op passthrough: with and without query
/// understanding the pipeline returns identical results.
#[tokio::test]
async fn simple_query_behavior_unchanged() {
    let storage = two_facts_fixture().await;
    let query = "observatory";

    let on = keyword_hybrid(storage.clone(), PipelineConfig::default())
        .search(query, 5)
        .await
        .unwrap();
    let mut off_cfg = PipelineConfig::default();
    off_cfg.enable_query_understanding = false;
    let off = keyword_hybrid(storage, off_cfg)
        .search(query, 5)
        .await
        .unwrap();

    let on_keys: Vec<_> = on.iter().map(|m| &m.entry.key).collect();
    let off_keys: Vec<_> = off.iter().map(|m| &m.entry.key).collect();
    assert_eq!(
        on_keys, off_keys,
        "simple query must be a no-op passthrough"
    );
}

/// End-to-end through AgentMemory::search with the default config (query
/// understanding ON): a temporal question surfaces the year-matching memory.
#[tokio::test]
async fn agent_memory_search_applies_query_understanding() {
    use synaptic::{AgentMemory, MemoryConfig};

    let mut config = MemoryConfig::default();
    config.enable_knowledge_graph = false;
    assert!(
        config.enable_query_understanding,
        "query understanding must default to ON"
    );
    let mut memory = AgentMemory::new(config).await.unwrap();

    memory
        .store("alice_2019", "in 2019 alice worked on the bridge project")
        .await
        .unwrap();
    memory
        .store("alice_2021", "in 2021 alice worked on the tunnel project")
        .await
        .unwrap();

    let results = memory
        .search("what did alice work on in 2019", 5)
        .await
        .unwrap();
    assert!(!results.is_empty(), "temporal question must return results");
    assert_eq!(
        results[0].entry.key,
        "alice_2019",
        "content-dated 2019 memory must rank first for a 2019 question: {:?}",
        results.iter().map(|m| &m.entry.key).collect::<Vec<_>>()
    );
}

// Silence unused-import lint until TemporalConstraint is exercised above via
// QueryPlan; the type is part of the public API surface under test.
#[test]
fn temporal_constraint_type_is_exported() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TemporalConstraint>();
    assert_send_sync::<QueryPlan>();
}
