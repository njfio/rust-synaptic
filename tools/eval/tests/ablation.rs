//! Ablation-harness tests: (a) every config really runs on the fixture and
//! produces distinct, comparable metric records with computed deltas;
//! (b) a crafted case where enabling the reranker changes the ranking (and
//! thus a metric) between the `+composite` and `+reranker` configs.

use synaptic_eval::ablation::{configs, run_ablation, run_config, AblationTable};
use synaptic_eval::dataset::{
    load_longmemeval_str, EvalConversation, EvalQuestion, QType, Session, Turn,
};

fn fixture_section(key: &str) -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/sample_conversations.json"
    );
    let raw = std::fs::read_to_string(path).expect("fixture must exist");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture must be valid JSON");
    serde_json::to_string(&value[key]).expect("fixture section must serialize")
}

// ---------------------------------------------------------------------------
// (a) all configs run on the fixture and yield comparable records + deltas
// ---------------------------------------------------------------------------
#[tokio::test]
async fn ablation_runs_every_config_and_computes_real_deltas() {
    let convs = load_longmemeval_str(&fixture_section("longmemeval")).expect("fixture must parse");
    let table = run_ablation(&convs, 5).await.expect("ablation must run");

    // Baseline plus one row per non-baseline config, in declaration order.
    let names: Vec<&str> = configs().iter().map(|c| c.name).collect();
    assert_eq!(names[0], "baseline");
    assert_eq!(table.baseline.name, "baseline");
    let row_names: Vec<&str> = table.rows.iter().map(|r| r.config.name.as_str()).collect();
    assert_eq!(row_names, &names[1..]);
    assert_eq!(
        row_names,
        vec!["+composite", "+reranker", "+graph_temporal", "+all"]
    );

    // Every config really ran: same question set, real per-question rows.
    let expected_questions: usize = convs.iter().map(|c| c.questions.len()).sum();
    assert_eq!(table.baseline.questions_evaluated, expected_questions);
    for row in &table.rows {
        assert_eq!(row.config.questions_evaluated, expected_questions);
        assert_eq!(row.config.per_question.len(), expected_questions);
        // Deltas are exactly (config - baseline) — computed, never fabricated.
        assert!(
            (row.delta_vs_baseline.mrr - (row.config.mrr - table.baseline.mrr)).abs() < 1e-12,
            "MRR delta must equal config minus baseline"
        );
        assert!(
            (row.delta_vs_baseline.recall_at_k
                - (row.config.mean_recall_at_k - table.baseline.mean_recall_at_k))
                .abs()
                < 1e-12
        );
        assert!(
            (row.delta_vs_baseline.precision_at_k
                - (row.config.mean_precision_at_k - table.baseline.mean_precision_at_k))
                .abs()
                < 1e-12
        );
        // Documented composition, and metrics are within valid metric range.
        assert!(!row.config.includes.is_empty());
        assert!((0.0..=1.0).contains(&row.config.mrr));
    }

    // The table serializes (it is the artifact `docs/evaluation.md` cites).
    let json = serde_json::to_string(&table).expect("table must serialize");
    let round: AblationTable = serde_json::from_str(&json).expect("table must deserialize");
    assert_eq!(round.rows.len(), table.rows.len());
    assert_eq!(round.k, 5);
}

// ---------------------------------------------------------------------------
// (b) crafted case: the reranker changes the ranking vs the same pipeline
//     without it, and the ablation captures the difference in a metric
// ---------------------------------------------------------------------------

/// A conversation crafted to expose the one structural asymmetry between the
/// composite-scored pipeline and the reranking stage: the reranker reorders
/// the top-K purely by query-focused cross-features (term overlap, embedding
/// agreement, recency with a one-week half-life) and ignores the fused
/// relevance score entirely, whereas composite scoring blends relevance at
/// weight 0.6 against recency at only 0.2.
///
/// The evidence turn is an exact query match but 600 hours old; the
/// distractor is a fresh paraphrase containing every query term plus two
/// noise words. The evidence's fused-relevance lead is large enough that
/// composite scoring keeps it on top (rank 1 in `baseline` and
/// `+composite`), but the relevance-blind reranker — equal term overlap,
/// slightly lower embedding agreement, decisively fresher — promotes the
/// distractor (evidence drops to rank 2 in `+reranker`).
fn crafted_reranker_conversation() -> EvalConversation {
    let now = chrono::Utc::now();
    let old = (now - chrono::Duration::hours(600)).to_rfc3339();
    let fresh = now.to_rfc3339();
    let turn = |text: &str, ts: &str| Turn {
        speaker: "user".to_string(),
        text: text.to_string(),
        timestamp: Some(ts.to_string()),
    };
    EvalConversation {
        id: "crafted_rerank".to_string(),
        sessions: vec![Session {
            id: "session_1".to_string(),
            timestamp: None,
            turns: vec![
                // D1:1 — fresh paraphrase distractor: full term overlap plus
                // two noise words (so it never ties the evidence turn's
                // dense score, keeping the fused ordering deterministic).
                turn("zephyr quill marmot dirigible spotted again", &fresh),
                // D1:2 — evidence: exact query match, 600 hours old.
                turn("zephyr quill marmot dirigible", &old),
                // D1:3 — anchor sharing three query terms, old.
                turn("quill marmot dirigible", &old),
            ],
        }],
        questions: vec![EvalQuestion {
            id: "crafted_q1".to_string(),
            text: "zephyr quill marmot dirigible".to_string(),
            qtype: QType::SingleHop,
            evidence_ids: vec!["D1:2".to_string()],
            gold_answer: "n/a".to_string(),
        }],
    }
}

#[tokio::test]
async fn reranker_config_changes_ranking_on_crafted_case() {
    let convs = vec![crafted_reranker_conversation()];
    let all = configs();
    let composite_cfg = all
        .iter()
        .find(|c| c.name == "+composite")
        .expect("+composite config exists");
    let reranker_cfg = all
        .iter()
        .find(|c| c.name == "+reranker")
        .expect("+reranker config exists");

    let without = run_config(composite_cfg, &convs, 5)
        .await
        .expect("+composite must run");
    let with = run_config(reranker_cfg, &convs, 5)
        .await
        .expect("+reranker must run");

    let rr_without = without.per_question[0].reciprocal_rank;
    let rr_with = with.per_question[0].reciprocal_rank;

    // Both configs retrieved the evidence somewhere (this is a retrieval
    // ranking test, not a recall test).
    assert!(
        rr_without > 0.0,
        "evidence must be retrieved without reranker"
    );
    assert!(rr_with > 0.0, "evidence must be retrieved with reranker");

    // The reranker really changes the ranking of the evidence turn, which
    // shows up as a different reciprocal rank. No fixed magnitude asserted —
    // only that the ablation captures a real, measured difference.
    assert_ne!(
        rr_with, rr_without,
        "enabling the reranker must change the evidence rank on the crafted case \
         (with={rr_with}, without={rr_without})"
    );
    // Direction on THIS crafted case (measured, not assumed): the
    // relevance-blind reranker prefers the fresh full-overlap paraphrase and
    // demotes the stale exact-match evidence to rank 2, while composite
    // scoring's 0.6 relevance weight had kept the evidence on top. The
    // ablation reports the honest (here: negative) delta.
    assert!(
        rr_with < rr_without,
        "reranker recency must promote the fresh paraphrase over the stale \
         evidence on this crafted case (with={rr_with}, without={rr_without})"
    );
}
