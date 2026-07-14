//! Tests for the model2vec static-embedding provider (`static-embeddings`).
//!
//! The semantic-quality test needs the potion-base-8M model fetched locally
//! (`scripts/fetch_embedding_model.sh --potion`), so it is `#[ignore]` by
//! default; run it with:
//!   cargo test --features static-embeddings --test static_embedding_tests -- --ignored
//! The fallback test runs without any download.

#![cfg(feature = "static-embeddings")]

use std::path::{Path, PathBuf};
use synaptic::memory::embeddings::{
    build_retrieval_provider, providers::StaticEmbeddingProvider, EmbeddingProvider,
    RetrievalEmbeddingConfig,
};

fn potion_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/potion-base-8M")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// With the real potion-base-8M table: related sentences must be clearly
/// closer than unrelated ones — proving the lookups/pooling produce genuine
/// semantic embeddings, not noise.
#[tokio::test]
#[ignore = "requires models/potion-base-8M (run scripts/fetch_embedding_model.sh --potion)"]
async fn static_embeddings_are_semantic() {
    let provider = StaticEmbeddingProvider::new(&potion_dir())
        .expect("potion-base-8M must load from models/potion-base-8M");
    assert_eq!(provider.embedding_dimension(), 256);
    assert!(provider.is_available());

    let a = provider
        .embed("The cat sat quietly on the warm windowsill.", None)
        .await
        .expect("embed must succeed on a loaded model");
    let b = provider
        .embed("A kitten was resting near the sunny window.", None)
        .await
        .expect("embed must succeed on a loaded model");
    let c = provider
        .embed("Quarterly corporate tax filings are due in April.", None)
        .await
        .expect("embed must succeed on a loaded model");

    // L2-normalized output.
    for e in [&a, &b, &c] {
        let norm: f32 = e.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "embedding must be L2-normalized");
    }

    let related = cosine(&a.vector, &b.vector);
    let unrelated = cosine(&a.vector, &c.vector);
    println!("related cosine = {related:.4}, unrelated cosine = {unrelated:.4}");
    assert!(
        related > unrelated + 0.15,
        "semantic separation too small: related={related:.4} unrelated={unrelated:.4}"
    );

    // Batch path agrees with the single path.
    let batch = provider
        .embed_for_scoring_batch(&["The cat sat quietly on the warm windowsill."])
        .await
        .expect("batch embed must succeed");
    let same = cosine(&a.vector, &batch[0].vector);
    assert!(same > 0.9999, "batch/single drift: cosine={same}");
}

/// Missing model dir: construction must fail closed (error, no panic) …
#[test]
fn missing_model_dir_fails_closed() {
    let err = StaticEmbeddingProvider::new(Path::new("models/does-not-exist"));
    assert!(
        err.is_err(),
        "missing model dir must be a construction error"
    );
}

/// … and the retrieval pipeline must degrade to TF-IDF, never hard-fail.
#[tokio::test]
async fn missing_model_dir_falls_back_to_tfidf() {
    let config = RetrievalEmbeddingConfig::Static {
        model_dir: PathBuf::from("models/does-not-exist"),
        dimension: 256,
    };
    let (provider, tfidf) = build_retrieval_provider(&config);
    // Fallback is the TF-IDF provider itself.
    assert_eq!(provider.name(), tfidf.name());
    let embedding = provider
        .embed("fallback still embeds", None)
        .await
        .expect("TF-IDF fallback must embed");
    assert!(!embedding.vector.is_empty());
}
