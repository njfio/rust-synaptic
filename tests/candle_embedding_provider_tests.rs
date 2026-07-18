//! Tests for the candle BERT embedding provider (feature `ml-models`).
//!
//! These tests exercise the REAL inference path (tokenize, BERT forward pass,
//! mean pooling) without any network access: the model is built from a tiny
//! `BertConfig` and a deterministic pseudo-random `VarBuilder` backend, and
//! the tokenizer is an in-memory word-level tokenizer. No weights are
//! downloaded.

#![cfg(feature = "ml-models")]
// Test code: unwrap/panic/diagnostic stdout are legitimate here.
#![allow(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]

use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{Config as BertConfig, HiddenAct, PositionEmbeddingType};
use std::collections::HashMap;
use std::sync::Arc;
use synaptic::memory::embeddings::provider::EmbeddingProvider;
use synaptic::memory::embeddings::providers::CandleEmbeddingProvider;
use synaptic::memory::embeddings::MultiProvider;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

/// A `VarBuilder` backend that fabricates every requested tensor from a
/// deterministic pseudo-random stream seeded by the tensor's name, so the
/// same names + shapes always produce the same (non-trivial) weights.
struct DeterministicBackend;

impl SimpleBackend for DeterministicBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let n = s.elem_count();
        let mut state: u64 = name.bytes().fold(0x9E37_79B9_7F4A_7C15u64, |acc, b| {
            acc.rotate_left(7) ^ (b as u64).wrapping_mul(0x0100_0000_01B3)
        });
        let mut vals = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            vals.push(unit * 0.2);
        }
        Tensor::from_vec(vals, s, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

const HIDDEN: usize = 32;

fn tiny_bert_config() -> BertConfig {
    BertConfig {
        vocab_size: 32,
        hidden_size: HIDDEN,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        intermediate_size: 64,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.0,
        max_position_embeddings: 64,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: false,
        classifier_dropout: None,
        model_type: Some("bert".to_string()),
    }
}

fn tiny_tokenizer() -> Tokenizer {
    let words = [
        "[UNK]",
        "database",
        "connection",
        "pooling",
        "config",
        "picnic",
        "volleyball",
        "summer",
        "sheet",
        "postgres",
        "tuning",
        "guide",
        "the",
        "for",
        "a",
        "of",
    ];
    let vocab: HashMap<String, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Whitespace {});
    tokenizer
}

fn build_provider() -> CandleEmbeddingProvider {
    let vb = VarBuilder::from_backend(Box::new(DeterministicBackend), DType::F32, Device::Cpu);
    CandleEmbeddingProvider::from_var_builder(vb, &tiny_bert_config(), tiny_tokenizer(), 64)
        .expect("model built from in-memory deterministic weights must load")
}

/// Loading from a directory with no model files must fail closed — never a
/// fake provider.
#[tokio::test]
async fn new_fails_closed_when_model_files_are_missing() {
    let dir = std::env::temp_dir().join("synaptic-candle-embed-missing");
    std::fs::create_dir_all(&dir).unwrap();
    let result = CandleEmbeddingProvider::new(&dir);
    assert!(
        result.is_err(),
        "constructing from a directory without model files must return Err"
    );
}

/// Real inference: correct dimension, deterministic output, non-trivial
/// (non-zero) vector, distinct texts produce distinct vectors.
#[tokio::test]
async fn embed_returns_deterministic_fixed_dimension_vector() {
    let provider = build_provider();

    assert_eq!(provider.embedding_dimension(), HIDDEN);
    assert_eq!(provider.name(), "CandleEmbeddingProvider");
    assert!(provider.is_available());
    assert!(!provider.model_id().is_empty());

    let a1 = provider
        .embed("database connection pooling config", None)
        .await
        .unwrap();
    let a2 = provider
        .embed("database connection pooling config", None)
        .await
        .unwrap();
    let b = provider
        .embed("summer picnic volleyball sheet", None)
        .await
        .unwrap();

    assert_eq!(a1.vector.len(), HIDDEN);
    assert_eq!(b.vector.len(), HIDDEN);

    // Deterministic: same text -> same vector.
    for (x, y) in a1.vector.iter().zip(a2.vector.iter()) {
        assert!((x - y).abs() < 1e-6, "same text must embed identically");
    }

    // Non-trivial: not the zero vector.
    assert!(a1.vector.iter().any(|v| v.abs() > f32::EPSILON));

    // Distinct texts through real inference produce distinct vectors.
    let differs = a1
        .vector
        .iter()
        .zip(b.vector.iter())
        .any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(differs, "distinct texts must produce distinct embeddings");
}

/// Batch embedding matches single embedding.
#[tokio::test]
async fn embed_batch_matches_single_embeds() {
    let provider = build_provider();
    let texts = vec![
        "postgres tuning guide".to_string(),
        "summer picnic".to_string(),
    ];
    let batch = provider.embed_batch(&texts, None).await.unwrap();
    assert_eq!(batch.len(), 2);
    let single = provider.embed(&texts[0], None).await.unwrap();
    for (x, y) in batch[0].vector.iter().zip(single.vector.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}

/// The candle provider is a first-class peer: selectable through
/// MultiProvider as the primary provider.
#[tokio::test]
async fn selectable_through_multi_provider() {
    let provider: Arc<dyn EmbeddingProvider> = Arc::new(build_provider());
    let multi = MultiProvider::new(Arc::clone(&provider));

    assert!(multi.is_available());
    assert_eq!(multi.embedding_dimension(), HIDDEN);

    let embedding = multi
        .embed("database connection pooling", None)
        .await
        .unwrap();
    assert_eq!(embedding.vector.len(), HIDDEN);

    // The embedding produced through MultiProvider is the candle model's own.
    let direct = provider
        .embed("database connection pooling", None)
        .await
        .unwrap();
    for (x, y) in embedding.vector.iter().zip(direct.vector.iter()) {
        assert!((x - y).abs() < 1e-6);
    }
}

/// A missing model directory selected via `RetrievalEmbeddingConfig::Candle`
/// must fall back to TF-IDF (no panic, no error, still embeds).
#[tokio::test]
async fn missing_model_dir_falls_back_to_tfidf() {
    use synaptic::memory::embeddings::{build_retrieval_provider, RetrievalEmbeddingConfig};

    let config = RetrievalEmbeddingConfig::Candle {
        model_dir: std::path::PathBuf::from("/nonexistent/synaptic-model-dir"),
        dimension: 384,
    };
    let (provider, _tfidf) = build_retrieval_provider(&config);
    let embedding = provider
        .embed("fallback still embeds via tfidf", None)
        .await
        .expect("fallback provider must embed without error");
    assert!(!embedding.vector.is_empty());
}

/// THE proof gate for real MiniLM inference: with the fetched
/// all-MiniLM-L6-v2 weights (run `scripts/fetch_embedding_model.sh` first),
/// semantically related sentences must be markedly more similar than
/// unrelated ones. Random or wrongly-mapped weights cannot pass this.
///
/// Ignored by default because it needs the ~90MB downloaded model:
///   cargo test --features ml-models --test candle_embedding_provider_tests -- --ignored
#[tokio::test]
#[ignore = "requires models/all-MiniLM-L6-v2 fetched via scripts/fetch_embedding_model.sh"]
async fn real_minilm_related_sentences_beat_unrelated() {
    let model_dir = std::env::var("SYNAPTIC_EMBED_MODEL_DIR")
        .unwrap_or_else(|_| "models/all-MiniLM-L6-v2".to_string());
    let provider = CandleEmbeddingProvider::new(std::path::Path::new(&model_dir))
        .expect("fetched all-MiniLM-L6-v2 model must load");
    assert_eq!(provider.embedding_dimension(), 384);

    let anchor = provider
        .embed("The cat sat quietly on the warm windowsill.", None)
        .await
        .unwrap();
    let related = provider
        .embed("A kitten was resting peacefully by the sunny window.", None)
        .await
        .unwrap();
    let unrelated = provider
        .embed(
            "Quarterly revenue projections exceeded analyst estimates.",
            None,
        )
        .await
        .unwrap();

    // Mean-pooled output is L2-normalized: dot product == cosine.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    let norm: f32 = anchor.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "embedding must be L2-normalized");

    let sim_related = cosine(&anchor.vector, &related.vector);
    let sim_unrelated = cosine(&anchor.vector, &unrelated.vector);
    println!("cosine(related) = {sim_related}, cosine(unrelated) = {sim_unrelated}");

    assert!(
        sim_related > sim_unrelated + 0.2,
        "real semantic inference must separate related ({sim_related}) from unrelated ({sim_unrelated}) by a clear margin"
    );
    assert!(
        sim_related > 0.5,
        "related sentences should be strongly similar, got {sim_related}"
    );
}

/// Empty tokenization fails with an error rather than a fabricated vector.
#[tokio::test]
async fn embed_empty_text_fails_closed() {
    let provider = build_provider();
    let result = provider.embed("", None).await;
    assert!(
        result.is_err(),
        "no tokens must be an error, not a fake vector"
    );
}

/// Padded-batch identity: `embed_for_scoring_batch` over texts of DIFFERENT
/// token lengths must return, for each text, the same vector as embedding
/// that text alone — proving the attention mask zeroes padding out of both
/// the forward pass and the mean-pool.
#[tokio::test]
async fn scoring_batch_matches_individual_embeds_despite_padding() {
    let provider = build_provider();
    // Deliberately different lengths so shorter texts are padded to the
    // batch max in the single batched forward pass.
    let texts = [
        "database",
        "postgres tuning guide for connection pooling",
        "summer picnic volleyball",
    ];

    let batched = provider.embed_for_scoring_batch(&texts).await.unwrap();
    assert_eq!(batched.len(), texts.len());

    for (i, text) in texts.iter().enumerate() {
        let single = provider.embed(text, None).await.unwrap();
        assert_eq!(batched[i].vector.len(), HIDDEN);
        for (d, (x, y)) in batched[i]
            .vector
            .iter()
            .zip(single.vector.iter())
            .enumerate()
        {
            assert!(
                (x - y).abs() < 1e-5,
                "text {} dim {}: padded-batch {} vs solo {} — padding leaked into the embedding",
                i,
                d,
                x,
                y
            );
        }
    }
}

/// The scoring cache serves repeats without changing results, and mixed
/// hit/miss batches stay correct (only misses are recomputed).
#[tokio::test]
async fn scoring_batch_cache_hits_return_identical_vectors() {
    let provider = build_provider();
    let first = provider
        .embed_for_scoring_batch(&["postgres tuning guide", "summer picnic"])
        .await
        .unwrap();
    // Second batch: one cached text, one new text.
    let second = provider
        .embed_for_scoring_batch(&["postgres tuning guide", "database connection"])
        .await
        .unwrap();
    for (x, y) in first[0].vector.iter().zip(second[0].vector.iter()) {
        assert!((x - y).abs() < 1e-6, "cache hit must be byte-stable");
    }
    let solo = provider.embed("database connection", None).await.unwrap();
    for (x, y) in second[1].vector.iter().zip(solo.vector.iter()) {
        assert!(
            (x - y).abs() < 1e-5,
            "mixed-batch miss must match solo embed"
        );
    }
}
