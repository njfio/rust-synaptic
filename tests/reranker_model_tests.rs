//! Tests for the optional candle cross-encoder reranker (feature `reranker-model`).
//!
//! These tests exercise the REAL inference path (tokenize (query, candidate)
//! pairs, BERT forward pass, classifier logit, re-order) without any network
//! access: the model is built from a tiny `BertConfig` and a deterministic
//! pseudo-random `VarBuilder` backend, and the tokenizer is an in-memory
//! word-level tokenizer. No weights are downloaded.

#![cfg(feature = "reranker-model")]
// Test code: unwrap/panic on failure is the intended behaviour.
#![allow(clippy::unwrap_used, clippy::panic)]

use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{Config as BertConfig, HiddenAct, PositionEmbeddingType};
use std::collections::HashMap;
use synaptic::memory::retrieval::{CrossEncoderReranker, Reranker, RetrievalSignal, ScoredMemory};
use synaptic::memory::types::{MemoryEntry, MemoryFragment, MemoryType};
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
            // Map the top bits to roughly [-0.1, 0.1).
            let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            vals.push(unit * 0.2);
        }
        Tensor::from_vec(vals, s, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn tiny_bert_config() -> BertConfig {
    BertConfig {
        vocab_size: 32,
        hidden_size: 32,
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

fn build_reranker() -> CrossEncoderReranker {
    let vb = VarBuilder::from_backend(Box::new(DeterministicBackend), DType::F32, Device::Cpu);
    CrossEncoderReranker::from_var_builder(vb, &tiny_bert_config(), tiny_tokenizer(), 64)
        .expect("model built from in-memory deterministic weights must load")
}

fn candidate(key: &str, content: &str, score: f64) -> ScoredMemory {
    let entry = MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::LongTerm);
    ScoredMemory::new(
        MemoryFragment::new(entry, score),
        score,
        RetrievalSignal::Hybrid,
    )
}

/// Loading from a directory with no model files must fail closed — never a
/// fake reranker.
#[test]
fn new_fails_closed_when_model_files_are_missing() {
    let dir = std::env::temp_dir().join("synaptic-reranker-model-missing");
    std::fs::create_dir_all(&dir).unwrap();
    let result = CrossEncoderReranker::new(&dir);
    assert!(
        result.is_err(),
        "constructing from a directory without model files must return Err"
    );
}

/// Real inference: the reranker's ordering must be deterministic across runs
/// and monotonic in the model's own pair logits, with candidates and their
/// original scores preserved.
#[tokio::test]
async fn rerank_is_deterministic_and_monotonic_in_model_logits() {
    let reranker = build_reranker();
    let query = "database connection pooling config";

    let a = candidate("a", "postgres connection pooling tuning guide", 0.2);
    let b = candidate("b", "summer picnic volleyball sheet", 0.9);

    // The model's own relevance logits for each (query, candidate) pair.
    let logit_a = reranker
        .score(query, "postgres connection pooling tuning guide")
        .unwrap();
    let logit_b = reranker
        .score(query, "summer picnic volleyball sheet")
        .unwrap();
    assert!(
        (logit_a - logit_b).abs() > f32::EPSILON,
        "distinct inputs through real inference must produce distinct logits"
    );

    let run1 = reranker
        .rerank(query, vec![a.clone(), b.clone()])
        .await
        .unwrap();
    let run2 = reranker.rerank(query, vec![a, b]).await.unwrap();

    // Deterministic: identical order on both runs.
    let order1: Vec<&str> = run1.iter().map(|c| c.memory.entry.key.as_str()).collect();
    let order2: Vec<&str> = run2.iter().map(|c| c.memory.entry.key.as_str()).collect();
    assert_eq!(order1, order2, "reranking must be deterministic");

    // Same candidates, no additions or drops; original scores preserved.
    assert_eq!(run1.len(), 2);
    let mut keys = order1.clone();
    keys.sort_unstable();
    assert_eq!(keys, vec!["a", "b"]);
    for c in &run1 {
        let expected = if c.memory.entry.key == "a" { 0.2 } else { 0.9 };
        assert!(
            (c.score - expected).abs() < f64::EPSILON,
            "original pipeline scores must travel through unchanged"
        );
    }

    // Monotonic in the model's logits: higher logit ranks first.
    let expected_first = if logit_a > logit_b { "a" } else { "b" };
    assert_eq!(
        order1[0], expected_first,
        "ordering must follow the model's relevance logits, descending"
    );

    // Scoring the same pair twice is itself deterministic.
    let logit_a_again = reranker
        .score(query, "postgres connection pooling tuning guide")
        .unwrap();
    assert!((logit_a - logit_a_again).abs() < 1e-6);
}

/// Fewer than two candidates: nothing to re-order; input passes through.
#[tokio::test]
async fn rerank_passes_through_single_candidate() {
    let reranker = build_reranker();
    let only = candidate("solo", "postgres connection pooling", 0.5);
    let out = reranker.rerank("pooling", vec![only]).await.unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].memory.entry.key, "solo");
    assert_eq!(reranker.name(), "CrossEncoderReranker");
}
