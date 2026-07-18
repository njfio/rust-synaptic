//! Optional LLM-backed implementation of [`MemoryReasoner`] (feature
//! `llm-reasoning`).
//!
//! Prompts an OpenAI-compatible chat-completions endpoint (works with
//! Ollama's `/v1` API) for structured-JSON extraction, conflict resolution,
//! and synthesis. Configuration comes from environment variables:
//!
//! - `SYNAPTIC_LLM_URL` — endpoint base, e.g. `http://localhost:11434/v1`
//!   (falls back to `SYNAPTIC_EVAL_LLM_URL` for consistency with the eval
//!   crate).
//! - `SYNAPTIC_LLM_MODEL` — model name (fallback `SYNAPTIC_EVAL_LLM_MODEL`).
//! - `SYNAPTIC_LLM_KEY` — optional bearer token (fallback
//!   `SYNAPTIC_EVAL_LLM_KEY`; omit for Ollama).
//!
//! FAIL-OPEN TO HEURISTIC: every method first tries the LLM and, on ANY
//! failure — transport error, non-2xx status, unparseable reply, invalid
//! verdict — returns the inner [`HeuristicReasoner`]'s result instead. The
//! write path therefore always receives a real extraction/resolution, never
//! a fabricated one and never a hard failure caused by the LLM. The first
//! fallback is logged at `warn`; subsequent ones at `debug`.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::Deserialize;

use crate::error::Result;
use crate::memory::embeddings::TfIdfProvider;
use crate::memory::reasoning::{
    ConflictResolution, Entity, EntityKind, Extraction, ExtractionContext, Fact, HeuristicReasoner,
    Insight, MemoryReasoner, NeighborFact, Relation,
};
use crate::memory::types::MemoryEntry;

/// Injectable chat call: `(system, user) -> assistant reply text`.
///
/// The production implementation posts to a chat-completions endpoint via
/// reqwest; tests inject canned or failing closures so no network is needed.
pub type ChatCall = Arc<
    dyn Fn(
            String,
            String,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/// LLM-backed [`MemoryReasoner`] with a deterministic heuristic fallback.
pub struct LlmReasoner {
    call: ChatCall,
    inner: HeuristicReasoner,
    warned: AtomicBool,
}

/// Wire format for the extraction reply (lenient: spans are recomputed).
#[derive(Deserialize)]
struct WireExtraction {
    facts: Vec<WireFact>,
}

#[derive(Deserialize)]
struct WireFact {
    text: String,
    #[serde(default)]
    entities: Vec<WireEntity>,
    #[serde(default)]
    relations: Vec<WireRelation>,
}

#[derive(Deserialize)]
struct WireEntity {
    name: String,
    #[serde(default)]
    kind: String,
}

#[derive(Deserialize)]
struct WireRelation {
    // Tolerate weaker models that omit a field: a partial relation should not
    // fail the whole extraction parse. Empty-field relations are dropped after
    // parsing (see `try_extract`).
    #[serde(default)]
    subject: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
}

/// Wire format for the conflict-resolution verdict.
#[derive(Deserialize)]
struct WireVerdict {
    action: String,
    #[serde(default)]
    old_id: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

/// Wire format for the synthesis reply.
#[derive(Deserialize)]
struct WireInsight {
    insight: String,
    #[serde(default = "default_confidence")]
    confidence: f64,
}

fn default_confidence() -> f64 {
    0.5
}

impl LlmReasoner {
    /// Environment variable holding the endpoint base URL.
    pub const ENV_URL: &'static str = "SYNAPTIC_LLM_URL";
    /// Environment variable holding the model name.
    pub const ENV_MODEL: &'static str = "SYNAPTIC_LLM_MODEL";
    /// Environment variable holding the optional bearer token.
    pub const ENV_KEY: &'static str = "SYNAPTIC_LLM_KEY";

    /// Build from environment variables (with `SYNAPTIC_EVAL_LLM_*`
    /// fallbacks). Returns `None` when no endpoint+model is configured —
    /// the caller keeps the plain heuristic reasoner in that case.
    pub fn from_env() -> Option<Self> {
        let env = |primary: &str, fallback: &str| {
            std::env::var(primary)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .or_else(|| {
                    std::env::var(fallback)
                        .ok()
                        .filter(|v| !v.trim().is_empty())
                })
                .map(|v| v.trim().to_string())
        };
        let url = env(Self::ENV_URL, "SYNAPTIC_EVAL_LLM_URL")?;
        let model = match env(Self::ENV_MODEL, "SYNAPTIC_EVAL_LLM_MODEL") {
            Some(m) => m,
            None => {
                tracing::warn!(
                    "{} is set but no model configured ({}); using heuristic reasoner",
                    Self::ENV_URL,
                    Self::ENV_MODEL
                );
                return None;
            }
        };
        let api_key = env(Self::ENV_KEY, "SYNAPTIC_EVAL_LLM_KEY");
        Some(Self::with_call(http_chat_call(url, model, api_key)))
    }

    /// Build with an injected chat call (used by tests and custom transports).
    pub fn with_call(call: ChatCall) -> Self {
        Self {
            call,
            inner: HeuristicReasoner::new(Arc::new(TfIdfProvider::default())),
            warned: AtomicBool::new(false),
        }
    }

    /// Log an LLM failure: `warn` the first time, `debug` afterwards.
    fn log_fallback(&self, method: &str, err: &str) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(method, error = %err, "LLM reasoner failed; falling back to heuristic");
        } else {
            tracing::debug!(method, error = %err, "LLM reasoner failed; falling back to heuristic");
        }
    }

    async fn try_extract(&self, text: &str) -> std::result::Result<Extraction, String> {
        let system = "You extract structured facts from text for a memory system. \
                      Reply with ONLY a JSON object, no prose, of the form: \
                      {\"facts\":[{\"text\":\"<sentence>\",\"entities\":[{\"name\":\"<surface form>\",\
                      \"kind\":\"Person|Place|Org|Date|Number|Quoted|Term\"}],\
                      \"relations\":[{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}]}]}. \
                      Entity names must be exact substrings of the fact text. \
                      Use snake_case predicates (e.g. lives_in, works_at).";
        let user = format!("Extract facts from this text:\n{text}");
        let reply = (self.call)(system.to_string(), user).await?;
        let wire: WireExtraction = serde_json::from_str(strip_fences(&reply))
            .map_err(|e| format!("unparseable extraction reply: {e}"))?;
        let facts = wire
            .facts
            .into_iter()
            .map(|f| {
                let entities = f
                    .entities
                    .into_iter()
                    .map(|e| {
                        let span = f
                            .text
                            .find(&e.name)
                            .map(|s| (s, s + e.name.len()))
                            .unwrap_or((0, 0));
                        Entity {
                            span,
                            kind: parse_kind(&e.kind),
                            name: e.name,
                        }
                    })
                    .collect();
                let relations = f
                    .relations
                    .into_iter()
                    // Drop partial relations (a field the model omitted defaults
                    // to empty) — an incomplete triple is not a usable relation.
                    .filter(|r| {
                        !r.subject.is_empty() && !r.predicate.is_empty() && !r.object.is_empty()
                    })
                    .map(|r| Relation {
                        subject: r.subject,
                        predicate: r.predicate,
                        object: r.object,
                    })
                    .collect();
                Fact {
                    text: f.text,
                    entities,
                    relations,
                }
            })
            .collect();
        Ok(Extraction { facts })
    }

    async fn try_resolve(
        &self,
        candidate: &Fact,
        neighbors: &[NeighborFact],
    ) -> std::result::Result<ConflictResolution, String> {
        let system = "You decide how a new fact interacts with existing memories. \
                      Reply with ONLY a JSON object, no prose: \
                      {\"action\":\"insert|update_in_place|supersede|append|noop\",\
                      \"old_id\":\"<id of the superseded memory, required for supersede>\",\
                      \"reason\":\"<short reason>\"}. \
                      Use supersede when the new fact replaces an outdated memory, \
                      noop for duplicates, update_in_place for refinements of the \
                      same memory, append to add detail, insert otherwise.";
        let neighbor_list = neighbors
            .iter()
            .map(|n| {
                format!(
                    "- id={} similarity={:.2} text={}",
                    n.id, n.similarity, n.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!(
            "New fact: {}\n\nExisting similar memories:\n{}",
            candidate.text,
            if neighbor_list.is_empty() {
                "(none)".to_string()
            } else {
                neighbor_list
            }
        );
        let reply = (self.call)(system.to_string(), user).await?;
        let wire: WireVerdict = serde_json::from_str(strip_fences(&reply))
            .map_err(|e| format!("unparseable resolve reply: {e}"))?;
        let reason = wire.reason.unwrap_or_else(|| "llm verdict".to_string());
        match wire.action.trim().to_ascii_lowercase().as_str() {
            "insert" => Ok(ConflictResolution::Insert),
            "update_in_place" => Ok(ConflictResolution::UpdateInPlace { reason }),
            "append" => Ok(ConflictResolution::Append { reason }),
            "noop" | "no_op" => Ok(ConflictResolution::NoOp { reason }),
            "supersede" => {
                let old_id = wire
                    .old_id
                    .ok_or_else(|| "supersede verdict missing old_id".to_string())?;
                if !neighbors.iter().any(|n| n.id == old_id) {
                    return Err(format!(
                        "supersede verdict targets unknown neighbor id {old_id}"
                    ));
                }
                Ok(ConflictResolution::Supersede { old_id, reason })
            }
            other => Err(format!("unknown resolve action {other:?}")),
        }
    }

    async fn try_synthesize(
        &self,
        cluster: &[MemoryEntry],
    ) -> std::result::Result<Option<Insight>, String> {
        if cluster.len() < 2 {
            return Ok(None);
        }
        let system = "You synthesize one insight from a cluster of related memories. \
                      Reply with ONLY a JSON object, no prose: \
                      {\"insight\":\"<one-sentence insight>\",\"confidence\":<0.0-1.0>}.";
        let memories = cluster
            .iter()
            .enumerate()
            .map(|(i, e)| format!("[{}] {}", i + 1, e.value))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("Related memories:\n{memories}");
        let reply = (self.call)(system.to_string(), user).await?;
        let wire: WireInsight = serde_json::from_str(strip_fences(&reply))
            .map_err(|e| format!("unparseable synthesize reply: {e}"))?;
        Ok(Some(Insight {
            text: wire.insight,
            derived_from: cluster.iter().map(|e| e.id().to_string()).collect(),
            confidence: wire.confidence.clamp(0.0, 1.0),
        }))
    }
}

#[async_trait::async_trait]
impl MemoryReasoner for LlmReasoner {
    async fn extract(&self, text: &str, ctx: &ExtractionContext) -> Result<Extraction> {
        match self.try_extract(text).await {
            Ok(extraction) => Ok(extraction),
            Err(e) => {
                self.log_fallback("extract", &e);
                self.inner.extract(text, ctx).await
            }
        }
    }

    async fn resolve(
        &self,
        candidate: &Fact,
        neighbors: &[NeighborFact],
    ) -> Result<ConflictResolution> {
        match self.try_resolve(candidate, neighbors).await {
            Ok(resolution) => Ok(resolution),
            Err(e) => {
                self.log_fallback("resolve", &e);
                self.inner.resolve(candidate, neighbors).await
            }
        }
    }

    async fn synthesize(&self, cluster: &[MemoryEntry]) -> Result<Option<Insight>> {
        match self.try_synthesize(cluster).await {
            Ok(insight) => Ok(insight),
            Err(e) => {
                self.log_fallback("synthesize", &e);
                self.inner.synthesize(cluster).await
            }
        }
    }

    fn name(&self) -> &str {
        "llm"
    }
}

/// Map a wire entity kind onto [`EntityKind`]; unknown kinds become `Term`.
fn parse_kind(kind: &str) -> EntityKind {
    match kind.trim().to_ascii_lowercase().as_str() {
        "person" => EntityKind::Person,
        "place" => EntityKind::Place,
        "org" | "organization" => EntityKind::Org,
        "date" => EntityKind::Date,
        "number" => EntityKind::Number,
        "quoted" | "quote" => EntityKind::Quoted,
        _ => EntityKind::Term,
    }
}

/// Strip a Markdown code fence (```json ... ```) around a reply, if present.
fn strip_fences(reply: &str) -> &str {
    let trimmed = reply.trim();
    let Some(inner) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let inner = inner.strip_suffix("```").unwrap_or(inner);
    // Drop an optional language tag on the first line (e.g. "json").
    match inner.find('\n') {
        Some(nl) if inner[..nl].chars().all(|c| c.is_ascii_alphanumeric()) => inner[nl..].trim(),
        _ => inner.trim(),
    }
}

/// Production chat call: POST to `{url}/chat/completions` (OpenAI-compatible,
/// including Ollama's `/v1`) and return `choices[0].message.content`.
fn http_chat_call(url: String, model: String, api_key: Option<String>) -> ChatCall {
    let client = reqwest::Client::new();
    let base = url.trim_end_matches('/').to_string();
    let endpoint = if base.ends_with("/chat/completions") {
        base
    } else {
        format!("{base}/chat/completions")
    };
    Arc::new(move |system: String, user: String| {
        let client = client.clone();
        let endpoint = endpoint.clone();
        let model = model.clone();
        let api_key = api_key.clone();
        Box::pin(async move {
            let body = serde_json::json!({
                "model": model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            });
            let mut req = client.post(&endpoint).json(&body);
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("request to LLM endpoint failed: {e}"))?;
            let status = resp.status();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("LLM endpoint returned {status}: {text}"));
            }
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| format!("LLM endpoint returned invalid JSON: {e}"))?;
            json.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .map(|s| s.trim().to_string())
                .ok_or_else(|| "LLM reply missing choices[0].message.content".to_string())
        })
    })
}

#[cfg(test)]
mod tolerant_parse_tests {
    use super::*;

    fn canned(reply: &'static str) -> ChatCall {
        Arc::new(move |_s: String, _u: String| {
            Box::pin(async move { Ok(reply.to_string()) })
        })
    }

    /// A weaker model can emit a relation missing a field. That must not fail
    /// the whole extraction (previously: "missing field `object`"); the fact
    /// still parses and the incomplete relation is dropped.
    #[tokio::test]
    async fn extract_tolerates_relation_missing_object() {
        let reply = r#"{"facts":[{"text":"Alice lives in Berlin.","entities":[{"name":"Alice","kind":"Person"}],"relations":[{"subject":"Alice","predicate":"lives_in"}]}]}"#;
        let r = LlmReasoner::with_call(canned(reply));
        let ctx = ExtractionContext {
            source_key: "k".into(),
            timestamp: chrono::Utc::now(),
        };
        let ex = r
            .extract("Alice lives in Berlin.", &ctx)
            .await
            .expect("extraction should succeed");
        assert_eq!(ex.facts.len(), 1, "the fact must parse despite partial relation");
        assert_eq!(ex.facts[0].text, "Alice lives in Berlin.");
        assert!(
            ex.facts[0].relations.is_empty(),
            "incomplete relation (no object) must be dropped, not kept empty"
        );
    }
}
