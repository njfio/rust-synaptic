//! Fallback wrapper around a semantic embedding provider.
//!
//! Wraps a configured semantic provider (e.g. Ollama `nomic-embed-text`) so
//! that if it is unreachable or fails at first use, the pipeline falls back
//! to the offline [`TfIdfProvider`] with a `tracing::warn` — the retrieval
//! pipeline NEVER hard-fails and never fabricates embeddings.
//!
//! Semantics:
//! - While the primary is healthy, every document-path [`embed`] call ALSO
//!   feeds the text into the TF-IDF fallback, so a later fail-over scores
//!   against a live corpus (real IDF) instead of an empty vocabulary.
//! - The first primary error marks the primary failed (sticky, per-instance)
//!   and all subsequent calls are served by the TF-IDF fallback.
//! - Real semantic providers ignore the TF-IDF document-vs-scoring (IDF)
//!   distinction: their `embed_for_scoring` delegates to `embed`, which is
//!   already read-only (no corpus-relative statistics to mutate).
//!
//! Note on dimensions: primary and fallback vector spaces differ (e.g. 768
//! vs hashed TF-IDF). A fail-over mid-corpus means embeddings created before
//! the failure do not compare against fallback vectors; cosine similarity on
//! mismatched dimensions is defined as 0.0, so ranking degrades gracefully
//! rather than erroring. The retrieval pipeline re-embeds candidates at query
//! time, so post-fail-over queries are fully consistent.
//!
//! [`embed`]: crate::memory::embeddings::provider::EmbeddingProvider::embed

use super::super::provider::{EmbedOptions, Embedding, EmbeddingProvider, ProviderCapabilities};
use super::tfidf::TfIdfProvider;
use crate::error::Result;
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Wraps a primary (semantic) provider with a TF-IDF fallback.
pub struct FallbackEmbeddingProvider {
    primary: Arc<dyn EmbeddingProvider>,
    fallback: Arc<TfIdfProvider>,
    /// Sticky flag: set on the first primary failure; all later calls go to
    /// the fallback without retrying the primary.
    primary_failed: AtomicBool,
}

impl FallbackEmbeddingProvider {
    /// Create a new fallback wrapper.
    pub fn new(primary: Arc<dyn EmbeddingProvider>, fallback: Arc<TfIdfProvider>) -> Self {
        Self {
            primary,
            fallback,
            primary_failed: AtomicBool::new(false),
        }
    }

    /// Whether the primary (semantic) provider is still being used.
    /// `false` after the first primary failure (fallback active).
    pub fn primary_active(&self) -> bool {
        !self.primary_failed.load(Ordering::Acquire)
    }

    /// Handle to the TF-IDF fallback (e.g. for corpus statistics).
    pub fn fallback_provider(&self) -> Arc<TfIdfProvider> {
        Arc::clone(&self.fallback)
    }

    fn mark_primary_failed(&self, path: &str, error: &crate::error::MemoryError) {
        if !self.primary_failed.swap(true, Ordering::AcqRel) {
            tracing::warn!(
                primary = self.primary.name(),
                model = %self.primary.model_id(),
                path = path,
                error = %error,
                "semantic embedding provider failed; falling back to TF-IDF for all subsequent embeddings"
            );
        }
    }
}

#[async_trait]
impl EmbeddingProvider for FallbackEmbeddingProvider {
    async fn embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding> {
        if self.primary_active() {
            // Keep the fallback corpus warm on the document path so a later
            // fail-over has real IDF statistics. Best-effort: a fallback feed
            // failure must not degrade a healthy primary.
            if let Err(e) = self.fallback.embed(text, options).await {
                tracing::debug!(error = %e, "TF-IDF fallback corpus feed failed (non-fatal)");
            }
            match self.primary.embed(text, options).await {
                Ok(embedding) => return Ok(embedding),
                Err(e) => self.mark_primary_failed("embed", &e),
            }
        }
        self.fallback.embed(text, options).await
    }

    async fn embed_for_scoring(
        &self,
        text: &str,
        options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        if self.primary_active() {
            match self.primary.embed_for_scoring(text, options).await {
                Ok(embedding) => return Ok(embedding),
                Err(e) => self.mark_primary_failed("embed_for_scoring", &e),
            }
        }
        self.fallback.embed_for_scoring(text, options).await
    }

    fn embedding_dimension(&self) -> usize {
        if self.primary_active() {
            self.primary.embedding_dimension()
        } else {
            self.fallback.embedding_dimension()
        }
    }

    fn name(&self) -> &'static str {
        "FallbackEmbeddingProvider"
    }

    fn model_id(&self) -> String {
        if self.primary_active() {
            self.primary.model_id()
        } else {
            self.fallback.model_id()
        }
    }

    /// Reflects the ACTIVE provider: the primary's availability while it is
    /// healthy, the TF-IDF fallback's (always true) after fail-over — the
    /// pipeline never becomes unavailable because a semantic endpoint is down.
    fn is_available(&self) -> bool {
        if self.primary_active() {
            self.primary.is_available() || self.fallback.is_available()
        } else {
            self.fallback.is_available()
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        if self.primary_active() {
            self.primary.capabilities()
        } else {
            self.fallback.capabilities()
        }
    }
}
