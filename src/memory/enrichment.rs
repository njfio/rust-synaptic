//! Async/deferred write-path enrichment: the raw store is fast; the slow LLM
//! enrichment (extract -> resolve -> supersede -> fact-store) runs later,
//! concurrently, on shared handles. See docs/superpowers/specs/2026-07-22-*.

/// How write-path LLM enrichment is scheduled relative to `store()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnrichmentMode {
    /// Enrich synchronously inside `store()` (today's behavior). Default.
    #[default]
    Inline,
    /// `store()` enqueues; enrichment runs on an explicit `enrich_pending()`.
    Deferred,
    /// `store()` enqueues + notifies a background worker that drains continuously.
    Background,
}

/// One entry awaiting enrichment (queued by `store()` in Deferred/Background).
#[derive(Debug, Clone)]
pub struct PendingEnrichment {
    pub key: String,
    pub value: String,
}
