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

// --- `enrich_one` engine: the write-path enrichment logic (extract ->
// resolve -> supersede -> fact-store -> tag raw), lifted from the `&mut
// self` methods in `src/lib.rs` onto shared `Arc` handles so later tasks can
// run it concurrently (Deferred/Background modes) instead of inline in
// `store()`. See docs/superpowers/specs/2026-07-22-*.

#[cfg(feature = "embeddings")]
mod enrich_one_engine {
    use crate::error::Result;
    use crate::memory::embeddings::provider::EmbeddingProvider;
    use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
    use crate::memory::reasoning::{ConflictResolution, Fact, MemoryReasoner, NeighborFact};
    use crate::memory::state::AgentState;
    use crate::memory::storage::Storage;
    use crate::memory::types::{MemoryEntry, MemoryType};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// Custom-field key recording when a memory was bi-temporally superseded.
    /// Mirrors `AgentMemory::SUPERSEDED_FIELD`.
    const SUPERSEDED_FIELD: &str = "superseded_at";
    /// Custom-field key recording when a raw turn was summarized by
    /// distillation. Mirrors `AgentMemory::RAW_SOURCE_FIELD`.
    const RAW_SOURCE_FIELD: &str = "raw_source";
    /// Upper bound on the keyword-hit candidates `neighbor_facts_shared`
    /// pulls per search. Mirrors `AgentMemory::NEIGHBOR_CANDIDATE_LIMIT`.
    const NEIGHBOR_CANDIDATE_LIMIT: usize = 20;

    /// Per-entry enrichment knobs, mirroring the relevant fields of
    /// `AgentMemory`/`MemoryConfig` that `enrich_one` needs.
    #[derive(Debug, Clone, Copy)]
    pub struct EnrichmentParams {
        /// Mirrors `AgentMemory.distillation_live`: whether extracted facts
        /// are persisted as their own retrievable memories.
        pub distillation_live: bool,
        /// Reserved for a future step (raw-source exclusion default);
        /// unused by `enrich_one` itself today.
        pub exclude_raw_default: bool,
    }

    /// Summary of what `enrich_one` did for a single entry.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct EnrichmentOutcome {
        pub facts_stored: usize,
        pub supersessions: usize,
        pub raw_tagged: bool,
    }

    /// Run the reasoner over a freshly stored entry `(key, value)`: extract
    /// facts, resolve each against the most similar existing memories, apply
    /// the resolution to the knowledge graph, optionally persist each fact as
    /// its own retrievable memory, and tag the raw turn as summarized once
    /// at least one fact was stored.
    ///
    /// Mirrors `AgentMemory::reason_over_store` exactly, but operates on
    /// shared `Arc` handles instead of `&mut self` so it can run outside the
    /// inline write path (Deferred/Background enrichment).
    pub async fn enrich_one(
        storage: &Arc<dyn Storage + Send + Sync>,
        kg: &Arc<RwLock<MemoryKnowledgeGraph>>,
        state: &Arc<RwLock<AgentState>>,
        reasoner: &Arc<dyn MemoryReasoner>,
        scoring: &Option<Arc<dyn EmbeddingProvider>>,
        params: EnrichmentParams,
        key: &str,
        value: &str,
    ) -> Result<EnrichmentOutcome> {
        let ctx = crate::memory::reasoning::ExtractionContext {
            source_key: key.to_string(),
            timestamp: chrono::Utc::now(),
        };
        let extraction = reasoner.extract(value, &ctx).await?;
        let mut outcome = EnrichmentOutcome::default();
        // Empty extraction: default behavior stays identical (no KG change
        // beyond what the earlier subsystems already applied).
        if extraction.facts.is_empty() {
            return Ok(outcome);
        }
        let mut stored_any_fact = false;
        for (i, fact) in extraction.facts.iter().enumerate() {
            let neighbors = neighbor_facts_shared(storage, key, fact, 5).await?;
            let resolution = reasoner.resolve(fact, &neighbors).await?;
            // The reasoner picks its target among the neighbors we supplied;
            // for id-less outcomes (UpdateInPlace/Append) the target is the
            // best neighbor by the same deterministic ordering.
            let best_neighbor_id = neighbors.first().map(|n| n.id.clone());
            let superseded =
                apply_resolution_shared(kg, key, fact, resolution, best_neighbor_id).await?;
            // Mark the superseded memory itself (not just its KG edges) so
            // bi-temporal validity is visible to retrieval.
            if let Some(old_key) = superseded {
                mark_field_shared(storage, state, &old_key, SUPERSEDED_FIELD).await?;
                outcome.supersessions += 1;
            }

            // Persist the extracted fact as its own retrievable memory so
            // search surfaces distilled facts, not just raw turns.
            if params.distillation_live && !fact.text.trim().is_empty() {
                let fact_key = format!("{key}::fact{i}");
                store_fact_shared(storage, state, scoring, &fact_key, &fact.text).await?;
                stored_any_fact = true;
                outcome.facts_stored += 1;
            }
        }
        // Failure-ordering invariant: tag the raw turn as a summarized source
        // ONLY after >=1 fact has been stored. If extraction or fact storage
        // failed above (propagated via `?`), we never reach here, so the raw
        // turn stays untagged and searchable -- a distillation outage
        // degrades to raw-turn retrieval rather than hiding a turn with no
        // replacement.
        if stored_any_fact {
            mark_field_shared(storage, state, key, RAW_SOURCE_FIELD).await?;
            outcome.raw_tagged = true;
        }
        Ok(outcome)
    }

    /// Store an extracted fact directly (not via a recursive
    /// `store_with_report`) so `enrich_one` never recurses through the
    /// inline enrichment path: raw storage + state, plus a best-effort
    /// scoring-provider embed so the fact stays pipeline-retrievable (this
    /// mirrors what `AgentMemory::store_with_report` does for every entry).
    async fn store_fact_shared(
        storage: &Arc<dyn Storage + Send + Sync>,
        state: &Arc<RwLock<AgentState>>,
        scoring: &Option<Arc<dyn EmbeddingProvider>>,
        fact_key: &str,
        fact_text: &str,
    ) -> Result<()> {
        let entry = MemoryEntry::new(
            fact_key.to_string(),
            fact_text.to_string(),
            MemoryType::LongTerm,
        );
        storage.store(&entry).await?;
        state.write().await.add_memory(entry.clone());
        if let Some(provider) = scoring {
            // Keep the retrieval scoring corpus live so the pipeline can
            // rank the fact; best-effort like the inline write path.
            let _ = provider.embed(fact_text, None).await;
        }
        Ok(())
    }

    /// Tag `key`'s stored entry with `field` = now (RFC3339), re-persist it,
    /// and keep in-memory state consistent if the key is tracked there.
    /// Mirrors `AgentMemory::mark_superseded`/`mark_raw_source`.
    async fn mark_field_shared(
        storage: &Arc<dyn Storage + Send + Sync>,
        state: &Arc<RwLock<AgentState>>,
        key: &str,
        field: &str,
    ) -> Result<()> {
        if let Some(mut entry) = storage.retrieve(key).await? {
            entry
                .metadata
                .custom_fields
                .insert(field.to_string(), chrono::Utc::now().to_rfc3339());
            storage.store(&entry).await?;
            if state.read().await.has_memory(key) {
                state.write().await.add_memory(entry);
            }
        }
        Ok(())
    }

    /// Build the top-`k` neighbor list for a candidate `fact`, excluding the
    /// entry being stored. Ported verbatim from `AgentMemory::neighbor_facts`
    /// (candidate source, ranking, and tie-breaking are unchanged).
    async fn neighbor_facts_shared(
        storage: &Arc<dyn Storage + Send + Sync>,
        current_key: &str,
        fact: &Fact,
        k: usize,
    ) -> Result<Vec<NeighborFact>> {
        fn tokens(text: &str) -> std::collections::HashSet<String> {
            text.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect()
        }
        let candidate_tokens = tokens(&fact.text);
        if candidate_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // (a) Build a keyword query from distinctive (>=3 char) tokens so a
        // handful of long, discriminating terms outrank many short stopword
        // matches. Fall back to the raw text if nothing survives the filter.
        let distinctive: Vec<&str> = fact
            .text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() >= 3)
            .collect();
        let query = if distinctive.is_empty() {
            fact.text.clone()
        } else {
            distinctive.join(" ")
        };

        // Collect a bounded, deduplicated candidate set: the keyword hits ...
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut candidates: Vec<crate::memory::types::MemoryFragment> = Vec::new();
        // +1 head-room: the search may return the entry being stored itself.
        for fragment in storage.search(&query, NEIGHBOR_CANDIDATE_LIMIT + 1).await? {
            if seen.insert(fragment.entry.key.clone()) {
                candidates.push(fragment);
            }
        }
        // (b) ... unioned with a targeted lookup per relation subject so a
        // same-subject memory can never be squeezed out of the candidate set.
        for relation in &fact.relations {
            if relation.subject.trim().is_empty() {
                continue;
            }
            for fragment in storage
                .search(&relation.subject, NEIGHBOR_CANDIDATE_LIMIT + 1)
                .await?
            {
                if seen.insert(fragment.entry.key.clone()) {
                    candidates.push(fragment);
                }
            }
        }

        let mut neighbors: Vec<NeighborFact> = candidates
            .iter()
            .filter(|fragment| fragment.entry.key != current_key)
            .filter_map(|fragment| {
                let mem_tokens = tokens(&fragment.entry.value);
                if mem_tokens.is_empty() {
                    return None;
                }
                let intersection = candidate_tokens.intersection(&mem_tokens).count() as f64;
                let similarity = intersection
                    / ((candidate_tokens.len() as f64).sqrt() * (mem_tokens.len() as f64).sqrt());
                if similarity > 0.0 {
                    Some(NeighborFact {
                        id: fragment.entry.key.clone(),
                        similarity,
                        text: fragment.entry.value.clone(),
                    })
                } else {
                    None
                }
            })
            .collect();
        neighbors.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.id.cmp(&b.id))
        });
        neighbors.truncate(k);
        Ok(neighbors)
    }

    /// Apply a conflict resolution outcome to the knowledge graph. Ported
    /// verbatim from `AgentMemory::apply_resolution`. The `kg` write guard is
    /// held only for the span of this function (the KG mutation itself);
    /// callers must never hold it across a `reasoner.extract`/`resolve`
    /// await.
    async fn apply_resolution_shared(
        kg: &Arc<RwLock<MemoryKnowledgeGraph>>,
        entry_key: &str,
        fact: &Fact,
        resolution: ConflictResolution,
        best_neighbor_id: Option<String>,
    ) -> Result<Option<String>> {
        let mut superseded_key: Option<String> = None;
        let mut kg = kg.write().await;
        match resolution {
            ConflictResolution::Insert => {
                kg.add_extracted_fact(entry_key, fact).await?;
            }
            ConflictResolution::Supersede { old_id, reason } => {
                tracing::debug!(old_id = %old_id, reason = %reason, "superseding fact");
                // Bi-temporally invalidate only the contradicted edges of
                // the old memory (same subject + predicate as the new fact).
                kg.supersede_matching_relations(&old_id, fact).await?;
                kg.add_extracted_fact(entry_key, fact).await?;
                // `old_id` is the superseded memory's key (see
                // `neighbor_facts_shared`): report it so the caller can mark
                // the memory itself superseded, making bi-temporal validity
                // visible to retrieval.
                superseded_key = Some(old_id);
            }
            ConflictResolution::UpdateInPlace { reason } => {
                tracing::debug!(reason = %reason, "updating fact in place");
                if let Some(target) = best_neighbor_id {
                    kg.update_memory_node_content(&target, &fact.text).await?;
                }
            }
            ConflictResolution::Append { reason } => {
                tracing::debug!(reason = %reason, "appending to existing fact");
                if let Some(target) = best_neighbor_id {
                    kg.append_memory_content(&target, &fact.text).await?;
                }
            }
            ConflictResolution::NoOp { reason } => {
                tracing::debug!(reason = %reason, "reasoner chose no-op");
            }
        }
        Ok(superseded_key)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::memory::knowledge_graph::GraphConfig;
        use crate::memory::storage::memory::MemoryStorage;

        #[derive(Clone)]
        struct StubReasoner;

        #[async_trait::async_trait]
        impl MemoryReasoner for StubReasoner {
            async fn extract(
                &self,
                _text: &str,
                _ctx: &crate::memory::reasoning::ExtractionContext,
            ) -> Result<crate::memory::reasoning::Extraction> {
                Ok(crate::memory::reasoning::Extraction {
                    facts: vec![Fact {
                        text: "Alice lives in Berlin".to_string(),
                        entities: vec![],
                        relations: vec![],
                    }],
                })
            }
            async fn resolve(
                &self,
                _candidate: &Fact,
                _neighbors: &[NeighborFact],
            ) -> Result<ConflictResolution> {
                Ok(ConflictResolution::Insert)
            }
            async fn synthesize(
                &self,
                _cluster: &[MemoryEntry],
            ) -> Result<Option<crate::memory::reasoning::Insight>> {
                Ok(None)
            }
            fn name(&self) -> &str {
                "StubReasoner"
            }
        }

        #[tokio::test]
        async fn enrich_one_stores_fact_and_tags_raw() {
            let storage: Arc<dyn Storage + Send + Sync> = Arc::new(MemoryStorage::new());
            let kg = Arc::new(RwLock::new(MemoryKnowledgeGraph::new(
                GraphConfig::default(),
            )));
            let state = Arc::new(RwLock::new(AgentState::new(uuid::Uuid::new_v4())));
            let reasoner: Arc<dyn MemoryReasoner> = Arc::new(StubReasoner);
            let scoring: Option<Arc<dyn EmbeddingProvider>> = None;

            let raw = MemoryEntry::new(
                "turn1".to_string(),
                "Alice: Berlin".to_string(),
                MemoryType::ShortTerm,
            );
            storage.store(&raw).await.unwrap();
            state.write().await.add_memory(raw.clone());

            let params = EnrichmentParams {
                distillation_live: true,
                exclude_raw_default: false,
            };

            let outcome = enrich_one(
                &storage,
                &kg,
                &state,
                &reasoner,
                &scoring,
                params,
                "turn1",
                "Alice: Berlin",
            )
            .await
            .unwrap();

            assert_eq!(outcome.facts_stored, 1);
            assert!(outcome.raw_tagged);
            assert_eq!(outcome.supersessions, 0);

            let stored_fact = storage.retrieve("turn1::fact0").await.unwrap();
            assert!(stored_fact.is_some());
            assert_eq!(stored_fact.unwrap().value, "Alice lives in Berlin");

            let raw_after = storage.retrieve("turn1").await.unwrap().unwrap();
            assert!(raw_after
                .metadata
                .custom_fields
                .contains_key(RAW_SOURCE_FIELD));
        }
    }
}

#[cfg(feature = "embeddings")]
pub use enrich_one_engine::{enrich_one, EnrichmentOutcome, EnrichmentParams};
