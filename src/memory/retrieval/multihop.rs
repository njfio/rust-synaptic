//! Multi-hop graph-expansion retrieval (HippoRAG-style).
//!
//! Iterative retrieve → expand → score over the knowledge graph:
//! 1. **Seed** with the top hits of the configured seed retrievers
//!    (dense/keyword) for the query.
//! 2. **Expand** along KG relation edges from the seed memories' nodes,
//!    breadth-first, up to `max_hops` hops with a bounded frontier.
//! 3. **Score** each reached memory as
//!    `seed_score × Π(edge_strength) × hop_decay^hop` (best path wins).
//! 4. Feed the expanded set into the existing hybrid fusion as a
//!    `GraphRelationship` signal, so multi-hop-reachable memories surface
//!    even when they share no terms with the query.
//!
//! The traversal is bounded (hop cap, per-hop frontier cap, total expansion
//! cap) and fully deterministic: seeds, frontiers, and neighbor lists are
//! sorted with total tie-breaks (score desc, then key/node id asc).
//!
//! Lock discipline: the knowledge-graph read guard is held only for the
//! in-memory traversal and dropped before any storage `.await`.

use super::pipeline::{PipelineConfig, RetrievalPipeline, RetrievalSignal, ScoredMemory};
use crate::error::Result;
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use crate::memory::storage::Storage;
use crate::memory::types::MemoryFragment;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Bounds and scoring parameters for multi-hop graph expansion.
#[derive(Debug, Clone)]
pub struct MultiHopConfig {
    /// Maximum number of relation hops from a seed (capped at 2).
    pub max_hops: usize,
    /// Maximum number of seed memories to expand from.
    pub max_seeds: usize,
    /// Maximum frontier nodes carried into each hop.
    pub max_frontier: usize,
    /// Maximum total expanded memories returned.
    pub max_expanded: usize,
    /// Multiplicative score decay applied per hop (0, 1].
    pub hop_decay: f64,
}

impl Default for MultiHopConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_seeds: 5,
            max_frontier: 16,
            max_expanded: 32,
            hop_decay: 0.8,
        }
    }
}

/// HippoRAG-style multi-hop graph-expansion retriever.
///
/// Seeds with the top results of `seed_retrievers`, expands along knowledge
/// graph edges, and emits the expanded memories as `GraphRelationship`
/// scored results for the hybrid fusion stage.
pub struct MultiHopGraphRetriever {
    storage: Arc<dyn Storage + Send + Sync>,
    knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
    seed_retrievers: Vec<Arc<dyn RetrievalPipeline>>,
    config: MultiHopConfig,
}

/// Best-known path state for a reached graph node during expansion.
#[derive(Debug, Clone)]
struct ReachedNode {
    score: f64,
    hop: usize,
    seed_key: String,
}

impl MultiHopGraphRetriever {
    pub fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        knowledge_graph: Option<Arc<tokio::sync::RwLock<MemoryKnowledgeGraph>>>,
        seed_retrievers: Vec<Arc<dyn RetrievalPipeline>>,
        config: MultiHopConfig,
    ) -> Self {
        Self {
            storage,
            knowledge_graph,
            seed_retrievers,
            config,
        }
    }

    /// Collect deterministic seeds: run each seed retriever, keep each
    /// memory's best score, order by (score desc, key asc), cap at
    /// `max_seeds`.
    ///
    /// NOTE (PL1): an earlier optimization reused the pipeline's already-
    /// computed dense/keyword results as seeds. That was NOT exact — the
    /// pipeline runs those retrievers with `limit = max_per_signal`, whose
    /// larger BM25 IDF candidate pool yields different scores (and therefore a
    /// different top-`max_seeds` seed set) than a direct `max_seeds`-limited
    /// run. The differing seed set changed multi-hop results and regressed
    /// recall, so the reuse was reverted: seeds are always collected fresh,
    /// exactly as before PL1. (The batched expansion `storage.retrieve` below
    /// is exact and was kept.)
    async fn collect_seeds(
        &self,
        query: &str,
        config: Option<&PipelineConfig>,
    ) -> Result<Vec<(String, f64)>> {
        let mut best: HashMap<String, f64> = HashMap::new();
        for retriever in &self.seed_retrievers {
            if !retriever.is_available() {
                continue;
            }
            let hits = match retriever.search(query, self.config.max_seeds, config).await {
                Ok(hits) => hits,
                Err(e) => {
                    tracing::warn!(
                        seed_retriever = retriever.name(),
                        error = %e,
                        "multi-hop seed retrieval failed; continuing with other seeds"
                    );
                    continue;
                }
            };
            for hit in hits {
                let entry = best.entry(hit.memory.entry.key.clone()).or_insert(0.0);
                if hit.score > *entry {
                    *entry = hit.score;
                }
            }
        }

        let mut seeds: Vec<(String, f64)> = best.into_iter().collect();
        seeds.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        seeds.truncate(self.config.max_seeds);
        Ok(seeds)
    }

    /// Bounded, deterministic BFS over KG edges from the seed nodes.
    ///
    /// Runs entirely under a single read guard on the knowledge graph (pure
    /// in-memory work, no storage access) and returns the reached memory
    /// keys with their propagated scores. The caller fetches the memory
    /// entries from storage *after* the guard is dropped.
    async fn expand_seeds(
        &self,
        kg: &MemoryKnowledgeGraph,
        seeds: &[(String, f64)],
    ) -> Result<Vec<(String, ReachedNode)>> {
        let max_hops = self.config.max_hops.min(2);
        let hop_decay = self.config.hop_decay.clamp(0.0, 1.0);

        // Resolve seed nodes; seeds without a graph node are skipped (they
        // are already surfaced by the seed retrievers themselves).
        let mut frontier: Vec<(Uuid, ReachedNode)> = Vec::new();
        let mut best_by_node: HashMap<Uuid, ReachedNode> = HashMap::new();
        let mut seed_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (key, score) in seeds {
            seed_keys.insert(key.clone());
            let Some(node_id) = kg.get_node_for_memory(key).await? else {
                continue;
            };
            let state = ReachedNode {
                score: *score,
                hop: 0,
                seed_key: key.clone(),
            };
            best_by_node.insert(node_id, state.clone());
            frontier.push((node_id, state));
        }

        for hop in 1..=max_hops {
            if frontier.is_empty() {
                break;
            }
            // Deterministic frontier: score desc, node id asc, capped.
            frontier.sort_by(|a, b| {
                b.1.score
                    .partial_cmp(&a.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            frontier.truncate(self.config.max_frontier);

            let mut next_frontier: Vec<(Uuid, ReachedNode)> = Vec::new();
            for (node_id, state) in &frontier {
                let mut neighbors = kg.get_connected_nodes(*node_id).await?;
                // Deterministic neighbor order: strength desc, node id asc.
                neighbors.sort_by(|a, b| {
                    b.2.partial_cmp(&a.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                for (neighbor, _rel_type, strength) in neighbors {
                    let score =
                        (state.score * strength.clamp(0.0, 1.0) * hop_decay).clamp(0.0, 1.0);
                    if score <= 0.0 {
                        continue;
                    }
                    let improved = match best_by_node.get(&neighbor) {
                        Some(existing) => score > existing.score,
                        None => true,
                    };
                    if !improved {
                        continue;
                    }
                    let reached = ReachedNode {
                        score,
                        hop,
                        seed_key: state.seed_key.clone(),
                    };
                    best_by_node.insert(neighbor, reached.clone());
                    next_frontier.push((neighbor, reached));
                }
            }
            frontier = next_frontier;
        }

        // Map reached nodes back to memory keys, excluding the seeds
        // themselves (hop 0) — expansion is purely additive.
        let mut reached: Vec<(String, ReachedNode)> = Vec::new();
        for (node_id, state) in best_by_node {
            if state.hop == 0 {
                continue;
            }
            if let Some(memory_key) = kg.get_memory_for_node(node_id).await? {
                if !seed_keys.contains(&memory_key) {
                    reached.push((memory_key, state));
                }
            }
        }

        // Deterministic output: score desc, key asc, capped.
        reached.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        reached.truncate(self.config.max_expanded);
        Ok(reached)
    }
}

#[async_trait]
impl RetrievalPipeline for MultiHopGraphRetriever {
    async fn search(
        &self,
        query: &str,
        limit: usize,
        config: Option<&PipelineConfig>,
    ) -> Result<Vec<ScoredMemory>> {
        let Some(ref kg) = self.knowledge_graph else {
            return Ok(Vec::new());
        };

        let seeds = self.collect_seeds(query, config).await?;
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Traverse under a single read guard, then DROP it before any
        // storage awaits — the KG lock is never held across `.await` on
        // storage (v2/v3 lock discipline).
        let reached = {
            let kg_guard = kg.read().await;
            self.expand_seeds(&kg_guard, &seeds).await?
        };

        // Fetch the reached memories from storage concurrently in one
        // batch (the reached set is already capped at `max_expanded`)
        // instead of one serial await per key.
        let fetched =
            futures::future::join_all(reached.iter().map(|(key, _)| self.storage.retrieve(key)))
                .await;

        let mut results = Vec::with_capacity(reached.len().min(limit));
        for ((memory_key, state), entry_result) in reached.into_iter().zip(fetched) {
            if results.len() >= limit {
                break;
            }
            let entry = match entry_result {
                Ok(Some(entry)) => entry,
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        key = %memory_key,
                        error = %e,
                        "multi-hop expansion candidate fetch failed"
                    );
                    continue;
                }
            };
            let fragment = MemoryFragment::new(entry, state.score);
            results.push(
                ScoredMemory::new(fragment, state.score, RetrievalSignal::GraphRelationship)
                    .with_explanation(format!(
                        "Surfaced via {}-hop graph expansion from seed '{}'",
                        state.hop, state.seed_key
                    )),
            );
        }

        tracing::debug!(
            query = %query,
            result_count = results.len(),
            "MultiHopGraphRetriever: expansion completed"
        );
        Ok(results)
    }

    fn name(&self) -> &'static str {
        "MultiHopGraphRetriever"
    }

    fn signal_type(&self) -> RetrievalSignal {
        RetrievalSignal::GraphRelationship
    }

    fn is_available(&self) -> bool {
        self.knowledge_graph.is_some() && !self.seed_retrievers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_bounded() {
        let config = MultiHopConfig::default();
        assert_eq!(config.max_hops, 2);
        assert!(config.max_seeds > 0);
        assert!(config.max_frontier > 0);
        assert!(config.max_expanded > 0);
        assert!(config.hop_decay > 0.0 && config.hop_decay <= 1.0);
    }
}
