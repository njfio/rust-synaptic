//! Triggered reflection: clustering accumulated memories and synthesizing
//! provenance-linked insights (Task 4.1, agent-memory-v2).
//!
//! A [`ReflectionEngine`] embeds the memories accumulated since the last
//! reflection, clusters them by connected components over pairwise cosine
//! similarity (deterministic: fixed iteration order, no randomness), and asks
//! the configured [`MemoryReasoner`] to synthesize an [`Insight`] for every
//! cluster that reaches `min_cluster` members.

use crate::error::Result;
use crate::memory::embeddings::{Embedding, EmbeddingProvider};
use crate::memory::reasoning::{Insight, MemoryReasoner};
use crate::memory::types::MemoryEntry;
use std::sync::Arc;

/// Configuration for triggered reflection.
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Reflection fires only once the importance accumulated since the last
    /// reflection reaches this threshold.
    pub importance_threshold: f64,
    /// Minimum cluster size that is worth synthesizing an insight from.
    pub min_cluster: usize,
    /// Pairwise embedding cosine similarity at or above which two memories
    /// are connected during clustering.
    pub similarity_threshold: f64,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            importance_threshold: 3.0,
            min_cluster: 2,
            similarity_threshold: 0.3,
        }
    }
}

/// Clusters accumulated memories and synthesizes insights over each cluster.
pub struct ReflectionEngine {
    config: ReflectionConfig,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl ReflectionEngine {
    /// Create an engine over the given embedding provider.
    pub fn new(config: ReflectionConfig, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { config, embedder }
    }

    /// The active configuration.
    pub fn config(&self) -> &ReflectionConfig {
        &self.config
    }

    /// Replace the active configuration.
    pub fn set_config(&mut self, config: ReflectionConfig) {
        self.config = config;
    }

    /// Cluster `entries` and synthesize one insight per cluster of at least
    /// `min_cluster` members. Entries are processed in the order given, so
    /// the output is deterministic for a given input order.
    pub async fn reflect(
        &self,
        entries: &[MemoryEntry],
        reasoner: &dyn MemoryReasoner,
    ) -> Result<Vec<Insight>> {
        if entries.len() < self.config.min_cluster {
            return Ok(Vec::new());
        }
        let mut embeddings: Vec<Embedding> = Vec::with_capacity(entries.len());
        for entry in entries {
            embeddings.push(self.embedder.embed(&entry.value, None).await?);
        }
        let clusters = connected_components(&embeddings, self.config.similarity_threshold);

        let mut insights = Vec::new();
        for cluster in clusters {
            if cluster.len() < self.config.min_cluster {
                continue;
            }
            let members: Vec<MemoryEntry> = cluster.iter().map(|&i| entries[i].clone()).collect();
            if let Some(insight) = reasoner.synthesize(&members).await? {
                insights.push(insight);
            }
        }
        Ok(insights)
    }
}

/// Connected components over the graph whose vertices are `embeddings` and
/// whose edges connect pairs with cosine similarity `>= threshold`.
///
/// Uses union-find with deterministic pair iteration; components are returned
/// ordered by their smallest member index, members in ascending index order.
fn connected_components(embeddings: &[Embedding], threshold: f64) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if embeddings[i].cosine_similarity(&embeddings[j]) >= threshold {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    // Attach the larger root to the smaller so component
                    // roots are the smallest member index (deterministic).
                    let (lo, hi) = if ri < rj { (ri, rj) } else { (rj, ri) };
                    parent[hi] = lo;
                }
            }
        }
    }

    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut root_to_component: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let idx = *root_to_component.entry(root).or_insert_with(|| {
            components.push(Vec::new());
            components.len() - 1
        });
        components[idx].push(i);
    }
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn embedding(vector: Vec<f32>) -> Embedding {
        Embedding {
            vector,
            model: "test".to_string(),
            version: None,
            created_at: Utc::now(),
            content_hash: String::new(),
            token_count: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn connected_components_groups_similar_vectors() {
        let embeddings = vec![
            embedding(vec![1.0, 0.0]),
            embedding(vec![0.9, 0.1]),
            embedding(vec![0.0, 1.0]),
        ];
        let clusters = connected_components(&embeddings, 0.8);
        assert_eq!(clusters, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn connected_components_is_transitive() {
        // 0~1 and 1~2 but 0 and 2 alone would not connect: one component.
        let embeddings = vec![
            embedding(vec![1.0, 0.0]),
            embedding(vec![0.7, 0.7]),
            embedding(vec![0.0, 1.0]),
        ];
        let clusters = connected_components(&embeddings, 0.7);
        assert_eq!(clusters, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn default_config_is_sensible() {
        let config = ReflectionConfig::default();
        assert_eq!(config.importance_threshold, 3.0);
        assert_eq!(config.min_cluster, 2);
        assert!(config.similarity_threshold > 0.0 && config.similarity_threshold < 1.0);
    }
}
