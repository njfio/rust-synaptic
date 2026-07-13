//! Knowledge graph implementation for AI agent memory system
//!
//! This module provides a graph-based representation of memories and their relationships,
//! enabling sophisticated querying and reasoning capabilities.

pub mod graph;
pub mod query;
pub mod reasoning;
pub mod types;

// Re-export commonly used types
pub use graph::{GraphConfig, GraphStats, KnowledgeGraph};
pub use query::{GraphQuery, GraphQueryBuilder, QueryResult, TraversalOptions};
pub use reasoning::{GraphReasoner, InferenceEngine, InferenceRule};
pub use types::{
    Edge, GraphEntity, GraphPath, GraphPattern, KnowledgeGraphMetadata, Node, NodeType,
    Relationship, RelationshipType,
};

// Distributed graph types
pub type MemoryNode = Node;
pub type MemoryEdge = Edge;

use crate::error::{MemoryError, Result};
use crate::memory::reasoning::{EntityKind, Fact};
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A subject-predicate-object relation materialized in the knowledge graph
/// from reasoner extraction, with its provenance and supersession state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    /// Label of the subject entity node.
    pub subject: String,
    /// Normalized predicate the edge was created with.
    pub predicate: String,
    /// Label of the object entity node.
    pub object: String,
    /// Key of the memory the relation was extracted from, if recorded.
    pub source_memory: Option<String>,
    /// True if the relation's edge is no longer bi-temporally valid (it was
    /// invalidated by a superseding fact).
    pub superseded: bool,
}

/// Planned knowledge-graph mutation for a stored memory, computed under a
/// read lock by [`MemoryKnowledgeGraph::plan_memory_node_update`] and applied
/// under a (short) write lock by
/// [`MemoryKnowledgeGraph::apply_memory_node_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryNodePlan {
    /// The memory already has a backing node: update it in place.
    UpdateExisting(Uuid),
    /// A sufficiently similar node exists: merge the memory into it.
    MergeWith(Uuid),
    /// No existing or similar node: create a new one.
    CreateNew,
}

/// Edge property key recording which memory a relation was extracted from.
const PROP_SOURCE_MEMORY: &str = "source_memory";
/// Edge property key carrying the normalized predicate.
const PROP_PREDICATE: &str = "predicate";

/// Main knowledge graph interface for the memory system
pub struct MemoryKnowledgeGraph {
    /// Core graph structure
    graph: KnowledgeGraph,
    /// Memory entry to node mapping
    memory_to_node: HashMap<String, Uuid>, // memory_key -> node_id
    /// Node to memory entry mapping
    node_to_memory: HashMap<Uuid, String>, // node_id -> memory_key
    /// Extracted entity nodes by lowercased entity name
    entity_nodes: HashMap<String, Uuid>,
    /// Graph configuration
    config: GraphConfig,
    /// Reasoning engine
    reasoner: GraphReasoner,
}

impl MemoryKnowledgeGraph {
    /// Create a new memory knowledge graph
    pub fn new(config: GraphConfig) -> Self {
        let graph = KnowledgeGraph::new(config.clone());
        let reasoner = GraphReasoner::new();

        Self {
            graph,
            memory_to_node: HashMap::new(),
            node_to_memory: HashMap::new(),
            entity_nodes: HashMap::new(),
            config,
            reasoner,
        }
    }

    /// Get or create the concept node for an extracted entity, keyed by its
    /// lowercased name so repeated mentions map onto one node.
    async fn get_or_create_entity_node(&mut self, name: &str, kind: &EntityKind) -> Result<Uuid> {
        let key = name.to_lowercase();
        if let Some(id) = self.entity_nodes.get(&key) {
            return Ok(*id);
        }
        let node_type = match kind {
            EntityKind::Person => NodeType::Person,
            EntityKind::Place => NodeType::Location,
            _ => NodeType::Concept,
        };
        let mut node = Node::new(node_type, name.to_string());
        node.add_property("concept".to_string(), key.clone());
        let node_id = self.graph.add_node(node).await?;
        self.entity_nodes.insert(key, node_id);
        Ok(node_id)
    }

    /// Materialize an extracted fact into the graph: one node per entity and
    /// one edge per relation, each edge recording its source memory key.
    pub async fn add_extracted_fact(&mut self, source_memory_key: &str, fact: &Fact) -> Result<()> {
        for entity in &fact.entities {
            self.get_or_create_entity_node(&entity.name, &entity.kind)
                .await?;
        }
        for relation in &fact.relations {
            let subject_id = match self.entity_nodes.get(&relation.subject.to_lowercase()) {
                Some(id) => *id,
                None => {
                    self.get_or_create_entity_node(&relation.subject, &EntityKind::Term)
                        .await?
                }
            };
            let object_id = match self.entity_nodes.get(&relation.object.to_lowercase()) {
                Some(id) => *id,
                None => {
                    self.get_or_create_entity_node(&relation.object, &EntityKind::Term)
                        .await?
                }
            };
            let mut properties = HashMap::new();
            properties.insert(PROP_PREDICATE.to_string(), relation.predicate.clone());
            properties.insert(
                PROP_SOURCE_MEMORY.to_string(),
                source_memory_key.to_string(),
            );
            let edge = Edge::new(
                subject_id,
                object_id,
                RelationshipType::Custom(relation.predicate.clone()),
                Some(properties),
            );
            self.graph.add_edge(edge).await?;
        }
        Ok(())
    }

    /// Bi-temporally invalidate the edges from `old_memory_key` that the
    /// superseding `fact` contradicts: for each relation in `fact`, only the
    /// still-valid edges with the same subject and predicate sourced from
    /// `old_memory_key` are invalidated (per-fact granularity — unrelated
    /// edges from the same source memory are left untouched). The edges
    /// remain stored but are no longer valid at or after now. Returns the
    /// number of edges invalidated.
    pub async fn supersede_matching_relations(
        &mut self,
        old_memory_key: &str,
        fact: &Fact,
    ) -> Result<usize> {
        let now = chrono::Utc::now();
        let mut invalidated = 0;
        for relation in &fact.relations {
            let Some(subject_id) = self.entity_nodes.get(&relation.subject.to_lowercase()) else {
                continue;
            };
            let edge_ids: Vec<Uuid> = self
                .graph
                .edges
                .iter()
                .filter(|edge| {
                    edge.from_node == *subject_id
                        && edge
                            .relationship
                            .properties
                            .get(PROP_SOURCE_MEMORY)
                            .map(String::as_str)
                            == Some(old_memory_key)
                        && edge
                            .relationship
                            .properties
                            .get(PROP_PREDICATE)
                            .map(String::as_str)
                            == Some(relation.predicate.as_str())
                        && edge.is_valid_at(now)
                })
                .map(|edge| edge.id)
                .collect();
            for edge_id in edge_ids {
                if self.graph.invalidate_edge(&edge_id.to_string(), now)? {
                    invalidated += 1;
                }
            }
        }
        Ok(invalidated)
    }

    /// Replace the content of the node backing `memory_key`. Returns true if
    /// the node existed and was updated.
    pub async fn update_memory_node_content(
        &mut self,
        memory_key: &str,
        content: &str,
    ) -> Result<bool> {
        let Some(node_id) = self.memory_to_node.get(memory_key) else {
            return Ok(false);
        };
        if let Some(mut node) = self.graph.nodes.get_mut(node_id) {
            node.description = Some(content.to_string());
            node.last_modified = Some(chrono::Utc::now());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Append `new_content` to the node backing `memory_key` using the
    /// existing content-merge behavior. Returns true if the node existed.
    pub async fn append_memory_content(
        &mut self,
        memory_key: &str,
        new_content: &str,
    ) -> Result<bool> {
        let Some(node_id) = self.memory_to_node.get(memory_key).copied() else {
            return Ok(false);
        };
        let Some(existing) = self.graph.get_node(node_id).await? else {
            return Ok(false);
        };
        let merged = self
            .merge_content(&existing.description.unwrap_or_default(), new_content)
            .await?;
        if let Some(mut node) = self.graph.nodes.get_mut(&node_id) {
            node.description = Some(merged);
            node.last_modified = Some(chrono::Utc::now());
        }
        Ok(true)
    }

    /// All extracted relations whose subject node is the entity named
    /// `entity_name` (case-insensitive), with supersession state.
    pub async fn relations_for_entity(&self, entity_name: &str) -> Result<Vec<ExtractedRelation>> {
        let Some(subject_id) = self.entity_nodes.get(&entity_name.to_lowercase()) else {
            return Ok(Vec::new());
        };
        let subject_label = self
            .graph
            .nodes
            .get(subject_id)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| entity_name.to_string());
        let mut relations = Vec::new();
        for edge in self.graph.edges.iter() {
            if edge.from_node != *subject_id {
                continue;
            }
            let Some(predicate) = edge.relationship.properties.get(PROP_PREDICATE).cloned() else {
                continue;
            };
            let object = self
                .graph
                .nodes
                .get(&edge.to_node)
                .map(|n| n.label.clone())
                .unwrap_or_default();
            relations.push(ExtractedRelation {
                subject: subject_label.clone(),
                predicate,
                object,
                source_memory: edge
                    .relationship
                    .properties
                    .get(PROP_SOURCE_MEMORY)
                    .cloned(),
                superseded: !edge.is_valid_at(chrono::Utc::now()),
            });
        }
        relations.sort_by(|a, b| {
            (&a.predicate, &a.object, &a.source_memory).cmp(&(
                &b.predicate,
                &b.object,
                &b.source_memory,
            ))
        });
        Ok(relations)
    }

    /// Add or update a memory entry in the knowledge graph
    #[tracing::instrument(skip(self, memory), fields(memory_key = %memory.key))]
    pub async fn add_or_update_memory_node(&mut self, memory: &MemoryEntry) -> Result<Uuid> {
        tracing::debug!("Adding or updating memory node in knowledge graph");

        // Check if this memory already has a node
        if let Some(existing_node_id) = self.memory_to_node.get(&memory.key) {
            tracing::debug!("Updating existing node: {}", existing_node_id);
            // Update existing node
            self.update_existing_node(*existing_node_id, memory).await
        } else {
            // Check for similar existing nodes that should be merged
            if let Some(similar_node_id) = self.find_similar_node(memory).await? {
                tracing::debug!("Merging with similar node: {}", similar_node_id);
                self.merge_with_existing_node(similar_node_id, memory).await
            } else {
                tracing::debug!("Creating new node");
                // Create new node
                self.create_new_node(memory).await
            }
        }
    }

    /// Legacy method for backward compatibility
    pub async fn add_memory_node(&mut self, memory: &MemoryEntry) -> Result<Uuid> {
        self.add_or_update_memory_node(memory).await
    }

    /// Plan the KG mutation for a stored memory WITHOUT mutating the graph.
    /// This carries the O(n) similarity scan (bounded by
    /// [`Self::KG_SCAN_LIMIT`]) so callers can run it under a read lock and
    /// keep the write lock ([`Self::apply_memory_node_plan`]) short.
    pub async fn plan_memory_node_update(&self, memory: &MemoryEntry) -> Result<MemoryNodePlan> {
        if let Some(existing_node_id) = self.memory_to_node.get(&memory.key) {
            return Ok(MemoryNodePlan::UpdateExisting(*existing_node_id));
        }
        if let Some(similar_node_id) = self.find_similar_node(memory).await? {
            return Ok(MemoryNodePlan::MergeWith(similar_node_id));
        }
        Ok(MemoryNodePlan::CreateNew)
    }

    /// Apply a plan produced by [`Self::plan_memory_node_update`]. The plan
    /// is revalidated against the current graph (it may be stale if the graph
    /// changed between the read and write locks); a stale plan degrades to
    /// creating a new node. Returns the node id and whether relationship
    /// detection ([`Self::detect_relationship_edges`]) still needs to run for
    /// a freshly created node.
    pub async fn apply_memory_node_plan(
        &mut self,
        plan: MemoryNodePlan,
        memory: &MemoryEntry,
    ) -> Result<(Uuid, bool)> {
        // Revalidate: if the memory got a node since planning, update it.
        if let Some(existing_node_id) = self.memory_to_node.get(&memory.key).copied() {
            let node_id = self.update_existing_node(existing_node_id, memory).await?;
            return Ok((node_id, false));
        }
        if let MemoryNodePlan::MergeWith(similar_node_id) = plan {
            if self.graph.get_node(similar_node_id).await?.is_some() {
                let node_id = self
                    .merge_with_existing_node(similar_node_id, memory)
                    .await?;
                return Ok((node_id, false));
            }
        }
        let node_id = self.create_node_without_detection(memory).await?;
        Ok((node_id, true))
    }

    /// Create a relationship between two memory entries
    #[tracing::instrument(skip(self, properties), fields(
        from_memory = %from_memory_key,
        to_memory = %to_memory_key,
        relationship_type = ?relationship_type
    ))]
    pub async fn create_relationship(
        &mut self,
        from_memory_key: &str,
        to_memory_key: &str,
        relationship_type: RelationshipType,
        properties: Option<HashMap<String, String>>,
    ) -> Result<Uuid> {
        tracing::debug!("Creating relationship between memories");

        let from_node_id =
            self.memory_to_node
                .get(from_memory_key)
                .ok_or_else(|| MemoryError::NotFound {
                    key: from_memory_key.to_string(),
                })?;

        let to_node_id =
            self.memory_to_node
                .get(to_memory_key)
                .ok_or_else(|| MemoryError::NotFound {
                    key: to_memory_key.to_string(),
                })?;

        tracing::debug!("Found nodes: {} -> {}", from_node_id, to_node_id);

        let edge = Edge::new(*from_node_id, *to_node_id, relationship_type, properties);
        let edge_id = self.graph.add_edge(edge).await?;

        tracing::debug!("Created relationship edge: {}", edge_id);
        Ok(edge_id)
    }

    /// Find related memories using graph traversal
    pub async fn find_related_memories(
        &self,
        memory_key: &str,
        max_depth: usize,
        relationship_types: Option<Vec<RelationshipType>>,
    ) -> Result<Vec<RelatedMemory>> {
        let node_id = self
            .memory_to_node
            .get(memory_key)
            .ok_or_else(|| MemoryError::NotFound {
                key: memory_key.to_string(),
            })?;

        let traversal_options = TraversalOptions {
            max_depth,
            relationship_types,
            direction: query::TraversalDirection::Both,
            ..Default::default()
        };

        let related_nodes = self
            .graph
            .traverse_from_node(*node_id, traversal_options)
            .await?;

        let mut related_memories = Vec::new();
        for (node_id, path) in related_nodes {
            if let Some(memory_key) = self.node_to_memory.get(&node_id) {
                let relationship_strength = self.calculate_relationship_strength(&path);
                let related_memory = RelatedMemory {
                    memory_key: memory_key.clone(),
                    node_id,
                    path,
                    relationship_strength,
                };
                related_memories.push(related_memory);
            }
        }

        // Sort by relationship strength
        related_memories.sort_by(|a, b| {
            b.relationship_strength
                .partial_cmp(&a.relationship_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(related_memories)
    }

    /// Query the knowledge graph using graph patterns
    pub async fn query_graph(&self, query: GraphQuery) -> Result<Vec<QueryResult>> {
        self.graph.execute_query(query).await
    }

    /// Find shortest path between two memories
    pub async fn find_path_between_memories(
        &self,
        from_memory: &str,
        to_memory: &str,
        max_depth: Option<usize>,
    ) -> Result<Option<GraphPath>> {
        let from_node =
            self.memory_to_node
                .get(from_memory)
                .ok_or_else(|| MemoryError::NotFound {
                    key: from_memory.to_string(),
                })?;

        let to_node = self
            .memory_to_node
            .get(to_memory)
            .ok_or_else(|| MemoryError::NotFound {
                key: to_memory.to_string(),
            })?;

        self.graph
            .find_shortest_path(*from_node, *to_node, max_depth)
            .await
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        self.graph.get_stats()
    }

    /// Perform inference to discover new relationships
    pub async fn infer_relationships(&mut self) -> Result<Vec<reasoning::InferenceResult>> {
        self.reasoner.infer_relationships(&self.graph).await
    }

    /// Get all memories connected to a specific concept or entity
    pub async fn get_memories_by_concept(&self, concept: &str) -> Result<Vec<String>> {
        let query = GraphQueryBuilder::new()
            .match_nodes_with_property("concept", concept)
            .build();

        let results = self.query_graph(query).await?;

        let mut memory_keys = Vec::new();
        for result in results {
            for node_id in result.nodes {
                if let Some(memory_key) = self.node_to_memory.get(&node_id) {
                    memory_keys.push(memory_key.clone());
                }
            }
        }

        Ok(memory_keys)
    }

    /// Get the node ID for a given memory key
    pub async fn get_node_for_memory(&self, memory_key: &str) -> Result<Option<Uuid>> {
        Ok(self.memory_to_node.get(memory_key).copied())
    }

    /// Bi-temporal validity of the node backing `memory_key` at instant `at`
    /// (event time and system time, via [`Node::is_valid_at`]). Returns
    /// `Ok(None)` when no graph node backs the memory key.
    pub async fn memory_node_valid_at(
        &self,
        memory_key: &str,
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<Option<bool>> {
        let Some(node_id) = self.memory_to_node.get(memory_key).copied() else {
            return Ok(None);
        };
        Ok(self
            .graph
            .get_node(node_id)
            .await?
            .map(|node| node.is_valid_at(at)))
    }

    /// Get the memory key for a given node ID
    pub async fn get_memory_for_node(&self, node_id: Uuid) -> Result<Option<String>> {
        Ok(self.node_to_memory.get(&node_id).cloned())
    }

    /// Get all connected nodes with their relationship types and strengths
    pub async fn get_connected_nodes(
        &self,
        node_id: Uuid,
    ) -> Result<Vec<(Uuid, RelationshipType, f64)>> {
        self.graph.get_connected_nodes(node_id).await
    }

    /// Cap on the nodes examined by per-store similarity/relationship
    /// scans. Bounds the write-path cost per store: at most this many
    /// deterministically-ordered candidates are examined.
    const KG_SCAN_LIMIT: usize = 256;

    /// Recency key for deterministic ordering: newest first. Uses
    /// `last_modified`, falling back to `created_at`.
    fn node_recency(node: &Node) -> chrono::DateTime<chrono::Utc> {
        node.last_modified
            .or(node.created_at)
            .unwrap_or(node.ingested_at)
    }

    /// Lowercased alphanumeric token set used for content-relevance scoring.
    fn scan_tokens(text: &str) -> std::collections::HashSet<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Memory-backed candidate nodes (excluding `exclude_key`), ordered
    /// DETERMINISTICALLY by content relevance to `query_text` (token-overlap
    /// desc), then recency (newest first), then node id, and capped at
    /// [`Self::KG_SCAN_LIMIT`]. Unlike iterating an unordered `HashMap`/
    /// `DashMap`, this keeps the examined candidate set reproducible run-to-run
    /// AND biased toward the memories most likely to match.
    async fn relevance_ordered_candidates(
        &self,
        exclude_key: &str,
        query_text: &str,
    ) -> Result<Vec<(Uuid, Node)>> {
        let query_tokens = Self::scan_tokens(query_text);
        let mut scored: Vec<(usize, chrono::DateTime<chrono::Utc>, Uuid, Node)> = Vec::new();
        for (mem_key, node_id) in &self.memory_to_node {
            if mem_key == exclude_key {
                continue;
            }
            if let Some(node) = self.graph.get_node(*node_id).await? {
                let overlap = node
                    .description
                    .as_deref()
                    .map(|d| {
                        let toks = Self::scan_tokens(d);
                        query_tokens.intersection(&toks).count()
                    })
                    .unwrap_or(0);
                let recency = Self::node_recency(&node);
                scored.push((overlap, recency, *node_id, node));
            }
        }
        // Overlap desc, then recency desc, then id asc (fully deterministic).
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        scored.truncate(Self::KG_SCAN_LIMIT);
        Ok(scored
            .into_iter()
            .map(|(_, _, id, node)| (id, node))
            .collect())
    }

    /// Memory-backed candidate nodes (excluding `exclude_key`), ordered
    /// DETERMINISTICALLY by recency (newest first) then node id — an explicit
    /// "most-recent-N nodes" window — capped at [`Self::KG_SCAN_LIMIT`]. Used
    /// for temporal detection, which is about creation time rather than
    /// content.
    async fn recency_ordered_candidates(&self, exclude_key: &str) -> Result<Vec<(Uuid, Node)>> {
        let mut nodes: Vec<(chrono::DateTime<chrono::Utc>, Uuid, Node)> = Vec::new();
        for (mem_key, node_id) in &self.memory_to_node {
            if mem_key == exclude_key {
                continue;
            }
            if let Some(node) = self.graph.get_node(*node_id).await? {
                let recency = Self::node_recency(&node);
                nodes.push((recency, *node_id, node));
            }
        }
        nodes.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        nodes.truncate(Self::KG_SCAN_LIMIT);
        Ok(nodes.into_iter().map(|(_, id, node)| (id, node)).collect())
    }

    /// Auto-detect relationships based on content similarity and metadata
    async fn auto_detect_relationships(
        &mut self,
        node_id: Uuid,
        memory: &MemoryEntry,
    ) -> Result<()> {
        let edges = self.detect_relationship_edges(node_id, memory).await?;
        self.add_detected_edges(edges).await
    }

    /// Read-only relationship detection for a node backing `memory`: computes
    /// tag-, temporal- and semantic-relationship edges WITHOUT mutating the
    /// graph, so callers can run it under a read lock and apply the returned
    /// edges under a short write lock ([`Self::add_detected_edges`]).
    /// Temporal/semantic candidate scans are capped at
    /// [`Self::KG_SCAN_LIMIT`] nodes.
    pub async fn detect_relationship_edges(
        &self,
        node_id: Uuid,
        memory: &MemoryEntry,
    ) -> Result<Vec<Edge>> {
        let mut edges = Vec::new();

        // Find similar memories based on tags
        for tag in &memory.metadata.tags {
            let similar_memories = self.get_memories_by_concept(tag).await?;
            for similar_memory_key in similar_memories {
                if similar_memory_key != memory.key {
                    if let Some(similar_node_id) = self.memory_to_node.get(&similar_memory_key) {
                        // Create a "related_to" relationship
                        edges.push(Edge::new(
                            node_id,
                            *similar_node_id,
                            RelationshipType::RelatedTo,
                            Some(HashMap::from([
                                ("reason".to_string(), "shared_tag".to_string()),
                                ("tag".to_string(), tag.clone()),
                            ])),
                        ));
                    }
                }
            }
        }

        // Detect temporal relationships over a deterministic most-recent-N
        // candidate window (creation-time based, so recency ordering is the
        // relevant one).
        let time_window = chrono::Duration::hours(1);
        let memory_time = memory.created_at();
        for (other_node_id, other_node) in self.recency_ordered_candidates(&memory.key).await? {
            if let Some(other_time) = other_node.created_at {
                let time_diff = (memory_time - other_time).abs();
                if time_diff <= time_window {
                    edges.push(Edge::new(
                        node_id,
                        other_node_id,
                        RelationshipType::TemporallyRelated,
                        Some(HashMap::from([(
                            "time_diff_minutes".to_string(),
                            time_diff.num_minutes().to_string(),
                        )])),
                    ));
                }
            }
        }

        // Detect semantic relationships (if embeddings are available) over a
        // deterministic content-relevance-ordered candidate window.
        if let Some(embedding) = &memory.embedding {
            let similarity_threshold = self.config.semantic_similarity_threshold;
            for (other_node_id, other_node) in self
                .relevance_ordered_candidates(&memory.key, &memory.value)
                .await?
            {
                if let Some(other_embedding) = &other_node.embedding {
                    let similarity = cosine_similarity(embedding, other_embedding);
                    if similarity >= similarity_threshold {
                        edges.push(Edge::new(
                            node_id,
                            other_node_id,
                            RelationshipType::SemanticallyRelated,
                            Some(HashMap::from([(
                                "similarity_score".to_string(),
                                similarity.to_string(),
                            )])),
                        ));
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Apply edges computed by [`Self::detect_relationship_edges`]. Kept
    /// separate so the O(n) detection can run outside the write lock and only
    /// this mutation needs `&mut self`. Individual edge failures are ignored
    /// (best-effort), matching the prior auto-detection behavior.
    pub async fn add_detected_edges(&mut self, edges: Vec<Edge>) -> Result<()> {
        for edge in edges {
            let _ = self.graph.add_edge(edge).await;
        }
        Ok(())
    }

    /// Update an existing node with new memory data
    #[tracing::instrument(skip(self, memory), fields(memory_key = %memory.key, node_id = %node_id))]
    async fn update_existing_node(&mut self, node_id: Uuid, memory: &MemoryEntry) -> Result<Uuid> {
        if let Some(mut node) = self.graph.get_node(node_id).await? {
            // Track changes for differential analysis
            let old_content = node.description.clone().unwrap_or_default();
            let new_content = memory.value.clone();

            // Update node with new information
            node.description = Some(memory.value.clone());
            node.last_modified = Some(chrono::Utc::now());
            node.importance = memory.metadata.importance;
            node.confidence = memory.metadata.confidence;

            // Merge tags (keep existing + add new)
            let mut updated_tags = node.tags.clone();
            for tag in &memory.metadata.tags {
                if !updated_tags.contains(tag) {
                    updated_tags.push(tag.clone());
                }
            }
            node.tags = updated_tags;

            // Update properties with new custom fields
            for (key, value) in &memory.metadata.custom_fields {
                node.properties.insert(key.clone(), value.clone());
            }

            // Update embedding if available
            if let Some(embedding) = &memory.embedding {
                node.embedding = Some(embedding.clone());
            }

            // Store the updated node back
            self.graph.nodes.insert(node_id, node);

            // Update relationships based on content changes
            self.update_relationships_for_changed_node(node_id, &old_content, &new_content)
                .await?;

            tracing::debug!(
                node_id = %node_id,
                memory_key = %memory.key,
                "Updated existing node for memory"
            );
        }

        Ok(node_id)
    }

    /// Find a similar existing node that could be merged
    #[tracing::instrument(skip(self, memory), fields(memory_key = %memory.key))]
    async fn find_similar_node(&self, memory: &MemoryEntry) -> Result<Option<Uuid>> {
        let similarity_threshold = self.config.semantic_similarity_threshold;
        let mut best_match: Option<(Uuid, f64)> = None;

        for (node_id, node) in self.relevance_ordered_graph_nodes(&memory.value) {
            // Skip if this is already mapped to a different memory
            if let Some(existing_memory_key) = self.node_to_memory.get(&node_id) {
                if existing_memory_key != &memory.key {
                    continue;
                }
            }

            // Calculate similarity
            let similarity = self.calculate_node_memory_similarity(&node, memory).await?;

            if similarity >= similarity_threshold {
                if let Some((_, best_similarity)) = best_match {
                    if similarity > best_similarity {
                        best_match = Some((node_id, similarity));
                    }
                } else {
                    best_match = Some((node_id, similarity));
                }
            }
        }

        Ok(best_match.map(|(node_id, _)| node_id))
    }

    /// All graph nodes ordered DETERMINISTICALLY by content relevance to
    /// `query_text` (token-overlap of the node's description/label desc), then
    /// recency (newest first), then id, capped at [`Self::KG_SCAN_LIMIT`].
    /// Includes non-memory (entity) nodes, so `find_similar_node` keeps its
    /// original candidate universe but examines a reproducible, relevant
    /// subset instead of an arbitrary map-iteration-order prefix.
    fn relevance_ordered_graph_nodes(&self, query_text: &str) -> Vec<(Uuid, Node)> {
        let query_tokens = Self::scan_tokens(query_text);
        let mut scored: Vec<(usize, chrono::DateTime<chrono::Utc>, Uuid, Node)> = self
            .graph
            .nodes
            .iter()
            .map(|node_ref| {
                let node = node_ref.value().clone();
                let text = node.description.as_deref().unwrap_or(&node.label);
                let overlap = query_tokens.intersection(&Self::scan_tokens(text)).count();
                let recency = Self::node_recency(&node);
                (overlap, recency, node.id, node)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        scored.truncate(Self::KG_SCAN_LIMIT);
        scored
            .into_iter()
            .map(|(_, _, id, node)| (id, node))
            .collect()
    }

    /// Test-only: the deterministic candidate node ordering
    /// `find_similar_node` will scan for `memory`, as (node_id) in order.
    /// Exposed so tests can prove the candidate set is reproducible and
    /// relevance-ordered past [`Self::KG_SCAN_LIMIT`].
    #[cfg(feature = "test-utils")]
    pub fn debug_similar_candidate_ids(&self, memory: &MemoryEntry) -> Vec<Uuid> {
        self.relevance_ordered_graph_nodes(&memory.value)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Merge memory with an existing similar node
    #[tracing::instrument(skip(self, memory), fields(memory_key = %memory.key, node_id = %node_id))]
    async fn merge_with_existing_node(
        &mut self,
        node_id: Uuid,
        memory: &MemoryEntry,
    ) -> Result<Uuid> {
        if let Some(mut node) = self.graph.get_node(node_id).await? {
            tracing::info!(
                memory_key = %memory.key,
                node_id = %node_id,
                "Merging memory with existing node"
            );

            // Combine content intelligently
            let combined_content = self
                .merge_content(&node.description.unwrap_or_default(), &memory.value)
                .await?;

            node.description = Some(combined_content);
            node.last_modified = Some(chrono::Utc::now());

            // Update importance and confidence (weighted average)
            let old_weight = 0.7; // Give more weight to existing data
            let new_weight = 0.3;
            node.importance =
                node.importance * old_weight + memory.metadata.importance * new_weight;
            node.confidence =
                node.confidence * old_weight + memory.metadata.confidence * new_weight;

            // Merge tags
            for tag in &memory.metadata.tags {
                if !node.tags.contains(tag) {
                    node.tags.push(tag.clone());
                }
            }

            // Merge properties
            for (key, value) in &memory.metadata.custom_fields {
                node.properties.insert(key.clone(), value.clone());
            }

            // Update embedding if the new one is available
            if memory.embedding.is_some() {
                node.embedding = memory.embedding.clone();
            }

            // Store updated node
            self.graph.nodes.insert(node_id, node);

            // Update mappings
            self.memory_to_node.insert(memory.key.clone(), node_id);
            self.node_to_memory.insert(node_id, memory.key.clone());
        }

        Ok(node_id)
    }

    /// Create a completely new node
    async fn create_new_node(&mut self, memory: &MemoryEntry) -> Result<Uuid> {
        let node_id = self.create_node_without_detection(memory).await?;

        // Auto-detect relationships with existing nodes
        self.auto_detect_relationships(node_id, memory).await?;

        Ok(node_id)
    }

    /// Create a new node and mappings WITHOUT relationship auto-detection,
    /// so detection can run separately outside the write lock.
    async fn create_node_without_detection(&mut self, memory: &MemoryEntry) -> Result<Uuid> {
        let node = Node::from_memory(memory);
        let node_id = self.graph.add_node(node).await?;

        // Update mappings
        self.memory_to_node.insert(memory.key.clone(), node_id);
        self.node_to_memory.insert(node_id, memory.key.clone());

        tracing::info!(
            node_id = %node_id,
            memory_key = %memory.key,
            "Created new node for memory"
        );
        Ok(node_id)
    }

    /// Find all memories within a specified graph distance from a reference memory
    pub async fn find_memories_within_distance(
        &self,
        reference_memory_key: &str,
        max_distance: usize,
    ) -> Result<Vec<(String, usize)>> {
        // Get the node ID for the reference memory
        let reference_node_id = self
            .memory_to_node
            .get(reference_memory_key)
            .ok_or_else(|| MemoryError::NotFound {
                key: reference_memory_key.to_string(),
            })?;

        // Use graph traversal to find all reachable nodes within max_distance
        let traversal_options = crate::memory::knowledge_graph::query::TraversalOptions {
            max_depth: max_distance,
            direction: crate::memory::knowledge_graph::query::TraversalDirection::Both,
            relationship_types: None, // Include all relationship types
            allow_cycles: false,
            sort_by_distance: true,
            limit: None,
            min_strength: None,
            min_confidence: None,
        };

        let reachable_nodes = self
            .graph
            .traverse_from_node(*reference_node_id, traversal_options)
            .await?;

        // Convert node IDs back to memory keys with distances
        let mut result = Vec::new();
        for (node_id, path) in reachable_nodes {
            if let Some(memory_key) = self.node_to_memory.get(&node_id) {
                let distance = path.edges.len(); // Path length as distance
                if distance <= max_distance {
                    result.push((memory_key.clone(), distance));
                }
            }
        }

        // Sort by distance
        result.sort_by_key(|(_, distance)| *distance);
        Ok(result)
    }

    /// Calculate relationship strength based on path
    fn calculate_relationship_strength(&self, path: &GraphPath) -> f64 {
        if path.edges.is_empty() {
            return 0.0;
        }

        let mut strength = 1.0;

        // Decay strength with distance
        let distance_decay = 0.8_f64.powi(path.edges.len() as i32);
        strength *= distance_decay;

        // Boost strength based on relationship types
        for edge_id in &path.edges {
            if let Some(edge) = self.graph.edges.get(edge_id) {
                let type_weight = match edge.relationship.relationship_type {
                    RelationshipType::CausedBy | RelationshipType::Causes => 1.0,
                    RelationshipType::PartOf | RelationshipType::Contains => 0.9,
                    RelationshipType::SemanticallyRelated => 0.8,
                    RelationshipType::RelatedTo => 0.7,
                    RelationshipType::TemporallyRelated => 0.6,
                    RelationshipType::References => 0.6,
                    RelationshipType::DependsOn => 0.8,
                    RelationshipType::SimilarTo => 0.7,
                    RelationshipType::Contradicts => 0.3,
                    RelationshipType::Custom(_) => 0.5,
                };
                strength *= type_weight;
            }
        }

        strength.clamp(0.0, 1.0)
    }

    /// Calculate similarity between a node and a memory entry
    async fn calculate_node_memory_similarity(
        &self,
        node: &Node,
        memory: &MemoryEntry,
    ) -> Result<f64> {
        let mut similarity_score = 0.0;
        let mut factor_count = 0;

        // Content similarity
        if let Some(node_content) = &node.description {
            let content_similarity = self.calculate_text_similarity(node_content, &memory.value);
            similarity_score += content_similarity;
            factor_count += 1;
        }

        // Tag similarity
        let node_tags: std::collections::HashSet<_> = node.tags.iter().collect();
        let memory_tags: std::collections::HashSet<_> = memory.metadata.tags.iter().collect();

        if !node_tags.is_empty() || !memory_tags.is_empty() {
            let intersection = node_tags.intersection(&memory_tags).count();
            let union = node_tags.union(&memory_tags).count();
            let tag_similarity = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };
            similarity_score += tag_similarity;
            factor_count += 1;
        }

        // Embedding similarity (if both have embeddings)
        if let (Some(node_embedding), Some(memory_embedding)) = (&node.embedding, &memory.embedding)
        {
            let embedding_similarity = cosine_similarity(node_embedding, memory_embedding);
            similarity_score += embedding_similarity;
            factor_count += 1;
        }

        // Key similarity (for exact matches or very similar keys)
        if let Some(memory_key) = self.node_to_memory.get(&node.id) {
            let key_similarity = self.calculate_text_similarity(memory_key, &memory.key);
            if key_similarity > 0.8 {
                similarity_score += key_similarity;
                factor_count += 1;
            }
        }

        Ok(if factor_count > 0 {
            similarity_score / factor_count as f64
        } else {
            0.0
        })
    }

    /// Calculate text similarity between two strings
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        if text1 == text2 {
            return 1.0;
        }

        if text1.is_empty() || text2.is_empty() {
            return 0.0;
        }

        // Use Jaccard similarity on word level
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        let words1: std::collections::HashSet<&str> = text1_lower.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Intelligently merge two pieces of content
    async fn merge_content(&self, existing_content: &str, new_content: &str) -> Result<String> {
        // If content is identical, return as-is
        if existing_content == new_content {
            return Ok(existing_content.to_string());
        }

        // If one is empty, return the other
        if existing_content.is_empty() {
            return Ok(new_content.to_string());
        }
        if new_content.is_empty() {
            return Ok(existing_content.to_string());
        }

        // Check if new content is an extension of existing content
        if new_content.contains(existing_content) {
            return Ok(new_content.to_string());
        }
        if existing_content.contains(new_content) {
            return Ok(existing_content.to_string());
        }

        // Check similarity to decide merge strategy
        let similarity = self.calculate_text_similarity(existing_content, new_content);

        if similarity > 0.7 {
            // High similarity: merge by combining unique information
            self.merge_similar_content(existing_content, new_content)
                .await
        } else {
            // Low similarity: create a structured combination
            Ok(format!(
                "{}\n\n--- Updated Information ---\n{}",
                existing_content, new_content
            ))
        }
    }

    /// Merge similar content by combining unique information (optimized)
    async fn merge_similar_content(&self, existing: &str, new: &str) -> Result<String> {
        // Optimized approach: use iterators and avoid unnecessary allocations
        let existing_sentences: Vec<&str> = existing
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let new_sentences: Vec<&str> = new
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        // Pre-allocate with estimated capacity to reduce reallocations
        let mut merged_sentences =
            Vec::with_capacity(existing_sentences.len() + new_sentences.len());
        merged_sentences.extend_from_slice(&existing_sentences);

        // Use iterator approach to avoid nested loops where possible
        for new_sentence in new_sentences {
            let is_duplicate = existing_sentences.iter().any(|&existing_sentence| {
                self.calculate_text_similarity(existing_sentence, new_sentence) > 0.8
            });

            if !is_duplicate {
                merged_sentences.push(new_sentence);
            }
        }

        Ok(merged_sentences.join(". ") + ".")
    }

    /// Update relationships when a node's content changes
    async fn update_relationships_for_changed_node(
        &mut self,
        node_id: Uuid,
        old_content: &str,
        new_content: &str,
    ) -> Result<()> {
        // Calculate content change significance
        let content_similarity = self.calculate_text_similarity(old_content, new_content);

        // If content changed significantly, re-evaluate relationships
        if content_similarity < 0.7 {
            // Remove weak relationships that may no longer be valid
            self.cleanup_weak_relationships(node_id).await?;

            // Re-detect relationships based on new content
            if let Some(memory_key) = self.node_to_memory.get(&node_id).cloned() {
                // Create a temporary memory entry for relationship detection
                let temp_memory = crate::memory::types::MemoryEntry::new(
                    memory_key,
                    new_content.to_string(),
                    crate::memory::types::MemoryType::LongTerm,
                );
                self.auto_detect_relationships(node_id, &temp_memory)
                    .await?;
            }
        } else {
            // Content is similar, just update relationship strengths
            self.update_relationship_strengths(node_id).await?;
        }

        Ok(())
    }

    /// Remove weak relationships that may no longer be valid
    async fn cleanup_weak_relationships(&mut self, node_id: Uuid) -> Result<()> {
        let weak_threshold = 0.3;
        let mut edges_to_remove = Vec::new();

        // Find edges connected to this node with low strength
        for edge_ref in self.graph.edges.iter() {
            let edge = edge_ref.value();
            if (edge.from_node == node_id || edge.to_node == node_id)
                && edge.relationship.strength < weak_threshold
            {
                edges_to_remove.push(edge.id);
            }
        }

        // Remove weak edges
        for edge_id in edges_to_remove {
            self.graph.remove_edge(edge_id).await?;
        }

        Ok(())
    }

    /// Update relationship strengths based on current node state
    async fn update_relationship_strengths(&mut self, node_id: Uuid) -> Result<()> {
        // Get all edges connected to this node
        let connected_edges: Vec<_> = self
            .graph
            .edges
            .iter()
            .filter(|edge_ref| {
                let edge = edge_ref.value();
                edge.from_node == node_id || edge.to_node == node_id
            })
            .map(|edge_ref| *edge_ref.key())
            .collect();

        // Update strength for each edge
        for edge_id in connected_edges {
            if let Some(mut edge) = self.graph.get_edge(edge_id).await? {
                // Recalculate relationship strength based on current node states
                let new_strength = self.calculate_edge_strength(&edge).await?;
                edge.relationship.set_strength(new_strength);
                edge.relationship.mark_modified();

                // Update the edge in the graph
                self.graph.edges.insert(edge_id, edge);
            }
        }

        Ok(())
    }

    /// Calculate the strength of an edge based on current node states
    async fn calculate_edge_strength(&self, edge: &Edge) -> Result<f64> {
        let from_node = self.graph.get_node(edge.from_node).await?;
        let to_node = self.graph.get_node(edge.to_node).await?;

        if let (Some(from_node), Some(to_node)) = (from_node, to_node) {
            // Calculate strength based on node similarity and relationship type
            let mut strength = 0.5; // Base strength

            // Content similarity factor
            if let (Some(from_content), Some(to_content)) =
                (&from_node.description, &to_node.description)
            {
                let content_similarity = self.calculate_text_similarity(from_content, to_content);
                strength += content_similarity * 0.3;
            }

            // Tag overlap factor
            let from_tags: std::collections::HashSet<_> = from_node.tags.iter().collect();
            let to_tags: std::collections::HashSet<_> = to_node.tags.iter().collect();
            let tag_overlap = if !from_tags.is_empty() && !to_tags.is_empty() {
                from_tags.intersection(&to_tags).count() as f64
                    / from_tags.union(&to_tags).count() as f64
            } else {
                0.0
            };
            strength += tag_overlap * 0.2;

            // Relationship type factor
            let type_factor = match edge.relationship.relationship_type {
                RelationshipType::Causes | RelationshipType::CausedBy => 0.9,
                RelationshipType::PartOf | RelationshipType::Contains => 0.8,
                RelationshipType::SemanticallyRelated => 0.7,
                _ => 0.5,
            };
            strength *= type_factor;

            Ok(strength.clamp(0.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
}

/// A memory and its relationship information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedMemory {
    pub memory_key: String,
    pub node_id: Uuid,
    pub path: GraphPath,
    pub relationship_strength: f64,
}

/// An inferred relationship discovered by the reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredRelationship {
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub relationship_type: RelationshipType,
    pub confidence: f64,
    pub reasoning: String,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b)) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!(similarity.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![-1.0, -2.0, -3.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![0.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let vec1: Vec<f32> = vec![];
        let vec2: Vec<f32> = vec![];
        let similarity = cosine_similarity(&vec1, &vec2);
        // Empty vectors should have zero similarity
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_partial_match() {
        let vec1 = vec![1.0, 2.0, 0.0];
        let vec2 = vec![1.0, 0.0, 2.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        // Should have some similarity but not identical
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_memory_knowledge_graph_new() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        // Graph should be initialized correctly
        assert_eq!(graph.memory_to_node.len(), 0);
        assert_eq!(graph.node_to_memory.len(), 0);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_text_similarity_identical() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let text1 = "The quick brown fox";
        let text2 = "The quick brown fox";
        let similarity = graph.calculate_text_similarity(text1, text2);

        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_text_similarity_empty() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let similarity = graph.calculate_text_similarity("", "test");
        assert_eq!(similarity, 0.0);

        let similarity = graph.calculate_text_similarity("test", "");
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_text_similarity_partial() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let text1 = "The quick brown fox";
        let text2 = "The quick red fox";
        let similarity = graph.calculate_text_similarity(text1, text2);

        // Should have high similarity due to shared words
        assert!(similarity > 0.5);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_text_similarity_case_insensitive() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let text1 = "THE QUICK BROWN FOX";
        let text2 = "the quick brown fox";
        let similarity = graph.calculate_text_similarity(text1, text2);

        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_text_similarity_no_overlap() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let text1 = "apple orange banana";
        let text2 = "car truck bus";
        let similarity = graph.calculate_text_similarity(text1, text2);

        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_memory_knowledge_graph_calculate_relationship_strength_empty_path() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let empty_path = GraphPath {
            nodes: vec![],
            edges: vec![],
            length: 0,
            weight: 0.0,
        };

        let strength = graph.calculate_relationship_strength(&empty_path);
        assert_eq!(strength, 0.0);
    }

    #[test]
    fn test_related_memory_clone() {
        let related_memory = RelatedMemory {
            memory_key: "test_key".to_string(),
            node_id: Uuid::new_v4(),
            path: GraphPath {
                nodes: vec![],
                edges: vec![],
                length: 0,
                weight: 0.0,
            },
            relationship_strength: 0.8,
        };

        let cloned = related_memory.clone();
        assert_eq!(related_memory.memory_key, cloned.memory_key);
        assert_eq!(related_memory.node_id, cloned.node_id);
        assert_eq!(
            related_memory.relationship_strength,
            cloned.relationship_strength
        );
    }

    #[test]
    fn test_inferred_relationship_clone() {
        let inferred = InferredRelationship {
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            relationship_type: RelationshipType::RelatedTo,
            confidence: 0.75,
            reasoning: "Similar content".to_string(),
        };

        let cloned = inferred.clone();
        assert_eq!(inferred.from_node, cloned.from_node);
        assert_eq!(inferred.to_node, cloned.to_node);
        assert_eq!(inferred.confidence, cloned.confidence);
        assert_eq!(inferred.reasoning, cloned.reasoning);
    }

    #[tokio::test]
    async fn test_memory_knowledge_graph_get_node_for_memory_empty() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let result = graph.get_node_for_memory("nonexistent").await;
        assert!(result.is_ok());
        assert!(result
            .expect("query on empty graph returns Ok(None), not an error")
            .is_none());
    }

    #[tokio::test]
    async fn test_memory_knowledge_graph_get_memory_for_node_empty() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let result = graph.get_memory_for_node(Uuid::new_v4()).await;
        assert!(result.is_ok());
        assert!(result
            .expect("query on empty graph returns Ok(None), not an error")
            .is_none());
    }

    #[test]
    fn test_memory_knowledge_graph_get_stats() {
        let config = GraphConfig::default();
        let graph = MemoryKnowledgeGraph::new(config);

        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
    }

    #[test]
    fn test_cosine_similarity_normalized_vectors() {
        // Unit vectors should have similarity equal to their dot product
        let vec1 = vec![0.6, 0.8];
        let vec2 = vec![0.8, 0.6];
        let similarity = cosine_similarity(&vec1, &vec2);

        // Expected: (0.6*0.8 + 0.8*0.6) / (sqrt(0.36+0.64) * sqrt(0.64+0.36))
        // = (0.48 + 0.48) / (1.0 * 1.0) = 0.96
        assert!((similarity - 0.96).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_symmetry() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let sim1 = cosine_similarity(&vec1, &vec2);
        let sim2 = cosine_similarity(&vec2, &vec1);

        // Cosine similarity should be symmetric
        assert!((sim1 - sim2).abs() < 0.0001);
    }

    #[test]
    fn test_related_memory_serialization() {
        let related_memory = RelatedMemory {
            memory_key: "test_key".to_string(),
            node_id: Uuid::new_v4(),
            path: GraphPath {
                nodes: vec![],
                edges: vec![],
                length: 0,
                weight: 0.0,
            },
            relationship_strength: 0.8,
        };

        // Test that it can be serialized and deserialized
        let serialized = serde_json::to_string(&related_memory)
            .expect("RelatedMemory derives Serialize with no fallible fields");
        let deserialized: RelatedMemory =
            serde_json::from_str(&serialized).expect("round-trip of value serialized just above");

        assert_eq!(related_memory.memory_key, deserialized.memory_key);
        assert_eq!(related_memory.node_id, deserialized.node_id);
        assert_eq!(
            related_memory.relationship_strength,
            deserialized.relationship_strength
        );
    }

    #[test]
    fn test_inferred_relationship_serialization() {
        let inferred = InferredRelationship {
            from_node: Uuid::new_v4(),
            to_node: Uuid::new_v4(),
            relationship_type: RelationshipType::SemanticallyRelated,
            confidence: 0.85,
            reasoning: "High semantic overlap".to_string(),
        };

        let serialized = serde_json::to_string(&inferred)
            .expect("InferredRelationship derives Serialize with no fallible fields");
        let deserialized: InferredRelationship =
            serde_json::from_str(&serialized).expect("round-trip of value serialized just above");

        assert_eq!(inferred.from_node, deserialized.from_node);
        assert_eq!(inferred.to_node, deserialized.to_node);
        assert_eq!(inferred.confidence, deserialized.confidence);
        assert_eq!(inferred.reasoning, deserialized.reasoning);
    }
}
