//! Core knowledge graph implementation

use super::query::{GraphQuery, QueryResult, TraversalDirection, TraversalOptions};
use super::types::{Edge, GraphPath, KnowledgeGraphMetadata, Node, RelationshipType};
use crate::error::{MemoryError, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Configuration for the knowledge graph
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Maximum number of edges
    pub max_edges: usize,
    /// Enable automatic relationship detection
    pub auto_detect_relationships: bool,
    /// Threshold for semantic similarity relationships
    pub semantic_similarity_threshold: f64,
    /// Maximum traversal depth for queries
    pub max_traversal_depth: usize,
    /// Enable graph persistence
    pub enable_persistence: bool,
    /// Path for graph persistence
    pub persistence_path: Option<String>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100000,
            max_edges: 500000,
            auto_detect_relationships: true,
            semantic_similarity_threshold: 0.7,
            max_traversal_depth: 10,
            enable_persistence: false,
            persistence_path: None,
        }
    }
}

/// Statistics about the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Number of connected components
    pub connected_components: usize,
    /// Average degree (connections per node)
    pub average_degree: f64,
    /// Graph density (actual edges / possible edges)
    pub density: f64,
    /// Most connected node
    pub most_connected_node: Option<Uuid>,
    /// Maximum degree
    pub max_degree: usize,
    /// Graph diameter (longest shortest path)
    pub diameter: Option<usize>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
}

impl Default for GraphStats {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStats {
    pub fn new() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            connected_components: 0,
            average_degree: 0.0,
            density: 0.0,
            most_connected_node: None,
            max_degree: 0,
            diameter: None,
            last_updated: Utc::now(),
        }
    }
}

/// Recency key used for deterministic candidate ordering: `last_modified`,
/// falling back to `created_at`, then `ingested_at` (newest first).
pub(crate) fn node_recency(node: &Node) -> DateTime<Utc> {
    node.last_modified
        .or(node.created_at)
        .unwrap_or(node.ingested_at)
}

/// Lowercased alphanumeric token set used for content-relevance scoring.
pub(crate) fn scan_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// The text a node is indexed/scored by: description, falling back to label.
fn node_scan_text(node: &Node) -> &str {
    node.description.as_deref().unwrap_or(&node.label)
}

/// Core knowledge graph structure
pub struct KnowledgeGraph {
    /// All nodes in the graph
    pub nodes: DashMap<Uuid, Node>,
    /// All edges in the graph
    pub edges: DashMap<Uuid, Edge>,
    /// Adjacency list for efficient traversal (node_id -> set of edge_ids)
    adjacency: DashMap<Uuid, HashSet<Uuid>>,
    /// Reverse adjacency list (node_id -> set of incoming edge_ids)
    reverse_adjacency: DashMap<Uuid, HashSet<Uuid>>,
    /// Graph metadata
    metadata: RwLock<KnowledgeGraphMetadata>,
    /// Configuration
    config: GraphConfig,
    /// Cached statistics
    stats_cache: RwLock<Option<(GraphStats, DateTime<Utc>)>>,
    /// Inverted token index over each node's scan text (description, falling
    /// back to label): token -> node ids. Maintained on every node mutation
    /// (add/update/remove) so candidate selection touches only posting lists
    /// for the query's tokens instead of scoring all nodes.
    token_index: RwLock<HashMap<String, HashSet<Uuid>>>,
    /// Nodes ordered newest-first with id-ascending tie-break:
    /// (Reverse(recency), id). Forward iteration yields the deterministic
    /// recency ordering used for candidate windows without sorting all nodes.
    recency_index: RwLock<std::collections::BTreeSet<(std::cmp::Reverse<DateTime<Utc>>, Uuid)>>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new(config: GraphConfig) -> Self {
        Self {
            nodes: DashMap::new(),
            edges: DashMap::new(),
            adjacency: DashMap::new(),
            reverse_adjacency: DashMap::new(),
            metadata: RwLock::new(KnowledgeGraphMetadata::new()),
            config,
            stats_cache: RwLock::new(None),
            token_index: RwLock::new(HashMap::new()),
            recency_index: RwLock::new(std::collections::BTreeSet::new()),
        }
    }

    /// Add `node` to the token and recency indexes.
    fn index_node(&self, node: &Node) {
        let mut tokens = self.token_index.write();
        for token in scan_tokens(node_scan_text(node)) {
            tokens.entry(token).or_default().insert(node.id);
        }
        drop(tokens);
        self.recency_index
            .write()
            .insert((std::cmp::Reverse(node_recency(node)), node.id));
    }

    /// Remove `node` (with its CURRENT indexed text/recency) from the indexes.
    fn unindex_node(&self, node: &Node) {
        let mut tokens = self.token_index.write();
        for token in scan_tokens(node_scan_text(node)) {
            if let Some(ids) = tokens.get_mut(&token) {
                ids.remove(&node.id);
                if ids.is_empty() {
                    tokens.remove(&token);
                }
            }
        }
        drop(tokens);
        self.recency_index
            .write()
            .remove(&(std::cmp::Reverse(node_recency(node)), node.id));
    }

    /// Insert or replace a node while keeping the token/recency indexes
    /// coherent. All in-place node content mutations must go through this
    /// (instead of writing `self.nodes` directly).
    pub(crate) fn put_node(&self, node: Node) {
        if let Some(old) = self.nodes.get(&node.id).map(|n| n.clone()) {
            self.unindex_node(&old);
        }
        self.index_node(&node);
        self.nodes.insert(node.id, node);
        self.invalidate_stats_cache();
        self.mark_modified();
    }

    /// For each node whose indexed text shares >= 1 token with
    /// `query_tokens`, the number of shared tokens. Cost is proportional to
    /// the total size of the touched posting lists, not the node count.
    pub(crate) fn token_overlap_counts(
        &self,
        query_tokens: &HashSet<String>,
    ) -> HashMap<Uuid, usize> {
        let index = self.token_index.read();
        let mut counts: HashMap<Uuid, usize> = HashMap::new();
        for token in query_tokens {
            if let Some(ids) = index.get(token) {
                for id in ids {
                    *counts.entry(*id).or_insert(0) += 1;
                }
            }
        }
        counts
    }

    /// Up to `limit` node ids in deterministic recency order (newest first,
    /// id-ascending tie-break) for which `keep` returns true. Early-stops at
    /// `limit`, so cost is O(limit + skipped), not O(total nodes) when the
    /// filter accepts most candidates.
    pub(crate) fn recency_ordered_ids<F>(&self, limit: usize, mut keep: F) -> Vec<Uuid>
    where
        F: FnMut(Uuid) -> bool,
    {
        self.recency_index
            .read()
            .iter()
            .map(|(_, id)| *id)
            .filter(|id| keep(*id))
            .take(limit)
            .collect()
    }

    /// Add a node to the graph
    pub async fn add_node(&self, node: Node) -> Result<Uuid> {
        if self.nodes.len() >= self.config.max_nodes {
            return Err(MemoryError::MemoryLimitExceeded {
                limit: self.config.max_nodes,
            });
        }

        let node_id = node.id;
        // Unindex a same-id node BEFORE indexing the new one so shared
        // tokens/recency keys survive the replacement.
        if let Some(old) = self.nodes.get(&node_id).map(|n| n.clone()) {
            self.unindex_node(&old);
        }
        self.index_node(&node);
        self.nodes.insert(node_id, node);
        self.adjacency.insert(node_id, HashSet::new());
        self.reverse_adjacency.insert(node_id, HashSet::new());

        self.invalidate_stats_cache();
        self.mark_modified();

        Ok(node_id)
    }

    /// Add an edge to the graph
    pub async fn add_edge(&self, edge: Edge) -> Result<Uuid> {
        if self.edges.len() >= self.config.max_edges {
            return Err(MemoryError::MemoryLimitExceeded {
                limit: self.config.max_edges,
            });
        }

        // Verify that both nodes exist
        if !self.nodes.contains_key(&edge.from_node) {
            return Err(MemoryError::NotFound {
                key: format!("node_{}", edge.from_node),
            });
        }
        if !self.nodes.contains_key(&edge.to_node) {
            return Err(MemoryError::NotFound {
                key: format!("node_{}", edge.to_node),
            });
        }

        let edge_id = edge.id;

        // Update adjacency lists
        self.adjacency
            .entry(edge.from_node)
            .or_default()
            .insert(edge_id);

        self.reverse_adjacency
            .entry(edge.to_node)
            .or_default()
            .insert(edge_id);

        self.edges.insert(edge_id, edge);

        self.invalidate_stats_cache();
        self.mark_modified();

        Ok(edge_id)
    }

    /// Get a node by ID
    pub async fn get_node(&self, node_id: Uuid) -> Result<Option<Node>> {
        Ok(self.nodes.get(&node_id).map(|node| node.clone()))
    }

    /// Get an edge by ID
    pub async fn get_edge(&self, edge_id: Uuid) -> Result<Option<Edge>> {
        Ok(self.edges.get(&edge_id).map(|edge| edge.clone()))
    }

    /// Remove a node and all its edges
    pub async fn remove_node(&self, node_id: Uuid) -> Result<bool> {
        if let Some((_, _node)) = self.nodes.remove(&node_id) {
            self.unindex_node(&_node);
            // Remove all edges connected to this node
            let mut edges_to_remove = Vec::new();

            // Collect outgoing edges
            if let Some((_, outgoing)) = self.adjacency.remove(&node_id) {
                edges_to_remove.extend(outgoing);
            }

            // Collect incoming edges
            if let Some((_, incoming)) = self.reverse_adjacency.remove(&node_id) {
                edges_to_remove.extend(incoming);
            }

            // Remove the edges
            for edge_id in edges_to_remove {
                self.remove_edge(edge_id).await?;
            }

            self.invalidate_stats_cache();
            self.mark_modified();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove an edge
    pub async fn remove_edge(&self, edge_id: Uuid) -> Result<bool> {
        if let Some((_, edge)) = self.edges.remove(&edge_id) {
            // Update adjacency lists
            if let Some(mut outgoing) = self.adjacency.get_mut(&edge.from_node) {
                outgoing.remove(&edge_id);
            }

            if let Some(mut incoming) = self.reverse_adjacency.get_mut(&edge.to_node) {
                incoming.remove(&edge_id);
            }

            self.invalidate_stats_cache();
            self.mark_modified();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// End an edge's bi-temporal validity at `at`: sets the event-time end
    /// (`valid_to = at`) and the system-time expiry (`expired_at = at`), so
    /// the edge is excluded from as-of queries at or after `at` while
    /// remaining stored in the graph. Returns whether an edge with the given
    /// id was found. A string that does not parse as a UUID matches no edge.
    pub fn invalidate_edge(&self, edge_id: &str, at: DateTime<Utc>) -> Result<bool> {
        let Ok(id) = Uuid::parse_str(edge_id) else {
            return Ok(false);
        };
        if let Some(mut edge) = self.edges.get_mut(&id) {
            edge.invalidate(at);
            edge.expire_at(at);
            self.mark_modified();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Point-in-time neighborhood query: all edges incident to `node_id`
    /// (either direction) that are bi-temporally valid at `at`. A string
    /// that does not parse as a UUID matches no node. Results are sorted by
    /// edge id for deterministic ordering.
    pub fn neighbors_as_of(&self, node_id: &str, at: DateTime<Utc>) -> Vec<Edge> {
        let Ok(id) = Uuid::parse_str(node_id) else {
            return Vec::new();
        };
        let mut edges: Vec<Edge> = self
            .edges
            .iter()
            .filter(|edge| (edge.from_node == id || edge.to_node == id) && edge.is_valid_at(at))
            .map(|edge| edge.clone())
            .collect();
        edges.sort_by_key(|edge| edge.id);
        edges
    }

    /// Get all neighbors of a node
    pub async fn get_neighbors(&self, node_id: Uuid) -> Result<Vec<Uuid>> {
        let mut neighbors = HashSet::new();

        // Add outgoing neighbors
        if let Some(outgoing_edges) = self.adjacency.get(&node_id) {
            for edge_id in outgoing_edges.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    neighbors.insert(edge.to_node);
                }
            }
        }

        // Add incoming neighbors
        if let Some(incoming_edges) = self.reverse_adjacency.get(&node_id) {
            for edge_id in incoming_edges.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    neighbors.insert(edge.from_node);
                }
            }
        }

        Ok(neighbors.into_iter().collect())
    }

    /// Traverse the graph from a starting node
    pub async fn traverse_from_node(
        &self,
        start_node: Uuid,
        options: TraversalOptions,
    ) -> Result<Vec<(Uuid, GraphPath)>> {
        let mut visited = HashSet::new();
        let mut results = Vec::new();
        let mut queue = VecDeque::new();

        // Initialize with start node
        let mut start_path = GraphPath::new();
        start_path.add_step(start_node, None);
        queue.push_back((start_node, start_path, 0));
        visited.insert(start_node);

        while let Some((current_node, current_path, depth)) = queue.pop_front() {
            if depth >= options.max_depth {
                continue;
            }

            // Get edges based on traversal direction
            let edge_ids = match options.direction {
                TraversalDirection::Outgoing => self
                    .adjacency
                    .get(&current_node)
                    .map(|edges| edges.clone())
                    .unwrap_or_default(),
                TraversalDirection::Incoming => self
                    .reverse_adjacency
                    .get(&current_node)
                    .map(|edges| edges.clone())
                    .unwrap_or_default(),
                TraversalDirection::Both => {
                    let mut all_edges = self
                        .adjacency
                        .get(&current_node)
                        .map(|edges| edges.clone())
                        .unwrap_or_default();

                    if let Some(incoming) = self.reverse_adjacency.get(&current_node) {
                        all_edges.extend(incoming.iter());
                    }
                    all_edges
                }
            };

            for edge_id in edge_ids {
                if let Some(edge) = self.edges.get(&edge_id) {
                    // Check relationship type filter
                    if let Some(ref allowed_types) = options.relationship_types {
                        if !allowed_types.contains(&edge.relationship.relationship_type) {
                            continue;
                        }
                    }

                    // Determine next node
                    let next_node = if edge.from_node == current_node {
                        edge.to_node
                    } else {
                        edge.from_node
                    };

                    if !visited.contains(&next_node) || options.allow_cycles {
                        let mut new_path = current_path.clone();
                        new_path.add_step(next_node, Some(edge_id));

                        results.push((next_node, new_path.clone()));

                        if !visited.contains(&next_node) {
                            visited.insert(next_node);
                            queue.push_back((next_node, new_path, depth + 1));
                        }
                    }
                }
            }
        }

        // Sort by path length if requested
        if options.sort_by_distance {
            results.sort_by_key(|(_, path)| path.length);
        }

        // Limit results
        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Find shortest path between two nodes using BFS
    pub async fn find_shortest_path(
        &self,
        from_node: Uuid,
        to_node: Uuid,
        max_depth: Option<usize>,
    ) -> Result<Option<GraphPath>> {
        if from_node == to_node {
            let mut path = GraphPath::new();
            path.add_step(from_node, None);
            return Ok(Some(path));
        }

        let max_depth = max_depth.unwrap_or(self.config.max_traversal_depth);
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<Uuid, (Uuid, Uuid)> = HashMap::new(); // node -> (parent_node, edge_id)

        queue.push_back((from_node, 0));
        visited.insert(from_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Check all neighbors
            let neighbors = self.get_neighbors(current_node).await?;
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);

                    // Find the edge connecting current_node to neighbor
                    if let Some(edge_id) = self.find_edge_between(current_node, neighbor).await? {
                        parent_map.insert(neighbor, (current_node, edge_id));

                        if neighbor == to_node {
                            // Reconstruct path
                            return Ok(Some(self.reconstruct_path(
                                from_node,
                                to_node,
                                &parent_map,
                            )));
                        }

                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Execute a graph query
    pub async fn execute_query(&self, query: GraphQuery) -> Result<Vec<QueryResult>> {
        // This is a simplified implementation
        // A full implementation would include a proper query engine
        let mut results = Vec::new();

        // For now, just return all nodes that match basic criteria
        for node_ref in self.nodes.iter() {
            let node = node_ref.value();
            let mut matches = true;

            // Check node type filter
            if let Some(ref node_type) = query.node_type_filter {
                if &node.node_type != node_type {
                    matches = false;
                }
            }

            // Check property filters
            for (key, value) in &query.property_filters {
                if node.get_property(key) != Some(value) {
                    matches = false;
                    break;
                }
            }

            if matches {
                results.push(QueryResult {
                    nodes: vec![node.id],
                    edges: Vec::new(),
                    paths: Vec::new(),
                    score: 1.0,
                });
            }
        }

        Ok(results)
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        // Check cache first
        {
            let cache = self.stats_cache.read();
            if let Some((stats, cached_at)) = cache.as_ref() {
                let age = Utc::now() - *cached_at;
                if age.num_seconds() < 60 {
                    // 1 minute cache
                    return stats.clone();
                }
            }
        }

        // Calculate fresh statistics
        let stats = self.calculate_stats();

        // Update cache
        {
            let mut cache = self.stats_cache.write();
            *cache = Some((stats.clone(), Utc::now()));
        }

        stats
    }

    /// Get all connected nodes with their relationship types and strengths
    pub async fn get_connected_nodes(
        &self,
        node_id: Uuid,
    ) -> Result<Vec<(Uuid, RelationshipType, f64)>> {
        let mut connected_nodes = Vec::new();

        // Get outgoing edges
        if let Some(edge_ids) = self.adjacency.get(&node_id) {
            for edge_id in edge_ids.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    let relationship_type = edge.relationship.relationship_type.clone();
                    let strength = edge.relationship.strength;
                    connected_nodes.push((edge.to_node, relationship_type, strength));
                }
            }
        }

        // Get incoming edges
        if let Some(edge_ids) = self.reverse_adjacency.get(&node_id) {
            for edge_id in edge_ids.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    let relationship_type = edge.relationship.relationship_type.clone();
                    let strength = edge.relationship.strength;
                    connected_nodes.push((edge.from_node, relationship_type, strength));
                }
            }
        }

        Ok(connected_nodes)
    }

    /// Find an edge between two nodes
    async fn find_edge_between(&self, node1: Uuid, node2: Uuid) -> Result<Option<Uuid>> {
        if let Some(outgoing_edges) = self.adjacency.get(&node1) {
            for edge_id in outgoing_edges.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.to_node == node2 {
                        return Ok(Some(*edge_id));
                    }
                }
            }
        }

        if let Some(outgoing_edges) = self.adjacency.get(&node2) {
            for edge_id in outgoing_edges.iter() {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.to_node == node1 {
                        return Ok(Some(*edge_id));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Reconstruct a path from parent map
    fn reconstruct_path(
        &self,
        start: Uuid,
        end: Uuid,
        parent_map: &HashMap<Uuid, (Uuid, Uuid)>,
    ) -> GraphPath {
        let mut path = GraphPath::new();
        let mut current = end;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Trace back from end to start
        while current != start {
            nodes.push(current);
            if let Some((parent, edge_id)) = parent_map.get(&current) {
                edges.push(*edge_id);
                current = *parent;
            } else {
                break;
            }
        }
        nodes.push(start);

        // Reverse to get correct order
        nodes.reverse();
        edges.reverse();

        path.nodes = nodes;
        path.edges = edges;
        path.length = path.nodes.len();

        path
    }

    /// Calculate graph statistics
    fn calculate_stats(&self) -> GraphStats {
        let mut stats = GraphStats::new();
        stats.node_count = self.nodes.len();
        stats.edge_count = self.edges.len();

        if stats.node_count > 0 {
            stats.average_degree = (stats.edge_count * 2) as f64 / stats.node_count as f64;

            let max_possible_edges = stats.node_count * (stats.node_count - 1) / 2;
            if max_possible_edges > 0 {
                stats.density = stats.edge_count as f64 / max_possible_edges as f64;
            }

            // Find most connected node
            let mut max_degree = 0;
            let mut most_connected = None;

            for node_ref in self.nodes.iter() {
                let node_id = *node_ref.key();
                let outgoing = self
                    .adjacency
                    .get(&node_id)
                    .map(|edges| edges.len())
                    .unwrap_or(0);
                let incoming = self
                    .reverse_adjacency
                    .get(&node_id)
                    .map(|edges| edges.len())
                    .unwrap_or(0);
                let degree = outgoing + incoming;

                if degree > max_degree {
                    max_degree = degree;
                    most_connected = Some(node_id);
                }
            }

            stats.max_degree = max_degree;
            stats.most_connected_node = most_connected;
        }

        stats.last_updated = Utc::now();
        stats
    }

    /// Invalidate the statistics cache
    fn invalidate_stats_cache(&self) {
        let mut cache = self.stats_cache.write();
        *cache = None;
    }

    /// Mark the graph as modified
    fn mark_modified(&self) {
        let mut metadata = self.metadata.write();
        metadata.mark_modified();
    }
}
