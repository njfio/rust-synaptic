//! Graph querying and traversal functionality

use super::types::{NodeType, RelationshipType, GraphPath};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Direction for graph traversal
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraversalDirection {
    /// Follow outgoing edges only
    Outgoing,
    /// Follow incoming edges only
    Incoming,
    /// Follow edges in both directions
    Both,
}

impl Default for TraversalDirection {
    fn default() -> Self {
        Self::Both
    }
}

/// Options for graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalOptions {
    /// Maximum depth to traverse
    pub max_depth: usize,
    /// Direction of traversal
    pub direction: TraversalDirection,
    /// Filter by relationship types
    pub relationship_types: Option<Vec<RelationshipType>>,
    /// Allow cycles in traversal
    pub allow_cycles: bool,
    /// Sort results by distance
    pub sort_by_distance: bool,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Minimum relationship strength
    pub min_strength: Option<f64>,
    /// Minimum relationship confidence
    pub min_confidence: Option<f64>,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: 5,
            direction: TraversalDirection::Both,
            relationship_types: None,
            allow_cycles: false,
            sort_by_distance: true,
            limit: None,
            min_strength: None,
            min_confidence: None,
        }
    }
}

/// A graph query for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQuery {
    /// Node type filter
    pub node_type_filter: Option<NodeType>,
    /// Property filters (key -> value)
    pub property_filters: HashMap<String, String>,
    /// Tag filters
    pub tag_filters: Vec<String>,
    /// Relationship type filters
    pub relationship_filters: Vec<RelationshipType>,
    /// Traversal options
    pub traversal_options: Option<TraversalOptions>,
    /// Return limit
    pub limit: Option<usize>,
    /// Sort criteria
    pub sort_by: Option<QuerySortBy>,
}

impl GraphQuery {
    /// Create a new empty query
    pub fn new() -> Self {
        Self {
            node_type_filter: None,
            property_filters: HashMap::new(),
            tag_filters: Vec::new(),
            relationship_filters: Vec::new(),
            traversal_options: None,
            limit: None,
            sort_by: None,
        }
    }
}

impl Default for GraphQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Sort criteria for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuerySortBy {
    /// Sort by node importance
    Importance,
    /// Sort by node confidence
    Confidence,
    /// Sort by creation time
    CreatedAt,
    /// Sort by last modification time
    LastModified,
    /// Sort by number of connections
    Degree,
    /// Sort by relevance score
    Relevance,
}

/// Builder for constructing graph queries
pub struct GraphQueryBuilder {
    query: GraphQuery,
}

impl GraphQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            query: GraphQuery::new(),
        }
    }

    /// Filter by node type
    pub fn with_node_type(mut self, node_type: NodeType) -> Self {
        self.query.node_type_filter = Some(node_type);
        self
    }

    /// Add a property filter
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.query.property_filters.insert(key, value);
        self
    }

    /// Add multiple property filters
    pub fn with_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.query.property_filters.extend(properties);
        self
    }

    /// Add a tag filter
    pub fn with_tag(mut self, tag: String) -> Self {
        self.query.tag_filters.push(tag);
        self
    }

    /// Add multiple tag filters
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.query.tag_filters.extend(tags);
        self
    }

    /// Add a relationship type filter
    pub fn with_relationship_type(mut self, relationship_type: RelationshipType) -> Self {
        self.query.relationship_filters.push(relationship_type);
        self
    }

    /// Add multiple relationship type filters
    pub fn with_relationship_types(mut self, relationship_types: Vec<RelationshipType>) -> Self {
        self.query.relationship_filters.extend(relationship_types);
        self
    }

    /// Set traversal options
    pub fn with_traversal_options(mut self, options: TraversalOptions) -> Self {
        self.query.traversal_options = Some(options);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }

    /// Set sort criteria
    pub fn sort_by(mut self, sort_by: QuerySortBy) -> Self {
        self.query.sort_by = Some(sort_by);
        self
    }

    /// Match nodes with a specific property value
    pub fn match_nodes_with_property(mut self, key: &str, value: &str) -> Self {
        self.query.property_filters.insert(key.to_string(), value.to_string());
        self
    }

    /// Match nodes with any of the specified tags
    pub fn match_nodes_with_any_tag(mut self, tags: Vec<String>) -> Self {
        self.query.tag_filters.extend(tags);
        self
    }

    /// Build the final query
    pub fn build(self) -> GraphQuery {
        self.query
    }
}

impl Default for GraphQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a graph query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Matching nodes
    pub nodes: Vec<Uuid>,
    /// Matching edges
    pub edges: Vec<Uuid>,
    /// Paths found (for traversal queries)
    pub paths: Vec<GraphPath>,
    /// Relevance score
    pub score: f64,
}

impl QueryResult {
    /// Create a new query result
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            paths: Vec::new(),
            score: 0.0,
        }
    }

    /// Add a node to the result
    pub fn add_node(&mut self, node_id: Uuid) {
        if !self.nodes.contains(&node_id) {
            self.nodes.push(node_id);
        }
    }

    /// Add an edge to the result
    pub fn add_edge(&mut self, edge_id: Uuid) {
        if !self.edges.contains(&edge_id) {
            self.edges.push(edge_id);
        }
    }

    /// Add a path to the result
    pub fn add_path(&mut self, path: GraphPath) {
        self.paths.push(path);
    }

    /// Set the relevance score
    pub fn set_score(&mut self, score: f64) {
        self.score = score.clamp(0.0, 1.0);
    }

    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty() && self.paths.is_empty()
    }

    /// Get the total number of entities in the result
    pub fn entity_count(&self) -> usize {
        self.nodes.len() + self.edges.len()
    }
}

impl Default for QueryResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced query patterns for complex graph matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPattern {
    /// Find nodes connected by a specific path pattern
    PathPattern {
        start_node_filter: Option<NodeFilter>,
        path_constraints: Vec<PathConstraint>,
        end_node_filter: Option<NodeFilter>,
    },
    /// Find subgraphs matching a pattern
    SubgraphPattern {
        node_patterns: Vec<NodePattern>,
        edge_patterns: Vec<EdgePattern>,
    },
    /// Find nodes within a certain distance
    ProximityPattern {
        center_node: Uuid,
        max_distance: usize,
        node_filter: Option<NodeFilter>,
    },
    /// Find strongly connected components
    ConnectedComponents {
        min_size: Option<usize>,
        max_size: Option<usize>,
    },
}

/// Filter for nodes in query patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFilter {
    pub node_type: Option<NodeType>,
    pub properties: HashMap<String, String>,
    pub tags: Vec<String>,
    pub min_importance: Option<f64>,
    pub min_confidence: Option<f64>,
}

/// Pattern for matching nodes in complex queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePattern {
    pub variable_name: String,
    pub filter: NodeFilter,
    pub optional: bool,
}

/// Pattern for matching edges in complex queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePattern {
    pub from_variable: String,
    pub to_variable: String,
    pub relationship_type: Option<RelationshipType>,
    pub properties: HashMap<String, String>,
    pub min_strength: Option<f64>,
    pub min_confidence: Option<f64>,
    pub optional: bool,
}

/// Constraint for path patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathConstraint {
    pub relationship_type: Option<RelationshipType>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub allow_cycles: bool,
}

/// Query execution context
pub struct QueryContext {
    /// Variable bindings (variable_name -> node_id)
    pub bindings: HashMap<String, Uuid>,
    /// Current path being explored
    pub current_path: GraphPath,
    /// Visited nodes (for cycle detection)
    pub visited: std::collections::HashSet<Uuid>,
}

impl QueryContext {
    /// Create a new query context
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            current_path: GraphPath::new(),
            visited: std::collections::HashSet::new(),
        }
    }

    /// Bind a variable to a node
    pub fn bind_variable(&mut self, variable: String, node_id: Uuid) {
        self.bindings.insert(variable, node_id);
    }

    /// Get the binding for a variable
    pub fn get_binding(&self, variable: &str) -> Option<Uuid> {
        self.bindings.get(variable).copied()
    }

    /// Check if a variable is bound
    pub fn is_bound(&self, variable: &str) -> bool {
        self.bindings.contains_key(variable)
    }

    /// Mark a node as visited
    pub fn visit_node(&mut self, node_id: Uuid) {
        self.visited.insert(node_id);
    }

    /// Check if a node has been visited
    pub fn is_visited(&self, node_id: Uuid) -> bool {
        self.visited.contains(&node_id)
    }

    /// Clone the context for branching
    pub fn branch(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            current_path: self.current_path.clone(),
            visited: self.visited.clone(),
        }
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new()
    }
}
