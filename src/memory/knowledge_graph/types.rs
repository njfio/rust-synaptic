//! Core types for the knowledge graph system

use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Types of nodes in the knowledge graph
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// Memory entry node
    Memory,
    /// Concept or entity node
    Concept,
    /// Event node
    Event,
    /// Person node
    Person,
    /// Location node
    Location,
    /// Topic node
    Topic,
    /// Custom node type
    Custom(String),
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Memory => write!(f, "memory"),
            NodeType::Concept => write!(f, "concept"),
            NodeType::Event => write!(f, "event"),
            NodeType::Person => write!(f, "person"),
            NodeType::Location => write!(f, "location"),
            NodeType::Topic => write!(f, "topic"),
            NodeType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Types of relationships between nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// General relationship
    RelatedTo,
    /// Causal relationship (A causes B)
    Causes,
    /// Reverse causal relationship (A is caused by B)
    CausedBy,
    /// Hierarchical relationship (A is part of B)
    PartOf,
    /// Reverse hierarchical relationship (A contains B)
    Contains,
    /// Temporal relationship (happened around the same time)
    TemporallyRelated,
    /// Semantic similarity relationship
    SemanticallyRelated,
    /// Reference relationship (A references B)
    References,
    /// Dependency relationship (A depends on B)
    DependsOn,
    /// Similarity relationship
    SimilarTo,
    /// Contradiction relationship
    Contradicts,
    /// Custom relationship type
    Custom(String),
}

impl std::fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationshipType::RelatedTo => write!(f, "related_to"),
            RelationshipType::Causes => write!(f, "causes"),
            RelationshipType::CausedBy => write!(f, "caused_by"),
            RelationshipType::PartOf => write!(f, "part_of"),
            RelationshipType::Contains => write!(f, "contains"),
            RelationshipType::TemporallyRelated => write!(f, "temporally_related"),
            RelationshipType::SemanticallyRelated => write!(f, "semantically_related"),
            RelationshipType::References => write!(f, "references"),
            RelationshipType::DependsOn => write!(f, "depends_on"),
            RelationshipType::SimilarTo => write!(f, "similar_to"),
            RelationshipType::Contradicts => write!(f, "contradicts"),
            RelationshipType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier
    pub id: Uuid,
    /// Node type
    pub node_type: NodeType,
    /// Node label/name
    pub label: String,
    /// Node description
    pub description: Option<String>,
    /// Node properties
    pub properties: HashMap<String, String>,
    /// Vector embedding for semantic operations
    pub embedding: Option<Vec<f32>>,
    /// When this node was created
    pub created_at: Option<DateTime<Utc>>,
    /// When this node was last modified
    pub last_modified: Option<DateTime<Utc>>,
    /// Importance score (0.0 to 1.0)
    pub importance: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Node {
    /// Create a new node
    pub fn new(node_type: NodeType, label: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            node_type,
            label,
            description: None,
            properties: HashMap::new(),
            embedding: None,
            created_at: Some(now),
            last_modified: Some(now),
            importance: 0.5,
            confidence: 1.0,
            tags: Vec::new(),
        }
    }

    /// Create a node from a memory entry
    pub fn from_memory(memory: &MemoryEntry) -> Self {
        let mut node = Self::new(NodeType::Memory, memory.key.clone());
        node.description = Some(memory.value.clone());
        node.embedding = memory.embedding.clone();
        node.created_at = Some(memory.created_at());
        node.last_modified = Some(memory.last_accessed());
        node.importance = memory.metadata.importance;
        node.confidence = memory.metadata.confidence;
        node.tags = memory.metadata.tags.clone();
        
        // Copy custom fields as properties
        for (key, value) in &memory.metadata.custom_fields {
            node.properties.insert(key.clone(), value.clone());
        }
        
        // Add memory-specific properties
        node.properties.insert("memory_type".to_string(), memory.memory_type.to_string());
        node.properties.insert("access_count".to_string(), memory.access_count().to_string());
        
        node
    }

    /// Add a property to the node
    pub fn add_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
        self.mark_modified();
    }

    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&String> {
        self.properties.get(key)
    }

    /// Add a tag to the node
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
            self.mark_modified();
        }
    }

    /// Remove a tag from the node
    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
        self.mark_modified();
    }

    /// Check if the node has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(&tag.to_string())
    }

    /// Mark the node as modified
    pub fn mark_modified(&mut self) {
        self.last_modified = Some(Utc::now());
    }

    /// Set the importance score
    pub fn set_importance(&mut self, importance: f64) {
        self.importance = importance.clamp(0.0, 1.0);
        self.mark_modified();
    }

    /// Set the confidence score
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence.clamp(0.0, 1.0);
        self.mark_modified();
    }
}

/// Relationship information between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Relationship properties
    pub properties: HashMap<String, String>,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f64,
    /// Confidence in this relationship (0.0 to 1.0)
    pub confidence: f64,
    /// When this relationship was created
    pub created_at: DateTime<Utc>,
    /// When this relationship was last modified
    pub last_modified: DateTime<Utc>,
}

impl Relationship {
    /// Create a new relationship
    pub fn new(relationship_type: RelationshipType) -> Self {
        let now = Utc::now();
        Self {
            relationship_type,
            properties: HashMap::new(),
            strength: 1.0,
            confidence: 1.0,
            created_at: now,
            last_modified: now,
        }
    }

    /// Create a relationship with properties
    pub fn with_properties(relationship_type: RelationshipType, properties: HashMap<String, String>) -> Self {
        let mut rel = Self::new(relationship_type);
        rel.properties = properties;
        rel
    }

    /// Add a property to the relationship
    pub fn add_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
        self.mark_modified();
    }

    /// Set the relationship strength
    pub fn set_strength(&mut self, strength: f64) {
        self.strength = strength.clamp(0.0, 1.0);
        self.mark_modified();
    }

    /// Set the relationship confidence
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence.clamp(0.0, 1.0);
        self.mark_modified();
    }

    /// Mark the relationship as modified
    pub fn mark_modified(&mut self) {
        self.last_modified = Utc::now();
    }
}

/// An edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier
    pub id: Uuid,
    /// Source node ID
    pub from_node: Uuid,
    /// Target node ID
    pub to_node: Uuid,
    /// Relationship information
    pub relationship: Relationship,
}

impl Edge {
    /// Create a new edge
    pub fn new(
        from_node: Uuid,
        to_node: Uuid,
        relationship_type: RelationshipType,
        properties: Option<HashMap<String, String>>,
    ) -> Self {
        let relationship = if let Some(props) = properties {
            Relationship::with_properties(relationship_type, props)
        } else {
            Relationship::new(relationship_type)
        };

        Self {
            id: Uuid::new_v4(),
            from_node,
            to_node,
            relationship,
        }
    }

    /// Check if this edge connects the given nodes (in either direction)
    pub fn connects(&self, node1: Uuid, node2: Uuid) -> bool {
        (self.from_node == node1 && self.to_node == node2) ||
        (self.from_node == node2 && self.to_node == node1)
    }

    /// Get the other node in this edge
    pub fn other_node(&self, node_id: Uuid) -> Option<Uuid> {
        if self.from_node == node_id {
            Some(self.to_node)
        } else if self.to_node == node_id {
            Some(self.from_node)
        } else {
            None
        }
    }
}

/// A path through the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Nodes in the path
    pub nodes: Vec<Uuid>,
    /// Edges in the path
    pub edges: Vec<Uuid>,
    /// Total path length
    pub length: usize,
    /// Path weight/cost
    pub weight: f64,
}

impl GraphPath {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            length: 0,
            weight: 0.0,
        }
    }

    /// Add a node and edge to the path
    pub fn add_step(&mut self, node_id: Uuid, edge_id: Option<Uuid>) {
        self.nodes.push(node_id);
        if let Some(edge_id) = edge_id {
            self.edges.push(edge_id);
        }
        self.length = self.nodes.len();
    }

    /// Check if the path is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the start node
    pub fn start_node(&self) -> Option<Uuid> {
        self.nodes.first().copied()
    }

    /// Get the end node
    pub fn end_node(&self) -> Option<Uuid> {
        self.nodes.last().copied()
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

/// A pattern for matching graph structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Node patterns to match
    pub node_patterns: Vec<NodePattern>,
    /// Edge patterns to match
    pub edge_patterns: Vec<EdgePattern>,
}

/// Pattern for matching nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePattern {
    /// Variable name for this node
    pub variable: String,
    /// Node type to match (optional)
    pub node_type: Option<NodeType>,
    /// Properties that must match
    pub properties: HashMap<String, String>,
    /// Tags that must be present
    pub tags: Vec<String>,
}

/// Pattern for matching edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePattern {
    /// Variable name for the source node
    pub from_variable: String,
    /// Variable name for the target node
    pub to_variable: String,
    /// Relationship type to match (optional)
    pub relationship_type: Option<RelationshipType>,
    /// Properties that must match
    pub properties: HashMap<String, String>,
}

/// Metadata for the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphMetadata {
    /// Graph creation time
    pub created_at: DateTime<Utc>,
    /// Last modification time
    pub last_modified: DateTime<Utc>,
    /// Graph version
    pub version: u64,
    /// Graph description
    pub description: Option<String>,
    /// Graph tags
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom_fields: HashMap<String, String>,
}

impl KnowledgeGraphMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            last_modified: now,
            version: 1,
            description: None,
            tags: Vec::new(),
            custom_fields: HashMap::new(),
        }
    }

    /// Mark as modified
    pub fn mark_modified(&mut self) {
        self.last_modified = Utc::now();
        self.version += 1;
    }
}

impl Default for KnowledgeGraphMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// A graph entity that can be either a node or an edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEntity {
    Node(Node),
    Edge(Edge),
}

impl GraphEntity {
    /// Get the entity ID
    pub fn id(&self) -> Uuid {
        match self {
            GraphEntity::Node(node) => node.id,
            GraphEntity::Edge(edge) => edge.id,
        }
    }

    /// Check if this is a node
    pub fn is_node(&self) -> bool {
        matches!(self, GraphEntity::Node(_))
    }

    /// Check if this is an edge
    pub fn is_edge(&self) -> bool {
        matches!(self, GraphEntity::Edge(_))
    }

    /// Get as node if it is one
    pub fn as_node(&self) -> Option<&Node> {
        match self {
            GraphEntity::Node(node) => Some(node),
            GraphEntity::Edge(_) => None,
        }
    }

    /// Get as edge if it is one
    pub fn as_edge(&self) -> Option<&Edge> {
        match self {
            GraphEntity::Node(_) => None,
            GraphEntity::Edge(edge) => Some(edge),
        }
    }
}
