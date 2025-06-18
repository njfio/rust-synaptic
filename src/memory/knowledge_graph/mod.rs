//! Knowledge graph implementation for AI agent memory system
//!
//! This module provides a graph-based representation of memories and their relationships,
//! enabling sophisticated querying and reasoning capabilities.

pub mod types;
pub mod graph;
pub mod query;
pub mod reasoning;

// Re-export commonly used types
pub use types::{
    Node, Edge, Relationship, RelationshipType, NodeType, GraphEntity,
    KnowledgeGraphMetadata, GraphPath, GraphPattern,
};
pub use graph::{KnowledgeGraph, GraphConfig, GraphStats};
pub use query::{GraphQuery, GraphQueryBuilder, QueryResult, TraversalOptions};
pub use reasoning::{GraphReasoner, InferenceRule, InferenceEngine};

// Distributed graph types
pub type MemoryNode = Node;
pub type MemoryEdge = Edge;

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

/// Main knowledge graph interface for the memory system
pub struct MemoryKnowledgeGraph {
    /// Core graph structure
    graph: KnowledgeGraph,
    /// Memory entry to node mapping
    memory_to_node: HashMap<String, Uuid>, // memory_key -> node_id
    /// Node to memory entry mapping
    node_to_memory: HashMap<Uuid, String>, // node_id -> memory_key
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
            config,
            reasoner,
        }
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

        let from_node_id = self.memory_to_node.get(from_memory_key)
            .ok_or_else(|| MemoryError::NotFound { key: from_memory_key.to_string() })?;

        let to_node_id = self.memory_to_node.get(to_memory_key)
            .ok_or_else(|| MemoryError::NotFound { key: to_memory_key.to_string() })?;

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
        let node_id = self.memory_to_node.get(memory_key)
            .ok_or_else(|| MemoryError::NotFound { key: memory_key.to_string() })?;

        let traversal_options = TraversalOptions {
            max_depth,
            relationship_types,
            direction: query::TraversalDirection::Both,
            ..Default::default()
        };

        let related_nodes = self.graph.traverse_from_node(*node_id, traversal_options).await?;
        
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
        related_memories.sort_by(|a, b| b.relationship_strength.partial_cmp(&a.relationship_strength).unwrap());
        
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
        let from_node = self.memory_to_node.get(from_memory)
            .ok_or_else(|| MemoryError::NotFound { key: from_memory.to_string() })?;
        
        let to_node = self.memory_to_node.get(to_memory)
            .ok_or_else(|| MemoryError::NotFound { key: to_memory.to_string() })?;

        self.graph.find_shortest_path(*from_node, *to_node, max_depth).await
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

    /// Get the memory key for a given node ID
    pub async fn get_memory_for_node(&self, node_id: Uuid) -> Result<Option<String>> {
        Ok(self.node_to_memory.get(&node_id).cloned())
    }

    /// Get all connected nodes with their relationship types and strengths
    pub async fn get_connected_nodes(&self, node_id: Uuid) -> Result<Vec<(Uuid, RelationshipType, f64)>> {
        self.graph.get_connected_nodes(node_id).await
    }

    /// Auto-detect relationships based on content similarity and metadata
    async fn auto_detect_relationships(&mut self, node_id: Uuid, memory: &MemoryEntry) -> Result<()> {
        // Find similar memories based on tags
        for tag in &memory.metadata.tags {
            let similar_memories = self.get_memories_by_concept(tag).await?;
            for similar_memory_key in similar_memories {
                if similar_memory_key != memory.key {
                    if let Some(similar_node_id) = self.memory_to_node.get(&similar_memory_key) {
                        // Create a "related_to" relationship
                        let edge = Edge::new(
                            node_id,
                            *similar_node_id,
                            RelationshipType::RelatedTo,
                            Some(HashMap::from([
                                ("reason".to_string(), "shared_tag".to_string()),
                                ("tag".to_string(), tag.clone()),
                            ])),
                        );
                        let _ = self.graph.add_edge(edge).await;
                    }
                }
            }
        }

        // Detect temporal relationships
        self.detect_temporal_relationships(node_id, memory).await?;
        
        // Detect semantic relationships (if embeddings are available)
        if memory.embedding.is_some() {
            self.detect_semantic_relationships(node_id, memory).await?;
        }

        Ok(())
    }

    /// Detect temporal relationships between memories
    async fn detect_temporal_relationships(&mut self, node_id: Uuid, memory: &MemoryEntry) -> Result<()> {
        // Find memories created around the same time
        let time_window = chrono::Duration::hours(1);
        let memory_time = memory.created_at();
        
        for (other_memory_key, other_node_id) in &self.memory_to_node {
            if other_memory_key != &memory.key {
                if let Some(other_node) = self.graph.get_node(*other_node_id).await? {
                    if let Some(other_time) = other_node.created_at {
                        let time_diff = (memory_time - other_time).abs();
                        if time_diff <= time_window {
                            let edge = Edge::new(
                                node_id,
                                *other_node_id,
                                RelationshipType::TemporallyRelated,
                                Some(HashMap::from([
                                    ("time_diff_minutes".to_string(), time_diff.num_minutes().to_string()),
                                ])),
                            );
                            let _ = self.graph.add_edge(edge).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Detect semantic relationships using embeddings
    async fn detect_semantic_relationships(&mut self, node_id: Uuid, memory: &MemoryEntry) -> Result<()> {
        if let Some(embedding) = &memory.embedding {
            let similarity_threshold = self.config.semantic_similarity_threshold;
            
            for (other_memory_key, other_node_id) in &self.memory_to_node {
                if other_memory_key != &memory.key {
                    if let Some(other_node) = self.graph.get_node(*other_node_id).await? {
                        if let Some(other_embedding) = &other_node.embedding {
                            let similarity = cosine_similarity(embedding, other_embedding);
                            if similarity >= similarity_threshold {
                                let edge = Edge::new(
                                    node_id,
                                    *other_node_id,
                                    RelationshipType::SemanticallyRelated,
                                    Some(HashMap::from([
                                        ("similarity_score".to_string(), similarity.to_string()),
                                    ])),
                                );
                                let _ = self.graph.add_edge(edge).await;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Update an existing node with new memory data
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
            self.update_relationships_for_changed_node(node_id, &old_content, &new_content).await?;

            println!("Updated existing node {} for memory '{}'", node_id, memory.key);
        }

        Ok(node_id)
    }

    /// Find a similar existing node that could be merged
    async fn find_similar_node(&self, memory: &MemoryEntry) -> Result<Option<Uuid>> {
        let similarity_threshold = self.config.semantic_similarity_threshold;
        let mut best_match: Option<(Uuid, f64)> = None;

        for node_ref in self.graph.nodes.iter() {
            let node = node_ref.value();

            // Skip if this is already mapped to a different memory
            if let Some(existing_memory_key) = self.node_to_memory.get(&node.id) {
                if existing_memory_key != &memory.key {
                    continue;
                }
            }

            // Calculate similarity
            let similarity = self.calculate_node_memory_similarity(node, memory).await?;

            if similarity >= similarity_threshold {
                if let Some((_, best_similarity)) = best_match {
                    if similarity > best_similarity {
                        best_match = Some((node.id, similarity));
                    }
                } else {
                    best_match = Some((node.id, similarity));
                }
            }
        }

        Ok(best_match.map(|(node_id, _)| node_id))
    }

    /// Merge memory with an existing similar node
    async fn merge_with_existing_node(&mut self, node_id: Uuid, memory: &MemoryEntry) -> Result<Uuid> {
        if let Some(mut node) = self.graph.get_node(node_id).await? {
            println!("Merging memory '{}' with existing node {}", memory.key, node_id);

            // Combine content intelligently
            let combined_content = self.merge_content(
                &node.description.unwrap_or_default(),
                &memory.value
            ).await?;

            node.description = Some(combined_content);
            node.last_modified = Some(chrono::Utc::now());

            // Update importance and confidence (weighted average)
            let old_weight = 0.7; // Give more weight to existing data
            let new_weight = 0.3;
            node.importance = node.importance * old_weight + memory.metadata.importance * new_weight;
            node.confidence = node.confidence * old_weight + memory.metadata.confidence * new_weight;

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
        let node = Node::from_memory(memory);
        let node_id = self.graph.add_node(node).await?;

        // Update mappings
        self.memory_to_node.insert(memory.key.clone(), node_id);
        self.node_to_memory.insert(node_id, memory.key.clone());

        // Auto-detect relationships with existing nodes
        self.auto_detect_relationships(node_id, memory).await?;

        println!("Created new node {} for memory '{}'", node_id, memory.key);
        Ok(node_id)
    }

    /// Find all memories within a specified graph distance from a reference memory
    pub async fn find_memories_within_distance(
        &self,
        reference_memory_key: &str,
        max_distance: usize,
    ) -> Result<Vec<(String, usize)>> {
        // Get the node ID for the reference memory
        let reference_node_id = self.memory_to_node.get(reference_memory_key)
            .ok_or_else(|| MemoryError::NotFound { key: reference_memory_key.to_string() })?;

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

        let reachable_nodes = self.graph.traverse_from_node(*reference_node_id, traversal_options).await?;

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
    async fn calculate_node_memory_similarity(&self, node: &Node, memory: &MemoryEntry) -> Result<f64> {
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
            let tag_similarity = if union > 0 { intersection as f64 / union as f64 } else { 0.0 };
            similarity_score += tag_similarity;
            factor_count += 1;
        }

        // Embedding similarity (if both have embeddings)
        if let (Some(node_embedding), Some(memory_embedding)) = (&node.embedding, &memory.embedding) {
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

        Ok(if factor_count > 0 { similarity_score / factor_count as f64 } else { 0.0 })
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

        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
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
            self.merge_similar_content(existing_content, new_content).await
        } else {
            // Low similarity: create a structured combination
            Ok(format!("{}\n\n--- Updated Information ---\n{}", existing_content, new_content))
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
        let mut merged_sentences = Vec::with_capacity(existing_sentences.len() + new_sentences.len());
        merged_sentences.extend_from_slice(&existing_sentences);

        // Use iterator approach to avoid nested loops where possible
        for new_sentence in new_sentences {
            let is_duplicate = existing_sentences
                .iter()
                .any(|&existing_sentence| self.calculate_text_similarity(existing_sentence, new_sentence) > 0.8);

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
                self.auto_detect_relationships(node_id, &temp_memory).await?;
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
            if (edge.from_node == node_id || edge.to_node == node_id) &&
               edge.relationship.strength < weak_threshold {
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
        let connected_edges: Vec<_> = self.graph.edges.iter()
            .filter(|edge_ref| {
                let edge = edge_ref.value();
                edge.from_node == node_id || edge.to_node == node_id
            })
            .map(|edge_ref| edge_ref.key().clone())
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
            if let (Some(from_content), Some(to_content)) = (&from_node.description, &to_node.description) {
                let content_similarity = self.calculate_text_similarity(from_content, to_content);
                strength += content_similarity * 0.3;
            }

            // Tag overlap factor
            let from_tags: std::collections::HashSet<_> = from_node.tags.iter().collect();
            let to_tags: std::collections::HashSet<_> = to_node.tags.iter().collect();
            let tag_overlap = if !from_tags.is_empty() && !to_tags.is_empty() {
                from_tags.intersection(&to_tags).count() as f64 / from_tags.union(&to_tags).count() as f64
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
