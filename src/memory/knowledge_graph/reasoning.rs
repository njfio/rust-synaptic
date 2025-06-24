//! Graph reasoning and inference engine

use super::types::{Node, RelationshipType};
use super::graph::KnowledgeGraph;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Graph reasoning engine for inferring new relationships
pub struct GraphReasoner {
    /// Inference rules
    rules: Vec<InferenceRule>,
    /// Configuration
    config: ReasoningConfig,
}

/// Configuration for the reasoning engine
#[derive(Debug, Clone)]
pub struct ReasoningConfig {
    /// Minimum confidence threshold for inferences
    pub min_confidence_threshold: f64,
    /// Maximum inference depth
    pub max_inference_depth: usize,
    /// Enable transitive reasoning
    pub enable_transitive_reasoning: bool,
    /// Enable similarity-based reasoning
    pub enable_similarity_reasoning: bool,
    /// Enable temporal reasoning
    pub enable_temporal_reasoning: bool,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            max_inference_depth: 3,
            enable_transitive_reasoning: true,
            enable_similarity_reasoning: true,
            enable_temporal_reasoning: true,
        }
    }
}

/// An inference rule for deriving new relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Rule pattern
    pub pattern: RulePattern,
    /// Confidence weight for this rule
    pub confidence_weight: f64,
    /// Whether this rule is enabled
    pub enabled: bool,
}

/// Pattern for inference rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RulePattern {
    /// Transitive rule: if A->B and B->C, then A->C
    Transitive {
        relationship_type: RelationshipType,
        inferred_type: RelationshipType,
    },
    /// Symmetric rule: if A->B, then B->A
    Symmetric {
        relationship_type: RelationshipType,
        inferred_type: RelationshipType,
    },
    /// Inverse rule: if A->B with type X, then B->A with type Y
    Inverse {
        source_type: RelationshipType,
        inverse_type: RelationshipType,
    },
    /// Similarity rule: if A similar to B and B->C, then A might relate to C
    Similarity {
        similarity_threshold: f64,
        source_relationship: RelationshipType,
        inferred_relationship: RelationshipType,
    },
    /// Temporal rule: if A and B happen close in time, they might be related
    Temporal {
        max_time_diff_hours: u64,
        inferred_relationship: RelationshipType,
    },
    /// Custom rule with arbitrary logic
    Custom {
        rule_logic: String, // Could be a script or function name
    },
}

/// Result of an inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Source node
    pub from_node: Uuid,
    /// Target node
    pub to_node: Uuid,
    /// Inferred relationship type
    pub relationship_type: RelationshipType,
    /// Confidence in the inference
    pub confidence: f64,
    /// Rule that generated this inference
    pub rule_id: String,
    /// Explanation of the reasoning
    pub explanation: String,
    /// Supporting evidence (node/edge IDs)
    pub evidence: Vec<Uuid>,
}

/// Inference engine for the knowledge graph
pub struct InferenceEngine {
    /// Reasoning configuration
    config: ReasoningConfig,
}

impl GraphReasoner {
    /// Create a new graph reasoner
    pub fn new() -> Self {
        let mut reasoner = Self {
            rules: Vec::new(),
            config: ReasoningConfig::default(),
        };
        
        // Add default inference rules
        reasoner.add_default_rules();
        reasoner
    }

    /// Create a reasoner with custom configuration
    pub fn with_config(config: ReasoningConfig) -> Self {
        let mut reasoner = Self {
            rules: Vec::new(),
            config,
        };
        
        reasoner.add_default_rules();
        reasoner
    }

    /// Add an inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Remove an inference rule
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        if let Some(pos) = self.rules.iter().position(|r| r.id == rule_id) {
            self.rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Enable or disable a rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) -> bool {
        if let Some(rule) = self.rules.iter_mut().find(|r| r.id == rule_id) {
            rule.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Infer new relationships from the graph
    pub async fn infer_relationships(&self, graph: &KnowledgeGraph) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            
            let rule_inferences = self.apply_rule(rule, graph).await?;
            inferences.extend(rule_inferences);
        }
        
        // Filter by confidence threshold
        inferences.retain(|inf| inf.confidence >= self.config.min_confidence_threshold);
        
        // Sort by confidence (highest first)
        inferences.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(inferences)
    }

    /// Apply a specific inference rule to the graph
    async fn apply_rule(&self, rule: &InferenceRule, graph: &KnowledgeGraph) -> Result<Vec<InferenceResult>> {
        match &rule.pattern {
            RulePattern::Transitive { relationship_type, inferred_type } => {
                self.apply_transitive_rule(rule, graph, relationship_type, inferred_type).await
            }
            RulePattern::Symmetric { relationship_type, inferred_type } => {
                self.apply_symmetric_rule(rule, graph, relationship_type, inferred_type).await
            }
            RulePattern::Inverse { source_type, inverse_type } => {
                self.apply_inverse_rule(rule, graph, source_type, inverse_type).await
            }
            RulePattern::Similarity { similarity_threshold, source_relationship, inferred_relationship } => {
                self.apply_similarity_rule(rule, graph, *similarity_threshold, source_relationship, inferred_relationship).await
            }
            RulePattern::Temporal { max_time_diff_hours, inferred_relationship } => {
                self.apply_temporal_rule(rule, graph, *max_time_diff_hours, inferred_relationship).await
            }
            RulePattern::Custom { rule_logic: _ } => {
                // Custom rules would require a scripting engine or plugin system
                Ok(Vec::new())
            }
        }
    }

    /// Apply transitive reasoning rule
    async fn apply_transitive_rule(
        &self,
        rule: &InferenceRule,
        graph: &KnowledgeGraph,
        relationship_type: &RelationshipType,
        inferred_type: &RelationshipType,
    ) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        // Find all edges of the specified type
        let relevant_edges: Vec<_> = graph.edges.iter()
            .filter(|edge_ref| edge_ref.relationship.relationship_type == *relationship_type)
            .map(|edge_ref| edge_ref.clone())
            .collect();
        
        // Look for transitive patterns: A->B and B->C implies A->C
        for edge1 in &relevant_edges {
            for edge2 in &relevant_edges {
                if edge1.to_node == edge2.from_node && edge1.from_node != edge2.to_node {
                    // Check if A->C relationship already exists
                    let existing_edge = graph.edges.iter()
                        .find(|e| e.from_node == edge1.from_node && 
                                 e.to_node == edge2.to_node &&
                                 e.relationship.relationship_type == *inferred_type);
                    
                    if existing_edge.is_none() {
                        let confidence = (edge1.relationship.confidence * edge2.relationship.confidence * rule.confidence_weight).min(1.0);
                        
                        if confidence >= self.config.min_confidence_threshold {
                            inferences.push(InferenceResult {
                                from_node: edge1.from_node,
                                to_node: edge2.to_node,
                                relationship_type: inferred_type.clone(),
                                confidence,
                                rule_id: rule.id.clone(),
                                explanation: format!(
                                    "Transitive inference: {} -> {} -> {} implies {} -> {}",
                                    edge1.from_node, edge1.to_node, edge2.to_node,
                                    edge1.from_node, edge2.to_node
                                ),
                                evidence: vec![edge1.id, edge2.id],
                            });
                        }
                    }
                }
            }
        }
        
        Ok(inferences)
    }

    /// Apply symmetric reasoning rule
    async fn apply_symmetric_rule(
        &self,
        rule: &InferenceRule,
        graph: &KnowledgeGraph,
        relationship_type: &RelationshipType,
        inferred_type: &RelationshipType,
    ) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        for edge_ref in graph.edges.iter() {
            let edge = edge_ref.value();
            if edge.relationship.relationship_type == *relationship_type {
                // Check if reverse relationship exists
                let reverse_exists = graph.edges.iter()
                    .any(|e| e.from_node == edge.to_node && 
                            e.to_node == edge.from_node &&
                            e.relationship.relationship_type == *inferred_type);
                
                if !reverse_exists {
                    let confidence = edge.relationship.confidence * rule.confidence_weight;
                    
                    if confidence >= self.config.min_confidence_threshold {
                        inferences.push(InferenceResult {
                            from_node: edge.to_node,
                            to_node: edge.from_node,
                            relationship_type: inferred_type.clone(),
                            confidence,
                            rule_id: rule.id.clone(),
                            explanation: format!(
                                "Symmetric inference: {} -> {} implies {} -> {}",
                                edge.from_node, edge.to_node, edge.to_node, edge.from_node
                            ),
                            evidence: vec![edge.id],
                        });
                    }
                }
            }
        }
        
        Ok(inferences)
    }

    /// Apply inverse reasoning rule
    async fn apply_inverse_rule(
        &self,
        rule: &InferenceRule,
        graph: &KnowledgeGraph,
        source_type: &RelationshipType,
        inverse_type: &RelationshipType,
    ) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        for edge_ref in graph.edges.iter() {
            let edge = edge_ref.value();
            if edge.relationship.relationship_type == *source_type {
                // Check if inverse relationship exists
                let inverse_exists = graph.edges.iter()
                    .any(|e| e.from_node == edge.to_node && 
                            e.to_node == edge.from_node &&
                            e.relationship.relationship_type == *inverse_type);
                
                if !inverse_exists {
                    let confidence = edge.relationship.confidence * rule.confidence_weight;
                    
                    if confidence >= self.config.min_confidence_threshold {
                        inferences.push(InferenceResult {
                            from_node: edge.to_node,
                            to_node: edge.from_node,
                            relationship_type: inverse_type.clone(),
                            confidence,
                            rule_id: rule.id.clone(),
                            explanation: format!(
                                "Inverse inference: {} {} {} implies {} {} {}",
                                edge.from_node, source_type, edge.to_node,
                                edge.to_node, inverse_type, edge.from_node
                            ),
                            evidence: vec![edge.id],
                        });
                    }
                }
            }
        }
        
        Ok(inferences)
    }

    /// Apply similarity-based reasoning
    async fn apply_similarity_rule(
        &self,
        rule: &InferenceRule,
        graph: &KnowledgeGraph,
        similarity_threshold: f64,
        source_relationship: &RelationshipType,
        inferred_relationship: &RelationshipType,
    ) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        // This would require implementing similarity calculation between nodes
        // For now, we'll use a simplified version based on shared tags
        
        for node1_ref in graph.nodes.iter() {
            for node2_ref in graph.nodes.iter() {
                let node1 = node1_ref.value();
                let node2 = node2_ref.value();
                
                if node1.id != node2.id {
                    let similarity = self.calculate_node_similarity(node1, node2);
                    
                    if similarity >= similarity_threshold {
                        // Look for relationships from node2 that could be inferred for node1
                        for edge_ref in graph.edges.iter() {
                            let edge = edge_ref.value();
                            if edge.from_node == node2.id && edge.relationship.relationship_type == *source_relationship {
                                // Check if similar relationship already exists for node1
                                let exists = graph.edges.iter()
                                    .any(|e| e.from_node == node1.id && 
                                            e.to_node == edge.to_node &&
                                            e.relationship.relationship_type == *inferred_relationship);
                                
                                if !exists {
                                    let confidence = similarity * edge.relationship.confidence * rule.confidence_weight;
                                    
                                    if confidence >= self.config.min_confidence_threshold {
                                        inferences.push(InferenceResult {
                                            from_node: node1.id,
                                            to_node: edge.to_node,
                                            relationship_type: inferred_relationship.clone(),
                                            confidence,
                                            rule_id: rule.id.clone(),
                                            explanation: format!(
                                                "Similarity inference: {} similar to {} (similarity: {:.2}), {} {} {} implies {} {} {}",
                                                node1.id, node2.id, similarity,
                                                node2.id, source_relationship, edge.to_node,
                                                node1.id, inferred_relationship, edge.to_node
                                            ),
                                            evidence: vec![edge.id],
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(inferences)
    }

    /// Apply temporal reasoning
    async fn apply_temporal_rule(
        &self,
        rule: &InferenceRule,
        graph: &KnowledgeGraph,
        max_time_diff_hours: u64,
        inferred_relationship: &RelationshipType,
    ) -> Result<Vec<InferenceResult>> {
        let mut inferences = Vec::new();
        
        let time_threshold = chrono::Duration::hours(max_time_diff_hours as i64);
        
        for node1_ref in graph.nodes.iter() {
            for node2_ref in graph.nodes.iter() {
                let node1 = node1_ref.value();
                let node2 = node2_ref.value();
                
                if node1.id != node2.id {
                    if let (Some(time1), Some(time2)) = (node1.created_at, node2.created_at) {
                        let time_diff = (time1 - time2).abs();
                        
                        if time_diff <= time_threshold {
                            // Check if temporal relationship already exists
                            let exists = graph.edges.iter()
                                .any(|e| (e.from_node == node1.id && e.to_node == node2.id) ||
                                        (e.from_node == node2.id && e.to_node == node1.id));
                            
                            if !exists {
                                let confidence = (1.0 - (time_diff.num_minutes() as f64 / (max_time_diff_hours * 60) as f64)) * rule.confidence_weight;
                                
                                if confidence >= self.config.min_confidence_threshold {
                                    inferences.push(InferenceResult {
                                        from_node: node1.id,
                                        to_node: node2.id,
                                        relationship_type: inferred_relationship.clone(),
                                        confidence,
                                        rule_id: rule.id.clone(),
                                        explanation: format!(
                                            "Temporal inference: {} and {} created within {} hours",
                                            node1.id, node2.id, time_diff.num_hours()
                                        ),
                                        evidence: vec![node1.id, node2.id],
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(inferences)
    }

    /// Calculate similarity between two nodes
    fn calculate_node_similarity(&self, node1: &Node, node2: &Node) -> f64 {
        let mut similarity = 0.0;
        let mut factors = 0;
        
        // Tag similarity
        if !node1.tags.is_empty() || !node2.tags.is_empty() {
            let tags1: HashSet<_> = node1.tags.iter().collect();
            let tags2: HashSet<_> = node2.tags.iter().collect();
            let intersection = tags1.intersection(&tags2).count();
            let union = tags1.union(&tags2).count();
            
            if union > 0 {
                similarity += intersection as f64 / union as f64;
                factors += 1;
            }
        }
        
        // Node type similarity
        if node1.node_type == node2.node_type {
            similarity += 1.0;
        }
        factors += 1;
        
        // Embedding similarity (if available)
        if let (Some(emb1), Some(emb2)) = (&node1.embedding, &node2.embedding) {
            let cosine_sim = cosine_similarity(emb1, emb2);
            similarity += cosine_sim;
            factors += 1;
        }
        
        if factors > 0 {
            similarity / factors as f64
        } else {
            0.0
        }
    }

    /// Add default inference rules
    fn add_default_rules(&mut self) {
        // Transitive "causes" rule
        self.add_rule(InferenceRule {
            id: "transitive_causes".to_string(),
            name: "Transitive Causation".to_string(),
            description: "If A causes B and B causes C, then A indirectly causes C".to_string(),
            pattern: RulePattern::Transitive {
                relationship_type: RelationshipType::Causes,
                inferred_type: RelationshipType::Causes,
            },
            confidence_weight: 0.8,
            enabled: true,
        });

        // Symmetric "related_to" rule
        self.add_rule(InferenceRule {
            id: "symmetric_related".to_string(),
            name: "Symmetric Relation".to_string(),
            description: "If A is related to B, then B is related to A".to_string(),
            pattern: RulePattern::Symmetric {
                relationship_type: RelationshipType::RelatedTo,
                inferred_type: RelationshipType::RelatedTo,
            },
            confidence_weight: 0.9,
            enabled: true,
        });

        // Inverse "part_of" and "contains" rule
        self.add_rule(InferenceRule {
            id: "inverse_part_contains".to_string(),
            name: "Part-Contains Inverse".to_string(),
            description: "If A is part of B, then B contains A".to_string(),
            pattern: RulePattern::Inverse {
                source_type: RelationshipType::PartOf,
                inverse_type: RelationshipType::Contains,
            },
            confidence_weight: 1.0,
            enabled: true,
        });
    }
}

impl Default for GraphReasoner {
    fn default() -> Self {
        Self::new()
    }
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
