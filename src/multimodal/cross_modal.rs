//! # Cross-Modal Relationship System
//!
//! Advanced cross-modal relationship detection and management for the Synaptic memory system.
//! Enables intelligent connections between different types of content (image, audio, code, text).

use super::{
    ContentSpecificMetadata, ContentType, CrossModalLink, CrossModalRelationship,
    MultiModalMemory, MultiModalResult,
};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-modal relationship analyzer
#[derive(Debug)]
pub struct CrossModalAnalyzer {
    /// Configuration for relationship detection
    config: CrossModalConfig,
    
    /// Relationship detection strategies
    strategies: Vec<Box<dyn RelationshipStrategy>>,
}

/// Configuration for cross-modal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Minimum confidence threshold for relationships
    pub min_confidence_threshold: f32,
    
    /// Enable text extraction relationships
    pub enable_text_extraction: bool,
    
    /// Enable semantic similarity relationships
    pub enable_semantic_similarity: bool,
    
    /// Enable temporal relationships
    pub enable_temporal_relationships: bool,
    
    /// Enable content generation relationships
    pub enable_generation_relationships: bool,
    
    /// Maximum number of relationships per memory
    pub max_relationships_per_memory: usize,
    
    /// Similarity threshold for "similar to" relationships
    pub similarity_threshold: f32,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            enable_text_extraction: true,
            enable_semantic_similarity: true,
            enable_temporal_relationships: true,
            enable_generation_relationships: true,
            max_relationships_per_memory: 10,
            similarity_threshold: 0.8,
        }
    }
}

/// Trait for relationship detection strategies
pub trait RelationshipStrategy: Send + Sync {
    /// Detect relationships between two memories
    fn detect_relationships(
        &self,
        source: &MultiModalMemory,
        target: &MultiModalMemory,
    ) -> MultiModalResult<Vec<CrossModalLink>>;
    
    /// Get the strategy name
    fn name(&self) -> &str;
}

/// Text extraction relationship strategy
#[derive(Debug)]
pub struct TextExtractionStrategy {
    confidence_threshold: f32,
}

impl TextExtractionStrategy {
    pub fn new(confidence_threshold: f32) -> Self {
        Self { confidence_threshold }
    }
}

impl RelationshipStrategy for TextExtractionStrategy {
    fn detect_relationships(
        &self,
        source: &MultiModalMemory,
        target: &MultiModalMemory,
    ) -> MultiModalResult<Vec<CrossModalLink>> {
        let mut relationships = Vec::new();

        // Check if target is extracted text from source
        match (&source.content_type, &target.content_type) {
            (ContentType::Image { .. }, ContentType::ExtractedText { source_modality, confidence }) => {
                if let ContentType::Image { .. } = source_modality.as_ref() {
                    if *confidence >= self.confidence_threshold {
                        relationships.push(CrossModalLink {
                            target_id: target.id.clone(),
                            relationship_type: CrossModalRelationship::ExtractedFrom,
                            confidence: *confidence,
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert("extraction_method".to_string(), serde_json::Value::String("ocr".to_string()));
                                meta
                            },
                        });
                    }
                }
            }
            (ContentType::Audio { .. }, ContentType::ExtractedText { source_modality, confidence }) => {
                if let ContentType::Audio { .. } = source_modality.as_ref() {
                    if *confidence >= self.confidence_threshold {
                        relationships.push(CrossModalLink {
                            target_id: target.id.clone(),
                            relationship_type: CrossModalRelationship::ExtractedFrom,
                            confidence: *confidence,
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert("extraction_method".to_string(), serde_json::Value::String("speech_to_text".to_string()));
                                meta
                            },
                        });
                    }
                }
            }
            _ => {}
        }

        Ok(relationships)
    }

    fn name(&self) -> &str {
        "text_extraction"
    }
}

/// Semantic similarity relationship strategy
#[derive(Debug)]
pub struct SemanticSimilarityStrategy {
    similarity_threshold: f32,
}

impl SemanticSimilarityStrategy {
    pub fn new(similarity_threshold: f32) -> Self {
        Self { similarity_threshold }
    }

    /// Calculate semantic similarity between text content
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Simple word overlap similarity (in production, use embeddings)
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Extract text content from memory for comparison
    fn extract_text_content(&self, memory: &MultiModalMemory) -> Option<String> {
        match &memory.metadata.content_specific {
            ContentSpecificMetadata::Image { text_regions, .. } => {
                if !text_regions.is_empty() {
                    Some(text_regions.iter().map(|r| &r.text).cloned().collect::<Vec<_>>().join(" "))
                } else {
                    None
                }
            }
            ContentSpecificMetadata::Audio { transcript, .. } => {
                transcript.clone()
            }
            ContentSpecificMetadata::Code { .. } => {
                // For code, we could extract comments or function names
                None
            }
        }
    }
}

impl RelationshipStrategy for SemanticSimilarityStrategy {
    fn detect_relationships(
        &self,
        source: &MultiModalMemory,
        target: &MultiModalMemory,
    ) -> MultiModalResult<Vec<CrossModalLink>> {
        let mut relationships = Vec::new();

        // Extract text content from both memories
        let source_text = self.extract_text_content(source);
        let target_text = self.extract_text_content(target);

        if let (Some(source_text), Some(target_text)) = (source_text, target_text) {
            let similarity = self.calculate_text_similarity(&source_text, &target_text);
            
            if similarity >= self.similarity_threshold {
                relationships.push(CrossModalLink {
                    target_id: target.id.clone(),
                    relationship_type: CrossModalRelationship::SimilarTo,
                    confidence: similarity,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("similarity_score".to_string(), serde_json::Value::Number(
                            serde_json::Number::from_f64(similarity as f64).unwrap()
                        ));
                        meta.insert("comparison_method".to_string(), serde_json::Value::String("text_overlap".to_string()));
                        meta
                    },
                });
            }
        }

        Ok(relationships)
    }

    fn name(&self) -> &str {
        "semantic_similarity"
    }
}

/// Temporal relationship strategy
#[derive(Debug)]
pub struct TemporalRelationshipStrategy {
    time_window_minutes: i64,
}

impl TemporalRelationshipStrategy {
    pub fn new(time_window_minutes: i64) -> Self {
        Self { time_window_minutes }
    }
}

impl RelationshipStrategy for TemporalRelationshipStrategy {
    fn detect_relationships(
        &self,
        source: &MultiModalMemory,
        target: &MultiModalMemory,
    ) -> MultiModalResult<Vec<CrossModalLink>> {
        let mut relationships = Vec::new();

        // Check if memories were created within the time window
        let time_diff = (target.created_at - source.created_at).num_minutes().abs();
        
        if time_diff <= self.time_window_minutes {
            let confidence = 1.0 - (time_diff as f32 / self.time_window_minutes as f32);
            
            relationships.push(CrossModalLink {
                target_id: target.id.clone(),
                relationship_type: CrossModalRelationship::Custom("temporal_proximity".to_string()),
                confidence,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("time_diff_minutes".to_string(), serde_json::Value::Number(
                        serde_json::Number::from(time_diff)
                    ));
                    meta
                },
            });
        }

        Ok(relationships)
    }

    fn name(&self) -> &str {
        "temporal_relationship"
    }
}

/// Content generation relationship strategy
#[derive(Debug)]
pub struct ContentGenerationStrategy;

impl RelationshipStrategy for ContentGenerationStrategy {
    fn detect_relationships(
        &self,
        source: &MultiModalMemory,
        target: &MultiModalMemory,
    ) -> MultiModalResult<Vec<CrossModalLink>> {
        let mut relationships = Vec::new();

        // Check for generation patterns based on content types and metadata
        match (&source.content_type, &target.content_type) {
            // Code generating documentation
            (ContentType::Code { .. }, ContentType::ExtractedText { .. }) => {
                if let Some(source_desc) = &source.metadata.source {
                    if source_desc.contains("documentation") || source_desc.contains("comment") {
                        relationships.push(CrossModalLink {
                            target_id: target.id.clone(),
                            relationship_type: CrossModalRelationship::GeneratedFrom,
                            confidence: 0.8,
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert("generation_type".to_string(), serde_json::Value::String("documentation".to_string()));
                                meta
                            },
                        });
                    }
                }
            }
            // Image generating description
            (ContentType::Image { .. }, ContentType::ExtractedText { .. }) => {
                if let Some(target_desc) = &target.metadata.description {
                    if target_desc.contains("description") || target_desc.contains("caption") {
                        relationships.push(CrossModalLink {
                            target_id: target.id.clone(),
                            relationship_type: CrossModalRelationship::Describes,
                            confidence: 0.9,
                            metadata: {
                                let mut meta = HashMap::new();
                                meta.insert("description_type".to_string(), serde_json::Value::String("image_caption".to_string()));
                                meta
                            },
                        });
                    }
                }
            }
            _ => {}
        }

        Ok(relationships)
    }

    fn name(&self) -> &str {
        "content_generation"
    }
}

impl CrossModalAnalyzer {
    /// Create a new cross-modal analyzer
    pub fn new(config: CrossModalConfig) -> Self {
        let mut strategies: Vec<Box<dyn RelationshipStrategy>> = Vec::new();

        if config.enable_text_extraction {
            strategies.push(Box::new(TextExtractionStrategy::new(config.min_confidence_threshold)));
        }

        if config.enable_semantic_similarity {
            strategies.push(Box::new(SemanticSimilarityStrategy::new(config.similarity_threshold)));
        }

        if config.enable_temporal_relationships {
            strategies.push(Box::new(TemporalRelationshipStrategy::new(60))); // 1 hour window
        }

        if config.enable_generation_relationships {
            strategies.push(Box::new(ContentGenerationStrategy));
        }

        Self { config, strategies }
    }

    /// Analyze relationships between a source memory and a collection of target memories
    pub fn analyze_relationships(
        &self,
        source: &MultiModalMemory,
        targets: &[MultiModalMemory],
    ) -> MultiModalResult<Vec<CrossModalLink>> {
        let mut all_relationships = Vec::new();

        for target in targets {
            // Skip self-relationships
            if source.id == target.id {
                continue;
            }

            // Apply each strategy
            for strategy in &self.strategies {
                let relationships = strategy.detect_relationships(source, target)?;
                all_relationships.extend(relationships);
            }
        }

        // Filter by confidence threshold
        all_relationships.retain(|link| link.confidence >= self.config.min_confidence_threshold);

        // Sort by confidence (highest first)
        all_relationships.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Limit number of relationships
        all_relationships.truncate(self.config.max_relationships_per_memory);

        Ok(all_relationships)
    }

    /// Update memory with detected relationships
    pub fn update_memory_relationships(
        &self,
        memory: &mut MultiModalMemory,
        targets: &[MultiModalMemory],
    ) -> MultiModalResult<()> {
        let relationships = self.analyze_relationships(memory, targets)?;
        memory.cross_modal_links = relationships;
        memory.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Find memories related to a given memory
    pub fn find_related_memories(
        &self,
        source: &MultiModalMemory,
        candidates: &[MultiModalMemory],
        relationship_types: Option<&[CrossModalRelationship]>,
    ) -> MultiModalResult<Vec<(MemoryId, CrossModalRelationship, f32)>> {
        let relationships = self.analyze_relationships(source, candidates)?;
        
        let mut related = Vec::new();
        for link in relationships {
            // Filter by relationship type if specified
            if let Some(types) = relationship_types {
                if !types.contains(&link.relationship_type) {
                    continue;
                }
            }
            
            related.push((link.target_id, link.relationship_type, link.confidence));
        }

        Ok(related)
    }

    /// Get relationship statistics
    pub fn get_relationship_statistics(&self, memories: &[MultiModalMemory]) -> RelationshipStatistics {
        let mut stats = RelationshipStatistics::default();
        
        for memory in memories {
            stats.total_memories += 1;
            stats.total_relationships += memory.cross_modal_links.len();
            
            for link in &memory.cross_modal_links {
                *stats.relationship_type_counts.entry(link.relationship_type.clone()).or_insert(0) += 1;
                stats.confidence_sum += link.confidence;
            }
        }

        if stats.total_relationships > 0 {
            stats.average_confidence = stats.confidence_sum / stats.total_relationships as f32;
        }

        stats
    }
}

/// Statistics about cross-modal relationships
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RelationshipStatistics {
    pub total_memories: usize,
    pub total_relationships: usize,
    pub relationship_type_counts: HashMap<CrossModalRelationship, usize>,
    pub average_confidence: f32,
    pub confidence_sum: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_memory(content_type: ContentType) -> MultiModalMemory {
        MultiModalMemory {
            id: Uuid::new_v4().to_string(),
            content_type,
            primary_content: vec![],
            metadata: super::MultiModalMetadata {
                title: None,
                description: None,
                tags: vec![],
                source: None,
                quality_score: 0.8,
                processing_info: super::ProcessingInfo {
                    processor_version: "1.0.0".to_string(),
                    processing_time_ms: 100,
                    algorithms_used: vec![],
                    confidence_scores: HashMap::new(),
                },
                content_specific: ContentSpecificMetadata::Image {
                    dominant_colors: vec![],
                    detected_objects: vec![],
                    text_regions: vec![],
                    visual_features: vec![],
                },
            },
            extracted_features: HashMap::new(),
            cross_modal_links: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_cross_modal_analyzer_creation() {
        let config = CrossModalConfig::default();
        let analyzer = CrossModalAnalyzer::new(config);
        assert!(!analyzer.strategies.is_empty());
    }

    #[test]
    fn test_text_extraction_strategy() {
        let strategy = TextExtractionStrategy::new(0.7);
        
        let source = create_test_memory(ContentType::Image {
            format: super::ImageFormat::Png,
            width: 100,
            height: 100,
        });
        
        let target = create_test_memory(ContentType::ExtractedText {
            source_modality: Box::new(ContentType::Image {
                format: super::ImageFormat::Png,
                width: 100,
                height: 100,
            }),
            confidence: 0.8,
        });

        let relationships = strategy.detect_relationships(&source, &target).unwrap();
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].relationship_type, CrossModalRelationship::ExtractedFrom);
    }
}
