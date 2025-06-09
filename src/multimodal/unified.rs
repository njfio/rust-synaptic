//! # Unified Multi-Modal Memory System
//!
//! Unified interface for managing multi-modal memories across different content types.
//! Provides seamless integration with the core Synaptic memory system.

use super::{
    audio::AudioMemoryProcessor,
    code::CodeMemoryProcessor,
    cross_modal::{CrossModalAnalyzer, CrossModalConfig},
    image::ImageMemoryProcessor,
    ContentType, MultiModalMemory, MultiModalProcessor, MultiModalResult,
};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use crate::AgentMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Unified multi-modal memory manager
#[derive(Debug)]
pub struct UnifiedMultiModalMemory {
    /// Core memory system
    core_memory: Arc<RwLock<AgentMemory>>,
    
    /// Image processor
    #[cfg(feature = "image-memory")]
    image_processor: Option<ImageMemoryProcessor>,
    
    /// Audio processor
    #[cfg(feature = "audio-memory")]
    audio_processor: Option<AudioMemoryProcessor>,
    
    /// Code processor
    #[cfg(feature = "code-memory")]
    code_processor: Option<CodeMemoryProcessor>,
    
    /// Cross-modal relationship analyzer
    cross_modal_analyzer: CrossModalAnalyzer,
    
    /// Multi-modal memory storage
    multimodal_memories: Arc<RwLock<HashMap<MemoryId, MultiModalMemory>>>,
    
    /// Configuration
    config: UnifiedMultiModalConfig,
}

/// Configuration for unified multi-modal memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMultiModalConfig {
    /// Enable image memory processing
    pub enable_image_memory: bool,
    
    /// Enable audio memory processing
    pub enable_audio_memory: bool,
    
    /// Enable code memory processing
    pub enable_code_memory: bool,
    
    /// Enable cross-modal relationship detection
    pub enable_cross_modal_analysis: bool,
    
    /// Auto-detect content types
    pub auto_detect_content_type: bool,
    
    /// Maximum memory storage size (bytes)
    pub max_storage_size: usize,
    
    /// Cross-modal analysis configuration
    pub cross_modal_config: CrossModalConfig,
}

impl Default for UnifiedMultiModalConfig {
    fn default() -> Self {
        Self {
            enable_image_memory: true,
            enable_audio_memory: true,
            enable_code_memory: true,
            enable_cross_modal_analysis: true,
            auto_detect_content_type: true,
            max_storage_size: 1024 * 1024 * 1024, // 1GB
            cross_modal_config: CrossModalConfig::default(),
        }
    }
}

/// Multi-modal search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalQuery {
    /// Content to search for
    pub content: Vec<u8>,
    
    /// Content type (if known)
    pub content_type: Option<ContentType>,
    
    /// Search across specific modalities
    pub modalities: Option<Vec<String>>,
    
    /// Minimum similarity threshold
    pub similarity_threshold: f32,
    
    /// Maximum number of results
    pub max_results: usize,
    
    /// Include cross-modal relationships in results
    pub include_relationships: bool,
}

impl Default for MultiModalQuery {
    fn default() -> Self {
        Self {
            content: vec![],
            content_type: None,
            modalities: None,
            similarity_threshold: 0.7,
            max_results: 10,
            include_relationships: true,
        }
    }
}

/// Multi-modal search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalSearchResult {
    /// Found memory
    pub memory: MultiModalMemory,
    
    /// Similarity score
    pub similarity: f32,
    
    /// Related memories (if requested)
    pub related_memories: Vec<(MemoryId, String, f32)>, // (id, relationship_type, confidence)
}

impl UnifiedMultiModalMemory {
    /// Create a new unified multi-modal memory system
    pub async fn new(
        core_memory: Arc<RwLock<AgentMemory>>,
        config: UnifiedMultiModalConfig,
    ) -> MultiModalResult<Self> {
        // Initialize processors based on configuration
        #[cfg(feature = "image-memory")]
        let image_processor = if config.enable_image_memory {
            Some(ImageMemoryProcessor::new(Default::default())?)
        } else {
            None
        };

        #[cfg(feature = "audio-memory")]
        let audio_processor = if config.enable_audio_memory {
            Some(AudioMemoryProcessor::new(Default::default())?)
        } else {
            None
        };

        #[cfg(feature = "code-memory")]
        let code_processor = if config.enable_code_memory {
            Some(CodeMemoryProcessor::new(Default::default())?)
        } else {
            None
        };

        let cross_modal_analyzer = CrossModalAnalyzer::new(config.cross_modal_config.clone());

        Ok(Self {
            core_memory,
            #[cfg(feature = "image-memory")]
            image_processor,
            #[cfg(feature = "audio-memory")]
            audio_processor,
            #[cfg(feature = "code-memory")]
            code_processor,
            cross_modal_analyzer,
            multimodal_memories: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }

    /// Store multi-modal content
    pub async fn store_multimodal(
        &self,
        key: &str,
        content: Vec<u8>,
        content_type: Option<ContentType>,
    ) -> MultiModalResult<MemoryId> {
        // Auto-detect content type if not provided
        let detected_content_type = if let Some(ct) = content_type {
            ct
        } else if self.config.auto_detect_content_type {
            self.detect_content_type(&content).await?
        } else {
            return Err(SynapticError::ProcessingError("Content type not provided and auto-detection disabled".to_string()));
        };

        // Process content with appropriate processor
        let memory = self.process_content(&content, &detected_content_type).await?;

        // Store in multi-modal memory
        {
            let mut memories = self.multimodal_memories.write().await;
            memories.insert(memory.id.clone(), memory.clone());
        }

        // Store reference in core memory
        {
            let mut core = self.core_memory.write().await;
            core.store(key, &format!("multimodal:{}", memory.id)).await
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to store in core memory: {}", e)))?;
        }

        // Update cross-modal relationships if enabled
        if self.config.enable_cross_modal_analysis {
            self.update_cross_modal_relationships(&memory.id).await?;
        }

        Ok(memory.id)
    }

    /// Retrieve multi-modal content
    pub async fn retrieve_multimodal(&self, key: &str) -> MultiModalResult<Option<MultiModalMemory>> {
        // Get reference from core memory
        let core_value = {
            let core = self.core_memory.read().await;
            core.retrieve(key).await
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to retrieve from core memory: {}", e)))?
        };

        if let Some(value) = core_value {
            if let Some(memory_id) = value.strip_prefix("multimodal:") {
                let memories = self.multimodal_memories.read().await;
                Ok(memories.get(memory_id).cloned())
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Search across all modalities
    pub async fn search_multimodal(&self, query: MultiModalQuery) -> MultiModalResult<Vec<MultiModalSearchResult>> {
        let mut results = Vec::new();

        // Extract features from query content
        let query_content_type = if let Some(ct) = query.content_type {
            ct
        } else {
            self.detect_content_type(&query.content).await?
        };

        let query_features = self.extract_features(&query.content, &query_content_type).await?;

        // Search in multi-modal memories
        let memories = self.multimodal_memories.read().await;
        
        for memory in memories.values() {
            // Filter by modality if specified
            if let Some(ref modalities) = query.modalities {
                let memory_modality = self.get_modality_name(&memory.content_type);
                if !modalities.contains(&memory_modality) {
                    continue;
                }
            }

            // Calculate similarity
            let memory_features = self.get_memory_features(memory).await?;
            let similarity = self.calculate_similarity(&query_features, &memory_features).await?;

            if similarity >= query.similarity_threshold {
                let mut result = MultiModalSearchResult {
                    memory: memory.clone(),
                    similarity,
                    related_memories: vec![],
                };

                // Include related memories if requested
                if query.include_relationships {
                    for link in &memory.cross_modal_links {
                        if let Some(related_memory) = memories.get(&link.target_id) {
                            result.related_memories.push((
                                link.target_id.clone(),
                                format!("{:?}", link.relationship_type),
                                link.confidence,
                            ));
                        }
                    }
                }

                results.push(result);
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        results.truncate(query.max_results);

        Ok(results)
    }

    /// Delete multi-modal content
    pub async fn delete_multimodal(&self, key: &str) -> MultiModalResult<bool> {
        // Get memory ID from core memory
        let core_value = {
            let core = self.core_memory.read().await;
            core.retrieve(key).await
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to retrieve from core memory: {}", e)))?
        };

        if let Some(value) = core_value {
            if let Some(memory_id) = value.strip_prefix("multimodal:") {
                // Remove from multi-modal storage
                {
                    let mut memories = self.multimodal_memories.write().await;
                    memories.remove(memory_id);
                }

                // Remove from core memory
                {
                    let mut core = self.core_memory.write().await;
                    core.delete(key).await
                        .map_err(|e| SynapticError::ProcessingError(format!("Failed to delete from core memory: {}", e)))?;
                }

                // Update cross-modal relationships
                if self.config.enable_cross_modal_analysis {
                    self.cleanup_cross_modal_relationships(memory_id).await?;
                }

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get statistics about multi-modal memories
    pub async fn get_statistics(&self) -> MultiModalResult<MultiModalStatistics> {
        let memories = self.multimodal_memories.read().await;
        
        let mut stats = MultiModalStatistics::default();
        stats.total_memories = memories.len();

        for memory in memories.values() {
            let modality = self.get_modality_name(&memory.content_type);
            *stats.memories_by_modality.entry(modality).or_insert(0) += 1;
            stats.total_size += memory.primary_content.len();
            stats.total_relationships += memory.cross_modal_links.len();
        }

        if self.config.enable_cross_modal_analysis {
            let relationship_stats = self.cross_modal_analyzer.get_relationship_statistics(&memories.values().cloned().collect::<Vec<_>>());
            stats.average_relationship_confidence = relationship_stats.average_confidence;
        }

        Ok(stats)
    }

    /// Auto-detect content type from raw bytes
    async fn detect_content_type(&self, content: &[u8]) -> MultiModalResult<ContentType> {
        // Try image detection first
        #[cfg(feature = "image-memory")]
        if let Some(ref processor) = self.image_processor {
            if let Ok(format) = processor.detect_format(content) {
                // Load image to get dimensions
                if let Ok(img) = processor.load_image(content) {
                    return Ok(ContentType::Image {
                        format,
                        width: img.width(),
                        height: img.height(),
                    });
                }
            }
        }

        // Try audio detection
        #[cfg(feature = "audio-memory")]
        if let Some(ref processor) = self.audio_processor {
            if let Ok(format) = processor.detect_format(content) {
                // For now, return with default values
                return Ok(ContentType::Audio {
                    format,
                    duration_ms: 0, // Would need to parse audio to get actual duration
                    sample_rate: 44100,
                    channels: 2,
                });
            }
        }

        // Try code detection
        if let Ok(text_content) = String::from_utf8(content.to_vec()) {
            #[cfg(feature = "code-memory")]
            if let Some(ref processor) = self.code_processor {
                let language = processor.detect_language(&text_content, None);
                if !matches!(language, super::CodeLanguage::Other(_)) {
                    return Ok(ContentType::Code {
                        language,
                        lines: text_content.lines().count() as u32,
                        complexity_score: 0.0, // Would need analysis to calculate
                    });
                }
            }
        }

        Err(SynapticError::ProcessingError("Unable to detect content type".to_string()))
    }

    /// Process content with appropriate processor
    async fn process_content(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        match content_type {
            #[cfg(feature = "image-memory")]
            ContentType::Image { .. } => {
                if let Some(ref processor) = self.image_processor {
                    processor.process(content, content_type).await
                } else {
                    Err(SynapticError::ProcessingError("Image processor not available".to_string()))
                }
            }
            #[cfg(feature = "audio-memory")]
            ContentType::Audio { .. } => {
                if let Some(ref processor) = self.audio_processor {
                    processor.process(content, content_type).await
                } else {
                    Err(SynapticError::ProcessingError("Audio processor not available".to_string()))
                }
            }
            #[cfg(feature = "code-memory")]
            ContentType::Code { .. } => {
                if let Some(ref processor) = self.code_processor {
                    processor.process(content, content_type).await
                } else {
                    Err(SynapticError::ProcessingError("Code processor not available".to_string()))
                }
            }
            _ => Err(SynapticError::ProcessingError("Unsupported content type".to_string())),
        }
    }

    /// Extract features from content
    async fn extract_features(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        match content_type {
            #[cfg(feature = "image-memory")]
            ContentType::Image { .. } => {
                if let Some(ref processor) = self.image_processor {
                    processor.extract_features(content, content_type).await
                } else {
                    Ok(vec![])
                }
            }
            #[cfg(feature = "audio-memory")]
            ContentType::Audio { .. } => {
                if let Some(ref processor) = self.audio_processor {
                    processor.extract_features(content, content_type).await
                } else {
                    Ok(vec![])
                }
            }
            #[cfg(feature = "code-memory")]
            ContentType::Code { .. } => {
                if let Some(ref processor) = self.code_processor {
                    processor.extract_features(content, content_type).await
                } else {
                    Ok(vec![])
                }
            }
            _ => Ok(vec![]),
        }
    }

    /// Get features from stored memory
    async fn get_memory_features(&self, memory: &MultiModalMemory) -> MultiModalResult<Vec<f32>> {
        match &memory.metadata.content_specific {
            super::ContentSpecificMetadata::Image { visual_features, .. } => Ok(visual_features.clone()),
            super::ContentSpecificMetadata::Audio { audio_features, .. } => Ok(audio_features.clone()),
            super::ContentSpecificMetadata::Code { .. } => {
                if let Some(features_value) = memory.extracted_features.get("semantic_features") {
                    serde_json::from_value(features_value.clone())
                        .map_err(|e| SynapticError::ProcessingError(format!("Failed to deserialize features: {}", e)))
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Calculate similarity between feature vectors
    async fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> MultiModalResult<f32> {
        if features1.is_empty() || features2.is_empty() || features1.len() != features2.len() {
            return Ok(0.0);
        }

        // Calculate cosine similarity
        let dot_product: f32 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    /// Get modality name from content type
    fn get_modality_name(&self, content_type: &ContentType) -> String {
        match content_type {
            ContentType::Image { .. } => "image".to_string(),
            ContentType::Audio { .. } => "audio".to_string(),
            ContentType::Code { .. } => "code".to_string(),
            ContentType::ExtractedText { .. } => "text".to_string(),
        }
    }

    /// Update cross-modal relationships for a memory
    async fn update_cross_modal_relationships(&self, memory_id: &MemoryId) -> MultiModalResult<()> {
        let memories = self.multimodal_memories.read().await;
        let all_memories: Vec<_> = memories.values().cloned().collect();
        drop(memories);

        if let Some(memory) = all_memories.iter().find(|m| m.id == *memory_id) {
            let relationships = self.cross_modal_analyzer.analyze_relationships(memory, &all_memories)?;
            
            let mut memories = self.multimodal_memories.write().await;
            if let Some(stored_memory) = memories.get_mut(memory_id) {
                stored_memory.cross_modal_links = relationships;
                stored_memory.updated_at = chrono::Utc::now();
            }
        }

        Ok(())
    }

    /// Clean up cross-modal relationships when a memory is deleted
    async fn cleanup_cross_modal_relationships(&self, deleted_memory_id: &MemoryId) -> MultiModalResult<()> {
        let mut memories = self.multimodal_memories.write().await;
        
        for memory in memories.values_mut() {
            memory.cross_modal_links.retain(|link| link.target_id != *deleted_memory_id);
            if memory.cross_modal_links.iter().any(|link| link.target_id == *deleted_memory_id) {
                memory.updated_at = chrono::Utc::now();
            }
        }

        Ok(())
    }
}

/// Statistics about multi-modal memory usage
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MultiModalStatistics {
    pub total_memories: usize,
    pub total_size: usize,
    pub total_relationships: usize,
    pub memories_by_modality: HashMap<String, usize>,
    pub average_relationship_confidence: f32,
}
