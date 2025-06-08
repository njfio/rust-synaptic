//! # Multi-Modal Memory System
//! 
//! This module provides comprehensive multi-modal memory capabilities for the Synaptic AI agent memory system.
//! It supports image, audio, and code memory with advanced processing, analysis, and cross-modal relationships.
//!
//! ## Features
//!
//! ### Image Memory
//! - Computer vision and image understanding
//! - OCR (Optical Character Recognition) for text extraction
//! - Visual similarity detection and search
//! - Image-text relationship mapping
//! - Visual content categorization and tagging
//!
//! ### Audio Memory  
//! - Speech-to-text conversion and transcription
//! - Audio pattern recognition and fingerprinting
//! - Voice-based memory queries and interactions
//! - Audio similarity detection and clustering
//! - Real-time audio processing and analysis
//!
//! ### Code Memory
//! - Syntax-aware code understanding and parsing
//! - Code similarity detection and clone analysis
//! - API usage pattern recognition and recommendations
//! - Cross-language code relationship mapping
//! - Semantic code search and retrieval
//!
//! ## Architecture
//!
//! The multi-modal system is built on a unified memory model that allows:
//! - Cross-modal relationships and associations
//! - Unified search across all modalities
//! - Semantic understanding across different data types
//! - Real-time processing and analysis
//! - Scalable storage and retrieval

use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "image-memory")]
pub mod image;

#[cfg(feature = "audio-memory")]
pub mod audio;

#[cfg(feature = "code-memory")]
pub mod code;

pub mod cross_modal;
pub mod unified;

/// Multi-modal memory entry that can contain different types of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalMemory {
    pub id: MemoryId,
    pub content_type: ContentType,
    pub primary_content: Vec<u8>,
    pub metadata: MultiModalMetadata,
    pub extracted_features: HashMap<String, serde_json::Value>,
    pub cross_modal_links: Vec<CrossModalLink>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Types of content supported by the multi-modal system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// Image content (PNG, JPEG, WebP, etc.)
    Image {
        format: ImageFormat,
        width: u32,
        height: u32,
    },
    /// Audio content (WAV, MP3, FLAC, etc.)
    Audio {
        format: AudioFormat,
        duration_ms: u64,
        sample_rate: u32,
        channels: u16,
    },
    /// Code content (Rust, Python, JavaScript, etc.)
    Code {
        language: CodeLanguage,
        lines: u32,
        complexity_score: f32,
    },
    /// Text content extracted from other modalities
    ExtractedText {
        source_modality: Box<ContentType>,
        confidence: f32,
    },
}

/// Supported image formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
    Gif,
    Bmp,
    Tiff,
}

/// Supported audio formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Aac,
    M4a,
}

/// Supported programming languages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Java,
    CSharp,
    Cpp,
    C,
    Go,
    Swift,
    Kotlin,
    Ruby,
    Php,
    Html,
    Css,
    Sql,
    Shell,
    Other(String),
}

/// Metadata for multi-modal content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub quality_score: f32,
    pub processing_info: ProcessingInfo,
    pub content_specific: ContentSpecificMetadata,
}

/// Information about how the content was processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInfo {
    pub processor_version: String,
    pub processing_time_ms: u64,
    pub algorithms_used: Vec<String>,
    pub confidence_scores: HashMap<String, f32>,
}

/// Content-specific metadata based on the type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentSpecificMetadata {
    Image {
        dominant_colors: Vec<String>,
        detected_objects: Vec<DetectedObject>,
        text_regions: Vec<TextRegion>,
        visual_features: Vec<f32>,
    },
    Audio {
        transcript: Option<String>,
        speaker_info: Option<SpeakerInfo>,
        audio_features: Vec<f32>,
        detected_events: Vec<AudioEvent>,
    },
    Code {
        ast_summary: Option<String>,
        dependencies: Vec<String>,
        functions: Vec<FunctionInfo>,
        complexity_metrics: ComplexityMetrics,
    },
}

/// Detected object in an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
}

/// Bounding box coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Text region detected in an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub language: Option<String>,
}

/// Speaker information for audio content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerInfo {
    pub speaker_id: Option<String>,
    pub gender: Option<String>,
    pub age_estimate: Option<u32>,
    pub emotion: Option<String>,
    pub confidence: f32,
}

/// Audio event detected in audio content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    pub event_type: String,
    pub start_time_ms: u64,
    pub duration_ms: u64,
    pub confidence: f32,
}

/// Function information extracted from code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub complexity: u32,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: u32,
    pub cognitive_complexity: u32,
    pub lines_of_code: u32,
    pub maintainability_index: f32,
}

/// Cross-modal link between different types of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalLink {
    pub target_id: MemoryId,
    pub relationship_type: CrossModalRelationship,
    pub confidence: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of relationships between different modalities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CrossModalRelationship {
    /// Text extracted from image/audio
    ExtractedFrom,
    /// Content describes or explains another
    Describes,
    /// Content is similar in meaning/purpose
    SimilarTo,
    /// Content is part of a larger work
    PartOf,
    /// Content references or mentions another
    References,
    /// Content is a translation/conversion of another
    TranslationOf,
    /// Content is generated from another
    GeneratedFrom,
    /// Custom relationship type
    Custom(String),
}

/// Result type for multi-modal operations
pub type MultiModalResult<T> = Result<T, SynapticError>;

/// Trait for multi-modal content processors
pub trait MultiModalProcessor {
    /// Process content and extract features
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory>;
    
    /// Extract features for similarity comparison
    async fn extract_features(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<Vec<f32>>;
    
    /// Calculate similarity between two pieces of content
    async fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> MultiModalResult<f32>;
    
    /// Search for similar content
    async fn search_similar(&self, query_features: &[f32], candidates: &[MultiModalMemory]) -> MultiModalResult<Vec<(MemoryId, f32)>>;
}
