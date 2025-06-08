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

// Phase 5B: Advanced Document Processing
#[cfg(feature = "document-memory")]
pub mod document;

#[cfg(feature = "data-memory")]
pub mod data;

#[cfg(feature = "folder-processing")]
pub mod folder;

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    /// Image content (PNG, JPEG, WebP, etc.)
    Image {
        format: String,
        width: u32,
        height: u32,
    },
    /// Audio content (WAV, MP3, FLAC, etc.)
    Audio {
        format: String,
        duration_ms: u64,
    },
    /// Code content (Rust, Python, JavaScript, etc.)
    Code {
        language: String,
        framework: Option<String>,
    },
    /// Document content (PDF, Word, Markdown, etc.)
    Document {
        format: String,
        language: Option<String>,
    },
    /// Data content (CSV, JSON, Parquet, etc.)
    Data {
        format: String,
        schema: Option<String>,
    },
    /// Text content extracted from other modalities
    Text {
        language: Option<String>,
    },
}



/// Metadata for multi-modal content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub quality_score: f64,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub extracted_features: HashMap<String, serde_json::Value>,
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
#[async_trait::async_trait]
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
