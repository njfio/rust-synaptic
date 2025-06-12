//! # Image Memory System
//!
//! Advanced image processing and computer vision capabilities for the Synaptic memory system.
//! Provides OCR, object detection, visual similarity, and image-text relationship mapping.

use super::{
    BoundingBox, ContentSpecificMetadata, ContentType, DetectedObject, ImageFormat,
    MultiModalMemory, MultiModalMetadata, MultiModalProcessor, MultiModalResult,
    ProcessingInfo, TextRegion,
};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "image-memory")]
use {
    image::{DynamicImage, ImageFormat as ImgFormat},
    imageproc::geometric_transformations::*,
    rusttype::{Font, Scale},
};

#[cfg(all(feature = "image-memory", feature = "tesseract"))]
use {
    std::sync::Mutex,
    tesseract::Tesseract,
};

#[cfg(all(feature = "image-memory", feature = "opencv"))]
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgproc::{self, INTER_LINEAR},
    objdetect::HOGDescriptor,
    prelude::*,
};

/// Image memory processor with computer vision capabilities
#[derive(Debug)]
pub struct ImageMemoryProcessor {
    /// OCR engine for text extraction
    #[cfg(feature = "tesseract")]
    ocr_engine: Option<Mutex<Tesseract>>,
    
    /// Object detection models
    #[cfg(feature = "opencv")]
    object_detector: Option<HOGDescriptor>,
    
    /// Configuration settings
    config: ImageProcessorConfig,
}

/// Configuration for image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessorConfig {
    /// Enable OCR text extraction
    pub enable_ocr: bool,
    
    /// Enable object detection
    pub enable_object_detection: bool,
    
    /// Enable visual feature extraction
    pub enable_visual_features: bool,
    
    /// Maximum image size for processing (width * height)
    pub max_image_size: u32,
    
    /// OCR confidence threshold
    pub ocr_confidence_threshold: f32,
    
    /// Object detection confidence threshold
    pub object_detection_threshold: f32,
    
    /// Supported image formats
    pub supported_formats: Vec<ImageFormat>,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            enable_ocr: true,
            enable_object_detection: true,
            enable_visual_features: true,
            max_image_size: 4096 * 4096, // 16MP max
            ocr_confidence_threshold: 0.7,
            object_detection_threshold: 0.5,
            supported_formats: vec![
                ImageFormat::Png,
                ImageFormat::Jpeg,
                ImageFormat::WebP,
                ImageFormat::Gif,
                ImageFormat::Bmp,
            ],
        }
    }
}

impl ImageMemoryProcessor {
    /// Create a new image memory processor
    pub fn new(config: ImageProcessorConfig) -> MultiModalResult<Self> {
        let mut processor = Self {
            #[cfg(feature = "tesseract")]
            ocr_engine: None,
            #[cfg(feature = "opencv")]
            object_detector: None,
            config,
        };

        // Initialize OCR engine
        #[cfg(feature = "tesseract")]
        if processor.config.enable_ocr {
            processor.ocr_engine = Some(Mutex::new(
                Tesseract::new(None, Some("eng"))
                    .map_err(|e| SynapticError::ProcessingError(format!("Failed to initialize OCR: {}", e)))?,
            ));
        }

        // Initialize object detector
        #[cfg(feature = "opencv")]
        if processor.config.enable_object_detection {
            let mut hog = HOGDescriptor::default()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to initialize HOG: {}", e)))?;
            hog.set_svm_detector(&HOGDescriptor::get_default_people_detector()?)
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to set SVM detector: {}", e)))?;
            processor.object_detector = Some(hog);
        }

        Ok(processor)
    }

    /// Load and validate an image from bytes
    #[cfg(feature = "image-memory")]
    pub fn load_image(&self, data: &[u8]) -> MultiModalResult<DynamicImage> {
        let img = image::load_from_memory(data)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to load image: {}", e)))?;

        // Check image size limits
        let (width, height) = (img.width(), img.height());
        if width * height > self.config.max_image_size {
            return Err(SynapticError::ProcessingError(format!(
                "Image too large: {}x{} exceeds limit of {} pixels",
                width, height, self.config.max_image_size
            )));
        }

        Ok(img)
    }

    /// Extract text from image using OCR
    #[cfg(all(feature = "image-memory", feature = "tesseract"))]
    pub async fn extract_text(&self, img: &DynamicImage) -> MultiModalResult<Vec<TextRegion>> {
        if !self.config.enable_ocr || self.ocr_engine.is_none() {
            return Ok(vec![]);
        }

        let mut ocr = self
            .ocr_engine
            .as_ref()
            .unwrap()
            .lock()
            .map_err(|_| SynapticError::ProcessingError("Failed to lock OCR engine".to_string()))?;
        
        // Convert image to grayscale for better OCR
        let gray_img = img.to_luma8();
        let (width, height) = gray_img.dimensions();
        
        // Set image data for OCR
        ocr.set_image(&gray_img.into_raw(), width as i32, height as i32, 1, width as i32)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to set OCR image: {}", e)))?;

        // Extract text with bounding boxes
        let text = ocr.get_text()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to extract text: {}", e)))?;

        // Get confidence scores and bounding boxes
        let confidences = ocr.get_confidences()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get confidences: {}", e)))?;

        let mut text_regions = Vec::new();
        
        if !text.trim().is_empty() {
            // For now, create a single text region covering the whole image
            // In a real implementation, you'd use tesseract's word/character level detection
            let avg_confidence = confidences.iter().sum::<i32>() as f32 / confidences.len() as f32 / 100.0;
            
            if avg_confidence >= self.config.ocr_confidence_threshold {
                text_regions.push(TextRegion {
                    text: text.trim().to_string(),
                    confidence: avg_confidence,
                    bounding_box: BoundingBox {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                    },
                    language: Some("en".to_string()),
                });
            }
        }

        Ok(text_regions)
    }

    /// Detect objects in image
    #[cfg(all(feature = "image-memory", feature = "opencv"))]
    pub async fn detect_objects(&self, img: &DynamicImage) -> MultiModalResult<Vec<DetectedObject>> {
        if !self.config.enable_object_detection || self.object_detector.is_none() {
            return Ok(vec![]);
        }

        let detector = self.object_detector.as_ref().unwrap();
        
        // Convert image to OpenCV Mat
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        
        let mat = Mat::from_slice_2d(&[rgb_img.into_raw()])
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create Mat: {}", e)))?;

        // Detect objects (people in this case with default HOG detector)
        let mut locations = Vector::<Rect>::new();
        let mut weights = Vector::<f64>::new();
        
        detector.detect_multi_scale(
            &mat,
            &mut locations,
            &mut weights,
            0.0, // hit threshold
            Size::new(8, 8), // win stride
            Size::new(32, 32), // padding
            1.05, // scale
            2, // final threshold
            false, // use meanshift grouping
        ).map_err(|e| SynapticError::ProcessingError(format!("Failed to detect objects: {}", e)))?;

        let mut detected_objects = Vec::new();
        
        for (i, rect) in locations.iter().enumerate() {
            let weight = weights.get(i).unwrap_or(0.0) as f32;
            
            if weight >= self.config.object_detection_threshold {
                detected_objects.push(DetectedObject {
                    label: "person".to_string(), // HOG default detector is for people
                    confidence: weight,
                    bounding_box: BoundingBox {
                        x: rect.x as f32,
                        y: rect.y as f32,
                        width: rect.width as f32,
                        height: rect.height as f32,
                    },
                });
            }
        }

        Ok(detected_objects)
    }

    /// Extract visual features for similarity comparison
    #[cfg(feature = "image-memory")]
    pub async fn extract_visual_features(&self, img: &DynamicImage) -> MultiModalResult<Vec<f32>> {
        if !self.config.enable_visual_features {
            return Ok(vec![]);
        }

        // Resize image to standard size for feature extraction
        let resized = img.resize(224, 224, image::imageops::FilterType::Lanczos3);
        let rgb_img = resized.to_rgb8();
        
        // Extract simple color histogram features
        let mut features = Vec::with_capacity(768); // 256 * 3 channels
        
        // Color histogram for each channel
        for channel in 0..3 {
            let mut histogram = vec![0u32; 256];
            
            for pixel in rgb_img.pixels() {
                let value = pixel.0[channel] as usize;
                histogram[value] += 1;
            }
            
            // Normalize histogram
            let total_pixels = (224 * 224) as f32;
            for &count in &histogram {
                features.push(count as f32 / total_pixels);
            }
        }

        Ok(features)
    }

    /// Calculate dominant colors in the image
    #[cfg(feature = "image-memory")]
    pub fn calculate_dominant_colors(&self, img: &DynamicImage, num_colors: usize) -> Vec<String> {
        let rgb_img = img.resize(100, 100, image::imageops::FilterType::Nearest).to_rgb8();
        let mut color_counts: HashMap<[u8; 3], u32> = HashMap::new();
        
        // Count color occurrences
        for pixel in rgb_img.pixels() {
            let rgb = [pixel.0[0], pixel.0[1], pixel.0[2]];
            *color_counts.entry(rgb).or_insert(0) += 1;
        }
        
        // Sort by frequency and take top colors
        let mut colors: Vec<_> = color_counts.into_iter().collect();
        colors.sort_by(|a, b| b.1.cmp(&a.1));
        
        colors
            .into_iter()
            .take(num_colors)
            .map(|(rgb, _)| format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]))
            .collect()
    }

    /// Determine image format from content
    pub fn detect_format(&self, data: &[u8]) -> MultiModalResult<ImageFormat> {
        // Check magic bytes to determine format
        if data.len() < 8 {
            return Err(SynapticError::ProcessingError("Image data too short".to_string()));
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return Ok(ImageFormat::Png);
        }

        // JPEG: FF D8 FF
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Ok(ImageFormat::Jpeg);
        }

        // WebP: RIFF ... WEBP
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return Ok(ImageFormat::WebP);
        }

        // GIF: GIF87a or GIF89a
        if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            return Ok(ImageFormat::Gif);
        }

        // BMP: BM
        if data.starts_with(b"BM") {
            return Ok(ImageFormat::Bmp);
        }

        // TIFF: II or MM
        if data.starts_with(b"II") || data.starts_with(b"MM") {
            return Ok(ImageFormat::Tiff);
        }

        Err(SynapticError::ProcessingError("Unknown image format".to_string()))
    }
}

#[cfg(feature = "image-memory")]
#[async_trait::async_trait]
impl MultiModalProcessor for ImageMemoryProcessor {
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        let start_time = std::time::Instant::now();
        
        // Validate content type
        let (format, width, height) = match content_type {
            ContentType::Image { format, width, height } => (format.clone(), *width, *height),
            _ => return Err(SynapticError::ProcessingError("Invalid content type for image processor".to_string())),
        };

        // Load image
        let img = self.load_image(content)?;
        
        // Extract text regions using OCR if available
        #[cfg(feature = "tesseract")]
        let text_regions = self.extract_text(&img).await?;
        #[cfg(not(feature = "tesseract"))]
        let text_regions = vec![];
        
        // Detect objects
        #[cfg(feature = "opencv")]
        let detected_objects = self.detect_objects(&img).await?;
        #[cfg(not(feature = "opencv"))]
        let detected_objects = vec![];
        
        // Extract visual features
        let visual_features = self.extract_visual_features(&img).await?;
        
        // Calculate dominant colors
        let dominant_colors = self.calculate_dominant_colors(&img, 5);
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let memory = MultiModalMemory {
            id: Uuid::new_v4().to_string(),
            content_type: content_type.clone(),
            primary_content: content.to_vec(),
            metadata: MultiModalMetadata {
                title: None,
                description: None,
                tags: vec![],
                source: None,
                quality_score: 0.8, // Default quality score
                processing_info: ProcessingInfo {
                    processor_version: "1.0.0".to_string(),
                    processing_time_ms: processing_time,
                    algorithms_used: vec!["color_histogram".to_string(), "object_detection".to_string()],
                    confidence_scores: HashMap::new(),
                },
                content_specific: ContentSpecificMetadata::Image {
                    dominant_colors,
                    detected_objects,
                    text_regions,
                    visual_features,
                },
            },
            extracted_features: HashMap::new(),
            cross_modal_links: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        Ok(memory)
    }

    async fn extract_features(&self, content: &[u8], _content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        let img = self.load_image(content)?;
        self.extract_visual_features(&img).await
    }

    async fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> MultiModalResult<f32> {
        if features1.len() != features2.len() {
            return Err(SynapticError::ProcessingError("Feature vectors must have same length".to_string()));
        }

        // Calculate cosine similarity
        let dot_product: f32 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }

    async fn search_similar(&self, query_features: &[f32], candidates: &[MultiModalMemory]) -> MultiModalResult<Vec<(MemoryId, f32)>> {
        let mut similarities = Vec::new();

        for candidate in candidates {
            if let ContentSpecificMetadata::Image { visual_features, .. } = &candidate.metadata.content_specific {
                let similarity = self.calculate_similarity(query_features, visual_features).await?;
                similarities.push((candidate.id.clone(), similarity));
            }
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities)
    }
}
