//! # Document Memory Processor
//!
//! Advanced document processing capabilities for PDF, Word documents, Markdown, and other text-based formats.
//! Provides intelligent content extraction, metadata analysis, and semantic understanding.

use super::{ContentType, MultiModalMemory, MultiModalMetadata, MultiModalProcessor, MultiModalResult};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Cursor};
use pdf_extract;
use zip::read::ZipArchive;
use quick_xml::Reader;
use quick_xml::events::Event;
use uuid::Uuid;

/// Document formats supported by the document processor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentFormat {
    /// PDF documents
    Pdf {
        pages: u32,
        version: String,
        encrypted: bool,
    },
    /// Microsoft Word documents (.docx)
    Docx {
        pages: u32,
        word_count: u32,
        has_images: bool,
    },
    /// Markdown documents
    Markdown {
        heading_count: u32,
        link_count: u32,
        image_count: u32,
    },
    /// Plain text documents
    PlainText {
        encoding: String,
        line_count: u32,
    },
    /// Rich Text Format
    Rtf {
        version: String,
        has_formatting: bool,
    },
    /// HTML documents
    Html {
        title: Option<String>,
        meta_description: Option<String>,
        link_count: u32,
    },
}

/// Document-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document format details
    pub format: DocumentFormat,
    /// Extracted text content
    pub text_content: String,
    /// Document title (extracted or inferred)
    pub title: Option<String>,
    /// Document author(s)
    pub authors: Vec<String>,
    /// Creation date
    pub created_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Last modified date
    pub modified_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Document language
    pub language: Option<String>,
    /// Word count
    pub word_count: u32,
    /// Character count
    pub char_count: u32,
    /// Extracted keywords
    pub keywords: Vec<String>,
    /// Document summary
    pub summary: Option<String>,
    /// Reading time estimate (minutes)
    pub reading_time_minutes: u32,
    /// Document structure (headings, sections)
    pub structure: DocumentStructure,
}

/// Document structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructure {
    /// Heading hierarchy
    pub headings: Vec<DocumentHeading>,
    /// Table of contents
    pub table_of_contents: Vec<TocEntry>,
    /// Sections and subsections
    pub sections: Vec<DocumentSection>,
    /// Footnotes and references
    pub references: Vec<String>,
}

/// Document heading information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentHeading {
    /// Heading level (1-6)
    pub level: u8,
    /// Heading text
    pub text: String,
    /// Position in document
    pub position: u32,
}

/// Table of contents entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Entry title
    pub title: String,
    /// Heading level
    pub level: u8,
    /// Page number (if applicable)
    pub page: Option<u32>,
    /// Position in document
    pub position: u32,
}

/// Document section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    /// Section title
    pub title: String,
    /// Section content
    pub content: String,
    /// Word count in section
    pub word_count: u32,
    /// Subsections
    pub subsections: Vec<DocumentSection>,
}

/// Document memory processor for text-based documents
#[derive(Debug, Clone)]
pub struct DocumentMemoryProcessor {
    /// Configuration for document processing
    config: DocumentProcessorConfig,
}

/// Configuration for document processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentProcessorConfig {
    /// Enable OCR for scanned documents
    pub enable_ocr: bool,
    /// Maximum document size to process (bytes)
    pub max_document_size: usize,
    /// Extract images from documents
    pub extract_images: bool,
    /// Generate document summary
    pub generate_summary: bool,
    /// Extract keywords automatically
    pub extract_keywords: bool,
    /// Supported languages for text analysis
    pub supported_languages: Vec<String>,
    /// Enable structure analysis
    pub analyze_structure: bool,
}

impl Default for DocumentProcessorConfig {
    fn default() -> Self {
        Self {
            enable_ocr: true,
            max_document_size: 100 * 1024 * 1024, // 100MB
            extract_images: true,
            generate_summary: true,
            extract_keywords: true,
            supported_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "it".to_string(),
                "pt".to_string(),
                "ru".to_string(),
                "zh".to_string(),
                "ja".to_string(),
                "ko".to_string(),
            ],
            analyze_structure: true,
        }
    }
}

impl DocumentMemoryProcessor {
    /// Create a new document processor
    pub fn new(config: DocumentProcessorConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DocumentProcessorConfig::default())
    }

    /// Detect document format from content
    pub fn detect_document_format(&self, content: &[u8]) -> MultiModalResult<DocumentFormat> {
        // Check file signatures and content patterns
        if content.starts_with(b"%PDF") {
            Ok(DocumentFormat::Pdf {
                pages: 0, // Would need proper parsing
                version: "1.4".to_string(), // Default version
                encrypted: false,
            })
        } else if content.starts_with(b"PK\x03\x04") {
            // ZIP-based format (likely DOCX)
            Ok(DocumentFormat::Docx {
                pages: 0,
                word_count: 0,
                has_images: false,
            })
        } else if self.is_markdown_content(content) {
            Ok(DocumentFormat::Markdown {
                heading_count: 0,
                link_count: 0,
                image_count: 0,
            })
        } else if content.starts_with(b"<!DOCTYPE html") || content.starts_with(b"<html") {
            Ok(DocumentFormat::Html {
                title: None,
                meta_description: None,
                link_count: 0,
            })
        } else if content.starts_with(b"{\\rtf") {
            Ok(DocumentFormat::Rtf {
                version: "1.0".to_string(),
                has_formatting: true,
            })
        } else {
            // Assume plain text
            let encoding = self.detect_encoding(content);
            let line_count = content.iter().filter(|&&b| b == b'\n').count() as u32;
            Ok(DocumentFormat::PlainText {
                encoding,
                line_count,
            })
        }
    }

    /// Check if content appears to be Markdown
    fn is_markdown_content(&self, content: &[u8]) -> bool {
        let text = String::from_utf8_lossy(content);
        let lines: Vec<&str> = text.lines().take(20).collect(); // Check first 20 lines
        
        let markdown_indicators = [
            "# ", "## ", "### ", // Headers
            "- ", "* ", "+ ",    // Lists
            "```", "~~~",        // Code blocks
            "[", "](", "![",     // Links and images
            "|", "---",          // Tables and horizontal rules
        ];
        
        let indicator_count = lines.iter()
            .map(|line| {
                markdown_indicators.iter()
                    .filter(|&indicator| line.contains(indicator))
                    .count()
            })
            .sum::<usize>();
        
        indicator_count >= 2 // At least 2 markdown indicators
    }

    /// Detect text encoding
    fn detect_encoding(&self, content: &[u8]) -> String {
        // Simple encoding detection - in a real implementation, use encoding_rs
        if content.starts_with(&[0xEF, 0xBB, 0xBF]) {
            "UTF-8 BOM".to_string()
        } else if content.iter().all(|&b| b < 128) {
            "ASCII".to_string()
        } else {
            "UTF-8".to_string()
        }
    }

    /// Extract text content from document
    pub async fn extract_text(&self, content: &[u8], format: &DocumentFormat) -> MultiModalResult<String> {
        match format {
            DocumentFormat::Pdf { .. } => self.extract_pdf_text(content).await,
            DocumentFormat::Docx { .. } => self.extract_docx_text(content).await,
            DocumentFormat::Markdown { .. } => self.extract_markdown_text(content).await,
            DocumentFormat::Html { .. } => self.extract_html_text(content).await,
            DocumentFormat::Rtf { .. } => self.extract_rtf_text(content).await,
            DocumentFormat::PlainText { .. } => {
                Ok(String::from_utf8_lossy(content).to_string())
            }
        }
    }

    /// Extract text from PDF using pdf-extract
    async fn extract_pdf_text(&self, content: &[u8]) -> MultiModalResult<String> {
        let mut cursor = std::io::Cursor::new(content);
        match pdf_extract::extract_text_from_reader(&mut cursor) {
            Ok(text) => Ok(text),
            Err(e) => Err(SynapticError::ProcessingError(format!("PDF extraction failed: {e}")))
        }
    }

    /// Extract text from DOCX using zip and quick-xml
    async fn extract_docx_text(&self, content: &[u8]) -> MultiModalResult<String> {
        let mut archive = ZipArchive::new(std::io::Cursor::new(content))
            .map_err(|e| SynapticError::ProcessingError(format!("DOCX zip error: {e}")))?;
        let mut doc_xml = String::new();
        archive
            .by_name("word/document.xml")
            .map_err(|e| SynapticError::ProcessingError(format!("DOCX entry error: {e}")))?
            .read_to_string(&mut doc_xml)
            .map_err(|e| SynapticError::ProcessingError(format!("DOCX read error: {e}")))?;
        let mut reader = Reader::from_str(&doc_xml);
        reader.trim_text(true);
        let mut buf = Vec::new();
        let mut text = String::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Text(e)) => text.push_str(&e.unescape().unwrap_or_default()),
                Ok(Event::Eof) => break,
                Err(e) => return Err(SynapticError::ProcessingError(format!("DOCX parse error: {e}"))),
                _ => {}
            }
            buf.clear();
        }
        Ok(text)
    }

    /// Extract text from Markdown
    async fn extract_markdown_text(&self, content: &[u8]) -> MultiModalResult<String> {
        let markdown_text = String::from_utf8_lossy(content);
        // In a real implementation, use pulldown-cmark to parse and extract plain text
        Ok(markdown_text.to_string())
    }

    /// Extract text from HTML
    async fn extract_html_text(&self, content: &[u8]) -> MultiModalResult<String> {
        let html_text = String::from_utf8_lossy(content);
        // Basic HTML tag removal (in real implementation, use html5ever or similar)
        let text = html_text
            .replace("<br>", "\n")
            .replace("<p>", "\n")
            .replace("</p>", "\n");
        
        // Remove HTML tags (very basic implementation)
        let mut result = String::new();
        let mut in_tag = false;
        for ch in text.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        
        Ok(result)
    }

    /// Extract text from RTF
    async fn extract_rtf_text(&self, content: &[u8]) -> MultiModalResult<String> {
        let rtf_text = String::from_utf8_lossy(content);
        // Basic RTF parsing (remove control words)
        let mut result = String::new();
        let mut in_control = false;
        
        for ch in rtf_text.chars() {
            match ch {
                '\\' => in_control = true,
                ' ' | '\n' | '\r' if in_control => {
                    in_control = false;
                    result.push(' ');
                }
                '{' | '}' => {} // Skip braces
                _ if !in_control => result.push(ch),
                _ => {}
            }
        }
        
        Ok(result.trim().to_string())
    }

    /// Analyze document structure
    pub async fn analyze_structure(&self, text: &str, format: &DocumentFormat) -> MultiModalResult<DocumentStructure> {
        let headings = self.extract_headings(text, format).await?;
        let table_of_contents = self.generate_toc(&headings);
        let sections = self.extract_sections(text, &headings).await?;
        let references = self.extract_references(text).await?;

        Ok(DocumentStructure {
            headings,
            table_of_contents,
            sections,
            references,
        })
    }

    /// Extract headings from text
    async fn extract_headings(&self, text: &str, format: &DocumentFormat) -> MultiModalResult<Vec<DocumentHeading>> {
        let mut headings = Vec::new();

        match format {
            DocumentFormat::Markdown { .. } => {
                for (line_num, line) in text.lines().enumerate() {
                    if line.starts_with('#') {
                        let level = line.chars().take_while(|&c| c == '#').count() as u8;
                        let text = line.trim_start_matches('#').trim().to_string();
                        if !text.is_empty() && level <= 6 {
                            headings.push(DocumentHeading {
                                level,
                                text,
                                position: line_num as u32,
                            });
                        }
                    }
                }
            }
            DocumentFormat::Html { .. } => {
                // Basic HTML heading extraction
                for (line_num, line) in text.lines().enumerate() {
                    for level in 1..=6 {
                        let start_tag = format!("<h{}>", level);
                        let end_tag = format!("</h{}>", level);
                        if let Some(start) = line.find(&start_tag) {
                            if let Some(end) = line.find(&end_tag) {
                                let heading_text = line[start + start_tag.len()..end].trim().to_string();
                                if !heading_text.is_empty() {
                                    headings.push(DocumentHeading {
                                        level: level as u8,
                                        text: heading_text,
                                        position: line_num as u32,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // For other formats, try to detect headings by patterns
                for (line_num, line) in text.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.len() > 0 && trimmed.len() < 100 {
                        // Check if line looks like a heading (all caps, short, etc.)
                        if trimmed.chars().all(|c| c.is_uppercase() || c.is_whitespace() || c.is_numeric()) {
                            headings.push(DocumentHeading {
                                level: 1,
                                text: trimmed.to_string(),
                                position: line_num as u32,
                            });
                        }
                    }
                }
            }
        }

        Ok(headings)
    }

    /// Generate table of contents from headings
    fn generate_toc(&self, headings: &[DocumentHeading]) -> Vec<TocEntry> {
        headings.iter().map(|heading| TocEntry {
            title: heading.text.clone(),
            level: heading.level,
            page: None, // Page numbers would need document layout analysis
            position: heading.position,
        }).collect()
    }

    /// Extract sections based on headings
    async fn extract_sections(&self, text: &str, headings: &[DocumentHeading]) -> MultiModalResult<Vec<DocumentSection>> {
        let lines: Vec<&str> = text.lines().collect();
        let mut sections = Vec::new();

        for (i, heading) in headings.iter().enumerate() {
            let start_line = heading.position as usize;
            let end_line = if i + 1 < headings.len() {
                headings[i + 1].position as usize
            } else {
                lines.len()
            };

            if start_line < lines.len() && end_line <= lines.len() {
                let section_lines = &lines[start_line + 1..end_line];
                let content = section_lines.join("\n").trim().to_string();
                let word_count = content.split_whitespace().count() as u32;

                sections.push(DocumentSection {
                    title: heading.text.clone(),
                    content,
                    word_count,
                    subsections: Vec::new(), // Could be implemented recursively
                });
            }
        }

        Ok(sections)
    }

    /// Extract references and citations
    async fn extract_references(&self, text: &str) -> MultiModalResult<Vec<String>> {
        let mut references = Vec::new();

        // Look for common reference patterns
        for line in text.lines() {
            // URLs
            if line.contains("http://") || line.contains("https://") {
                references.push(line.trim().to_string());
            }
            // DOI patterns
            if line.contains("doi:") || line.contains("DOI:") {
                references.push(line.trim().to_string());
            }
            // Citation patterns [1], (Smith, 2023), etc.
            if line.contains('[') && line.contains(']') {
                references.push(line.trim().to_string());
            }
        }

        Ok(references)
    }

    /// Extract keywords from text
    pub async fn extract_keywords(&self, text: &str) -> MultiModalResult<Vec<String>> {
        let words: Vec<&str> = text
            .split_whitespace()
            .filter(|word| word.len() > 3) // Filter short words
            .collect();

        // Simple keyword extraction (in real implementation, use TF-IDF or NLP)
        let mut word_counts = HashMap::new();
        for word in words {
            let clean_word = word.to_lowercase()
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_string();
            if !clean_word.is_empty() && !self.is_stop_word(&clean_word) {
                *word_counts.entry(clean_word).or_insert(0) += 1;
            }
        }

        // Get most frequent words as keywords
        let mut keywords: Vec<(String, usize)> = word_counts.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(keywords.into_iter()
            .take(10) // Top 10 keywords
            .map(|(word, _)| word)
            .collect())
    }

    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "this", "that", "these", "those", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "shall", "a", "an",
        ];
        STOP_WORDS.contains(&word)
    }

    /// Generate document summary
    pub async fn generate_summary(&self, text: &str) -> MultiModalResult<String> {
        let sentences: Vec<&str> = text.split('.').collect();
        if sentences.len() <= 3 {
            return Ok(text.to_string());
        }

        // Simple extractive summarization (take first and last sentences)
        let summary = format!("{}. {}",
            sentences.first().unwrap_or(&"").trim(),
            sentences.last().unwrap_or(&"").trim()
        );

        Ok(summary)
    }

    /// Calculate reading time estimate
    pub fn calculate_reading_time(&self, word_count: u32) -> u32 {
        // Average reading speed: 200-250 words per minute
        const WORDS_PER_MINUTE: u32 = 225;
        (word_count + WORDS_PER_MINUTE - 1) / WORDS_PER_MINUTE // Round up
    }
}

#[async_trait::async_trait]
impl MultiModalProcessor for DocumentMemoryProcessor {
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        if content.len() > self.config.max_document_size {
            return Err(SynapticError::ProcessingError(
                format!("Document size {} exceeds maximum {}", content.len(), self.config.max_document_size)
            ));
        }

        // Detect document format
        let format = self.detect_document_format(content)?;

        // Extract text content
        let text_content = self.extract_text(content, &format).await?;

        // Calculate basic metrics
        let word_count = text_content.split_whitespace().count() as u32;
        let char_count = text_content.chars().count() as u32;
        let reading_time = self.calculate_reading_time(word_count);

        // Extract keywords if enabled
        let keywords = if self.config.extract_keywords {
            self.extract_keywords(&text_content).await?
        } else {
            Vec::new()
        };

        // Generate summary if enabled
        let summary = if self.config.generate_summary {
            Some(self.generate_summary(&text_content).await?)
        } else {
            None
        };

        // Analyze structure if enabled
        let structure = if self.config.analyze_structure {
            self.analyze_structure(&text_content, &format).await?
        } else {
            DocumentStructure {
                headings: Vec::new(),
                table_of_contents: Vec::new(),
                sections: Vec::new(),
                references: Vec::new(),
            }
        };

        // Create document metadata
        let doc_metadata = DocumentMetadata {
            format,
            text_content: text_content.clone(),
            title: None, // Would be extracted from document properties
            authors: Vec::new(),
            created_date: None,
            modified_date: None,
            language: Some("en".to_string()), // Would use language detection
            word_count,
            char_count,
            keywords,
            summary,
            reading_time_minutes: reading_time,
            structure,
        };

        // Create multi-modal metadata
        let metadata = MultiModalMetadata {
            title: doc_metadata.title.clone(),
            description: doc_metadata.summary.clone(),
            tags: doc_metadata.keywords.clone(),
            quality_score: 0.8, // Default quality score
            confidence: 0.9,
            processing_time_ms: 100, // Placeholder
            extracted_features: HashMap::new(),
        };

        // Create memory entry
        let memory = MultiModalMemory {
            id: Uuid::new_v4().to_string(),
            content_type: content_type.clone(),
            primary_content: content.to_vec(),
            metadata,
            extracted_features: {
                let mut features = HashMap::new();
                features.insert("document_metadata".to_string(), serde_json::to_value(doc_metadata)?);
                features
            },
            cross_modal_links: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        Ok(memory)
    }

    async fn extract_features(&self, content: &[u8], _content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        // Extract text and create feature vector
        let format = self.detect_document_format(content)?;
        let text = self.extract_text(content, &format).await?;

        // Simple feature extraction (in real implementation, use embeddings)
        let word_count = text.split_whitespace().count() as f32;
        let char_count = text.chars().count() as f32;
        let line_count = text.lines().count() as f32;
        let avg_word_length = if word_count > 0.0 { char_count / word_count } else { 0.0 };

        Ok(vec![word_count, char_count, line_count, avg_word_length])
    }

    async fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> MultiModalResult<f32> {
        if features1.len() != features2.len() {
            return Ok(0.0);
        }

        // Cosine similarity
        let dot_product: f32 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    async fn search_similar(&self, query_features: &[f32], candidates: &[MultiModalMemory]) -> MultiModalResult<Vec<(MemoryId, f32)>> {
        let mut results = Vec::new();

        for memory in candidates {
            // Extract features from stored memory
            if let Some(features_value) = memory.extracted_features.get("features") {
                if let Ok(features) = serde_json::from_value::<Vec<f32>>(features_value.clone()) {
                    let similarity = self.calculate_similarity(query_features, &features).await?;
                    results.push((memory.id.clone(), similarity));
                }
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}
