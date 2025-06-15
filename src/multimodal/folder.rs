//! # Folder Memory Processor
//!
//! Advanced folder and batch processing capabilities for handling multiple files and directories.
//! Provides intelligent content discovery, batch processing, and hierarchical organization.

use super::{ContentType, MultiModalMemory, MultiModalMetadata, MultiModalProcessor, MultiModalResult};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[cfg(feature = "folder-processing")]
use walkdir::WalkDir;

/// Folder processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderProcessingResult {
    /// Root folder path
    pub root_path: PathBuf,
    /// Total files processed
    pub total_files: u64,
    /// Total directories processed
    pub total_directories: u64,
    /// Total size processed (bytes)
    pub total_size: u64,
    /// Processing duration
    pub processing_duration: std::time::Duration,
    /// File type distribution
    pub file_type_distribution: HashMap<String, u64>,
    /// Processing errors
    pub errors: Vec<ProcessingError>,
    /// Processed memories
    pub memories: Vec<MultiModalMemory>,
    /// Folder hierarchy
    pub hierarchy: FolderHierarchy,
}

/// Processing error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingError {
    /// File path that caused the error
    pub file_path: PathBuf,
    /// Error message
    pub error_message: String,
    /// Error type
    pub error_type: ErrorType,
}

/// Types of processing errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ErrorType {
    FileNotFound,
    PermissionDenied,
    UnsupportedFormat,
    ProcessingFailed,
    SizeExceeded,
    CorruptedFile,
}

/// Folder hierarchy representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderHierarchy {
    /// Root node
    pub root: FolderNode,
    /// Total depth
    pub max_depth: u32,
    /// Total nodes
    pub total_nodes: u64,
}

/// Folder node in hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderNode {
    /// Node name
    pub name: String,
    /// Full path
    pub path: PathBuf,
    /// Node type
    pub node_type: NodeType,
    /// File size (for files)
    pub size: Option<u64>,
    /// Last modified time
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    /// Child nodes
    pub children: Vec<FolderNode>,
    /// Associated memory ID (if processed)
    pub memory_id: Option<MemoryId>,
    /// Content type (if file)
    pub content_type: Option<ContentType>,
}

/// Types of nodes in folder hierarchy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    Directory,
    File,
    Symlink,
    Unknown,
}

/// Folder processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderProcessorConfig {
    /// Maximum recursion depth
    pub max_depth: u32,
    /// Maximum number of files to process
    pub max_files: u64,
    /// Maximum total size to process (bytes)
    pub max_total_size: u64,
    /// File patterns to include
    pub include_patterns: Vec<String>,
    /// File patterns to exclude
    pub exclude_patterns: Vec<String>,
    /// Follow symbolic links
    pub follow_symlinks: bool,
    /// Process hidden files
    pub process_hidden: bool,
    /// Parallel processing threads
    pub parallel_threads: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable content deduplication
    pub enable_deduplication: bool,
    /// Supported file extensions
    pub supported_extensions: Vec<String>,
}

impl Default for FolderProcessorConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_files: 10000,
            max_total_size: 10 * 1024 * 1024 * 1024, // 10GB
            include_patterns: vec!["*".to_string()],
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.log".to_string(),
                ".git/*".to_string(),
                "node_modules/*".to_string(),
                "target/*".to_string(),
            ],
            follow_symlinks: false,
            process_hidden: false,
            parallel_threads: 4,
            batch_size: 100,
            enable_deduplication: true,
            supported_extensions: vec![
                "txt".to_string(), "md".to_string(), "pdf".to_string(),
                "doc".to_string(), "docx".to_string(), "csv".to_string(),
                "json".to_string(), "xml".to_string(), "html".to_string(),
                "rs".to_string(), "py".to_string(), "js".to_string(),
                "ts".to_string(), "java".to_string(), "cpp".to_string(),
                "png".to_string(), "jpg".to_string(), "jpeg".to_string(),
                "wav".to_string(), "mp3".to_string(), "flac".to_string(),
            ],
        }
    }
}

/// Folder memory processor for batch operations
#[derive(Debug, Clone)]
pub struct FolderMemoryProcessor {
    /// Configuration for folder processing
    config: FolderProcessorConfig,
    /// Document processor
    document_processor: super::document::DocumentMemoryProcessor,
    /// Data processor
    data_processor: super::data::DataMemoryProcessor,
    /// Image processor
    image_processor: super::image::ImageMemoryProcessor,
    /// Audio processor
    audio_processor: super::audio::AudioMemoryProcessor,
    /// Code processor
    code_processor: super::code::CodeMemoryProcessor,
}

impl FolderMemoryProcessor {
    /// Create a new folder processor
    pub fn new(config: FolderProcessorConfig) -> Self {
        Self {
            config,
            document_processor: super::document::DocumentMemoryProcessor::default(),
            data_processor: super::data::DataMemoryProcessor::default(),
            image_processor: super::image::ImageMemoryProcessor::default(),
            audio_processor: super::audio::AudioMemoryProcessor::default(),
            code_processor: super::code::CodeMemoryProcessor::default(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(FolderProcessorConfig::default())
    }

    /// Process entire folder structure
    pub async fn process_folder<P: AsRef<Path>>(&self, folder_path: P) -> MultiModalResult<FolderProcessingResult> {
        let start_time = std::time::Instant::now();
        let root_path = folder_path.as_ref().to_path_buf();
        
        if !root_path.exists() {
            return Err(SynapticError::ProcessingError(
                format!("Folder does not exist: {}", root_path.display())
            ));
        }
        
        if !root_path.is_dir() {
            return Err(SynapticError::ProcessingError(
                format!("Path is not a directory: {}", root_path.display())
            ));
        }
        
        // Discover files
        let discovered_files = self.discover_files(&root_path).await?;
        
        // Build hierarchy
        let hierarchy = self.build_hierarchy(&root_path, &discovered_files).await?;
        
        // Process files in batches
        let (memories, errors) = self.process_files_batch(&discovered_files).await?;
        
        // Calculate statistics
        let total_files = discovered_files.len() as u64;
        let total_directories = self.count_directories(&hierarchy);
        let total_size = discovered_files.iter()
            .filter_map(|path| std::fs::metadata(path).ok())
            .map(|metadata| metadata.len())
            .sum();
        
        let file_type_distribution = self.calculate_file_type_distribution(&discovered_files);
        
        let processing_duration = start_time.elapsed();
        
        Ok(FolderProcessingResult {
            root_path,
            total_files,
            total_directories,
            total_size,
            processing_duration,
            file_type_distribution,
            errors,
            memories,
            hierarchy,
        })
    }

    /// Discover files in folder structure
    async fn discover_files(&self, root_path: &Path) -> MultiModalResult<Vec<PathBuf>> {
        let mut discovered_files = Vec::new();
        let mut total_size = 0u64;
        
        // Use walkdir for recursive directory traversal
        let walker = walkdir::WalkDir::new(root_path)
            .max_depth(self.config.max_depth as usize)
            .follow_links(self.config.follow_symlinks);
        
        for entry in walker {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    
                    // Skip if not a file
                    if !path.is_file() {
                        continue;
                    }
                    
                    // Skip hidden files if not enabled
                    if !self.config.process_hidden && self.is_hidden_file(path) {
                        continue;
                    }
                    
                    // Check include/exclude patterns
                    if !self.matches_patterns(path) {
                        continue;
                    }
                    
                    // Check file extension
                    if !self.is_supported_extension(path) {
                        continue;
                    }
                    
                    // Check size limits
                    if let Ok(metadata) = std::fs::metadata(path) {
                        let file_size = metadata.len();
                        if total_size + file_size > self.config.max_total_size {
                            break;
                        }
                        total_size += file_size;
                    }
                    
                    discovered_files.push(path.to_path_buf());
                    
                    // Check file count limit
                    if discovered_files.len() >= self.config.max_files as usize {
                        break;
                    }
                }
                Err(_) => {
                    // Skip files we can't access
                    continue;
                }
            }
        }
        
        Ok(discovered_files)
    }

    /// Check if file is hidden
    fn is_hidden_file(&self, path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false)
    }

    /// Check if file matches include/exclude patterns
    fn matches_patterns(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        // Check exclude patterns first
        for pattern in &self.config.exclude_patterns {
            if self.matches_glob_pattern(&path_str, pattern) {
                return false;
            }
        }
        
        // Check include patterns
        for pattern in &self.config.include_patterns {
            if self.matches_glob_pattern(&path_str, pattern) {
                return true;
            }
        }
        
        false
    }

    /// Simple glob pattern matching
    fn matches_glob_pattern(&self, text: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return text.starts_with(prefix) && text.ends_with(suffix);
            }
        }
        
        text.contains(pattern)
    }

    /// Check if file extension is supported
    fn is_supported_extension(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                return self.config.supported_extensions.contains(&ext_str.to_lowercase());
            }
        }
        false
    }

    /// Build folder hierarchy
    async fn build_hierarchy(&self, root_path: &Path, files: &[PathBuf]) -> MultiModalResult<FolderHierarchy> {
        let root_node = self.build_node(root_path, files, 0).await?;
        let max_depth = self.calculate_max_depth(&root_node);
        let total_nodes = self.count_total_nodes(&root_node);
        
        Ok(FolderHierarchy {
            root: root_node,
            max_depth,
            total_nodes,
        })
    }

    /// Build individual node
    async fn build_node(&self, path: &Path, all_files: &[PathBuf], depth: u32) -> MultiModalResult<FolderNode> {
        let metadata = std::fs::metadata(path).ok();
        let modified = metadata.as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| chrono::DateTime::from_timestamp(
                t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64, 0
            ));
        
        let node_type = if path.is_dir() {
            NodeType::Directory
        } else if path.is_file() {
            NodeType::File
        } else {
            NodeType::Unknown
        };
        
        let size = if path.is_file() {
            metadata.map(|m| m.len())
        } else {
            None
        };
        
        let mut children = Vec::new();
        
        // Add child nodes for directories
        if path.is_dir() && depth < self.config.max_depth {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let child_path = entry.path();
                    if all_files.contains(&child_path) || child_path.is_dir() {
                        let child_node = self.build_node(&child_path, all_files, depth + 1).await?;
                        children.push(child_node);
                    }
                }
            }
        }
        
        Ok(FolderNode {
            name: path.file_name()
                .unwrap_or_else(|| path.as_os_str())
                .to_string_lossy()
                .to_string(),
            path: path.to_path_buf(),
            node_type,
            size,
            modified,
            children,
            memory_id: None,
            content_type: None,
        })
    }

    /// Calculate maximum depth of hierarchy
    fn calculate_max_depth(&self, node: &FolderNode) -> u32 {
        if node.children.is_empty() {
            0
        } else {
            1 + node.children.iter()
                .map(|child| self.calculate_max_depth(child))
                .max()
                .unwrap_or(0)
        }
    }

    /// Count total nodes in hierarchy
    fn count_total_nodes(&self, node: &FolderNode) -> u64 {
        1 + node.children.iter()
            .map(|child| self.count_total_nodes(child))
            .sum::<u64>()
    }

    /// Count directories in hierarchy
    fn count_directories(&self, hierarchy: &FolderHierarchy) -> u64 {
        self.count_directories_recursive(&hierarchy.root)
    }

    /// Count directories recursively
    fn count_directories_recursive(&self, node: &FolderNode) -> u64 {
        let current = if node.node_type == NodeType::Directory { 1 } else { 0 };
        current + node.children.iter()
            .map(|child| self.count_directories_recursive(child))
            .sum::<u64>()
    }

    /// Calculate file type distribution
    fn calculate_file_type_distribution(&self, files: &[PathBuf]) -> HashMap<String, u64> {
        let mut distribution = HashMap::new();
        
        for file in files {
            let extension = file.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown")
                .to_lowercase();
            
            *distribution.entry(extension).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Process files in batches
    async fn process_files_batch(&self, files: &[PathBuf]) -> MultiModalResult<(Vec<MultiModalMemory>, Vec<ProcessingError>)> {
        let mut memories = Vec::new();
        let mut errors = Vec::new();

        // Process files in batches
        for chunk in files.chunks(self.config.batch_size) {
            let batch_results = self.process_file_batch(chunk).await;

            for result in batch_results {
                match result {
                    Ok(memory) => memories.push(memory),
                    Err(error) => errors.push(error),
                }
            }
        }

        // Deduplicate if enabled
        if self.config.enable_deduplication {
            memories = self.deduplicate_memories(memories).await?;
        }

        Ok((memories, errors))
    }

    /// Process a single batch of files
    async fn process_file_batch(&self, files: &[PathBuf]) -> Vec<Result<MultiModalMemory, ProcessingError>> {
        let mut results = Vec::new();

        for file_path in files {
            let result = self.process_single_file(file_path).await;
            results.push(result);
        }

        results
    }

    /// Process a single file
    async fn process_single_file(&self, file_path: &Path) -> Result<MultiModalMemory, ProcessingError> {
        // Read file content
        let content = match std::fs::read(file_path) {
            Ok(content) => content,
            Err(e) => {
                return Err(ProcessingError {
                    file_path: file_path.to_path_buf(),
                    error_message: e.to_string(),
                    error_type: match e.kind() {
                        std::io::ErrorKind::NotFound => ErrorType::FileNotFound,
                        std::io::ErrorKind::PermissionDenied => ErrorType::PermissionDenied,
                        _ => ErrorType::ProcessingFailed,
                    },
                });
            }
        };

        // Detect content type
        let content_type = match self.detect_content_type(file_path, &content) {
            Ok(ct) => ct,
            Err(_) => {
                return Err(ProcessingError {
                    file_path: file_path.to_path_buf(),
                    error_message: "Failed to detect content type".to_string(),
                    error_type: ErrorType::UnsupportedFormat,
                });
            }
        };

        // Process based on content type
        let memory = match content_type {
            ContentType::Document { .. } => {
                self.document_processor.process(&content, &content_type).await
            }
            ContentType::Data { .. } => {
                self.data_processor.process(&content, &content_type).await
            }
            ContentType::Image { .. } => {
                self.image_processor.process(&content, &content_type).await
            }
            ContentType::Audio { .. } => {
                self.audio_processor.process(&content, &content_type).await
            }
            ContentType::Code { .. } => {
                self.code_processor.process(&content, &content_type).await
            }
            _ => {
                return Err(ProcessingError {
                    file_path: file_path.to_path_buf(),
                    error_message: "Unsupported content type".to_string(),
                    error_type: ErrorType::UnsupportedFormat,
                });
            }
        };

        match memory {
            Ok(mut mem) => {
                // Add file path information
                mem.extracted_features.insert(
                    "file_path".to_string(),
                    serde_json::to_value(file_path.to_string_lossy().to_string()).unwrap(),
                );
                Ok(mem)
            }
            Err(_) => Err(ProcessingError {
                file_path: file_path.to_path_buf(),
                error_message: "Processing failed".to_string(),
                error_type: ErrorType::ProcessingFailed,
            }),
        }
    }

    /// Detect content type from file path and content
    fn detect_content_type(&self, file_path: &Path, content: &[u8]) -> MultiModalResult<ContentType> {
        let extension = file_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            // Document formats
            "pdf" => Ok(ContentType::Document {
                format: "PDF".to_string(),
                language: Some("en".to_string()),
            }),
            "doc" | "docx" => Ok(ContentType::Document {
                format: "Word".to_string(),
                language: Some("en".to_string()),
            }),
            "md" | "markdown" => Ok(ContentType::Document {
                format: "Markdown".to_string(),
                language: Some("en".to_string()),
            }),
            "txt" => Ok(ContentType::Document {
                format: "PlainText".to_string(),
                language: Some("en".to_string()),
            }),
            "html" | "htm" => Ok(ContentType::Document {
                format: "HTML".to_string(),
                language: Some("en".to_string()),
            }),

            // Data formats
            "csv" => Ok(ContentType::Data {
                format: "CSV".to_string(),
                schema: None,
            }),
            "json" => Ok(ContentType::Data {
                format: "JSON".to_string(),
                schema: None,
            }),
            "parquet" => Ok(ContentType::Data {
                format: "Parquet".to_string(),
                schema: None,
            }),
            "xlsx" | "xls" => Ok(ContentType::Data {
                format: "Excel".to_string(),
                schema: None,
            }),

            // Image formats
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tiff" => Ok(ContentType::Image {
                format: extension.to_uppercase(),
                width: 0,
                height: 0,
            }),

            // Audio formats
            "wav" | "mp3" | "flac" | "ogg" | "m4a" => Ok(ContentType::Audio {
                format: extension.to_uppercase(),
                duration_ms: 0,
            }),

            // Code formats
            "rs" => Ok(ContentType::Code {
                language: "rust".to_string(),
                framework: None,
            }),
            "py" => Ok(ContentType::Code {
                language: "python".to_string(),
                framework: None,
            }),
            "js" => Ok(ContentType::Code {
                language: "javascript".to_string(),
                framework: None,
            }),
            "ts" => Ok(ContentType::Code {
                language: "typescript".to_string(),
                framework: None,
            }),
            "java" => Ok(ContentType::Code {
                language: "java".to_string(),
                framework: None,
            }),
            "cpp" | "cc" | "cxx" => Ok(ContentType::Code {
                language: "cpp".to_string(),
                framework: None,
            }),
            "c" => Ok(ContentType::Code {
                language: "c".to_string(),
                framework: None,
            }),
            "go" => Ok(ContentType::Code {
                language: "go".to_string(),
                framework: None,
            }),

            _ => {
                // Try to detect from content
                if content.starts_with(b"%PDF") {
                    Ok(ContentType::Document {
                        format: "PDF".to_string(),
                        language: Some("en".to_string()),
                    })
                } else if content.starts_with(b"PK\x03\x04") {
                    Ok(ContentType::Data {
                        format: "Archive".to_string(),
                        schema: None,
                    })
                } else {
                    Err(SynapticError::ProcessingError("Unknown content type".to_string()))
                }
            }
        }
    }

    /// Deduplicate memories based on content hash
    async fn deduplicate_memories(&self, memories: Vec<MultiModalMemory>) -> MultiModalResult<Vec<MultiModalMemory>> {
        let mut unique_memories = Vec::new();
        let mut seen_hashes = std::collections::HashSet::new();

        for memory in memories {
            // Calculate content hash
            let content_hash = self.calculate_content_hash(&memory.primary_content);

            if !seen_hashes.contains(&content_hash) {
                seen_hashes.insert(content_hash);
                unique_memories.push(memory);
            }
        }

        Ok(unique_memories)
    }

    /// Calculate content hash for deduplication
    fn calculate_content_hash(&self, content: &[u8]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Get processing statistics
    pub fn get_processing_stats(&self, result: &FolderProcessingResult) -> ProcessingStats {
        let success_rate = if result.total_files > 0 {
            (result.memories.len() as f64 / result.total_files as f64) * 100.0
        } else {
            0.0
        };

        let avg_file_size = if result.total_files > 0 {
            result.total_size as f64 / result.total_files as f64
        } else {
            0.0
        };

        let processing_speed = if result.processing_duration.as_secs() > 0 {
            result.total_files as f64 / result.processing_duration.as_secs() as f64
        } else {
            0.0
        };

        ProcessingStats {
            success_rate,
            avg_file_size,
            processing_speed,
            error_rate: 100.0 - success_rate,
            total_processing_time: result.processing_duration,
        }
    }
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Success rate percentage
    pub success_rate: f64,
    /// Average file size in bytes
    pub avg_file_size: f64,
    /// Processing speed (files per second)
    pub processing_speed: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
}

#[async_trait::async_trait]
impl MultiModalProcessor for FolderMemoryProcessor {
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        // For folder processor, content should be a path
        let path_str = String::from_utf8_lossy(content);
        let path = Path::new(path_str.as_ref());

        if !path.exists() {
            return Err(SynapticError::ProcessingError(
                format!("Path does not exist: {}", path.display())
            ));
        }

        if path.is_file() {
            // Process single file
            match self.process_single_file(path).await {
                Ok(memory) => Ok(memory),
                Err(error) => Err(SynapticError::ProcessingError(error.error_message)),
            }
        } else if path.is_dir() {
            // Process folder and create summary memory
            let result = self.process_folder(path).await?;

            // Create summary memory for the folder
            let metadata = MultiModalMetadata {
                title: Some(format!("Folder: {}", path.display())),
                description: Some(format!(
                    "Processed {} files in {} directories",
                    result.total_files, result.total_directories
                )),
                tags: result.file_type_distribution.keys().cloned().collect(),
                quality_score: if result.errors.is_empty() { 1.0 } else { 0.8 },
                confidence: 0.9,
                processing_time_ms: result.processing_duration.as_millis() as u64,
                extracted_features: HashMap::new(),
            };

            let memory = MultiModalMemory {
                id: Uuid::new_v4().to_string(),
                content_type: content_type.clone(),
                primary_content: content.to_vec(),
                metadata,
                extracted_features: {
                    let mut features = HashMap::new();
                    features.insert("folder_result".to_string(), serde_json::to_value(result)?);
                    features
                },
                cross_modal_links: Vec::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };

            Ok(memory)
        } else {
            Err(SynapticError::ProcessingError(
                format!("Unsupported path type: {}", path.display())
            ))
        }
    }

    async fn extract_features(&self, content: &[u8], _content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        // For folder processing, extract features from path
        let path_str = String::from_utf8_lossy(content);
        let path = Path::new(path_str.as_ref());

        if path.is_dir() {
            // Count files and directories
            let file_count = walkdir::WalkDir::new(path)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .count() as f32;

            let dir_count = walkdir::WalkDir::new(path)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir() && e.path() != path)
                .count() as f32;

            Ok(vec![file_count, dir_count, 0.0, 0.0])
        } else {
            // Single file features
            let file_size = std::fs::metadata(path)
                .map(|m| m.len() as f32)
                .unwrap_or(0.0);

            Ok(vec![1.0, 0.0, file_size, 0.0])
        }
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
