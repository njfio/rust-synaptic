//! # Code Memory System
//!
//! Advanced code analysis and understanding capabilities for the Synaptic memory system.
//! Provides syntax parsing, similarity detection, API pattern recognition, and semantic code search.

use super::{
    CodeLanguage, ComplexityMetrics, ContentSpecificMetadata, ContentType, FunctionInfo,
    MultiModalMemory, MultiModalMetadata, MultiModalProcessor, MultiModalResult, ProcessingInfo,
};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "code-memory")]
use {
    syn::{parse_file, visit::Visit, File, Item, ItemFn, Type},
    tree_sitter::{Language, Node, Parser, Query, QueryCursor, Tree},
};

#[cfg(all(feature = "code-memory", feature = "tree-sitter-rust"))]
extern "C" {
    fn tree_sitter_rust() -> Language;
}

#[cfg(all(feature = "code-memory", feature = "tree-sitter-python"))]
extern "C" {
    fn tree_sitter_python() -> Language;
}

#[cfg(all(feature = "code-memory", feature = "tree-sitter-javascript"))]
extern "C" {
    fn tree_sitter_javascript() -> Language;
}

/// Code memory processor with syntax analysis and pattern recognition
#[derive(Debug)]
pub struct CodeMemoryProcessor {
    /// Tree-sitter parsers for different languages
    #[cfg(feature = "code-memory")]
    parsers: HashMap<CodeLanguage, Parser>,
    
    /// Configuration settings
    config: CodeProcessorConfig,
}

/// Configuration for code processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProcessorConfig {
    /// Enable syntax parsing and AST analysis
    pub enable_syntax_analysis: bool,
    
    /// Enable function extraction and analysis
    pub enable_function_analysis: bool,
    
    /// Enable complexity metrics calculation
    pub enable_complexity_metrics: bool,
    
    /// Enable dependency analysis
    pub enable_dependency_analysis: bool,
    
    /// Enable semantic similarity detection
    pub enable_semantic_similarity: bool,
    
    /// Maximum file size for processing (bytes)
    pub max_file_size: usize,
    
    /// Supported programming languages
    pub supported_languages: Vec<CodeLanguage>,
}

impl Default for CodeProcessorConfig {
    fn default() -> Self {
        Self {
            enable_syntax_analysis: true,
            enable_function_analysis: true,
            enable_complexity_metrics: true,
            enable_dependency_analysis: true,
            enable_semantic_similarity: true,
            max_file_size: 1024 * 1024, // 1MB max
            supported_languages: vec![
                CodeLanguage::Rust,
                CodeLanguage::Python,
                CodeLanguage::JavaScript,
                CodeLanguage::TypeScript,
                CodeLanguage::Java,
                CodeLanguage::CSharp,
                CodeLanguage::Cpp,
                CodeLanguage::C,
                CodeLanguage::Go,
            ],
        }
    }
}

impl CodeMemoryProcessor {
    /// Create a new code memory processor
    pub fn new(config: CodeProcessorConfig) -> MultiModalResult<Self> {
        let mut processor = Self {
            #[cfg(feature = "code-memory")]
            parsers: HashMap::new(),
            config,
        };

        // Initialize tree-sitter parsers
        #[cfg(feature = "code-memory")]
        processor.initialize_parsers()?;

        Ok(processor)
    }

    /// Initialize tree-sitter parsers for supported languages
    #[cfg(feature = "code-memory")]
    fn initialize_parsers(&mut self) -> MultiModalResult<()> {
        for language in &self.config.supported_languages {
            let mut parser = Parser::new();
            
            let tree_sitter_language = match language {
                #[cfg(feature = "tree-sitter-rust")]
                CodeLanguage::Rust => unsafe { tree_sitter_rust() },
                #[cfg(feature = "tree-sitter-python")]
                CodeLanguage::Python => unsafe { tree_sitter_python() },
                #[cfg(feature = "tree-sitter-javascript")]
                CodeLanguage::JavaScript => unsafe { tree_sitter_javascript() },
                _ => continue, // Skip unsupported languages
            };

            parser.set_language(tree_sitter_language)
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to set language: {}", e)))?;
            
            self.parsers.insert(language.clone(), parser);
        }

        Ok(())
    }

    /// Detect programming language from content
    pub fn detect_language(&self, content: &str, filename: Option<&str>) -> CodeLanguage {
        // Check file extension first
        if let Some(filename) = filename {
            if let Some(ext) = filename.split('.').last() {
                match ext.to_lowercase().as_str() {
                    "rs" => return CodeLanguage::Rust,
                    "py" => return CodeLanguage::Python,
                    "js" => return CodeLanguage::JavaScript,
                    "ts" => return CodeLanguage::TypeScript,
                    "java" => return CodeLanguage::Java,
                    "cs" => return CodeLanguage::CSharp,
                    "cpp" | "cc" | "cxx" => return CodeLanguage::Cpp,
                    "c" => return CodeLanguage::C,
                    "go" => return CodeLanguage::Go,
                    "swift" => return CodeLanguage::Swift,
                    "kt" => return CodeLanguage::Kotlin,
                    "rb" => return CodeLanguage::Ruby,
                    "php" => return CodeLanguage::Php,
                    "html" => return CodeLanguage::Html,
                    "css" => return CodeLanguage::Css,
                    "sql" => return CodeLanguage::Sql,
                    "sh" | "bash" => return CodeLanguage::Shell,
                    _ => {}
                }
            }
        }

        // Analyze content for language-specific patterns
        if content.contains("fn main()") || content.contains("use std::") {
            CodeLanguage::Rust
        } else if content.contains("def ") && content.contains("import ") {
            CodeLanguage::Python
        } else if content.contains("function ") || content.contains("const ") {
            CodeLanguage::JavaScript
        } else if content.contains("public class ") {
            CodeLanguage::Java
        } else if content.contains("using System") {
            CodeLanguage::CSharp
        } else if content.contains("#include") {
            CodeLanguage::Cpp
        } else if content.contains("package main") {
            CodeLanguage::Go
        } else {
            CodeLanguage::Other("unknown".to_string())
        }
    }

    /// Parse code using tree-sitter
    #[cfg(feature = "code-memory")]
    pub fn parse_code(&mut self, content: &str, language: &CodeLanguage) -> MultiModalResult<Option<Tree>> {
        if !self.config.enable_syntax_analysis {
            return Ok(None);
        }

        let parser = self.parsers.get_mut(language);
        if let Some(parser) = parser {
            let tree = parser.parse(content, None)
                .ok_or_else(|| SynapticError::ProcessingError("Failed to parse code".to_string()))?;
            Ok(Some(tree))
        } else {
            Ok(None)
        }
    }

    /// Extract functions from parsed AST
    #[cfg(feature = "code-memory")]
    pub fn extract_functions(&self, tree: &Tree, content: &str, language: &CodeLanguage) -> MultiModalResult<Vec<FunctionInfo>> {
        if !self.config.enable_function_analysis {
            return Ok(vec![]);
        }

        let mut functions = Vec::new();
        let root_node = tree.root_node();

        match language {
            CodeLanguage::Rust => {
                self.extract_rust_functions(&root_node, content, &mut functions)?;
            }
            CodeLanguage::Python => {
                self.extract_python_functions(&root_node, content, &mut functions)?;
            }
            CodeLanguage::JavaScript => {
                self.extract_javascript_functions(&root_node, content, &mut functions)?;
            }
            _ => {
                // Generic function extraction
                self.extract_generic_functions(&root_node, content, &mut functions)?;
            }
        }

        Ok(functions)
    }

    /// Extract Rust functions using tree-sitter
    #[cfg(feature = "code-memory")]
    fn extract_rust_functions(&self, node: &Node, content: &str, functions: &mut Vec<FunctionInfo>) -> MultiModalResult<()> {
        let mut cursor = node.walk();
        
        for child in node.children(&mut cursor) {
            if child.kind() == "function_item" {
                let function_info = self.parse_rust_function(&child, content)?;
                functions.push(function_info);
            }
            
            // Recursively search child nodes
            self.extract_rust_functions(&child, content, functions)?;
        }
        
        Ok(())
    }

    /// Parse a Rust function node
    #[cfg(feature = "code-memory")]
    fn parse_rust_function(&self, node: &Node, content: &str) -> MultiModalResult<FunctionInfo> {
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        
        // Extract function name
        let name = if let Some(name_node) = node.child_by_field_name("name") {
            name_node.utf8_text(content.as_bytes())
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to extract function name: {}", e)))?
                .to_string()
        } else {
            "unknown".to_string()
        };

        // Extract parameters
        let mut parameters = Vec::new();
        if let Some(params_node) = node.child_by_field_name("parameters") {
            for param in params_node.children(&mut params_node.walk()) {
                if param.kind() == "parameter" {
                    if let Ok(param_text) = param.utf8_text(content.as_bytes()) {
                        parameters.push(param_text.to_string());
                    }
                }
            }
        }

        // Extract return type
        let return_type = if let Some(return_node) = node.child_by_field_name("return_type") {
            return_node.utf8_text(content.as_bytes()).ok().map(|s| s.to_string())
        } else {
            None
        };

        // Calculate complexity (simplified)
        let complexity = self.calculate_function_complexity(node, content)?;

        Ok(FunctionInfo {
            name,
            parameters,
            return_type,
            line_start: start_line,
            line_end: end_line,
            complexity,
        })
    }

    /// Extract Python functions using tree-sitter
    #[cfg(feature = "code-memory")]
    fn extract_python_functions(&self, node: &Node, content: &str, functions: &mut Vec<FunctionInfo>) -> MultiModalResult<()> {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "function_definition" {
                let function_info = self.parse_python_function(&child, content)?;
                functions.push(function_info);
            }

            // Recursively search child nodes
            self.extract_python_functions(&child, content, functions)?;
        }

        Ok(())
    }

    /// Parse a Python function node
    #[cfg(feature = "code-memory")]
    fn parse_python_function(&self, node: &Node, content: &str) -> MultiModalResult<FunctionInfo> {
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        // Extract function name
        let name = if let Some(name_node) = node.child_by_field_name("name") {
            name_node.utf8_text(content.as_bytes())
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to extract function name: {}", e)))?
                .to_string()
        } else {
            "unknown".to_string()
        };

        // Extract parameters
        let mut parameters = Vec::new();
        if let Some(params_node) = node.child_by_field_name("parameters") {
            for param in params_node.children(&mut params_node.walk()) {
                if param.kind() == "identifier" || param.kind() == "typed_parameter" || param.kind() == "default_parameter" {
                    if let Ok(param_text) = param.utf8_text(content.as_bytes()) {
                        parameters.push(param_text.to_string());
                    }
                }
            }
        }

        // Extract return type (Python type hints)
        let return_type = if let Some(return_node) = node.child_by_field_name("return_type") {
            return_node.utf8_text(content.as_bytes()).ok().map(|s| s.to_string())
        } else {
            None
        };

        // Calculate complexity using Python-specific patterns
        let complexity = self.calculate_python_function_complexity(node, content)?;

        Ok(FunctionInfo {
            name,
            parameters,
            return_type,
            line_start: start_line,
            line_end: end_line,
            complexity,
        })
    }

    /// Calculate function complexity for Python
    #[cfg(feature = "code-memory")]
    fn calculate_python_function_complexity(&self, node: &Node, content: &str) -> MultiModalResult<u32> {
        let mut complexity = 1; // Base complexity
        let mut cursor = node.walk();

        // Count decision points (if, while, for, try, etc.)
        for child in node.children(&mut cursor) {
            match child.kind() {
                "if_statement" | "while_statement" | "for_statement" | "try_statement" |
                "with_statement" | "elif_clause" | "except_clause" => {
                    complexity += 1;
                }
                "and" | "or" => {
                    complexity += 1; // Logical operators add complexity
                }
                _ => {}
            }

            // Recursively count in child nodes
            complexity += self.calculate_python_function_complexity(&child, content).unwrap_or(0);
        }

        Ok(complexity)
    }

    /// Extract JavaScript functions using tree-sitter
    #[cfg(feature = "code-memory")]
    fn extract_javascript_functions(&self, node: &Node, content: &str, functions: &mut Vec<FunctionInfo>) -> MultiModalResult<()> {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "function_declaration" | "function_expression" | "arrow_function" | "method_definition" => {
                    let function_info = self.parse_javascript_function(&child, content)?;
                    functions.push(function_info);
                }
                _ => {}
            }

            // Recursively search child nodes
            self.extract_javascript_functions(&child, content, functions)?;
        }

        Ok(())
    }

    /// Parse a JavaScript function node
    #[cfg(feature = "code-memory")]
    fn parse_javascript_function(&self, node: &Node, content: &str) -> MultiModalResult<FunctionInfo> {
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;

        // Extract function name
        let name = if let Some(name_node) = node.child_by_field_name("name") {
            name_node.utf8_text(content.as_bytes())
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to extract function name: {}", e)))?
                .to_string()
        } else if node.kind() == "arrow_function" {
            "anonymous_arrow".to_string()
        } else {
            "anonymous".to_string()
        };

        // Extract parameters
        let mut parameters = Vec::new();
        if let Some(params_node) = node.child_by_field_name("parameters") {
            for param in params_node.children(&mut params_node.walk()) {
                if param.kind() == "identifier" || param.kind() == "formal_parameter" || param.kind() == "rest_parameter" {
                    if let Ok(param_text) = param.utf8_text(content.as_bytes()) {
                        parameters.push(param_text.to_string());
                    }
                }
            }
        }

        // JavaScript doesn't have explicit return types (unless TypeScript)
        let return_type = None;

        // Calculate complexity using JavaScript-specific patterns
        let complexity = self.calculate_javascript_function_complexity(node, content)?;

        Ok(FunctionInfo {
            name,
            parameters,
            return_type,
            line_start: start_line,
            line_end: end_line,
            complexity,
        })
    }

    /// Calculate function complexity for JavaScript
    #[cfg(feature = "code-memory")]
    fn calculate_javascript_function_complexity(&self, node: &Node, content: &str) -> MultiModalResult<u32> {
        let mut complexity = 1; // Base complexity
        let mut cursor = node.walk();

        // Count decision points (if, while, for, switch, etc.)
        for child in node.children(&mut cursor) {
            match child.kind() {
                "if_statement" | "while_statement" | "for_statement" | "for_in_statement" |
                "for_of_statement" | "switch_statement" | "try_statement" | "catch_clause" |
                "conditional_expression" => {
                    complexity += 1;
                }
                "&&" | "||" => {
                    complexity += 1; // Logical operators add complexity
                }
                _ => {}
            }

            // Recursively count in child nodes
            complexity += self.calculate_javascript_function_complexity(&child, content).unwrap_or(0);
        }

        Ok(complexity)
    }

    /// Generic function extraction for unsupported languages
    #[cfg(feature = "code-memory")]
    fn extract_generic_functions(&self, _node: &Node, _content: &str, _functions: &mut Vec<FunctionInfo>) -> MultiModalResult<()> {
        // Basic pattern matching for function detection
        Ok(())
    }

    /// Calculate function complexity (generic version for Rust)
    #[cfg(feature = "code-memory")]
    fn calculate_function_complexity(&self, node: &Node, content: &str) -> MultiModalResult<u32> {
        let mut complexity = 1; // Base complexity
        let mut cursor = node.walk();

        // Count decision points (if, while, for, match, etc.) - Rust specific
        for child in node.children(&mut cursor) {
            match child.kind() {
                "if_expression" | "while_expression" | "for_expression" | "match_expression" |
                "loop_expression" | "if_let_expression" | "while_let_expression" => {
                    complexity += 1;
                }
                _ => {}
            }

            // Recursively count in child nodes
            complexity += self.calculate_function_complexity(&child, content).unwrap_or(0);
        }

        Ok(complexity)
    }

    /// Calculate code complexity metrics
    pub fn calculate_complexity_metrics(&self, content: &str, functions: &[FunctionInfo]) -> ComplexityMetrics {
        let lines_of_code = content.lines().count() as u32;
        
        let cyclomatic_complexity = functions.iter().map(|f| f.complexity).sum::<u32>().max(1);
        
        // Simplified cognitive complexity (would need more sophisticated analysis)
        let cognitive_complexity = cyclomatic_complexity;
        
        // Maintainability index (simplified formula)
        let maintainability_index = if lines_of_code > 0 {
            171.0 - 5.2 * (lines_of_code as f32).ln() - 0.23 * cyclomatic_complexity as f32
        } else {
            100.0
        };

        ComplexityMetrics {
            cyclomatic_complexity,
            cognitive_complexity,
            lines_of_code,
            maintainability_index: maintainability_index.max(0.0).min(100.0),
        }
    }

    /// Extract dependencies from code
    pub fn extract_dependencies(&self, content: &str, language: &CodeLanguage) -> Vec<String> {
        if !self.config.enable_dependency_analysis {
            return vec![];
        }

        let mut dependencies = Vec::new();

        match language {
            CodeLanguage::Rust => {
                // Extract use statements
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("use ") && !trimmed.starts_with("use std::") {
                        if let Some(dep) = trimmed.strip_prefix("use ") {
                            if let Some(dep) = dep.split("::").next() {
                                dependencies.push(dep.to_string());
                            }
                        }
                    }
                }
            }
            CodeLanguage::Python => {
                // Extract import statements
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("import ") {
                        if let Some(dep) = trimmed.strip_prefix("import ") {
                            if let Some(dep) = dep.split('.').next() {
                                dependencies.push(dep.to_string());
                            }
                        }
                    } else if trimmed.starts_with("from ") {
                        if let Some(rest) = trimmed.strip_prefix("from ") {
                            if let Some(dep) = rest.split(' ').next() {
                                dependencies.push(dep.to_string());
                            }
                        }
                    }
                }
            }
            CodeLanguage::JavaScript | CodeLanguage::TypeScript => {
                // Extract import/require statements
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.contains("import ") && trimmed.contains(" from ") {
                        // ES6 imports
                        if let Some(from_part) = trimmed.split(" from ").nth(1) {
                            let dep = from_part.trim_matches(|c| c == '"' || c == '\'' || c == ';');
                            dependencies.push(dep.to_string());
                        }
                    } else if trimmed.contains("require(") {
                        // CommonJS requires
                        if let Some(start) = trimmed.find("require(") {
                            let after_require = &trimmed[start + 8..];
                            if let Some(end) = after_require.find(')') {
                                let dep = after_require[..end].trim_matches(|c| c == '"' || c == '\'');
                                dependencies.push(dep.to_string());
                            }
                        }
                    }
                }
            }
            _ => {
                // Generic dependency extraction
            }
        }

        dependencies.sort();
        dependencies.dedup();
        dependencies
    }

    /// Extract semantic features for similarity comparison
    pub async fn extract_semantic_features(&self, content: &str, language: &CodeLanguage) -> MultiModalResult<Vec<f32>> {
        if !self.config.enable_semantic_similarity {
            return Ok(vec![]);
        }

        let mut features = Vec::new();

        // Basic lexical features
        let lines = content.lines().count() as f32;
        let chars = content.len() as f32;
        let words = content.split_whitespace().count() as f32;
        
        features.push(lines);
        features.push(chars);
        features.push(words);
        features.push(chars / lines.max(1.0)); // Average line length

        // Language-specific keyword frequencies
        let keywords = match language {
            CodeLanguage::Rust => vec!["fn", "let", "mut", "if", "else", "match", "for", "while", "impl", "struct"],
            CodeLanguage::Python => vec!["def", "class", "if", "else", "elif", "for", "while", "import", "from", "return"],
            CodeLanguage::JavaScript => vec!["function", "var", "let", "const", "if", "else", "for", "while", "return", "class"],
            _ => vec!["function", "if", "else", "for", "while", "return"],
        };

        for keyword in keywords {
            let count = content.matches(keyword).count() as f32;
            features.push(count / words.max(1.0)); // Normalized frequency
        }

        // Structural features
        let brace_count = content.matches('{').count() as f32;
        let paren_count = content.matches('(').count() as f32;
        let bracket_count = content.matches('[').count() as f32;
        
        features.push(brace_count / chars.max(1.0));
        features.push(paren_count / chars.max(1.0));
        features.push(bracket_count / chars.max(1.0));

        Ok(features)
    }
}

#[cfg(feature = "code-memory")]
#[async_trait::async_trait]
impl MultiModalProcessor for CodeMemoryProcessor {
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        let start_time = std::time::Instant::now();
        
        // Validate content type and extract info
        let (language, lines, complexity_score) = match content_type {
            ContentType::Code { language, lines, complexity_score } => {
                (language.clone(), *lines, *complexity_score)
            }
            _ => return Err(SynapticError::ProcessingError("Invalid content type for code processor".to_string())),
        };

        // Convert bytes to string
        let code_content = String::from_utf8(content.to_vec())
            .map_err(|e| SynapticError::ProcessingError(format!("Invalid UTF-8 content: {}", e)))?;

        // Check size limits
        if code_content.len() > self.config.max_file_size {
            return Err(SynapticError::ProcessingError(format!(
                "Code file too large: {} bytes exceeds limit of {} bytes",
                code_content.len(), self.config.max_file_size
            )));
        }

        // Parse code and extract functions
        #[cfg(feature = "code-memory")]
        let functions = {
            if self.config.enable_syntax_analysis && self.config.enable_function_analysis {
                let mut parser = Parser::new();

                let ts_lang_opt = match &language {
                    #[cfg(feature = "tree-sitter-rust")]
                    CodeLanguage::Rust => Some(unsafe { tree_sitter_rust() }),
                    #[cfg(feature = "tree-sitter-python")]
                    CodeLanguage::Python => Some(unsafe { tree_sitter_python() }),
                    #[cfg(feature = "tree-sitter-javascript")]
                    CodeLanguage::JavaScript => Some(unsafe { tree_sitter_javascript() }),
                    _ => None,
                };

                if let Some(ts_lang) = ts_lang_opt {
                    if parser.set_language(ts_lang).is_ok() {
                        if let Some(tree) = parser.parse(&code_content, None) {
                            self.extract_functions(&tree, &code_content, &language)?
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        };

        #[cfg(not(feature = "code-memory"))]
        let functions = Vec::new();
        
        // Calculate complexity metrics
        let complexity_metrics = self.calculate_complexity_metrics(&code_content, &functions);
        
        // Extract dependencies
        let dependencies = self.extract_dependencies(&code_content, &language);
        
        // Extract semantic features
        let semantic_features = self.extract_semantic_features(&code_content, &language).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let memory = MultiModalMemory {
            id: Uuid::new_v4().to_string(),
            content_type: content_type.clone(),
            primary_content: content.to_vec(),
            metadata: MultiModalMetadata {
                title: None,
                description: None,
                tags: vec![language.to_string()],
                source: None,
                quality_score: complexity_metrics.maintainability_index / 100.0,
                processing_info: ProcessingInfo {
                    processor_version: "1.0.0".to_string(),
                    processing_time_ms: processing_time,
                    algorithms_used: vec!["ast_parsing".to_string(), "complexity_analysis".to_string()],
                    confidence_scores: HashMap::new(),
                },
                content_specific: ContentSpecificMetadata::Code {
                    ast_summary: Some(format!("Functions: {}, Lines: {}", functions.len(), lines)),
                    dependencies,
                    functions,
                    complexity_metrics,
                },
            },
            extracted_features: {
                let mut features = HashMap::new();
                let semantic_value = serde_json::to_value(semantic_features)
                    .map_err(|e| SynapticError::ProcessingError(format!("Failed to serialize semantic features: {}", e)))?;
                features.insert("semantic_features".to_string(), semantic_value);
                features
            },
            cross_modal_links: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        Ok(memory)
    }

    async fn extract_features(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        let language = match content_type {
            ContentType::Code { language, .. } => language,
            _ => return Err(SynapticError::ProcessingError("Invalid content type".to_string())),
        };

        let code_content = String::from_utf8(content.to_vec())
            .map_err(|e| SynapticError::ProcessingError(format!("Invalid UTF-8 content: {}", e)))?;

        self.extract_semantic_features(&code_content, language).await
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
            if let Some(features_value) = candidate.extracted_features.get("semantic_features") {
                if let Ok(features) = serde_json::from_value::<Vec<f32>>(features_value.clone()) {
                    let similarity = self.calculate_similarity(query_features, &features).await?;
                    similarities.push((candidate.id.clone(), similarity));
                }
            }
        }

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities)
    }
}

impl ToString for CodeLanguage {
    fn to_string(&self) -> String {
        match self {
            CodeLanguage::Rust => "rust".to_string(),
            CodeLanguage::Python => "python".to_string(),
            CodeLanguage::JavaScript => "javascript".to_string(),
            CodeLanguage::TypeScript => "typescript".to_string(),
            CodeLanguage::Java => "java".to_string(),
            CodeLanguage::CSharp => "csharp".to_string(),
            CodeLanguage::Cpp => "cpp".to_string(),
            CodeLanguage::C => "c".to_string(),
            CodeLanguage::Go => "go".to_string(),
            CodeLanguage::Swift => "swift".to_string(),
            CodeLanguage::Kotlin => "kotlin".to_string(),
            CodeLanguage::Ruby => "ruby".to_string(),
            CodeLanguage::Php => "php".to_string(),
            CodeLanguage::Html => "html".to_string(),
            CodeLanguage::Css => "css".to_string(),
            CodeLanguage::Sql => "sql".to_string(),
            CodeLanguage::Shell => "shell".to_string(),
            CodeLanguage::Other(name) => name.clone(),
        }
    }
}
