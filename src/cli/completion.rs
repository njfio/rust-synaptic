//! Auto-completion Support for Synaptic CLI
//!
//! This module provides sophisticated auto-completion for SyQL queries,
//! shell commands, and file paths with context-aware suggestions.

use crate::error::Result;
use std::collections::HashMap;

/// Auto-completion engine
pub struct CompletionEngine {
    /// SyQL keywords
    syql_keywords: Vec<String>,
    /// Shell commands
    shell_commands: Vec<String>,
    /// Memory types
    memory_types: Vec<String>,
    /// Relationship types
    relationship_types: Vec<String>,
    /// Function names
    functions: Vec<String>,
    /// Completion cache
    cache: HashMap<String, Vec<CompletionItem>>,
}

/// Completion item
#[derive(Debug, Clone)]
pub struct CompletionItem {
    /// Completion text
    pub text: String,
    /// Display text (may include additional info)
    pub display: String,
    /// Item type
    pub item_type: CompletionType,
    /// Description
    pub description: Option<String>,
    /// Priority (higher = more relevant)
    pub priority: u32,
}

/// Completion types
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionType {
    Keyword,
    Command,
    Function,
    MemoryType,
    RelationshipType,
    Variable,
    File,
    Directory,
    MemoryId,
    Property,
}

impl CompletionEngine {
    /// Create a new completion engine
    pub fn new() -> Self {
        Self {
            syql_keywords: vec![
                "SELECT".to_string(), "FROM".to_string(), "WHERE".to_string(),
                "ORDER".to_string(), "BY".to_string(), "LIMIT".to_string(),
                "GROUP".to_string(), "HAVING".to_string(), "MATCH".to_string(),
                "CREATE".to_string(), "UPDATE".to_string(), "DELETE".to_string(),
                "SHOW".to_string(), "EXPLAIN".to_string(), "DESCRIBE".to_string(),
                "AND".to_string(), "OR".to_string(), "NOT".to_string(),
                "IN".to_string(), "BETWEEN".to_string(), "LIKE".to_string(),
                "REGEX".to_string(), "EXISTS".to_string(), "CASE".to_string(),
                "WHEN".to_string(), "THEN".to_string(), "ELSE".to_string(),
                "END".to_string(), "AS".to_string(), "DISTINCT".to_string(),
                "PATH".to_string(), "SHORTEST".to_string(), "ALL".to_string(),
                "PATHS".to_string(), "CONNECTED".to_string(), "MEMORY".to_string(),
                "RELATIONSHIP".to_string(), "NODE".to_string(), "EDGE".to_string(),
            ],
            shell_commands: vec![
                "help".to_string(), "exit".to_string(), "quit".to_string(),
                "clear".to_string(), "history".to_string(), "set".to_string(),
                "get".to_string(), "format".to_string(), "timing".to_string(),
                "stats".to_string(), "explain".to_string(),
            ],
            memory_types: vec![
                "text".to_string(), "image".to_string(), "audio".to_string(),
                "video".to_string(), "document".to_string(), "code".to_string(),
                "data".to_string(), "structured".to_string(), "unstructured".to_string(),
            ],
            relationship_types: vec![
                "related_to".to_string(), "contains".to_string(), "references".to_string(),
                "similar_to".to_string(), "derived_from".to_string(), "depends_on".to_string(),
                "part_of".to_string(), "follows".to_string(), "precedes".to_string(),
            ],
            functions: vec![
                "COUNT".to_string(), "SUM".to_string(), "AVG".to_string(),
                "MIN".to_string(), "MAX".to_string(), "FIRST".to_string(),
                "LAST".to_string(), "COLLECT".to_string(), "LENGTH".to_string(),
                "SIZE".to_string(), "TYPE".to_string(), "ID".to_string(),
                "PROPERTIES".to_string(), "KEYS".to_string(), "VALUES".to_string(),
                "SUBSTRING".to_string(), "LOWER".to_string(), "UPPER".to_string(),
                "TRIM".to_string(), "REPLACE".to_string(), "SPLIT".to_string(),
                "CONCAT".to_string(), "NOW".to_string(), "DATE".to_string(),
                "SIMILARITY".to_string(), "DISTANCE".to_string(), "CENTRALITY".to_string(),
                "SHORTEST_PATH".to_string(), "ALL_PATHS".to_string(),
            ],
            cache: HashMap::new(),
        }
    }

    /// Get completions for a given input
    pub async fn get_completions(&mut self, input: &str, cursor_position: usize) -> Result<Vec<CompletionItem>> {
        // Check cache first
        let cache_key = format!("{}:{}", input, cursor_position);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let mut completions = Vec::new();
        
        // Analyze the input context
        let context = self.analyze_context(input, cursor_position);
        let current_word = self.get_current_word(input, cursor_position);

        match context {
            CompletionContext::SyqlKeyword => {
                completions.extend(self.complete_syql_keywords(&current_word));
            },
            CompletionContext::ShellCommand => {
                completions.extend(self.complete_shell_commands(&current_word));
            },
            CompletionContext::Function => {
                completions.extend(self.complete_functions(&current_word));
            },
            CompletionContext::MemoryType => {
                completions.extend(self.complete_memory_types(&current_word));
            },
            CompletionContext::RelationshipType => {
                completions.extend(self.complete_relationship_types(&current_word));
            },
            CompletionContext::Property => {
                completions.extend(self.complete_properties(&current_word));
            },
            CompletionContext::File => {
                completions.extend(self.complete_files(&current_word).await?);
            },
            CompletionContext::Variable => {
                completions.extend(self.complete_variables(&current_word));
            },
        }

        // Sort by priority and relevance
        completions.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| a.text.len().cmp(&b.text.len()))
                .then_with(|| a.text.cmp(&b.text))
        });

        // Cache the result
        self.cache.insert(cache_key, completions.clone());

        Ok(completions)
    }

    /// Analyze completion context
    fn analyze_context(&self, input: &str, cursor_position: usize) -> CompletionContext {
        let before_cursor = &input[..cursor_position.min(input.len())];
        let tokens: Vec<&str> = before_cursor.split_whitespace().collect();

        if tokens.is_empty() {
            return CompletionContext::SyqlKeyword;
        }

        let last_token = tokens.last().map(|t| t.to_uppercase()).unwrap_or_default();

        // Check for shell commands (starting with no SyQL keywords)
        if tokens.len() == 1 && self.shell_commands.iter().any(|cmd| cmd.to_uppercase().starts_with(&last_token)) {
            return CompletionContext::ShellCommand;
        }

        // Check for SyQL contexts
        match last_token.as_str() {
            "SELECT" | "MATCH" | "WHERE" | "HAVING" | "ORDER" | "GROUP" => CompletionContext::SyqlKeyword,
            "FROM" | "JOIN" => CompletionContext::MemoryType,
            "TYPE" | "RELATIONSHIP" => CompletionContext::RelationshipType,
            _ => {
                // Check if we're in a function context
                if before_cursor.contains('(') && !before_cursor.ends_with(')') {
                    CompletionContext::Function
                } else if before_cursor.contains('.') {
                    CompletionContext::Property
                } else if before_cursor.contains('/') || before_cursor.contains('\\') {
                    CompletionContext::File
                } else if before_cursor.contains('$') {
                    CompletionContext::Variable
                } else {
                    CompletionContext::SyqlKeyword
                }
            }
        }
    }

    /// Get the current word being typed
    fn get_current_word(&self, input: &str, cursor_position: usize) -> String {
        let before_cursor = &input[..cursor_position.min(input.len())];
        
        // Find the start of the current word
        let word_start = before_cursor.rfind(|c: char| c.is_whitespace() || "()[]{},.;".contains(c))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        before_cursor[word_start..].to_string()
    }

    /// Complete SyQL keywords
    fn complete_syql_keywords(&self, prefix: &str) -> Vec<CompletionItem> {
        self.syql_keywords
            .iter()
            .filter(|keyword| keyword.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|keyword| CompletionItem {
                text: keyword.clone(),
                display: keyword.clone(),
                item_type: CompletionType::Keyword,
                description: Some(format!("SyQL keyword: {}", keyword)),
                priority: if keyword.to_lowercase() == prefix.to_lowercase() { 100 } else { 80 },
            })
            .collect()
    }

    /// Complete shell commands
    fn complete_shell_commands(&self, prefix: &str) -> Vec<CompletionItem> {
        self.shell_commands
            .iter()
            .filter(|cmd| cmd.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|cmd| CompletionItem {
                text: cmd.clone(),
                display: cmd.clone(),
                item_type: CompletionType::Command,
                description: Some(format!("Shell command: {}", cmd)),
                priority: if cmd.to_lowercase() == prefix.to_lowercase() { 100 } else { 90 },
            })
            .collect()
    }

    /// Complete functions
    fn complete_functions(&self, prefix: &str) -> Vec<CompletionItem> {
        self.functions
            .iter()
            .filter(|func| func.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|func| CompletionItem {
                text: format!("{}()", func),
                display: format!("{}()", func),
                item_type: CompletionType::Function,
                description: Some(format!("SyQL function: {}", func)),
                priority: if func.to_lowercase() == prefix.to_lowercase() { 100 } else { 85 },
            })
            .collect()
    }

    /// Complete memory types
    fn complete_memory_types(&self, prefix: &str) -> Vec<CompletionItem> {
        self.memory_types
            .iter()
            .filter(|mtype| mtype.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|mtype| CompletionItem {
                text: mtype.clone(),
                display: mtype.clone(),
                item_type: CompletionType::MemoryType,
                description: Some(format!("Memory type: {}", mtype)),
                priority: 75,
            })
            .collect()
    }

    /// Complete relationship types
    fn complete_relationship_types(&self, prefix: &str) -> Vec<CompletionItem> {
        self.relationship_types
            .iter()
            .filter(|rtype| rtype.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|rtype| CompletionItem {
                text: rtype.clone(),
                display: rtype.clone(),
                item_type: CompletionType::RelationshipType,
                description: Some(format!("Relationship type: {}", rtype)),
                priority: 75,
            })
            .collect()
    }

    /// Complete properties
    fn complete_properties(&self, prefix: &str) -> Vec<CompletionItem> {
        let properties = vec![
            "id", "content", "type", "created_at", "updated_at", "tags",
            "metadata", "size", "checksum", "version", "author", "source",
        ];

        properties
            .iter()
            .filter(|prop| prop.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|prop| CompletionItem {
                text: prop.to_string(),
                display: prop.to_string(),
                item_type: CompletionType::Property,
                description: Some(format!("Property: {}", prop)),
                priority: 70,
            })
            .collect()
    }

    /// Complete file paths
    async fn complete_files(&self, prefix: &str) -> Result<Vec<CompletionItem>> {
        let mut completions = Vec::new();
        
        // Simple file completion (in a real implementation, you'd use proper file system APIs)
        let path = std::path::Path::new(prefix);
        let (dir, filename_prefix) = if prefix.ends_with('/') || prefix.ends_with('\\') {
            (path, "")
        } else {
            (path.parent().unwrap_or(std::path::Path::new(".")), 
             path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
        };

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(filename_prefix) {
                    let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                    completions.push(CompletionItem {
                        text: if is_dir { format!("{}/", name) } else { name.clone() },
                        display: name,
                        item_type: if is_dir { CompletionType::Directory } else { CompletionType::File },
                        description: Some(if is_dir { "Directory".to_string() } else { "File".to_string() }),
                        priority: if is_dir { 60 } else { 50 },
                    });
                }
            }
        }

        Ok(completions)
    }

    /// Complete variables
    fn complete_variables(&self, prefix: &str) -> Vec<CompletionItem> {
        // In a real implementation, you'd get variables from the shell state
        let variables = vec!["$USER", "$HOME", "$PATH", "$PWD"];
        
        variables
            .iter()
            .filter(|var| var.to_lowercase().starts_with(&prefix.to_lowercase()))
            .map(|var| CompletionItem {
                text: var.to_string(),
                display: var.to_string(),
                item_type: CompletionType::Variable,
                description: Some(format!("Environment variable: {}", var)),
                priority: 65,
            })
            .collect()
    }

    /// Clear completion cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Add custom completion items
    pub fn add_custom_completions(&mut self, _items: Vec<CompletionItem>) {
        // In a real implementation, you'd store these and include them in completions
        // For now, this is a placeholder
    }
}

/// Completion context types
#[derive(Debug, Clone, PartialEq)]
enum CompletionContext {
    SyqlKeyword,
    ShellCommand,
    Function,
    MemoryType,
    RelationshipType,
    Property,
    File,
    Variable,
}

impl Default for CompletionEngine {
    fn default() -> Self {
        Self::new()
    }
}
