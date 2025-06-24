//! Command History Management for Synaptic CLI
//!
//! This module provides sophisticated command history management with
//! persistent storage, search capabilities, and intelligent deduplication.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};

/// Command history manager
pub struct HistoryManager {
    /// History entries
    entries: VecDeque<HistoryEntry>,
    /// Maximum history size
    max_size: usize,
    /// History file path
    file_path: Option<PathBuf>,
    /// Whether to deduplicate consecutive entries
    deduplicate: bool,
    /// Whether to ignore commands starting with space
    ignore_space: bool,
}

/// History entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Command text
    pub command: String,
    /// Execution timestamp
    pub timestamp: DateTime<Utc>,
    /// Execution duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Whether the command succeeded
    pub success: bool,
    /// Command type
    pub command_type: CommandType,
    /// Session ID
    pub session_id: String,
}

/// Command types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommandType {
    /// SyQL query command
    SyqlQuery,
    /// Shell command
    ShellCommand,
    /// System command
    SystemCommand,
}

impl HistoryManager {
    /// Create a new history manager
    pub fn new(max_size: usize, file_path: Option<PathBuf>) -> Self {
        Self {
            entries: VecDeque::new(),
            max_size,
            file_path,
            deduplicate: true,
            ignore_space: true,
        }
    }

    /// Load history from file
    pub async fn load(&mut self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if path.exists() {
                let content = tokio::fs::read_to_string(path).await?;
                let entries: Vec<HistoryEntry> = serde_json::from_str(&content)?;
                
                self.entries.clear();
                for entry in entries {
                    self.entries.push_back(entry);
                }
                
                // Ensure we don't exceed max size
                while self.entries.len() > self.max_size {
                    self.entries.pop_front();
                }
            }
        }
        Ok(())
    }

    /// Save history to file
    pub async fn save(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            // Create parent directories if they don't exist
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            let entries: Vec<&HistoryEntry> = self.entries.iter().collect();
            let content = serde_json::to_string_pretty(&entries)?;
            tokio::fs::write(path, content).await?;
        }
        Ok(())
    }

    /// Add a command to history
    pub fn add_command(
        &mut self,
        command: String,
        duration_ms: Option<u64>,
        success: bool,
        command_type: CommandType,
        session_id: String,
    ) {
        // Skip empty commands
        if command.trim().is_empty() {
            return;
        }

        // Skip commands starting with space if ignore_space is enabled
        if self.ignore_space && command.starts_with(' ') {
            return;
        }

        // Check for deduplication
        if self.deduplicate {
            if let Some(last_entry) = self.entries.back() {
                if last_entry.command == command {
                    return; // Skip duplicate
                }
            }
        }

        let entry = HistoryEntry {
            command,
            timestamp: Utc::now(),
            duration_ms,
            success,
            command_type,
            session_id,
        };

        self.entries.push_back(entry);

        // Maintain max size
        while self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
    }

    /// Get all history entries
    pub fn get_entries(&self) -> Vec<&HistoryEntry> {
        self.entries.iter().collect()
    }

    /// Get recent entries
    pub fn get_recent(&self, count: usize) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .rev()
            .take(count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Search history
    pub fn search(&self, query: &str) -> Vec<&HistoryEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .filter(|entry| entry.command.to_lowercase().contains(&query_lower))
            .collect()
    }

    /// Search history with regex
    pub fn search_regex(&self, pattern: &str) -> Result<Vec<&HistoryEntry>> {
        let regex = regex::Regex::new(pattern)?;
        Ok(self.entries
            .iter()
            .filter(|entry| regex.is_match(&entry.command))
            .collect())
    }

    /// Get history by command type
    pub fn get_by_type(&self, command_type: CommandType) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.command_type == command_type)
            .collect()
    }

    /// Get history by session
    pub fn get_by_session(&self, session_id: &str) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.session_id == session_id)
            .collect()
    }

    /// Get history statistics
    pub fn get_statistics(&self) -> HistoryStatistics {
        let total_commands = self.entries.len();
        let successful_commands = self.entries.iter().filter(|e| e.success).count();
        let failed_commands = total_commands - successful_commands;

        let syql_commands = self.entries.iter().filter(|e| e.command_type == CommandType::SyqlQuery).count();
        let shell_commands = self.entries.iter().filter(|e| e.command_type == CommandType::ShellCommand).count();
        let system_commands = self.entries.iter().filter(|e| e.command_type == CommandType::SystemCommand).count();

        let avg_duration = if total_commands > 0 {
            let total_duration: u64 = self.entries
                .iter()
                .filter_map(|e| e.duration_ms)
                .sum();
            Some(total_duration / total_commands as u64)
        } else {
            None
        };

        let most_used_commands = self.get_most_used_commands(10);

        HistoryStatistics {
            total_commands,
            successful_commands,
            failed_commands,
            syql_commands,
            shell_commands,
            system_commands,
            avg_duration_ms: avg_duration,
            most_used_commands,
        }
    }

    /// Get most frequently used commands
    pub fn get_most_used_commands(&self, limit: usize) -> Vec<(String, usize)> {
        let mut command_counts = std::collections::HashMap::new();
        
        for entry in &self.entries {
            *command_counts.entry(entry.command.clone()).or_insert(0) += 1;
        }

        let mut sorted_commands: Vec<(String, usize)> = command_counts.into_iter().collect();
        sorted_commands.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_commands.truncate(limit);
        
        sorted_commands
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Remove entries older than specified duration
    pub fn cleanup_old_entries(&mut self, max_age_days: u32) {
        let cutoff = Utc::now() - chrono::Duration::days(max_age_days as i64);
        self.entries.retain(|entry| entry.timestamp > cutoff);
    }

    /// Export history to different formats
    pub async fn export(&self, path: &Path, format: ExportFormat) -> Result<()> {
        let content = match format {
            ExportFormat::Json => {
                serde_json::to_string_pretty(&self.entries)?
            },
            ExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("timestamp,command,duration_ms,success,command_type,session_id\n");
                
                for entry in &self.entries {
                    csv.push_str(&format!(
                        "{},{},{},{},{:?},{}\n",
                        entry.timestamp.to_rfc3339(),
                        entry.command.replace(',', "\\,"),
                        entry.duration_ms.unwrap_or(0),
                        entry.success,
                        entry.command_type,
                        entry.session_id
                    ));
                }
                csv
            },
            ExportFormat::Text => {
                let mut text = String::new();
                for entry in &self.entries {
                    text.push_str(&format!(
                        "[{}] {} ({}ms) - {}\n",
                        entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                        entry.command,
                        entry.duration_ms.unwrap_or(0),
                        if entry.success { "SUCCESS" } else { "FAILED" }
                    ));
                }
                text
            },
        };

        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Import history from file
    pub async fn import(&mut self, path: &Path, format: ImportFormat) -> Result<usize> {
        let content = tokio::fs::read_to_string(path).await?;
        let mut imported_count = 0;

        match format {
            ImportFormat::Json => {
                let entries: Vec<HistoryEntry> = serde_json::from_str(&content)?;
                for entry in entries {
                    self.entries.push_back(entry);
                    imported_count += 1;
                }
            },
            ImportFormat::Text => {
                // Simple text format: one command per line
                for line in content.lines() {
                    if !line.trim().is_empty() {
                        let entry = HistoryEntry {
                            command: line.to_string(),
                            timestamp: Utc::now(),
                            duration_ms: None,
                            success: true,
                            command_type: CommandType::SyqlQuery, // Default assumption
                            session_id: "imported".to_string(),
                        };
                        self.entries.push_back(entry);
                        imported_count += 1;
                    }
                }
            },
        }

        // Maintain max size
        while self.entries.len() > self.max_size {
            self.entries.pop_front();
        }

        Ok(imported_count)
    }

    /// Get configuration
    pub fn get_config(&self) -> HistoryConfig {
        HistoryConfig {
            max_size: self.max_size,
            file_path: self.file_path.clone(),
            deduplicate: self.deduplicate,
            ignore_space: self.ignore_space,
        }
    }

    /// Update configuration
    pub fn update_config(&mut self, config: HistoryConfig) {
        self.max_size = config.max_size;
        self.file_path = config.file_path;
        self.deduplicate = config.deduplicate;
        self.ignore_space = config.ignore_space;

        // Adjust current entries if max_size changed
        while self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
    }
}

/// History statistics
#[derive(Debug, Clone)]
pub struct HistoryStatistics {
    /// Total number of commands executed
    pub total_commands: usize,
    /// Number of successful commands
    pub successful_commands: usize,
    /// Number of failed commands
    pub failed_commands: usize,
    /// Number of SyQL commands
    pub syql_commands: usize,
    /// Number of shell commands
    pub shell_commands: usize,
    pub system_commands: usize,
    pub avg_duration_ms: Option<u64>,
    pub most_used_commands: Vec<(String, usize)>,
}

/// History configuration
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    pub max_size: usize,
    pub file_path: Option<PathBuf>,
    pub deduplicate: bool,
    pub ignore_space: bool,
}

/// Export formats
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Text,
}

/// Import formats
#[derive(Debug, Clone)]
pub enum ImportFormat {
    Json,
    Text,
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self::new(1000, None)
    }
}
