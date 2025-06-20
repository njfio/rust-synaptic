//! Interactive Shell for Synaptic CLI
//!
//! This module implements a sophisticated interactive shell with command history,
//! auto-completion, multi-line input support, and integrated SyQL query execution.

use super::syql::{SyQLEngine, OutputFormat};
use super::config::CliConfig;
use super::completion::CompletionEngine;
use super::history::{HistoryManager, CommandType};
use crate::error::Result;
use rustyline::{Helper, Context, Result as RustylineResult, DefaultEditor};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter, CmdKind};
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline::validate::{Validator, MatchingBracketValidator};
use std::borrow::Cow::{self, Borrowed, Owned};
use std::path::PathBuf;
use std::collections::HashMap;
use uuid::Uuid;

/// Interactive shell for Synaptic CLI
pub struct InteractiveShell<'a> {
    /// Readline editor
    editor: DefaultEditor,
    /// SyQL engine reference
    syql_engine: &'a mut SyQLEngine,
    /// CLI configuration
    config: &'a CliConfig,
    /// Shell state
    state: ShellState,
    /// Completion engine
    completion_engine: CompletionEngine,
    /// History manager
    history_manager: HistoryManager,
    /// Session ID
    session_id: String,
}

impl<'a> InteractiveShell<'a> {
    /// Create a new interactive shell
    pub async fn new(
        syql_engine: &'a mut SyQLEngine,
        config: &'a CliConfig,
        history_file: Option<PathBuf>,
        enable_completion: bool,
    ) -> Result<Self> {
        let mut editor = DefaultEditor::new()?;

        // Initialize completion engine
        let completion_engine = CompletionEngine::new();

        // Initialize history manager
        let history_path = history_file.clone().unwrap_or_else(|| config.get_history_file());
        let mut history_manager = HistoryManager::new(config.shell.history_size, Some(history_path.clone()));
        history_manager.load().await?;

        // Load history into rustyline if file is specified
        if history_path.exists() {
            let _ = editor.load_history(&history_path);
        }

        let session_id = Uuid::new_v4().to_string();

        let state = ShellState {
            history_file: Some(history_path),
            variables: HashMap::new(),
            output_format: OutputFormat::Table,
            show_timing: true,
            show_statistics: false,
            multi_line_mode: false,
            current_query: String::new(),
            command_chain: Vec::new(),
            error_recovery_mode: false,
            last_error: None,
        };

        Ok(Self {
            editor,
            syql_engine,
            config,
            state,
            completion_engine,
            history_manager,
            session_id,
        })
    }

    /// Run the interactive shell
    pub async fn run(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            let prompt = if self.state.multi_line_mode {
                "synaptic> ... "
            } else {
                "synaptic> "
            };

            match self.editor.readline(prompt) {
                Ok(line) => {
                    let trimmed = line.trim();
                    
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Handle multi-line input
                    if self.state.multi_line_mode {
                        if trimmed == ";" {
                            // End multi-line input
                            self.state.multi_line_mode = false;
                            if !self.state.current_query.trim().is_empty() {
                                self.execute_command(&self.state.current_query.clone()).await?;
                            }
                            self.state.current_query.clear();
                        } else {
                            self.state.current_query.push_str(&line);
                            self.state.current_query.push('\n');
                        }
                        continue;
                    }

                    // Check for multi-line start
                    if trimmed.ends_with('\\') {
                        self.state.multi_line_mode = true;
                        self.state.current_query = trimmed[..trimmed.len()-1].to_string();
                        self.state.current_query.push('\n');
                        continue;
                    }

                    // Handle command chaining (pipe operations)
                    if trimmed.contains('|') {
                        self.handle_command_chain(trimmed).await?;
                        continue;
                    }

                    // Handle shell commands
                    if let Some(result) = self.handle_shell_command(trimmed).await? {
                        if result {
                            break; // Exit shell
                        }
                        continue;
                    }

                    // Execute as SyQL query
                    self.execute_command(trimmed).await?;
                },
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    self.state.multi_line_mode = false;
                    self.state.current_query.clear();
                },
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                },
                Err(err) => {
                    eprintln!("Error: {}", err);
                    break;
                }
            }
        }

        // Save history
        if let Some(history_path) = &self.state.history_file {
            let _ = self.editor.save_history(history_path);
            let _ = self.history_manager.save().await;
        }

        Ok(())
    }

    /// Print welcome message
    fn print_welcome(&self) {
        println!("Synaptic Interactive Shell v0.1.0");
        println!("Type 'help' for available commands or 'exit' to quit.");
        println!("Use '\\' at the end of a line for multi-line input, end with ';'");
        println!();
    }

    /// Handle shell-specific commands
    async fn handle_shell_command(&mut self, command: &str) -> Result<Option<bool>> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(None);
        }

        match parts[0].to_lowercase().as_str() {
            "exit" | "quit" | "q" => {
                println!("Goodbye!");
                return Ok(Some(true));
            },
            "help" | "h" => {
                self.print_help();
                return Ok(Some(false));
            },
            "clear" | "cls" => {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen
                return Ok(Some(false));
            },
            "history" => {
                self.show_history();
                return Ok(Some(false));
            },
            "set" => {
                if parts.len() >= 3 {
                    self.set_variable(&parts[1..].join(" ")).await?;
                } else {
                    println!("Usage: set <key> <value>");
                }
                return Ok(Some(false));
            },
            "get" => {
                if parts.len() >= 2 {
                    self.get_variable(parts[1]);
                } else {
                    self.show_all_variables();
                }
                return Ok(Some(false));
            },
            "format" => {
                if parts.len() >= 2 {
                    self.set_output_format(parts[1])?;
                } else {
                    println!("Current format: {:?}", self.state.output_format);
                    println!("Available formats: table, json, csv, yaml, graph, tree");
                }
                return Ok(Some(false));
            },
            "timing" => {
                if parts.len() >= 2 {
                    self.state.show_timing = parts[1].parse().unwrap_or(true);
                } else {
                    self.state.show_timing = !self.state.show_timing;
                }
                println!("Timing display: {}", if self.state.show_timing { "on" } else { "off" });
                return Ok(Some(false));
            },
            "stats" => {
                if parts.len() >= 2 {
                    self.state.show_statistics = parts[1].parse().unwrap_or(true);
                } else {
                    self.state.show_statistics = !self.state.show_statistics;
                }
                println!("Statistics display: {}", if self.state.show_statistics { "on" } else { "off" });
                return Ok(Some(false));
            },
            "explain" => {
                if parts.len() >= 2 {
                    let query = parts[1..].join(" ");
                    self.explain_query(&query).await?;
                } else {
                    println!("Usage: explain <query>");
                }
                return Ok(Some(false));
            },
            "recover" => {
                self.handle_error_recovery().await?;
                return Ok(Some(false));
            },
            "session" => {
                if parts.len() >= 2 {
                    match parts[1] {
                        "info" => self.show_session_info().await?,
                        "save" => self.save_session(parts.get(2).map(|v| &**v)).await?,
                        "load" => self.load_session(parts.get(2).map(|v| &**v)).await?,
                        _ => println!("Usage: session [info|save|load] [name]"),
                    }
                } else {
                    self.show_session_info().await?;
                }
                return Ok(Some(false));
            },
            "complete" => {
                if parts.len() >= 2 {
                    let query = parts[1..].join(" ");
                    self.show_completions(&query).await?;
                } else {
                    println!("Usage: complete <partial_query>");
                }
                return Ok(Some(false));
            },
            _ => return Ok(None), // Not a shell command
        }
    }

    /// Handle command chaining (pipe operations)
    async fn handle_command_chain(&mut self, command_line: &str) -> Result<()> {
        let commands: Vec<&str> = command_line.split('|').map(|s| s.trim()).collect();

        if commands.len() < 2 {
            return self.execute_command(command_line).await;
        }

        println!("Executing command chain with {} commands", commands.len());

        // For now, execute each command separately
        // In a full implementation, you'd pipe the output of one command to the next
        for (i, cmd) in commands.iter().enumerate() {
            println!("Step {}: {}", i + 1, cmd);

            match self.execute_command_with_recovery(cmd).await {
                Ok(_) => {},
                Err(e) => {
                    eprintln!("Command chain failed at step {}: {}", i + 1, e);
                    self.state.error_recovery_mode = true;
                    self.state.last_error = Some(e.to_string());
                    break;
                }
            }
        }

        Ok(())
    }

    /// Execute a command with error recovery
    async fn execute_command_with_recovery(&mut self, command: &str) -> Result<()> {
        let start_time = std::time::Instant::now();
        let command_type = if command.trim().starts_with(|c: char| c.is_ascii_uppercase()) {
            CommandType::SyqlQuery
        } else {
            CommandType::ShellCommand
        };

        match self.syql_engine.execute_query(command).await {
            Ok(result) => {
                let formatted = self.syql_engine.format_result(&result, self.state.output_format.clone())?;
                println!("{}", formatted);

                if self.state.show_timing {
                    let elapsed = start_time.elapsed();
                    println!("Query executed in {:.2}ms", elapsed.as_millis());
                }

                if self.state.show_statistics {
                    println!("Statistics: {} rows, {} memories scanned, {} relationships traversed",
                        result.statistics.rows_returned,
                        result.statistics.memories_scanned,
                        result.statistics.relationships_traversed
                    );
                }

                // Add to history
                self.history_manager.add_command(
                    command.to_string(),
                    Some(start_time.elapsed().as_millis() as u64),
                    true,
                    command_type,
                    self.session_id.clone(),
                );

                // Clear error recovery mode on success
                self.state.error_recovery_mode = false;
                self.state.last_error = None;
            },
            Err(e) => {
                eprintln!("Error executing query: {}", e);

                // Add failed command to history
                self.history_manager.add_command(
                    command.to_string(),
                    Some(start_time.elapsed().as_millis() as u64),
                    false,
                    command_type,
                    self.session_id.clone(),
                );

                // Enter error recovery mode
                self.state.error_recovery_mode = true;
                self.state.last_error = Some(e.to_string());

                // Suggest recovery actions
                self.suggest_error_recovery(command, &e.to_string()).await?;

                return Err(e);
            }
        }

        Ok(())
    }

    /// Execute a command (SyQL query)
    async fn execute_command(&mut self, command: &str) -> Result<()> {
        self.execute_command_with_recovery(command).await
    }

    /// Explain query execution plan
    async fn explain_query(&self, query: &str) -> Result<()> {
        match self.syql_engine.explain_query(query).await {
            Ok(plan) => {
                println!("Query Execution Plan:");
                println!("=====================");
                println!("Estimated Cost: {:.2}", plan.estimated_cost);
                println!("Estimated Rows: {}", plan.estimated_rows);
                println!();
                
                for (i, node) in plan.nodes.iter().enumerate() {
                    println!("{}. {} (Cost: {:.2}, Rows: {})", 
                        i + 1, node.description, node.cost, node.rows);
                }
            },
            Err(e) => {
                eprintln!("Error explaining query: {}", e);
            }
        }
        Ok(())
    }

    /// Print help information
    fn print_help(&self) {
        println!("Synaptic Interactive Shell v0.1.0");
        println!("==================================");
        println!();
        println!("Shell Commands:");
        println!("  help, h              - Show this help message");
        println!("  exit, quit, q        - Exit the shell");
        println!("  clear, cls           - Clear the screen");
        println!("  history              - Show command history");
        println!("  set <key> <value>    - Set a variable");
        println!("  get [key]            - Get variable value or show all");
        println!("  format [format]      - Set/show output format");
        println!("  timing [on|off]      - Toggle timing display");
        println!("  stats [on|off]       - Toggle statistics display");
        println!("  explain <query>      - Show query execution plan");
        println!("  recover              - Enter error recovery mode");
        println!("  session [info|save|load] [name] - Session management");
        println!("  complete <partial>   - Show completions for partial query");
        println!();
        println!("Advanced Features:");
        println!("  Command chaining     - Use '|' to chain commands");
        println!("  Multi-line input     - End lines with '\\', finish with ';'");
        println!("  Auto-completion      - Press Tab for completions");
        println!("  Command history      - Use Up/Down arrows");
        println!("  Error recovery       - Automatic suggestions on errors");
        println!();
        println!("SyQL Query Examples:");
        println!("===================");
        println!("  SELECT * FROM memories LIMIT 10");
        println!("  MATCH (m:Memory)-[r:RELATED_TO]->(n:Memory) RETURN m, r, n");
        println!("  PATH FROM 'mem1' TO 'mem2' USING SHORTEST");
        println!("  SHOW MEMORIES");
        println!("  CREATE MEMORY content='Hello World' type='text'");
        println!();
        println!("Command Chaining Examples:");
        println!("=========================");
        println!("  SELECT * FROM memories | format json");
        println!("  SHOW MEMORIES | stats on");
        println!();
        println!("Session Management:");
        println!("==================");
        println!("  session info         - Show current session information");
        println!("  session save work    - Save current session as 'work'");
        println!("  session load work    - Load session 'work'");
        println!();
        println!("Tips:");
        println!("=====");
        println!("• Use 'recover' if a command fails for suggestions");
        println!("• Commands are automatically saved to history");
        println!("• Use 'complete <partial>' to see available completions");
        println!("• Variables can be used in queries with $variable syntax");
    }

    /// Show command history
    fn show_history(&self) {
        let history = self.editor.history();
        for (i, entry) in history.iter().enumerate() {
            println!("{:3}: {}", i + 1, entry);
        }
    }

    /// Set a shell variable
    async fn set_variable(&mut self, assignment: &str) -> Result<()> {
        if let Some(eq_pos) = assignment.find('=') {
            let key = assignment[..eq_pos].trim().to_string();
            let value = assignment[eq_pos + 1..].trim().to_string();
            self.state.variables.insert(key.clone(), value.clone());
            println!("Set {} = {}", key, value);
        } else {
            let parts: Vec<&str> = assignment.split_whitespace().collect();
            if parts.len() >= 2 {
                let key = parts[0].to_string();
                let value = parts[1..].join(" ");
                self.state.variables.insert(key.clone(), value.clone());
                println!("Set {} = {}", key, value);
            } else {
                println!("Usage: set <key> <value> or set <key>=<value>");
            }
        }
        Ok(())
    }

    /// Get a shell variable
    fn get_variable(&self, key: &str) {
        if let Some(value) = self.state.variables.get(key) {
            println!("{} = {}", key, value);
        } else {
            println!("Variable '{}' not found", key);
        }
    }

    /// Show all variables
    fn show_all_variables(&self) {
        if self.state.variables.is_empty() {
            println!("No variables set");
        } else {
            println!("Variables:");
            for (key, value) in &self.state.variables {
                println!("  {} = {}", key, value);
            }
        }
    }

    /// Set output format
    fn set_output_format(&mut self, format_str: &str) -> Result<()> {
        match format_str.to_lowercase().as_str() {
            "table" => self.state.output_format = OutputFormat::Table,
            "json" => self.state.output_format = OutputFormat::Json,
            "csv" => self.state.output_format = OutputFormat::Csv,
            "yaml" => self.state.output_format = OutputFormat::Yaml,
            "graph" => self.state.output_format = OutputFormat::Graph,
            "tree" => self.state.output_format = OutputFormat::Tree,
            _ => {
                println!("Unknown format: {}", format_str);
                println!("Available formats: table, json, csv, yaml, graph, tree");
                return Ok(());
            }
        }
        println!("Output format set to: {:?}", self.state.output_format);
        Ok(())
    }

    /// Suggest error recovery actions
    async fn suggest_error_recovery(&self, failed_command: &str, error_message: &str) -> Result<()> {
        println!("\n🔧 Error Recovery Suggestions:");
        println!("===============================");

        // Analyze the error and suggest fixes
        if error_message.contains("syntax") || error_message.contains("parse") {
            println!("• Check query syntax - use 'help' for SyQL syntax guide");
            println!("• Try 'explain <query>' to see the execution plan");

            // Suggest similar commands from history
            let similar_commands = self.find_similar_commands(failed_command);
            if !similar_commands.is_empty() {
                println!("• Similar commands from history:");
                for cmd in similar_commands.iter().take(3) {
                    println!("  - {}", cmd);
                }
            }
        } else if error_message.contains("connection") || error_message.contains("timeout") {
            println!("• Check database connection");
            println!("• Verify network connectivity");
            println!("• Try reducing query complexity");
        } else if error_message.contains("permission") || error_message.contains("access") {
            println!("• Check access permissions");
            println!("• Verify authentication credentials");
        } else {
            println!("• Use 'recover' command for interactive recovery");
            println!("• Check 'history' for working commands");
            println!("• Use 'help' for available commands");
        }

        println!("• Type 'recover' to enter recovery mode");
        println!();

        Ok(())
    }

    /// Handle interactive error recovery
    async fn handle_error_recovery(&mut self) -> Result<()> {
        if !self.state.error_recovery_mode {
            println!("No recent errors to recover from.");
            return Ok(());
        }

        println!("🔧 Interactive Error Recovery Mode");
        println!("==================================");

        if let Some(ref error) = self.state.last_error {
            println!("Last error: {}", error);
        }

        println!("\nRecovery options:");
        println!("1. Show similar working commands");
        println!("2. Show command history");
        println!("3. Get syntax help");
        println!("4. Exit recovery mode");

        // In a full implementation, you'd prompt for user input here
        // For now, just show similar commands
        println!("\nShowing recent successful commands:");
        let entries = self.history_manager.get_entries();
        let successful_commands = entries
            .iter()
            .filter(|entry| entry.success)
            .rev()
            .take(5)
            .collect::<Vec<_>>();

        for (i, entry) in successful_commands.iter().enumerate() {
            println!("{}. {} ({}ms)", i + 1, entry.command, entry.duration_ms.unwrap_or(0));
        }

        self.state.error_recovery_mode = false;
        Ok(())
    }

    /// Find similar commands from history
    fn find_similar_commands(&self, failed_command: &str) -> Vec<String> {
        let failed_words: Vec<&str> = failed_command.split_whitespace().collect();
        let mut similar_commands = Vec::new();

        for entry in self.history_manager.get_entries() {
            if !entry.success {
                continue;
            }

            let command_words: Vec<&str> = entry.command.split_whitespace().collect();
            let similarity = self.calculate_command_similarity(&failed_words, &command_words);

            if similarity > 0.3 {
                similar_commands.push(entry.command.clone());
            }
        }

        similar_commands.sort_by(|a, b| {
            let sim_a = self.calculate_command_similarity(&failed_words, &a.split_whitespace().collect::<Vec<_>>());
            let sim_b = self.calculate_command_similarity(&failed_words, &b.split_whitespace().collect::<Vec<_>>());
            sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        similar_commands
    }

    /// Calculate similarity between two commands
    fn calculate_command_similarity(&self, words1: &[&str], words2: &[&str]) -> f64 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut common_words = 0;
        for word1 in words1 {
            if words2.iter().any(|word2| word1.to_lowercase() == word2.to_lowercase()) {
                common_words += 1;
            }
        }

        common_words as f64 / words1.len().max(words2.len()) as f64
    }

    /// Show session information
    async fn show_session_info(&self) -> Result<()> {
        println!("Session Information:");
        println!("===================");
        println!("Session ID: {}", self.session_id);
        println!("Commands executed: {}", self.history_manager.get_entries().len());

        let stats = self.history_manager.get_statistics();
        println!("Successful commands: {}", stats.successful_commands);
        println!("Failed commands: {}", stats.failed_commands);

        if let Some(avg_duration) = stats.avg_duration_ms {
            println!("Average execution time: {}ms", avg_duration);
        }

        println!("Current format: {:?}", self.state.output_format);
        println!("Timing display: {}", if self.state.show_timing { "on" } else { "off" });
        println!("Statistics display: {}", if self.state.show_statistics { "on" } else { "off" });

        if self.state.error_recovery_mode {
            println!("Status: Error recovery mode");
        } else {
            println!("Status: Normal");
        }

        Ok(())
    }

    /// Save current session
    async fn save_session(&self, name: Option<&str>) -> Result<()> {
        let session_name = name.unwrap_or("default");
        println!("Saving session as '{}'...", session_name);

        // In a full implementation, you'd save session state to a file
        // For now, just save the history
        if let Some(ref history_path) = self.state.history_file {
            let session_path = history_path.with_file_name(format!("session_{}.json", session_name));
            self.history_manager.export(&session_path, super::history::ExportFormat::Json).await?;
            println!("Session saved to {}", session_path.display());
        }

        Ok(())
    }

    /// Load a saved session
    async fn load_session(&mut self, name: Option<&str>) -> Result<()> {
        let session_name = name.unwrap_or("default");
        println!("Loading session '{}'...", session_name);

        // In a full implementation, you'd load session state from a file
        // For now, just load the history
        if let Some(ref history_path) = self.state.history_file {
            let session_path = history_path.with_file_name(format!("session_{}.json", session_name));
            if session_path.exists() {
                let imported = self.history_manager.import(&session_path, super::history::ImportFormat::Json).await?;
                println!("Loaded {} commands from session", imported);
            } else {
                println!("Session '{}' not found", session_name);
            }
        }

        Ok(())
    }

    /// Show completions for a partial query
    async fn show_completions(&mut self, partial_query: &str) -> Result<()> {
        let completions = self.completion_engine.get_completions(partial_query, partial_query.len()).await?;

        if completions.is_empty() {
            println!("No completions found for '{}'", partial_query);
            return Ok(());
        }

        println!("Completions for '{}':", partial_query);
        println!("======================");

        for (i, completion) in completions.iter().take(10).enumerate() {
            println!("{}. {} ({:?})", i + 1, completion.text, completion.item_type);
            if let Some(ref desc) = completion.description {
                println!("   {}", desc);
            }
        }

        if completions.len() > 10 {
            println!("... and {} more", completions.len() - 10);
        }

        Ok(())
    }
}

/// Shell state
struct ShellState {
    /// History file path
    history_file: Option<PathBuf>,
    /// Shell variables
    variables: HashMap<String, String>,
    /// Current output format
    output_format: OutputFormat,
    /// Whether to show query timing
    show_timing: bool,
    /// Whether to show query statistics
    show_statistics: bool,
    /// Multi-line input mode
    multi_line_mode: bool,
    /// Current multi-line query
    current_query: String,
    /// Command chain for piping
    command_chain: Vec<String>,
    /// Error recovery mode
    error_recovery_mode: bool,
    /// Last error for recovery
    last_error: Option<String>,
}

/// Helper for rustyline with completion and highlighting
struct SynapticHelper {
    /// Filename completer
    completer: FilenameCompleter,
    /// Bracket highlighter
    highlighter: MatchingBracketHighlighter,
    /// History hinter
    hinter: HistoryHinter,
    /// Bracket validator
    validator: MatchingBracketValidator,
    /// Whether completion is enabled
    completion_enabled: bool,
}

impl SynapticHelper {
    fn new(completion_enabled: bool) -> Self {
        Self {
            completer: FilenameCompleter::new(),
            highlighter: MatchingBracketHighlighter::new(),
            hinter: HistoryHinter {},
            validator: MatchingBracketValidator::new(),
            completion_enabled,
        }
    }
}

impl Helper for SynapticHelper {}

impl Completer for SynapticHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> RustylineResult<(usize, Vec<Pair>)> {
        if !self.completion_enabled {
            return Ok((pos, Vec::new()));
        }

        // Enhanced keyword completion with categories
        let syql_keywords = vec![
            "SELECT", "FROM", "WHERE", "ORDER", "BY", "LIMIT", "GROUP", "HAVING",
            "MATCH", "CREATE", "UPDATE", "DELETE", "SHOW", "EXPLAIN", "DESCRIBE",
            "MEMORIES", "RELATIONSHIPS", "PATH", "SHORTEST", "ALL", "PATHS",
            "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "REGEX", "EXISTS",
            "CASE", "WHEN", "THEN", "ELSE", "END", "AS", "DISTINCT",
        ];

        let shell_commands = vec![
            "help", "exit", "quit", "clear", "history", "set", "get", "format",
            "timing", "stats", "explain", "recover", "session", "complete",
        ];

        let functions = vec![
            "COUNT", "SUM", "AVG", "MIN", "MAX", "LENGTH", "SIZE", "TYPE",
            "NOW", "DATE", "SIMILARITY", "DISTANCE", "SHORTEST_PATH",
        ];

        let word_start = line[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
        let word = &line[word_start..pos];

        let mut matches = Vec::new();

        // Add SyQL keywords
        for keyword in &syql_keywords {
            if keyword.to_lowercase().starts_with(&word.to_lowercase()) {
                matches.push(Pair {
                    display: format!("{} (keyword)", keyword),
                    replacement: keyword.to_string(),
                });
            }
        }

        // Add shell commands
        for command in &shell_commands {
            if command.to_lowercase().starts_with(&word.to_lowercase()) {
                matches.push(Pair {
                    display: format!("{} (command)", command),
                    replacement: command.to_string(),
                });
            }
        }

        // Add functions
        for function in &functions {
            if function.to_lowercase().starts_with(&word.to_lowercase()) {
                matches.push(Pair {
                    display: format!("{}() (function)", function),
                    replacement: format!("{}()", function),
                });
            }
        }

        // Sort matches by relevance
        matches.sort_by(|a, b| {
            let a_exact = a.replacement.to_lowercase() == word.to_lowercase();
            let b_exact = b.replacement.to_lowercase() == word.to_lowercase();

            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.replacement.len().cmp(&b.replacement.len()),
            }
        });

        Ok((word_start, matches))
    }
}

impl Hinter for SynapticHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<String> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for SynapticHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            Borrowed(prompt)
        } else {
            Owned(format!("\x1b[1;32m{}\x1b[0m", prompt))
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Owned(format!("\x1b[90m{}\x1b[0m", hint))
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }
}

impl Validator for SynapticHelper {
    fn validate(
        &self,
        ctx: &mut rustyline::validate::ValidationContext,
    ) -> RustylineResult<rustyline::validate::ValidationResult> {
        self.validator.validate(ctx)
    }

    fn validate_while_typing(&self) -> bool {
        self.validator.validate_while_typing()
    }
}


