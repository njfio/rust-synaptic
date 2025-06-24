//! Synaptic CLI Module
//!
//! This module implements the command-line interface for the Synaptic memory system,
//! providing an interactive shell with SyQL query support, performance profiling,
//! and comprehensive memory exploration capabilities.

pub mod syql;
pub mod shell;
pub mod commands;
pub mod completion;
pub mod history;
pub mod config;
pub mod profiler;

use crate::error::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Synaptic CLI application
#[derive(Parser, Clone)]
#[command(name = "synaptic")]
#[command(about = "Synaptic AI Agent Memory System CLI")]
#[command(version = "0.1.0")]
#[command(long_about = "Interactive command-line interface for the Synaptic AI agent memory system with graph query language (SyQL) support")]
pub struct SynapticCli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    /// Disable colors in output
    #[arg(long)]
    pub no_color: bool,

    /// Enable interactive shell mode
    #[arg(short, long)]
    pub interactive: bool,

    /// Execute query and exit
    #[arg(short, long)]
    pub query: Option<String>,

    /// Database connection string
    #[arg(long)]
    pub database_url: Option<String>,

    /// Subcommands
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// CLI subcommands
#[derive(Subcommand, Clone)]
pub enum Commands {
    /// Start interactive shell
    Shell {
        /// Shell configuration
        #[arg(long)]
        history_file: Option<PathBuf>,
        
        /// Enable auto-completion
        #[arg(long, default_value_t = true)]
        completion: bool,
    },
    
    /// Execute SyQL query
    Query {
        /// SyQL query to execute
        query: String,
        
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Explain query execution plan
        #[arg(long)]
        explain: bool,
    },
    
    /// Memory management commands
    Memory {
        #[command(subcommand)]
        /// Memory management action to perform
        action: MemoryAction,
    },
    
    /// Graph operations
    Graph {
        #[command(subcommand)]
        /// Graph operation action to perform
        action: GraphAction,
    },
    
    /// Performance profiling
    Profile {
        /// Duration to profile (in seconds)
        #[arg(short, long, default_value_t = 30)]
        duration: u64,
        
        /// Output file for profile data
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Enable real-time monitoring
        #[arg(long)]
        realtime: bool,
    },
    
    /// Configuration management
    Config {
        #[command(subcommand)]
        /// Configuration action to perform
        action: ConfigAction,
    },
    
    /// Export data
    Export {
        /// Export format
        #[arg(short, long, value_enum, default_value_t = ExportFormat::Json)]
        format: ExportFormat,
        
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
        
        /// Filter expression
        #[arg(short, long)]
        filter: Option<String>,
    },
    
    /// Import data
    Import {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Import format
        #[arg(short, long, value_enum)]
        format: Option<ImportFormat>,
        
        /// Merge strategy
        #[arg(short, long, value_enum, default_value_t = MergeStrategy::Skip)]
        merge: MergeStrategy,
    },
    
    /// Show system information
    Info {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
}

/// Memory management actions
#[derive(Subcommand, Clone)]
pub enum MemoryAction {
    /// List memories
    List {
        /// Limit number of results
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        
        /// Filter by type
        #[arg(short, long)]
        memory_type: Option<String>,
    },
    
    /// Show memory details
    Show {
        /// Memory ID
        id: String,
    },
    
    /// Create new memory
    Create {
        /// Memory content
        content: String,
        
        /// Memory type
        #[arg(short, long, default_value = "text")]
        memory_type: String,
        
        /// Tags
        #[arg(short, long)]
        tags: Vec<String>,
    },
    
    /// Update memory
    Update {
        /// Memory ID
        id: String,
        
        /// New content
        #[arg(short, long)]
        content: Option<String>,
        
        /// New tags
        #[arg(short, long)]
        tags: Option<Vec<String>>,
    },
    
    /// Delete memory
    Delete {
        /// Memory ID
        id: String,
        
        /// Force deletion without confirmation
        #[arg(short, long)]
        force: bool,
    },
    
    /// Search memories
    Search {
        /// Search query
        query: String,
        
        /// Limit number of results
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
        
        /// Similarity threshold
        #[arg(short, long, default_value_t = 0.7)]
        threshold: f64,
    },
}

/// Graph operations
#[derive(Subcommand, Clone)]
pub enum GraphAction {
    /// Visualize graph
    Visualize {
        /// Output format
        #[arg(short, long, value_enum, default_value_t = GraphFormat::Ascii)]
        format: GraphFormat,
        
        /// Maximum depth
        #[arg(short, long, default_value_t = 3)]
        depth: usize,
        
        /// Starting node ID
        #[arg(short, long)]
        start: Option<String>,
    },
    
    /// Find paths between nodes
    Path {
        /// Source node ID
        from: String,
        
        /// Target node ID
        to: String,
        
        /// Maximum path length
        #[arg(short, long, default_value_t = 5)]
        max_length: usize,
        
        /// Path finding algorithm
        #[arg(short, long, value_enum, default_value_t = PathAlgorithm::Shortest)]
        algorithm: PathAlgorithm,
    },
    
    /// Analyze graph structure
    Analyze {
        /// Analysis type
        #[arg(short, long, value_enum, default_value_t = AnalysisType::Overview)]
        analysis_type: AnalysisType,
    },
    
    /// Export graph
    Export {
        /// Export format
        #[arg(short, long, value_enum, default_value_t = GraphExportFormat::Graphml)]
        format: GraphExportFormat,
        
        /// Output file
        #[arg(short, long)]
        output: PathBuf,
    },
}

/// Configuration actions
#[derive(Subcommand, Clone)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    
    /// Set configuration value
    Set {
        /// Configuration key
        key: String,
        
        /// Configuration value
        value: String,
    },
    
    /// Get configuration value
    Get {
        /// Configuration key
        key: String,
    },
    
    /// Reset configuration to defaults
    Reset {
        /// Force reset without confirmation
        #[arg(short, long)]
        force: bool,
    },
}

/// Output formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Table format output
    Table,
    /// JSON format output
    Json,
    /// CSV format output
    Csv,
    /// YAML format output
    Yaml,
    /// Graph format output
    Graph,
    Tree,
}

/// Export formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ExportFormat {
    Json,
    Csv,
    Yaml,
    Xml,
    Parquet,
}

/// Import formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ImportFormat {
    Json,
    Csv,
    Yaml,
    Xml,
}

/// Merge strategies for imports
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum MergeStrategy {
    Skip,
    Overwrite,
    Merge,
    Fail,
}

/// Graph visualization formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum GraphFormat {
    Ascii,
    Dot,
    Svg,
    Png,
}

impl std::fmt::Display for GraphFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphFormat::Ascii => write!(f, "ascii"),
            GraphFormat::Dot => write!(f, "dot"),
            GraphFormat::Svg => write!(f, "svg"),
            GraphFormat::Png => write!(f, "png"),
        }
    }
}

/// Path finding algorithms
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PathAlgorithm {
    Shortest,
    All,
    Dijkstra,
    AStar,
}

impl std::fmt::Display for PathAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathAlgorithm::Shortest => write!(f, "shortest"),
            PathAlgorithm::All => write!(f, "all"),
            PathAlgorithm::Dijkstra => write!(f, "dijkstra"),
            PathAlgorithm::AStar => write!(f, "astar"),
        }
    }
}

/// Graph analysis types
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum AnalysisType {
    Overview,
    Centrality,
    Clustering,
    Components,
    Metrics,
}

impl std::fmt::Display for AnalysisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisType::Overview => write!(f, "overview"),
            AnalysisType::Centrality => write!(f, "centrality"),
            AnalysisType::Clustering => write!(f, "clustering"),
            AnalysisType::Components => write!(f, "components"),
            AnalysisType::Metrics => write!(f, "metrics"),
        }
    }
}

/// Graph export formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum GraphExportFormat {
    Graphml,
    Gexf,
    Json,
    Csv,
}

impl std::fmt::Display for GraphExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphExportFormat::Graphml => write!(f, "graphml"),
            GraphExportFormat::Gexf => write!(f, "gexf"),
            GraphExportFormat::Json => write!(f, "json"),
            GraphExportFormat::Csv => write!(f, "csv"),
        }
    }
}

/// CLI application runner
pub struct CliRunner {
    /// CLI configuration
    config: config::CliConfig,
    /// SyQL engine
    syql_engine: syql::SyQLEngine,
    /// Agent memory system
    agent_memory: crate::AgentMemory,
}

impl CliRunner {
    /// Create a new CLI runner
    pub async fn new(args: SynapticCli) -> Result<Self> {
        let config = config::CliConfig::load(args.config.as_ref().map(|v| &**v)).await?;
        let syql_engine = syql::SyQLEngine::new()?;

        // Initialize AgentMemory with default configuration
        let memory_config = crate::MemoryConfig {
            enable_knowledge_graph: true,
            enable_temporal_tracking: true,
            enable_advanced_management: true,
            ..Default::default()
        };
        let agent_memory = crate::AgentMemory::new(memory_config).await?;

        Ok(Self {
            config,
            syql_engine,
            agent_memory,
        })
    }

    /// Run the CLI application
    pub async fn run(&mut self, args: SynapticCli) -> Result<()> {
        match args.command {
            Some(Commands::Shell { history_file, completion }) => {
                let mut shell = shell::InteractiveShell::new(
                    &mut self.syql_engine,
                    &self.config,
                    history_file,
                    completion,
                ).await?;
                shell.run().await
            },
            Some(Commands::Query { query, output, explain }) => {
                self.execute_query(&query, output.as_ref(), explain).await
            },
            Some(Commands::Memory { action }) => {
                self.handle_memory_action(action).await
            },
            Some(Commands::Graph { action }) => {
                self.handle_graph_action(action).await
            },
            Some(Commands::Profile { duration, output, realtime }) => {
                self.run_profiler(duration, output.as_ref(), realtime).await
            },
            Some(Commands::Config { action }) => {
                self.handle_config_action(action).await
            },
            Some(Commands::Export { format, output, filter }) => {
                self.export_data(format, &output, filter.as_ref()).await
            },
            Some(Commands::Import { input, format, merge }) => {
                self.import_data(&input, format, merge).await
            },
            Some(Commands::Info { detailed }) => {
                self.show_info(detailed).await
            },
            None => {
                if args.interactive || args.query.is_some() {
                    if let Some(query) = args.query {
                        self.execute_query(&query, None, false).await
                    } else {
                        // Start interactive shell
                        let mut shell = shell::InteractiveShell::new(
                            &mut self.syql_engine,
                            &self.config,
                            None,
                            true,
                        ).await?;
                        shell.run().await
                    }
                } else {
                    // Show help
                    println!("Use --help for usage information or --interactive to start the shell");
                    Ok(())
                }
            }
        }
    }

    /// Execute a SyQL query
    async fn execute_query(&mut self, query: &str, output_file: Option<&PathBuf>, explain: bool) -> Result<()> {
        commands::SyQLCommands::execute(&mut self.syql_engine, query, output_file.map(|p| p.as_path()), explain).await
    }



    /// Handle memory actions
    async fn handle_memory_action(&mut self, action: MemoryAction) -> Result<()> {
        match action {
            MemoryAction::List { limit, memory_type } => {
                commands::MemoryCommands::list(&mut self.agent_memory, limit, memory_type).await
            },
            MemoryAction::Show { id } => {
                commands::MemoryCommands::show(&mut self.agent_memory, &id).await
            },
            MemoryAction::Create { content, memory_type, tags } => {
                commands::MemoryCommands::create(&mut self.agent_memory, &content, &memory_type, &tags).await
            },
            MemoryAction::Update { id, content, tags } => {
                commands::MemoryCommands::update(&mut self.agent_memory, &id, content.as_deref(), tags.as_deref()).await
            },
            MemoryAction::Delete { id, force: _ } => {
                commands::MemoryCommands::delete(&mut self.agent_memory, &id).await
            },
            MemoryAction::Search { query, limit, threshold: _ } => {
                commands::MemoryCommands::search(&mut self.agent_memory, &query, limit).await
            },
        }
    }

    /// Handle graph actions
    async fn handle_graph_action(&mut self, action: GraphAction) -> Result<()> {
        match action {
            GraphAction::Visualize { format, depth, start } => {
                commands::GraphCommands::visualize(&mut self.agent_memory, &format.to_string(), depth, start.as_deref()).await
            },
            GraphAction::Path { from, to, max_length, algorithm } => {
                commands::GraphCommands::find_path(&mut self.agent_memory, &from, &to, max_length, &algorithm.to_string()).await
            },
            GraphAction::Analyze { analysis_type } => {
                commands::GraphCommands::analyze(&mut self.agent_memory, &analysis_type.to_string()).await
            },
            GraphAction::Export { format, output } => {
                commands::GraphCommands::export(&mut self.agent_memory, &format.to_string(), &output).await
            },
        }
    }

    /// Run performance profiler
    async fn run_profiler(&self, duration: u64, output: Option<&PathBuf>, realtime: bool) -> Result<()> {
        commands::ProfilerCommands::run_profiler(duration, output.map(|v| &**v), realtime).await
    }

    /// Handle configuration actions
    async fn handle_config_action(&self, action: ConfigAction) -> Result<()> {
        match action {
            ConfigAction::Show => {
                commands::ConfigCommands::show(&self.config).await
            },
            ConfigAction::Set { key, value } => {
                commands::ConfigCommands::set(&key, &value).await
            },
            ConfigAction::Get { key } => {
                commands::ConfigCommands::get(&self.config, &key).await
            },
            ConfigAction::Reset { force } => {
                commands::ConfigCommands::reset(force).await
            },
        }
    }

    /// Export data
    async fn export_data(&self, format: ExportFormat, output: &PathBuf, filter: Option<&String>) -> Result<()> {
        let format_str = match format {
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
            ExportFormat::Yaml => "yaml",
            ExportFormat::Xml => "xml",
            ExportFormat::Parquet => "parquet",
        };
        commands::DataCommands::export(&**self.agent_memory.storage(), format_str, output, filter.map(|s| s.as_str())).await
    }

    /// Import data
    async fn import_data(&self, input: &PathBuf, format: Option<ImportFormat>, merge: MergeStrategy) -> Result<()> {
        let format_str = format.map(|f| match f {
            ImportFormat::Json => "json",
            ImportFormat::Csv => "csv",
            ImportFormat::Yaml => "yaml",
            ImportFormat::Xml => "xml",
        });
        let merge_str = match merge {
            MergeStrategy::Skip => "skip",
            MergeStrategy::Overwrite => "overwrite",
            MergeStrategy::Merge => "merge",
            MergeStrategy::Fail => "fail",
        };
        commands::DataCommands::import(&**self.agent_memory.storage(), input, format_str, merge_str).await
    }

    /// Show system information (placeholder)
    async fn show_info(&self, _detailed: bool) -> Result<()> {
        println!("System info not yet implemented");
        Ok(())
    }
}
