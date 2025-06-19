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
        action: MemoryAction,
    },
    
    /// Graph operations
    Graph {
        #[command(subcommand)]
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
    Table,
    Json,
    Csv,
    Yaml,
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

/// Path finding algorithms
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PathAlgorithm {
    Shortest,
    All,
    Dijkstra,
    AStar,
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

/// Graph export formats
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum GraphExportFormat {
    Graphml,
    Gexf,
    Json,
    Csv,
}

/// CLI application runner
pub struct CliRunner {
    /// CLI configuration
    config: config::CliConfig,
    /// SyQL engine
    syql_engine: syql::SyQLEngine,
}

impl CliRunner {
    /// Create a new CLI runner
    pub async fn new(args: SynapticCli) -> Result<Self> {
        let config = config::CliConfig::load(args.config.as_ref().map(|v| &**v)).await?;
        let syql_engine = syql::SyQLEngine::new()?;
        
        Ok(Self {
            config,
            syql_engine,
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
        if explain {
            let plan = self.syql_engine.explain_query(query).await?;
            println!("Query Execution Plan:");
            println!("{:#?}", plan);
        } else {
            let result = self.syql_engine.execute_query(query).await?;
            let formatted = self.syql_engine.format_result(&result, syql::OutputFormat::Table)?;
            
            if let Some(output_path) = output_file {
                std::fs::write(output_path, formatted)?;
                println!("Results written to {}", output_path.display());
            } else {
                println!("{}", formatted);
            }
        }
        Ok(())
    }

    /// Handle memory actions (placeholder)
    async fn handle_memory_action(&self, _action: MemoryAction) -> Result<()> {
        println!("Memory action not yet implemented");
        Ok(())
    }

    /// Handle graph actions (placeholder)
    async fn handle_graph_action(&self, _action: GraphAction) -> Result<()> {
        println!("Graph action not yet implemented");
        Ok(())
    }

    /// Run performance profiler (placeholder)
    async fn run_profiler(&self, _duration: u64, _output: Option<&PathBuf>, _realtime: bool) -> Result<()> {
        println!("Performance profiler not yet implemented");
        Ok(())
    }

    /// Handle configuration actions (placeholder)
    async fn handle_config_action(&self, _action: ConfigAction) -> Result<()> {
        println!("Config action not yet implemented");
        Ok(())
    }

    /// Export data (placeholder)
    async fn export_data(&self, _format: ExportFormat, _output: &PathBuf, _filter: Option<&String>) -> Result<()> {
        println!("Data export not yet implemented");
        Ok(())
    }

    /// Import data (placeholder)
    async fn import_data(&self, _input: &PathBuf, _format: Option<ImportFormat>, _merge: MergeStrategy) -> Result<()> {
        println!("Data import not yet implemented");
        Ok(())
    }

    /// Show system information (placeholder)
    async fn show_info(&self, _detailed: bool) -> Result<()> {
        println!("System info not yet implemented");
        Ok(())
    }
}
