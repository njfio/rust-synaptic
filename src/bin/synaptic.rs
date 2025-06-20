//! Synaptic CLI Binary
//!
//! This is the main entry point for the Synaptic command-line interface,
//! providing comprehensive memory management, graph querying, and system
//! administration capabilities.

use synaptic::cli::{SynapticCli, CliRunner};
use synaptic::error::Result;
use clap::Parser;
use tracing::{info, error};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    // Parse command line arguments
    let args = SynapticCli::parse();

    // Set up verbose logging if requested
    if args.verbose {
        info!("Verbose logging enabled");
    }

    // Create and run CLI
    match CliRunner::new(args.clone()).await {
        Ok(mut runner) => {
            info!("Starting Synaptic CLI");
            
            if let Err(e) = runner.run(args).await {
                error!("CLI execution failed: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to initialize CLI: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
