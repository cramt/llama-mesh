mod coord;
mod ffi;
mod protocol;
mod rpc_server;
mod worker;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llama-mesh")]
#[command(about = "Dynamic distributed LLM inference over llama-cpp RPC")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as a worker node — exposes local GPU(s) over RPC and registers with coordinator
    Worker(worker::Args),
    /// Run as the coordinator — accepts workers, manages topology, drives llama-swap
    Coord(coord::Args),
    /// (internal) Run the RPC server directly — used by the worker as a child process
    #[command(hide = true)]
    RpcServer(rpc_server::Args),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Worker(args) => worker::run(args).await,
        Commands::Coord(args) => coord::run(args).await,
        Commands::RpcServer(args) => rpc_server::run(args),
    }
}
