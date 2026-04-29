//! Hidden subcommand that just runs the ggml RPC server.
//!
//! The worker spawns `llama-mesh rpc-server` as a child process so it can
//! kill and restart it cleanly for GPU preemption — ggml_backend_rpc_start_server
//! blocks forever and has no shutdown API.

use anyhow::Result;
use tracing::info;

use crate::ffi;

#[derive(clap::Args)]
pub struct Args {
    /// RPC server endpoint (host:port)
    #[arg(long)]
    endpoint: String,
}

/// Enumerate GPU devices and run the RPC server. **Blocks forever.**
pub fn run(args: Args) -> Result<()> {
    let devices = ffi::enumerate_devices();
    let gpu_indices: Vec<usize> = devices.iter().filter(|d| d.is_gpu()).map(|d| d.index).collect();

    if gpu_indices.is_empty() {
        anyhow::bail!("no GPU devices found");
    }

    for &i in &gpu_indices {
        let d = &devices[i];
        info!(
            "serving device {}: {} — {} ({}MB)",
            d.index, d.name, d.description, d.vram_total_mb,
        );
    }

    info!("RPC server starting on {}", args.endpoint);

    // This never returns
    ffi::run_rpc_server(&args.endpoint, &gpu_indices);

    Ok(())
}
