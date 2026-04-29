use anyhow::{Context, Result};
use clap;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use crate::ffi;
use crate::protocol::{CoordinatorMsg, GpuType, WorkerAnnounce, WorkerMsg};

#[derive(clap::Args)]
pub struct Args {
    /// WebSocket URL of the coordinator (e.g. ws://192.168.178.24:50050)
    #[arg(long)]
    coordinator: String,

    /// GPU type reported to coordinator: cuda, rocm, or cpu
    #[arg(long)]
    gpu: String,

    /// Port for the RPC server
    #[arg(long, default_value_t = 50052)]
    rpc_port: u16,

    /// Bind address for the RPC server
    #[arg(long, default_value = "0.0.0.0")]
    rpc_host: String,

    /// Process names that trigger GPU preemption (comma-separated)
    #[arg(long, value_delimiter = ',')]
    preempt_triggers: Vec<String>,

    /// Node identifier (defaults to hostname)
    #[arg(long)]
    node_id: Option<String>,
}

fn parse_gpu(s: &str) -> Result<GpuType> {
    match s {
        "cuda" => Ok(GpuType::Cuda),
        "rocm" => Ok(GpuType::Rocm),
        "cpu" => Ok(GpuType::Cpu),
        other => anyhow::bail!("unknown GPU type: {other} (expected cuda, rocm, or cpu)"),
    }
}

fn get_hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .unwrap_or_else(|_| "unknown".into())
        .trim()
        .to_string()
}

fn is_process_running(name: &str) -> bool {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return false;
    };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|s| s.chars().all(|c| c.is_ascii_digit()))
        })
        .any(|e| {
            std::fs::read_to_string(e.path().join("comm"))
                .is_ok_and(|comm| comm.trim() == name)
        })
}

fn any_trigger_running(triggers: &[String]) -> Option<String> {
    triggers
        .iter()
        .find(|t| is_process_running(t))
        .cloned()
}

/// Block until no preempt-trigger processes are running.
async fn wait_for_preempt_clear(triggers: &[String]) {
    loop {
        match any_trigger_running(triggers) {
            Some(name) => {
                info!("waiting for {name} to exit before starting RPC server...");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            None => return,
        }
    }
}

/// Run a single coordinator session. Returns cleanly on disconnect.
/// Calls `std::process::exit(0)` on preemption — the RPC server thread
/// can only be stopped by terminating the process.
async fn run_session(args: &Args, node_id: &str, vram_mb: u64, gpu: &GpuType) -> Result<()> {
    info!("connecting to coordinator at {}", args.coordinator);
    let (ws, _) = tokio_tungstenite::connect_async(&args.coordinator)
        .await
        .context("failed to connect to coordinator")?;

    let (mut sink, mut stream) = ws.split();

    // Announce
    let announce = WorkerMsg::Announce(WorkerAnnounce {
        node_id: node_id.to_string(),
        gpu: gpu.clone(),
        vram_mb,
        rpc_port: args.rpc_port,
        preemptible: !args.preempt_triggers.is_empty(),
    });
    sink.send(Message::Text(serde_json::to_string(&announce)?))
        .await
        .context("failed to send announce")?;

    info!("announced to coordinator as {node_id}");

    // Preemption watcher
    let (preempt_tx, mut preempt_rx) = mpsc::channel::<String>(1);
    let preempt_handle = if !args.preempt_triggers.is_empty() {
        let triggers = args.preempt_triggers.clone();
        let tx = preempt_tx;
        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if let Some(name) = any_trigger_running(&triggers) {
                    let _ = tx.send(name).await;
                    return;
                }
            }
        }))
    } else {
        drop(preempt_tx);
        None
    };

    loop {
        tokio::select! {
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<CoordinatorMsg>(&text) {
                            Ok(m) => info!("coordinator: {m:?}"),
                            Err(e) => warn!("bad coordinator message: {e}"),
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sink.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        info!("coordinator disconnected");
                        break;
                    }
                    Some(Err(e)) => {
                        warn!("websocket error: {e}");
                        break;
                    }
                    _ => {}
                }
            }

            trigger = preempt_rx.recv() => {
                if let Some(process_name) = trigger {
                    info!("preemption triggered by {process_name} — exiting for GPU release");
                    let drain = WorkerMsg::Draining {
                        reason: format!("{process_name} launched"),
                    };
                    let _ = sink
                        .send(Message::Text(serde_json::to_string(&drain).unwrap()))
                        .await;
                    // Brief pause so the message actually sends
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    // Exit the process — the RPC server thread cannot be stopped gracefully.
                    // systemd restarts us; we'll wait for the game to exit before re-joining.
                    std::process::exit(0);
                }
            }
        }
    }

    if let Some(h) = preempt_handle {
        h.abort();
    }
    Ok(())
}

pub async fn run(args: Args) -> Result<()> {
    let gpu = parse_gpu(&args.gpu)?;
    let node_id = args.node_id.clone().unwrap_or_else(get_hostname);

    info!("llama-mesh worker starting as {node_id} ({gpu})");

    // Wait for any preempt trigger to clear (e.g. game still running from last preemption)
    if !args.preempt_triggers.is_empty() {
        wait_for_preempt_clear(&args.preempt_triggers).await;
    }

    // Enumerate GPU devices
    let devices = ffi::enumerate_devices();
    for d in &devices {
        info!(
            "  device {}: {} — {} ({}MB free / {}MB total, {})",
            d.index,
            d.name,
            d.description,
            d.vram_free_mb,
            d.vram_total_mb,
            if d.is_gpu() { "GPU" } else { "CPU" },
        );
    }

    let gpu_indices: Vec<usize> = devices.iter().filter(|d| d.is_gpu()).map(|d| d.index).collect();
    if gpu_indices.is_empty() {
        anyhow::bail!("no GPU devices found — nothing to serve over RPC");
    }

    let total_vram_mb: u64 = devices
        .iter()
        .filter(|d| d.is_gpu())
        .map(|d| d.vram_total_mb)
        .sum();

    // Start the RPC server in a dedicated thread (blocks forever)
    let endpoint = format!("{}:{}", args.rpc_host, args.rpc_port);
    let indices = gpu_indices.clone();
    std::thread::spawn(move || {
        ffi::run_rpc_server(&endpoint, &indices);
    });

    info!(
        "RPC server started on {}:{} ({} GPU device(s), {}MB total VRAM)",
        args.rpc_host,
        args.rpc_port,
        gpu_indices.len(),
        total_vram_mb,
    );

    // Give the RPC server a moment to bind
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Reconnect loop — RPC server stays running across coordinator reconnects
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        match run_session(&args, &node_id, total_vram_mb, &gpu).await {
            Ok(()) => backoff = Duration::from_secs(1),
            Err(e) => error!("session error: {e:#}"),
        }

        info!("reconnecting in {}s...", backoff.as_secs());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}
