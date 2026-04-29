use anyhow::{Context, Result};
use clap;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::process::{Child, Command};
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

/// Spawn `llama-mesh rpc-server` as a child process.
fn spawn_rpc_child(rpc_host: &str, rpc_port: u16) -> Result<Child> {
    let self_exe = std::env::current_exe().context("failed to find own binary path")?;
    let endpoint = format!("{rpc_host}:{rpc_port}");

    let child = Command::new(self_exe)
        .arg("rpc-server")
        .arg("--endpoint")
        .arg(&endpoint)
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn rpc-server child process")?;

    info!("spawned rpc-server child on {endpoint}");
    Ok(child)
}

enum SessionEnd {
    /// Coordinator disconnected — reconnect, RPC server stays up
    Disconnected,
    /// A preempt trigger was detected — caller should kill RPC and wait
    Preempted(String),
}

async fn run_session(
    args: &Args,
    node_id: &str,
    vram_mb: u64,
    gpu: &GpuType,
) -> Result<SessionEnd> {
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

    let result = loop {
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
                        break SessionEnd::Disconnected;
                    }
                    Some(Err(e)) => {
                        warn!("websocket error: {e}");
                        break SessionEnd::Disconnected;
                    }
                    _ => {}
                }
            }

            trigger = preempt_rx.recv() => {
                if let Some(process_name) = trigger {
                    info!("preemption triggered by {process_name}");
                    let drain = WorkerMsg::Draining {
                        reason: format!("{process_name} launched"),
                    };
                    let _ = sink
                        .send(Message::Text(serde_json::to_string(&drain).unwrap()))
                        .await;
                    break SessionEnd::Preempted(process_name);
                }
            }
        }
    };

    if let Some(h) = preempt_handle {
        h.abort();
    }
    Ok(result)
}

pub async fn run(args: Args) -> Result<()> {
    let gpu = parse_gpu(&args.gpu)?;
    let node_id = args.node_id.clone().unwrap_or_else(get_hostname);

    info!("llama-mesh worker starting as {node_id} ({gpu})");

    // Auto-detect GPU devices and VRAM
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

    let total_vram_mb: u64 = devices
        .iter()
        .filter(|d| d.is_gpu())
        .map(|d| d.vram_total_mb)
        .sum();

    if total_vram_mb == 0 {
        anyhow::bail!("no GPU devices found — nothing to serve over RPC");
    }

    info!(
        "{} GPU device(s), {}MB total VRAM",
        devices.iter().filter(|d| d.is_gpu()).count(),
        total_vram_mb,
    );

    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        // Wait for any game to exit before starting the RPC server
        if !args.preempt_triggers.is_empty() {
            wait_for_preempt_clear(&args.preempt_triggers).await;
        }

        // Start RPC server as a child process (so we can kill it for preemption)
        let mut rpc_child = spawn_rpc_child(&args.rpc_host, args.rpc_port)?;

        // Give the RPC server a moment to bind
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Reconnect loop — RPC server stays running across coordinator reconnects
        loop {
            match run_session(&args, &node_id, total_vram_mb, &gpu).await {
                Ok(SessionEnd::Disconnected) => {
                    backoff = Duration::from_secs(1);
                    info!("reconnecting in {}s...", backoff.as_secs());
                    tokio::time::sleep(backoff).await;
                    continue; // RPC server still running, just reconnect
                }
                Ok(SessionEnd::Preempted(process_name)) => {
                    // Kill the RPC server to free the GPU
                    info!("killing RPC server for preemption");
                    rpc_child.kill().await.ok();
                    rpc_child.wait().await.ok();

                    info!("waiting for {process_name} to exit...");
                    loop {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        if !is_process_running(&process_name) {
                            break;
                        }
                    }
                    info!("{process_name} exited, restarting RPC server");

                    backoff = Duration::from_secs(1);
                    break; // break inner loop → outer loop restarts RPC + reconnects
                }
                Err(e) => {
                    error!("session error: {e:#}");
                    info!("reconnecting in {}s...", backoff.as_secs());
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(max_backoff);
                    continue;
                }
            }
        }
    }
}
