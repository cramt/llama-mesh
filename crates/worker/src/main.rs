use anyhow::{Context, Result};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use llama_mesh_protocol::{CoordinatorMsg, GpuType, WorkerAnnounce, WorkerMsg};
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "llama-mesh-worker")]
#[command(about = "Worker node for llama-mesh distributed inference")]
struct Args {
    /// WebSocket URL of the coordinator (e.g. ws://192.168.178.24:50050)
    #[arg(long)]
    coordinator: String,

    /// GPU type: cuda, rocm, or cpu
    #[arg(long)]
    gpu: String,

    /// Port for llama-rpc-server to listen on
    #[arg(long, default_value_t = 50052)]
    rpc_port: u16,

    /// Bind address for llama-rpc-server
    #[arg(long, default_value = "0.0.0.0")]
    rpc_host: String,

    /// Path to llama-rpc-server binary
    #[arg(long)]
    rpc_bin: PathBuf,

    /// Process names that trigger GPU preemption (comma-separated)
    #[arg(long, value_delimiter = ',')]
    preempt_triggers: Vec<String>,

    /// ROCm GFX version override (HSA_OVERRIDE_GFX_VERSION)
    #[arg(long)]
    rocm_version: Option<String>,

    /// GPU device visibility (CUDA_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES)
    #[arg(long)]
    visible_devices: Option<String>,

    /// Total GPU VRAM in MB
    #[arg(long)]
    vram_mb: u64,

    /// Node identifier (defaults to hostname)
    #[arg(long)]
    node_id: Option<String>,

    /// Extra arguments passed to llama-rpc-server
    #[arg(last = true)]
    extra_rpc_args: Vec<String>,
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

fn start_rpc_server(args: &Args, gpu: &GpuType) -> Result<Child> {
    let mut cmd = Command::new(&args.rpc_bin);
    cmd.arg("-H")
        .arg(&args.rpc_host)
        .arg("-p")
        .arg(args.rpc_port.to_string());

    match gpu {
        GpuType::Cuda => {
            if let Some(ref devices) = args.visible_devices {
                cmd.env("CUDA_VISIBLE_DEVICES", devices);
            }
        }
        GpuType::Rocm => {
            if let Some(ref devices) = args.visible_devices {
                cmd.env("ROCR_VISIBLE_DEVICES", devices);
            }
            if let Some(ref version) = args.rocm_version {
                cmd.env("HSA_OVERRIDE_GFX_VERSION", version);
            }
        }
        GpuType::Cpu => {}
    }

    cmd.args(&args.extra_rpc_args);

    let child = cmd
        .kill_on_drop(true)
        .spawn()
        .context("failed to start llama-rpc-server")?;

    info!(
        "started llama-rpc-server on {}:{}",
        args.rpc_host, args.rpc_port
    );
    Ok(child)
}

async fn run_session(args: &Args, gpu: &GpuType, node_id: &str) -> Result<()> {
    // Start the RPC server
    let mut rpc_child = start_rpc_server(args, gpu)?;

    // Give the RPC server a moment to bind its port
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Connect to coordinator
    info!("connecting to coordinator at {}", args.coordinator);
    let (ws_stream, _) = tokio_tungstenite::connect_async(&args.coordinator)
        .await
        .context("failed to connect to coordinator")?;

    let (mut sink, mut stream) = ws_stream.split();

    // Announce ourselves
    let announce = WorkerMsg::Announce(WorkerAnnounce {
        node_id: node_id.to_string(),
        gpu: gpu.clone(),
        vram_mb: args.vram_mb,
        rpc_port: args.rpc_port,
        preemptible: !args.preempt_triggers.is_empty(),
    });
    sink.send(Message::Text(serde_json::to_string(&announce)?))
        .await
        .context("failed to send announce")?;

    info!("announced to coordinator as {node_id}");

    // Preemption watcher channel
    let (preempt_tx, mut preempt_rx) = mpsc::channel::<String>(1);

    let preempt_handle = if !args.preempt_triggers.is_empty() {
        let triggers = args.preempt_triggers.clone();
        let tx = preempt_tx;
        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                for trigger in &triggers {
                    if is_process_running(trigger) {
                        let _ = tx.send(trigger.clone()).await;
                        return;
                    }
                }
            }
        }))
    } else {
        drop(preempt_tx);
        None
    };

    // Main event loop — hold the connection open
    loop {
        tokio::select! {
            msg = stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<CoordinatorMsg>(&text) {
                            Ok(coord_msg) => info!("coordinator: {coord_msg:?}"),
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
                    info!("preemption triggered by {process_name}");

                    // Tell coordinator we're leaving
                    let drain_msg = WorkerMsg::Draining {
                        reason: format!("{process_name} launched"),
                    };
                    let _ = sink
                        .send(Message::Text(serde_json::to_string(&drain_msg).unwrap()))
                        .await;

                    // Free the GPU
                    info!("stopping llama-rpc-server for preemption");
                    rpc_child.kill().await.ok();
                    rpc_child.wait().await.ok();

                    // Wait for the game to exit
                    info!("waiting for {process_name} to exit...");
                    loop {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        if !is_process_running(&process_name) {
                            break;
                        }
                    }
                    info!("{process_name} exited, will reconnect");
                    break;
                }
            }

            status = rpc_child.wait() => {
                match status {
                    Ok(exit) => warn!("llama-rpc-server exited unexpectedly: {exit}"),
                    Err(e) => warn!("llama-rpc-server error: {e}"),
                }
                break;
            }
        }
    }

    // Cleanup
    if let Some(handle) = preempt_handle {
        handle.abort();
    }
    rpc_child.kill().await.ok();
    rpc_child.wait().await.ok();

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let gpu = parse_gpu(&args.gpu)?;
    let node_id = args.node_id.clone().unwrap_or_else(get_hostname);

    info!("llama-mesh worker starting as {node_id} ({gpu})");

    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(30);

    loop {
        match run_session(&args, &gpu, &node_id).await {
            Ok(()) => {
                // Clean session end (preemption cycle or coordinator disconnect)
                backoff = Duration::from_secs(1);
            }
            Err(e) => {
                error!("session error: {e:#}");
            }
        }

        info!("reconnecting in {}s...", backoff.as_secs());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}
