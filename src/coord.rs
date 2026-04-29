use anyhow::{Context, Result};
use clap;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use crate::protocol::{CoordinatorMsg, WorkerAnnounce, WorkerMsg};
use crate::tunnel;

static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(clap::Args)]
pub struct Args {
    /// Address to listen for worker WebSocket connections
    #[arg(long, default_value = "0.0.0.0:50050")]
    listen: String,

    /// Path to coordinator config file (TOML)
    #[arg(long)]
    config: PathBuf,
}

// ---------------------------------------------------------------------------
// Config (TOML)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Config {
    swap_bin: PathBuf,
    swap_listen: String,
    llama_server_bin: PathBuf,
    swap_config_path: PathBuf,
    #[serde(default)]
    local_vram_mb: u64,
    #[serde(default)]
    models: Vec<ModelConfig>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    name: String,
    path: PathBuf,
    #[serde(default = "default_model_args")]
    args: Vec<String>,
    #[serde(default = "default_ttl")]
    ttl: u64,
}

fn default_model_args() -> Vec<String> {
    vec![
        "-ngl".into(),
        "999".into(),
        "-c".into(),
        "16384".into(),
        "--flash-attn".into(),
        "on".into(),
    ]
}

fn default_ttl() -> u64 {
    300
}

// ---------------------------------------------------------------------------
// llama-swap YAML
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SwapConfig {
    health_check_timeout: u64,
    log_level: String,
    models: HashMap<String, SwapModelEntry>,
}

#[derive(Serialize)]
struct SwapModelEntry {
    cmd: String,
    ttl: u64,
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

struct TopologyPlan {
    rpc_endpoints: Vec<String>,
    tensor_split: Option<String>,
}

struct ConnectedWorker {
    conn_id: u64,
    announce: WorkerAnnounce,
    proxy_port: u16,
    draining: bool,
}

fn plan_topology(local_vram_mb: u64, workers: &HashMap<String, ConnectedWorker>) -> TopologyPlan {
    let mut ready: Vec<&ConnectedWorker> = workers.values().filter(|w| !w.draining).collect();

    ready.sort_by(|a, b| {
        a.announce
            .preemptible
            .cmp(&b.announce.preemptible)
            .then_with(|| b.announce.vram_mb.cmp(&a.announce.vram_mb))
    });

    if ready.is_empty() {
        return TopologyPlan {
            rpc_endpoints: vec![],
            tensor_split: None,
        };
    }

    // RPC traffic tunnels through WebSocket — llama-server connects to local proxy ports
    let rpc_endpoints: Vec<String> = ready
        .iter()
        .map(|w| format!("127.0.0.1:{}", w.proxy_port))
        .collect();

    let mut splits = Vec::with_capacity(ready.len() + 1);
    if local_vram_mb > 0 {
        splits.push(local_vram_mb.to_string());
    }
    for w in &ready {
        splits.push(w.announce.vram_mb.to_string());
    }

    TopologyPlan {
        rpc_endpoints,
        tensor_split: Some(splits.join(",")),
    }
}

fn generate_swap_config(config: &Config, plan: &TopologyPlan) -> SwapConfig {
    let mut models = HashMap::new();

    for model in &config.models {
        let mut cmd_parts: Vec<String> = vec![
            config.llama_server_bin.display().to_string(),
            "--model".into(),
            model.path.display().to_string(),
            "--port".into(),
            "${PORT}".into(),
            "--host".into(),
            "127.0.0.1".into(),
        ];

        if !plan.rpc_endpoints.is_empty() {
            cmd_parts.push("--rpc".into());
            cmd_parts.push(plan.rpc_endpoints.join(","));
        }

        if let Some(ref ts) = plan.tensor_split {
            cmd_parts.push("--tensor-split".into());
            cmd_parts.push(ts.clone());
        }

        cmd_parts.extend(model.args.iter().cloned());

        models.insert(
            model.name.clone(),
            SwapModelEntry {
                cmd: cmd_parts.join(" "),
                ttl: model.ttl,
            },
        );
    }

    SwapConfig {
        health_check_timeout: 600,
        log_level: "info".into(),
        models,
    }
}

// ---------------------------------------------------------------------------
// Swap process management
// ---------------------------------------------------------------------------

async fn write_and_reload_swap(
    config: &Config,
    plan: &TopologyPlan,
    swap_child: &mut Option<Child>,
) -> Result<()> {
    let swap_config = generate_swap_config(config, plan);
    let yaml = serde_yaml::to_string(&swap_config)?;

    if let Some(parent) = config.swap_config_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }

    tokio::fs::write(&config.swap_config_path, &yaml)
        .await
        .context("failed to write swap config")?;

    info!(
        "topology: {} worker(s), rpc=[{}]",
        plan.rpc_endpoints.len(),
        plan.rpc_endpoints.join(", ")
    );

    if let Some(ref mut child) = swap_child {
        info!("restarting llama-swap for topology change");
        child.kill().await.ok();
        child.wait().await.ok();
    }

    let child = Command::new(&config.swap_bin)
        .arg("-config")
        .arg(&config.swap_config_path)
        .arg("-listen")
        .arg(&config.swap_listen)
        .kill_on_drop(true)
        .spawn()
        .context("failed to start llama-swap")?;

    info!(
        "llama-swap listening on {} (config: {})",
        config.swap_listen,
        config.swap_config_path.display()
    );
    *swap_child = Some(child);

    Ok(())
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

enum CoordEvent {
    Joined {
        conn_id: u64,
        node_id: String,
        announce: WorkerAnnounce,
        proxy_port: u16,
    },
    Left {
        conn_id: u64,
        node_id: String,
    },
    Draining {
        conn_id: u64,
        node_id: String,
    },
    Resuming {
        conn_id: u64,
        node_id: String,
    },
}

fn apply_event(workers: &mut HashMap<String, ConnectedWorker>, event: CoordEvent) -> bool {
    match event {
        CoordEvent::Joined {
            conn_id,
            node_id,
            announce,
            proxy_port,
        } => {
            workers.insert(
                node_id,
                ConnectedWorker {
                    conn_id,
                    announce,
                    proxy_port,
                    draining: false,
                },
            );
            true
        }
        CoordEvent::Left { conn_id, node_id } => {
            if workers.get(&node_id).is_some_and(|w| w.conn_id == conn_id) {
                workers.remove(&node_id);
                true
            } else {
                false
            }
        }
        CoordEvent::Draining { conn_id, node_id } => {
            if let Some(w) = workers.get_mut(&node_id) {
                if w.conn_id == conn_id && !w.draining {
                    w.draining = true;
                    return true;
                }
            }
            false
        }
        CoordEvent::Resuming { conn_id, node_id } => {
            if let Some(w) = workers.get_mut(&node_id) {
                if w.conn_id == conn_id && w.draining {
                    w.draining = false;
                    return true;
                }
            }
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinator task
// ---------------------------------------------------------------------------

async fn coordinator_task(config: Config, mut rx: mpsc::Receiver<CoordEvent>) -> Result<()> {
    let mut workers: HashMap<String, ConnectedWorker> = HashMap::new();
    let mut swap_child: Option<Child> = None;

    let plan = plan_topology(config.local_vram_mb, &workers);
    write_and_reload_swap(&config, &plan, &mut swap_child).await?;

    while let Some(event) = rx.recv().await {
        let changed = apply_event(&mut workers, event);

        if changed {
            // Debounce: batch events arriving close together
            tokio::time::sleep(Duration::from_millis(500)).await;
            while let Ok(extra) = rx.try_recv() {
                apply_event(&mut workers, extra);
            }

            let plan = plan_topology(config.local_vram_mb, &workers);
            if let Err(e) = write_and_reload_swap(&config, &plan, &mut swap_child).await {
                error!("failed to reload swap: {e:#}");
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Proxy TCP accept task — one per worker connection
// ---------------------------------------------------------------------------

/// Accepts TCP connections from llama-server on the proxy port and sets up
/// tunnel streams through the WebSocket to the remote worker.
async fn proxy_accept_task(
    listener: TcpListener,
    ws_tx: mpsc::Sender<Message>,
    close_tx: mpsc::Sender<u32>,
    streams: Arc<Mutex<HashMap<u32, mpsc::Sender<Vec<u8>>>>>,
) {
    let mut next_id = 1u32;
    loop {
        match listener.accept().await {
            Ok((tcp_stream, _)) => {
                let stream_id = next_id;
                next_id += 1;

                // Send TunnelOpen before spawning stream tasks — this guarantees
                // the worker sees TunnelOpen before any binary data frames, since
                // both go through the same ordered ws_tx channel.
                let open = CoordinatorMsg::TunnelOpen { stream_id };
                if ws_tx
                    .send(Message::Text(serde_json::to_string(&open).unwrap()))
                    .await
                    .is_err()
                {
                    break; // WebSocket closed
                }

                let data_tx =
                    tunnel::spawn_stream(stream_id, tcp_stream, ws_tx.clone(), close_tx.clone());
                streams.lock().unwrap().insert(stream_id, data_tx);

                info!("tunnel stream {stream_id} opened");
            }
            Err(e) => {
                warn!("proxy accept error: {e}");
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket connection handler
// ---------------------------------------------------------------------------

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_addr: SocketAddr,
    tx: mpsc::Sender<CoordEvent>,
) {
    let conn_id = NEXT_CONN_ID.fetch_add(1, Ordering::Relaxed);

    let ws = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            warn!("websocket handshake failed from {peer_addr}: {e}");
            return;
        }
    };

    let (mut sink, mut ws_stream) = ws.split();

    // Shared WebSocket writer — tunnel stream tasks + control messages all
    // send through this channel; a dedicated task drains it into the sink.
    let (ws_tx, mut ws_rx) = mpsc::channel::<Message>(64);
    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = ws_rx.recv().await {
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    // First message must be Announce
    let announce = match ws_stream.next().await {
        Some(Ok(Message::Text(text))) => match serde_json::from_str::<WorkerMsg>(&text) {
            Ok(WorkerMsg::Announce(a)) => a,
            Ok(other) => {
                warn!("expected Announce from {peer_addr}, got {other:?}");
                return;
            }
            Err(e) => {
                warn!("invalid message from {peer_addr}: {e}");
                return;
            }
        },
        other => {
            warn!("unexpected first frame from {peer_addr}: {other:?}");
            return;
        }
    };

    let node_id = announce.node_id.clone();

    // Bind a local proxy port for this worker — llama-server connects here,
    // and the data tunnels through the WebSocket to the worker's RPC server.
    let proxy_listener = match TcpListener::bind("127.0.0.1:0").await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind proxy listener for {node_id}: {e}");
            return;
        }
    };
    let proxy_port = proxy_listener.local_addr().unwrap().port();

    info!(
        "worker connected: {} ({}, {}MB VRAM, preemptible={}) [conn {}, proxy port {}]",
        node_id, announce.gpu, announce.vram_mb, announce.preemptible, conn_id, proxy_port
    );

    let _ = tx
        .send(CoordEvent::Joined {
            conn_id,
            node_id: node_id.clone(),
            announce,
            proxy_port,
        })
        .await;

    // Send Ack through the shared writer
    let ack = CoordinatorMsg::Ack {
        node_id: node_id.clone(),
    };
    let _ = ws_tx
        .send(Message::Text(serde_json::to_string(&ack).unwrap()))
        .await;

    // Tunnel stream tracking — shared with the proxy accept task
    let streams: Arc<Mutex<HashMap<u32, mpsc::Sender<Vec<u8>>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let (close_tx, mut close_rx) = mpsc::channel::<u32>(32);

    let proxy_handle = tokio::spawn(proxy_accept_task(
        proxy_listener,
        ws_tx.clone(),
        close_tx,
        streams.clone(),
    ));

    // Main loop — multiplex control messages and tunnel binary frames
    loop {
        tokio::select! {
            msg = ws_stream.next() => {
                match msg {
                    Some(Ok(Message::Binary(data))) => {
                        if let Some((stream_id, payload)) = tunnel::decode_frame(&data) {
                            let sender = streams.lock().unwrap().get(&stream_id).cloned();
                            if let Some(sender) = sender {
                                let _ = sender.send(payload.to_vec()).await;
                            }
                        }
                    }
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<WorkerMsg>(&text) {
                            Ok(WorkerMsg::Draining { reason }) => {
                                info!("worker {node_id} draining: {reason}");
                                let _ = tx
                                    .send(CoordEvent::Draining {
                                        conn_id,
                                        node_id: node_id.clone(),
                                    })
                                    .await;
                            }
                            Ok(WorkerMsg::Resuming) => {
                                info!("worker {node_id} resuming");
                                let _ = tx
                                    .send(CoordEvent::Resuming {
                                        conn_id,
                                        node_id: node_id.clone(),
                                    })
                                    .await;
                            }
                            Ok(WorkerMsg::TunnelClose { stream_id }) => {
                                streams.lock().unwrap().remove(&stream_id);
                            }
                            Ok(WorkerMsg::Announce(_)) => {
                                warn!("unexpected re-announce from {node_id}, ignoring");
                            }
                            Err(e) => warn!("bad message from {node_id}: {e}"),
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = ws_tx.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => {
                        warn!("websocket error from {node_id}: {e}");
                        break;
                    }
                    _ => {}
                }
            }

            closed_id = close_rx.recv() => {
                // A tunnel stream's TCP side closed — notify the worker
                if let Some(stream_id) = closed_id {
                    streams.lock().unwrap().remove(&stream_id);
                    let close = CoordinatorMsg::TunnelClose { stream_id };
                    let _ = ws_tx
                        .send(Message::Text(serde_json::to_string(&close).unwrap()))
                        .await;
                }
            }
        }
    }

    // Cleanup — aborting the writer drops ws_rx, which makes all ws_tx.send()
    // calls in tunnel stream tasks fail, cascading a clean shutdown.
    proxy_handle.abort();
    writer_handle.abort();

    info!("worker disconnected: {node_id} [conn {conn_id}]");
    let _ = tx.send(CoordEvent::Left { conn_id, node_id }).await;
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

pub async fn run(args: Args) -> Result<()> {
    let config_str = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config: {}", args.config.display()))?;
    let config: Config = toml::from_str(&config_str).context("failed to parse config")?;

    info!("llama-mesh coordinator starting on {}", args.listen);
    info!("local VRAM: {}MB", config.local_vram_mb);
    info!(
        "models: {}",
        config
            .models
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let (tx, rx) = mpsc::channel(32);

    tokio::spawn(async move {
        if let Err(e) = coordinator_task(config, rx).await {
            error!("coordinator task failed: {e:#}");
            std::process::exit(1);
        }
    });

    let listener = TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("failed to bind {}", args.listen))?;

    info!("listening for workers on {}", args.listen);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let tx = tx.clone();
                tokio::spawn(handle_connection(stream, addr, tx));
            }
            Err(e) => error!("accept error: {e}"),
        }
    }
}
