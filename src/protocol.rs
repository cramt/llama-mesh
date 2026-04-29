use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GpuType {
    Cuda,
    Rocm,
    Cpu,
}

impl std::fmt::Display for GpuType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "cuda"),
            Self::Rocm => write!(f, "rocm"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerAnnounce {
    pub node_id: String,
    pub gpu: GpuType,
    pub vram_mb: u64,
    pub rpc_port: u16,
    pub preemptible: bool,
}

/// Messages sent from worker to coordinator over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum WorkerMsg {
    Announce(WorkerAnnounce),
    Draining { reason: String },
    Resuming,
}

/// Messages sent from coordinator to worker over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum CoordinatorMsg {
    Ack { node_id: String },
    TopologyUpdate { active_workers: usize },
}
