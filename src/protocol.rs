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
    TunnelClose { stream_id: u32 },
}

/// Messages sent from coordinator to worker over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum CoordinatorMsg {
    Ack { node_id: String },
    TopologyUpdate { active_workers: usize },
    TunnelOpen { stream_id: u32 },
    TunnelClose { stream_id: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_worker(msg: &WorkerMsg) -> WorkerMsg {
        let json = serde_json::to_string(msg).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    fn roundtrip_coord(msg: &CoordinatorMsg) -> CoordinatorMsg {
        let json = serde_json::to_string(msg).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn worker_announce_roundtrip() {
        let msg = WorkerMsg::Announce(WorkerAnnounce {
            node_id: "test".into(),
            gpu: GpuType::Cuda,
            vram_mb: 8192,
            rpc_port: 50052,
            preemptible: true,
        });
        match roundtrip_worker(&msg) {
            WorkerMsg::Announce(a) => {
                assert_eq!(a.node_id, "test");
                assert_eq!(a.gpu, GpuType::Cuda);
                assert_eq!(a.vram_mb, 8192);
                assert_eq!(a.rpc_port, 50052);
                assert!(a.preemptible);
            }
            other => panic!("expected Announce, got {other:?}"),
        }
    }

    #[test]
    fn worker_draining_roundtrip() {
        let msg = WorkerMsg::Draining {
            reason: "game launched".into(),
        };
        match roundtrip_worker(&msg) {
            WorkerMsg::Draining { reason } => assert_eq!(reason, "game launched"),
            other => panic!("expected Draining, got {other:?}"),
        }
    }

    #[test]
    fn worker_resuming_roundtrip() {
        match roundtrip_worker(&WorkerMsg::Resuming) {
            WorkerMsg::Resuming => {}
            other => panic!("expected Resuming, got {other:?}"),
        }
    }

    #[test]
    fn worker_tunnel_close_roundtrip() {
        let msg = WorkerMsg::TunnelClose { stream_id: 42 };
        match roundtrip_worker(&msg) {
            WorkerMsg::TunnelClose { stream_id } => assert_eq!(stream_id, 42),
            other => panic!("expected TunnelClose, got {other:?}"),
        }
    }

    #[test]
    fn coordinator_ack_roundtrip() {
        let msg = CoordinatorMsg::Ack {
            node_id: "node1".into(),
        };
        match roundtrip_coord(&msg) {
            CoordinatorMsg::Ack { node_id } => assert_eq!(node_id, "node1"),
            other => panic!("expected Ack, got {other:?}"),
        }
    }

    #[test]
    fn coordinator_topology_update_roundtrip() {
        let msg = CoordinatorMsg::TopologyUpdate { active_workers: 3 };
        match roundtrip_coord(&msg) {
            CoordinatorMsg::TopologyUpdate { active_workers } => assert_eq!(active_workers, 3),
            other => panic!("expected TopologyUpdate, got {other:?}"),
        }
    }

    #[test]
    fn coordinator_tunnel_open_roundtrip() {
        let msg = CoordinatorMsg::TunnelOpen { stream_id: 7 };
        match roundtrip_coord(&msg) {
            CoordinatorMsg::TunnelOpen { stream_id } => assert_eq!(stream_id, 7),
            other => panic!("expected TunnelOpen, got {other:?}"),
        }
    }

    #[test]
    fn coordinator_tunnel_close_roundtrip() {
        let msg = CoordinatorMsg::TunnelClose { stream_id: 7 };
        match roundtrip_coord(&msg) {
            CoordinatorMsg::TunnelClose { stream_id } => assert_eq!(stream_id, 7),
            other => panic!("expected TunnelClose, got {other:?}"),
        }
    }

    #[test]
    fn gpu_type_display() {
        assert_eq!(GpuType::Cuda.to_string(), "cuda");
        assert_eq!(GpuType::Rocm.to_string(), "rocm");
        assert_eq!(GpuType::Cpu.to_string(), "cpu");
    }

    #[test]
    fn gpu_type_serde_lowercase() {
        let json = serde_json::to_string(&GpuType::Cuda).unwrap();
        assert_eq!(json, "\"cuda\"");
        let parsed: GpuType = serde_json::from_str("\"rocm\"").unwrap();
        assert_eq!(parsed, GpuType::Rocm);
    }
}
