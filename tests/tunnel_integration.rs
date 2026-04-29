//! End-to-end tunnel integration test.
//!
//! Verifies the full data path:
//!   TCP client → coordinator proxy → WebSocket → worker → echo server → back
//!
//! No GPU hardware or ggml libraries are exercised — this tests the
//! networking layer only.

use futures_util::{SinkExt, StreamExt};
use llama_mesh::protocol::{CoordinatorMsg, GpuType, WorkerAnnounce, WorkerMsg};
use llama_mesh::tunnel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

/// Full round-trip: data enters the coordinator's proxy TCP port, travels
/// through the WebSocket tunnel to the worker, hits an echo server, and
/// comes back the same way.
#[tokio::test]
async fn data_roundtrips_through_tunnel() {
    // ---- 1. Echo server (simulates the RPC server on the worker machine) ----
    let echo = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let echo_port = echo.local_addr().unwrap().port();
    tokio::spawn(async move {
        let (stream, _) = echo.accept().await.unwrap();
        let (mut r, mut w) = stream.into_split();
        tokio::io::copy(&mut r, &mut w).await.ok();
    });

    // ---- 2. WebSocket server (coordinator side) ----
    let ws_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ws_addr = ws_listener.local_addr().unwrap();
    let (proxy_port_tx, proxy_port_rx) = tokio::sync::oneshot::channel::<u16>();

    // Coordinator task — accepts one worker, opens one tunnel stream
    tokio::spawn(async move {
        let (tcp, _) = ws_listener.accept().await.unwrap();
        let ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
        let (mut sink, mut ws_stream) = ws.split();

        // Shared WebSocket writer
        let (ws_tx, mut ws_rx) = mpsc::channel::<Message>(32);
        tokio::spawn(async move {
            while let Some(msg) = ws_rx.recv().await {
                if sink.send(msg).await.is_err() {
                    break;
                }
            }
        });

        // Read Announce
        let msg = ws_stream.next().await.unwrap().unwrap();
        assert!(matches!(msg, Message::Text(_)));

        // Send Ack
        let ack = CoordinatorMsg::Ack {
            node_id: "test".into(),
        };
        ws_tx
            .send(Message::Text(serde_json::to_string(&ack).unwrap()))
            .await
            .unwrap();

        // Bind proxy port
        let proxy = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = proxy.local_addr().unwrap().port();
        proxy_port_tx.send(port).unwrap();

        // Tunnel stream tracking
        let streams: Arc<Mutex<HashMap<u32, mpsc::Sender<Vec<u8>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (close_tx, mut close_rx) = mpsc::channel::<u32>(8);

        // Accept one proxy TCP connection, set up tunnel
        let streams2 = streams.clone();
        let ws_tx2 = ws_tx.clone();
        let close_tx2 = close_tx;
        tokio::spawn(async move {
            let (tcp, _) = proxy.accept().await.unwrap();
            let stream_id = 1u32;

            // TunnelOpen must be queued before stream tasks start sending
            let open = CoordinatorMsg::TunnelOpen { stream_id };
            ws_tx2
                .send(Message::Text(serde_json::to_string(&open).unwrap()))
                .await
                .unwrap();

            let data_tx = tunnel::spawn_stream(stream_id, tcp, ws_tx2, close_tx2);
            streams2.lock().unwrap().insert(stream_id, data_tx);
        });

        // Route binary frames from worker back to proxy TCP
        loop {
            tokio::select! {
                msg = ws_stream.next() => {
                    match msg {
                        Some(Ok(Message::Binary(data))) => {
                            if let Some((stream_id, payload)) = tunnel::decode_frame(&data) {
                                let tx = streams.lock().unwrap().get(&stream_id).cloned();
                                if let Some(tx) = tx {
                                    tx.send(payload.to_vec()).await.ok();
                                }
                            }
                        }
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(WorkerMsg::TunnelClose { stream_id }) =
                                serde_json::from_str(&text)
                            {
                                streams.lock().unwrap().remove(&stream_id);
                            }
                        }
                        None | Some(Ok(Message::Close(_))) => break,
                        _ => {}
                    }
                }
                id = close_rx.recv() => {
                    if let Some(stream_id) = id {
                        streams.lock().unwrap().remove(&stream_id);
                        let close = CoordinatorMsg::TunnelClose { stream_id };
                        ws_tx
                            .send(Message::Text(serde_json::to_string(&close).unwrap()))
                            .await
                            .ok();
                    }
                }
            }
        }
    });

    // ---- 3. Worker side (WebSocket client) ----
    let (ws, _) = tokio_tungstenite::connect_async(format!("ws://{ws_addr}"))
        .await
        .unwrap();
    let (mut sink, mut ws_stream) = ws.split();

    // Shared WebSocket writer
    let (ws_tx, mut ws_rx) = mpsc::channel::<Message>(32);
    tokio::spawn(async move {
        while let Some(msg) = ws_rx.recv().await {
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Send Announce
    let announce = WorkerMsg::Announce(WorkerAnnounce {
        node_id: "test-worker".into(),
        gpu: GpuType::Cpu,
        vram_mb: 1024,
        rpc_port: echo_port,
        preemptible: false,
    });
    ws_tx
        .send(Message::Text(serde_json::to_string(&announce).unwrap()))
        .await
        .unwrap();

    // Read Ack
    let _ack = ws_stream.next().await.unwrap().unwrap();

    // Worker tunnel tracking + message handler
    let ws_tx2 = ws_tx.clone();
    let (close_tx, mut close_rx) = mpsc::channel::<u32>(8);
    let worker_handle = tokio::spawn(async move {
        let mut streams: HashMap<u32, mpsc::Sender<Vec<u8>>> = HashMap::new();
        loop {
            tokio::select! {
                msg = ws_stream.next() => {
                    match msg {
                        Some(Ok(Message::Binary(data))) => {
                            if let Some((stream_id, payload)) = tunnel::decode_frame(&data) {
                                if let Some(tx) = streams.get(&stream_id) {
                                    tx.send(payload.to_vec()).await.ok();
                                }
                            }
                        }
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<CoordinatorMsg>(&text) {
                                Ok(CoordinatorMsg::TunnelOpen { stream_id }) => {
                                    let tcp = tokio::net::TcpStream::connect(
                                        format!("127.0.0.1:{echo_port}")
                                    ).await.unwrap();
                                    let data_tx = tunnel::spawn_stream(
                                        stream_id, tcp, ws_tx2.clone(), close_tx.clone(),
                                    );
                                    streams.insert(stream_id, data_tx);
                                }
                                Ok(CoordinatorMsg::TunnelClose { stream_id }) => {
                                    streams.remove(&stream_id);
                                }
                                _ => {}
                            }
                        }
                        None | Some(Ok(Message::Close(_))) => break,
                        _ => {}
                    }
                }
                id = close_rx.recv() => {
                    if let Some(stream_id) = id {
                        streams.remove(&stream_id);
                        let close = WorkerMsg::TunnelClose { stream_id };
                        ws_tx2
                            .send(Message::Text(serde_json::to_string(&close).unwrap()))
                            .await
                            .ok();
                    }
                }
            }
        }
    });

    // ---- 4. TCP client connects to the coordinator's proxy port ----
    let proxy_port = proxy_port_rx.await.unwrap();

    // Small delay for the proxy accept task to start listening
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut client = tokio::net::TcpStream::connect(format!("127.0.0.1:{proxy_port}"))
        .await
        .unwrap();

    // Give the tunnel time to set up (TunnelOpen → worker connect → stream spawn)
    tokio::time::sleep(Duration::from_millis(200)).await;

    // ---- 5. Send data and verify it echoes back ----
    let test_data = b"hello through the tunnel!";
    client.write_all(test_data).await.unwrap();

    let mut buf = vec![0u8; test_data.len()];
    tokio::time::timeout(Duration::from_secs(5), client.read_exact(&mut buf))
        .await
        .expect("timeout waiting for echo")
        .unwrap();
    assert_eq!(&buf, test_data);

    // ---- 6. Verify the tunnel handles multiple messages ----
    let test_data2 = b"second message through tunnel";
    client.write_all(test_data2).await.unwrap();

    let mut buf2 = vec![0u8; test_data2.len()];
    tokio::time::timeout(Duration::from_secs(5), client.read_exact(&mut buf2))
        .await
        .expect("timeout on second echo")
        .unwrap();
    assert_eq!(&buf2, test_data2);

    // Cleanup
    drop(client);
    worker_handle.abort();
}
