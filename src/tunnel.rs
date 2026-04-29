//! TCP-over-WebSocket tunnel.
//!
//! Multiplexes TCP streams over a single WebSocket connection using binary
//! frames with a 4-byte stream-ID header. Text frames carry JSON control
//! messages (the existing protocol).

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

const TCP_BUF_SIZE: usize = 64 * 1024;

/// Prefix `data` with a big-endian stream ID.
pub fn encode_frame(stream_id: u32, data: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(4 + data.len());
    frame.extend_from_slice(&stream_id.to_be_bytes());
    frame.extend_from_slice(data);
    frame
}

/// Extract stream ID and payload from a binary frame.
pub fn decode_frame(frame: &[u8]) -> Option<(u32, &[u8])> {
    if frame.len() < 4 {
        return None;
    }
    let id = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]);
    Some((id, &frame[4..]))
}

/// Wire up bidirectional forwarding between a TCP stream and the WebSocket.
///
/// Returns the sender used to push data from the WebSocket into this
/// stream's TCP write half. The caller stores it keyed by `stream_id`.
pub fn spawn_stream(
    stream_id: u32,
    tcp: tokio::net::TcpStream,
    ws_tx: mpsc::Sender<Message>,
    close_tx: mpsc::Sender<u32>,
) -> mpsc::Sender<Vec<u8>> {
    let (tcp_read, tcp_write) = tcp.into_split();

    // TCP → WebSocket (binary frames)
    tokio::spawn(tcp_to_ws(stream_id, tcp_read, ws_tx, close_tx));

    // WebSocket → TCP
    let (data_tx, data_rx) = mpsc::channel(32);
    tokio::spawn(ws_to_tcp(tcp_write, data_rx));

    data_tx
}

async fn tcp_to_ws(
    stream_id: u32,
    mut reader: OwnedReadHalf,
    ws_tx: mpsc::Sender<Message>,
    close_tx: mpsc::Sender<u32>,
) {
    let mut buf = vec![0u8; TCP_BUF_SIZE];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let frame = encode_frame(stream_id, &buf[..n]);
                if ws_tx.send(Message::Binary(frame)).await.is_err() {
                    break;
                }
            }
        }
    }
    let _ = close_tx.send(stream_id).await;
}

async fn ws_to_tcp(mut writer: OwnedWriteHalf, mut rx: mpsc::Receiver<Vec<u8>>) {
    while let Some(data) = rx.recv().await {
        if writer.write_all(&data).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let data = b"test payload";
        let frame = encode_frame(42, data);
        let (id, payload) = decode_frame(&frame).unwrap();
        assert_eq!(id, 42);
        assert_eq!(payload, data);
    }

    #[test]
    fn decode_rejects_short_frames() {
        assert!(decode_frame(&[]).is_none());
        assert!(decode_frame(&[1]).is_none());
        assert!(decode_frame(&[1, 2]).is_none());
        assert!(decode_frame(&[1, 2, 3]).is_none());
    }

    #[test]
    fn decode_accepts_header_only() {
        let (id, data) = decode_frame(&[0, 0, 0, 1]).unwrap();
        assert_eq!(id, 1);
        assert!(data.is_empty());
    }

    #[test]
    fn stream_id_is_big_endian() {
        let frame = encode_frame(0x01020304, &[]);
        assert_eq!(&frame[..4], &[1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn spawn_stream_forwards_bidirectionally() {
        // Echo server
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (mut r, mut w) = stream.into_split();
            tokio::io::copy(&mut r, &mut w).await.ok();
        });

        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (ws_tx, mut ws_rx) = mpsc::channel::<Message>(16);
        let (close_tx, mut close_rx) = mpsc::channel::<u32>(4);

        let data_tx = spawn_stream(1, tcp, ws_tx, close_tx);

        // WS -> TCP -> echo -> TCP -> WS
        data_tx.send(b"hello".to_vec()).await.unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(2), ws_rx.recv())
            .await
            .expect("timeout waiting for echo")
            .unwrap();

        match msg {
            Message::Binary(frame) => {
                let (id, payload) = decode_frame(&frame).unwrap();
                assert_eq!(id, 1);
                assert_eq!(payload, b"hello");
            }
            other => panic!("expected Binary, got {other:?}"),
        }

        // Close and verify notification
        drop(data_tx);
        let closed = tokio::time::timeout(std::time::Duration::from_secs(2), close_rx.recv())
            .await
            .expect("timeout waiting for close")
            .unwrap();
        assert_eq!(closed, 1);
    }
}
