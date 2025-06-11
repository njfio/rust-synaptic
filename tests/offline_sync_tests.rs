#[cfg(feature = "cross-platform")]
use synaptic::cross_platform::offline::{OfflineAdapter, InMemoryRemoteBackend, RemoteBackend, SyncOperation};
#[cfg(feature = "cross-platform")]
use std::sync::Arc;
#[cfg(feature = "cross-platform")]
use std::net::TcpListener;
#[cfg(feature = "cross-platform")]
use std::thread;

#[cfg(feature = "cross-platform")]
fn start_server() -> (std::thread::JoinHandle<()>, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        // Accept a single connection then drop
        let _ = listener.accept();
    });
    (handle, format!("{}", addr))
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_sync_upload() {
    let remote = Arc::new(InMemoryRemoteBackend::default());
    let adapter = OfflineAdapter::with_remote_backend(remote.clone() as Arc<dyn RemoteBackend>).unwrap();

    let (handle, addr) = start_server();
    std::env::set_var("SYNC_ONLINE_TEST_ADDR", &addr);

    adapter.store("key1", b"data1").unwrap();
    adapter.sync_with_remote().await.unwrap();
    handle.join().unwrap();

    assert!(remote.download("key1").unwrap().is_some());
    assert!(adapter.get_pending_operations().unwrap().is_empty());
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_sync_delete() {
    let remote = Arc::new(InMemoryRemoteBackend::default());
    remote.upload("key2", b"data2", 1).unwrap();
    let adapter = OfflineAdapter::with_remote_backend(remote.clone() as Arc<dyn RemoteBackend>).unwrap();

    let (handle, addr) = start_server();
    std::env::set_var("SYNC_ONLINE_TEST_ADDR", &addr);

    adapter.queue_sync_operation(SyncOperation::Delete { key: "key2".into() }).unwrap();
    adapter.sync_with_remote().await.unwrap();
    handle.join().unwrap();

    assert!(remote.download("key2").unwrap().is_none());
}

#[cfg(feature = "cross-platform")]
#[tokio::test]
async fn test_sync_download() {
    let remote = Arc::new(InMemoryRemoteBackend::default());
    remote.upload("key3", b"data3", 1).unwrap();
    let adapter = OfflineAdapter::with_remote_backend(remote.clone() as Arc<dyn RemoteBackend>).unwrap();

    let (handle, addr) = start_server();
    std::env::set_var("SYNC_ONLINE_TEST_ADDR", &addr);

    adapter.queue_sync_operation(SyncOperation::Download { key: "key3".into(), remote_version: 1 }).unwrap();
    adapter.sync_with_remote().await.unwrap();
    handle.join().unwrap();

    let data = adapter.retrieve("key3").unwrap();
    assert_eq!(data, Some(b"data3".to_vec()));
}
