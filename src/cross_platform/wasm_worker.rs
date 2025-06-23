//! Web Worker support for WebAssembly memory operations
//!
//! This module provides web worker integration for offloading heavy memory operations
//! to background threads in the browser environment.

#[cfg(feature = "wasm")]
use {
    js_sys::{Array, Function, Object, Promise, Uint8Array},
    wasm_bindgen::{prelude::*, JsCast},
    wasm_bindgen_futures::JsFuture,
    web_sys::{console, window, Worker, MessageEvent, DedicatedWorkerGlobalScope},
};

use crate::error::MemoryError as SynapticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Web Worker message types for memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerMessage {
    /// Store data operation
    Store {
        key: String,
        data: Vec<u8>,
        compress: bool,
    },
    /// Retrieve data operation
    Retrieve {
        key: String,
    },
    /// Delete data operation
    Delete {
        key: String,
    },
    /// List keys operation
    ListKeys,
    /// Compress data operation
    Compress {
        data: Vec<u8>,
    },
    /// Decompress data operation
    Decompress {
        data: Vec<u8>,
    },
    /// Search operation
    Search {
        query: String,
        limit: usize,
    },
}

/// Web Worker response types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerResponse {
    /// Operation completed successfully
    Success {
        request_id: String,
        data: Option<Vec<u8>>,
    },
    /// Operation failed with error
    Error {
        request_id: String,
        message: String,
    },
    /// List of keys response
    Keys {
        request_id: String,
        keys: Vec<String>,
    },
    /// Search results response
    SearchResults {
        request_id: String,
        results: Vec<SearchResult>,
    },
}

/// Search result from web worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub key: String,
    pub score: f64,
    pub snippet: String,
}

/// Web Worker manager for memory operations
#[derive(Debug)]
pub struct WebWorkerManager {
    #[cfg(feature = "wasm")]
    worker: Option<Worker>,
    pending_requests: Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<WorkerResponse>>>>,
    request_counter: Arc<Mutex<u64>>,
}

impl WebWorkerManager {
    /// Create a new web worker manager
    pub fn new() -> Result<Self, SynapticError> {
        Ok(Self {
            #[cfg(feature = "wasm")]
            worker: None,
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            request_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Initialize the web worker
    #[cfg(feature = "wasm")]
    pub async fn initialize(&mut self) -> Result<(), SynapticError> {
        // Create web worker from inline script
        let worker_script = include_str!("../../assets/memory_worker.js");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(
            &js_sys::Array::of1(&JsValue::from_str(worker_script)),
            web_sys::BlobPropertyBag::new().type_("application/javascript"),
        ).map_err(|e| SynapticError::ProcessingError(format!("Failed to create blob: {:?}", e)))?;

        let url = web_sys::Url::create_object_url_with_blob(&blob)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create object URL: {:?}", e)))?;

        let worker = Worker::new(&url)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create worker: {:?}", e)))?;

        // Set up message handler
        let pending_requests = self.pending_requests.clone();
        let onmessage_callback = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Ok(response_data) = event.data().into_serde::<WorkerResponse>() {
                let mut requests = pending_requests.lock().unwrap();
                match &response_data {
                    WorkerResponse::Success { request_id, .. } |
                    WorkerResponse::Error { request_id, .. } |
                    WorkerResponse::Keys { request_id, .. } |
                    WorkerResponse::SearchResults { request_id, .. } => {
                        if let Some(sender) = requests.remove(request_id) {
                            let _ = sender.send(response_data);
                        }
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);

        worker.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget();

        self.worker = Some(worker);
        Ok(())
    }

    /// Send a message to the web worker and wait for response
    #[cfg(feature = "wasm")]
    pub async fn send_message(&self, message: WorkerMessage) -> Result<WorkerResponse, SynapticError> {
        let worker = self.worker.as_ref()
            .ok_or_else(|| SynapticError::ProcessingError("Worker not initialized".to_string()))?;

        // Generate unique request ID
        let request_id = {
            let mut counter = self.request_counter.lock().unwrap();
            *counter += 1;
            format!("req_{}", *counter)
        };

        // Create response channel
        let (sender, receiver) = tokio::sync::oneshot::channel();
        {
            let mut requests = self.pending_requests.lock().unwrap();
            requests.insert(request_id.clone(), sender);
        }

        // Send message to worker
        let message_with_id = serde_json::json!({
            "request_id": request_id,
            "message": message
        });

        worker.post_message(&JsValue::from_str(&message_with_id.to_string()))
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to send message to worker: {:?}", e)))?;

        // Wait for response with timeout
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            receiver
        ).await
            .map_err(|_| SynapticError::ProcessingError("Worker request timeout".to_string()))?
            .map_err(|_| SynapticError::ProcessingError("Worker response channel closed".to_string()))?;

        Ok(response)
    }

    /// Store data using web worker
    pub async fn store_async(&self, key: &str, data: &[u8], compress: bool) -> Result<(), SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::Store {
                key: key.to_string(),
                data: data.to_vec(),
                compress,
            };

            match self.send_message(message).await? {
                WorkerResponse::Success { .. } => Ok(()),
                WorkerResponse::Error { message, .. } => {
                    Err(SynapticError::ProcessingError(format!("Worker store failed: {}", message)))
                }
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// Retrieve data using web worker
    pub async fn retrieve_async(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::Retrieve {
                key: key.to_string(),
            };

            match self.send_message(message).await? {
                WorkerResponse::Success { data, .. } => Ok(data),
                WorkerResponse::Error { message, .. } => {
                    Err(SynapticError::ProcessingError(format!("Worker retrieve failed: {}", message)))
                }
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// Delete data using web worker
    pub async fn delete_async(&self, key: &str) -> Result<bool, SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::Delete {
                key: key.to_string(),
            };

            match self.send_message(message).await? {
                WorkerResponse::Success { .. } => Ok(true),
                WorkerResponse::Error { .. } => Ok(false),
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// List keys using web worker
    pub async fn list_keys_async(&self) -> Result<Vec<String>, SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::ListKeys;

            match self.send_message(message).await? {
                WorkerResponse::Keys { keys, .. } => Ok(keys),
                WorkerResponse::Error { message, .. } => {
                    Err(SynapticError::ProcessingError(format!("Worker list keys failed: {}", message)))
                }
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// Compress data using web worker
    pub async fn compress_async(&self, data: &[u8]) -> Result<Vec<u8>, SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::Compress {
                data: data.to_vec(),
            };

            match self.send_message(message).await? {
                WorkerResponse::Success { data: Some(compressed), .. } => Ok(compressed),
                WorkerResponse::Error { message, .. } => {
                    Err(SynapticError::ProcessingError(format!("Worker compression failed: {}", message)))
                }
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// Search data using web worker
    pub async fn search_async(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SynapticError> {
        #[cfg(feature = "wasm")]
        {
            let message = WorkerMessage::Search {
                query: query.to_string(),
                limit,
            };

            match self.send_message(message).await? {
                WorkerResponse::SearchResults { results, .. } => Ok(results),
                WorkerResponse::Error { message, .. } => {
                    Err(SynapticError::ProcessingError(format!("Worker search failed: {}", message)))
                }
                _ => Err(SynapticError::ProcessingError("Unexpected worker response".to_string())),
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            Err(SynapticError::ProcessingError("Web workers not available".to_string()))
        }
    }

    /// Terminate the web worker
    #[cfg(feature = "wasm")]
    pub fn terminate(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.terminate();
        }
        
        // Clear pending requests
        let mut requests = self.pending_requests.lock().unwrap();
        requests.clear();
    }
}

impl Drop for WebWorkerManager {
    fn drop(&mut self) {
        #[cfg(feature = "wasm")]
        self.terminate();
    }
}
