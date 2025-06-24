//! # WebAssembly Support
//!
//! WebAssembly adapter for running Synaptic memory system in web browsers.
//! Provides IndexedDB storage, web worker support, and browser-specific optimizations.

use super::{
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, PlatformInfo, Platform,
    PerformanceProfile, StorageBackend, StorageStats,
};
use crate::error::MemoryError as SynapticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::rc::Rc;

mod wasm_worker;
use wasm_worker::WebWorkerManager;

#[cfg(feature = "wasm")]
use {
    js_sys::{Array, Object, Promise, Uint8Array},
    wasm_bindgen::{prelude::*, JsCast},
    wasm_bindgen_futures::JsFuture,
    web_sys::{
        console, window, IdbDatabase, IdbFactory, IdbObjectStore, IdbOpenDbRequest, IdbRequest,
        IdbTransaction, IdbTransactionMode, Storage,
    },
};

/// WebAssembly memory adapter
#[derive(Debug)]
pub struct WasmAdapter {
    /// IndexedDB database connection
    #[cfg(feature = "wasm")]
    db: Option<IdbDatabase>,

    /// Local storage fallback
    #[cfg(feature = "wasm")]
    local_storage: Option<Storage>,

    /// In-memory cache with interior mutability
    memory_cache: RefCell<HashMap<String, Vec<u8>>>,

    /// Web worker manager for background operations
    worker_manager: RefCell<Option<WebWorkerManager>>,

    /// Configuration
    config: WasmConfig,
}

/// WebAssembly-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// IndexedDB database name
    pub db_name: String,

    /// IndexedDB version
    pub db_version: u32,

    /// Object store name
    pub store_name: String,

    /// Enable local storage fallback
    pub enable_local_storage_fallback: bool,

    /// Enable memory cache
    pub enable_memory_cache: bool,

    /// Maximum cache size (bytes)
    pub max_cache_size: usize,

    /// Enable compression
    pub enable_compression: bool,

    /// Enable web worker for background operations
    pub enable_web_worker: bool,

    /// Web worker timeout (seconds)
    pub worker_timeout_seconds: u64,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            db_name: "synaptic_memory".to_string(),
            db_version: 1,
            store_name: "memories".to_string(),
            enable_local_storage_fallback: true,
            enable_memory_cache: true,
            max_cache_size: 10 * 1024 * 1024, // 10MB
            enable_compression: true,
            enable_web_worker: true,
            worker_timeout_seconds: 30,
        }
    }
}

impl WasmAdapter {
    /// Create a new WebAssembly adapter
    pub fn new() -> Result<Self, SynapticError> {
        Ok(Self {
            #[cfg(feature = "wasm")]
            db: None,
            #[cfg(feature = "wasm")]
            local_storage: None,
            memory_cache: RefCell::new(HashMap::new()),
            worker_manager: RefCell::new(None),
            config: WasmConfig::default(),
        })
    }

    /// Initialize IndexedDB connection
    #[cfg(feature = "wasm")]
    async fn initialize_indexeddb(&mut self) -> Result<(), SynapticError> {
        let window = window().ok_or_else(|| SynapticError::ProcessingError("No window object available".to_string()))?;
        
        let idb_factory = window
            .indexed_db()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get IndexedDB: {:?}", e)))?
            .ok_or_else(|| SynapticError::ProcessingError("IndexedDB not supported".to_string()))?;

        // Open database
        let open_request = idb_factory
            .open_with_u32(&self.config.db_name, self.config.db_version)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to open database: {:?}", e)))?;

        // Set up upgrade handler
        let upgrade_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
            if let Some(target) = event.target() {
                if let Ok(request) = target.dyn_into::<IdbOpenDbRequest>() {
                    if let Ok(result) = request.result() {
                        if let Ok(db) = result.dyn_into::<IdbDatabase>() {
                            // Create object store if it doesn't exist
                            if !db.object_store_names().contains(&"memories".into()) {
                                if let Ok(_store) = db.create_object_store("memories") {
                                    console::log_1(&"Created object store".into());
                                } else {
                                    console::log_1(&"Failed to create object store".into());
                                }
                            }
                        } else {
                            console::log_1(&"Failed to cast to IdbDatabase".into());
                        }
                    } else {
                        console::log_1(&"Failed to get request result".into());
                    }
                } else {
                    console::log_1(&"Failed to cast to IdbOpenDbRequest".into());
                }
            } else {
                console::log_1(&"Event target is None".into());
            }
        }) as Box<dyn FnMut(_)>);

        open_request.set_onupgradeneeded(Some(upgrade_closure.as_ref().unchecked_ref()));
        upgrade_closure.forget();

        // Wait for database to open
        let promise = Promise::resolve(&JsFuture::from(open_request).await
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to open IndexedDB: {:?}", e)))?);
        
        let db_result = JsFuture::from(promise).await
            .map_err(|e| SynapticError::ProcessingError(format!("Database open failed: {:?}", e)))?;
        
        self.db = Some(db_result.dyn_into::<IdbDatabase>()
            .map_err(|e| SynapticError::ProcessingError(format!("Invalid database object: {:?}", e)))?);

        Ok(())
    }

    /// Initialize local storage fallback
    #[cfg(feature = "wasm")]
    fn initialize_local_storage(&mut self) -> Result<(), SynapticError> {
        if !self.config.enable_local_storage_fallback {
            return Ok(());
        }

        let window = window().ok_or_else(|| SynapticError::ProcessingError("No window object available".to_string()))?;

        self.local_storage = window
            .local_storage()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get local storage: {:?}", e)))?;

        Ok(())
    }

    /// Initialize web worker for background operations
    #[cfg(feature = "wasm")]
    async fn initialize_web_worker(&self) -> Result<(), SynapticError> {
        if !self.config.enable_web_worker {
            return Ok(());
        }

        let mut worker_manager = WebWorkerManager::new()?;
        worker_manager.initialize().await?;

        *self.worker_manager.borrow_mut() = Some(worker_manager);

        tracing::info!("Web worker initialized for background operations");
        Ok(())
    }

    /// Store data in IndexedDB
    #[cfg(feature = "wasm")]
    async fn store_indexeddb(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let db = self.db.as_ref().ok_or_else(|| SynapticError::ProcessingError("IndexedDB not initialized".to_string()))?;
        
        let transaction = db
            .transaction_with_str_and_mode(&self.config.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create transaction: {:?}", e)))?;
        
        let store = transaction
            .object_store(&self.config.store_name)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get object store: {:?}", e)))?;

        // Convert data to Uint8Array
        let uint8_array = Uint8Array::new_with_length(data.len() as u32);
        uint8_array.copy_from(data);

        // Store data
        let request = store
            .put_with_key(&uint8_array, &JsValue::from_str(key))
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to store data: {:?}", e)))?;

        // Wait for completion
        JsFuture::from(request).await
            .map_err(|e| SynapticError::ProcessingError(format!("Store operation failed: {:?}", e)))?;

        Ok(())
    }

    /// Retrieve data from IndexedDB
    #[cfg(feature = "wasm")]
    async fn retrieve_indexeddb(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        let db = self.db.as_ref().ok_or_else(|| SynapticError::ProcessingError("IndexedDB not initialized".to_string()))?;
        
        let transaction = db
            .transaction_with_str(&self.config.store_name)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create transaction: {:?}", e)))?;
        
        let store = transaction
            .object_store(&self.config.store_name)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get object store: {:?}", e)))?;

        let request = store
            .get(&JsValue::from_str(key))
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to get data: {:?}", e)))?;

        let result = JsFuture::from(request).await
            .map_err(|e| SynapticError::ProcessingError(format!("Retrieve operation failed: {:?}", e)))?;

        if result.is_undefined() {
            return Ok(None);
        }

        // Convert Uint8Array back to Vec<u8>
        let uint8_array: Uint8Array = result.dyn_into()
            .map_err(|e| SynapticError::ProcessingError(format!("Invalid data format: {:?}", e)))?;
        
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        Ok(Some(data))
    }

    /// Store data in local storage (fallback)
    #[cfg(feature = "wasm")]
    fn store_local_storage(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let storage = self.local_storage.as_ref().ok_or_else(|| SynapticError::ProcessingError("Local storage not available".to_string()))?;
        
        // Encode data as base64
        let encoded = base64::encode(data);
        
        storage
            .set_item(&format!("synaptic_{}", key), &encoded)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to store in local storage: {:?}", e)))?;

        Ok(())
    }

    /// Retrieve data from local storage (fallback)
    #[cfg(feature = "wasm")]
    fn retrieve_local_storage(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        let storage = self.local_storage.as_ref().ok_or_else(|| SynapticError::ProcessingError("Local storage not available".to_string()))?;
        
        let encoded = storage
            .get_item(&format!("synaptic_{}", key))
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to retrieve from local storage: {:?}", e)))?;

        if let Some(encoded) = encoded {
            let data = base64::decode(&encoded)
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to decode data: {}", e)))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Compress data if enabled
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, SynapticError> {
        if self.config.enable_compression && data.len() > 1024 {
            // Simple compression using deflate (in a real implementation, use a proper compression library)
            // For now, just return the original data
            Ok(data.to_vec())
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decompress data if needed
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, SynapticError> {
        // In a real implementation, detect if data is compressed and decompress
        Ok(data.to_vec())
    }

    /// Update memory cache
    fn update_cache(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        if !self.config.enable_memory_cache {
            return Ok(());
        }

        let mut cache = self.memory_cache.borrow_mut();

        // Check cache size limit
        let current_size: usize = cache.values().map(|v| v.len()).sum();
        if current_size + data.len() > self.config.max_cache_size {
            // Simple LRU eviction (remove first entry)
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key.to_string(), data.to_vec());
        Ok(())
    }

    /// Get browser performance information
    #[cfg(feature = "wasm")]
    fn get_browser_performance(&self) -> PerformanceProfile {
        let window = window();
        let performance = window.and_then(|w| w.performance());
        
        // Get memory information if available
        let memory_info = performance.and_then(|p| {
            js_sys::Reflect::get(&p, &JsValue::from_str("memory")).ok()
        });

        let (memory_score, cpu_score) = if let Some(memory) = memory_info {
            let used_heap = js_sys::Reflect::get(&memory, &JsValue::from_str("usedJSHeapSize"))
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            
            let total_heap = js_sys::Reflect::get(&memory, &JsValue::from_str("totalJSHeapSize"))
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            let memory_usage = used_heap / total_heap;
            let memory_score = (1.0 - memory_usage).max(0.0).min(1.0) as f32;
            
            // Estimate CPU score based on memory pressure
            let cpu_score = if memory_usage < 0.5 { 0.9 } else if memory_usage < 0.8 { 0.7 } else { 0.5 };
            
            (memory_score, cpu_score)
        } else {
            (0.8, 0.8) // Default values
        };

        PerformanceProfile {
            cpu_score,
            memory_score,
            storage_score: 0.7, // IndexedDB is generally good
            network_score: 0.9, // Browsers have good network capabilities
            battery_optimization: true, // Always optimize for battery in browsers
        }
    }
}

impl CrossPlatformAdapter for WasmAdapter {
    fn initialize(&mut self, config: &PlatformConfig) -> Result<(), SynapticError> {
        // Initialize storage backends
        #[cfg(feature = "wasm")]
        {
            // Initialize IndexedDB
            if config.storage_backends.contains(&StorageBackend::IndexedDB) {
                // Note: In a real implementation, this would be async
                // For now, we'll defer initialization to first use
            }

            // Initialize local storage fallback
            self.initialize_local_storage()?;

            // Initialize web worker (async operation deferred to first use)
            // The web worker will be initialized on first async operation
        }

        Ok(())
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        // Compress data if enabled
        let compressed_data = self.compress_data(data)?;

        // Try IndexedDB first
        #[cfg(feature = "wasm")]
        if self.db.is_some() {
            // Note: This should be async in a real implementation
            // For now, we'll use local storage as fallback
        }

        // Fallback to local storage
        #[cfg(feature = "wasm")]
        if self.local_storage.is_some() {
            self.store_local_storage(key, &compressed_data)?;
        }

        // Update cache using safe interior mutability
        let _ = self.update_cache(key, data);

        Ok(())
    }

    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        // Check cache first
        if self.config.enable_memory_cache {
            let cache = self.memory_cache.borrow();
            if let Some(data) = cache.get(key) {
                return Ok(Some(data.clone()));
            }
        }

        // Try IndexedDB
        #[cfg(feature = "wasm")]
        if self.db.is_some() {
            // Note: This should be async in a real implementation
        }

        // Fallback to local storage
        #[cfg(feature = "wasm")]
        if let Some(data) = self.retrieve_local_storage(key)? {
            let decompressed = self.decompress_data(&data)?;
            return Ok(Some(decompressed));
        }

        Ok(None)
    }

    fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        let mut deleted = false;

        // Remove from cache using safe interior mutability
        if self.config.enable_memory_cache {
            let mut cache = self.memory_cache.borrow_mut();
            deleted = cache.remove(key).is_some();
        }

        // Remove from local storage
        #[cfg(feature = "wasm")]
        if let Some(ref storage) = self.local_storage {
            storage
                .remove_item(&format!("synaptic_{}", key))
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to delete from local storage: {:?}", e)))?;
            deleted = true;
        }

        Ok(deleted)
    }

    fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        let mut keys = Vec::new();

        // Get keys from cache
        {
            let cache = self.memory_cache.borrow();
            keys.extend(cache.keys().cloned());
        }

        // Get keys from local storage
        #[cfg(feature = "wasm")]
        if let Some(ref storage) = self.local_storage {
            let length = storage.length()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to get storage length: {:?}", e)))?;

            for i in 0..length {
                if let Ok(Some(key)) = storage.key(i) {
                    if let Some(synaptic_key) = key.strip_prefix("synaptic_") {
                        keys.push(synaptic_key.to_string());
                    }
                }
            }
        }

        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        let keys = self.list_keys()?;
        let item_count = keys.len();
        
        let mut total_size = 0;
        for key in &keys {
            if let Ok(Some(data)) = self.retrieve(key) {
                total_size += data.len();
            }
        }

        let average_item_size = if item_count > 0 {
            total_size / item_count
        } else {
            0
        };

        Ok(StorageStats {
            used_storage: total_size,
            available_storage: 50 * 1024 * 1024, // Estimate 50MB available
            item_count,
            average_item_size,
            backend: StorageBackend::IndexedDB,
        })
    }

    fn supports_feature(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::FileSystemAccess => false, // Limited in browsers
            PlatformFeature::NetworkAccess => true,
            PlatformFeature::BackgroundProcessing => false, // Limited in browsers
            PlatformFeature::PushNotifications => true,
            PlatformFeature::HardwareAcceleration => true,
            PlatformFeature::MultiThreading => false, // Web Workers are different
            PlatformFeature::LargeMemoryAllocation => false, // Limited in browsers
        }
    }

    fn get_platform_info(&self) -> PlatformInfo {
        #[cfg(feature = "wasm")]
        let performance_profile = self.get_browser_performance();
        #[cfg(not(feature = "wasm"))]
        let performance_profile = PerformanceProfile {
            cpu_score: 0.8,
            memory_score: 0.7,
            storage_score: 0.7,
            network_score: 0.9,
            battery_optimization: true,
        };

        PlatformInfo {
            platform: Platform::WebAssembly,
            version: "1.0.0".to_string(),
            available_memory: 50 * 1024 * 1024, // 50MB estimate
            available_storage: 100 * 1024 * 1024, // 100MB estimate
            supported_features: vec![
                PlatformFeature::NetworkAccess,
                PlatformFeature::PushNotifications,
                PlatformFeature::HardwareAcceleration,
            ],
            performance_profile,
        }
    }
}

impl WasmAdapter {
    /// Async store operation using web worker when available
    pub async fn store_async(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        // Try web worker first if enabled
        if self.config.enable_web_worker {
            // Initialize worker if not already done
            if self.worker_manager.borrow().is_none() {
                #[cfg(feature = "wasm")]
                self.initialize_web_worker().await?;
            }

            if let Some(ref worker) = *self.worker_manager.borrow() {
                return worker.store_async(key, data, self.config.enable_compression).await;
            }
        }

        // Fallback to synchronous operation
        self.store(key, data)
    }

    /// Async retrieve operation using web worker when available
    pub async fn retrieve_async(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        // Try web worker first if enabled
        if self.config.enable_web_worker {
            // Initialize worker if not already done
            if self.worker_manager.borrow().is_none() {
                #[cfg(feature = "wasm")]
                self.initialize_web_worker().await?;
            }

            if let Some(ref worker) = *self.worker_manager.borrow() {
                return worker.retrieve_async(key).await;
            }
        }

        // Fallback to synchronous operation
        self.retrieve(key)
    }

    /// Async delete operation using web worker when available
    pub async fn delete_async(&self, key: &str) -> Result<bool, SynapticError> {
        // Try web worker first if enabled
        if self.config.enable_web_worker {
            // Initialize worker if not already done
            if self.worker_manager.borrow().is_none() {
                #[cfg(feature = "wasm")]
                self.initialize_web_worker().await?;
            }

            if let Some(ref worker) = *self.worker_manager.borrow() {
                return worker.delete_async(key).await;
            }
        }

        // Fallback to synchronous operation
        self.delete(key)
    }

    /// Async list keys operation using web worker when available
    pub async fn list_keys_async(&self) -> Result<Vec<String>, SynapticError> {
        // Try web worker first if enabled
        if self.config.enable_web_worker {
            // Initialize worker if not already done
            if self.worker_manager.borrow().is_none() {
                #[cfg(feature = "wasm")]
                self.initialize_web_worker().await?;
            }

            if let Some(ref worker) = *self.worker_manager.borrow() {
                return worker.list_keys_async().await;
            }
        }

        // Fallback to synchronous operation
        self.list_keys()
    }

    /// Search operation using web worker
    pub async fn search_async(&self, query: &str, limit: usize) -> Result<Vec<wasm_worker::SearchResult>, SynapticError> {
        // Initialize worker if not already done
        if self.worker_manager.borrow().is_none() {
            #[cfg(feature = "wasm")]
            self.initialize_web_worker().await?;
        }

        if let Some(ref worker) = *self.worker_manager.borrow() {
            return worker.search_async(query, limit).await;
        }

        Err(SynapticError::ProcessingError("Web worker not available for search".to_string()))
    }

    /// Get web worker statistics
    pub fn get_worker_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        stats.insert("worker_enabled".to_string(), self.config.enable_web_worker.to_string());
        stats.insert("worker_initialized".to_string(), self.worker_manager.borrow().is_some().to_string());
        stats.insert("worker_timeout".to_string(), format!("{}s", self.config.worker_timeout_seconds));

        stats
    }
}
