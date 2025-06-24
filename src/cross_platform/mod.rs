//! # Cross-Platform Support
//!
//! Cross-platform capabilities for the Synaptic memory system, including WebAssembly,
//! mobile platforms, and offline-first functionality.

use crate::error::MemoryError as SynapticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "mobile")]
pub mod mobile;

pub mod offline;
pub mod sync;

/// Cross-platform configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformConfig {
    /// Enable WebAssembly support
    pub enable_wasm: bool,
    
    /// Enable mobile platform support
    pub enable_mobile: bool,
    
    /// Enable offline-first capabilities
    pub enable_offline: bool,
    
    /// Enable cross-platform synchronization
    pub enable_sync: bool,
    
    /// Platform-specific configurations
    pub platform_configs: HashMap<Platform, PlatformConfig>,
}

impl Default for CrossPlatformConfig {
    fn default() -> Self {
        Self {
            enable_wasm: true,
            enable_mobile: true,
            enable_offline: true,
            enable_sync: true,
            platform_configs: HashMap::new(),
        }
    }
}

/// Supported platforms
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    /// Web browsers via WebAssembly
    WebAssembly,
    /// iOS mobile platform
    iOS,
    /// Android mobile platform
    Android,
    /// Desktop platforms (Windows, macOS, Linux)
    Desktop,
    /// Server/cloud platforms
    Server,
}

/// Platform-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,
    
    /// Enable local storage
    pub enable_local_storage: bool,
    
    /// Enable network synchronization
    pub enable_network_sync: bool,
    
    /// Storage backend preferences
    pub storage_backends: Vec<StorageBackend>,
    
    /// Performance optimization settings
    pub performance_settings: PerformanceSettings,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 100 * 1024 * 1024, // 100MB default
            enable_local_storage: true,
            enable_network_sync: true,
            storage_backends: vec![StorageBackend::Memory, StorageBackend::IndexedDB],
            performance_settings: PerformanceSettings::default(),
        }
    }
}

/// Available storage backends
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StorageBackend {
    /// In-memory storage
    Memory,
    /// Browser IndexedDB (WebAssembly)
    IndexedDB,
    /// SQLite database
    SQLite,
    /// Core Data (iOS)
    CoreData,
    /// Room database (Android)
    Room,
    /// File system storage
    FileSystem,
}

/// Performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Enable lazy loading
    pub lazy_loading: bool,
    
    /// Enable compression
    pub compression: bool,
    
    /// Enable caching
    pub caching: bool,
    
    /// Batch size for operations
    pub batch_size: usize,
    
    /// Worker thread count
    pub worker_threads: usize,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            lazy_loading: true,
            compression: true,
            caching: true,
            batch_size: 100,
            worker_threads: 4,
        }
    }
}

/// Cross-platform memory adapter
pub trait CrossPlatformAdapter: Send + Sync {
    /// Initialize the adapter for the target platform
    fn initialize(&mut self, config: &PlatformConfig) -> Result<(), SynapticError>;
    
    /// Store data on the platform
    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError>;
    
    /// Retrieve data from the platform
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError>;
    
    /// Delete data from the platform
    fn delete(&self, key: &str) -> Result<bool, SynapticError>;
    
    /// List all keys
    fn list_keys(&self) -> Result<Vec<String>, SynapticError>;
    
    /// Get storage statistics
    fn get_stats(&self) -> Result<StorageStats, SynapticError>;
    
    /// Check if platform supports feature
    fn supports_feature(&self, feature: PlatformFeature) -> bool;
    
    /// Get platform information
    fn get_platform_info(&self) -> PlatformInfo;
}

/// Platform feature capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlatformFeature {
    /// Local file system access
    FileSystemAccess,
    /// Network connectivity
    NetworkAccess,
    /// Background processing
    BackgroundProcessing,
    /// Push notifications
    PushNotifications,
    /// Hardware acceleration
    HardwareAcceleration,
    /// Multi-threading
    MultiThreading,
    /// Large memory allocation
    LargeMemoryAllocation,
}

/// Platform information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Platform type
    pub platform: Platform,
    
    /// Platform version
    pub version: String,
    
    /// Available memory (bytes)
    pub available_memory: usize,
    
    /// Available storage (bytes)
    pub available_storage: usize,
    
    /// Supported features
    pub supported_features: Vec<PlatformFeature>,
    
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Performance profile for the platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// CPU performance score (0.0 - 1.0)
    pub cpu_score: f32,
    
    /// Memory performance score (0.0 - 1.0)
    pub memory_score: f32,
    
    /// Storage performance score (0.0 - 1.0)
    pub storage_score: f32,
    
    /// Network performance score (0.0 - 1.0)
    pub network_score: f32,
    
    /// Battery optimization needed
    pub battery_optimization: bool,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total storage used (bytes)
    pub used_storage: usize,
    
    /// Available storage (bytes)
    pub available_storage: usize,
    
    /// Number of stored items
    pub item_count: usize,
    
    /// Average item size (bytes)
    pub average_item_size: usize,
    
    /// Storage backend in use
    pub backend: StorageBackend,
}

/// Cross-platform memory manager
pub struct CrossPlatformMemoryManager {
    /// Platform adapters
    adapters: HashMap<Platform, Box<dyn CrossPlatformAdapter>>,
    
    /// Current platform
    current_platform: Platform,
    
    /// Configuration
    config: CrossPlatformConfig,
    
    /// Synchronization manager
    sync_manager: Option<sync::SyncManager>,
}

impl CrossPlatformMemoryManager {
    /// Create a new cross-platform memory manager
    pub fn new(config: CrossPlatformConfig) -> Result<Self, SynapticError> {
        let current_platform = Self::detect_platform();
        
        let mut manager = Self {
            adapters: HashMap::new(),
            current_platform,
            config: config.clone(),
            sync_manager: None,
        };

        // Initialize platform-specific adapters
        manager.initialize_adapters()?;

        // Initialize sync manager if enabled
        if config.enable_sync {
            manager.sync_manager = Some(sync::SyncManager::new(sync::SyncConfig::default())?);
        }

        Ok(manager)
    }

    /// Detect the current platform
    fn detect_platform() -> Platform {
        #[cfg(target_arch = "wasm32")]
        {
            Platform::WebAssembly
        }
        #[cfg(target_os = "ios")]
        {
            Platform::iOS
        }
        #[cfg(target_os = "android")]
        {
            Platform::Android
        }
        #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
        {
            Platform::Desktop
        }
        #[cfg(not(any(target_arch = "wasm32", target_os = "ios", target_os = "android", target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Platform::Server
        }
    }

    /// Initialize platform-specific adapters
    fn initialize_adapters(&mut self) -> Result<(), SynapticError> {
        // Initialize WebAssembly adapter
        #[cfg(feature = "wasm")]
        if self.config.enable_wasm {
            let adapter = wasm::WasmAdapter::new()?;
            self.adapters.insert(Platform::WebAssembly, Box::new(adapter));
        }

        // Initialize mobile adapters
        #[cfg(feature = "mobile")]
        if self.config.enable_mobile {
            #[cfg(target_os = "ios")]
            {
                let adapter = mobile::iOSAdapter::new()?;
                self.adapters.insert(Platform::iOS, Box::new(adapter));
            }

            #[cfg(target_os = "android")]
            {
                let adapter = mobile::AndroidAdapter::new()?;
                self.adapters.insert(Platform::Android, Box::new(adapter));
            }
        }

        // Initialize offline adapter
        if self.config.enable_offline {
            let adapter = offline::OfflineAdapter::new()?;
            self.adapters.insert(self.current_platform.clone(), Box::new(adapter));
        }

        Ok(())
    }

    /// Get the adapter for the current platform
    fn get_current_adapter(&self) -> Result<&dyn CrossPlatformAdapter, SynapticError> {
        self.adapters
            .get(&self.current_platform)
            .map(|adapter| adapter.as_ref())
            .ok_or_else(|| SynapticError::ProcessingError(format!("No adapter available for platform: {:?}", self.current_platform)))
    }

    /// Store data using the current platform adapter
    pub fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let adapter = self.get_current_adapter()?;
        adapter.store(key, data)?;

        // Sync if enabled
        if let Some(ref sync_manager) = self.sync_manager {
            sync_manager.queue_sync_operation(sync::SyncOperation::Store {
                key: key.to_string(),
                data: data.to_vec(),
            })?;
        }

        Ok(())
    }

    /// Retrieve data using the current platform adapter
    pub fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        let adapter = self.get_current_adapter()?;
        adapter.retrieve(key)
    }

    /// Delete data using the current platform adapter
    pub fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        let adapter = self.get_current_adapter()?;
        let result = adapter.delete(key)?;

        // Sync if enabled
        if let Some(ref sync_manager) = self.sync_manager {
            sync_manager.queue_sync_operation(sync::SyncOperation::Delete {
                key: key.to_string(),
            })?;
        }

        Ok(result)
    }

    /// Get platform information
    pub fn get_platform_info(&self) -> Result<PlatformInfo, SynapticError> {
        let adapter = self.get_current_adapter()?;
        Ok(adapter.get_platform_info())
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> Result<StorageStats, SynapticError> {
        let adapter = self.get_current_adapter()?;
        adapter.get_stats()
    }

    /// Check if a feature is supported on the current platform
    pub fn supports_feature(&self, feature: PlatformFeature) -> Result<bool, SynapticError> {
        let adapter = self.get_current_adapter()?;
        Ok(adapter.supports_feature(feature))
    }

    /// Optimize for the current platform
    pub fn optimize_for_platform(&mut self) -> Result<(), SynapticError> {
        let platform_info = self.get_platform_info()?;
        
        // Adjust configuration based on platform capabilities
        if let Some(platform_config) = self.config.platform_configs.get_mut(&self.current_platform) {
            // Adjust memory usage based on available memory
            if platform_info.available_memory < 512 * 1024 * 1024 { // Less than 512MB
                platform_config.max_memory_usage = platform_info.available_memory / 4; // Use 25% of available memory
                platform_config.performance_settings.batch_size = 50; // Smaller batches
                platform_config.performance_settings.worker_threads = 2; // Fewer threads
            }

            // Enable battery optimization for mobile platforms
            if matches!(self.current_platform, Platform::iOS | Platform::Android) {
                platform_config.performance_settings.lazy_loading = true;
                platform_config.performance_settings.compression = true;
            }

            // Optimize for WebAssembly constraints
            if matches!(self.current_platform, Platform::WebAssembly) {
                platform_config.max_memory_usage = 50 * 1024 * 1024; // 50MB limit
                platform_config.performance_settings.worker_threads = 1; // Single-threaded
                platform_config.storage_backends = vec![StorageBackend::IndexedDB, StorageBackend::Memory];
            }
        }

        Ok(())
    }

    /// Synchronize data across platforms
    pub async fn sync(&self) -> Result<(), SynapticError> {
        if let Some(ref sync_manager) = self.sync_manager {
            sync_manager.sync().await?;
        }
        Ok(())
    }
}

/// Result type for cross-platform operations
pub type CrossPlatformResult<T> = Result<T, SynapticError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = CrossPlatformMemoryManager::detect_platform();
        // Platform detection should work on any target
        assert!(matches!(platform, Platform::Desktop | Platform::Server | Platform::WebAssembly | Platform::iOS | Platform::Android));
    }

    #[test]
    fn test_cross_platform_config() {
        let config = CrossPlatformConfig::default();
        assert!(config.enable_wasm);
        assert!(config.enable_mobile);
        assert!(config.enable_offline);
        assert!(config.enable_sync);
    }

    #[test]
    fn test_platform_config() {
        let config = PlatformConfig::default();
        assert_eq!(config.max_memory_usage, 100 * 1024 * 1024);
        assert!(config.enable_local_storage);
        assert!(config.enable_network_sync);
        assert!(!config.storage_backends.is_empty());
    }

    #[test]
    fn test_performance_settings() {
        let settings = PerformanceSettings::default();
        assert!(settings.lazy_loading);
        assert!(settings.compression);
        assert!(settings.caching);
        assert_eq!(settings.batch_size, 100);
        assert_eq!(settings.worker_threads, 4);
    }
}
