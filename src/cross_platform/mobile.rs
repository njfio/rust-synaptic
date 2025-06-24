//! # Mobile Platform Support
//!
//! Mobile adapters for iOS and Android platforms with platform-specific optimizations,
//! battery management, and native storage integration.

use super::{
    CrossPlatformAdapter, PlatformConfig, PlatformFeature, PlatformInfo, Platform,
    PerformanceProfile, StorageBackend, StorageStats,
};
use crate::error::MemoryError as SynapticError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::path::PathBuf;
use std::fs;

// Platform-specific imports
#[cfg(feature = "mobile")]
use {
    #[cfg(target_os = "ios")]
    swift_bridge,
    #[cfg(target_os = "android")]
    jni::{JNIEnv, JavaVM, objects::{JClass, JString, JObject}},
};

/// Mobile-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Enable battery optimization
    pub enable_battery_optimization: bool,

    /// Enable background sync
    pub enable_background_sync: bool,

    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_pressure_threshold: f32,

    /// Cache size limit (bytes)
    pub cache_size_limit: usize,

    /// Enable compression for storage
    pub enable_compression: bool,

    /// Sync interval (seconds)
    pub sync_interval_seconds: u64,

    /// Enable offline mode
    pub enable_offline_mode: bool,

    /// Data directory path
    pub data_directory: Option<PathBuf>,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            enable_battery_optimization: true,
            enable_background_sync: true,
            memory_pressure_threshold: 0.8,
            cache_size_limit: 50 * 1024 * 1024, // 50MB
            enable_compression: true,
            sync_interval_seconds: 300, // 5 minutes
            enable_offline_mode: true,
            data_directory: None,
        }
    }
}

/// Mobile storage backend
#[derive(Debug)]
pub struct MobileStorage {
    /// In-memory cache for fast access
    cache: Arc<RwLock<HashMap<String, (Vec<u8>, Instant)>>>,

    /// Persistent storage path
    storage_path: PathBuf,

    /// Configuration
    config: MobileConfig,

    /// Memory pressure monitor
    memory_pressure: Arc<Mutex<f32>>,

    /// Last cleanup time
    last_cleanup: Arc<Mutex<Instant>>,
}

impl MobileStorage {
    pub fn new(config: MobileConfig) -> Result<Self, SynapticError> {
        let storage_path = config.data_directory.clone()
            .unwrap_or_else(|| {
                #[cfg(target_os = "ios")]
                {
                    // iOS Documents directory
                    dirs::document_dir()
                        .unwrap_or_else(|| PathBuf::from("/tmp"))
                        .join("synaptic")
                }
                #[cfg(target_os = "android")]
                {
                    // Android internal storage
                    PathBuf::from("/data/data/com.synaptic.memory/files")
                }
                #[cfg(not(any(target_os = "ios", target_os = "android")))]
                {
                    dirs::data_dir()
                        .unwrap_or_else(|| PathBuf::from("/tmp"))
                        .join("synaptic")
                }
            });

        // Create storage directory
        fs::create_dir_all(&storage_path)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to create storage directory: {}", e)))?;

        Ok(Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            storage_path,
            config,
            memory_pressure: Arc::new(Mutex::new(0.0)),
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
        })
    }

    /// Store data with automatic compression and caching
    pub fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        // Compress data if enabled
        let stored_data = if self.config.enable_compression {
            self.compress_data(data)?
        } else {
            data.to_vec()
        };

        // Store to persistent storage
        let file_path = self.storage_path.join(format!("{}.dat", key));
        fs::write(&file_path, &stored_data)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to write to storage: {}", e)))?;

        // Update cache
        self.update_cache(key, data)?;

        // Check memory pressure and cleanup if needed
        self.check_memory_pressure()?;

        Ok(())
    }

    /// Retrieve data from cache or storage
    pub fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        // Check cache first
        {
            let cache = self.cache.read()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cache lock: {}", e)))?;

            if let Some((data, _)) = cache.get(key) {
                return Ok(Some(data.clone()));
            }
        }

        // Load from persistent storage
        let file_path = self.storage_path.join(format!("{}.dat", key));
        if !file_path.exists() {
            return Ok(None);
        }

        let stored_data = fs::read(&file_path)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to read from storage: {}", e)))?;

        // Decompress if needed
        let data = if self.config.enable_compression {
            self.decompress_data(&stored_data)?
        } else {
            stored_data
        };

        // Update cache
        self.update_cache(key, &data)?;

        Ok(Some(data))
    }

    /// Delete data from cache and storage
    pub fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        // Remove from cache
        let cache_existed = {
            let mut cache = self.cache.write()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cache lock: {}", e)))?;
            cache.remove(key).is_some()
        };

        // Remove from persistent storage
        let file_path = self.storage_path.join(format!("{}.dat", key));
        let storage_existed = if file_path.exists() {
            fs::remove_file(&file_path)
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to delete from storage: {}", e)))?;
            true
        } else {
            false
        };

        Ok(cache_existed || storage_existed)
    }

    /// List all keys
    pub fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        let mut keys = Vec::new();

        // Get keys from storage directory
        let entries = fs::read_dir(&self.storage_path)
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to read storage directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to read directory entry: {}", e)))?;

            if let Some(file_name) = entry.file_name().to_str() {
                if let Some(key) = file_name.strip_suffix(".dat") {
                    keys.push(key.to_string());
                }
            }
        }

        keys.sort();
        Ok(keys)
    }

    /// Update cache with size and time limits
    fn update_cache(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let mut cache = self.cache.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cache lock: {}", e)))?;

        // Check cache size limit
        let current_size: usize = cache.values().map(|(data, _)| data.len()).sum();
        if current_size + data.len() > self.config.cache_size_limit {
            // Remove oldest entries
            let mut entries: Vec<_> = cache.iter().map(|(k, (_, time))| (k.clone(), *time)).collect();
            entries.sort_by_key(|(_, time)| *time);

            let mut removed_size = 0;
            for (old_key, _) in entries {
                if let Some((old_data, _)) = cache.remove(&old_key) {
                    removed_size += old_data.len();
                    if current_size - removed_size + data.len() <= self.config.cache_size_limit {
                        break;
                    }
                }
            }
        }

        cache.insert(key.to_string(), (data.to_vec(), Instant::now()));
        Ok(())
    }

    /// Simple compression using flate2
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, SynapticError> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data)
            .map_err(|e| SynapticError::ProcessingError(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| SynapticError::ProcessingError(format!("Compression finalization failed: {}", e)))
    }

    /// Simple decompression using flate2
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, SynapticError> {
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| SynapticError::ProcessingError(format!("Decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    /// Check memory pressure and cleanup if needed
    fn check_memory_pressure(&self) -> Result<(), SynapticError> {
        let current_pressure = self.get_memory_pressure();

        {
            let mut pressure = self.memory_pressure.lock()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire pressure lock: {}", e)))?;
            *pressure = current_pressure;
        }

        if current_pressure > self.config.memory_pressure_threshold {
            self.cleanup_cache()?;
        }

        Ok(())
    }

    /// Get current memory pressure (0.0 - 1.0)
    fn get_memory_pressure(&self) -> f32 {
        // Simple heuristic based on cache size
        let cache_size: usize = {
            let cache = self.cache.read().unwrap_or_else(|_| {
                // If lock is poisoned, return empty map
                return 0;
            });
            cache.values().map(|(data, _)| data.len()).sum()
        };

        (cache_size as f32) / (self.config.cache_size_limit as f32)
    }

    /// Cleanup cache to reduce memory pressure
    fn cleanup_cache(&self) -> Result<(), SynapticError> {
        let mut last_cleanup = self.last_cleanup.lock()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cleanup lock: {}", e)))?;

        // Only cleanup if enough time has passed
        if last_cleanup.elapsed() < Duration::from_secs(60) {
            return Ok(());
        }

        let mut cache = self.cache.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cache lock: {}", e)))?;

        // Remove oldest 50% of entries
        let mut entries: Vec<_> = cache.iter().map(|(k, (_, time))| (k.clone(), *time)).collect();
        entries.sort_by_key(|(_, time)| *time);

        let remove_count = entries.len() / 2;
        for (key, _) in entries.into_iter().take(remove_count) {
            cache.remove(&key);
        }

        *last_cleanup = Instant::now();
        tracing::info!("Cache cleanup completed, removed {} entries", remove_count);

        Ok(())
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        let keys = self.list_keys()?;
        let item_count = keys.len();

        let mut total_size = 0;
        for key in &keys {
            let file_path = self.storage_path.join(format!("{}.dat", key));
            if let Ok(metadata) = fs::metadata(&file_path) {
                total_size += metadata.len() as usize;
            }
        }

        let cache_size = {
            let cache = self.cache.read()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire cache lock: {}", e)))?;
            cache.len()
        };

        let average_item_size = if item_count > 0 {
            total_size / item_count
        } else {
            0
        };

        Ok(StorageStats {
            used_storage: total_size,
            available_storage: self.get_available_storage(),
            item_count,
            average_item_size,
            backend: self.get_storage_backend(),
        })
    }

    /// Get available storage space
    fn get_available_storage(&self) -> usize {
        // Platform-specific implementation would query actual available space
        // For now, return a conservative estimate
        1024 * 1024 * 1024 // 1GB
    }

    /// Get the appropriate storage backend for the platform
    fn get_storage_backend(&self) -> StorageBackend {
        #[cfg(target_os = "ios")]
        return StorageBackend::CoreData;

        #[cfg(target_os = "android")]
        return StorageBackend::Room;

        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        return StorageBackend::FileSystem;
    }
}

/// iOS-specific adapter
#[cfg(target_os = "ios")]
pub struct iOSAdapter {
    storage: MobileStorage,
    config: Option<PlatformConfig>,
    mobile_config: MobileConfig,
    battery_optimization: bool,
}

/// Android-specific adapter
#[cfg(target_os = "android")]
pub struct AndroidAdapter {
    storage: MobileStorage,
    config: Option<PlatformConfig>,
    mobile_config: MobileConfig,
    battery_optimization: bool,
    #[cfg(feature = "mobile")]
    jvm: Option<Arc<JavaVM>>,
}

/// Generic mobile adapter for other mobile platforms
pub struct GenericMobileAdapter {
    storage: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    config: Option<PlatformConfig>,
    stats: Arc<RwLock<StorageStats>>,
    platform: Platform,
    battery_optimization: bool,
}

#[cfg(target_os = "ios")]
impl iOSAdapter {
    pub fn new() -> Result<Self, SynapticError> {
        let mobile_config = MobileConfig::default();
        let storage = MobileStorage::new(mobile_config.clone())?;

        Ok(Self {
            storage,
            config: None,
            mobile_config,
            battery_optimization: true,
        })
    }

    pub fn new_with_config(mobile_config: MobileConfig) -> Result<Self, SynapticError> {
        let storage = MobileStorage::new(mobile_config.clone())?;

        Ok(Self {
            storage,
            config: None,
            mobile_config,
            battery_optimization: mobile_config.enable_battery_optimization,
        })
    }

    /// Initialize iOS-specific features
    fn initialize_ios_features(&mut self) -> Result<(), SynapticError> {
        // Initialize Core Data integration
        self.setup_core_data_integration()?;

        // Setup background app refresh handling
        self.setup_background_refresh()?;

        // Configure memory pressure handling
        self.setup_memory_pressure_handling()?;

        // Setup iOS-specific optimizations
        self.setup_ios_optimizations()?;

        Ok(())
    }

    fn setup_core_data_integration(&self) -> Result<(), SynapticError> {
        // In a real implementation, this would use Swift bridge to integrate with Core Data
        // For now, we simulate Core Data behavior with file-based persistence

        #[cfg(feature = "mobile")]
        {
            // This would call Swift code via swift-bridge
            // swift_bridge::setup_core_data_stack();
        }

        tracing::info!("iOS Core Data integration initialized with file-based persistence");
        Ok(())
    }

    fn setup_background_refresh(&self) -> Result<(), SynapticError> {
        // Setup background app refresh handling
        // In a real implementation, this would register for background refresh notifications

        if self.mobile_config.enable_background_sync {
            tracing::info!("iOS background refresh configured for sync interval: {}s",
                         self.mobile_config.sync_interval_seconds);
        }

        Ok(())
    }

    fn setup_memory_pressure_handling(&self) -> Result<(), SynapticError> {
        // Setup iOS memory pressure notifications
        // In a real implementation, this would register for memory pressure notifications

        tracing::info!("iOS memory pressure handling configured with threshold: {}",
                     self.mobile_config.memory_pressure_threshold);
        Ok(())
    }

    fn setup_ios_optimizations(&self) -> Result<(), SynapticError> {
        // iOS-specific optimizations
        if self.battery_optimization {
            // Reduce background activity
            // Optimize for battery life
            tracing::info!("iOS battery optimizations enabled");
        }

        // Configure for iOS memory constraints
        if self.mobile_config.cache_size_limit > 100 * 1024 * 1024 {
            tracing::warn!("Cache size limit may be too large for iOS devices");
        }

        Ok(())
    }

    fn get_ios_performance_profile(&self) -> PerformanceProfile {
        // iOS-specific performance characteristics
        PerformanceProfile {
            cpu_score: 0.9, // iOS devices generally have good CPUs
            memory_score: 0.7, // Memory is more constrained
            storage_score: 0.8, // Good storage performance
            network_score: 0.9, // Excellent network capabilities
            battery_optimization: true,
        }
    }
}

#[cfg(target_os = "android")]
impl AndroidAdapter {
    pub fn new() -> Result<Self, SynapticError> {
        let mobile_config = MobileConfig::default();
        let storage = MobileStorage::new(mobile_config.clone())?;

        Ok(Self {
            storage,
            config: None,
            mobile_config,
            battery_optimization: true,
            #[cfg(feature = "mobile")]
            jvm: None,
        })
    }

    pub fn new_with_config(mobile_config: MobileConfig) -> Result<Self, SynapticError> {
        let storage = MobileStorage::new(mobile_config.clone())?;

        Ok(Self {
            storage,
            config: None,
            mobile_config,
            battery_optimization: mobile_config.enable_battery_optimization,
            #[cfg(feature = "mobile")]
            jvm: None,
        })
    }

    /// Initialize JNI integration for Android
    #[cfg(feature = "mobile")]
    pub fn initialize_jni(&mut self, jvm: Arc<JavaVM>) -> Result<(), SynapticError> {
        self.jvm = Some(jvm);
        tracing::info!("Android JNI integration initialized");
        Ok(())
    }

    /// Initialize Android-specific features
    fn initialize_android_features(&mut self) -> Result<(), SynapticError> {
        // Initialize SQLite integration
        self.setup_sqlite_integration()?;

        // Setup doze mode handling
        self.setup_doze_mode_handling()?;

        // Configure memory management
        self.setup_android_memory_management()?;

        // Setup Android-specific optimizations
        self.setup_android_optimizations()?;

        Ok(())
    }

    fn setup_sqlite_integration(&self) -> Result<(), SynapticError> {
        // In a real implementation, this would use JNI to integrate with Android SQLite/Room

        #[cfg(feature = "mobile")]
        if let Some(ref jvm) = self.jvm {
            // This would call Java/Kotlin code via JNI
            // let env = jvm.get_env()?;
            // Call Android Room database setup
        }

        tracing::info!("Android SQLite/Room integration initialized with file-based persistence");
        Ok(())
    }

    fn setup_doze_mode_handling(&self) -> Result<(), SynapticError> {
        // Setup Android doze mode handling
        // In a real implementation, this would register for doze mode changes

        if self.mobile_config.enable_background_sync {
            tracing::info!("Android doze mode handling configured for background sync");
        }

        Ok(())
    }

    fn setup_android_memory_management(&self) -> Result<(), SynapticError> {
        // Setup Android-specific memory management
        // In a real implementation, this would register for memory trim callbacks

        tracing::info!("Android memory management configured with cache limit: {} MB",
                     self.mobile_config.cache_size_limit / (1024 * 1024));
        Ok(())
    }

    fn setup_android_optimizations(&self) -> Result<(), SynapticError> {
        // Android-specific optimizations
        if self.battery_optimization {
            // Configure for doze mode and app standby
            tracing::info!("Android battery optimizations enabled for doze mode compatibility");
        }

        // Configure for Android memory management
        if self.mobile_config.enable_compression {
            tracing::info!("Compression enabled for Android storage optimization");
        }

        Ok(())
    }

    /// Get Android system information via JNI
    #[cfg(feature = "mobile")]
    fn get_android_system_info(&self) -> Result<HashMap<String, String>, SynapticError> {
        let mut info = HashMap::new();

        if let Some(ref jvm) = self.jvm {
            // In a real implementation, this would call Android APIs via JNI
            // let env = jvm.get_env()?;
            // Get system information like available memory, storage, etc.

            info.insert("platform".to_string(), "Android".to_string());
            info.insert("jni_available".to_string(), "true".to_string());
        } else {
            info.insert("jni_available".to_string(), "false".to_string());
        }

        Ok(info)
    }

    fn get_android_performance_profile(&self) -> PerformanceProfile {
        // Android-specific performance characteristics
        PerformanceProfile {
            cpu_score: 0.8, // Varies widely across Android devices
            memory_score: 0.6, // Often more constrained than iOS
            storage_score: 0.7, // Variable storage performance
            network_score: 0.9, // Good network capabilities
            battery_optimization: true,
        }
    }
}

impl GenericMobileAdapter {
    pub fn new(platform: Platform) -> Result<Self, SynapticError> {
        Ok(Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            config: None,
            stats: Arc::new(RwLock::new(StorageStats {
                used_storage: 0,
                available_storage: 1024 * 1024 * 1024, // 1GB estimate
                item_count: 0,
                average_item_size: 0,
                backend: StorageBackend::Memory,
            })),
            platform,
            battery_optimization: true,
        })
    }

    fn get_generic_mobile_performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            cpu_score: 0.7,
            memory_score: 0.6,
            storage_score: 0.7,
            network_score: 0.8,
            battery_optimization: true,
        }
    }
}

// Shared mobile functionality
trait MobileOptimizations {
    fn optimize_for_battery(&mut self) -> Result<(), SynapticError>;
    fn handle_memory_pressure(&mut self) -> Result<(), SynapticError>;
    fn configure_background_sync(&mut self) -> Result<(), SynapticError>;
}

impl MobileOptimizations for GenericMobileAdapter {
    fn optimize_for_battery(&mut self) -> Result<(), SynapticError> {
        self.battery_optimization = true;
        tracing::info!("Battery optimization enabled for mobile platform");
        Ok(())
    }

    fn handle_memory_pressure(&mut self) -> Result<(), SynapticError> {
        // Clear non-essential cached data
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;
        
        // In a real implementation, this would clear caches and temporary data
        let initial_size = storage.len();
        storage.retain(|key, _| !key.starts_with("cache_"));
        let cleared = initial_size - storage.len();
        
        tracing::info!("Memory pressure handled: cleared {} cached entries", cleared);
        Ok(())
    }

    fn configure_background_sync(&mut self) -> Result<(), SynapticError> {
        // Configure background synchronization for mobile
        tracing::info!("Background sync configured for mobile platform");
        Ok(())
    }
}

// Implementation of CrossPlatformAdapter for iOS
#[cfg(target_os = "ios")]
impl CrossPlatformAdapter for iOSAdapter {
    fn initialize(&mut self, config: &PlatformConfig) -> Result<(), SynapticError> {
        self.config = Some(config.clone());
        self.initialize_ios_features()?;
        Ok(())
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        self.storage.store(key, data)
    }

    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        self.storage.retrieve(key)
    }

    fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        self.storage.delete(key)
    }

    fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        self.storage.list_keys()
    }

    fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        self.storage.get_stats()
    }

    fn supports_feature(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::FileSystemAccess => true,
            PlatformFeature::NetworkAccess => true,
            PlatformFeature::BackgroundProcessing => true,
            PlatformFeature::PushNotifications => true,
            PlatformFeature::HardwareAcceleration => true,
            PlatformFeature::MultiThreading => true,
            PlatformFeature::LargeMemoryAllocation => false, // iOS has memory constraints
        }
    }

    fn get_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            platform: Platform::iOS,
            version: "1.0.0".to_string(),
            available_memory: 512 * 1024 * 1024, // 512MB estimate
            available_storage: 2 * 1024 * 1024 * 1024, // 2GB estimate
            supported_features: vec![
                PlatformFeature::FileSystemAccess,
                PlatformFeature::NetworkAccess,
                PlatformFeature::BackgroundProcessing,
                PlatformFeature::PushNotifications,
                PlatformFeature::HardwareAcceleration,
                PlatformFeature::MultiThreading,
            ],
            performance_profile: self.get_ios_performance_profile(),
        }
    }
}

// Implementation of CrossPlatformAdapter for Android
#[cfg(target_os = "android")]
impl CrossPlatformAdapter for AndroidAdapter {
    fn initialize(&mut self, config: &PlatformConfig) -> Result<(), SynapticError> {
        self.config = Some(config.clone());
        self.initialize_android_features()?;
        Ok(())
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        self.storage.store(key, data)
    }

    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        self.storage.retrieve(key)
    }

    fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        self.storage.delete(key)
    }

    fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        self.storage.list_keys()
    }

    fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        self.storage.get_stats()
    }

    fn supports_feature(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::FileSystemAccess => true,
            PlatformFeature::NetworkAccess => true,
            PlatformFeature::BackgroundProcessing => true,
            PlatformFeature::PushNotifications => true,
            PlatformFeature::HardwareAcceleration => true,
            PlatformFeature::MultiThreading => true,
            PlatformFeature::LargeMemoryAllocation => true, // Android generally allows larger allocations
        }
    }

    fn get_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            platform: Platform::Android,
            version: "1.0.0".to_string(),
            available_memory: 1024 * 1024 * 1024, // 1GB estimate
            available_storage: 4 * 1024 * 1024 * 1024, // 4GB estimate
            supported_features: vec![
                PlatformFeature::FileSystemAccess,
                PlatformFeature::NetworkAccess,
                PlatformFeature::BackgroundProcessing,
                PlatformFeature::PushNotifications,
                PlatformFeature::HardwareAcceleration,
                PlatformFeature::MultiThreading,
                PlatformFeature::LargeMemoryAllocation,
            ],
            performance_profile: self.get_android_performance_profile(),
        }
    }
}

// Implementation of CrossPlatformAdapter for GenericMobileAdapter
impl CrossPlatformAdapter for GenericMobileAdapter {
    fn initialize(&mut self, config: &PlatformConfig) -> Result<(), SynapticError> {
        self.config = Some(config.clone());
        self.optimize_for_battery()?;
        self.configure_background_sync()?;
        Ok(())
    }

    fn store(&self, key: &str, data: &[u8]) -> Result<(), SynapticError> {
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;

        storage.insert(key.to_string(), data.to_vec());

        // Update stats
        let mut stats = self.stats.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire stats lock: {}", e)))?;
        stats.item_count = storage.len();
        stats.used_storage = storage.values().map(|v| v.len()).sum();
        stats.average_item_size = if stats.item_count > 0 { stats.used_storage / stats.item_count } else { 0 };

        Ok(())
    }

    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>, SynapticError> {
        let storage = self.storage.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;

        Ok(storage.get(key).cloned())
    }

    fn delete(&self, key: &str) -> Result<bool, SynapticError> {
        let mut storage = self.storage.write()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;

        let existed = storage.remove(key).is_some();

        if existed {
            // Update stats
            let mut stats = self.stats.write()
                .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire stats lock: {}", e)))?;
            stats.item_count = storage.len();
            stats.used_storage = storage.values().map(|v| v.len()).sum();
            stats.average_item_size = if stats.item_count > 0 { stats.used_storage / stats.item_count } else { 0 };
        }

        Ok(existed)
    }

    fn list_keys(&self) -> Result<Vec<String>, SynapticError> {
        let storage = self.storage.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire storage lock: {}", e)))?;

        Ok(storage.keys().cloned().collect())
    }

    fn get_stats(&self) -> Result<StorageStats, SynapticError> {
        let stats = self.stats.read()
            .map_err(|e| SynapticError::ProcessingError(format!("Failed to acquire stats lock: {}", e)))?;

        Ok(stats.clone())
    }

    fn supports_feature(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::FileSystemAccess => true,
            PlatformFeature::NetworkAccess => true,
            PlatformFeature::BackgroundProcessing => true,
            PlatformFeature::PushNotifications => false, // Generic mobile may not support
            PlatformFeature::HardwareAcceleration => false, // Conservative assumption
            PlatformFeature::MultiThreading => true,
            PlatformFeature::LargeMemoryAllocation => false, // Conservative for mobile
        }
    }

    fn get_platform_info(&self) -> PlatformInfo {
        PlatformInfo {
            platform: self.platform.clone(),
            version: "1.0.0".to_string(),
            available_memory: 512 * 1024 * 1024, // 512MB conservative estimate
            available_storage: 1024 * 1024 * 1024, // 1GB conservative estimate
            supported_features: vec![
                PlatformFeature::FileSystemAccess,
                PlatformFeature::NetworkAccess,
                PlatformFeature::BackgroundProcessing,
                PlatformFeature::MultiThreading,
            ],
            performance_profile: self.get_generic_mobile_performance_profile(),
        }
    }
}
