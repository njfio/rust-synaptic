# Platform Compatibility Guide

This guide helps developers understand which Synaptic features are available on different platforms and how to write cross-platform compatible code.

## Quick Reference

### Platform Detection

```rust
use synaptic::cross_platform::{Platform, detect_platform};

let platform = detect_platform();
match platform {
    Platform::WebAssembly => {
        // Browser environment
        println!("Running in browser");
    },
    Platform::iOS => {
        // iOS device
        println!("Running on iOS");
    },
    Platform::Android => {
        // Android device
        println!("Running on Android");
    },
    Platform::Desktop => {
        // Desktop environment (Windows, macOS, Linux)
        println!("Running on desktop");
    },
    Platform::Server => {
        // Server environment
        println!("Running on server");
    },
}
```

### Feature Availability Check

```rust
use synaptic::cross_platform::{CrossPlatformAdapter, PlatformFeature};

let adapter = synaptic::cross_platform::create_adapter()?;

// Check if specific features are available
if adapter.supports_feature(PlatformFeature::FileSystemAccess) {
    // Use file system storage
    adapter.enable_file_storage()?;
} else {
    // Fallback to memory or other storage
    adapter.enable_memory_storage()?;
}

if adapter.supports_feature(PlatformFeature::BackgroundProcessing) {
    // Enable background sync
    adapter.enable_background_sync()?;
}
```

## Platform-Specific Code Patterns

### Conditional Compilation

Use feature flags for platform-specific code:

```rust
// WASM-specific code
#[cfg(feature = "wasm")]
mod wasm_specific {
    use synaptic::cross_platform::wasm::WasmAdapter;
    
    pub async fn setup_web_worker(adapter: &WasmAdapter) -> Result<(), Error> {
        adapter.initialize_web_worker().await?;
        Ok(())
    }
}

// Mobile-specific code
#[cfg(any(target_os = "ios", target_os = "android"))]
mod mobile_specific {
    use synaptic::cross_platform::mobile::MobileConfig;
    
    pub fn optimize_for_mobile() -> MobileConfig {
        MobileConfig {
            enable_battery_optimization: true,
            memory_pressure_threshold: 0.8,
            cache_size_limit: 50 * 1024 * 1024, // 50MB
            ..Default::default()
        }
    }
}

// Desktop/Server code
#[cfg(not(any(feature = "wasm", target_os = "ios", target_os = "android")))]
mod desktop_specific {
    pub fn setup_high_performance() -> Config {
        Config {
            cache_size_limit: 1024 * 1024 * 1024, // 1GB
            enable_multi_threading: true,
            ..Default::default()
        }
    }
}
```

### Runtime Platform Adaptation

```rust
use synaptic::cross_platform::{CrossPlatformAdapter, Platform};

pub struct AdaptiveMemoryManager {
    adapter: Box<dyn CrossPlatformAdapter>,
    config: AdaptiveConfig,
}

impl AdaptiveMemoryManager {
    pub fn new() -> Result<Self, Error> {
        let adapter = synaptic::cross_platform::create_adapter()?;
        let config = Self::create_adaptive_config(&adapter);
        
        Ok(Self { adapter, config })
    }
    
    fn create_adaptive_config(adapter: &dyn CrossPlatformAdapter) -> AdaptiveConfig {
        let platform_info = adapter.get_platform_info();
        
        match platform_info.platform {
            Platform::WebAssembly => AdaptiveConfig {
                cache_size: 10 * 1024 * 1024,  // 10MB
                enable_compression: true,
                sync_interval: 60,  // 1 minute
                batch_size: 100,
            },
            Platform::iOS => AdaptiveConfig {
                cache_size: 50 * 1024 * 1024,  // 50MB
                enable_compression: true,
                sync_interval: 300, // 5 minutes
                batch_size: 500,
            },
            Platform::Android => AdaptiveConfig {
                cache_size: 100 * 1024 * 1024, // 100MB
                enable_compression: true,
                sync_interval: 300, // 5 minutes
                batch_size: 1000,
            },
            Platform::Desktop | Platform::Server => AdaptiveConfig {
                cache_size: 500 * 1024 * 1024, // 500MB
                enable_compression: false,      // CPU is less constrained
                sync_interval: 30,              // 30 seconds
                batch_size: 5000,
            },
        }
    }
}
```

## Storage Backend Selection

### Automatic Backend Selection

```rust
use synaptic::cross_platform::{StorageBackend, select_optimal_backend};

pub fn setup_storage() -> Result<Box<dyn Storage>, Error> {
    let available_backends = synaptic::cross_platform::get_available_backends();
    let optimal_backend = select_optimal_backend(&available_backends)?;
    
    match optimal_backend {
        StorageBackend::IndexedDB => {
            // WASM environment
            Ok(Box::new(IndexedDBStorage::new()?))
        },
        StorageBackend::CoreData => {
            // iOS environment
            Ok(Box::new(CoreDataStorage::new()?))
        },
        StorageBackend::Room => {
            // Android environment
            Ok(Box::new(RoomStorage::new()?))
        },
        StorageBackend::FileSystem => {
            // Desktop/Server environment
            Ok(Box::new(FileSystemStorage::new()?))
        },
        StorageBackend::Memory => {
            // Fallback for any environment
            Ok(Box::new(MemoryStorage::new()))
        },
    }
}
```

### Manual Backend Configuration

```rust
use synaptic::cross_platform::{CrossPlatformAdapter, StorageBackend};

pub fn configure_storage_priority(adapter: &mut dyn CrossPlatformAdapter) -> Result<(), Error> {
    let platform_info = adapter.get_platform_info();
    
    let preferred_backends = match platform_info.platform {
        Platform::WebAssembly => vec![
            StorageBackend::IndexedDB,
            StorageBackend::Memory,
        ],
        Platform::iOS => vec![
            StorageBackend::CoreData,
            StorageBackend::FileSystem,
            StorageBackend::Memory,
        ],
        Platform::Android => vec![
            StorageBackend::Room,
            StorageBackend::FileSystem,
            StorageBackend::Memory,
        ],
        Platform::Desktop | Platform::Server => vec![
            StorageBackend::FileSystem,
            StorageBackend::Memory,
        ],
    };
    
    for backend in preferred_backends {
        if adapter.supports_backend(backend) {
            adapter.set_primary_backend(backend)?;
            break;
        }
    }
    
    Ok(())
}
```

## Performance Optimization Patterns

### Memory Management

```rust
use synaptic::cross_platform::{Platform, MemoryManager};

pub struct PlatformOptimizedMemoryManager {
    manager: MemoryManager,
    platform: Platform,
}

impl PlatformOptimizedMemoryManager {
    pub fn new() -> Result<Self, Error> {
        let platform = synaptic::cross_platform::detect_platform();
        let manager = MemoryManager::new_for_platform(platform)?;
        
        Ok(Self { manager, platform })
    }
    
    pub fn optimize_for_platform(&mut self) -> Result<(), Error> {
        match self.platform {
            Platform::WebAssembly => {
                // Aggressive memory management for browser
                self.manager.set_gc_threshold(0.7)?;
                self.manager.enable_aggressive_cleanup(true)?;
            },
            Platform::iOS => {
                // iOS memory pressure handling
                self.manager.set_memory_pressure_threshold(0.8)?;
                self.manager.enable_background_cleanup(true)?;
            },
            Platform::Android => {
                // Android memory management
                self.manager.set_memory_pressure_threshold(0.7)?;
                self.manager.enable_doze_mode_handling(true)?;
            },
            Platform::Desktop | Platform::Server => {
                // Less aggressive for desktop/server
                self.manager.set_gc_threshold(0.9)?;
                self.manager.enable_aggressive_cleanup(false)?;
            },
        }
        
        Ok(())
    }
}
```

### Async/Sync Patterns

```rust
use synaptic::cross_platform::{Platform, AsyncCapability};

pub trait PlatformAwareOperation {
    fn execute_sync(&self, data: &[u8]) -> Result<Vec<u8>, Error>;
    
    #[cfg(feature = "async")]
    async fn execute_async(&self, data: &[u8]) -> Result<Vec<u8>, Error>;
}

pub struct AdaptiveProcessor {
    platform: Platform,
}

impl AdaptiveProcessor {
    pub fn new() -> Self {
        Self {
            platform: synaptic::cross_platform::detect_platform(),
        }
    }
    
    pub async fn process(&self, data: &[u8]) -> Result<Vec<u8>, Error> {
        match self.platform {
            Platform::WebAssembly => {
                // Use async for WASM to avoid blocking main thread
                #[cfg(feature = "wasm")]
                {
                    self.execute_async(data).await
                }
                #[cfg(not(feature = "wasm"))]
                {
                    self.execute_sync(data)
                }
            },
            _ => {
                // Use sync for other platforms unless async is beneficial
                if self.should_use_async(data) {
                    #[cfg(feature = "async")]
                    {
                        self.execute_async(data).await
                    }
                    #[cfg(not(feature = "async"))]
                    {
                        self.execute_sync(data)
                    }
                } else {
                    self.execute_sync(data)
                }
            }
        }
    }
    
    fn should_use_async(&self, data: &[u8]) -> bool {
        // Use async for large data or when background processing is available
        data.len() > 1024 * 1024 || // > 1MB
        synaptic::cross_platform::supports_background_processing()
    }
}
```

## Error Handling Patterns

### Platform-Specific Error Handling

```rust
use synaptic::error::{SynapticError, PlatformError};

pub fn handle_platform_error(error: SynapticError) -> Result<(), SynapticError> {
    match error {
        SynapticError::PlatformError(platform_error) => {
            match platform_error {
                PlatformError::WasmError(msg) => {
                    // Handle WASM-specific errors
                    eprintln!("WASM error: {}", msg);
                    // Maybe fallback to different storage
                    Ok(())
                },
                PlatformError::iOSError(msg) => {
                    // Handle iOS-specific errors
                    eprintln!("iOS error: {}", msg);
                    // Maybe request more memory or reduce cache
                    Ok(())
                },
                PlatformError::AndroidError(msg) => {
                    // Handle Android-specific errors
                    eprintln!("Android error: {}", msg);
                    // Maybe handle doze mode or permissions
                    Ok(())
                },
                _ => Err(SynapticError::PlatformError(platform_error)),
            }
        },
        other => Err(other),
    }
}
```

## Testing Patterns

### Cross-Platform Testing

```rust
#[cfg(test)]
mod cross_platform_tests {
    use super::*;
    
    #[test]
    fn test_platform_detection() {
        let platform = synaptic::cross_platform::detect_platform();
        assert!(matches!(platform, Platform::WebAssembly | Platform::iOS | 
                        Platform::Android | Platform::Desktop | Platform::Server));
    }
    
    #[cfg(feature = "wasm")]
    #[wasm_bindgen_test]
    async fn test_wasm_specific_features() {
        let adapter = WasmAdapter::new().unwrap();
        assert!(adapter.supports_feature(PlatformFeature::NetworkAccess));
        assert!(!adapter.supports_feature(PlatformFeature::FileSystemAccess));
    }
    
    #[cfg(target_os = "ios")]
    #[test]
    fn test_ios_specific_features() {
        let adapter = iOSAdapter::new().unwrap();
        assert!(adapter.supports_feature(PlatformFeature::BackgroundProcessing));
        assert!(adapter.supports_feature(PlatformFeature::PushNotifications));
    }
    
    #[cfg(target_os = "android")]
    #[test]
    fn test_android_specific_features() {
        let adapter = AndroidAdapter::new().unwrap();
        assert!(adapter.supports_feature(PlatformFeature::LargeMemoryAllocation));
        assert!(adapter.supports_feature(PlatformFeature::MultiThreading));
    }
}
```

## Best Practices

### 1. Always Check Feature Availability

```rust
// Good
if adapter.supports_feature(PlatformFeature::FileSystemAccess) {
    adapter.use_file_storage()?;
} else {
    adapter.use_memory_storage()?;
}

// Bad - assumes feature is available
adapter.use_file_storage()?; // May fail on WASM
```

### 2. Use Adaptive Configuration

```rust
// Good - adapts to platform capabilities
let config = create_adaptive_config_for_platform(platform);

// Bad - one-size-fits-all
let config = Config {
    cache_size: 1024 * 1024 * 1024, // Too large for mobile
    ..Default::default()
};
```

### 3. Handle Platform Errors Gracefully

```rust
// Good
match result {
    Ok(data) => process_data(data),
    Err(SynapticError::PlatformError(_)) => {
        // Try alternative approach
        fallback_processing()
    },
    Err(e) => return Err(e),
}

// Bad - doesn't handle platform-specific failures
let data = result.unwrap(); // May panic on platform errors
```

### 4. Use Feature Flags Appropriately

```rust
// Good - conditional compilation
#[cfg(feature = "wasm")]
use synaptic::cross_platform::wasm::WasmAdapter;

#[cfg(feature = "mobile")]
use synaptic::cross_platform::mobile::{iOSAdapter, AndroidAdapter};

// Bad - runtime feature detection for compile-time features
if cfg!(feature = "wasm") {
    // This doesn't work as expected
}
```

This guide provides the foundation for writing robust cross-platform applications with Synaptic. Always test on your target platforms and refer to the platform-specific documentation for detailed implementation guidance.
