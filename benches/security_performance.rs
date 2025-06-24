use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use synaptic::security::encryption::{EncryptionManager, EncryptionConfig};
use synaptic::security::access_control::{AccessControlManager, Permission, Role};
use synaptic::security::audit::{AuditLogger, AuditEvent};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use tokio::runtime::Runtime;

/// Create test data for security benchmarks
fn create_security_test_data(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            format!("Sensitive data entry {} with confidential information", i)
                .into_bytes()
        })
        .collect()
}

/// Benchmark encryption operations
fn bench_encryption_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("encryption_operations");
    
    // Test different data sizes
    let data_sizes = [1024, 10240, 102400]; // 1KB, 10KB, 100KB
    
    for &size in data_sizes.iter() {
        let test_data = vec![42u8; size];
        
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark AES-256-GCM encryption
        group.bench_with_input(
            BenchmarkId::new("aes_256_gcm_encrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let config = EncryptionConfig::default();
                        let encryption_manager = EncryptionManager::new(config).await.unwrap();
                        
                        encryption_manager.encrypt(black_box(&test_data)).await.unwrap()
                    })
                })
            },
        );
        
        // Benchmark AES-256-GCM decryption
        group.bench_with_input(
            BenchmarkId::new("aes_256_gcm_decrypt", size),
            &size,
            |b, _| {
                let encrypted_data = rt.block_on(async {
                    let config = EncryptionConfig::default();
                    let encryption_manager = EncryptionManager::new(config).await.unwrap();
                    encryption_manager.encrypt(&test_data).await.unwrap()
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        let config = EncryptionConfig::default();
                        let encryption_manager = EncryptionManager::new(config).await.unwrap();
                        
                        encryption_manager.decrypt(black_box(&encrypted_data)).await.unwrap()
                    })
                })
            },
        );
        
        // Benchmark ChaCha20-Poly1305 encryption
        group.bench_with_input(
            BenchmarkId::new("chacha20_poly1305_encrypt", size),
            &size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut config = EncryptionConfig::default();
                        config.algorithm = synaptic::security::encryption::EncryptionAlgorithm::ChaCha20Poly1305;
                        let encryption_manager = EncryptionManager::new(config).await.unwrap();
                        
                        encryption_manager.encrypt(black_box(&test_data)).await.unwrap()
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark access control operations
fn bench_access_control(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("access_control");
    
    // Test different numbers of users and permissions
    for user_count in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*user_count as u64));
        
        // Benchmark permission checking
        group.bench_with_input(
            BenchmarkId::new("permission_check", user_count),
            user_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let access_control = AccessControlManager::new().await.unwrap();
                        
                        // Create users and roles
                        for i in 0..count {
                            let user_id = format!("user_{}", i);
                            let role = if i % 3 == 0 { Role::Admin } else { Role::User };
                            
                            access_control.create_user(&user_id, role).await.unwrap();
                        }
                        
                        // Check permissions for all users
                        for i in 0..count {
                            let user_id = format!("user_{}", i);
                            access_control.check_permission(
                                black_box(&user_id),
                                black_box(&Permission::Read),
                                black_box("memory_resource"),
                            ).await.unwrap();
                        }
                    })
                })
            },
        );
        
        // Benchmark role assignment
        group.bench_with_input(
            BenchmarkId::new("role_assignment", user_count),
            user_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let access_control = AccessControlManager::new().await.unwrap();
                        
                        // Create and assign roles to users
                        for i in 0..count {
                            let user_id = format!("user_{}", i);
                            let role = match i % 4 {
                                0 => Role::Admin,
                                1 => Role::User,
                                2 => Role::ReadOnly,
                                _ => Role::Guest,
                            };
                            
                            access_control.create_user(&user_id, Role::Guest).await.unwrap();
                            access_control.assign_role(black_box(&user_id), black_box(role)).await.unwrap();
                        }
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark audit logging
fn bench_audit_logging(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("audit_logging");
    
    // Test different numbers of audit events
    for event_count in [1000, 10000, 50000].iter() {
        group.throughput(Throughput::Elements(*event_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("audit_event_logging", event_count),
            event_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let audit_logger = AuditLogger::new().await.unwrap();
                        
                        // Log various types of audit events
                        for i in 0..count {
                            let event = match i % 5 {
                                0 => AuditEvent::MemoryAccess {
                                    user_id: format!("user_{}", i % 100),
                                    memory_id: format!("memory_{}", i),
                                    action: "read".to_string(),
                                    timestamp: chrono::Utc::now(),
                                },
                                1 => AuditEvent::MemoryModification {
                                    user_id: format!("user_{}", i % 100),
                                    memory_id: format!("memory_{}", i),
                                    action: "update".to_string(),
                                    timestamp: chrono::Utc::now(),
                                },
                                2 => AuditEvent::SecurityViolation {
                                    user_id: format!("user_{}", i % 100),
                                    violation_type: "unauthorized_access".to_string(),
                                    resource: format!("resource_{}", i),
                                    timestamp: chrono::Utc::now(),
                                },
                                3 => AuditEvent::SystemEvent {
                                    event_type: "backup_created".to_string(),
                                    details: format!("Backup {} created", i),
                                    timestamp: chrono::Utc::now(),
                                },
                                _ => AuditEvent::ConfigurationChange {
                                    user_id: format!("admin_{}", i % 10),
                                    setting: format!("setting_{}", i),
                                    old_value: "old".to_string(),
                                    new_value: "new".to_string(),
                                    timestamp: chrono::Utc::now(),
                                },
                            };
                            
                            audit_logger.log_event(black_box(event)).await.unwrap();
                        }
                    })
                })
            },
        );
        
        // Benchmark audit query performance
        group.bench_with_input(
            BenchmarkId::new("audit_query", event_count),
            event_count,
            |b, &count| {
                let audit_logger = rt.block_on(async {
                    let audit_logger = AuditLogger::new().await.unwrap();
                    
                    // Pre-populate with audit events
                    for i in 0..count {
                        let event = AuditEvent::MemoryAccess {
                            user_id: format!("user_{}", i % 100),
                            memory_id: format!("memory_{}", i),
                            action: "read".to_string(),
                            timestamp: chrono::Utc::now(),
                        };
                        audit_logger.log_event(event).await.unwrap();
                    }
                    
                    audit_logger
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        // Query audit events by user
                        let user_id = format!("user_{}", black_box(50));
                        audit_logger.query_events_by_user(&user_id, 100).await.unwrap();
                        
                        // Query audit events by time range
                        let start_time = chrono::Utc::now() - chrono::Duration::hours(1);
                        let end_time = chrono::Utc::now();
                        audit_logger.query_events_by_time_range(start_time, end_time, 100).await.unwrap();
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark secure memory operations
fn bench_secure_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("secure_memory_operations");
    
    // Test different numbers of encrypted memory entries
    for entry_count in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*entry_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("encrypted_memory_store", entry_count),
            entry_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let encryption_config = EncryptionConfig::default();
                        let encryption_manager = EncryptionManager::new(encryption_config).await.unwrap();
                        
                        // Store encrypted memory entries
                        for i in 0..count {
                            let content = format!("Sensitive memory content {}", i);
                            let encrypted_content = encryption_manager.encrypt(content.as_bytes()).await.unwrap();
                            
                            let entry = MemoryEntry::new(
                                format!("secure_key_{}", i),
                                String::from_utf8(encrypted_content).unwrap(),
                                MemoryType::LongTerm,
                            );
                            
                            // Simulate secure storage operation
                            black_box(entry);
                        }
                    })
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("encrypted_memory_retrieve", entry_count),
            entry_count,
            |b, &count| {
                let encrypted_entries = rt.block_on(async {
                    let encryption_config = EncryptionConfig::default();
                    let encryption_manager = EncryptionManager::new(encryption_config).await.unwrap();
                    
                    let mut entries = Vec::new();
                    for i in 0..count {
                        let content = format!("Sensitive memory content {}", i);
                        let encrypted_content = encryption_manager.encrypt(content.as_bytes()).await.unwrap();
                        entries.push(encrypted_content);
                    }
                    entries
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        let encryption_config = EncryptionConfig::default();
                        let encryption_manager = EncryptionManager::new(encryption_config).await.unwrap();
                        
                        // Decrypt and retrieve memory entries
                        for encrypted_content in &encrypted_entries {
                            let decrypted_content = encryption_manager.decrypt(black_box(encrypted_content)).await.unwrap();
                            black_box(decrypted_content);
                        }
                    })
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark key management operations
fn bench_key_management(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("key_management");
    
    // Benchmark key generation
    group.bench_function("key_generation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let encryption_config = EncryptionConfig::default();
                let encryption_manager = EncryptionManager::new(encryption_config).await.unwrap();
                
                // Generate multiple keys
                for _ in 0..10 {
                    encryption_manager.generate_key().await.unwrap();
                }
            })
        })
    });
    
    // Benchmark key rotation
    group.bench_function("key_rotation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let encryption_config = EncryptionConfig::default();
                let encryption_manager = EncryptionManager::new(encryption_config).await.unwrap();
                
                // Simulate key rotation process
                let old_key = encryption_manager.get_current_key().await.unwrap();
                let new_key = encryption_manager.generate_key().await.unwrap();
                
                // Re-encrypt data with new key
                let test_data = b"Sensitive data for key rotation test";
                let encrypted_old = encryption_manager.encrypt_with_key(test_data, &old_key).await.unwrap();
                let decrypted = encryption_manager.decrypt_with_key(&encrypted_old, &old_key).await.unwrap();
                let encrypted_new = encryption_manager.encrypt_with_key(&decrypted, black_box(&new_key)).await.unwrap();
                
                black_box(encrypted_new);
            })
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_encryption_operations,
    bench_access_control,
    bench_audit_logging,
    bench_secure_memory_operations,
    bench_key_management
);
criterion_main!(benches);
