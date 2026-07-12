use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::security::access_control::{
    AccessControlManager, AuthenticationCredentials, AuthenticationType,
};
use synaptic::security::audit::AuditLogger;
use synaptic::security::encryption::EncryptionManager;
use synaptic::security::key_management::KeyManager;
use synaptic::security::{AuditConfig, Permission, SecurityConfig, SecurityContext};
use tokio::runtime::Runtime;

/// Build a SecurityConfig that does not require MFA, so benchmark contexts
/// validate without an interactive challenge.
fn bench_security_config() -> SecurityConfig {
    let mut config = SecurityConfig::default();
    config.access_control_policy.require_mfa = false;
    config
}

/// Construct an encryption manager for benchmarking.
async fn new_encryption_manager() -> EncryptionManager {
    let config = bench_security_config();
    let key_manager = KeyManager::new(&config).await.unwrap();
    EncryptionManager::new(&config, key_manager).await.unwrap()
}

/// Construct a valid (MFA-satisfied, valid-session) security context.
fn bench_context(user_id: &str) -> SecurityContext {
    let mut ctx = SecurityContext::new(user_id.to_string(), vec!["user".to_string()]);
    ctx.mfa_verified = true;
    ctx
}

/// Build a memory entry of a given content size.
fn sized_entry(key: &str, size: usize) -> MemoryEntry {
    MemoryEntry::new(key.to_string(), "x".repeat(size), MemoryType::LongTerm)
}

/// Benchmark encryption operations.
fn bench_encryption_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("encryption_operations");

    // Test different data sizes
    let data_sizes = [1024, 10240, 102400]; // 1KB, 10KB, 100KB

    for &size in data_sizes.iter() {
        group.throughput(Throughput::Bytes(size as u64));

        // Benchmark AES-256-GCM encryption (standard_encrypt).
        group.bench_with_input(
            BenchmarkId::new("aes_256_gcm_encrypt", size),
            &size,
            |b, &size| {
                let ctx = bench_context("bench_user");
                let entry = sized_entry("enc_key", size);
                b.iter(|| {
                    rt.block_on(async {
                        let mut manager = new_encryption_manager().await;
                        manager
                            .standard_encrypt(black_box(&entry), &ctx)
                            .await
                            .unwrap()
                    })
                })
            },
        );

        // Benchmark AES-256-GCM decryption (standard_decrypt).
        group.bench_with_input(
            BenchmarkId::new("aes_256_gcm_decrypt", size),
            &size,
            |b, &size| {
                let ctx = bench_context("bench_user");
                let entry = sized_entry("enc_key", size);
                // Encrypt once with a manager we keep, then decrypt with the
                // same manager (the key lives in the manager's key store).
                b.iter_batched(
                    || {
                        rt.block_on(async {
                            let mut manager = new_encryption_manager().await;
                            let encrypted = manager.standard_encrypt(&entry, &ctx).await.unwrap();
                            (manager, encrypted)
                        })
                    },
                    |(mut manager, encrypted)| {
                        rt.block_on(async {
                            manager
                                .standard_decrypt(black_box(&encrypted), &ctx)
                                .await
                                .unwrap()
                        })
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark access control operations.
fn bench_access_control(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("access_control");

    for user_count in [100, 1000, 5000].iter() {
        let user_count = *user_count;
        group.throughput(Throughput::Elements(user_count as u64));

        // Benchmark authentication + permission checking.
        group.bench_with_input(
            BenchmarkId::new("permission_check", user_count),
            &user_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let config = bench_security_config();
                        let mut access_control = AccessControlManager::new(&config).await.unwrap();
                        access_control
                            .add_role(
                                "user".to_string(),
                                vec![Permission::ReadMemory, Permission::WriteMemory],
                            )
                            .await
                            .unwrap();

                        for i in 0..count {
                            let user_id = format!("user_{}", i);
                            // Authentication is real (argon2) as of Task 4.7:
                            // provision the user before authenticating.
                            access_control
                                .set_password(&user_id, "password123")
                                .unwrap();
                            let creds = AuthenticationCredentials {
                                auth_type: AuthenticationType::Password,
                                password: Some("password123".to_string()),
                                api_key: None,
                                certificate: None,
                                mfa_token: None,
                                ip_address: None,
                                user_agent: None,
                            };
                            let ctx = access_control.authenticate(user_id, creds).await.unwrap();

                            let _ = access_control
                                .check_permission(
                                    black_box(&ctx),
                                    black_box(Permission::ReadMemory),
                                )
                                .await;
                        }
                    })
                })
            },
        );

        // Benchmark role registration.
        group.bench_with_input(
            BenchmarkId::new("role_assignment", user_count),
            &user_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let config = bench_security_config();
                        let mut access_control = AccessControlManager::new(&config).await.unwrap();

                        for i in 0..count {
                            let role_name = format!("role_{}", i);
                            let perms = match i % 4 {
                                0 => vec![Permission::ReadMemory, Permission::WriteMemory],
                                1 => vec![Permission::ReadMemory],
                                2 => vec![Permission::ReadAnalytics],
                                _ => vec![Permission::ExecuteQueries],
                            };
                            access_control
                                .add_role(black_box(role_name), black_box(perms))
                                .await
                                .unwrap();
                        }
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark audit logging.
fn bench_audit_logging(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("audit_logging");

    for event_count in [1000, 10000, 50000].iter() {
        let event_count = *event_count;
        group.throughput(Throughput::Elements(event_count as u64));

        group.bench_with_input(
            BenchmarkId::new("audit_event_logging", event_count),
            &event_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut audit_logger =
                            AuditLogger::new(&AuditConfig::default()).await.unwrap();

                        for i in 0..count {
                            let ctx = bench_context(&format!("user_{}", i % 100));
                            match i % 4 {
                                0 => audit_logger
                                    .log_memory_operation(black_box(&ctx), "read", true)
                                    .await
                                    .unwrap(),
                                1 => audit_logger
                                    .log_access_decision(black_box(&ctx), "ReadMemory", true)
                                    .await
                                    .unwrap(),
                                2 => audit_logger
                                    .log_system_event(
                                        "backup_created",
                                        &format!("Backup {} created", i),
                                        synaptic::security::audit::RiskLevel::Low,
                                    )
                                    .await
                                    .unwrap(),
                                _ => audit_logger
                                    .log_authentication_event(
                                        &format!("user_{}", i % 100),
                                        "password",
                                        true,
                                        None,
                                    )
                                    .await
                                    .unwrap(),
                            }
                        }
                    })
                })
            },
        );

        // Benchmark audit query performance.
        group.bench_with_input(
            BenchmarkId::new("audit_query", event_count),
            &event_count,
            |b, &count| {
                let audit_logger = rt.block_on(async {
                    let mut audit_logger = AuditLogger::new(&AuditConfig::default()).await.unwrap();

                    for i in 0..count {
                        let ctx = bench_context(&format!("user_{}", i % 100));
                        audit_logger
                            .log_memory_operation(&ctx, "read", true)
                            .await
                            .unwrap();
                    }

                    audit_logger
                });

                b.iter(|| {
                    rt.block_on(async {
                        // Query audit events by user.
                        let user_id = format!("user_{}", black_box(50));
                        audit_logger
                            .get_user_audit_events(&user_id, Some(100))
                            .await
                            .unwrap();

                        // Query audit events by time range.
                        let start_time = chrono::Utc::now() - chrono::Duration::hours(1);
                        let end_time = chrono::Utc::now() + chrono::Duration::hours(1);
                        audit_logger
                            .get_audit_events(start_time, end_time)
                            .await
                            .unwrap();
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark secure memory operations (encrypt then decrypt entries).
fn bench_secure_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("secure_memory_operations");

    for entry_count in [100, 500, 1000].iter() {
        let entry_count = *entry_count;
        group.throughput(Throughput::Elements(entry_count as u64));

        group.bench_with_input(
            BenchmarkId::new("encrypted_memory_store", entry_count),
            &entry_count,
            |b, &count| {
                let ctx = bench_context("bench_user");
                b.iter(|| {
                    rt.block_on(async {
                        let mut manager = new_encryption_manager().await;

                        for i in 0..count {
                            let entry = MemoryEntry::new(
                                format!("secure_key_{}", i),
                                format!("Sensitive memory content {}", i),
                                MemoryType::LongTerm,
                            );
                            let encrypted = manager
                                .standard_encrypt(black_box(&entry), &ctx)
                                .await
                                .unwrap();
                            black_box(encrypted);
                        }
                    })
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("encrypted_memory_retrieve", entry_count),
            &entry_count,
            |b, &count| {
                let ctx = bench_context("bench_user");
                b.iter_batched(
                    || {
                        rt.block_on(async {
                            let mut manager = new_encryption_manager().await;
                            let mut encrypted = Vec::new();
                            for i in 0..count {
                                let entry = MemoryEntry::new(
                                    format!("secure_key_{}", i),
                                    format!("Sensitive memory content {}", i),
                                    MemoryType::LongTerm,
                                );
                                encrypted
                                    .push(manager.standard_encrypt(&entry, &ctx).await.unwrap());
                            }
                            (manager, encrypted)
                        })
                    },
                    |(mut manager, encrypted)| {
                        rt.block_on(async {
                            for enc in &encrypted {
                                let decrypted = manager
                                    .standard_decrypt(black_box(enc), &ctx)
                                    .await
                                    .unwrap();
                                black_box(decrypted);
                            }
                        })
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark key management operations.
fn bench_key_management(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("key_management");

    // Benchmark key manager creation (key store initialization).
    group.bench_function("key_manager_init", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = bench_security_config();
                let key_manager = KeyManager::new(black_box(&config)).await.unwrap();
                black_box(key_manager);
            })
        })
    });

    // Benchmark encrypt/decrypt round-trip (exercises key generation/lookup).
    group.bench_function("encrypt_decrypt_roundtrip", |b| {
        let ctx = bench_context("bench_user");
        let entry = sized_entry("rotation_key", 256);
        b.iter(|| {
            rt.block_on(async {
                let mut manager = new_encryption_manager().await;
                let encrypted = manager.standard_encrypt(&entry, &ctx).await.unwrap();
                let decrypted = manager
                    .standard_decrypt(black_box(&encrypted), &ctx)
                    .await
                    .unwrap();
                black_box(decrypted);
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
