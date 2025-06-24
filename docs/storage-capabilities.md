# Storage Backend Capabilities

This document provides a comprehensive overview of the capabilities and limitations of each storage backend in the Synaptic memory system.

## Overview

The Synaptic memory system supports multiple storage backends, each with different capabilities and use cases. All backends implement the core `Storage` trait, but some features may not be available or may behave differently depending on the backend.

## Storage Backends

### 1. Memory Storage (`MemoryStorage`)

**Use Case**: Fast, temporary storage for development, testing, and caching scenarios.

**Capabilities**:
- ✅ **Full CRUD Operations**: Store, retrieve, update, delete
- ✅ **Search**: Full-text search across memory entries
- ✅ **Backup/Restore**: JSON-based backup with versioning
- ✅ **Statistics**: Real-time storage statistics
- ✅ **Maintenance**: Memory cleanup and optimization
- ✅ **Batch Operations**: Efficient bulk operations
- ✅ **Thread Safety**: Concurrent access using DashMap
- ✅ **Transactions**: In-memory transaction support

**Limitations**:
- ❌ **Persistence**: Data is lost when the application stops
- ❌ **Scalability**: Limited by available RAM
- ❌ **Cross-Process**: Cannot be shared between processes

**Backup Format**:
```json
{
  "entries": [...],
  "created_at": "2024-01-01T00:00:00Z",
  "backup_timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0",
  "entry_count": 1000
}
```

### 2. File Storage (`FileStorage`)

**Use Case**: Persistent storage for single-node applications with moderate data volumes.

**Capabilities**:
- ✅ **Full CRUD Operations**: Store, retrieve, update, delete
- ✅ **Search**: Full-text search with indexing
- ✅ **Backup/Restore**: JSON export/import functionality
- ✅ **Statistics**: File-based storage statistics
- ✅ **Maintenance**: Database compaction and optimization
- ✅ **Persistence**: Data survives application restarts
- ✅ **ACID Properties**: Atomic operations via Sled database
- ✅ **Compression**: Optional data compression

**Limitations**:
- ❌ **Concurrent Writers**: Limited concurrent write performance
- ❌ **Network Access**: Local file system only
- ❌ **Distributed**: Cannot be shared across multiple nodes

**Backup Process**:
1. Export all entries to JSON format
2. Include metadata and timestamps
3. Atomic file operations ensure consistency

### 3. SQL Storage (`DatabaseClient`) [Optional Feature]

**Use Case**: Enterprise applications requiring ACID compliance, complex queries, and high availability.

**Capabilities**:
- ✅ **Full CRUD Operations**: Store, retrieve, update, delete
- ✅ **Advanced Search**: SQL-based queries and full-text search
- ✅ **Backup/Restore**: Database-native backup mechanisms
- ✅ **Statistics**: Comprehensive database statistics
- ✅ **Maintenance**: Database optimization and cleanup
- ✅ **Transactions**: Full ACID transaction support
- ✅ **Scalability**: Horizontal and vertical scaling
- ✅ **Concurrent Access**: High-performance concurrent operations
- ✅ **Network Access**: Remote database connections
- ✅ **Replication**: Database replication and clustering

**Supported Databases**:
- PostgreSQL (recommended)
- SQLite (for development)

**Backup Methods**:
- Database-native backup tools (pg_dump, etc.)
- Application-level JSON export
- Point-in-time recovery (PostgreSQL)

## Feature Comparison Matrix

| Feature | Memory Storage | File Storage | SQL Storage |
|---------|---------------|--------------|-------------|
| **Core Operations** |
| Store/Retrieve | ✅ | ✅ | ✅ |
| Update/Delete | ✅ | ✅ | ✅ |
| Search | ✅ | ✅ | ✅ |
| **Persistence** |
| Data Persistence | ❌ | ✅ | ✅ |
| Crash Recovery | ❌ | ✅ | ✅ |
| **Backup/Restore** |
| Application Backup | ✅ | ✅ | ✅ |
| Native Backup | ❌ | ❌ | ✅ |
| Point-in-time Recovery | ❌ | ❌ | ✅ |
| **Performance** |
| Read Performance | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Write Performance | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Concurrent Access | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Scalability** |
| Data Volume | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| Concurrent Users | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| Distributed | ❌ | ❌ | ✅ |

## Usage Recommendations

### Development and Testing
```rust
use synaptic::memory::storage::memory::MemoryStorage;

let storage = MemoryStorage::new();
// Fast, no persistence needed
```

### Single-Node Applications
```rust
use synaptic::memory::storage::file::FileStorage;

let storage = FileStorage::new("./data/memories.db").await?;
// Persistent, good performance for moderate data
```

### Enterprise Applications
```rust
use synaptic::integrations::database::DatabaseClient;

let storage = DatabaseClient::new(DatabaseConfig {
    database_url: "postgresql://user:pass@host/db".to_string(),
    max_connections: 20,
    // ... other config
}).await?;
// Full ACID, scalable, enterprise-ready
```

## Backup Best Practices

### Memory Storage
```rust
// Create backup
storage.backup("./backups/memory_backup.json").await?;

// Restore from backup
storage.restore("./backups/memory_backup.json").await?;
```

### File Storage
```rust
// Export to JSON
storage.backup("./backups/file_export.json").await?;

// Import from JSON
storage.restore("./backups/file_export.json").await?;
```

### SQL Storage
```rust
// Application-level backup
storage.backup("./backups/sql_export.json").await?;

// Use database-native tools for production:
// pg_dump -h host -U user -d database > backup.sql
```

## Error Handling

All storage backends return `Result<T>` types with detailed error information:

```rust
match storage.backup("./backup.json").await {
    Ok(()) => println!("Backup successful"),
    Err(e) => eprintln!("Backup failed: {}", e),
}
```

## Configuration

Storage backends can be configured through the `StorageConfig` struct:

```rust
let config = StorageConfig {
    enable_compression: true,
    enable_encryption: false,
    max_entry_size: 1024 * 1024, // 1MB
    cache_size: 1000,
    sync_interval_seconds: 60,
    backup_interval_hours: 24,
};
```

## Migration Between Backends

Data can be migrated between storage backends using the backup/restore functionality:

```rust
// Export from source
source_storage.backup("./migration.json").await?;

// Import to destination
dest_storage.restore("./migration.json").await?;
```

## Monitoring and Maintenance

All storage backends provide statistics and maintenance capabilities:

```rust
// Get storage statistics
let stats = storage.stats().await?;
println!("Total entries: {}", stats.total_entries);

// Perform maintenance
storage.maintenance().await?;
```
