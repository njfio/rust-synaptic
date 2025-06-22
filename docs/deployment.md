# Synaptic Deployment Guide

This guide covers deploying Synaptic in various environments, from development to production.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration Management](#configuration-management)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Backup and Recovery](#backup-and-recovery)
8. [Security Considerations](#security-considerations)

## Development Setup

### Prerequisites

- Rust 1.79+ 
- PostgreSQL 13+ (optional, for SQL storage)
- Redis 6+ (optional, for distributed features)
- Docker and Docker Compose (optional)

### Quick Development Setup

```bash
# Clone the repository
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic

# Build the project
cargo build

# Run tests
cargo test

# Start development services (optional)
just setup  # or docker-compose up -d
```

### Using Justfile

The project includes a Justfile for common development tasks:

```bash
# Show available commands
just

# Development commands
just build          # Build the project
just test           # Run all tests
just test-features  # Run tests with all features
just clippy         # Run lints
just fmt            # Format code

# Infrastructure commands
just setup          # Start all services
just services-up    # Start Docker services
just services-down  # Stop Docker services
just clean          # Clean build artifacts
```

### Environment Configuration

Create a `.env` file for development:

```env
# Database configuration
DATABASE_URL=postgresql://synaptic:password@localhost:5432/synaptic_dev
REDIS_URL=redis://localhost:6379

# Security settings
ENCRYPTION_KEY=your-32-byte-encryption-key-here
JWT_SECRET=your-jwt-secret-here

# Feature flags
ENABLE_ANALYTICS=true
ENABLE_SECURITY=true
ENABLE_MULTIMODAL=false

# Logging
RUST_LOG=synaptic=debug,info
LOG_LEVEL=debug
```

## Production Deployment

### System Requirements

#### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- Network: 100 Mbps

#### Recommended Requirements
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 100GB+ SSD
- Network: 1 Gbps

### Binary Deployment

```bash
# Build optimized release binary
cargo build --release --features "full"

# Copy binary to target server
scp target/release/synaptic user@server:/opt/synaptic/

# Create systemd service
sudo cp deployment/synaptic.service /etc/systemd/system/
sudo systemctl enable synaptic
sudo systemctl start synaptic
```

### Systemd Service Configuration

Create `/etc/systemd/system/synaptic.service`:

```ini
[Unit]
Description=Synaptic AI Memory System
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=synaptic
Group=synaptic
WorkingDirectory=/opt/synaptic
ExecStart=/opt/synaptic/synaptic
Restart=always
RestartSec=10

# Environment
Environment=RUST_LOG=synaptic=info
EnvironmentFile=/opt/synaptic/.env

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/synaptic/data

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

### Database Setup

#### PostgreSQL Setup

```sql
-- Create database and user
CREATE DATABASE synaptic_prod;
CREATE USER synaptic WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE synaptic_prod TO synaptic;

-- Connect to the database
\c synaptic_prod

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO synaptic;
```

#### Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf

# Key settings:
# bind 127.0.0.1
# requirepass your_redis_password
# maxmemory 2gb
# maxmemory-policy allkeys-lru

# Restart Redis
sudo systemctl restart redis-server
```

## Docker Deployment

### Single Container

```dockerfile
# Dockerfile
FROM rust:1.79 as builder

WORKDIR /app
COPY . .
RUN cargo build --release --features "full"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/synaptic /usr/local/bin/synaptic

EXPOSE 8080
CMD ["synaptic"]
```

Build and run:

```bash
# Build image
docker build -t synaptic:latest .

# Run container
docker run -d \
  --name synaptic \
  -p 8080:8080 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/synaptic \
  -e REDIS_URL=redis://redis:6379 \
  -v synaptic_data:/data \
  synaptic:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  synaptic:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://synaptic:password@postgres:5432/synaptic
      - REDIS_URL=redis://redis:6379
      - RUST_LOG=synaptic=info
    depends_on:
      - postgres
      - redis
    volumes:
      - synaptic_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=synaptic
      - POSTGRES_USER=synaptic
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass password
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Optional: Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  synaptic_data:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

Deploy with Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f synaptic

# Scale the application
docker-compose up -d --scale synaptic=3

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: synaptic

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: synaptic-config
  namespace: synaptic
data:
  RUST_LOG: "synaptic=info"
  ENABLE_ANALYTICS: "true"
  ENABLE_SECURITY: "true"
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: synaptic-secrets
  namespace: synaptic
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  encryption-key: <base64-encoded-encryption-key>
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synaptic
  namespace: synaptic
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synaptic
  template:
    metadata:
      labels:
        app: synaptic
    spec:
      containers:
      - name: synaptic
        image: synaptic:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: synaptic-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: synaptic-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: synaptic-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: synaptic-service
  namespace: synaptic
spec:
  selector:
    app: synaptic
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: synaptic-ingress
  namespace: synaptic
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - synaptic.yourdomain.com
    secretName: synaptic-tls
  rules:
  - host: synaptic.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: synaptic-service
            port:
              number: 80
```

Deploy to Kubernetes:

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n synaptic
kubectl get services -n synaptic

# View logs
kubectl logs -f deployment/synaptic -n synaptic

# Scale deployment
kubectl scale deployment synaptic --replicas=5 -n synaptic
```

## Configuration Management

### Environment-based Configuration

```rust
// config.rs
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub database_url: String,
    pub redis_url: Option<String>,
    pub log_level: String,
    pub enable_analytics: bool,
    pub enable_security: bool,
    pub max_memory_size: usize,
}

impl AppConfig {
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(AppConfig {
            database_url: env::var("DATABASE_URL")?,
            redis_url: env::var("REDIS_URL").ok(),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            enable_analytics: env::var("ENABLE_ANALYTICS")
                .unwrap_or_else(|_| "false".to_string())
                .parse()?,
            enable_security: env::var("ENABLE_SECURITY")
                .unwrap_or_else(|_| "true".to_string())
                .parse()?,
            max_memory_size: env::var("MAX_MEMORY_SIZE")
                .unwrap_or_else(|_| "1073741824".to_string()) // 1GB
                .parse()?,
        })
    }
}
```

### Configuration Files

```toml
# config/production.toml
[database]
url = "postgresql://synaptic:password@localhost:5432/synaptic_prod"
max_connections = 20
connection_timeout = 30

[redis]
url = "redis://localhost:6379"
password = "redis_password"
max_connections = 10

[security]
enable_encryption = true
enable_access_control = true
enable_audit_logging = true
encryption_algorithm = "AES-256-GCM"

[analytics]
enable_behavioral_analysis = true
enable_performance_monitoring = true
metrics_retention_days = 90

[logging]
level = "info"
format = "json"
output = "stdout"

[performance]
max_memory_size = 2147483648  # 2GB
cache_size = 100000
enable_compression = true
```

## Monitoring and Observability

### Health Checks

```rust
// health.rs
use axum::{http::StatusCode, response::Json, routing::get, Router};
use serde_json::{json, Value};

pub fn health_routes() -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        .route("/metrics", get(metrics))
}

async fn health_check() -> Result<Json<Value>, StatusCode> {
    // Basic health check
    Ok(Json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION")
    })))
}

async fn readiness_check() -> Result<Json<Value>, StatusCode> {
    // Check dependencies (database, redis, etc.)
    // Return 503 if not ready
    Ok(Json(json!({
        "status": "ready",
        "dependencies": {
            "database": "connected",
            "redis": "connected"
        }
    })))
}
```

### Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'synaptic'
    static_configs:
      - targets: ['synaptic:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Synaptic Monitoring",
    "panels": [
      {
        "title": "Memory Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(synaptic_memory_operations_total[5m])",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(synaptic_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/synaptic"
DB_NAME="synaptic_prod"

# Create backup directory
mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump -h localhost -U synaptic -d $DB_NAME | gzip > $BACKUP_DIR/synaptic_db_$DATE.sql.gz

# Redis backup
redis-cli --rdb $BACKUP_DIR/synaptic_redis_$DATE.rdb

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete
```

### Automated Backup with Cron

```bash
# Add to crontab
0 2 * * * /opt/synaptic/scripts/backup.sh
```

### Recovery Procedures

```bash
# Database recovery
gunzip -c /opt/backups/synaptic/synaptic_db_20240115_020000.sql.gz | psql -h localhost -U synaptic -d synaptic_prod

# Redis recovery
redis-cli --rdb /opt/backups/synaptic/synaptic_redis_20240115_020000.rdb
```

## Security Considerations

### Network Security

- Use TLS/SSL for all connections
- Implement proper firewall rules
- Use VPN for administrative access
- Enable fail2ban for brute force protection

### Application Security

- Enable encryption at rest and in transit
- Implement proper authentication and authorization
- Use secure session management
- Regular security updates

### Monitoring Security

- Enable audit logging
- Monitor for suspicious activities
- Set up security alerts
- Regular security assessments

This deployment guide provides comprehensive instructions for deploying Synaptic in various environments. For specific use cases or advanced configurations, consult the other documentation files.
