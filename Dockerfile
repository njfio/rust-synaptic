# Synaptic AI Agent Memory System - Production Dockerfile
# Multi-stage build for optimal image size and security

# Stage 1: Builder
FROM rust:1.79-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/synaptic

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples
COPY tests ./tests

# Build release binary
# We only build the library for now, but you can add --bin synaptic if you have a binary
RUN cargo build --release --lib

# Build tests to ensure everything compiles
RUN cargo test --release --no-run || true

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash synaptic

# Create app directory
WORKDIR /app

# Copy built library from builder
COPY --from=builder /usr/src/synaptic/target/release/libsynaptic.rlib /app/ || true
COPY --from=builder /usr/src/synaptic/target/release/libsynaptic.so /app/ || true

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R synaptic:synaptic /app

# Switch to non-root user
USER synaptic

# Expose ports
# 9090 - Prometheus metrics
# 8080 - HTTP API (if implemented)
# 14268 - Jaeger collector (if using)
EXPOSE 9090 8080 14268

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD [ "true" ]

# Set environment variables
ENV RUST_LOG=info \
    RUST_BACKTRACE=1 \
    SYNAPTIC_DATA_DIR=/app/data \
    SYNAPTIC_LOG_DIR=/app/logs

# Labels for metadata
LABEL org.opencontainers.image.title="Synaptic AI Agent Memory System" \
      org.opencontainers.image.description="Intelligent AI agent memory system with knowledge graphs" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="Nicholas Ferguson <me@njf.io>" \
      org.opencontainers.image.source="https://github.com/njfio/rust-synaptic" \
      org.opencontainers.image.licenses="MIT"

# Default command - since this is a library, we'll use a placeholder
# In a real deployment, you'd run your actual application binary
CMD ["/bin/bash", "-c", "echo 'Synaptic is a library. Build your application with it.' && sleep infinity"]
