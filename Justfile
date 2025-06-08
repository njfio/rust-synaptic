# Synaptic 🧠 - Justfile for Development and Operations
# State-of-the-Art AI Agent Memory System

# Default recipe - show available commands
default:
    @echo "🧠 Synaptic - AI Agent Memory System"
    @echo "=================================="
    @echo ""
    @echo "📋 Available Commands:"
    @echo ""
    @echo "🔧 Development:"
    @echo "  just build          - Build the project"
    @echo "  just test           - Run all tests"
    @echo "  just test-quiet     - Run tests quietly"
    @echo "  just test-features  - Run tests with all features"
    @echo "  just check          - Run cargo check"
    @echo "  just fmt            - Format code"
    @echo "  just clippy         - Run clippy lints"
    @echo "  just clean          - Clean build artifacts"
    @echo ""
    @echo "🐳 Infrastructure:"
    @echo "  just setup          - Complete setup (env + services)"
    @echo "  just services-up    - Start all Docker services"
    @echo "  just services-down  - Stop all Docker services"
    @echo "  just postgres-up    - Start PostgreSQL only"
    @echo "  just redis-up       - Start Redis only"
    @echo "  just kafka-up       - Start Kafka + Zookeeper"
    @echo "  just services-logs  - Show service logs"
    @echo "  just services-status - Check service health"
    @echo ""
    @echo "🚀 Examples:"
    @echo "  just run-basic      - Basic usage example"
    @echo "  just run-kg         - Knowledge graph example"
    @echo "  just run-updates    - Intelligent updates demo"
    @echo "  just run-distributed - Distributed systems demo"
    @echo "  just run-integrations - External integrations demo"
    @echo "  just run-analytics  - Advanced analytics demo"
    @echo "  just run-combined   - Full system demo"
    @echo ""
    @echo "📊 Monitoring:"
    @echo "  just show-visuals   - Open latest visualizations"
    @echo "  just health-check   - Check all system health"
    @echo ""

# 🔧 Development Commands

# Build the project
build:
    @echo "🔨 Building Synaptic..."
    cargo build

# Build with all features
build-all:
    @echo "🔨 Building Synaptic with all features..."
    cargo build --features "distributed,external-integrations,embeddings,analytics"

# Run all tests
test:
    @echo "🧪 Running all tests..."
    cargo test

# Run tests quietly
test-quiet:
    @echo "🧪 Running tests quietly..."
    cargo test --quiet

# Run tests with all features
test-features:
    @echo "🧪 Running tests with all features..."
    cargo test --features "distributed,external-integrations,embeddings,analytics"

# Run specific test suites
test-integration:
    @echo "🧪 Running integration tests..."
    cargo test integration_tests

test-embeddings:
    @echo "🧪 Running embeddings tests..."
    cargo test --features embeddings embeddings_tests

test-distributed:
    @echo "🧪 Running distributed tests..."
    cargo test --features distributed distributed_tests

# Run performance benchmarks
bench:
    @echo "⚡ Running performance benchmarks..."
    cargo test --release -- --ignored performance

# Check code without building
check:
    @echo "✅ Checking code..."
    cargo check

# Format code
fmt:
    @echo "🎨 Formatting code..."
    cargo fmt

# Run clippy lints
clippy:
    @echo "📎 Running clippy..."
    cargo clippy -- -D warnings

# Clean build artifacts
clean:
    @echo "🧹 Cleaning build artifacts..."
    cargo clean

# 🐳 Infrastructure Commands

# Complete setup - environment and services
setup:
    @echo "🚀 Setting up Synaptic environment..."
    @if [ ! -f .env ]; then echo "📝 Creating .env file..."; cp .env.example .env 2>/dev/null || echo "⚠️  Please create .env file manually"; fi
    @echo "🐳 Starting all services..."
    docker-compose up -d
    @echo "⏳ Waiting for services to be ready..."
    sleep 10
    just health-check
    @echo "✅ Setup complete!"

# Start all Docker services
services-up:
    @echo "🐳 Starting all Docker services..."
    docker-compose up -d

# Stop all Docker services
services-down:
    @echo "🛑 Stopping all Docker services..."
    docker-compose down

# Start PostgreSQL only
postgres-up:
    @echo "🐘 Starting PostgreSQL..."
    docker-compose up -d postgres

# Start Redis only
redis-up:
    @echo "🔴 Starting Redis..."
    docker-compose up -d redis

# Start Kafka infrastructure
kafka-up:
    @echo "📡 Starting Kafka infrastructure..."
    docker-compose up -d zookeeper kafka

# Show service logs
services-logs:
    @echo "📋 Showing service logs..."
    docker-compose logs -f

# Check service status
services-status:
    @echo "🔍 Checking service status..."
    docker-compose ps

# Health check for all services
health-check:
    @echo "🏥 Checking system health..."
    @echo "PostgreSQL:" && docker-compose exec -T postgres pg_isready -U synaptic_user -d synaptic_db || echo "❌ PostgreSQL not ready"
    @echo "Redis:" && docker-compose exec -T redis redis-cli ping || echo "❌ Redis not ready"
    @echo "Kafka:" && docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1 && echo "✅ Kafka ready" || echo "❌ Kafka not ready"

# 🚀 Example Commands

# Basic usage example
run-basic:
    @echo "🧠 Running basic usage example..."
    cargo run --example basic_usage

# Knowledge graph example
run-kg:
    @echo "🕸️ Running knowledge graph example..."
    cargo run --example knowledge_graph_usage

# Intelligent updates demo
run-updates:
    @echo "🔄 Running intelligent updates demo..."
    cargo run --example intelligent_updates

# Simple updates demo
run-simple:
    @echo "🔄 Running simple updates demo..."
    cargo run --example simple_intelligent_updates

# Distributed systems demo (requires Kafka)
run-distributed:
    @echo "🕸️ Running distributed systems demo..."
    @echo "🔍 Checking Kafka availability..."
    @docker-compose ps kafka | grep -q "Up" || (echo "❌ Kafka not running. Starting..." && just kafka-up && sleep 15)
    cargo run --example phase2_distributed_system --features "distributed,embeddings"

# External integrations demo (requires PostgreSQL + Redis)
run-integrations:
    @echo "🔗 Running external integrations demo..."
    @echo "🔍 Checking services availability..."
    @docker-compose ps postgres | grep -q "Up" || (echo "❌ PostgreSQL not running. Starting..." && just postgres-up && sleep 10)
    @docker-compose ps redis | grep -q "Up" || (echo "❌ Redis not running. Starting..." && just redis-up && sleep 5)
    cargo run --example real_integrations --features external-integrations

# Advanced analytics demo
run-analytics:
    @echo "📊 Running advanced analytics demo..."
    cargo run --example phase3_analytics --features analytics

# Combined full system demo (requires all services)
run-combined:
    @echo "🎯 Running combined full system demo..."
    @echo "🔍 Ensuring all services are running..."
    just services-up
    @echo "⏳ Waiting for services to be ready..."
    sleep 15
    just health-check
    cargo run --example combined_full_system --features "distributed,external-integrations,embeddings,analytics"

# 📊 Monitoring and Visualization Commands

# Open latest visualizations
show-visuals:
    @echo "📊 Opening latest visualizations..."
    @ls -t visualizations/memory_network_*.png 2>/dev/null | head -1 | xargs -I {} sh -c 'echo "🕸️ Opening memory network: {}"; open "{}" 2>/dev/null || echo "📁 Latest network: {}"'
    @ls -t visualizations/analytics_timeline_*.png 2>/dev/null | head -1 | xargs -I {} sh -c 'echo "📈 Opening analytics timeline: {}"; open "{}" 2>/dev/null || echo "📁 Latest timeline: {}"'
    @test -d visualizations && ls visualizations/*.png >/dev/null 2>&1 || echo "📂 No visualizations found. Run an example with visualization features first."

# List all visualizations
list-visuals:
    @echo "📊 Available visualizations:"
    @ls -la visualizations/ 2>/dev/null || echo "📂 No visualizations directory found"

# Clean old visualizations (keep last 5)
clean-visuals:
    @echo "🧹 Cleaning old visualizations (keeping last 5)..."
    @cd visualizations && ls -t memory_network_*.png 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    @cd visualizations && ls -t analytics_timeline_*.png 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    @echo "✅ Cleanup complete"

# Show visualization statistics
visual-stats:
    @echo "📊 Visualization Statistics:"
    @echo "Memory Networks: $(ls visualizations/memory_network_*.png 2>/dev/null | wc -l | tr -d ' ')"
    @echo "Analytics Timelines: $(ls visualizations/analytics_timeline_*.png 2>/dev/null | wc -l | tr -d ' ')"
    @echo "Total Size: $(du -sh visualizations/ 2>/dev/null | cut -f1 || echo '0B')"
    @echo "Latest Files:"
    @ls -lt visualizations/*.png 2>/dev/null | head -5 | awk '{print "  " $9 " (" $5 " bytes)"}' || echo "  No visualizations found"

# Generate sample visualizations for demo
demo-visuals:
    @echo "🎨 Generating demo visualizations..."
    @echo "🔄 Running knowledge graph example..."
    cargo run --example knowledge_graph_usage --quiet
    @echo "🔄 Running analytics example..."
    cargo run --example phase3_analytics --features analytics --quiet
    @echo "✅ Demo visualizations generated!"
    just show-visuals

# 🔧 Utility Commands

# Install development dependencies
install-deps:
    @echo "📦 Installing development dependencies..."
    cargo install cargo-watch cargo-expand cargo-audit

# Watch for changes and run tests
watch:
    @echo "👀 Watching for changes..."
    cargo watch -x test

# Security audit
audit:
    @echo "🔒 Running security audit..."
    cargo audit

# Generate documentation
docs:
    @echo "📚 Generating documentation..."
    cargo doc --open --features "distributed,external-integrations,embeddings,analytics"

# Quick development cycle
dev: fmt clippy test
    @echo "✅ Development cycle complete!"

# Full CI pipeline
ci: fmt clippy test-features bench
    @echo "✅ CI pipeline complete!"

# Reset everything (clean + rebuild)
reset: clean services-down
    @echo "🔄 Resetting everything..."
    just build-all
    @echo "✅ Reset complete!"
