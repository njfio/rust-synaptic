# Makefile for Synaptic AI Agent Memory System
# Provides organized test execution and development commands

.PHONY: help test test-all test-critical test-high test-medium test-low
.PHONY: test-core test-integration test-security test-performance test-lifecycle
.PHONY: test-multimodal test-temporal test-analytics test-search test-external
.PHONY: test-logging test-visualization test-sync
.PHONY: build clean fmt clippy check coverage audit docs
.PHONY: ci-setup ci-test ci-full
.PHONY: docker-build docker-up docker-down docker-logs docker-clean
.PHONY: install-hooks dev watch bench format lint

# Default target
help:
	@echo "Synaptic AI Agent Memory - Development Commands"
	@echo ""
	@echo "Test Commands:"
	@echo "  test-all          Run all 161 tests across all categories"
	@echo "  test-critical     Run critical tests (core, integration, security)"
	@echo "  test-high         Run high priority tests (performance, lifecycle, multimodal)"
	@echo "  test-medium       Run medium priority tests (temporal, analytics, search, external)"
	@echo "  test-low          Run low priority tests (logging, visualization, sync)"
	@echo ""
	@echo "Category-Specific Tests:"
	@echo "  test-core         Run core library tests (29 tests)"
	@echo "  test-integration  Run integration tests (13 tests)"
	@echo "  test-security     Run security tests (28 tests)"
	@echo "  test-performance  Run performance tests (28 tests)"
	@echo "  test-lifecycle    Run lifecycle tests (11 tests)"
	@echo "  test-multimodal   Run multimodal tests (21 tests)"
	@echo "  test-temporal     Run temporal tests (17 tests)"
	@echo "  test-analytics    Run analytics tests (19 tests)"
	@echo "  test-search       Run search tests (6 tests)"
	@echo "  test-external     Run external integration tests (8 tests)"
	@echo "  test-logging      Run logging tests (20 tests)"
	@echo "  test-visualization Run visualization tests (2 tests)"
	@echo "  test-sync         Run sync tests (2 tests)"
	@echo ""
	@echo "Development Commands:"
	@echo "  build             Build the project"
	@echo "  clean             Clean build artifacts"
	@echo "  fmt               Format code with rustfmt"
	@echo "  clippy            Run clippy linter"
	@echo "  check             Run cargo check"
	@echo "  coverage          Generate code coverage report"
	@echo "  audit             Run security audit"
	@echo "  docs              Build documentation"
	@echo ""
	@echo "CI Commands:"
	@echo "  ci-setup          Setup CI environment"
	@echo "  ci-test           Run CI test suite"
	@echo "  ci-full           Run full CI pipeline"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-up         Start all services with Docker Compose"
	@echo "  docker-down       Stop all services"
	@echo "  docker-logs       Show Docker Compose logs"
	@echo "  docker-clean      Clean Docker resources"
	@echo ""
	@echo "Utilities:"
	@echo "  install-hooks     Install pre-commit hooks"
	@echo "  dev               Quick development check (fmt + clippy + test)"
	@echo "  watch             Watch for changes and run tests"
	@echo "  bench             Run performance benchmarks"

# Test execution using the organized test runner
test-all:
	@echo "Running all 161 tests across 14 categories..."
	./scripts/run_tests.sh all

test-critical:
	@echo "Running critical priority tests..."
	./scripts/run_tests.sh critical

test-high:
	@echo "Running high priority tests..."
	./scripts/run_tests.sh high

test-medium:
	@echo "Running medium priority tests..."
	./scripts/run_tests.sh medium

test-low:
	@echo "Running low priority tests..."
	./scripts/run_tests.sh low

# Category-specific test targets
test-core:
	@echo "Running core library tests..."
	./scripts/run_tests.sh core

test-integration:
	@echo "Running integration tests..."
	./scripts/run_tests.sh integration

test-security:
	@echo "Running security tests..."
	./scripts/run_tests.sh security

test-performance:
	@echo "Running performance tests..."
	./scripts/run_tests.sh performance

test-lifecycle:
	@echo "Running lifecycle tests..."
	./scripts/run_tests.sh lifecycle

test-multimodal:
	@echo "Running multimodal tests..."
	./scripts/run_tests.sh multimodal

test-temporal:
	@echo "Running temporal tests..."
	./scripts/run_tests.sh temporal

test-analytics:
	@echo "Running analytics tests..."
	./scripts/run_tests.sh analytics

test-search:
	@echo "Running search tests..."
	./scripts/run_tests.sh search

test-external:
	@echo "Running external integration tests..."
	./scripts/run_tests.sh external

test-logging:
	@echo "Running logging tests..."
	./scripts/run_tests.sh logging

test-visualization:
	@echo "Running visualization tests..."
	./scripts/run_tests.sh visualization

test-sync:
	@echo "Running sync tests..."
	./scripts/run_tests.sh sync

# Development commands
build:
	@echo "Building Synaptic..."
	cargo build --all-features

clean:
	@echo "Cleaning build artifacts..."
	cargo clean

fmt:
	@echo "Formatting code..."
	cargo fmt --all

clippy:
	@echo "Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

check:
	@echo "Running cargo check..."
	cargo check --all-features

coverage:
	@echo "Generating code coverage..."
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "Coverage report generated: lcov.info"

audit:
	@echo "Running security audit..."
	cargo audit

docs:
	@echo "Building documentation..."
	cargo doc --all-features --no-deps --open

# CI commands
ci-setup:
	@echo "Setting up CI environment..."
	rustup component add rustfmt clippy llvm-tools-preview
	cargo install cargo-llvm-cov cargo-audit

ci-test:
	@echo "Running CI test suite..."
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo build --verbose --all-features
	./scripts/run_tests.sh all

ci-full: ci-setup ci-test coverage audit docs
	@echo "Full CI pipeline completed successfully!"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t synaptic:latest .

docker-up:
	@echo "Starting services..."
	docker-compose up -d

docker-down:
	@echo "Stopping services..."
	docker-compose down

docker-logs:
	@echo "Showing logs..."
	docker-compose logs -f

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f

# Utility commands
install-hooks:
	@echo "Installing git hooks..."
	@if [ -d .git ]; then \
		cp scripts/pre-commit.sh .git/hooks/pre-commit && \
		chmod +x .git/hooks/pre-commit && \
		echo "Pre-commit hook installed!"; \
	else \
		echo "Not a git repository!"; \
		exit 1; \
	fi

dev: fmt clippy test-critical
	@echo "Development checks completed!"

watch:
	@echo "Watching for changes..."
	cargo watch -x check -x test -x clippy

bench:
	@echo "Running benchmarks..."
	cargo bench

# Convenience aliases
test: test-all
lint: clippy
format: fmt
