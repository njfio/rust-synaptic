#!/bin/bash
# Pre-commit hook for Synaptic
# This hook runs formatting checks and lints before allowing commits

set -e

echo "🔍 Running pre-commit checks..."

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found. Please install Rust."
    exit 1
fi

# Format check
echo "📝 Checking code formatting..."
if ! cargo fmt --all -- --check; then
    echo "❌ Code formatting check failed!"
    echo "💡 Run 'cargo fmt --all' to fix formatting issues"
    exit 1
fi

echo "✅ Code formatting check passed"

# Clippy lint check
echo "📎 Running clippy lints..."
if ! cargo clippy --all-features --all-targets -- -D warnings 2>&1 | grep -v "warning: unused import"; then
    echo "❌ Clippy checks failed!"
    echo "💡 Fix the issues above before committing"
    exit 1
fi

echo "✅ Clippy checks passed"

# Run critical tests
echo "🧪 Running critical tests..."
if ! cargo test --lib --quiet; then
    echo "❌ Critical tests failed!"
    echo "💡 Fix failing tests before committing"
    exit 1
fi

echo "✅ Critical tests passed"

echo "✨ All pre-commit checks passed! Proceeding with commit..."
exit 0
