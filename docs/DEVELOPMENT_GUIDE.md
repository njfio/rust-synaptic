# Synaptic Development Guide

This guide provides comprehensive information for developers working on the Synaptic intelligent memory system.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation Standards](#documentation-standards)
7. [Performance Guidelines](#performance-guidelines)
8. [Security Considerations](#security-considerations)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Troubleshooting](#troubleshooting)

## Development Environment Setup

### Prerequisites

- **Rust**: Version 1.70+ (stable, beta, nightly supported)
- **Git**: Version 2.30+
- **Docker**: For containerized development (optional)
- **PostgreSQL**: For database integration testing (optional)
- **Redis**: For cache integration testing (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/njfio/rust-synaptic.git
cd rust-synaptic

# Install Rust dependencies
cargo build

# Run tests to verify setup
cargo test

# Install development tools
cargo install cargo-watch cargo-llvm-cov cargo-audit
```

### IDE Setup

#### VS Code
Recommended extensions:
- `rust-analyzer`: Rust language support
- `CodeLLDB`: Debugging support
- `Better TOML`: TOML file support
- `GitLens`: Git integration

#### RustRover/IntelliJ
- Install Rust plugin
- Configure code style settings
- Set up debugging configurations

### Environment Variables

Create a `.env` file in the project root:

```env
# Database configuration
DATABASE_URL=postgresql://user:password@localhost/synaptic_dev
REDIS_URL=redis://localhost:6379

# Security configuration
ENCRYPTION_KEY=your-32-byte-encryption-key-here
JWT_SECRET=your-jwt-secret-here

# Development settings
RUST_LOG=debug
RUST_BACKTRACE=1
```

## Project Structure

```
synaptic/
├── src/                          # Source code
│   ├── lib.rs                   # Library entry point
│   ├── config/                  # Configuration management
│   ├── error/                   # Error types and handling
│   ├── memory/                  # Core memory system
│   │   ├── embeddings/         # Vector embeddings
│   │   ├── knowledge_graph/    # Graph operations
│   │   ├── management/         # Memory management
│   │   ├── storage/            # Storage backends
│   │   ├── temporal/           # Temporal analysis
│   │   └── types.rs            # Core data types
│   ├── integrations/           # External integrations
│   ├── multimodal/             # Multimodal processing
│   ├── security/               # Security and privacy
│   └── utils/                  # Utility functions
├── tests/                       # Integration tests
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── scripts/                     # Development scripts
├── .github/                     # GitHub workflows
└── Cargo.toml                   # Project configuration
```

### Module Organization

Each module follows a consistent structure:
```
module/
├── mod.rs                       # Module entry point
├── types.rs                     # Type definitions
├── error.rs                     # Module-specific errors
├── config.rs                    # Configuration
├── implementation.rs            # Core implementation
└── tests.rs                     # Unit tests
```

## Development Workflow

### Git Workflow

1. **Feature Branches**: Create feature branches from `develop`
2. **Conventional Commits**: Use conventional commit format
3. **Pull Requests**: All changes go through PR review
4. **CI/CD**: Automated testing and deployment

#### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(memory): add advanced similarity search
fix(storage): resolve file locking issue
docs(api): update API documentation
test(security): add encryption test cases
```

### Development Commands

```bash
# Development with auto-reload
cargo watch -x check -x test -x run

# Run specific test category
./scripts/test-runner.sh category security

# Generate documentation
cargo doc --all-features --open

# Run benchmarks
cargo test --test performance_tests -- --ignored

# Security audit
cargo audit

# Format code
cargo fmt

# Lint code
cargo clippy --all-targets --all-features -- -D warnings
```

## Coding Standards

### Rust Style Guidelines

Follow the official Rust style guide with these additions:

#### Naming Conventions
- **Types**: `PascalCase` (e.g., `MemoryEntry`)
- **Functions**: `snake_case` (e.g., `store_memory`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_MEMORY_SIZE`)
- **Modules**: `snake_case` (e.g., `knowledge_graph`)

#### Code Organization
```rust
// 1. External crate imports
use std::collections::HashMap;
use tokio::time::Duration;

// 2. Internal crate imports
use crate::memory::types::MemoryEntry;
use crate::error::MemoryError;

// 3. Type definitions
pub struct MemoryManager {
    // fields
}

// 4. Implementation blocks
impl MemoryManager {
    // associated functions first
    pub fn new() -> Self { }
    
    // public methods
    pub async fn store(&self) -> Result<()> { }
    
    // private methods
    async fn internal_operation(&self) -> Result<()> { }
}

// 5. Trait implementations
impl Default for MemoryManager {
    fn default() -> Self { }
}

// 6. Tests (in separate module)
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_storage() {
        // test implementation
    }
}
```

#### Error Handling
- Use `Result<T, E>` for fallible operations
- Create specific error types for each module
- Provide meaningful error messages
- Use `?` operator for error propagation

```rust
pub enum MemoryError {
    #[error("Memory not found: {key}")]
    NotFound { key: String },
    
    #[error("Storage error: {source}")]
    Storage { source: StorageError },
}
```

#### Documentation
- Document all public APIs with `///`
- Include examples in documentation
- Use `#[doc = include_str!("../README.md")]` for module docs

```rust
/// Stores a memory entry in the system.
///
/// # Arguments
///
/// * `key` - Unique identifier for the memory
/// * `content` - The content to store
/// * `metadata` - Optional metadata
///
/// # Returns
///
/// Returns the ID of the stored memory entry.
///
/// # Examples
///
/// ```rust
/// let memory = AgentMemory::new().await?;
/// let id = memory.store("key", "content", None).await?;
/// ```
pub async fn store(&self, key: &str, content: &str, metadata: Option<MemoryMetadata>) -> Result<String> {
    // implementation
}
```

### Performance Guidelines

#### Async Programming
- Use `async`/`await` for I/O operations
- Avoid blocking operations in async contexts
- Use `tokio::spawn` for CPU-intensive tasks
- Implement proper cancellation handling

#### Memory Management
- Use `Arc<T>` for shared ownership
- Use `Rc<T>` for single-threaded shared ownership
- Prefer `&str` over `String` for temporary strings
- Use `Cow<str>` for conditional ownership

#### Optimization Patterns
```rust
// Prefer iterators over loops
let results: Vec<_> = items
    .iter()
    .filter(|item| item.is_valid())
    .map(|item| item.process())
    .collect();

// Use lazy evaluation
let expensive_computation = || {
    // expensive operation
};

// Cache frequently accessed data
use std::sync::OnceLock;
static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
```

## Testing Guidelines

### Test Organization

Tests are organized into categories:
- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **Security Tests**: Test security features
- **End-to-End Tests**: Test complete workflows

### Test Naming Convention

```rust
#[tokio::test]
async fn test_<functionality>_<scenario>_<expected_outcome>() {
    // test implementation
}

// Examples:
async fn test_memory_store_valid_input_returns_id() { }
async fn test_memory_store_duplicate_key_returns_error() { }
async fn test_search_empty_query_returns_all_results() { }
```

### Test Structure

```rust
#[tokio::test]
async fn test_memory_storage() {
    // Arrange
    let memory = AgentMemory::new().await.unwrap();
    let test_key = "test_key";
    let test_content = "test content";
    
    // Act
    let result = memory.store(test_key, test_content, None).await;
    
    // Assert
    assert!(result.is_ok());
    let stored_memory = memory.retrieve(test_key).await.unwrap();
    assert!(stored_memory.is_some());
    assert_eq!(stored_memory.unwrap().content, test_content);
}
```

### Test Utilities

```rust
// Test helpers
pub mod test_utils {
    use super::*;
    
    pub async fn create_test_memory() -> AgentMemory {
        AgentMemory::new().await.unwrap()
    }
    
    pub fn create_test_entry(key: &str, content: &str) -> MemoryEntry {
        MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            key: key.to_string(),
            content: content.to_string(),
            // ... other fields
        }
    }
}
```

### Coverage Requirements

- **Minimum Coverage**: 90% line coverage
- **Critical Paths**: 100% coverage for security and data integrity
- **Error Paths**: All error conditions must be tested
- **Edge Cases**: Boundary conditions and edge cases

## Documentation Standards

### Code Documentation

- All public APIs must have documentation
- Include examples for complex functions
- Document error conditions
- Use proper Rust doc syntax

### Architecture Documentation

- Keep architecture docs up to date
- Include diagrams for complex systems
- Document design decisions and trade-offs
- Maintain API changelog

### User Documentation

- Provide clear setup instructions
- Include comprehensive examples
- Document configuration options
- Maintain troubleshooting guides

## Security Considerations

### Secure Coding Practices

- Validate all inputs
- Use secure random number generation
- Implement proper authentication/authorization
- Follow principle of least privilege
- Regular security audits

### Sensitive Data Handling

```rust
// Use SecureString for sensitive data
use zeroize::Zeroize;

#[derive(Zeroize)]
struct SensitiveData {
    secret: String,
}

impl Drop for SensitiveData {
    fn drop(&mut self) {
        self.zeroize();
    }
}
```

### Dependency Management

- Regular dependency updates
- Security vulnerability scanning
- Minimal dependency principle
- Pin critical dependencies

## Contributing Guidelines

### Before Contributing

1. Read the code of conduct
2. Check existing issues and PRs
3. Discuss major changes in issues first
4. Follow the development workflow

### Pull Request Process

1. Create feature branch from `develop`
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit PR with clear description
6. Address review feedback
7. Merge after approval

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Error handling is proper

## Troubleshooting

### Common Issues

#### Compilation Errors
```bash
# Clear cargo cache
cargo clean

# Update dependencies
cargo update

# Check for conflicting features
cargo tree --duplicates
```

#### Test Failures
```bash
# Run specific test
cargo test test_name -- --nocapture

# Run tests with logging
RUST_LOG=debug cargo test

# Run tests serially
cargo test -- --test-threads=1
```

#### Performance Issues
```bash
# Profile with perf
cargo build --release
perf record --call-graph=dwarf target/release/synaptic
perf report

# Memory profiling
valgrind --tool=massif target/release/synaptic
```

### Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions
- **Community**: Join project Discord/Slack

### Development Tools

```bash
# Install useful tools
cargo install cargo-watch cargo-expand cargo-udeps
cargo install flamegraph cargo-bloat cargo-audit

# Use cargo-watch for development
cargo watch -x check -x test -x run

# Expand macros for debugging
cargo expand

# Find unused dependencies
cargo udeps

# Generate flame graphs
cargo flamegraph --bin synaptic
```

## Additional Resources

### Learning Resources
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Async Programming in Rust](https://rust-lang.github.io/async-book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Project-Specific Resources
- [API Documentation](./API_DOCUMENTATION.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Test Organization](./TEST_ORGANIZATION.md)
- [Phase Documentation](./PHASE5_MULTIMODAL_CROSSPLATFORM.md)
