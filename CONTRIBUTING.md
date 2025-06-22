# Contributing to Synaptic

Thank you for your interest in contributing to Synaptic! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Requirements](#testing-requirements)
8. [Documentation](#documentation)
9. [Issue Reporting](#issue-reporting)
10. [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## Getting Started

### Prerequisites

- Rust 1.79 or later
- Git
- Basic understanding of Rust and async programming
- Familiarity with AI/ML concepts (helpful but not required)

### Development Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install additional tools
cargo install cargo-watch cargo-tarpaulin just

# Optional: Install Docker for integration testing
# Follow Docker installation instructions for your platform
```

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rust-synaptic.git
cd rust-synaptic

# Add upstream remote
git remote add upstream https://github.com/njfio/rust-synaptic.git
```

### 2. Set Up Development Environment

```bash
# Install dependencies and build
cargo build

# Run tests to ensure everything works
cargo test

# Start development services (optional)
just setup
```

### 3. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix existing issues or problems
- **Features**: Add new functionality or improve existing features
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize performance or reduce resource usage
- **Refactoring**: Improve code structure without changing functionality

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions about your idea
2. **Create an issue**: For significant changes, create an issue to discuss the approach
3. **Small changes**: For small fixes, you can directly create a pull request

### Development Workflow

1. **Write code**: Implement your changes following our coding standards
2. **Add tests**: Ensure your changes are covered by tests
3. **Update documentation**: Update relevant documentation
4. **Run tests**: Ensure all tests pass
5. **Commit changes**: Use conventional commit messages
6. **Push and create PR**: Push your branch and create a pull request

## Pull Request Process

### 1. Prepare Your Pull Request

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run the full test suite
cargo test --all-features

# Run linting
cargo clippy --all-features -- -D warnings

# Check formatting
cargo fmt -- --check

# Run benchmarks (if applicable)
cargo bench --features "full"
```

### 2. Create Pull Request

- Use a clear, descriptive title
- Fill out the pull request template
- Reference any related issues
- Include screenshots for UI changes
- Add reviewers if you know who should review

### 3. Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is commented, particularly in hard-to-understand areas
- [ ] Corresponding changes to documentation made
- [ ] No new warnings introduced
```

### 4. Review Process

- All PRs require at least one review
- Address feedback promptly and professionally
- Update your PR based on review comments
- Maintain a clean commit history

## Coding Standards

### Rust Style Guidelines

```rust
// Use descriptive names
fn calculate_memory_similarity(entry1: &MemoryEntry, entry2: &MemoryEntry) -> f64 {
    // Implementation
}

// Prefer explicit error handling
async fn store_memory(key: &str, value: &str) -> Result<(), SynapticError> {
    // Use ? operator for error propagation
    validate_key(key)?;
    storage.store(key, value).await?;
    Ok(())
}

// Use appropriate visibility
pub struct PublicStruct {
    pub public_field: String,
    private_field: i32,
}

// Document public APIs
/// Stores a memory entry in the system.
/// 
/// # Arguments
/// 
/// * `key` - The unique identifier for the memory
/// * `value` - The content to store
/// 
/// # Returns
/// 
/// Returns `Ok(())` on success, or a `SynapticError` on failure.
/// 
/// # Examples
/// 
/// ```
/// let memory = AgentMemory::new(config).await?;
/// memory.store("greeting", "Hello, World!").await?;
/// ```
pub async fn store(&mut self, key: &str, value: &str) -> Result<()> {
    // Implementation
}
```

### Code Organization

- **Modules**: Organize code into logical modules
- **Traits**: Use traits for abstraction and testability
- **Error Handling**: Use `Result` types consistently
- **Async**: Use async/await for I/O operations
- **Dependencies**: Minimize external dependencies

### Performance Guidelines

- **Avoid allocations**: Minimize unnecessary allocations in hot paths
- **Use references**: Prefer borrowing over cloning when possible
- **Batch operations**: Implement batch operations for better performance
- **Lazy evaluation**: Use lazy evaluation where appropriate
- **Profiling**: Profile performance-critical code

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 90% for new code
- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark performance-critical code
- **Security tests**: Test security-related functionality

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_store_success() {
        // Arrange
        let mut memory = create_test_memory().await;
        
        // Act
        let result = memory.store("test_key", "test_value").await;
        
        // Assert
        assert!(result.is_ok());
        let retrieved = memory.retrieve("test_key").await.unwrap();
        assert_eq!(retrieved.unwrap().value, "test_value");
    }

    #[tokio::test]
    async fn test_memory_store_invalid_key() {
        let mut memory = create_test_memory().await;
        
        let result = memory.store("", "value").await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SynapticError::ValidationError(_) => {}, // Expected
            _ => panic!("Expected ValidationError"),
        }
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with coverage
cargo tarpaulin --all-features

# Run specific test
cargo test test_memory_store

# Run tests in watch mode
cargo watch -x test
```

## Documentation

### Documentation Requirements

- **Public APIs**: All public functions, structs, and modules must be documented
- **Examples**: Include examples in documentation
- **Error conditions**: Document possible errors
- **Safety**: Document any unsafe code or invariants

### Documentation Style

```rust
/// A brief one-line description.
/// 
/// A more detailed description that explains what this function does,
/// when you might want to use it, and any important considerations.
/// 
/// # Arguments
/// 
/// * `param1` - Description of the first parameter
/// * `param2` - Description of the second parameter
/// 
/// # Returns
/// 
/// Description of what the function returns.
/// 
/// # Errors
/// 
/// This function will return an error if:
/// * Condition 1 occurs
/// * Condition 2 happens
/// 
/// # Examples
/// 
/// ```
/// use synaptic::AgentMemory;
/// 
/// let memory = AgentMemory::new(config).await?;
/// let result = memory.some_function(param1, param2).await?;
/// ```
/// 
/// # Panics
/// 
/// This function panics if the invariant X is violated.
/// 
/// # Safety
/// 
/// This function is safe to call as long as...
pub async fn some_function(&self, param1: Type1, param2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

### Building Documentation

```bash
# Generate documentation
cargo doc --all-features --no-deps

# Open documentation in browser
cargo doc --all-features --no-deps --open

# Check for documentation warnings
cargo doc --all-features 2>&1 | grep warning
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Rust version, Synaptic version
- **Logs**: Relevant log output or error messages

### Feature Requests

For feature requests, please include:

- **Description**: Clear description of the feature
- **Use case**: Why this feature would be useful
- **Proposed solution**: How you think it should work
- **Alternatives**: Alternative solutions you've considered

### Issue Templates

Use the provided issue templates when creating new issues. This helps ensure all necessary information is included.

## Community

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code review and collaboration

### Getting Help

- Check the [documentation](docs/) first
- Search existing issues and discussions
- Create a new issue or discussion if needed
- Be patient and respectful when asking for help

### Recognition

Contributors will be recognized in:

- The project's README
- Release notes for significant contributions
- The CHANGELOG for their contributions

## Development Tips

### Useful Commands

```bash
# Watch for changes and run tests
cargo watch -x test

# Run clippy with all features
cargo clippy --all-features -- -D warnings

# Format code
cargo fmt

# Check for unused dependencies
cargo machete

# Update dependencies
cargo update

# Run benchmarks
cargo bench --features "full"
```

### IDE Setup

Recommended VS Code extensions:
- rust-analyzer
- CodeLLDB (for debugging)
- Better TOML
- GitLens

### Debugging

```rust
// Use tracing for structured logging
use tracing::{debug, info, warn, error};

debug!("Processing memory entry: {}", key);
info!("Memory stored successfully");
warn!("Memory consolidation taking longer than expected");
error!("Failed to store memory: {}", error);
```

Thank you for contributing to Synaptic! Your contributions help make this project better for everyone.
