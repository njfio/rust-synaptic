# Contributing to Synaptic 🧠

Thank you for your interest in contributing to Synaptic! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Rust 1.79 or later
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rust-synaptic.git
   cd rust-synaptic
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/njfio/rust-synaptic.git
   ```
4. **Install dependencies**:
   ```bash
   cargo build
   ```
5. **Run tests** to ensure everything works:
   ```bash
   cargo test
   ```

## 🔧 Development Workflow

### Before You Start

1. **Check existing issues** to see if your idea is already being worked on
2. **Create an issue** to discuss new features or major changes
3. **Keep changes focused** - one feature or fix per pull request

### Making Changes

1. **Create a new branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the full test suite**:
   ```bash
   cargo test
   cargo clippy
   cargo fmt --check
   ```

### Submitting Changes

1. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "feat: add intelligent memory merging algorithm"
   ```
2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Code Style

- Follow standard Rust formatting (`cargo fmt`)
- Use `cargo clippy` and address all warnings
- Write clear, self-documenting code
- Add comments for complex algorithms or business logic

### Naming Conventions

- Use descriptive names for functions, variables, and types
- Follow Rust naming conventions (snake_case for functions/variables, PascalCase for types)
- Prefix private functions with underscore when appropriate

### Documentation

- Add rustdoc comments for all public APIs
- Include examples in documentation when helpful
- Update README.md for significant changes

### Testing

- Write unit tests for all new functionality
- Add integration tests for complex features
- Ensure all tests pass before submitting
- Aim for good test coverage

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_knowledge_graph

# Run benchmarks
cargo bench
```

### Writing Tests

- Place unit tests in the same file as the code being tested
- Use integration tests in the `tests/` directory for end-to-end testing
- Mock external dependencies when appropriate
- Test both success and error cases

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused and atomic

### PR Description

Include in your PR description:
- **What** changes you made
- **Why** you made them
- **How** to test the changes
- **Any breaking changes** or migration notes

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:
- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Rust version, etc.)
- **Code samples** if applicable

### Feature Requests

For new features, please include:
- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach
- **Breaking changes** considerations

## 🏷️ Commit Message Format

We follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(knowledge-graph): add intelligent node merging
fix(storage): resolve memory leak in file backend
docs(readme): update installation instructions
```

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Performance optimizations
- Additional storage backends
- Vector embedding support
- Memory compression algorithms

### Medium Priority
- Additional relationship types
- Graph visualization tools
- Benchmarking improvements
- Documentation enhancements

### Good First Issues
- Adding more examples
- Improving error messages
- Writing additional tests
- Documentation fixes

## 💬 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Code Review**: Ask questions in PR comments

## 📄 License

By contributing to Synaptic, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- The project README
- Release notes for significant contributions
- The project's contributor list

Thank you for helping make Synaptic better! 🧠✨
