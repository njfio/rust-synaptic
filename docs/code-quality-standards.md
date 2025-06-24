# Code Quality Standards

This document defines the code quality standards, linting rules, and automated checks for the Synaptic memory system to ensure consistent, maintainable, and production-ready code.

## Code Quality Principles

### 1. Safety First
- **No `unwrap()` in production code**: Use proper error handling with `Result` types
- **No `panic!` in library code**: Handle errors gracefully and return appropriate error types
- **Memory safety**: Avoid unsafe code unless absolutely necessary and well-documented
- **Thread safety**: Use appropriate synchronization primitives and avoid data races

### 2. Clarity and Maintainability
- **Clear naming**: Use descriptive names for variables, functions, and types
- **Documentation**: All public APIs must have comprehensive documentation
- **Small functions**: Keep functions focused and under 50 lines when possible
- **Consistent style**: Follow Rust community conventions and project-specific guidelines

### 3. Performance and Efficiency
- **Avoid unnecessary allocations**: Use references and borrowing appropriately
- **Efficient algorithms**: Choose appropriate data structures and algorithms
- **Lazy evaluation**: Use lazy initialization and computation when beneficial
- **Resource management**: Properly manage file handles, network connections, and memory

## Linting Configuration

### 1. Clippy Configuration

Create `.clippy.toml`:
```toml
# Clippy configuration for Synaptic
avoid-breaking-exported-api = false
msrv = "1.70.0"

# Allowed lints (use sparingly)
allow = [
    # Allow complex types in some cases
    "clippy::type_complexity",
]

# Denied lints (enforce strictly)
deny = [
    # Safety and correctness
    "clippy::unwrap_used",
    "clippy::expect_used",
    "clippy::panic",
    "clippy::unreachable",
    "clippy::todo",
    "clippy::unimplemented",
    
    # Performance
    "clippy::inefficient_to_string",
    "clippy::clone_on_ref_ptr",
    "clippy::large_enum_variant",
    "clippy::large_stack_arrays",
    
    # Style and clarity
    "clippy::cognitive_complexity",
    "clippy::too_many_arguments",
    "clippy::too_many_lines",
    "clippy::similar_names",
    
    # Documentation
    "clippy::missing_docs_in_private_items",
    "clippy::missing_errors_doc",
    "clippy::missing_panics_doc",
    "clippy::missing_safety_doc",
]

# Warn lints (should be addressed but not blocking)
warn = [
    "clippy::pedantic",
    "clippy::nursery",
    "clippy::cargo",
]
```

### 2. Rustfmt Configuration

Create `.rustfmt.toml`:
```toml
# Rustfmt configuration for Synaptic
edition = "2021"
max_width = 100
hard_tabs = false
tab_spaces = 4

# Import organization
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true

# Function formatting
fn_args_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"

# String formatting
format_strings = true
format_macro_matchers = true

# Comment formatting
comment_width = 80
wrap_comments = true
normalize_comments = true

# Misc formatting
trailing_comma = "Vertical"
match_arm_trailing_comma = true
use_field_init_shorthand = true
use_try_shorthand = true
```

### 3. Cargo Configuration

Update `Cargo.toml` with quality settings:
```toml
[package]
name = "synaptic"
version = "0.1.0"
edition = "2021"
rust-version = "1.70.0"

# Quality metadata
authors = ["Synaptic Team"]
description = "Advanced AI agent memory system"
license = "MIT OR Apache-2.0"
repository = "https://github.com/njfio/rust-synaptic"
documentation = "https://docs.rs/synaptic"
readme = "README.md"
keywords = ["ai", "memory", "agent", "knowledge-graph"]
categories = ["data-structures", "database", "science"]

[lints.rust]
unsafe_code = "forbid"
missing_docs = "warn"
unused_imports = "warn"
unused_variables = "warn"
dead_code = "warn"

[lints.clippy]
# Enforce safety
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
todo = "deny"
unimplemented = "deny"

# Enforce performance
inefficient_to_string = "deny"
clone_on_ref_ptr = "deny"

# Enforce style
cognitive_complexity = "warn"
too_many_arguments = "warn"
missing_errors_doc = "warn"
missing_panics_doc = "warn"
```

## Automated Quality Checks

### 1. Pre-commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt
        language: system
        types: [rust]
        pass_filenames: false
        
      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy
        language: system
        types: [rust]
        pass_filenames: false
        args: [--all-targets, --all-features, --, -D, warnings]
        
      - id: cargo-test
        name: cargo test
        entry: cargo test
        language: system
        types: [rust]
        pass_filenames: false
        args: [--lib, --bins]
        
      - id: cargo-doc
        name: cargo doc
        entry: cargo doc
        language: system
        types: [rust]
        pass_filenames: false
        args: [--no-deps, --document-private-items]
        
      - id: check-unwrap
        name: check for unwrap() in production code
        entry: bash -c 'if grep -r "\.unwrap()" src/ --exclude-dir=tests; then echo "Error: Found unwrap() calls in production code"; exit 1; fi'
        language: system
        types: [rust]
        pass_filenames: false
        
      - id: check-println
        name: check for println! in production code
        entry: bash -c 'if grep -r "println!\|eprintln!" src/ --exclude="src/bin/" --exclude="src/examples/"; then echo "Error: Found println!/eprintln! in production code"; exit 1; fi'
        language: system
        types: [rust]
        pass_filenames: false
```

### 2. GitHub Actions Workflow

Create `.github/workflows/quality.yml`:
```yaml
name: Code Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  formatting:
    name: Check Formatting
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - name: Check formatting
        run: cargo fmt --all -- --check

  linting:
    name: Clippy Linting
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: Run clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  documentation:
    name: Documentation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Check documentation
        run: cargo doc --no-deps --document-private-items --all-features
        env:
          RUSTDOCFLAGS: -D warnings

  security-audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v1.4.1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  dependency-check:
    name: Dependency Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-deny
        run: cargo install cargo-deny
      - name: Check dependencies
        run: cargo deny check

  code-coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
      - name: Generate coverage
        run: cargo tarpaulin --out xml --engine llvm
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./cobertura.xml
          fail_ci_if_error: true

  quality-gates:
    name: Quality Gates
    runs-on: ubuntu-latest
    needs: [formatting, linting, documentation, security-audit, dependency-check]
    steps:
      - name: Quality check passed
        run: echo "All quality checks passed!"
```

### 3. Dependency Management

Create `deny.toml`:
```toml
[graph]
targets = [
    { triple = "x86_64-unknown-linux-gnu" },
    { triple = "x86_64-pc-windows-msvc" },
    { triple = "x86_64-apple-darwin" },
]

[advisories]
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"
notice = "warn"
ignore = []

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
]
deny = [
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-1.0",
    "AGPL-3.0",
]
copyleft = "warn"
allow-osi-fsf-free = "neither"
default = "deny"
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "allow"
highlight = "all"
workspace-default-features = "allow"
external-default-features = "allow"
allow = []
deny = [
    # Deny old versions of security-sensitive crates
    { name = "openssl", version = "<0.10.55" },
    { name = "rustls", version = "<0.21.0" },
]

[sources]
unknown-registry = "warn"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

## Code Review Guidelines

### 1. Review Checklist

**Safety and Correctness**:
- [ ] No `unwrap()` or `expect()` in production code
- [ ] Proper error handling with `Result` types
- [ ] No potential panics or undefined behavior
- [ ] Thread safety considerations addressed

**Performance**:
- [ ] Efficient algorithms and data structures
- [ ] Minimal unnecessary allocations
- [ ] Appropriate use of references vs. owned values
- [ ] No obvious performance bottlenecks

**Code Quality**:
- [ ] Clear and descriptive naming
- [ ] Functions are focused and reasonably sized
- [ ] Proper documentation for public APIs
- [ ] Consistent with project style guidelines

**Testing**:
- [ ] Adequate test coverage for new functionality
- [ ] Tests cover both happy path and error conditions
- [ ] Integration tests for complex interactions
- [ ] Performance tests for critical paths

### 2. Review Process

1. **Automated Checks**: All CI checks must pass before review
2. **Self Review**: Author reviews their own code first
3. **Peer Review**: At least one team member reviews the code
4. **Documentation Review**: Ensure documentation is accurate and complete
5. **Final Approval**: Senior team member approves before merge

## Metrics and Monitoring

### 1. Code Quality Metrics

**Coverage Targets**:
- Line coverage: >90%
- Branch coverage: >85%
- Function coverage: >95%

**Complexity Limits**:
- Cyclomatic complexity: <10 per function
- Cognitive complexity: <15 per function
- Lines per function: <50 (guideline, not strict)

**Documentation Coverage**:
- Public APIs: 100% documented
- Private APIs: >80% documented
- Examples for complex functions

### 2. Quality Dashboard

Track quality metrics over time:
- Test coverage trends
- Code complexity metrics
- Security vulnerability count
- Dependency freshness
- Build success rate
- Review turnaround time

## Enforcement and Exceptions

### 1. Enforcement Levels

**Blocking (CI Failure)**:
- Compilation errors
- Test failures
- Clippy deny-level lints
- Security vulnerabilities
- Formatting violations

**Warning (Review Required)**:
- Coverage decrease
- Complexity increases
- New dependencies
- Documentation gaps

### 2. Exception Process

For rare cases where standards must be relaxed:

1. **Document Justification**: Clear explanation of why exception is needed
2. **Risk Assessment**: Evaluate potential impact and mitigation strategies
3. **Time-bound**: Set expiration date for temporary exceptions
4. **Review Required**: Senior team member must approve exceptions
5. **Track Technical Debt**: Add to technical debt backlog for future resolution

## Continuous Improvement

### 1. Regular Reviews

- **Monthly**: Review quality metrics and trends
- **Quarterly**: Assess and update quality standards
- **Annually**: Major review of tooling and processes

### 2. Tool Updates

- Keep linting tools and dependencies up to date
- Evaluate new quality tools and techniques
- Incorporate community best practices
- Monitor Rust ecosystem developments

This comprehensive code quality framework ensures that the Synaptic memory system maintains high standards of safety, performance, and maintainability throughout its development lifecycle.
