# Scripts Directory

This directory contains utility scripts for development, testing, and CI/CD.

## Available Scripts

### `run_tests.sh`

Priority-based test runner for the Synaptic test suite.

**Usage:**
```bash
./scripts/run_tests.sh [priority]
```

**Priority Levels:**
- `critical` - Run critical priority tests (core, integration, security)
- `high` - Run high priority tests (performance, lifecycle, multimodal)
- `medium` - Run medium priority tests (temporal, knowledge_graph, analytics, search, integrations)
- `low` - Run low priority tests (logging, visualization, sync)
- `all` - Run all tests (default)

**Examples:**
```bash
# Run critical tests only
./scripts/run_tests.sh critical

# Run all tests
./scripts/run_tests.sh all

# Show help
./scripts/run_tests.sh --help
```

**Features:**
- Color-coded output for better visibility
- Detailed test category descriptions
- Graceful handling of feature-gated tests
- Summary report at the end

### `pre-commit.sh`

Git pre-commit hook for code quality checks.

**Usage:**
```bash
# Install as git hook
make install-hooks

# Or manually install
cp scripts/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Checks Performed:**
1. Code formatting with `cargo fmt`
2. Lint checks with `cargo clippy`
3. Critical unit tests with `cargo test --lib`

**Note:** The hook will prevent commits if any checks fail.

## Adding New Scripts

When adding new scripts to this directory:

1. Make the script executable:
   ```bash
   chmod +x scripts/your_script.sh
   ```

2. Add a shebang at the top:
   ```bash
   #!/bin/bash
   ```

3. Include error handling:
   ```bash
   set -e  # Exit on error
   ```

4. Add the script to this README

5. Update the Makefile if appropriate

## Script Guidelines

- **Shell**: Use `bash` for compatibility
- **Error Handling**: Always include `set -e` or explicit error handling
- **Output**: Use colored output for better UX (see `run_tests.sh` for examples)
- **Help**: Include a `--help` option
- **Comments**: Document complex logic
- **Exit Codes**: Return 0 for success, non-zero for failure

## Color Codes

Scripts use these ANSI color codes:

```bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
```

Example usage:
```bash
echo -e "${GREEN}Success!${NC}"
echo -e "${RED}Error!${NC}"
```
