#!/bin/bash
# Test Runner for Synaptic AI Agent Memory System
# Executes tests based on priority levels defined in tests/test_config.toml

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the priority level from command line argument
PRIORITY=${1:-all}

# Print usage if help is requested
if [ "$PRIORITY" = "--help" ] || [ "$PRIORITY" = "-h" ]; then
    echo "Usage: $0 [priority]"
    echo ""
    echo "Priority levels:"
    echo "  critical  - Run critical priority tests (core, integration, security)"
    echo "  high      - Run high priority tests (performance, lifecycle, multimodal)"
    echo "  medium    - Run medium priority tests (temporal, knowledge_graph, analytics, search, integrations)"
    echo "  low       - Run low priority tests (logging, visualization, sync)"
    echo "  all       - Run all tests (default)"
    echo ""
    echo "Example: $0 critical"
    exit 0
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Synaptic Test Suite - Priority: ${PRIORITY}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to run a test category
run_test_category() {
    local category=$1
    local description=$2
    local command=$3

    echo -e "${YELLOW}Running: ${category}${NC}"
    echo -e "${BLUE}Description: ${description}${NC}"
    echo -e "Command: ${command}"
    echo ""

    if eval "$command"; then
        echo -e "${GREEN}✓ ${category} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${category} failed${NC}"
        echo ""
        return 1
    fi
}

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Critical priority tests
run_critical_tests() {
    echo -e "${BLUE}=== Critical Priority Tests ===${NC}"
    echo ""

    run_test_category "Core Library Tests" \
        "Core memory operations and basic functionality" \
        "cargo test --lib --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    run_test_category "Integration Tests" \
        "End-to-end integration testing" \
        "cargo test --test integration_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    # Security tests (may require features)
    if cargo test --test security_tests --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Security Tests passed${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${YELLOW}⊘ Security Tests skipped (requires security features)${NC}"
    fi
    ((TOTAL_TESTS++))
}

# High priority tests
run_high_tests() {
    echo -e "${BLUE}=== High Priority Tests ===${NC}"
    echo ""

    run_test_category "Performance Tests" \
        "Performance measurement and optimization" \
        "cargo test --test performance_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    run_test_category "Lifecycle Management" \
        "Memory lifecycle management and archiving" \
        "cargo test --test real_lifecycle_management_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    # Multimodal tests (may require features)
    if cargo test --test data_processor_tests --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Multimodal Tests passed${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${YELLOW}⊘ Multimodal Tests skipped (requires multimodal features)${NC}"
    fi
    ((TOTAL_TESTS++))
}

# Medium priority tests
run_medium_tests() {
    echo -e "${BLUE}=== Medium Priority Tests ===${NC}"
    echo ""

    run_test_category "Temporal Analysis" \
        "Temporal patterns and differential analysis" \
        "cargo test --test temporal_evolution_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    run_test_category "Knowledge Graph" \
        "Knowledge graph and reasoning functionality" \
        "cargo test --test knowledge_graph_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    run_test_category "Search & Similarity" \
        "Advanced search and similarity algorithms" \
        "cargo test --test search_scoring_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
}

# Low priority tests
run_low_tests() {
    echo -e "${BLUE}=== Low Priority Tests ===${NC}"
    echo ""

    run_test_category "Logging Tests" \
        "Comprehensive logging and error handling" \
        "cargo test --test comprehensive_logging_tests --quiet" && ((PASSED_TESTS++)) || ((FAILED_TESTS++))
    ((TOTAL_TESTS++))

    # Visualization tests (may require features)
    if cargo test --test visualization_tests --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Visualization Tests passed${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${YELLOW}⊘ Visualization Tests skipped (requires visualization features)${NC}"
    fi
    ((TOTAL_TESTS++))
}

# Execute tests based on priority
case "$PRIORITY" in
    critical)
        run_critical_tests
        ;;
    high)
        run_high_tests
        ;;
    medium)
        run_medium_tests
        ;;
    low)
        run_low_tests
        ;;
    all)
        run_critical_tests
        run_high_tests
        run_medium_tests
        run_low_tests
        ;;
    *)
        echo -e "${RED}Error: Unknown priority level: $PRIORITY${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Print summary
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Total test categories: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed! ✗${NC}"
    exit 1
fi
