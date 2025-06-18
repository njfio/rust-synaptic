#!/bin/bash

# Synaptic Test Runner Script
# Provides organized test execution with categorization and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to get test commands for category
get_test_command() {
    case "$1" in
        "core")
            echo "--lib"
            ;;
        "integration")
            echo "--test integration_tests"
            ;;
        "security")
            echo "--test phase4_security_tests --test security_tests --test zero_knowledge_tests --test homomorphic_encryption_tests"
            ;;
        "performance")
            echo "--test real_performance_measurement_tests --test comprehensive_optimization_tests"
            ;;
        "multimodal")
            echo "--test phase5_multimodal_tests --test phase5b_document_tests --test data_processor_tests"
            ;;
        "temporal")
            echo "--test temporal_evolution_tests --test temporal_summary_tests --test myers_diff_tests --test diff_analyzer_tests"
            ;;
        "analytics")
            echo "--test phase3_analytics --test advanced_theme_extraction_tests --test summarization_tests --test knowledge_graph_tests --test enhanced_similarity_search_tests --test real_lifecycle_management_tests"
            ;;
        "infrastructure")
            echo "--test external_integrations_tests --test phase1_embeddings_tests --test phase2_distributed_tests --test comprehensive_logging_tests --test enhanced_error_handling_tests --test visualization_tests --test offline_sync_tests"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run tests for a category
run_category_tests() {
    local category=$1
    local tests=$(get_test_command "$category")

    if [ -z "$tests" ]; then
        print_status $RED "Unknown test category: $category"
        return 1
    fi

    print_status $BLUE "Running $category tests..."
    echo "Command: cargo test $tests"

    if cargo test $tests; then
        print_status $GREEN "✓ $category tests passed"
        return 0
    else
        print_status $RED "✗ $category tests failed"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Synaptic Test Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  all                    Run all tests (default)"
    echo "  category <name>        Run tests for specific category"
    echo "  list                   List all test categories"
    echo "  quick                  Run core and integration tests only"
    echo "  security               Run all security-related tests"
    echo "  performance            Run performance and optimization tests"
    echo "  coverage               Generate code coverage report"
    echo "  benchmark              Run performance benchmarks"
    echo "  clean                  Clean test artifacts"
    echo ""
    echo "Categories:"
    echo "  - core"
    echo "  - integration"
    echo "  - security"
    echo "  - performance"
    echo "  - multimodal"
    echo "  - temporal"
    echo "  - analytics"
    echo "  - infrastructure"
    echo ""
    echo "Options:"
    echo "  --verbose              Show detailed output"
    echo "  --nocapture            Show test output"
    echo "  --help                 Show this help message"
}

# Function to run all tests
run_all_tests() {
    local verbose=$1
    local nocapture=$2
    local failed_categories=()
    local total_categories=8
    local passed_categories=0
    local categories="core integration security performance multimodal temporal analytics infrastructure"

    print_status $BLUE "Running all test categories ($total_categories total)..."
    echo ""

    for category in $categories; do
        if run_category_tests "$category"; then
            passed_categories=$((passed_categories + 1))
        else
            failed_categories="$failed_categories $category"
        fi
        echo ""
    done
    
    # Summary
    print_status $BLUE "Test Summary:"
    print_status $GREEN "Passed: $passed_categories/$total_categories categories"

    if [ -n "$failed_categories" ]; then
        print_status $RED "Failed categories:$failed_categories"
        return 1
    else
        print_status $GREEN "All test categories passed!"
        return 0
    fi
}

# Function to generate coverage report
generate_coverage() {
    print_status $BLUE "Generating code coverage report..."
    
    if command -v cargo-llvm-cov >/dev/null 2>&1; then
        cargo llvm-cov --all-features --workspace --html
        print_status $GREEN "Coverage report generated in target/llvm-cov/html/"
    else
        print_status $YELLOW "cargo-llvm-cov not installed. Installing..."
        cargo install cargo-llvm-cov
        cargo llvm-cov --all-features --workspace --html
        print_status $GREEN "Coverage report generated in target/llvm-cov/html/"
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_status $BLUE "Running performance benchmarks..."
    cargo test --test performance_tests -- --ignored --nocapture
}

# Function to clean test artifacts
clean_tests() {
    print_status $BLUE "Cleaning test artifacts..."
    cargo clean
    rm -rf target/llvm-cov/
    print_status $GREEN "Test artifacts cleaned"
}

# Main script logic
case "${1:-all}" in
    "all")
        run_all_tests "$2" "$3"
        ;;
    "category")
        if [ -z "$2" ]; then
            print_status $RED "Category name required"
            show_usage
            exit 1
        fi
        run_category_tests "$2"
        ;;
    "list")
        echo "Available test categories:"
        echo "  - core"
        echo "  - integration"
        echo "  - security"
        echo "  - performance"
        echo "  - multimodal"
        echo "  - temporal"
        echo "  - analytics"
        echo "  - infrastructure"
        ;;
    "quick")
        print_status $BLUE "Running quick test suite (core + integration)..."
        run_category_tests "core" && run_category_tests "integration"
        ;;
    "security")
        run_category_tests "security"
        ;;
    "performance")
        run_category_tests "performance"
        ;;
    "coverage")
        generate_coverage
        ;;
    "benchmark")
        run_benchmarks
        ;;
    "clean")
        clean_tests
        ;;
    "--help"|"-h"|"help")
        show_usage
        ;;
    *)
        print_status $RED "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
