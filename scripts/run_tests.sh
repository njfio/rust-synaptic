#!/bin/bash

# Test Runner Script for Synaptic AI Agent Memory
# Provides organized test execution with categories and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to get test command for a category
get_test_command() {
    case $1 in
        "core")
            echo "cargo test --lib"
            ;;
        "integration")
            echo "cargo test --test integration_tests"
            ;;
        "security")
            echo "cargo test --test phase4_security_tests && cargo test --test security_tests && cargo test --test zero_knowledge_tests && cargo test --test homomorphic_encryption_tests"
            ;;
        "performance")
            echo "cargo test --test real_performance_measurement_tests && cargo test --test performance_tests && cargo test --test comprehensive_optimization_tests && cargo test --test advanced_performance_optimization_tests"
            ;;
        "lifecycle")
            echo "cargo test --test real_lifecycle_management_tests"
            ;;
        "multimodal")
            echo "cargo test --test phase5_multimodal_tests && cargo test --test phase5b_document_tests && cargo test --test data_processor_tests"
            ;;
        "temporal")
            echo "cargo test --test temporal_evolution_tests && cargo test --test temporal_summary_tests && cargo test --test myers_diff_tests && cargo test --test diff_analyzer_tests"
            ;;
        "knowledge_graph")
            echo "cargo test --test knowledge_graph_tests"
            ;;
        "analytics")
            echo "cargo test --test phase3_analytics && cargo test --test advanced_theme_extraction_tests && cargo test --test summarization_tests"
            ;;
        "search")
            echo "cargo test --test enhanced_similarity_search_tests"
            ;;
        "external")
            echo "cargo test --test external_integrations_tests && cargo test --test phase1_embeddings_tests && cargo test --test phase2_distributed_tests"
            ;;
        "logging")
            echo "cargo test --test comprehensive_logging_tests && cargo test --test enhanced_error_handling_tests"
            ;;
        "visualization")
            echo "cargo test --test visualization_tests"
            ;;
        "sync")
            echo "cargo test --test offline_sync_tests"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Priority order
CRITICAL_TESTS=("core" "integration" "security")
HIGH_TESTS=("performance" "lifecycle" "multimodal")
MEDIUM_TESTS=("temporal" "knowledge_graph" "analytics" "search" "external")
LOW_TESTS=("logging" "visualization" "sync")

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run a test category
run_test_category() {
    local category=$1
    local command=$(get_test_command "$category")

    if [ -z "$command" ]; then
        print_status $RED "Unknown test category: $category"
        return 1
    fi

    print_status $BLUE "Running $category tests..."
    echo "Command: $command"

    if eval $command; then
        print_status $GREEN "✅ $category tests PASSED"
        return 0
    else
        print_status $RED "❌ $category tests FAILED"
        return 1
    fi
}

# Function to run tests by priority
run_by_priority() {
    local priority=$1
    local tests_array
    
    case $priority in
        "critical")
            tests_array=("${CRITICAL_TESTS[@]}")
            ;;
        "high")
            tests_array=("${HIGH_TESTS[@]}")
            ;;
        "medium")
            tests_array=("${MEDIUM_TESTS[@]}")
            ;;
        "low")
            tests_array=("${LOW_TESTS[@]}")
            ;;
        *)
            print_status $RED "Unknown priority: $priority"
            exit 1
            ;;
    esac
    
    print_status $YELLOW "Running $priority priority tests..."
    
    local failed_tests=()
    for test in "${tests_array[@]}"; do
        if ! run_test_category "$test"; then
            failed_tests+=("$test")
        fi
        echo ""
    done
    
    if [ ${#failed_tests[@]} -eq 0 ]; then
        print_status $GREEN "All $priority priority tests passed!"
        return 0
    else
        print_status $RED "Failed $priority priority tests: ${failed_tests[*]}"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    print_status $YELLOW "Running all 161 tests across all categories..."
    
    local failed_categories=()
    local total_categories=0
    
    for priority in "critical" "high" "medium" "low"; do
        print_status $BLUE "\n=== $priority PRIORITY TESTS ==="
        if ! run_by_priority "$priority"; then
            failed_categories+=("$priority")
        fi
        ((total_categories++))
        echo ""
    done
    
    # Summary
    print_status $BLUE "=== TEST SUMMARY ==="
    if [ ${#failed_categories[@]} -eq 0 ]; then
        print_status $GREEN "🎉 ALL TESTS PASSED! (161 tests across 14 categories)"
    else
        print_status $RED "❌ Some test categories failed: ${failed_categories[*]}"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all                    Run all tests (161 tests)"
    echo "  critical              Run critical priority tests (core, integration, security)"
    echo "  high                  Run high priority tests (performance, lifecycle, multimodal)"
    echo "  medium                Run medium priority tests (temporal, knowledge_graph, analytics, search, external)"
    echo "  low                   Run low priority tests (logging, visualization, sync)"
    echo "  <category>            Run specific test category"
    echo ""
    echo "Available categories:"
    echo "  - core"
    echo "  - integration"
    echo "  - security"
    echo "  - performance"
    echo "  - lifecycle"
    echo "  - multimodal"
    echo "  - temporal"
    echo "  - knowledge_graph"
    echo "  - analytics"
    echo "  - search"
    echo "  - external"
    echo "  - logging"
    echo "  - visualization"
    echo "  - sync"
    echo ""
    echo "Examples:"
    echo "  $0 all                # Run all tests"
    echo "  $0 critical           # Run only critical tests"
    echo "  $0 security           # Run only security tests"
    echo "  $0 performance        # Run only performance tests"
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    case $1 in
        "all")
            run_all_tests
            ;;
        "critical"|"high"|"medium"|"low")
            run_by_priority "$1"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            command=$(get_test_command "$1")
            if [ -n "$command" ]; then
                run_test_category "$1"
            else
                print_status $RED "Unknown option: $1"
                show_usage
                exit 1
            fi
            ;;
    esac
}

main "$@"
