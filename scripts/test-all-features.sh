#!/bin/bash

# Test All Features Compilation Script
# This script tests compilation with various feature combinations to catch feature-gated issues

set -e

echo "🚀 Testing Synaptic Feature Compilation Matrix"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test feature compilation
test_features() {
    local features="$1"
    local description="$2"
    local extra_args="$3"
    
    echo -e "\n${YELLOW}Testing: $description${NC}"
    echo "Features: $features"
    echo "Command: cargo check --features \"$features\" $extra_args"
    
    if cargo check --features "$features" $extra_args; then
        echo -e "${GREEN}✅ SUCCESS: $description${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED: $description${NC}"
        return 1
    fi
}

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=()

# Function to run test and track results
run_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if test_features "$1" "$2" "$3"; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS+=("$2")
    fi
}

echo -e "\n${YELLOW}Phase 1: Core Feature Tests${NC}"
echo "============================"

run_test "minimal" "Minimal features only" "--no-default-features"
run_test "core" "Core features only" "--no-default-features"
run_test "storage" "Storage features" "--no-default-features"
run_test "core,storage" "Core + Storage" "--no-default-features"

echo -e "\n${YELLOW}Phase 2: Essential Feature Tests${NC}"
echo "================================="

run_test "embeddings" "Embeddings only" "--no-default-features"
run_test "analytics" "Analytics only" "--no-default-features"
run_test "embeddings,analytics" "Embeddings + Analytics" "--no-default-features"
run_test "" "Default features" ""

echo -e "\n${YELLOW}Phase 3: Advanced Feature Tests${NC}"
echo "================================"

run_test "security" "Security features" "--no-default-features"
run_test "distributed" "Distributed features" "--no-default-features"
run_test "external-integrations" "External integrations" "--no-default-features"

echo -e "\n${YELLOW}Phase 4: Multi-modal Feature Tests${NC}"
echo "==================================="

run_test "document-processing" "Document processing" "--no-default-features"
run_test "code-analysis" "Code analysis" "--no-default-features"
run_test "image-processing" "Image processing" "--no-default-features"
run_test "audio-processing" "Audio processing" "--no-default-features"

echo -e "\n${YELLOW}Phase 5: Integration Feature Tests${NC}"
echo "==================================="

run_test "reqwest" "HTTP client (reqwest)" "--no-default-features"
run_test "openai-embeddings" "OpenAI embeddings" "--no-default-features"
run_test "llm-integration" "LLM integration" "--no-default-features"
run_test "visualization" "Visualization" "--no-default-features"

echo -e "\n${YELLOW}Phase 6: Combination Tests${NC}"
echo "==========================="

run_test "embeddings,analytics,security" "Core + Security" "--no-default-features"
run_test "storage,embeddings,analytics" "Storage + Analytics" "--no-default-features"
run_test "external-integrations,embeddings" "External + Embeddings" "--no-default-features"

echo -e "\n${YELLOW}Phase 7: Library and Examples Tests${NC}"
echo "====================================="

TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo -e "\n${YELLOW}Testing: Library compilation${NC}"
if cargo check --lib; then
    echo -e "${GREEN}✅ SUCCESS: Library compilation${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ FAILED: Library compilation${NC}"
    FAILED_TESTS+=("Library compilation")
fi

# Test examples
EXAMPLES=(
    "basic_usage:"
    "phase3_analytics:analytics"
    "real_integrations:external-integrations"
    "openai_embeddings_test:openai-embeddings"
)

for example_spec in "${EXAMPLES[@]}"; do
    IFS=':' read -r example features <<< "$example_spec"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "\n${YELLOW}Testing: Example $example${NC}"
    
    if [ -n "$features" ]; then
        cmd="cargo check --example $example --features \"$features\""
    else
        cmd="cargo check --example $example"
    fi
    
    echo "Command: $cmd"
    if eval $cmd; then
        echo -e "${GREEN}✅ SUCCESS: Example $example${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: Example $example${NC}"
        FAILED_TESTS+=("Example $example")
    fi
done

# Summary
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}           TEST SUMMARY${NC}"
echo -e "${YELLOW}========================================${NC}"

echo -e "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $((TOTAL_TESTS - PASSED_TESTS))${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}❌ $test${NC}"
    done
    echo -e "\n${RED}Some feature combinations failed compilation!${NC}"
    exit 1
else
    echo -e "\n${GREEN}🎉 All feature combinations compiled successfully!${NC}"
    exit 0
fi
