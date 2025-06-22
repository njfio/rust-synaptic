# Logging Standards for Synaptic

## Overview

This document defines the logging standards and best practices for the Synaptic memory system. Consistent logging is crucial for production monitoring, debugging, and observability.

## Logging Framework

We use the `tracing` crate for structured logging throughout the codebase. This provides:
- Structured, machine-readable logs
- Hierarchical spans for request tracing
- Configurable log levels
- Rich context information

## Log Levels

### ERROR
Use for errors that require immediate attention or indicate system failures:
```rust
tracing::error!(
    error = %err,
    operation = "memory_store",
    key = %memory_key,
    "Failed to store memory entry"
);
```

**When to use:**
- Database connection failures
- Critical system errors
- Security violations
- Data corruption

### WARN
Use for potentially problematic situations that don't stop execution:
```rust
tracing::warn!(
    threshold = %threshold,
    current_usage = %usage,
    "Memory usage approaching threshold"
);
```

**When to use:**
- Resource usage warnings
- Deprecated API usage
- Configuration issues
- Performance degradation

### INFO
Use for general operational information:
```rust
tracing::info!(
    component = "memory_manager",
    operation = "consolidation",
    memories_processed = %count,
    duration_ms = %duration.as_millis(),
    "Memory consolidation completed"
);
```

**When to use:**
- System startup/shutdown
- Major operation completion
- Configuration changes
- User actions

### DEBUG
Use for detailed diagnostic information:
```rust
tracing::debug!(
    query = %query_text,
    results_count = %results.len(),
    execution_time_ms = %execution_time.as_millis(),
    "Search query executed"
);
```

**When to use:**
- Algorithm execution details
- Internal state changes
- Performance metrics
- Detailed operation flow

### TRACE
Use for very detailed diagnostic information:
```rust
tracing::trace!(
    step = "similarity_calculation",
    vector_a_len = %vector_a.len(),
    vector_b_len = %vector_b.len(),
    similarity_score = %score,
    "Calculated vector similarity"
);
```

**When to use:**
- Fine-grained execution flow
- Loop iterations
- Mathematical calculations
- Low-level operations

## Structured Logging Patterns

### Standard Fields

Always include these fields when relevant:

```rust
tracing::info!(
    component = "memory_manager",     // Module/component name
    operation = "store",              // Operation being performed
    user_id = %user_id,              // User context (if applicable)
    session_id = %session_id,        // Session context (if applicable)
    duration_ms = %duration.as_millis(), // Operation duration
    "Operation completed successfully"
);
```

### Error Logging Pattern

```rust
tracing::error!(
    error = %err,                    // Error details
    error_type = "ValidationError",  // Error classification
    component = "input_validator",   // Where error occurred
    operation = "validate_memory",   // What was being done
    input_size = %input.len(),       // Relevant context
    "Input validation failed"
);
```

### Performance Logging Pattern

```rust
tracing::info!(
    component = "search_engine",
    operation = "semantic_search",
    query_complexity = %complexity,
    results_count = %results.len(),
    cache_hit = %cache_hit,
    duration_ms = %duration.as_millis(),
    memory_usage_mb = %memory_usage / 1024 / 1024,
    "Search operation completed"
);
```

## Spans for Request Tracing

Use spans to trace operations across multiple functions:

```rust
#[tracing::instrument(
    name = "memory_consolidation",
    fields(
        memory_count = %memories.len(),
        strategy = %strategy_name
    )
)]
async fn consolidate_memories(
    memories: &[MemoryEntry],
    strategy_name: &str
) -> Result<ConsolidationResult> {
    tracing::info!("Starting memory consolidation");
    
    // Operation implementation
    
    tracing::info!(
        consolidated_count = %result.consolidated_count,
        "Consolidation completed successfully"
    );
    
    Ok(result)
}
```

## Context Propagation

Use the `tracing::Span::current()` to add context to existing spans:

```rust
async fn process_memory(&self, memory: &MemoryEntry) -> Result<()> {
    let span = tracing::Span::current();
    span.record("memory_id", &memory.id);
    span.record("memory_type", &format!("{:?}", memory.memory_type));
    
    // Processing logic
}
```

## What NOT to Log

### Avoid println!/eprintln!
Never use `println!` or `eprintln!` in production code. Use tracing instead:

```rust
// ❌ Don't do this
println!("Processing memory: {}", memory_id);

// ✅ Do this instead
tracing::info!(
    memory_id = %memory_id,
    "Processing memory entry"
);
```

### Sensitive Information
Never log sensitive data:
- Passwords or API keys
- Personal information
- Encryption keys
- Session tokens

```rust
// ❌ Don't do this
tracing::info!("User password: {}", password);

// ✅ Do this instead
tracing::info!(
    user_id = %user_id,
    "User authentication successful"
);
```

### High-Frequency Logs
Avoid logging in tight loops or high-frequency operations without rate limiting:

```rust
// ❌ Don't do this
for item in large_collection {
    tracing::debug!("Processing item: {:?}", item);
}

// ✅ Do this instead
tracing::debug!(
    item_count = %large_collection.len(),
    "Processing collection"
);
for (index, item) in large_collection.iter().enumerate() {
    if index % 1000 == 0 {
        tracing::debug!(
            processed = %index,
            total = %large_collection.len(),
            "Processing progress"
        );
    }
}
```

## Configuration

### Log Levels by Environment

- **Development**: `TRACE` or `DEBUG`
- **Testing**: `INFO` or `WARN`
- **Production**: `INFO` or `WARN`
- **Critical Systems**: `WARN` or `ERROR`

### Environment Variables

```bash
# Set log level
RUST_LOG=synaptic=info

# Component-specific levels
RUST_LOG=synaptic::memory=debug,synaptic::search=info

# JSON output for production
SYNAPTIC_LOG_FORMAT=json
```

## Testing Logging

Use `tracing-test` for testing log output:

```rust
#[cfg(test)]
mod tests {
    use tracing_test::traced_test;
    
    #[traced_test]
    #[tokio::test]
    async fn test_memory_operation() {
        // Test implementation
        
        // Verify logs were emitted
        assert!(logs_contain("Memory operation completed"));
    }
}
```

## Monitoring Integration

Logs should integrate with monitoring systems:

```rust
// Add correlation IDs for distributed tracing
tracing::info!(
    trace_id = %trace_id,
    span_id = %span_id,
    component = "memory_manager",
    "Operation started"
);
```

## Performance Considerations

- Use lazy evaluation for expensive log formatting
- Consider log sampling for high-volume operations
- Use appropriate log levels to control verbosity
- Avoid string allocation in hot paths

```rust
// ❌ Expensive formatting always executed
tracing::debug!("Complex data: {}", expensive_format(&data));

// ✅ Lazy evaluation
tracing::debug!(data = ?data, "Complex data processed");
```

This logging standard ensures consistent, structured, and useful logging throughout the Synaptic codebase.
