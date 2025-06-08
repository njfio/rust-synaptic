-- Initialize Synaptic database schema

-- Create schema
CREATE SCHEMA IF NOT EXISTS synaptic;

-- Memory entries table
CREATE TABLE IF NOT EXISTS synaptic.memory_entries (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    importance DOUBLE PRECISION DEFAULT 0.5,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding TEXT, -- JSON array of floats for now
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0
);

-- Analytics events table
CREATE TABLE IF NOT EXISTS synaptic.analytics_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_context VARCHAR(255)
);

-- Knowledge graph relationships table
CREATE TABLE IF NOT EXISTS synaptic.relationships (
    id SERIAL PRIMARY KEY,
    source_key VARCHAR(255) NOT NULL,
    target_key VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    strength REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_key, target_key, relationship_type)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_memory_entries_key ON synaptic.memory_entries(key);
CREATE INDEX IF NOT EXISTS idx_memory_entries_created_at ON synaptic.memory_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_entries_importance ON synaptic.memory_entries(importance);
CREATE INDEX IF NOT EXISTS idx_memory_entries_tags ON synaptic.memory_entries USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memory_entries_metadata ON synaptic.memory_entries USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON synaptic.analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_events_timestamp ON synaptic.analytics_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON synaptic.relationships(source_key);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON synaptic.relationships(target_key);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON synaptic.relationships(relationship_type);

-- Update trigger for memory entries
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_memory_entries_updated_at
    BEFORE UPDATE ON synaptic.memory_entries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
