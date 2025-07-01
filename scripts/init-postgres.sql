-- PostgreSQL initialization script for CleoAI development
-- This script runs when PostgreSQL container starts for the first time

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For pgvector support

-- Create schema
CREATE SCHEMA IF NOT EXISTS cleoai;

-- Set default schema
SET search_path TO cleoai, public;

-- Create tables for conversation metadata
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    summary TEXT,
    token_count INTEGER DEFAULT 0
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);

-- Create tables for user preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    source TEXT DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category, key)
);

CREATE INDEX idx_preferences_user_id ON user_preferences(user_id);
CREATE INDEX idx_preferences_category ON user_preferences(category);
CREATE INDEX idx_preferences_updated_at ON user_preferences(updated_at DESC);

-- Create tables for episodic memories with vector embeddings
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- For OpenAI embeddings
    importance FLOAT DEFAULT 0.5,
    emotion TEXT,
    tags TEXT[],
    relations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX idx_memories_user_id ON episodic_memories(user_id);
CREATE INDEX idx_memories_created_at ON episodic_memories(created_at DESC);
CREATE INDEX idx_memories_importance ON episodic_memories(importance DESC);
CREATE INDEX idx_memories_embedding ON episodic_memories USING ivfflat (embedding vector_cosine_ops);

-- Create tables for procedural memories
CREATE TABLE IF NOT EXISTS procedural_memories (
    protocol_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    trigger_conditions JSONB NOT NULL,
    steps JSONB NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_executed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE INDEX idx_protocols_user_id ON procedural_memories(user_id);
CREATE INDEX idx_protocols_last_executed ON procedural_memories(last_executed DESC);

-- Create audit/analytics table
CREATE TABLE IF NOT EXISTS memory_access_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    memory_id TEXT,
    action TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    latency_ms FLOAT,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

CREATE INDEX idx_access_log_user_id ON memory_access_log(user_id);
CREATE INDEX idx_access_log_timestamp ON memory_access_log(timestamp DESC);
CREATE INDEX idx_access_log_memory_type ON memory_access_log(memory_type);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for auto-updating timestamps
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_protocols_updated_at BEFORE UPDATE ON procedural_memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create view for memory statistics
CREATE OR REPLACE VIEW memory_statistics AS
SELECT 
    user_id,
    COUNT(DISTINCT c.conversation_id) as conversation_count,
    COUNT(DISTINCT p.preference_id) as preference_count,
    COUNT(DISTINCT e.memory_id) as episodic_memory_count,
    COUNT(DISTINCT pr.protocol_id) as procedural_memory_count,
    MAX(GREATEST(
        c.created_at, 
        p.updated_at, 
        e.created_at, 
        pr.updated_at
    )) as last_activity
FROM 
    conversations c
    FULL OUTER JOIN user_preferences p USING (user_id)
    FULL OUTER JOIN episodic_memories e USING (user_id)
    FULL OUTER JOIN procedural_memories pr USING (user_id)
GROUP BY user_id;

-- Insert initialization record
INSERT INTO memory_access_log (user_id, memory_type, action, success)
VALUES ('system', 'initialization', 'database_setup', true);

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON SCHEMA cleoai TO cleoai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cleoai TO cleoai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cleoai TO cleoai;