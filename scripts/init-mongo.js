// MongoDB initialization script for CleoAI development
// This script runs when MongoDB container starts for the first time

// Switch to cleoai_dev database
db = db.getSiblingDB('cleoai_dev');

// Create collections with indexes for optimal performance
db.createCollection('conversations');
db.conversations.createIndex({ 'user_id': 1, 'created_at': -1 });
db.conversations.createIndex({ 'conversation_id': 1 }, { unique: true });

db.createCollection('user_preferences');
db.user_preferences.createIndex({ 'user_id': 1, 'category': 1, 'key': 1 });
db.user_preferences.createIndex({ 'updated_at': -1 });

db.createCollection('episodic_memories');
db.episodic_memories.createIndex({ 'user_id': 1, 'created_at': -1 });
db.episodic_memories.createIndex({ 'importance': -1 });
db.episodic_memories.createIndex({ 'memory_id': 1 }, { unique: true });

db.createCollection('archived_conversations');
db.archived_conversations.createIndex({ 'user_id': 1, 'archived_at': -1 });
db.archived_conversations.createIndex({ 'conversation_id': 1 });

// Create a test document to verify setup
db.system_info.insertOne({
    type: 'initialization',
    timestamp: new Date(),
    message: 'CleoAI MongoDB initialized successfully',
    version: '1.0.0'
});

print('CleoAI MongoDB initialization completed successfully');