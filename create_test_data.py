import sqlite3
import json
import uuid
from datetime import datetime, timedelta
import random
import os

# Connect to database
db_path = 'data/memory/cleoai_memory.db'
os.makedirs('data/memory', exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Read and execute schema
schema_file = 'data/memory/init_schema.sql'
if os.path.exists(schema_file):
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
        cursor.executescript(schema_sql)
    print("Database initialized with schema")
else:
    print("Warning: Schema file not found. Creating basic tables...")
    # Create basic tables if schema file doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            summary TEXT,
            token_count INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

# Create test data
test_user_id = "test_user_001"

# Create a test conversation
conv_id = str(uuid.uuid4())
cursor.execute("""
    INSERT INTO conversations (conversation_id, user_id, metadata, summary, token_count)
    VALUES (?, ?, ?, ?, ?)
""", (conv_id, test_user_id, json.dumps({"test": True, "source": "setup_script"}), 
      "Test conversation about AI and memory systems", 256))

# Add messages
messages = [
    ("user", "Hello! Can you tell me about your memory system?"),
    ("assistant", "I have a sophisticated memory system with multiple components:\n\n1. Short-term memory for conversations\n2. Long-term memory for user preferences\n3. Episodic memory for important events\n4. Procedural memory for learned tasks"),
    ("user", "That's interesting! How do you store user preferences?"),
    ("assistant", "User preferences are stored with confidence scores and sources. For example, if you tell me you prefer concise responses, I'll remember that with high confidence."),
]

for role, content in messages:
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
    """, (conv_id, role, content))

# Try to add preferences if table exists
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            preference_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            preference_key TEXT NOT NULL,
            preference_value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            source TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, category, preference_key)
        )
    """)
    
    preferences = [
        ("communication", "style", "detailed", 0.8, "explicit"),
        ("communication", "language", "English", 1.0, "explicit"),
        ("interests", "topics", json.dumps(["AI", "technology", "science"]), 0.7, "inferred"),
        ("behavior", "response_length", "medium", 0.6, "inferred"),
    ]
    
    for category, key, value, confidence, source in preferences:
        pref_id = f"{test_user_id}:{category}:{key}"
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences 
            (preference_id, user_id, category, preference_key, preference_value, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pref_id, test_user_id, category, key, value, confidence, source))
    print(f"Added {len(preferences)} user preferences")
except Exception as e:
    print(f"Could not create preferences: {e}")

# Try to add episodic memories if table exists
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            emotion TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    memories = [
        ("Learning about AI", "User expressed strong interest in understanding AI architectures", 0.9, "curious"),
        ("First conversation", "Initial interaction where user tested the system", 0.7, "neutral"),
        ("Preference discovery", "User mentioned they work in technology", 0.8, "engaged"),
    ]
    
    for title, content, importance, emotion in memories:
        memory_id = str(uuid.uuid4())
        tags = json.dumps(["test", "setup", "initial"])
        cursor.execute("""
            INSERT INTO episodic_memories 
            (memory_id, user_id, title, content, importance, emotion, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, test_user_id, title, content, importance, emotion, tags))
    print(f"Added {len(memories)} episodic memories")
except Exception as e:
    print(f"Could not create episodic memories: {e}")

# Commit changes
conn.commit()
conn.close()

print(f"\nTest data created successfully!")
print(f"  - 1 conversation with {len(messages)} messages")
print(f"\nTest user ID: {test_user_id}")
print("\nDatabase location: " + os.path.abspath(db_path))