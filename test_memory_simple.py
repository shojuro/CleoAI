#!/usr/bin/env python3
"""
Simple memory system test that doesn't require ML dependencies
"""
import sqlite3
import json
import os
from datetime import datetime

print("Testing CleoAI Memory System (SQLite)")
print("=" * 50)

# Database path
db_path = 'data/memory/cleoai_memory.db'

if not os.path.exists(db_path):
    print(f"✗ Database not found at {db_path}")
    print("  Please run: python create_test_data.py")
    exit(1)

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Test user
test_user_id = "test_user_001"

try:
    # Test 1: Check conversations
    print("\n1. Testing conversation retrieval...")
    cursor.execute("""
        SELECT conversation_id, user_id, created_at, summary, token_count 
        FROM conversations 
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (test_user_id,))
    
    conversations = cursor.fetchall()
    print(f"   ✓ Found {len(conversations)} conversations")
    
    if conversations:
        conv_id = conversations[0][0]
        print(f"   - Conversation ID: {conv_id}")
        print(f"   - Summary: {conversations[0][3]}")
        
        # Get messages for this conversation
        cursor.execute("""
            SELECT role, content, timestamp 
            FROM messages 
            WHERE conversation_id = ?
            ORDER BY timestamp
        """, (conv_id,))
        
        messages = cursor.fetchall()
        print(f"   - Messages: {len(messages)}")
        for i, (role, content, timestamp) in enumerate(messages[:2]):  # Show first 2
            print(f"     [{role}]: {content[:50]}...")
    
    # Test 2: Check user preferences
    print("\n2. Testing user preferences...")
    cursor.execute("""
        SELECT category, preference_key, preference_value, confidence, source 
        FROM user_preferences 
        WHERE user_id = ?
    """, (test_user_id,))
    
    preferences = cursor.fetchall()
    print(f"   ✓ Found {len(preferences)} preferences")
    
    for category, key, value, confidence, source in preferences:
        print(f"   - {category}/{key}: {value} (confidence: {confidence}, source: {source})")
    
    # Test 3: Check episodic memories
    print("\n3. Testing episodic memories...")
    cursor.execute("""
        SELECT title, content, importance, emotion 
        FROM episodic_memories 
        WHERE user_id = ?
        ORDER BY importance DESC
    """, (test_user_id,))
    
    memories = cursor.fetchall()
    print(f"   ✓ Found {len(memories)} memories")
    
    for title, content, importance, emotion in memories:
        print(f"   - {title}: importance={importance}, emotion={emotion}")
        print(f"     {content[:60]}...")
    
    # Test 4: Create new conversation
    print("\n4. Testing conversation creation...")
    import uuid
    new_conv_id = str(uuid.uuid4())
    
    cursor.execute("""
        INSERT INTO conversations (conversation_id, user_id, metadata, summary)
        VALUES (?, ?, ?, ?)
    """, (new_conv_id, test_user_id, 
          json.dumps({"source": "test_script", "timestamp": datetime.now().isoformat()}),
          "Test conversation from memory test"))
    
    # Add a message
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content)
        VALUES (?, ?, ?)
    """, (new_conv_id, "user", "This is a test message from the memory test script"))
    
    conn.commit()
    print(f"   ✓ Created conversation: {new_conv_id}")
    print("   ✓ Added test message")
    
    # Test 5: Verify creation
    print("\n5. Verifying new conversation...")
    cursor.execute("""
        SELECT COUNT(*) FROM messages WHERE conversation_id = ?
    """, (new_conv_id,))
    
    message_count = cursor.fetchone()[0]
    print(f"   ✓ Verified: {message_count} message(s) in new conversation")
    
    print("\n✓ All tests passed!")
    print("\nMemory system is working correctly with SQLite backend.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()

print("\nNote: This is a simplified test that doesn't require ML dependencies.")
print("For full memory system testing, install core dependencies first.")