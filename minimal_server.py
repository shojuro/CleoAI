#!/usr/bin/env python3
"""
Ultra-minimal HTTP server for CleoAI using only standard library
No external dependencies required
"""
import http.server
import json
import sqlite3
import urllib.parse
from datetime import datetime
import uuid
import os

# Configuration
PORT = 8000
DB_PATH = 'data/memory/cleoai_memory.db'

class CleoAIHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "name": "CleoAI API (Ultra-Minimal)",
                "version": "1.0.0", 
                "status": "running",
                "mode": "stdlib-only",
                "endpoints": {
                    "/": "API info",
                    "/health": "Health check",
                    "/conversations?user_id=XXX": "Get user conversations",
                    "/memories?user_id=XXX": "Get user memories"
                }
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check database
            db_exists = os.path.exists(DB_PATH)
            db_size = os.path.getsize(DB_PATH) if db_exists else 0
            
            response = {
                "status": "healthy",
                "database": {
                    "exists": db_exists,
                    "path": DB_PATH,
                    "size_mb": round(db_size / 1024 / 1024, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        elif self.path.startswith('/conversations'):
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            user_id = params.get('user_id', ['test_user_001'])[0]
            
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT conversation_id, created_at, summary, token_count 
                    FROM conversations 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
                
                conversations = []
                for conv_id, created_at, summary, token_count in cursor.fetchall():
                    conversations.append({
                        "id": conv_id,
                        "created_at": created_at,
                        "summary": summary,
                        "token_count": token_count
                    })
                
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "user_id": user_id,
                    "conversations": conversations,
                    "count": len(conversations)
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
                
        elif self.path.startswith('/memories'):
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            user_id = params.get('user_id', ['test_user_001'])[0]
            
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Get episodic memories
                cursor.execute("""
                    SELECT title, content, importance, emotion, created_at
                    FROM episodic_memories 
                    WHERE user_id = ?
                    ORDER BY importance DESC
                """, (user_id,))
                
                memories = []
                for title, content, importance, emotion, created_at in cursor.fetchall():
                    memories.append({
                        "title": title,
                        "content": content,
                        "importance": importance,
                        "emotion": emotion,
                        "created_at": created_at
                    })
                
                # Get preferences
                cursor.execute("""
                    SELECT category, preference_key, preference_value, confidence
                    FROM user_preferences 
                    WHERE user_id = ?
                """, (user_id,))
                
                preferences = {}
                for category, key, value, confidence in cursor.fetchall():
                    if category not in preferences:
                        preferences[category] = {}
                    preferences[category][key] = {
                        "value": value,
                        "confidence": confidence
                    }
                
                conn.close()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "user_id": user_id,
                    "episodic_memories": memories,
                    "preferences": preferences
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

def main():
    print(f"🚀 Starting CleoAI Ultra-Minimal Server on port {PORT}")
    print(f"📁 Database: {DB_PATH}")
    print(f"🌐 Access at: http://localhost:{PORT}")
    print("\nAvailable endpoints:")
    print("  GET /              - API info")
    print("  GET /health        - Health check") 
    print("  GET /conversations - Get conversations (add ?user_id=XXX)")
    print("  GET /memories      - Get memories (add ?user_id=XXX)")
    print("\nPress Ctrl+C to stop\n")
    
    server = http.server.HTTPServer(('', PORT), CleoAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()

if __name__ == '__main__':
    main()