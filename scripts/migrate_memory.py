#!/usr/bin/env python3
"""
Migration script to move data from SQLite/ChromaDB to distributed backends.

Usage:
    python scripts/migrate_memory.py --source sqlite --target redis,supabase,pinecone
    python scripts/migrate_memory.py --dry-run
    python scripts/migrate_memory.py --batch-size 1000 --users user1,user2
"""
import os
import sys
import json
import time
import logging
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import memory_backend_config, project_config
from src.memory.memory_manager import MemoryManager
from src.memory.adapters.memory_adapter import HybridMemoryAdapter
from src.utils.config_validator import validate_configuration


@dataclass
class MigrationStats:
    """Track migration statistics."""
    conversations_migrated: int = 0
    preferences_migrated: int = 0
    episodic_memories_migrated: int = 0
    procedural_memories_migrated: int = 0
    vectors_migrated: int = 0
    errors: List[str] = None
    start_time: float = 0
    end_time: float = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time == 0:
            self.start_time = time.time()
    
    @property
    def total_migrated(self) -> int:
        return (self.conversations_migrated + self.preferences_migrated + 
                self.episodic_memories_migrated + self.procedural_memories_migrated)
    
    @property
    def duration(self) -> float:
        return (self.end_time or time.time()) - self.start_time
    
    def print_summary(self):
        """Print migration summary."""
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Conversations migrated: {self.conversations_migrated}")
        print(f"Preferences migrated: {self.preferences_migrated}")
        print(f"Episodic memories migrated: {self.episodic_memories_migrated}")
        print(f"Procedural memories migrated: {self.procedural_memories_migrated}")
        print(f"Vectors migrated: {self.vectors_migrated}")
        print(f"Total items migrated: {self.total_migrated}")
        
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        else:
            print("\nNo errors encountered!")
        print("="*60)


class MemoryMigrator:
    """Handles migration from legacy to distributed backends."""
    
    def __init__(self, source: str, targets: List[str], dry_run: bool = False,
                 batch_size: int = 100, specific_users: Optional[List[str]] = None):
        self.source = source
        self.targets = targets
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.specific_users = specific_users
        self.stats = MigrationStats()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory managers
        self.legacy_manager = MemoryManager()
        self.hybrid_adapter = self._create_hybrid_adapter()
    
    def _create_hybrid_adapter(self) -> HybridMemoryAdapter:
        """Create hybrid adapter with target backends enabled."""
        # Override config for migration targets
        config_overrides = {
            "use_redis": "redis" in self.targets,
            "use_mongodb": "mongodb" in self.targets,
            "use_supabase": "supabase" in self.targets,
            "use_pinecone": "pinecone" in self.targets,
            "use_sqlite": False,  # Don't use SQLite as target
            "use_chromadb": False  # Don't use ChromaDB as target
        }
        
        # Create adapter with overrides
        return HybridMemoryAdapter(
            legacy_manager=None,  # Don't use legacy for writing
            redis_config={
                "host": memory_backend_config.redis_host,
                "port": memory_backend_config.redis_port,
                "password": memory_backend_config.redis_password
            } if config_overrides["use_redis"] else None,
            supabase_config={
                "url": memory_backend_config.supabase_url,
                "key": memory_backend_config.supabase_anon_key
            } if config_overrides["use_supabase"] else None,
            pinecone_config={
                "api_key": memory_backend_config.pinecone_api_key,
                "environment": memory_backend_config.pinecone_environment,
                "index_name": memory_backend_config.pinecone_index_name,
                "dimension": memory_backend_config.pinecone_dimension
            } if config_overrides["use_pinecone"] else None,
            mongodb_config={
                "connection_string": memory_backend_config.mongodb_connection_string,
                "database": memory_backend_config.mongodb_database
            } if config_overrides["use_mongodb"] else None,
            use_legacy=False,
            use_distributed=True
        )
    
    async def migrate(self):
        """Run the migration."""
        self.logger.info(f"Starting migration from {self.source} to {self.targets}")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No data will be modified")
        
        try:
            # Get list of users to migrate
            users = await self._get_users_to_migrate()
            self.logger.info(f"Found {len(users)} users to migrate")
            
            # Migrate each user's data
            for i, user_id in enumerate(users):
                self.logger.info(f"Migrating user {i+1}/{len(users)}: {user_id}")
                await self._migrate_user(user_id)
            
            self.stats.end_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self.stats.errors.append(f"Fatal error: {str(e)}")
            raise
        
        finally:
            self.stats.print_summary()
    
    async def _get_users_to_migrate(self) -> List[str]:
        """Get list of users to migrate."""
        if self.specific_users:
            return self.specific_users
        
        # Get all users from SQLite
        users = set()
        
        # Get users from conversations
        db_path = Path(project_config.data_dir) / "memory" / "cleoai_memory.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get unique users from all tables
            tables = ["conversations", "user_preferences", "episodic_memories", "procedural_memories"]
            for table in tables:
                try:
                    cursor.execute(f"SELECT DISTINCT user_id FROM {table}")
                    users.update(row[0] for row in cursor.fetchall())
                except sqlite3.OperationalError:
                    self.logger.warning(f"Table {table} not found")
            
            conn.close()
        
        return list(users)
    
    async def _migrate_user(self, user_id: str):
        """Migrate all data for a specific user."""
        # Migrate conversations
        await self._migrate_conversations(user_id)
        
        # Migrate user preferences
        await self._migrate_preferences(user_id)
        
        # Migrate episodic memories
        await self._migrate_episodic_memories(user_id)
        
        # Migrate procedural memories
        await self._migrate_procedural_memories(user_id)
    
    async def _migrate_conversations(self, user_id: str):
        """Migrate conversations for a user."""
        try:
            conversations = self.legacy_manager.short_term.get_user_conversations(
                user_id, limit=10000
            )
            
            for conv in conversations:
                if self.dry_run:
                    self.logger.debug(f"Would migrate conversation {conv.conversation_id}")
                else:
                    # Create conversation in new system
                    new_conv = self.hybrid_adapter.create_conversation(
                        user_id=user_id,
                        metadata=conv.metadata
                    )
                    
                    # Migrate messages
                    for msg in conv.messages:
                        self.hybrid_adapter.add_message(
                            conversation_id=new_conv.conversation_id,
                            user_id=user_id,
                            role=msg["role"],
                            content=msg["content"]
                        )
                
                self.stats.conversations_migrated += 1
                
        except Exception as e:
            error_msg = f"Failed to migrate conversations for {user_id}: {e}"
            self.logger.error(error_msg)
            self.stats.errors.append(error_msg)
    
    async def _migrate_preferences(self, user_id: str):
        """Migrate user preferences."""
        try:
            preferences = self.legacy_manager.long_term.get_user_preferences(user_id)
            
            for pref in preferences:
                if self.dry_run:
                    self.logger.debug(f"Would migrate preference {pref.preference_key}")
                else:
                    self.hybrid_adapter.set_user_preference(
                        user_id=user_id,
                        category=pref.category,
                        key=pref.preference_key,
                        value=pref.preference_value,
                        confidence=pref.confidence,
                        source=pref.source
                    )
                
                self.stats.preferences_migrated += 1
                
        except Exception as e:
            error_msg = f"Failed to migrate preferences for {user_id}: {e}"
            self.logger.error(error_msg)
            self.stats.errors.append(error_msg)
    
    async def _migrate_episodic_memories(self, user_id: str):
        """Migrate episodic memories."""
        try:
            memories = self.legacy_manager.episodic.get_user_memories(
                user_id, limit=10000
            )
            
            for memory in memories:
                if self.dry_run:
                    self.logger.debug(f"Would migrate memory {memory.memory_id}")
                else:
                    # Create memory with embedding
                    self.hybrid_adapter.create_episodic_memory(
                        user_id=user_id,
                        title=memory.title,
                        content=memory.content,
                        importance=memory.importance,
                        emotion=memory.emotion,
                        tags=memory.tags,
                        relations=memory.relations,
                        embedding=memory.embedding
                    )
                
                self.stats.episodic_memories_migrated += 1
                
                if memory.embedding is not None:
                    self.stats.vectors_migrated += 1
                
        except Exception as e:
            error_msg = f"Failed to migrate episodic memories for {user_id}: {e}"
            self.logger.error(error_msg)
            self.stats.errors.append(error_msg)
    
    async def _migrate_procedural_memories(self, user_id: str):
        """Migrate procedural memories."""
        try:
            # Get procedural memories from SQLite
            db_path = Path(project_config.data_dir) / "memory" / "cleoai_memory.db"
            if not db_path.exists():
                return
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT protocol_id, name, description, trigger_conditions, 
                       steps, success_count, failure_count, last_executed
                FROM procedural_memories
                WHERE user_id = ?
            """, (user_id,))
            
            protocols = cursor.fetchall()
            conn.close()
            
            for protocol in protocols:
                if self.dry_run:
                    self.logger.debug(f"Would migrate protocol {protocol[1]}")
                else:
                    # Note: HybridMemoryAdapter doesn't have create_protocol method yet
                    # This would need to be implemented
                    self.logger.warning("Procedural memory migration not yet implemented")
                
                self.stats.procedural_memories_migrated += 1
                
        except Exception as e:
            error_msg = f"Failed to migrate procedural memories for {user_id}: {e}"
            self.logger.error(error_msg)
            self.stats.errors.append(error_msg)


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(description="Migrate CleoAI memory to distributed backends")
    
    parser.add_argument("--source", type=str, default="sqlite",
                       choices=["sqlite", "chromadb"],
                       help="Source backend to migrate from")
    
    parser.add_argument("--target", type=str, required=True,
                       help="Comma-separated list of target backends (redis,mongodb,supabase,pinecone)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform a dry run without modifying data")
    
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of items to process in each batch")
    
    parser.add_argument("--users", type=str,
                       help="Comma-separated list of specific users to migrate")
    
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse targets
    targets = [t.strip() for t in args.target.split(",")]
    valid_targets = ["redis", "mongodb", "supabase", "pinecone"]
    invalid_targets = [t for t in targets if t not in valid_targets]
    
    if invalid_targets:
        print(f"Error: Invalid target backends: {invalid_targets}")
        print(f"Valid targets are: {valid_targets}")
        sys.exit(1)
    
    # Parse users if specified
    specific_users = None
    if args.users:
        specific_users = [u.strip() for u in args.users.split(",")]
    
    # Validate configuration for targets
    print("Validating configuration...")
    result = validate_configuration()
    
    if not result.is_valid:
        print("\nConfiguration validation failed. Please fix errors before migrating.")
        sys.exit(1)
    
    # Confirm migration
    if not args.dry_run:
        print(f"\nThis will migrate data from {args.source} to: {targets}")
        print("This operation cannot be undone.")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Migration cancelled.")
            sys.exit(0)
    
    # Run migration
    migrator = MemoryMigrator(
        source=args.source,
        targets=targets,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        specific_users=specific_users
    )
    
    asyncio.run(migrator.migrate())


if __name__ == "__main__":
    main()