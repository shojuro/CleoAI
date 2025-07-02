#!/usr/bin/env python3
"""
Database management CLI for CleoAI.

This script provides commands for:
- Running migrations
- Creating backups
- Restoring from backups
- Database initialization
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_database, create_all_tables, drop_all_tables
from src.database.migrations import migration_manager, backup_manager
from src.monitoring import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


def init_db(args):
    """Initialize database with tables."""
    logger.info("Initializing database...")
    
    try:
        # Initialize connection
        init_database()
        
        # Create tables
        create_all_tables()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)


def create_migration(args):
    """Create a new migration."""
    logger.info(f"Creating migration: {args.message}")
    
    try:
        migration_file = migration_manager.create_migration(
            message=args.message,
            autogenerate=not args.empty
        )
        
        logger.info(f"Migration created: {migration_file}")
        
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        sys.exit(1)


def run_migrations(args):
    """Run pending migrations."""
    logger.info("Running migrations...")
    
    try:
        # Check pending migrations
        pending = migration_manager.get_pending_migrations()
        if not pending:
            logger.info("No pending migrations")
            return
        
        logger.info(f"Found {len(pending)} pending migrations")
        
        # Run migrations
        migration_manager.run_migrations(target=args.target)
        
        logger.info("Migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        sys.exit(1)


def rollback_migration(args):
    """Rollback migrations."""
    logger.warning(f"Rolling back to: {args.target}")
    
    try:
        # Confirm if not forced
        if not args.force:
            response = input("Are you sure you want to rollback? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Rollback cancelled")
                return
        
        # Rollback
        migration_manager.rollback_migration(target=args.target)
        
        logger.info("Rollback completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        sys.exit(1)


def show_status(args):
    """Show migration status."""
    try:
        # Current revision
        current = migration_manager.get_current_revision()
        logger.info(f"Current revision: {current or 'None (empty database)'}")
        
        # Pending migrations
        pending = migration_manager.get_pending_migrations()
        if pending:
            logger.info(f"Pending migrations: {len(pending)}")
            for rev in pending:
                logger.info(f"  - {rev}")
        else:
            logger.info("No pending migrations")
        
        # Validation
        if migration_manager.validate_migrations():
            logger.info("Schema is in sync with models")
        else:
            logger.warning("Schema differs from models - create a new migration")
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        sys.exit(1)


def create_backup(args):
    """Create database backup."""
    logger.info("Creating database backup...")
    
    try:
        backup_file = backup_manager.create_backup(
            name=args.name,
            compress=not args.no_compress
        )
        
        logger.info(f"Backup created: {backup_file}")
        
        # Show backup size
        size_mb = backup_file.stat().st_size / (1024 * 1024)
        logger.info(f"Backup size: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        sys.exit(1)


def restore_backup(args):
    """Restore from backup."""
    backup_file = Path(args.backup_file)
    
    if not backup_file.exists():
        logger.error(f"Backup file not found: {backup_file}")
        sys.exit(1)
    
    logger.warning(f"Restoring from: {backup_file}")
    logger.warning("This will replace all current data!")
    
    try:
        # Confirm if not forced
        if not args.force:
            response = input("Are you sure you want to restore? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Restore cancelled")
                return
        
        # Restore
        backup_manager.restore_backup(backup_file)
        
        logger.info("Database restored successfully")
        
    except Exception as e:
        logger.error(f"Failed to restore: {e}")
        sys.exit(1)


def list_backups(args):
    """List available backups."""
    try:
        backups = backup_manager.list_backups()
        
        if not backups:
            logger.info("No backups found")
            return
        
        logger.info(f"Found {len(backups)} backups:")
        
        for backup in backups:
            size_mb = backup['size'] / (1024 * 1024)
            compressed = " (compressed)" if backup['compressed'] else ""
            logger.info(
                f"  - {backup['name']}: "
                f"{size_mb:.2f} MB, "
                f"created {backup['created'].strftime('%Y-%m-%d %H:%M:%S')}"
                f"{compressed}"
            )
        
    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        sys.exit(1)


def cleanup_backups(args):
    """Clean up old backups."""
    logger.info(f"Cleaning up backups older than {args.days} days...")
    
    try:
        removed = backup_manager.cleanup_old_backups(keep_days=args.days)
        
        if removed > 0:
            logger.info(f"Removed {removed} old backups")
        else:
            logger.info("No old backups to remove")
        
    except Exception as e:
        logger.error(f"Failed to cleanup backups: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CleoAI Database Management"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run"
    )
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize database with tables"
    )
    init_parser.set_defaults(func=init_db)
    
    # Migration commands
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migration commands"
    )
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_command")
    
    # Create migration
    create_parser = migrate_subparsers.add_parser(
        "create",
        help="Create a new migration"
    )
    create_parser.add_argument(
        "message",
        help="Migration message"
    )
    create_parser.add_argument(
        "--empty",
        action="store_true",
        help="Create empty migration (no autogenerate)"
    )
    create_parser.set_defaults(func=create_migration)
    
    # Run migrations
    run_parser = migrate_subparsers.add_parser(
        "run",
        help="Run pending migrations"
    )
    run_parser.add_argument(
        "--target",
        default="head",
        help="Target revision (default: head)"
    )
    run_parser.set_defaults(func=run_migrations)
    
    # Rollback
    rollback_parser = migrate_subparsers.add_parser(
        "rollback",
        help="Rollback migrations"
    )
    rollback_parser.add_argument(
        "--target",
        default="-1",
        help="Target revision (default: -1)"
    )
    rollback_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation"
    )
    rollback_parser.set_defaults(func=rollback_migration)
    
    # Status
    status_parser = migrate_subparsers.add_parser(
        "status",
        help="Show migration status"
    )
    status_parser.set_defaults(func=show_status)
    
    # Backup commands
    backup_parser = subparsers.add_parser(
        "backup",
        help="Backup commands"
    )
    backup_subparsers = backup_parser.add_subparsers(dest="backup_command")
    
    # Create backup
    create_backup_parser = backup_subparsers.add_parser(
        "create",
        help="Create database backup"
    )
    create_backup_parser.add_argument(
        "--name",
        help="Backup name (default: timestamp)"
    )
    create_backup_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress backup"
    )
    create_backup_parser.set_defaults(func=create_backup)
    
    # Restore backup
    restore_parser = backup_subparsers.add_parser(
        "restore",
        help="Restore from backup"
    )
    restore_parser.add_argument(
        "backup_file",
        help="Path to backup file"
    )
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation"
    )
    restore_parser.set_defaults(func=restore_backup)
    
    # List backups
    list_parser = backup_subparsers.add_parser(
        "list",
        help="List available backups"
    )
    list_parser.set_defaults(func=list_backups)
    
    # Cleanup backups
    cleanup_parser = backup_subparsers.add_parser(
        "cleanup",
        help="Clean up old backups"
    )
    cleanup_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Keep backups newer than this many days (default: 30)"
    )
    cleanup_parser.set_defaults(func=cleanup_backups)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # Handle subcommands without function
        if args.command == "migrate" and not args.migrate_command:
            migrate_parser.print_help()
        elif args.command == "backup" and not args.backup_command:
            backup_parser.print_help()
        else:
            parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()