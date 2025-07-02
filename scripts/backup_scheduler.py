#!/usr/bin/env python3
"""
Automated database backup scheduler for CleoAI.

This script runs as a daemon and performs scheduled backups based on configuration.
It can be run standalone or as a systemd service.
"""
import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import schedule

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.migrations import backup_manager
from src.monitoring import (
    setup_logging,
    audit_logger,
    capture_exception,
    add_breadcrumb,
    track_error
)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = threading.Event()


class BackupScheduler:
    """Manages automated database backups."""
    
    def __init__(self):
        """Initialize backup scheduler."""
        # Configuration from environment
        self.enabled = os.getenv("BACKUP_ENABLED", "false").lower() == "true"
        self.schedule_cron = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")  # 2 AM daily
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.compress = os.getenv("BACKUP_COMPRESS", "true").lower() == "true"
        
        # S3 configuration for remote backups
        self.s3_enabled = os.getenv("BACKUP_S3_ENABLED", "false").lower() == "true"
        self.s3_bucket = os.getenv("BACKUP_S3_BUCKET", "")
        self.s3_prefix = os.getenv("BACKUP_S3_PREFIX", "cleoai-backups/")
        
        # Notification settings
        self.notify_on_success = os.getenv("BACKUP_NOTIFY_SUCCESS", "false").lower() == "true"
        self.notify_on_failure = os.getenv("BACKUP_NOTIFY_FAILURE", "true").lower() == "true"
        
        # Parse schedule
        self._parse_schedule()
        
    def _parse_schedule(self):
        """Parse cron-like schedule into schedule.py format."""
        # Simple cron parser for common patterns
        parts = self.schedule_cron.split()
        
        if len(parts) != 5:
            logger.warning(f"Invalid cron format: {self.schedule_cron}, using daily at 2 AM")
            self.schedule_time = "02:00"
            self.schedule_type = "daily"
            return
        
        minute, hour, day, month, weekday = parts
        
        # Daily backup
        if day == "*" and month == "*" and weekday == "*":
            self.schedule_type = "daily"
            self.schedule_time = f"{hour.zfill(2)}:{minute.zfill(2)}"
        # Weekly backup
        elif day == "*" and month == "*" and weekday != "*":
            self.schedule_type = "weekly"
            self.schedule_time = f"{hour.zfill(2)}:{minute.zfill(2)}"
            self.schedule_day = int(weekday) if weekday.isdigit() else 0
        # Hourly backup
        elif hour == "*":
            self.schedule_type = "hourly"
            self.schedule_time = f":{minute.zfill(2)}"
        else:
            # Default to daily
            logger.warning(f"Complex cron pattern not fully supported: {self.schedule_cron}")
            self.schedule_type = "daily"
            self.schedule_time = "02:00"
    
    @track_error
    def perform_backup(self):
        """Perform a scheduled backup."""
        start_time = time.time()
        backup_file = None
        
        try:
            logger.info("Starting scheduled backup")
            add_breadcrumb("Starting scheduled backup", "backup")
            
            # Create local backup
            backup_file = backup_manager.create_backup(compress=self.compress)
            
            # Get backup size
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            
            # Upload to S3 if enabled
            if self.s3_enabled and backup_file:
                self._upload_to_s3(backup_file)
            
            # Clean up old backups
            removed = backup_manager.cleanup_old_backups(self.retention_days)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old backups")
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log success
            logger.info(
                f"Backup completed successfully: {backup_file.name} "
                f"({size_mb:.2f} MB) in {duration:.2f} seconds"
            )
            
            # Audit log
            audit_logger.log_event(
                event_type="scheduled_backup",
                resource_type="database",
                action="create",
                result="success",
                metadata={
                    "backup_file": str(backup_file),
                    "size_mb": size_mb,
                    "duration_seconds": duration,
                    "compressed": self.compress,
                    "uploaded_to_s3": self.s3_enabled
                }
            )
            
            # Send success notification if enabled
            if self.notify_on_success:
                self._send_notification(
                    "Backup Success",
                    f"Database backup completed: {backup_file.name} ({size_mb:.2f} MB)"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            capture_exception(e, context={"component": "backup_scheduler"})
            
            # Audit log
            audit_logger.log_event(
                event_type="scheduled_backup",
                resource_type="database",
                action="create",
                result="failure",
                error_message=str(e)
            )
            
            # Send failure notification
            if self.notify_on_failure:
                self._send_notification(
                    "Backup Failed",
                    f"Database backup failed: {str(e)}",
                    level="error"
                )
            
            return False
    
    def _upload_to_s3(self, backup_file: Path):
        """Upload backup to S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Generate S3 key
            s3_key = f"{self.s3_prefix}{backup_file.name}"
            
            logger.info(f"Uploading backup to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Upload file
            s3_client.upload_file(
                str(backup_file),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA',  # Infrequent access for cost savings
                    'Metadata': {
                        'backup-date': datetime.utcnow().isoformat(),
                        'backup-type': 'scheduled',
                        'compressed': str(self.compress)
                    }
                }
            )
            
            logger.info("Backup uploaded to S3 successfully")
            
            # Clean up old S3 backups
            self._cleanup_s3_backups()
            
        except ClientError as e:
            logger.error(f"Failed to upload backup to S3: {e}")
            raise
        except ImportError:
            logger.error("boto3 not installed, cannot upload to S3")
            raise
    
    def _cleanup_s3_backups(self):
        """Clean up old backups from S3."""
        try:
            import boto3
            
            s3_client = boto3.client('s3')
            
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # List objects
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            # Find old backups
            objects_to_delete = []
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        objects_to_delete.append({'Key': obj['Key']})
            
            # Delete old backups
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=self.s3_bucket,
                    Delete={'Objects': objects_to_delete}
                )
                logger.info(f"Deleted {len(objects_to_delete)} old backups from S3")
                
        except Exception as e:
            logger.error(f"Failed to cleanup S3 backups: {e}")
    
    def _send_notification(self, subject: str, message: str, level: str = "info"):
        """Send notification about backup status."""
        try:
            # Try multiple notification methods
            
            # Slack webhook
            slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
            if slack_webhook:
                import requests
                
                color = "good" if level == "info" else "danger"
                payload = {
                    "attachments": [{
                        "color": color,
                        "title": subject,
                        "text": message,
                        "footer": "CleoAI Backup Scheduler",
                        "ts": int(time.time())
                    }]
                }
                
                response = requests.post(slack_webhook, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.debug("Slack notification sent")
                    return
            
            # Email notification
            smtp_host = os.getenv("SMTP_HOST")
            if smtp_host:
                import smtplib
                from email.mime.text import MIMEText
                
                msg = MIMEText(message)
                msg['Subject'] = f"[CleoAI] {subject}"
                msg['From'] = os.getenv("ALERT_EMAIL_FROM", "cleoai@localhost")
                msg['To'] = os.getenv("ALERT_EMAIL_TO", "admin@localhost")
                
                with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", "587"))) as server:
                    if os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD"):
                        server.starttls()
                        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                    server.send_message(msg)
                    logger.debug("Email notification sent")
                    
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def setup_schedule(self):
        """Set up the backup schedule."""
        if not self.enabled:
            logger.info("Backup scheduler is disabled")
            return
        
        logger.info(
            f"Setting up backup schedule: {self.schedule_type} at {self.schedule_time}"
        )
        
        # Clear existing jobs
        schedule.clear()
        
        # Set up schedule based on type
        if self.schedule_type == "daily":
            schedule.every().day.at(self.schedule_time).do(self.perform_backup)
        elif self.schedule_type == "weekly":
            schedule.every().week.at(self.schedule_time).do(self.perform_backup)
        elif self.schedule_type == "hourly":
            schedule.every().hour.at(self.schedule_time).do(self.perform_backup)
        
        # Log next run time
        next_run = schedule.next_run()
        if next_run:
            logger.info(f"Next backup scheduled for: {next_run}")
    
    def run(self):
        """Run the scheduler."""
        if not self.enabled:
            logger.info("Backup scheduler is disabled, exiting")
            return
        
        logger.info("Starting backup scheduler")
        
        # Set up schedule
        self.setup_schedule()
        
        # Run scheduler loop
        while not shutdown_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                capture_exception(e)
                time.sleep(300)  # Wait 5 minutes on error
        
        logger.info("Backup scheduler stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()


def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run scheduler
    scheduler = BackupScheduler()
    
    # Check if we should run once or as daemon
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run single backup
        logger.info("Running single backup")
        success = scheduler.perform_backup()
        sys.exit(0 if success else 1)
    else:
        # Run as daemon
        try:
            scheduler.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Scheduler failed: {e}")
            capture_exception(e)
            sys.exit(1)


if __name__ == "__main__":
    main()