#!/usr/bin/env python3
"""
Backup monitoring service for CleoAI.

Monitors backup jobs across all databases and sends alerts on failures.
Exposes metrics to Prometheus for monitoring.
"""
import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import requests
import boto3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import Gauge, Counter, Histogram, push_to_gateway, CollectorRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()

backup_last_success_timestamp = Gauge(
    'cleoai_backup_last_success_timestamp',
    'Timestamp of last successful backup',
    ['database', 'backup_type'],
    registry=registry
)

backup_size_bytes = Gauge(
    'cleoai_backup_size_bytes',
    'Size of last backup in bytes',
    ['database', 'backup_type'],
    registry=registry
)

backup_duration_seconds = Histogram(
    'cleoai_backup_duration_seconds',
    'Backup duration in seconds',
    ['database', 'backup_type'],
    buckets=(30, 60, 120, 300, 600, 1200, 3600),
    registry=registry
)

backup_failures_total = Counter(
    'cleoai_backup_failures_total',
    'Total number of backup failures',
    ['database', 'backup_type'],
    registry=registry
)

backup_success_total = Counter(
    'cleoai_backup_success_total',
    'Total number of successful backups',
    ['database', 'backup_type'],
    registry=registry
)


class BackupMonitor:
    """Monitor backup status across all databases."""
    
    def __init__(self):
        """Initialize backup monitor."""
        # S3 configuration
        self.s3_bucket = os.getenv("BACKUP_S3_BUCKET")
        self.s3_client = boto3.client('s3') if self.s3_bucket else None
        
        # Monitoring configuration
        self.check_interval = int(os.getenv("BACKUP_CHECK_INTERVAL", "3600"))  # 1 hour
        self.alert_threshold_hours = int(os.getenv("BACKUP_ALERT_THRESHOLD", "26"))  # Alert if no backup in 26 hours
        
        # Prometheus pushgateway
        self.pushgateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY_URL")
        
        # Alert configuration
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.alert_email = os.getenv("ALERT_EMAIL_TO")
        
        # Databases to monitor
        self.databases = [
            {"name": "postgres", "prefix": "postgres/", "pattern": "*.sql.gz"},
            {"name": "mongodb", "prefix": "mongodb/", "pattern": "*.tar.gz"},
            {"name": "redis", "prefix": "redis/", "pattern": "*.rdb.gz"}
        ]
    
    def check_s3_backups(self, database: dict) -> dict:
        """Check S3 for latest backup of a database."""
        if not self.s3_client:
            return None
        
        try:
            # List objects in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=database['prefix'],
                MaxKeys=1000
            )
            
            if 'Contents' not in response:
                logger.warning(f"No backups found for {database['name']}")
                return None
            
            # Find latest backup
            latest = None
            for obj in response['Contents']:
                if latest is None or obj['LastModified'] > latest['LastModified']:
                    latest = obj
            
            if latest:
                return {
                    'database': database['name'],
                    'file': latest['Key'],
                    'size': latest['Size'],
                    'timestamp': latest['LastModified'],
                    'age_hours': (datetime.utcnow().replace(tzinfo=None) - 
                                 latest['LastModified'].replace(tzinfo=None)).total_seconds() / 3600
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check S3 backups for {database['name']}: {e}")
            return None
    
    def check_local_backups(self, database: dict) -> dict:
        """Check local filesystem for backups."""
        backup_dir = Path(f"/backup/{database['name']}")
        
        if not backup_dir.exists():
            return None
        
        try:
            # Find latest backup file
            backup_files = list(backup_dir.glob(database['pattern']))
            
            if not backup_files:
                return None
            
            # Get latest file
            latest = max(backup_files, key=lambda f: f.stat().st_mtime)
            
            stat = latest.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime)
            
            return {
                'database': database['name'],
                'file': str(latest),
                'size': stat.st_size,
                'timestamp': timestamp,
                'age_hours': (datetime.utcnow() - timestamp).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"Failed to check local backups for {database['name']}: {e}")
            return None
    
    def update_metrics(self, backup_info: dict):
        """Update Prometheus metrics."""
        if not backup_info:
            return
        
        # Update metrics
        backup_last_success_timestamp.labels(
            database=backup_info['database'],
            backup_type='scheduled'
        ).set(backup_info['timestamp'].timestamp())
        
        backup_size_bytes.labels(
            database=backup_info['database'],
            backup_type='scheduled'
        ).set(backup_info['size'])
        
        # Increment success counter if backup is recent
        if backup_info['age_hours'] < 1:  # Backup completed in last hour
            backup_success_total.labels(
                database=backup_info['database'],
                backup_type='scheduled'
            ).inc()
    
    def send_alert(self, message: str, level: str = "error"):
        """Send alert via configured channels."""
        logger.error(f"ALERT: {message}")
        
        # Slack notification
        if self.slack_webhook:
            try:
                color = "danger" if level == "error" else "warning"
                payload = {
                    "attachments": [{
                        "color": color,
                        "title": "CleoAI Backup Alert",
                        "text": message,
                        "footer": "Backup Monitor",
                        "ts": int(time.time())
                    }]
                }
                
                response = requests.post(self.slack_webhook, json=payload, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to send Slack alert: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
        
        # Could add email, PagerDuty, etc. here
    
    def push_metrics(self):
        """Push metrics to Prometheus pushgateway."""
        if not self.pushgateway_url:
            return
        
        try:
            push_to_gateway(
                self.pushgateway_url,
                job='backup_monitor',
                registry=registry
            )
            logger.debug("Metrics pushed to Prometheus")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")
    
    def check_all_backups(self):
        """Check all database backups."""
        alerts = []
        
        for database in self.databases:
            logger.info(f"Checking backups for {database['name']}")
            
            # Check S3 first (preferred)
            backup_info = self.check_s3_backups(database)
            
            # Fall back to local if S3 not available
            if not backup_info:
                backup_info = self.check_local_backups(database)
            
            if backup_info:
                logger.info(
                    f"{database['name']}: Latest backup is {backup_info['age_hours']:.1f} hours old, "
                    f"size: {backup_info['size'] / (1024*1024):.2f} MB"
                )
                
                # Update metrics
                self.update_metrics(backup_info)
                
                # Check if backup is too old
                if backup_info['age_hours'] > self.alert_threshold_hours:
                    alerts.append(
                        f"{database['name']}: No backup for {backup_info['age_hours']:.1f} hours"
                    )
            else:
                logger.error(f"No backup found for {database['name']}")
                alerts.append(f"{database['name']}: No backup found")
                
                # Increment failure counter
                backup_failures_total.labels(
                    database=database['name'],
                    backup_type='scheduled'
                ).inc()
        
        # Send alerts if any
        if alerts:
            message = "Backup issues detected:\n" + "\n".join(f"• {alert}" for alert in alerts)
            self.send_alert(message)
        
        # Push metrics
        self.push_metrics()
    
    def run(self):
        """Run monitoring loop."""
        logger.info("Starting backup monitor")
        
        while True:
            try:
                self.check_all_backups()
            except Exception as e:
                logger.error(f"Monitor check failed: {e}")
                self.send_alert(f"Backup monitor error: {str(e)}")
            
            # Wait for next check
            logger.info(f"Next check in {self.check_interval} seconds")
            time.sleep(self.check_interval)


def main():
    """Main entry point."""
    monitor = BackupMonitor()
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        logger.info("Backup monitor stopped")
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()