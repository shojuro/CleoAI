#!/bin/sh
# Redis backup script for Docker container

set -e

# Configuration
BACKUP_DIR="/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="redis_backup_${TIMESTAMP}.rdb"

# Install required tools
apk add --no-cache aws-cli dcron

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Function to perform backup
perform_backup() {
    echo "[$(date)] Starting Redis backup..."
    
    # Connect to Redis and create backup
    redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD} --no-auth-warning BGSAVE
    
    # Wait for backup to complete
    while [ $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD} --no-auth-warning LASTSAVE) -eq $(redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD} --no-auth-warning LASTSAVE) ]; do
        sleep 1
    done
    
    # Copy backup file
    redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD} --no-auth-warning --rdb ${BACKUP_DIR}/${BACKUP_FILE}
    
    # Compress backup
    gzip ${BACKUP_DIR}/${BACKUP_FILE}
    
    # Upload to S3 if configured
    if [ -n "${S3_BUCKET}" ]; then
        echo "[$(date)] Uploading to S3..."
        aws s3 cp ${BACKUP_DIR}/${BACKUP_FILE}.gz s3://${S3_BUCKET}/${S3_PREFIX}${BACKUP_FILE}.gz \
            --storage-class STANDARD_IA \
            --metadata "backup-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        
        # Clean up old S3 backups
        if [ -n "${BACKUP_RETENTION_DAYS}" ]; then
            echo "[$(date)] Cleaning up old backups..."
            CUTOFF_DATE=$(date -d "${BACKUP_RETENTION_DAYS} days ago" +%Y-%m-%d)
            
            aws s3 ls s3://${S3_BUCKET}/${S3_PREFIX} | while read -r line; do
                FILE_DATE=$(echo $line | awk '{print $1}')
                FILE_NAME=$(echo $line | awk '{print $4}')
                
                if [ "${FILE_DATE}" \< "${CUTOFF_DATE}" ]; then
                    echo "Deleting old backup: ${FILE_NAME}"
                    aws s3 rm s3://${S3_BUCKET}/${S3_PREFIX}${FILE_NAME}
                fi
            done
        fi
    fi
    
    # Clean up local backups older than 7 days
    find ${BACKUP_DIR} -name "redis_backup_*.gz" -mtime +7 -delete
    
    echo "[$(date)] Redis backup completed: ${BACKUP_FILE}.gz"
}

# If running as cron job
if [ -n "${BACKUP_SCHEDULE}" ]; then
    echo "Setting up cron schedule: ${BACKUP_SCHEDULE}"
    echo "${BACKUP_SCHEDULE} /scripts/backup.sh backup" > /etc/crontabs/root
    
    # Run initial backup
    perform_backup
    
    # Start cron daemon
    crond -f -l 8
else
    # Run single backup
    perform_backup
fi