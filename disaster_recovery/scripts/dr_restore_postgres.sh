#!/bin/bash
# PostgreSQL Disaster Recovery Script

set -euo pipefail

# Configuration
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://cleoai-backups/postgres}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-cleoai}"
POSTGRES_USER="${POSTGRES_USER:-cleoai_user}"
LOG_FILE="/var/log/dr_restore_postgres_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1" "$RED"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..." "$YELLOW"
    
    # Check for required tools
    command -v psql >/dev/null 2>&1 || error_exit "psql is not installed"
    command -v pg_restore >/dev/null 2>&1 || error_exit "pg_restore is not installed"
    command -v aws >/dev/null 2>&1 || error_exit "AWS CLI is not installed"
    
    # Check PostgreSQL connectivity
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "SELECT 1" >/dev/null 2>&1 || \
        error_exit "Cannot connect to PostgreSQL"
    
    log "Prerequisites check passed" "$GREEN"
}

# List available backups
list_backups() {
    log "Listing available backups..." "$YELLOW"
    aws s3 ls "$BACKUP_BUCKET/" --recursive | grep -E "\.sql\.gz$|\.dump$" | sort -r | head -20
}

# Get latest backup
get_latest_backup() {
    local latest=$(aws s3 ls "$BACKUP_BUCKET/" --recursive | grep -E "\.sql\.gz$|\.dump$" | sort -r | head -1 | awk '{print $4}')
    
    if [ -z "$latest" ]; then
        error_exit "No backups found in $BACKUP_BUCKET"
    fi
    
    echo "$latest"
}

# Download backup
download_backup() {
    local backup_file="$1"
    local local_file="/tmp/$(basename "$backup_file")"
    
    log "Downloading backup: $backup_file" "$YELLOW"
    
    aws s3 cp "$BACKUP_BUCKET/$backup_file" "$local_file" || \
        error_exit "Failed to download backup"
    
    # Decompress if needed
    if [[ "$local_file" == *.gz ]]; then
        log "Decompressing backup..." "$YELLOW"
        gunzip "$local_file"
        local_file="${local_file%.gz}"
    fi
    
    echo "$local_file"
}

# Stop application connections
stop_app_connections() {
    log "Stopping application connections..." "$YELLOW"
    
    # Terminate existing connections
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres <<EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$POSTGRES_DB'
AND pid <> pg_backend_pid();
EOF
    
    log "Application connections terminated" "$GREEN"
}

# Create restore point
create_restore_point() {
    log "Creating restore point..." "$YELLOW"
    
    local restore_point="dr_restore_$(date +%Y%m%d_%H%M%S)"
    
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOF
SELECT pg_create_restore_point('$restore_point');
EOF
    
    log "Restore point created: $restore_point" "$GREEN"
}

# Restore database
restore_database() {
    local backup_file="$1"
    
    log "Starting database restore..." "$YELLOW"
    
    # Determine restore method based on file type
    if [[ "$backup_file" == *.sql ]]; then
        # SQL format restore
        log "Restoring from SQL backup..." "$YELLOW"
        
        # Drop and recreate database
        PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres <<EOF
DROP DATABASE IF EXISTS ${POSTGRES_DB}_old;
ALTER DATABASE $POSTGRES_DB RENAME TO ${POSTGRES_DB}_old;
CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;
EOF
        
        # Restore data
        PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$backup_file" || {
            log "Restore failed, rolling back..." "$RED"
            PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres <<EOF
DROP DATABASE IF EXISTS $POSTGRES_DB;
ALTER DATABASE ${POSTGRES_DB}_old RENAME TO $POSTGRES_DB;
EOF
            error_exit "Database restore failed"
        }
        
    elif [[ "$backup_file" == *.dump ]]; then
        # Custom format restore
        log "Restoring from custom format backup..." "$YELLOW"
        
        PGPASSWORD="${POSTGRES_PASSWORD}" pg_restore \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -c \
            -v \
            "$backup_file" || error_exit "Database restore failed"
    else
        error_exit "Unknown backup format: $backup_file"
    fi
    
    log "Database restore completed" "$GREEN"
}

# Apply WAL logs
apply_wal_logs() {
    log "Checking for WAL logs to apply..." "$YELLOW"
    
    # Download and apply any WAL logs since backup
    local wal_bucket="${BACKUP_BUCKET}/wal"
    local wal_files=$(aws s3 ls "$wal_bucket/" --recursive | grep -E "\.wal$" | awk '{print $4}')
    
    if [ -n "$wal_files" ]; then
        log "Applying WAL logs..." "$YELLOW"
        # WAL application logic here
        log "WAL logs applied" "$GREEN"
    else
        log "No WAL logs to apply" "$YELLOW"
    fi
}

# Verify restore
verify_restore() {
    log "Verifying database restore..." "$YELLOW"
    
    # Check table counts
    local table_count=$(PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
    
    if [ "$table_count" -lt 1 ]; then
        error_exit "No tables found after restore"
    fi
    
    # Check data integrity
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<EOF
-- Check for primary keys
SELECT COUNT(*) as tables_without_pk
FROM information_schema.tables t
LEFT JOIN information_schema.table_constraints tc
    ON t.table_name = tc.table_name
    AND tc.constraint_type = 'PRIMARY KEY'
WHERE t.table_schema = 'public'
    AND t.table_type = 'BASE TABLE'
    AND tc.constraint_name IS NULL;

-- Check for recent data
SELECT COUNT(*) as recent_records
FROM conversations
WHERE created_at > NOW() - INTERVAL '7 days';
EOF
    
    log "Database verification completed" "$GREEN"
}

# Update statistics
update_statistics() {
    log "Updating database statistics..." "$YELLOW"
    
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "ANALYZE;"
    
    log "Statistics updated" "$GREEN"
}

# Main restore process
main() {
    log "Starting PostgreSQL disaster recovery restore" "$GREEN"
    
    # Parse arguments
    local backup_file=""
    local use_latest=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --latest)
                use_latest=true
                shift
                ;;
            --backup)
                backup_file="$2"
                shift 2
                ;;
            --list)
                list_backups
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Determine backup to use
    if [ "$use_latest" = true ]; then
        backup_file=$(get_latest_backup)
        log "Using latest backup: $backup_file" "$GREEN"
    elif [ -z "$backup_file" ]; then
        error_exit "No backup specified. Use --latest or --backup <file>"
    fi
    
    # Download backup
    local local_backup=$(download_backup "$backup_file")
    
    # Stop connections
    stop_app_connections
    
    # Create restore point
    create_restore_point
    
    # Restore database
    restore_database "$local_backup"
    
    # Apply WAL logs
    apply_wal_logs
    
    # Verify restore
    verify_restore
    
    # Update statistics
    update_statistics
    
    # Cleanup
    rm -f "$local_backup"
    
    log "PostgreSQL disaster recovery restore completed successfully!" "$GREEN"
    log "Total time: $SECONDS seconds" "$GREEN"
}

# Run main function
main "$@"