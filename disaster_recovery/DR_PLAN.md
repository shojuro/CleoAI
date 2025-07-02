# CleoAI Disaster Recovery Plan

## Table of Contents
1. [Overview](#overview)
2. [Recovery Objectives](#recovery-objectives)
3. [Disaster Scenarios](#disaster-scenarios)
4. [Recovery Procedures](#recovery-procedures)
5. [Testing Schedule](#testing-schedule)
6. [Contact Information](#contact-information)

## Overview

This document outlines the disaster recovery (DR) procedures for CleoAI, ensuring business continuity in the event of system failures, data loss, or catastrophic events.

## Recovery Objectives

### Recovery Time Objective (RTO)
- **Critical Services**: 1 hour
- **Non-Critical Services**: 4 hours
- **Full System Recovery**: 8 hours

### Recovery Point Objective (RPO)
- **Database**: 15 minutes (continuous replication)
- **File Storage**: 1 hour (hourly snapshots)
- **Configuration**: Real-time (Git-based)

## Disaster Scenarios

### 1. Database Failure
**Impact**: Loss of conversation history, user data, model states  
**Likelihood**: Medium  
**Severity**: High

### 2. Infrastructure Failure
**Impact**: Service unavailability  
**Likelihood**: Low  
**Severity**: High

### 3. Data Corruption
**Impact**: Inconsistent or unusable data  
**Likelihood**: Low  
**Severity**: Critical

### 4. Security Breach
**Impact**: Data exposure, system compromise  
**Likelihood**: Medium  
**Severity**: Critical

### 5. Natural Disaster
**Impact**: Complete data center loss  
**Likelihood**: Very Low  
**Severity**: Critical

## Recovery Procedures

### Phase 1: Initial Response (0-15 minutes)

1. **Incident Detection**
   ```bash
   # Check system status
   ./scripts/dr_check_status.sh
   
   # Verify backup integrity
   ./scripts/dr_verify_backups.sh
   ```

2. **Incident Classification**
   - Determine severity level
   - Activate appropriate response team
   - Begin incident documentation

3. **Communication**
   - Notify stakeholders
   - Update status page
   - Alert on-call personnel

### Phase 2: Assessment (15-30 minutes)

1. **Damage Assessment**
   ```bash
   # Run comprehensive health check
   ./scripts/dr_health_check.sh --comprehensive
   
   # Generate damage report
   ./scripts/dr_damage_report.sh > damage_$(date +%Y%m%d_%H%M%S).log
   ```

2. **Recovery Strategy Selection**
   - Hot standby failover
   - Warm standby activation
   - Cold recovery from backups

### Phase 3: Recovery Execution (30 minutes - 4 hours)

#### Database Recovery

1. **PostgreSQL Recovery**
   ```bash
   # Stop corrupted instance
   systemctl stop postgresql
   
   # Restore from latest backup
   ./scripts/dr_restore_postgres.sh --latest
   
   # Apply transaction logs
   ./scripts/dr_apply_wal.sh --from-backup
   
   # Verify data integrity
   ./scripts/dr_verify_postgres.sh
   ```

2. **MongoDB Recovery**
   ```bash
   # Restore from backup
   ./scripts/dr_restore_mongo.sh --latest
   
   # Rebuild indexes
   ./scripts/dr_rebuild_indexes.sh
   
   # Verify collections
   ./scripts/dr_verify_mongo.sh
   ```

3. **Redis Recovery**
   ```bash
   # Restore from RDB snapshot
   ./scripts/dr_restore_redis.sh --latest
   
   # Reload AOF if available
   ./scripts/dr_reload_aof.sh
   
   # Warm cache
   ./scripts/dr_warm_cache.sh
   ```

#### Application Recovery

1. **Container Recovery**
   ```bash
   # Pull latest stable images
   ./scripts/dr_pull_images.sh --stable
   
   # Deploy with DR configuration
   kubectl apply -f k8s/disaster-recovery/
   
   # Verify deployments
   ./scripts/dr_verify_deployments.sh
   ```

2. **Data Validation**
   ```bash
   # Run data integrity checks
   ./scripts/dr_validate_data.sh
   
   # Check user access
   ./scripts/dr_test_auth.sh
   
   # Verify API endpoints
   ./scripts/dr_test_api.sh
   ```

### Phase 4: Validation (1-2 hours)

1. **System Testing**
   ```bash
   # Run smoke tests
   pytest tests/dr/test_smoke.py -v
   
   # Run integration tests
   pytest tests/dr/test_integration.py -v
   
   # Performance baseline
   ./scripts/dr_performance_test.sh
   ```

2. **User Acceptance**
   - Test critical user flows
   - Verify data consistency
   - Check performance metrics

### Phase 5: Switchover (30 minutes)

1. **DNS Update**
   ```bash
   # Update DNS records
   ./scripts/dr_update_dns.sh --production
   
   # Verify propagation
   ./scripts/dr_verify_dns.sh
   ```

2. **Load Balancer Configuration**
   ```bash
   # Update load balancer pools
   ./scripts/dr_update_lb.sh --dr-site
   
   # Enable traffic flow
   ./scripts/dr_enable_traffic.sh
   ```

## Backup Procedures

### Automated Backups

1. **Database Backups**
   ```yaml
   # backup-cronjob.yaml
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: database-backup
   spec:
     schedule: "*/15 * * * *"  # Every 15 minutes
     jobTemplate:
       spec:
         template:
           spec:
             containers:
             - name: backup
               image: cleoai/backup-agent:latest
               command: ["/scripts/backup.sh"]
               env:
               - name: BACKUP_TYPE
                 value: "incremental"
   ```

2. **File System Snapshots**
   ```bash
   # Hourly snapshots
   0 * * * * /scripts/create_snapshot.sh
   
   # Daily full backup
   0 2 * * * /scripts/full_backup.sh
   
   # Weekly offsite sync
   0 3 * * 0 /scripts/offsite_sync.sh
   ```

### Manual Recovery Procedures

1. **Point-in-Time Recovery**
   ```bash
   # Restore to specific timestamp
   ./scripts/pitr_restore.sh --timestamp "2024-01-15 14:30:00"
   
   # Verify restoration
   ./scripts/pitr_verify.sh
   ```

2. **Selective Data Recovery**
   ```bash
   # Restore specific tables/collections
   ./scripts/selective_restore.sh --table users --date yesterday
   
   # Restore specific user data
   ./scripts/user_restore.sh --user-id <id> --date <date>
   ```

## Testing Schedule

### Monthly Tests
- Backup restoration drill
- Failover simulation
- Data integrity verification

### Quarterly Tests
- Full DR simulation
- Cross-region failover
- Performance under DR conditions

### Annual Tests
- Complete data center failure simulation
- Multi-region coordination test
- Third-party service failure simulation

## Monitoring and Alerts

### Health Checks
```yaml
# dr-monitoring.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dr-health
spec:
  selector:
    matchLabels:
      app: cleoai-dr
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Alert Rules
```yaml
# dr-alerts.yaml
groups:
- name: disaster_recovery
  rules:
  - alert: BackupFailed
    expr: backup_last_success_timestamp < time() - 3600
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Backup has not succeeded in the last hour"
      
  - alert: ReplicationLag
    expr: replication_lag_seconds > 300
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Database replication lag exceeds 5 minutes"
```

## Contact Information

### Escalation Matrix

| Level | Role | Contact | Response Time |
|-------|------|---------|---------------|
| L1 | On-Call Engineer | PagerDuty | 15 minutes |
| L2 | Team Lead | +1-XXX-XXX-XXXX | 30 minutes |
| L3 | CTO | +1-XXX-XXX-XXXX | 1 hour |

### External Contacts

- **Cloud Provider Support**: 1-800-XXX-XXXX
- **Database Vendor**: support@vendor.com
- **Security Team**: security@cleoai.com

## Post-Incident Procedures

1. **Documentation**
   - Complete incident report
   - Update runbooks
   - Document lessons learned

2. **Analysis**
   - Root cause analysis
   - Impact assessment
   - Prevention measures

3. **Improvements**
   - Update DR procedures
   - Enhance monitoring
   - Schedule additional training

## Appendices

### A. Recovery Scripts Location
- Production: `/opt/cleoai/dr/scripts/`
- Backup: `s3://cleoai-dr-scripts/`
- Documentation: `https://wiki.cleoai.com/dr/`

### B. Configuration Templates
- DR Kubernetes manifests: `k8s/disaster-recovery/`
- Database configs: `configs/dr/`
- Network configs: `network/dr/`

### C. Compliance Requirements
- SOC 2 Type II: Annual DR testing required
- HIPAA: 72-hour breach notification
- GDPR: Data restoration capabilities

---

**Document Version**: 1.0  
**Last Updated**: January 2024  
**Next Review**: April 2024  
**Owner**: Infrastructure Team