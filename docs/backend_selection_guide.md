# CleoAI Backend Selection Guide

This guide helps you choose the right memory storage backends for your CleoAI deployment based on your specific requirements, scale, and use case.

## Table of Contents
- [Overview](#overview)
- [Backend Comparison](#backend-comparison)
- [Use Case Recommendations](#use-case-recommendations)
- [Configuration Examples](#configuration-examples)
- [Performance Considerations](#performance-considerations)
- [Cost Analysis](#cost-analysis)
- [Migration Strategies](#migration-strategies)

## Overview

CleoAI supports multiple memory storage backends, each optimized for different aspects of the memory system:

### Memory Types and Their Requirements

1. **Short-Term Memory** (Active Conversations)
   - High-speed read/write
   - TTL (Time To Live) support
   - Small data size per item
   - Frequent updates

2. **Long-Term Memory** (User Preferences)
   - Persistent storage
   - Structured queries
   - Moderate read/write frequency
   - ACID compliance beneficial

3. **Episodic Memory** (Events & Experiences)
   - Vector similarity search
   - Metadata filtering
   - Large storage capacity
   - Infrequent updates

4. **Procedural Memory** (Protocols & Tasks)
   - JSON document storage
   - Complex query support
   - Version tracking
   - Transactional updates

## Backend Comparison

### Redis
**Best for:** Short-term memory, conversation cache, session management

**Pros:**
- ✅ Sub-millisecond latency
- ✅ Built-in TTL support
- ✅ Excellent for hot data
- ✅ Pub/sub capabilities
- ✅ Simple setup

**Cons:**
- ❌ Memory-limited (expensive for large datasets)
- ❌ No complex queries
- ❌ Persistence not guaranteed
- ❌ Single-threaded for writes

**Recommended Configuration:**
```env
USE_REDIS=true
REDIS_TTL_MINUTES=60  # Keep conversations for 1 hour
```

### MongoDB
**Best for:** Archival storage, document-based memories, flexible schemas

**Pros:**
- ✅ Excellent for unstructured data
- ✅ Flexible schema
- ✅ Good for time-series data
- ✅ Built-in aggregation
- ✅ Horizontal scaling

**Cons:**
- ❌ Higher latency than Redis
- ❌ No built-in vector search
- ❌ Can be memory-intensive
- ❌ Complex for relational data

**Recommended Configuration:**
```env
USE_MONGODB=true
MONGODB_ARCHIVE_DAYS=90  # Archive after 90 days
```

### Supabase (PostgreSQL)
**Best for:** Structured data, user preferences, relational queries, real-time sync

**Pros:**
- ✅ ACID compliance
- ✅ Complex queries with SQL
- ✅ Real-time subscriptions
- ✅ Built-in auth & RLS
- ✅ pgvector for embeddings
- ✅ Managed service

**Cons:**
- ❌ Network latency
- ❌ Not ideal for unstructured data
- ❌ Limited by connection pool
- ❌ Costs can scale with usage

**Recommended Configuration:**
```env
USE_SUPABASE=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-key
```

### Pinecone
**Best for:** Vector embeddings, semantic search, similarity matching

**Pros:**
- ✅ Purpose-built for vectors
- ✅ Blazing fast similarity search
- ✅ Metadata filtering
- ✅ Fully managed
- ✅ Namespaces for multi-tenancy

**Cons:**
- ❌ Only for vector data
- ❌ Requires separate metadata store
- ❌ Can be expensive at scale
- ❌ Limited to embeddings

**Recommended Configuration:**
```env
USE_PINECONE=true
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-east-1
```

### SQLite (Legacy)
**Best for:** Development, single-user deployments, offline-first applications

**Pros:**
- ✅ Zero configuration
- ✅ No external dependencies
- ✅ File-based (easy backup)
- ✅ Good for development

**Cons:**
- ❌ Single writer limitation
- ❌ No network access
- ❌ Limited concurrency
- ❌ Not suitable for production

### ChromaDB (Legacy)
**Best for:** Local vector storage, development environments

**Pros:**
- ✅ Simple setup
- ✅ Local vector search
- ✅ Good for prototyping

**Cons:**
- ❌ Not production-ready
- ❌ Limited scalability
- ❌ No cloud offering

## Use Case Recommendations

### 🚀 High-Performance Production
**Requirements:** Low latency, high concurrency, 24/7 availability

**Recommended Stack:**
- **Redis**: Short-term memory & cache
- **Supabase**: User preferences & structured data
- **Pinecone**: Vector embeddings for semantic search
- **MongoDB**: Long-term archival

```env
USE_REDIS=true
USE_SUPABASE=true
USE_PINECONE=true
USE_MONGODB=true
USE_SQLITE=false
USE_CHROMADB=false
```

### 💰 Cost-Optimized Production
**Requirements:** Good performance, managed costs, moderate scale

**Recommended Stack:**
- **Redis**: Short-term memory only
- **PostgreSQL** (self-hosted): All relational data + pgvector
- **MongoDB**: Archival and documents

```env
USE_REDIS=true
USE_MONGODB=true
USE_SQLITE=false  # Or true for read-only fallback
USE_CHROMADB=false
USE_SUPABASE=false
USE_PINECONE=false
```

### 🧪 Development Environment
**Requirements:** Easy setup, low cost, rapid iteration

**Recommended Stack:**
- **SQLite**: All relational data
- **ChromaDB**: Vector storage
- **Redis** (optional): If testing distributed features

```env
USE_SQLITE=true
USE_CHROMADB=true
USE_REDIS=false
USE_MONGODB=false
USE_SUPABASE=false
USE_PINECONE=false
```

### 🏢 Enterprise Deployment
**Requirements:** Compliance, audit trails, high availability, disaster recovery

**Recommended Stack:**
- **Redis Cluster**: HA caching layer
- **Supabase/PostgreSQL**: Primary database with replication
- **Pinecone**: Managed vector search
- **MongoDB Atlas**: Managed document store with backup

```env
USE_REDIS=true
USE_SUPABASE=true
USE_PINECONE=true
USE_MONGODB=true
# Enable audit logging
ENABLE_AUDIT_LOG=true
ENABLE_ENCRYPTION=true
```

### 📱 Edge/Mobile Deployment
**Requirements:** Offline capability, sync when online, minimal footprint

**Recommended Stack:**
- **SQLite**: Local storage
- **Supabase**: Cloud sync when online
- **Redis**: Optional local cache

```env
USE_SQLITE=true
USE_SUPABASE=true  # For sync
USE_REDIS=false
USE_MONGODB=false
USE_PINECONE=false
USE_CHROMADB=false
```

## Configuration Examples

### Minimal Configuration (Development)
```env
# .env for development
USE_SQLITE=true
USE_CHROMADB=true
# All others default to false
```

### Balanced Configuration (Small Production)
```env
# .env for small production
USE_REDIS=true
REDIS_HOST=localhost
REDIS_PORT=6379

USE_MONGODB=true
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/cleoai

USE_SQLITE=true  # Fallback
USE_CHROMADB=false
USE_SUPABASE=false
USE_PINECONE=false
```

### Full Configuration (Large Production)
```env
# .env for large production
USE_REDIS=true
REDIS_HOST=redis.internal.company.com
REDIS_PORT=6379
REDIS_PASSWORD=secure_password
REDIS_TTL_MINUTES=120

USE_MONGODB=true
MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/

USE_SUPABASE=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

USE_PINECONE=true
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east-1

USE_SQLITE=false
USE_CHROMADB=false

# Performance tuning
MAX_MEMORY_RESULTS=100
MONGODB_ARCHIVE_DAYS=365
```

## Performance Considerations

### Latency Comparison
| Backend | Read Latency | Write Latency | Use Case |
|---------|--------------|---------------|----------|
| Redis | <1ms | <1ms | Hot cache |
| SQLite | 1-5ms | 5-20ms | Local only |
| PostgreSQL | 5-20ms | 10-30ms | Structured |
| MongoDB | 10-30ms | 20-50ms | Documents |
| Pinecone | 20-50ms | 50-100ms | Vectors |

### Throughput Guidelines
- **Redis**: 100k+ ops/sec
- **PostgreSQL**: 10k ops/sec
- **MongoDB**: 50k ops/sec
- **Pinecone**: 1k queries/sec

### Memory Requirements
- **Development**: 8GB RAM minimum
- **Small Production**: 16GB RAM
- **Large Production**: 32GB+ RAM

## Cost Analysis

### Estimated Monthly Costs

#### Small Deployment (1k users)
- **Redis** (2GB): $15/month
- **MongoDB** (10GB): $57/month
- **Supabase**: Free tier
- **Pinecone**: Free tier
- **Total**: ~$72/month

#### Medium Deployment (10k users)
- **Redis** (8GB): $60/month
- **MongoDB** (100GB): $180/month
- **Supabase**: $25/month
- **Pinecone**: $70/month
- **Total**: ~$335/month

#### Large Deployment (100k+ users)
- **Redis Cluster** (32GB): $500/month
- **MongoDB Atlas** (1TB): $1,200/month
- **Supabase Pro**: $599/month
- **Pinecone** (10M vectors): $700/month
- **Total**: ~$3,000/month

## Migration Strategies

### Gradual Migration
1. **Phase 1**: Add Redis for caching (keep SQLite)
2. **Phase 2**: Add MongoDB for archival
3. **Phase 3**: Move to Supabase for structured data
4. **Phase 4**: Add Pinecone for vectors
5. **Phase 5**: Disable legacy backends

### Big Bang Migration
1. Set up all distributed backends
2. Run migration script
3. Verify data integrity
4. Switch over during maintenance window
5. Keep legacy as read-only backup

### Hybrid Approach (Recommended)
1. Enable new backends alongside legacy
2. Write to both systems
3. Gradually move reads to new system
4. Monitor performance
5. Disable legacy after validation

### Migration Commands
```bash
# Dry run first
python scripts/migrate_memory.py --source sqlite --target redis,mongodb,supabase,pinecone --dry-run

# Migrate specific users
python scripts/migrate_memory.py --source sqlite --target redis,mongodb --users user1,user2

# Full migration
python scripts/migrate_memory.py --source sqlite --target redis,mongodb,supabase,pinecone
```

## Best Practices

1. **Always validate configuration** before starting
   ```bash
   python -c "from src.utils.config_validator import validate_configuration; validate_configuration()"
   ```

2. **Monitor health endpoints**
   - `/health` - Basic health check
   - `/health/detailed` - Comprehensive status

3. **Use connection pooling** for databases

4. **Implement retry logic** for network operations

5. **Set up monitoring** for all backends

6. **Regular backups** of persistent stores

7. **Test failover scenarios**

## Troubleshooting

### Common Issues

1. **Redis Connection Refused**
   - Check if Redis is running
   - Verify host/port configuration
   - Check firewall rules

2. **MongoDB Timeout**
   - Verify connection string
   - Check network connectivity
   - Increase timeout values

3. **Pinecone Index Not Found**
   - Create index first
   - Verify index name in config
   - Check API key permissions

4. **Supabase Rate Limits**
   - Implement exponential backoff
   - Use connection pooling
   - Consider upgrading plan

### Debug Commands
```bash
# Check backend connectivity
curl http://localhost:8000/health/detailed

# Test specific backend
python -c "from src.utils.health_check import HealthChecker; import asyncio; asyncio.run(HealthChecker(config).check_all())"

# Verify migration
python scripts/migrate_memory.py --source sqlite --target redis --dry-run --log-level DEBUG
```

## Conclusion

Choosing the right backend combination depends on your specific requirements:

- **For Development**: SQLite + ChromaDB
- **For Small Production**: Redis + MongoDB + SQLite
- **For Large Production**: Redis + Supabase + Pinecone + MongoDB
- **For Enterprise**: All backends with HA configuration

Remember to:
1. Start simple and scale as needed
2. Monitor performance metrics
3. Plan for growth
4. Test thoroughly before production
5. Keep backups of critical data

For questions or issues, please refer to the [GitHub Issues](https://github.com/yourusername/CleoAI/issues) page.