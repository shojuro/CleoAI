# Docker Setup Guide for CleoAI

## Installing Docker and Docker Compose

### Windows Installation

#### Option 1: Docker Desktop (Recommended)
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Docker Desktop includes Docker Compose v2 built-in
4. After installation, restart your computer
5. Open Docker Desktop and ensure it's running

#### Option 2: Docker Compose Standalone
If you have Docker but not Docker Compose:
```powershell
# Download Docker Compose (PowerShell as Administrator)
Invoke-WebRequest "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-windows-x86_64.exe" -OutFile "$Env:ProgramFiles\Docker\docker-compose.exe"
```

### Verify Installation

```bash
# Check Docker
docker --version

# Check Docker Compose (v2 syntax)
docker compose version

# Or legacy syntax
docker-compose --version
```

## Using Docker Compose with CleoAI

### Modern Docker Compose (v2) Syntax

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Check service status
docker compose ps
```

### Legacy Docker Compose Syntax

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down
```

## Alternative: Run Services Without Docker

If you prefer not to use Docker, you can run the services locally:

### 1. Redis
```bash
# Windows (using WSL or Redis for Windows)
# Download from: https://github.com/microsoftarchive/redis/releases

# Or use Memurai (Redis for Windows)
# Download from: https://www.memurai.com/
```

### 2. MongoDB
```bash
# Download MongoDB Community Server
# https://www.mongodb.com/try/download/community

# Start MongoDB
mongod --dbpath C:\data\db
```

### 3. Start CleoAI with Local Services

Update your `.env` file:
```env
REDIS_HOST=localhost
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/cleoai
```

Then start CleoAI:
```bash
python main.py api
```

## Quick Start Script (No Docker)

Create `start_local.ps1` for PowerShell:
```powershell
# Start Redis (if installed)
Start-Process -FilePath "redis-server.exe"

# Start MongoDB (if installed)
Start-Process -FilePath "mongod.exe" -ArgumentList "--dbpath", "C:\data\db"

# Wait for services
Start-Sleep -Seconds 5

# Start CleoAI API
python main.py api
```

## Troubleshooting

### "docker-compose: command not found"
- Install Docker Desktop (includes Docker Compose)
- Or use the new syntax: `docker compose` (with space)

### "Cannot connect to Docker daemon"
- Ensure Docker Desktop is running
- On Windows, check if WSL2 is properly configured

### Port conflicts
- Check if ports are already in use:
  ```bash
  netstat -an | findstr :6379  # Redis
  netstat -an | findstr :27017 # MongoDB
  netstat -an | findstr :8000  # API
  ```

## Running Without Docker

For development without Docker, you can:

1. **Use SQLite + ChromaDB only** (no setup required):
   ```env
   USE_SQLITE=true
   USE_CHROMADB=true
   USE_REDIS=false
   USE_MONGODB=false
   ```

2. **Install services locally** and configure `.env`

3. **Use cloud services**:
   - Redis: Use Redis Cloud (free tier available)
   - MongoDB: Use MongoDB Atlas (free tier available)
   - Already configured for Supabase and Pinecone