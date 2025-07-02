# Setup Redis for Windows
# This script helps install and configure Redis for CleoAI development

param(
    [Parameter(Position=0)]
    [ValidateSet("check", "install", "start", "test")]
    [string]$Action = "check"
)

Write-Host "`nRedis Setup for Windows" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

function Test-RedisInstalled {
    try {
        $redis = Get-Command redis-cli -ErrorAction SilentlyContinue
        if ($redis) {
            return $redis.Source
        }
        
        # Check common installation paths
        $paths = @(
            "C:\Program Files\Redis\redis-cli.exe",
            "C:\Redis\redis-cli.exe",
            "$env:USERPROFILE\scoop\apps\redis\current\redis-cli.exe",
            "C:\tools\redis\redis-cli.exe"
        )
        
        foreach ($path in $paths) {
            if (Test-Path $path) {
                return $path
            }
        }
        
        return $null
    } catch {
        return $null
    }
}

function Test-RedisRunning {
    try {
        $tcpConnection = Test-NetConnection -ComputerName localhost -Port 6379 -ErrorAction SilentlyContinue
        return $tcpConnection.TcpTestSucceeded
    } catch {
        return $false
    }
}

switch ($Action) {
    "check" {
        Write-Host "`nChecking Redis status..." -ForegroundColor Yellow
        
        $redisPath = Test-RedisInstalled
        if ($redisPath) {
            Write-Host "  ✓ Redis installed at: $redisPath" -ForegroundColor Green
            
            if (Test-RedisRunning) {
                Write-Host "  ✓ Redis is running on port 6379" -ForegroundColor Green
                
                # Try to connect
                try {
                    $result = & redis-cli ping 2>$null
                    if ($result -eq "PONG") {
                        Write-Host "  ✓ Redis connection successful" -ForegroundColor Green
                    }
                } catch {
                    Write-Host "  ⚠ Redis running but cannot connect" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  ✗ Redis is not running" -ForegroundColor Red
                Write-Host "`nTo start Redis, run: .\setup_redis_windows.ps1 start" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ✗ Redis is not installed" -ForegroundColor Red
            Write-Host "`nTo install Redis, run: .\setup_redis_windows.ps1 install" -ForegroundColor Yellow
        }
    }
    
    "install" {
        Write-Host "`nRedis Installation Options:" -ForegroundColor Cyan
        Write-Host "1. Memurai (Recommended - Redis for Windows)" -ForegroundColor White
        Write-Host "   - Download from: https://www.memurai.com/get-memurai" -ForegroundColor Gray
        Write-Host "   - Free Developer Edition available" -ForegroundColor Gray
        Write-Host ""
        Write-Host "2. Redis Windows Port (Legacy)" -ForegroundColor White
        Write-Host "   - Download from: https://github.com/microsoftarchive/redis/releases" -ForegroundColor Gray
        Write-Host "   - Version 3.2.100 (older but stable)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "3. WSL2 (Windows Subsystem for Linux)" -ForegroundColor White
        Write-Host "   - Install WSL2: wsl --install" -ForegroundColor Gray
        Write-Host "   - Then: sudo apt update && sudo apt install redis-server" -ForegroundColor Gray
        Write-Host ""
        Write-Host "4. Docker (if Docker Desktop installed)" -ForegroundColor White
        Write-Host "   - docker run -d -p 6379:6379 redis:alpine" -ForegroundColor Gray
        
        Write-Host "`nAfter installation, run: .\setup_redis_windows.ps1 check" -ForegroundColor Yellow
    }
    
    "start" {
        Write-Host "`nStarting Redis..." -ForegroundColor Yellow
        
        $redisPath = Test-RedisInstalled
        if (-not $redisPath) {
            Write-Host "  ✗ Redis is not installed" -ForegroundColor Red
            Write-Host "  Run: .\setup_redis_windows.ps1 install" -ForegroundColor Yellow
            exit 1
        }
        
        if (Test-RedisRunning) {
            Write-Host "  ✓ Redis is already running" -ForegroundColor Green
            exit 0
        }
        
        # Try to find redis-server
        $redisServer = $redisPath -replace "redis-cli\.exe", "redis-server.exe"
        if (Test-Path $redisServer) {
            Write-Host "  Starting Redis server..." -ForegroundColor Cyan
            Start-Process -FilePath $redisServer -WindowStyle Minimized
            Start-Sleep -Seconds 2
            
            if (Test-RedisRunning) {
                Write-Host "  ✓ Redis started successfully" -ForegroundColor Green
            } else {
                Write-Host "  ✗ Failed to start Redis" -ForegroundColor Red
            }
        } else {
            Write-Host "  ✗ Cannot find redis-server.exe" -ForegroundColor Red
        }
    }
    
    "test" {
        Write-Host "`nTesting Redis connection..." -ForegroundColor Yellow
        
        if (-not (Test-RedisRunning)) {
            Write-Host "  ✗ Redis is not running" -ForegroundColor Red
            Write-Host "  Run: .\setup_redis_windows.ps1 start" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "  Running Redis tests..." -ForegroundColor Cyan
        
        # Test 1: Ping
        try {
            $result = & redis-cli ping 2>$null
            if ($result -eq "PONG") {
                Write-Host "  ✓ PING test passed" -ForegroundColor Green
            }
        } catch {
            Write-Host "  ✗ PING test failed" -ForegroundColor Red
        }
        
        # Test 2: Set/Get
        try {
            & redis-cli SET test_key "Hello from CleoAI" | Out-Null
            $value = & redis-cli GET test_key
            if ($value -eq "Hello from CleoAI") {
                Write-Host "  ✓ SET/GET test passed" -ForegroundColor Green
            }
            & redis-cli DEL test_key | Out-Null
        } catch {
            Write-Host "  ✗ SET/GET test failed" -ForegroundColor Red
        }
        
        # Test 3: Python Redis client
        Write-Host "`n  Testing Python Redis client..." -ForegroundColor Cyan
        
        $pythonTest = @"
import sys
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print('  ✓ Python Redis client working')
except ImportError:
    print('  ✗ Redis Python package not installed')
    print('    Run: pip install redis')
    sys.exit(1)
except Exception as e:
    print(f'  ✗ Python Redis connection failed: {e}')
    sys.exit(1)
"@
        
        $pythonTest | python
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ Redis is properly configured for CleoAI!" -ForegroundColor Green
            
            # Update .env to enable Redis
            Write-Host "`nWould you like to enable Redis in your .env file? (y/n)" -ForegroundColor Yellow
            $response = Read-Host
            if ($response -eq 'y') {
                $envContent = Get-Content .env
                $envContent = $envContent -replace 'USE_REDIS=false', 'USE_REDIS=true'
                $envContent | Set-Content .env
                Write-Host "  ✓ Updated .env to enable Redis" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""