# Test script for CleoAI API endpoints

param(
    [string]$BaseUrl = "http://localhost:8000",
    [switch]$Verbose
)

Write-Host "`nCleoAI API Test Suite" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host "Testing API at: $BaseUrl" -ForegroundColor White

# Function to test endpoint
function Test-Endpoint {
    param(
        [string]$Method,
        [string]$Endpoint,
        [object]$Body = $null,
        [string]$Description = ""
    )
    
    Write-Host "`nTesting: $Method $Endpoint" -ForegroundColor Yellow
    if ($Description) {
        Write-Host "  $Description" -ForegroundColor Gray
    }
    
    try {
        $uri = "$BaseUrl$Endpoint"
        $params = @{
            Method = $Method
            Uri = $uri
            ContentType = "application/json"
        }
        
        if ($Body) {
            $params.Body = ($Body | ConvertTo-Json -Depth 10)
        }
        
        $response = Invoke-RestMethod @params
        
        Write-Host "  ✓ Success" -ForegroundColor Green
        if ($Verbose) {
            Write-Host "  Response:" -ForegroundColor Gray
            $response | ConvertTo-Json -Depth 10 | Write-Host -ForegroundColor DarkGray
        }
        
        return $response
    }
    catch {
        Write-Host "  ✗ Failed: $_" -ForegroundColor Red
        if ($Verbose -and $_.Exception.Response) {
            $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "  Error details: $responseBody" -ForegroundColor DarkRed
        }
        return $null
    }
}

# Test if API is running
Write-Host "`nChecking if API is running..." -ForegroundColor Cyan
try {
    $testConnection = Invoke-WebRequest -Uri $BaseUrl -Method Head -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ API is running" -ForegroundColor Green
}
catch {
    Write-Host "✗ API is not running at $BaseUrl" -ForegroundColor Red
    Write-Host "`nPlease start the API first:" -ForegroundColor Yellow
    Write-Host "  python main_api_minimal.py" -ForegroundColor White
    Write-Host "  OR" -ForegroundColor Gray
    Write-Host "  python main.py api" -ForegroundColor White
    exit 1
}

# Run tests
Write-Host "`nRunning API tests..." -ForegroundColor Cyan

# Test 1: Root endpoint
$root = Test-Endpoint -Method "GET" -Endpoint "/" -Description "Get API information"

# Test 2: Health check
$health = Test-Endpoint -Method "GET" -Endpoint "/health" -Description "Check system health"

# Test 3: Test endpoint (if using minimal API)
$test = Test-Endpoint -Method "GET" -Endpoint "/api/test" -Description "Test endpoint"

# Test 4: Echo endpoint (if using minimal API)
$testData = @{
    message = "Hello from PowerShell"
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    test = $true
}
$echo = Test-Endpoint -Method "POST" -Endpoint "/api/echo" -Body $testData -Description "Test POST with echo"

# Test 5: GraphQL endpoint (if available)
$graphqlQuery = @{
    query = @"
{
    healthCheck {
        status
        timestamp
        services
    }
}
"@
}
$graphql = Test-Endpoint -Method "POST" -Endpoint "/graphql" -Body $graphqlQuery -Description "GraphQL health query"

# Summary
Write-Host "`n" + "="*50 -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "="*50 -ForegroundColor Cyan

$tests = @(
    @("Root Endpoint", $null -ne $root),
    @("Health Check", $null -ne $health),
    @("Test Endpoint", $null -ne $test),
    @("Echo Endpoint", $null -ne $echo),
    @("GraphQL", $null -ne $graphql)
)

$passed = 0
foreach ($test in $tests) {
    if ($test[1]) {
        Write-Host "✓ $($test[0])" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "✗ $($test[0])" -ForegroundColor Red
    }
}

Write-Host "`nPassed: $passed/$($tests.Count)" -ForegroundColor $(if ($passed -eq $tests.Count) { "Green" } else { "Yellow" })

# Additional information
if ($health) {
    Write-Host "`nSystem Status:" -ForegroundColor Cyan
    if ($health.status) {
        Write-Host "  Status: $($health.status)" -ForegroundColor $(if ($health.status -eq "healthy") { "Green" } else { "Yellow" })
    }
    if ($health.mode) {
        Write-Host "  Mode: $($health.mode)" -ForegroundColor White
    }
    if ($health.system) {
        Write-Host "  CPU: $($health.system.cpu_percent)%" -ForegroundColor White
        Write-Host "  Memory: $($health.system.memory_percent)%" -ForegroundColor White
    }
}

Write-Host "`nAPI testing complete!" -ForegroundColor Green