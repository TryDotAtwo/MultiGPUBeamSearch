[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$serviceRoot = Split-Path -Parent $PSScriptRoot
$helper = Join-Path $serviceRoot 'scripts/invoke-staging-deployment.ps1'
$binDir = Join-Path $serviceRoot 'node_modules/.bin'
$fakeWrangler = Join-Path $binDir 'wrangler.cmd'
$deployDir = Join-Path $serviceRoot '.staging-deploy-private'
$manifest = Join-Path $serviceRoot 'staging-resources.private.json'
$calls = Join-Path $serviceRoot '.staging-deploy-test-calls.txt'
$seedDb = Join-Path $serviceRoot '.staging-deploy-test.sqlite3'
$createdBinDir = $false

function Assert-True([bool]$Condition, [string]$Message) {
  if (-not $Condition) { throw "ASSERTION_FAILED: $Message" }
}

if (Test-Path -LiteralPath $fakeWrangler) { throw 'test refuses to replace a real wrangler.cmd' }
if (Test-Path -LiteralPath $deployDir) { throw 'test refuses to replace an existing private deploy directory' }
if (Test-Path -LiteralPath $manifest) { throw 'test refuses to replace an existing staging manifest' }

try {
  if (-not (Test-Path -LiteralPath $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    $createdBinDir = $true
  }
  @"
@echo off
echo %*>>"$calls"
exit /b 0
"@ | Set-Content -LiteralPath $fakeWrangler -Encoding ASCII -NoNewline
  @'
{
  "account_label": "test-only",
  "d1_database_name": "cayleypy-results-staging",
  "d1_database_id": "11111111-2222-4333-8444-555555555555",
  "r2_bucket_name": "cayleypy-results-raw-staging",
  "validate_queue_name": "cayleypy-validate-staging",
  "validate_dlq_name": "cayleypy-validate-dlq-staging"
}
'@ | Set-Content -LiteralPath $manifest -Encoding UTF8 -NoNewline

  & $helper -Phase store_only -ResourceManifest $manifest | Out-Null
  $generated = Join-Path $deployDir 'wrangler.generated.json'
  Assert-True (Test-Path -LiteralPath $generated) 'helper did not generate a staging config'
  $config = Get-Content -LiteralPath $generated -Raw -Encoding UTF8 | ConvertFrom-Json
  $binding = @($config.env.staging.d1_databases | Where-Object binding -eq 'RESULTS_DB')
  Assert-True ($binding.Count -eq 1) 'generated config lost RESULTS_DB'
  $actualMigrations = [System.IO.Path]::GetFullPath([string]$binding[0].migrations_dir)
  $expectedMigrations = [System.IO.Path]::GetFullPath((Join-Path $serviceRoot 'migrations'))
  Assert-True ($actualMigrations -ceq $expectedMigrations) "generated migrations_dir does not resolve to service migrations: $actualMigrations"
  Assert-True (Test-Path -LiteralPath (Join-Path $actualMigrations '0001_initial.sql')) '0001 missing through generated resolver'
  Assert-True (Test-Path -LiteralPath (Join-Path $actualMigrations '0002_ingest_rate_limits.sql')) '0002 missing through generated resolver'
  Assert-True (Test-Path -LiteralPath (Join-Path $actualMigrations '0003_remove_legacy_status_ip_limits.sql')) '0003 missing through generated resolver'

  $recorded = @(Get-Content -LiteralPath $calls -Encoding UTF8)
  Assert-True ($recorded.Count -eq 5) 'unexpected fake Wrangler command count'
  Assert-True ($recorded[3] -match '^d1 migrations apply cayleypy-results-staging --remote --config ') 'migration command missing or reordered'
  Assert-True ($recorded[4] -match '^deploy --config ') 'deploy did not follow migration apply'

  $python = @'
import pathlib, sqlite3, sys
root = pathlib.Path(sys.argv[1])
db_path = sys.argv[2]
conn = sqlite3.connect(db_path)
names = [path.name for path in sorted(root.glob('*.sql'))]
assert names == ['0001_initial.sql', '0002_ingest_rate_limits.sql', '0003_remove_legacy_status_ip_limits.sql']
conn.executescript((root / names[0]).read_text(encoding='utf-8'))
conn.execute('INSERT INTO submissions (submission_id,idempotency_key,run_id,author_name,competition,puzzle_type,puzzle_id,state,raw_r2_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)', ('s','i','r','a','c','p',1,'received','raw','t','t'))
conn.executescript((root / names[1]).read_text(encoding='utf-8'))
scopes = [('status-ip:198.51.100.7', 1, 1), ('global', 1, 2), ('ip:198.51.100.7', 1, 3), ('status-bucket:198.51.100.0/24', 1, 4)]
conn.executemany('INSERT INTO ingest_rate_limits (scope,window_start,count) VALUES (?,?,?)', scopes)
conn.executescript((root / names[2]).read_text(encoding='utf-8'))
assert conn.execute('SELECT COUNT(*) FROM submissions WHERE submission_id=?', ('s',)).fetchone()[0] == 1
assert conn.execute('SELECT COUNT(*) FROM sqlite_master WHERE type=? AND name=?', ('table','ingest_rate_limits')).fetchone()[0] == 1
assert conn.execute('SELECT COUNT(*) FROM sqlite_master WHERE type=? AND name=?', ('index','submissions_recovery')).fetchone()[0] == 1
assert conn.execute('SELECT scope FROM ingest_rate_limits ORDER BY scope').fetchall() == [('global',), ('ip:198.51.100.7',), ('status-bucket:198.51.100.0/24',)]
conn.close()
'@
  & py -c $python $expectedMigrations $seedDb
  if ($LASTEXITCODE -ne 0) { throw 'seeded SQLite migration upgrade failed' }
  Write-Output 'STAGING_MIGRATIONS_RESOLVER_TEST_OK'
}
finally {
  foreach ($path in @($fakeWrangler, $manifest, $calls, $seedDb)) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
  }
  if (Test-Path -LiteralPath $deployDir) { Remove-Item -LiteralPath $deployDir -Recurse -Force }
  if ($createdBinDir -and (Test-Path -LiteralPath $binDir)) { Remove-Item -LiteralPath $binDir -Force }
}
