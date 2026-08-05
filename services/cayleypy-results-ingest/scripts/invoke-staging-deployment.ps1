[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('preflight', 'store_only', 'activate_normal', 'rollback_store_only')]
  [string]$Phase,
  [Parameter(Mandatory = $true)]
  [string]$ResourceManifest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$serviceRoot = Split-Path -Parent $PSScriptRoot
$wrangler = Join-Path $serviceRoot 'node_modules/.bin/wrangler.cmd'
$baseConfig = Join-Path $serviceRoot 'wrangler.jsonc'
$migrationDir = Join-Path $serviceRoot 'migrations'

function Require-CommandResult([string[]]$Arguments) {
  & $wrangler @Arguments
  if ($LASTEXITCODE -ne 0) { throw "wrangler command failed: $($Arguments -join ' ')" }
}

function Require-ValidConfig([string]$ConfigPath) {
  $validationDir = Join-Path $serviceRoot '.staging-deploy-private/dry-run'
  New-Item -ItemType Directory -Force -Path $validationDir | Out-Null
  Require-CommandResult @('deploy', '--dry-run', '--outdir', $validationDir, '--config', $ConfigPath, '--env', 'staging')
}

function Read-ResourceManifest([string]$Path) {
  $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
  $value = (Get-Content -LiteralPath $resolved -Raw -Encoding UTF8) | ConvertFrom-Json
  $expectedProperties = @('account_label', 'cloudflare_account_id', 'd1_database_name', 'd1_database_id', 'r2_bucket_name', 'validate_queue_name', 'validate_dlq_name')
  $actualProperties = @($value.PSObject.Properties.Name)
  if (@(Compare-Object -ReferenceObject $expectedProperties -DifferenceObject $actualProperties).Count -ne 0) {
    throw 'manifest must contain exactly the reviewed staging resource fields'
  }
  $expected = @{ d1_database_name = 'cayleypy-results-staging'; r2_bucket_name = 'cayleypy-results-raw-staging'; validate_queue_name = 'cayleypy-validate-staging'; validate_dlq_name = 'cayleypy-validate-dlq-staging' }
  foreach ($entry in $expected.GetEnumerator()) {
    if ([string]$value.($entry.Key) -cne $entry.Value) { throw "manifest mismatch: $($entry.Key)" }
  }
  if ([string]::IsNullOrWhiteSpace([string]$value.account_label)) { throw 'manifest account_label must identify the reviewed Cloudflare account' }
  $accountId = [string]$value.cloudflare_account_id
  if ($accountId -notmatch '^[0-9a-fA-F]{32}$') { throw 'manifest cloudflare_account_id must be the exact 32-hex id returned by current wrangler whoami --json' }
  $value.cloudflare_account_id = $accountId.ToLowerInvariant()
  $id = [string]$value.d1_database_id
  if ($id -notmatch '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') { throw 'manifest d1_database_id must be the exact UUID returned by the current wrangler command' }
  return $value
}

function New-GeneratedConfig($Resources, [string]$Mode) {
  $base = (Get-Content -LiteralPath $baseConfig -Raw -Encoding UTF8) | ConvertFrom-Json
  $entryPoint = [System.IO.Path]::GetFullPath((Join-Path $serviceRoot ([string]$base.main)))
  if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) { throw "Worker entry point is missing: $entryPoint" }
  $base.main = $entryPoint
  $staging = $base.env.staging
  if ($null -eq $staging) { throw 'staging environment missing from tracked config' }
  $d1 = @($staging.d1_databases | Where-Object { $_.binding -eq 'RESULTS_DB' })
  if ($d1.Count -ne 1 -or $d1[0].database_name -ne $Resources.d1_database_name) { throw 'unexpected staging D1 binding' }
  $base | Add-Member -NotePropertyName account_id -NotePropertyValue ([string]$Resources.cloudflare_account_id) -Force
  $d1[0] | Add-Member -NotePropertyName database_id -NotePropertyValue ([string]$Resources.d1_database_id) -Force
  $d1[0] | Add-Member -NotePropertyName migrations_dir -NotePropertyValue ([System.IO.Path]::GetFullPath($migrationDir)) -Force
  $staging.vars.INGEST_MODE = $Mode
  $stageDir = Join-Path $serviceRoot '.staging-deploy-private'
  New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
  $generated = Join-Path $stageDir 'wrangler.generated.json'
  $generatedJson = $base | ConvertTo-Json -Depth 100
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($generated, $generatedJson, $utf8NoBom)
  return $generated
}

if (-not (Test-Path -LiteralPath $wrangler)) { throw 'pinned wrangler is unavailable; run the existing exact dependency gate first' }
if (-not (Test-Path -LiteralPath $baseConfig)) { throw 'tracked wrangler.jsonc is missing' }
if (-not (Test-Path -LiteralPath $migrationDir)) { throw 'D1 migrations directory is missing' }
$resources = Read-ResourceManifest $ResourceManifest
$accountOverride = [string]$env:CLOUDFLARE_ACCOUNT_ID
if (-not [string]::IsNullOrWhiteSpace($accountOverride) -and $accountOverride.ToLowerInvariant() -cne [string]$resources.cloudflare_account_id) {
  throw 'CLOUDFLARE_ACCOUNT_ID conflicts with the reviewed private resource manifest'
}
Require-CommandResult @('--version')
Require-CommandResult @('whoami', '--account', [string]$resources.cloudflare_account_id, '--json')
Require-ValidConfig $baseConfig
if ($Phase -eq 'preflight') { Write-Output 'PREFLIGHT_OK: no Cloudflare resource or Worker mutation was attempted'; exit 0 }
$targetMode = if ($Phase -eq 'activate_normal') { 'normal' } else { 'store_only' }
$generatedConfig = New-GeneratedConfig $resources $targetMode
Require-ValidConfig $generatedConfig
if ($Phase -eq 'store_only') { Require-CommandResult @('d1', 'migrations', 'apply', $resources.d1_database_name, '--remote', '--config', $generatedConfig, '--env', 'staging') }
Require-CommandResult @('deploy', '--config', $generatedConfig, '--env', 'staging')
Write-Output "DEPLOYED_MODE=$targetMode"
