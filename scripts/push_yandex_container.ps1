param(
    [string]$Registry = "cr.yandex/crp7o66ucs8c14sjctp5",
    [string]$ImageName = "multigpu-beam-search",
    [string]$Tag = "a100-kaggle-2t4-baseline",
    [string]$LocalImage = "cayley-beam-h100:latest",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$RemoteImage = "$Registry/$ImageName`:$Tag"

Write-Host "entity_id=yandex_push; stage=auth_check; registry=$Registry"
$yc = Get-Command yc -ErrorAction SilentlyContinue
if ($yc) {
    yc container registry configure-docker
} else {
    Write-Host "entity_id=yandex_push; stage=auth_hint; yc_cli=missing"
    Write-Host "Run one auth option before push:"
    Write-Host "  yc container registry configure-docker"
    Write-Host "  docker login --username oauth --password <YANDEX_OAUTH_TOKEN> cr.yandex"
}

if (-not $SkipBuild) {
    Write-Host "entity_id=yandex_push; stage=build; image=$LocalImage"
    docker build -t $LocalImage .
}

Write-Host "entity_id=yandex_push; stage=tag; local=$LocalImage; remote=$RemoteImage"
docker tag $LocalImage $RemoteImage

Write-Host "entity_id=yandex_push; stage=push; remote=$RemoteImage"
docker push $RemoteImage

Write-Host "entity_id=yandex_push; stage=done; image=$RemoteImage"
