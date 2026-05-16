@echo off
setlocal enabledelayedexpansion

REM Run Stage6 smoke tests in Docker

cd /d "c:\Users\Иван Литвак\source\repos\CayleyBeam100H100"

echo Running Stage6 Dispatcher Skeleton Smoke Tests in Docker...
echo.

docker-compose -f docker-compose.2h100.yml run --rm -T beam-2h100 bash run_stage6_smoke_tests.sh

if errorlevel 1 (
    echo.
    echo ERROR: Docker tests failed
    exit /b 1
) else (
    echo.
    echo SUCCESS: All Stage6 smoke tests passed
    exit /b 0
)
