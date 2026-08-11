# run_map_eval.ps1
# XRayEarth -- Run mAP evaluation for all versions sequentially
#
# Usage:
#   .\scripts\run_map_eval.ps1
#   .\scripts\run_map_eval.ps1 -Versions v1,v2,v3
#   .\scripts\run_map_eval.ps1 -Map50Only
#   .\scripts\run_map_eval.ps1 -MaxTiles 100   # quick debug

param(
    [string[]]$Versions   = @("v1","v2","v3","v4","v5","v6","v7","v8","v9","v10"),
    [switch]$Map50Only,
    [int]$MaxTiles        = 0,
    [string]$OutputDir    = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  XRayEarth -- mAP Evaluation" -ForegroundColor Cyan
Write-Host "  Versions: $($Versions -join ', ')" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$failed  = @()
$results = @()

foreach ($v in $Versions) {
    $config = "configs\$v.yaml"
    $ckpt   = "outputs\checkpoints\${v}_best.pth"

    if (-Not (Test-Path $config)) {
        Write-Host "[SKIP] $v -- config not found: $config" -ForegroundColor Yellow
        continue
    }
    if (-Not (Test-Path $ckpt)) {
        Write-Host "[WARN] $v -- best checkpoint not found, will try latest" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "---- $v ----------------------------------------" -ForegroundColor Cyan

    # Build argument list
    $args_list = @("src/evaluate_map.py", "--config", $config)

    if ($Map50Only)      { $args_list += "--map50-only" }
    if ($MaxTiles -gt 0) { $args_list += @("--max-tiles", $MaxTiles) }
    if ($OutputDir -ne "") { $args_list += @("--output-dir", $OutputDir) }

    $start = Get-Date
    python @args_list
    $exit_code = $LASTEXITCODE
    $elapsed   = ((Get-Date) - $start).TotalMinutes

    if ($exit_code -ne 0) {
        Write-Host "[FAIL] $v -- exit code $exit_code" -ForegroundColor Red
        $failed += $v
    } else {
        Write-Host "[OK]   $v -- completed in $([math]::Round($elapsed,1)) min" -ForegroundColor Green
        $results += $v
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  mAP Evaluation Complete" -ForegroundColor Cyan
Write-Host "  Passed : $($results -join ', ')" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "  Failed : $($failed -join ', ')" -ForegroundColor Red
}
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Regenerate final summary table and charts
Write-Host "Generating summary table..." -ForegroundColor Cyan
$summary_args = @("src/evaluate_map.py", "--config", "configs/v1.yaml", "--summary")
if ($OutputDir -ne "") { $summary_args += @("--output-dir", $OutputDir) }
python @summary_args

Write-Host ""
Write-Host "Results saved to: outputs\map_results\" -ForegroundColor Green
