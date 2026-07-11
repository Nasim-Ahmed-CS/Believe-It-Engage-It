#!/usr/bin/env pwsh
# PowerShell script to train all engagement classification models
# Run with: .\train_all_models.ps1 or right-click -> Run with PowerShell

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "ENGAGEMENT CLASSIFICATION MODEL TRAINING SUITE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will train all 6 engagement classification models sequentially."
Write-Host "Expected total time: 15-30 minutes depending on hardware."
Write-Host ""
Write-Host "Models to be trained:" -ForegroundColor Yellow
Write-Host "  1. LSTM"
Write-Host "  2. GRU"
Write-Host "  3. XGBoost"
Write-Host "  4. Random Forest"
Write-Host "  5. Decision Tree"
Write-Host "  6. EBM (Explainable Boosting Machine)"
Write-Host ""
Write-Host "Trained models will be saved in: models\saved_models\" -ForegroundColor Green
Write-Host ""

$continue = Read-Host "Press Enter to continue or Ctrl+C to cancel"

# Change to script directory
Set-Location $PSScriptRoot
Write-Host ""
Write-Host "Current directory: $PWD" -ForegroundColor Gray
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Create saved_models directory if it doesn't exist
if (-not (Test-Path "saved_models")) {
    New-Item -ItemType Directory -Path "saved_models" | Out-Null
}

# Track results
$results = @()
$startTime = Get-Date

# Define models
$models = @(
    @{Name="LSTM"; Script="train_lstm_optimized.py"},
    @{Name="GRU"; Script="train_gru_optimized.py"},
    @{Name="XGBoost"; Script="train_xgboost_optimized.py"},
    @{Name="Random Forest"; Script="train_randomforest_optimized.py"},
    @{Name="Decision Tree"; Script="train_decisiontree_optimized.py"},
    @{Name="EBM"; Script="train_ebm_optimized.py"}
)

$modelCount = $models.Count
$currentModel = 0

# Train each model
foreach ($model in $models) {
    $currentModel++
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "[$currentModel/$modelCount] Training $($model.Name) Model..." -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    
    $modelStartTime = Get-Date
    
    try {
        python $model.Script
        $exitCode = $LASTEXITCODE
        
        $modelEndTime = Get-Date
        $duration = ($modelEndTime - $modelStartTime).TotalSeconds
        
        if ($exitCode -eq 0 -or $null -eq $exitCode) {
            Write-Host ""
            Write-Host "✓ SUCCESS: $($model.Name) model trained in $([math]::Round($duration, 1))s" -ForegroundColor Green
            $results += [PSCustomObject]@{
                Model = $model.Name
                Status = "SUCCESS"
                Time = "$([math]::Round($duration, 1))s"
            }
        } else {
            Write-Host ""
            Write-Host "✗ WARNING: $($model.Name) training failed after $([math]::Round($duration, 1))s" -ForegroundColor Yellow
            $results += [PSCustomObject]@{
                Model = $model.Name
                Status = "FAILED"
                Time = "$([math]::Round($duration, 1))s"
            }
        }
    } catch {
        $modelEndTime = Get-Date
        $duration = ($modelEndTime - $modelStartTime).TotalSeconds
        Write-Host ""
        Write-Host "✗ ERROR: $($model.Name) training encountered an error: $_" -ForegroundColor Red
        $results += [PSCustomObject]@{
            Model = $model.Name
            Status = "ERROR"
            Time = "$([math]::Round($duration, 1))s"
        }
    }
    
    Write-Host ""
}

$endTime = Get-Date
$totalDuration = ($endTime - $startTime).TotalMinutes

# Print summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "TRAINING SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$results | Format-Table -AutoSize

Write-Host "Total training time: $([math]::Round($totalDuration, 1)) minutes" -ForegroundColor Yellow
Write-Host ""

$successful = ($results | Where-Object { $_.Status -eq "SUCCESS" }).Count
$failed = ($results | Where-Object { $_.Status -ne "SUCCESS" }).Count

Write-Host "Successful: $successful/$modelCount" -ForegroundColor Green
if ($failed -gt 0) {
    Write-Host "Failed: $failed/$modelCount" -ForegroundColor Yellow
    $failedModels = ($results | Where-Object { $_.Status -ne "SUCCESS" }).Model -join ", "
    Write-Host "Failed models: $failedModels" -ForegroundColor Yellow
}

Write-Host ""

# List saved models
if (Test-Path "saved_models") {
    $savedFiles = Get-ChildItem "saved_models" -File
    Write-Host "Saved models directory: $PWD\saved_models" -ForegroundColor Green
    Write-Host "Total saved model files: $($savedFiles.Count)" -ForegroundColor Green
    foreach ($file in $savedFiles | Sort-Object Name) {
        $sizeMB = [math]::Round($file.Length / 1MB, 1)
        Write-Host "  - $($file.Name) ($sizeMB MB)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "ALL TRAINING COMPLETE!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
