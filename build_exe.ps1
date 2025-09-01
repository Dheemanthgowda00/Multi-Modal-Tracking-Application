# Build Multi-Modal Tracking Application Executable
# PowerShell Script

Write-Host "Building Multi-Modal Tracking Application Executable..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv_py310\Scripts\Activate.ps1"

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "*.spec") { Remove-Item -Force "*.spec" }

# Build executable
Write-Host "Building executable with PyInstaller..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan

try {
    pyinstaller --onefile `
                --windowed `
                --name "MultiModalTracking" `
                --add-data "config.py;." `
                --add-data "README.md;." `
                --hidden-import mediapipe `
                --hidden-import cv2 `
                --hidden-import numpy `
                --hidden-import pygame `
                --hidden-import json `
                --hidden-import typing `
                --hidden-import time `
                --hidden-import sys `
                --hidden-import os `
                enhanced_tracking_app.py

    # Check if build was successful
    if (Test-Path "dist\MultiModalTracking.exe") {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Executable created: dist\MultiModalTracking.exe" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run the application without Python installed!" -ForegroundColor Green
        Write-Host ""
        
        # Show file size
        $fileSize = (Get-Item "dist\MultiModalTracking.exe").Length
        $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
        Write-Host "File size: $fileSizeMB MB" -ForegroundColor Cyan
        
        # Copy to current directory for easy access
        Copy-Item "dist\MultiModalTracking.exe" ".\MultiModalTracking.exe" -Force
        Write-Host "Copied to: .\MultiModalTracking.exe" -ForegroundColor Cyan
    } else {
        throw "Executable not found after build"
    }
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "BUILD FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above." -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
