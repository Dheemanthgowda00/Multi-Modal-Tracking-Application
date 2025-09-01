# Final Build Script for Multi-Modal Tracking Application
# This creates a completely standalone executable

Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-Modal Tracking Application Builder" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating Python 3.10 virtual environment..." -ForegroundColor Yellow
& ".\venv_py310\Scripts\Activate.ps1"

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "*.spec") { Remove-Item -Force "*.spec" }

# Build executable
Write-Host "Building standalone executable..." -ForegroundColor Yellow
Write-Host "This will take 5-10 minutes depending on your system..." -ForegroundColor Cyan
Write-Host ""

try {
    pyinstaller --onefile `
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
                --hidden-import mediapipe.solutions.pose `
                --hidden-import mediapipe.solutions.hands `
                --hidden-import mediapipe.solutions.face_mesh `
                --hidden-import mediapipe.solutions.drawing_utils `
                --hidden-import mediapipe.solutions.drawing_styles `
                --collect-all mediapipe `
                --collect-all cv2 `
                --collect-all numpy `
                --collect-all pygame `
                launcher.py

    # Check if build was successful
    if (Test-Path "dist\MultiModalTracking.exe") {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        
        # Show file size
        $fileSize = (Get-Item "dist\MultiModalTracking.exe").Length
        $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
        Write-Host "Executable created: dist\MultiModalTracking.exe" -ForegroundColor Green
        Write-Host "File size: $fileSizeMB MB" -ForegroundColor Cyan
        Write-Host ""
        
        # Copy to current directory for easy access
        Copy-Item "dist\MultiModalTracking.exe" ".\MultiModalTracking.exe" -Force
        Write-Host "Copied to: .\MultiModalTracking.exe" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "YOUR STANDALONE APP IS READY!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Features:" -ForegroundColor Yellow
        Write-Host "✓ No Python installation required" -ForegroundColor Green
        Write-Host "✓ No dependencies to install" -ForegroundColor Green
        Write-Host "✓ Works on any Windows 10/11 machine" -ForegroundColor Green
        Write-Host "✓ Includes all tracking modes" -ForegroundColor Green
        Write-Host "✓ Ready to distribute" -ForegroundColor Green
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "1. Double-click MultiModalTracking.exe" -ForegroundColor White
        Write-Host "2. Grant camera permissions when prompted" -ForegroundColor White
        Write-Host "3. Use keyboard controls to toggle tracking modes" -ForegroundColor White
        Write-Host ""
        Write-Host "Controls:" -ForegroundColor Yellow
        Write-Host "B - Toggle body tracking" -ForegroundColor White
        Write-Host "H - Toggle hand tracking" -ForegroundColor White
        Write-Host "E - Toggle eye tracking" -ForegroundColor White
        Write-Host "T - Toggle head tracking" -ForegroundColor White
        Write-Host "S - Show tracking summary" -ForegroundColor White
        Write-Host "C - Save tracking data" -ForegroundColor White
        Write-Host "Q - Quit application" -ForegroundColor White
        
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
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Ensure all dependencies are installed in the virtual environment" -ForegroundColor White
    Write-Host "2. Check that you have sufficient disk space (at least 2GB free)" -ForegroundColor White
    Write-Host "3. Try running as administrator if permission issues occur" -ForegroundColor White
    Write-Host "4. Check the error messages above for specific issues" -ForegroundColor White
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
