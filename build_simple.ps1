# Simple Build Script for Multi-Modal Tracking Application

Write-Host "Building Multi-Modal Tracking Application..." -ForegroundColor Green

# Activate virtual environment
& ".\venv_py310\Scripts\Activate.ps1"

# Clean previous builds
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# Build executable
Write-Host "Building executable (this will take 5-10 minutes)..." -ForegroundColor Yellow

pyinstaller --onefile --name "MultiModalTracking" --add-data "config.py;." --add-data "README.md;." --hidden-import mediapipe --hidden-import cv2 --hidden-import numpy --hidden-import pygame --collect-all mediapipe --collect-all cv2 --collect-all numpy --collect-all pygame launcher.py

# Check if build was successful
if (Test-Path "dist\MultiModalTracking.exe") {
    Write-Host ""
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Executable: dist\MultiModalTracking.exe" -ForegroundColor Green
    
    # Copy to current directory
    Copy-Item "dist\MultiModalTracking.exe" ".\MultiModalTracking.exe" -Force
    Write-Host "Copied to: .\MultiModalTracking.exe" -ForegroundColor Green
} else {
    Write-Host "BUILD FAILED!" -ForegroundColor Red
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
