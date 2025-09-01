@echo off
echo Building Multi-Modal Tracking Application Executable...
echo.

REM Activate virtual environment
call venv_py310\Scripts\activate.bat

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del "*.spec"

REM Build executable
echo Building executable...
pyinstaller --onefile --windowed --name "MultiModalTracking" --add-data "config.py;." --add-data "README.md;." --hidden-import mediapipe --hidden-import cv2 --hidden-import numpy --hidden-import pygame enhanced_tracking_app.py

REM Check if build was successful
if exist "dist\MultiModalTracking.exe" (
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Executable created: dist\MultiModalTracking.exe
    echo.
    echo You can now run the application without Python installed!
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Check the error messages above.
    echo.
    pause
)
