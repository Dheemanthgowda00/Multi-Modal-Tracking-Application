# Building Standalone Executable

This guide will help you create a completely standalone `.exe` file that includes all dependencies and can run on any Windows machine without Python installation.

## 🚀 Quick Build

### Option 1: PowerShell Script (Recommended)
```powershell
# Right-click on build_final.ps1 and select "Run with PowerShell"
# OR run from PowerShell:
.\build_final.ps1
```

### Option 2: Batch File
```cmd
# Double-click build_exe.bat
# OR run from Command Prompt:
build_exe.bat
```

### Option 3: Manual PyInstaller Command
```powershell
# Activate virtual environment
.\venv_py310\Scripts\Activate.ps1

# Build executable
pyinstaller --onefile --name "MultiModalTracking" --add-data "config.py;." --add-data "README.md;." --hidden-import mediapipe --hidden-import cv2 --hidden-import numpy --hidden-import pygame --collect-all mediapipe --collect-all cv2 --collect-all numpy --collect-all pygame launcher.py
```

## 📋 Prerequisites

- ✅ Python 3.10 virtual environment activated
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ PyInstaller installed (`pip install pyinstaller`)
- ✅ At least 2GB free disk space
- ✅ Windows 10/11

## ⏱️ Build Time

- **First build**: 5-10 minutes
- **Subsequent builds**: 3-5 minutes
- **File size**: Approximately 200-400 MB

## 📁 Output Files

After successful build, you'll find:
- `dist\MultiModalTracking.exe` - Main executable
- `.\MultiModalTracking.exe` - Copy in current directory
- `build\` - Build cache (can be deleted)
- `*.spec` - PyInstaller spec files (can be deleted)

## 🎯 What You Get

### Standalone Application Features:
- ✅ **No Python Required** - Runs on any Windows machine
- ✅ **No Dependencies** - Everything bundled inside
- ✅ **Portable** - Can be moved to any folder
- ✅ **Distributable** - Share with others easily
- ✅ **All Tracking Modes** - Body, hands, eyes, head tracking
- ✅ **Configuration Support** - Uses config.py settings

### Application Controls:
- **B** - Toggle body tracking
- **H** - Toggle hand tracking  
- **E** - Toggle eye tracking
- **T** - Toggle head tracking
- **S** - Show tracking summary
- **C** - Save tracking data
- **Q** - Quit application

## 🔧 Troubleshooting

### Build Fails:
1. **Check virtual environment**: Ensure `venv_py310` is activated
2. **Verify dependencies**: Run `pip list` to see installed packages
3. **Disk space**: Ensure at least 2GB free space
4. **Permissions**: Try running as administrator
5. **Antivirus**: Temporarily disable if blocking PyInstaller

### Executable Won't Run:
1. **Camera permissions**: Grant camera access when prompted
2. **Windows Defender**: May flag as suspicious (safe to allow)
3. **Missing DLLs**: Ensure Visual C++ Redistributable is installed
4. **Antivirus**: Add to exclusions if blocked

### Performance Issues:
1. **Reduce resolution** in `config.py`
2. **Disable unused modes** (eyes, head tracking)
3. **Close other applications** to free up resources

## 📦 Distribution

### Single File Distribution:
- Copy `MultiModalTracking.exe` to any Windows machine
- No installation required
- No dependencies needed
- Works immediately

### Package Contents:
- `MultiModalTracking.exe` - Main application
- `README.md` - User documentation
- `config.py` - Configuration file (optional)

## 🎉 Success!

Once built successfully, you have a professional-grade standalone application that:
- Works on any Windows 10/11 machine
- Requires no technical knowledge to run
- Includes all advanced tracking features
- Is ready for distribution to end users

**Happy Tracking! 🚀**
