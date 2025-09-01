#!/usr/bin/env python3
"""
Demo script for Multi-Modal Tracking Application
This script tests the basic installation and provides a simple interface.
"""

import cv2
import sys
import time

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🔍 Testing dependencies...")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe installed successfully")
    except ImportError:
        print("❌ MediaPipe not found. Install with: pip install mediapipe")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy installed successfully")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        import pygame
        print("✅ Pygame installed successfully")
    except ImportError:
        print("❌ Pygame not found. Install with: pip install pygame")
        return False
    
    print("✅ All dependencies are available!")
    return True

def test_camera():
    """Test if camera is accessible"""
    print("\n📹 Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible. Check camera connection and permissions.")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read from camera.")
        cap.release()
        return False
    
    print(f"✅ Camera accessible. Frame size: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True

def show_demo():
    """Show a simple demo of camera feed"""
    print("\n🎥 Starting camera demo (press 'q' to quit)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    start_time = time.time()
    frame_count = 0
    fps = 0.0  # Initialize fps variable
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Add info overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Demo Mode - Basic Camera Feed", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Camera Demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main demo function"""
    print("🚀 Multi-Modal Tracking Application - Demo")
    print("=" * 50)
    
    # Test dependencies
    if not test_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test camera
    if not test_camera():
        print("\n❌ Camera test failed. Please check your camera setup.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Your system is ready for tracking.")
    print("\nNext steps:")
    print("1. Run 'python tracking_app.py' for basic tracking")
    print("2. Run 'python enhanced_tracking_app.py' for advanced features")
    print("3. Edit 'config.py' to customize settings")
    
    # Ask if user wants to see demo
    try:
        response = input("\nWould you like to see a camera demo? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            show_demo()
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    
    print("\n✨ Demo completed! Happy tracking!")

if __name__ == "__main__":
    main()
