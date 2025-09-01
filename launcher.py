#!/usr/bin/env python3
"""
Multi-Modal Tracking Application Launcher
This script serves as the entry point for the executable.
"""

import sys
import os
import traceback

def main():
    """Main launcher function"""
    try:
        # Add current directory to path for imports
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            application_path = os.path.dirname(sys.executable)
        else:
            # Running as script
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        sys.path.insert(0, application_path)
        
        # Import and run the main application
        from enhanced_tracking_app import main as run_app
        run_app()
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("This executable may be corrupted or missing dependencies.")
        input("Press Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        print("\nPlease report this error if it persists.")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
