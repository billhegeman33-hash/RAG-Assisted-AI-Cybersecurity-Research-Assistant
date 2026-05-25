#!/usr/bin/env python
"""
Simple launcher script for the Streamlit app.
Run this from the project root to start the app.
"""
import os
import sys
import subprocess

# Change to the directory containing this script
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Run streamlit
cmd = [sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"]
print(f"Running: {' '.join(cmd)}")
print(f"Working directory: {os.getcwd()}\n")

try:
    subprocess.run(cmd, check=False)
except KeyboardInterrupt:
    print("\n\nShutting down...")
    sys.exit(0)
