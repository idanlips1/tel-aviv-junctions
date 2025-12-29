#!/usr/bin/env python3
"""
Copy accident data CSV files from the ML-Applied project.
Finds all H*AccData.csv files and copies them to the data/ folder.
"""

import os
import shutil
from pathlib import Path

# Source and destination paths
SOURCE_DIR = Path("/Users/idanlipschitz/Projects/ML-Applied/Scooter-accident-prevention-machine-learning/Data")
DEST_DIR = Path(__file__).parent / "data"

def main():
    # Create destination directory
    DEST_DIR.mkdir(exist_ok=True)
    
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Destination directory: {DEST_DIR}")
    print()
    
    copied_files = []
    
    # Iterate through all subdirectories
    for folder in SOURCE_DIR.iterdir():
        if not folder.is_dir():
            continue
        
        # Look for H*AccData.csv files in each folder
        for file in folder.glob("H*AccData.csv"):
            dest_path = DEST_DIR / file.name
            
            print(f"Copying: {file.name}")
            shutil.copy2(file, dest_path)
            copied_files.append(file.name)
    
    print()
    print(f"Copied {len(copied_files)} files:")
    for f in sorted(copied_files):
        print(f"  - {f}")
    
    print(f"\nAll files saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()

