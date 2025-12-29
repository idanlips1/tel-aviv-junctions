#!/usr/bin/env python3
"""
Copy vehicle data files from the ML-Applied project to the data folder.

Copies all H**Vehdata.csv files from:
/Users/idanlipschitz/Projects/ML-Applied/Scooter-accident-prevention-machine-learning/Data/H**/

to: data/
"""

import shutil
from pathlib import Path

# Source directory
SOURCE_DIR = Path("/Users/idanlipschitz/Projects/ML-Applied/Scooter-accident-prevention-machine-learning/Data")

# Destination directory
DEST_DIR = Path("data")

def main():
    # Create destination directory if it doesn't exist
    DEST_DIR.mkdir(exist_ok=True)
    
    # Find all H**Vehdata.csv and H**VehData.csv files recursively (case varies by year)
    patterns = ["H*Vehdata.csv", "H*VehData.csv"]
    files = []
    for pattern in patterns:
        files.extend(SOURCE_DIR.rglob(pattern))
    
    # Remove duplicates (in case both patterns match same file on case-insensitive fs)
    files = list({f.resolve(): f for f in files}.values())
    
    print(f"Found {len(files)} vehicle data files")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR.absolute()}")
    print("-" * 60)
    
    copied = 0
    for src_file in sorted(files):
        dest_file = DEST_DIR / src_file.name
        print(f"Copying: {src_file.name}")
        shutil.copy2(src_file, dest_file)
        copied += 1
    
    print("-" * 60)
    print(f"Copied {copied} files to {DEST_DIR}/")
    
    # List copied files
    print("\nCopied files:")
    veh_files = list(DEST_DIR.glob("H*Vehdata.csv")) + list(DEST_DIR.glob("H*VehData.csv"))
    for f in sorted(set(veh_files), key=lambda x: x.name):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

