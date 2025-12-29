#!/usr/bin/env python3
"""
Generate the panel dataset with junctions and temporal features.
Run this script to create the ML-ready dataset.
"""

from tel_aviv_junctions.panel import build_full_pipeline
from tel_aviv_junctions.config import OUTPUT_DIR

if __name__ == "__main__":
    print("Generating Tel Aviv Junctions Panel Dataset...")
    print("This will:")
    print("  1. Extract junctions from OSM (current data)")
    print("  2. Query ohsome API for historical features (2015-2024)")
    print("  3. Build panel dataset (junction × year)")
    print()
    
    # Build the dataset without accident labels
    panel_df = build_full_pipeline(
        accidents_csv=None,  # No accident labels yet
        use_cache=True,      # Cache ohsome API results
    )
    
    print("\n" + "=" * 60)
    print("DATASET GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}/tel_aviv_junctions_panel.csv")
    print(f"\nDataset shape: {panel_df.shape}")
    print(f"Columns: {list(panel_df.columns)}")

