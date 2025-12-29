#!/usr/bin/env python3
"""
Add accident labels to the panel dataset.

This script:
1. Loads the panel dataset (junction × year with features)
2. Loads Israeli CBS accident data (filtered to micromobility only)
3. Filters to accurate locations (STATUS_IGUN == 1) and Tel Aviv
4. Assigns each accident to its nearest junction
5. Aggregates accident counts by junction-year
6. Adds 'accident_count' column to the panel

Output: output/tel_aviv_junctions_panel_labeled.csv
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from tel_aviv_junctions.extraction import get_junctions_geodataframe
from tel_aviv_junctions.spatial import (
    load_micromobility_accidents,
    accident_location_to_junction,
)


def main():
    # Paths
    panel_path = "output/tel_aviv_junctions_panel.csv"
    output_path = "output/tel_aviv_junctions_panel_labeled.csv"
    
    # Collect all accident and vehicle files (multiple naming conventions)
    import glob
    
    # Accident files: H*AccData.csv (various cases) + klali-*.csv (2024-2025 format)
    accident_files = (
        glob.glob("data/H*AccData.csv") + 
        glob.glob("data/H*Accdata.csv") +
        glob.glob("data/klali-*.csv")  # 2024-2025 format
    )
    accident_files = sorted(set(accident_files))
    
    # Vehicle files: H*VehData.csv (various cases) + rehev-*.csv (2024-2025 format)
    vehicle_files = (
        glob.glob("data/H*VehData.csv") + 
        glob.glob("data/H*Vehdata.csv") +
        glob.glob("data/rehev-*.csv")  # 2024-2025 format
    )
    vehicle_files = sorted(set(vehicle_files))
    
    print("=" * 70)
    print("DATA FILES INVENTORY")
    print("=" * 70)
    print(f"\nAccident files ({len(accident_files)}):")
    for f in accident_files:
        print(f"  - {f}")
    print(f"\nVehicle files ({len(vehicle_files)}):")
    for f in vehicle_files:
        print(f"  - {f}")
    
    # Load junctions GeoDataFrame (needed for spatial join)
    print("Loading junctions...")
    junctions_gdf = get_junctions_geodataframe()
    print(f"Loaded {len(junctions_gdf)} junctions\n")
    
    # Load micromobility accidents only
    accidents = load_micromobility_accidents(
        accident_csv_paths=accident_files,
        vehicle_csv_paths=vehicle_files,
        filter_tel_aviv=True,
        filter_accurate_location=True,
        vehicle_types=[15, 21, 23],  # e-bikes, e-scooters, other micromobility
    )
    
    # Assign to junctions
    print("\nAssigning accidents to junctions...")
    accidents_with_junctions = accident_location_to_junction(
        accidents,
        junctions_gdf,
        max_distance_meters=50.0,
    )
    
    # Load panel
    print(f"\nLoading panel dataset: {panel_path}")
    panel_df = pd.read_csv(panel_path)
    print(f"Panel shape: {panel_df.shape}")
    panel_years = set(panel_df['year'].unique())
    
    # Filter to panel years
    accidents_filtered = accidents_with_junctions[
        accidents_with_junctions['year'].isin(panel_years)
    ].copy()
    print(f"Filtered to panel years: {len(accidents_filtered)} accidents")
    
    # Aggregate by junction and year
    print("\nAggregating accidents by junction-year...")
    accident_counts = accidents_filtered.groupby(
        ['junction_id', 'year']
    ).size().reset_index(name='accident_count')
    
    print(f"Junction-years with accidents: {len(accident_counts)}")
    print(f"Total accidents: {accident_counts['accident_count'].sum()}")
    
    # Merge with panel
    print("\nMerging with panel dataset...")
    result = panel_df.merge(
        accident_counts,
        on=['junction_id', 'year'],
        how='left'
    )
    result['accident_count'] = result['accident_count'].fillna(0).astype(int)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final panel shape: {result.shape}")
    print(f"Total micromobility accidents: {result['accident_count'].sum()}")
    print(f"Junction-years with >= 1 accident: {(result['accident_count'] > 0).sum()}")
    print(f"Junction-years with 0 accidents: {(result['accident_count'] == 0).sum()}")
    
    print("\nMicromobility accidents by year:")
    by_year = result.groupby('year')['accident_count'].sum()
    for year, count in by_year.items():
        print(f"  {year}: {count}")
    
    print("\nTop 10 junction-years by micromobility accidents:")
    top = result.nlargest(10, 'accident_count')[['junction_id', 'year', 'latitude', 'longitude', 'accident_count']]
    print(top.to_string(index=False))
    
    # Save
    print(f"\nSaving to: {output_path}")
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows")
    
    print("\nDone!")
    return result


if __name__ == "__main__":
    main()

