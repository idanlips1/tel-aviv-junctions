#!/usr/bin/env python3
"""
Add accident labels to the junction panel dataset.

This script:
1. Loads the panel dataset (junction × year with features)
2. Loads Israeli CBS accident data (filtered to micromobility only)
3. Filters to accurate locations (STATUS_IGUN == 1) and Tel Aviv (SEMEL_YISHUV == 5000)
4. Converts accident coordinates from ITM to UTM Zone 36N (to match junction coordinates)
5. Assigns each accident to its nearest junction
6. Aggregates accident counts by junction-year
7. Adds 'accident_count' column to the panel

NOTE that It appears to be unique per year, 
so the same physical junction can have different cluster_id values across years.

Output: output/tel_aviv_junctions_panel_labeled.csv
"""

import glob
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


# Configuration
DATA_DIR = "/Volumes/Encrypted Extreme SSD/Projects/applied ML project/data"
PANEL_PATH = "/Volumes/Encrypted Extreme SSD/Projects/applied ML project/tel-aviv-junctions/tlv_junctions_history.csv"
OUTPUT_PATH = "/Volumes/Encrypted Extreme SSD/Projects/applied ML project/tel-aviv-junctions/output/tel_aviv_junctions_panel_labeled.csv"

# Coordinate systems
# Israeli CBS accident data uses ITM (Israeli Transverse Mercator)
ITM_CRS = "EPSG:2039"
# Junction data appears to be in UTM Zone 36N
UTM_CRS = "EPSG:32636"

# Micromobility vehicle types (SUG_REHEV_LMS codes)
# 15 = Electric bicycle (אופניים חשמליים)
# 21 = Electric scooter (קורקינט חשמלי)  
# 23 = Other micromobility (אחר - רכב תחבורה זעירה)
MICROMOBILITY_VEHICLE_TYPES = [15, 21, 23]

# Tel Aviv city code
TEL_AVIV_SEMEL = 5000

# Max distance (in meters) to assign an accident to a junction
# Accidents within this radius are assigned to the nearest junction
MAX_DISTANCE_METERS = 50.0

# History feature configuration
# Number of past years to include in history count
HISTORY_N_YEARS = 3
# Radius (in meters) for counting historical accidents near a junction
HISTORY_RADIUS_METERS = 50.0


def load_accident_files():
    """Find and load all accident CSV files."""
    accident_files = (
        glob.glob(f"{DATA_DIR}/H*/H*AccData.csv") + 
        glob.glob(f"{DATA_DIR}/H*/H*Accdata.csv") +
        glob.glob(f"{DATA_DIR}/puf_*/klali*.csv")  # 2024-2025 format
    )
    accident_files = sorted(set(accident_files))
    
    print(f"Found {len(accident_files)} accident files:")
    for f in accident_files:
        print(f"  - {Path(f).name}")
    
    dfs = []
    for f in accident_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total accident records: {len(combined)}")
    return combined


def load_vehicle_files():
    """Find and load all vehicle CSV files."""
    vehicle_files = (
        glob.glob(f"{DATA_DIR}/H*/H*VehData.csv") + 
        glob.glob(f"{DATA_DIR}/H*/H*Vehdata.csv") +
        glob.glob(f"{DATA_DIR}/puf_*/rehev*.csv")  # 2024-2025 format
    )
    vehicle_files = sorted(set(vehicle_files))
    
    print(f"Found {len(vehicle_files)} vehicle files:")
    for f in vehicle_files:
        print(f"  - {Path(f).name}")
    
    dfs = []
    for f in vehicle_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total vehicle records: {len(combined)}")
    return combined


def get_micromobility_accident_ids(vehicle_df: pd.DataFrame) -> set:
    """Get accident IDs that involve micromobility vehicles."""
    # Filter to micromobility vehicle types
    micro = vehicle_df[vehicle_df['SUG_REHEV_LMS'].isin(MICROMOBILITY_VEHICLE_TYPES)]
    
    # Get unique accident IDs
    accident_ids = set(micro['pk_teuna_fikt'].unique())
    
    print(f"\nMicromobility filtering:")
    print(f"  Vehicle types: {MICROMOBILITY_VEHICLE_TYPES}")
    print(f"  Matching vehicle records: {len(micro)}")
    print(f"  Unique accidents: {len(accident_ids)}")
    
    # Vehicle type breakdown
    type_names = {15: "Electric bicycle", 21: "Electric scooter", 23: "Other micromobility"}
    for vtype in MICROMOBILITY_VEHICLE_TYPES:
        count = (micro['SUG_REHEV_LMS'] == vtype).sum()
        print(f"    {vtype} ({type_names.get(vtype, 'Unknown')}): {count}")
    
    return accident_ids


def filter_accidents(accident_df: pd.DataFrame, micromobility_ids: set) -> pd.DataFrame:
    """Filter accidents to Tel Aviv, accurate location, micromobility."""
    df = accident_df.copy()
    print(f"\nFiltering accidents (starting with {len(df)} records):")
    
    # Filter to Tel Aviv
    if 'SEMEL_YISHUV' in df.columns:
        df = df[df['SEMEL_YISHUV'] == TEL_AVIV_SEMEL]
        print(f"  After Tel Aviv filter (SEMEL={TEL_AVIV_SEMEL}): {len(df)}")
    
    # Filter to accurate location
    if 'STATUS_IGUN' in df.columns:
        df = df[df['STATUS_IGUN'] == 1]
        print(f"  After accurate location filter (STATUS_IGUN=1): {len(df)}")
    
    # Filter to micromobility accidents
    df = df[df['pk_teuna_fikt'].isin(micromobility_ids)]
    print(f"  After micromobility filter: {len(df)}")
    
    # Filter to valid coordinates
    df = df[df['X'].notna() & df['Y'].notna() & (df['X'] != 0) & (df['Y'] != 0)]
    print(f"  After valid coordinates filter: {len(df)}")
    
    return df


def convert_itm_to_utm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert accident coordinates from ITM (EPSG:2039) to UTM Zone 36N (EPSG:32636).
    
    Israeli CBS data uses ITM where Tel Aviv is around X=180000, Y=660000.
    Junction data uses UTM 36N where Tel Aviv is around X=670000, Y=3550000.
    """
    print(f"\nConverting coordinates from ITM to UTM Zone 36N...")
    
    # Create transformer
    transformer = Transformer.from_crs(ITM_CRS, UTM_CRS, always_xy=True)
    
    # Sample before conversion
    print(f"  Sample ITM coords: X={df['X'].iloc[0]:.0f}, Y={df['Y'].iloc[0]:.0f}")
    
    # Transform coordinates
    x_utm, y_utm = transformer.transform(df['X'].values, df['Y'].values)
    
    # Add new columns
    df = df.copy()
    df['x_utm'] = x_utm
    df['y_utm'] = y_utm
    
    # Sample after conversion
    print(f"  Sample UTM coords: x_utm={df['x_utm'].iloc[0]:.0f}, y_utm={df['y_utm'].iloc[0]:.0f}")
    
    return df


def assign_accidents_to_junctions(
    accidents: pd.DataFrame,
    junctions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each accident to its nearest junction based on UTM coordinates.
    Matches accidents to junctions from the same year.
    
    Both accident and junction coordinates should be in UTM Zone 36N at this point.
    """
    print(f"\nAssigning {len(accidents)} accidents to junctions...")
    print(f"  Max distance: {MAX_DISTANCE_METERS}m")
    
    if len(accidents) == 0:
        return pd.DataFrame(columns=['pk_teuna_fikt', 'year', 'cluster_id', 'distance'])
    
    # Ensure year is int for consistent matching
    accidents['year'] = accidents['year'].astype(int)
    junctions['year'] = junctions['year'].astype(int)
    
    # Group by year and assign within each year
    results = []
    
    for year in sorted(accidents['year'].unique()):
        year_accidents = accidents[accidents['year'] == year]
        year_junctions = junctions[junctions['year'] == year]
        
        if len(year_junctions) == 0:
            print(f"  Warning: No junctions found for year {year}, skipping {len(year_accidents)} accidents")
            continue
        
        # Get unique junctions for this year
        unique_junctions = year_junctions.drop_duplicates('cluster_id')[['cluster_id', 'x_utm', 'y_utm']]
        
        # Build KD-tree from junction coordinates
        junction_coords = unique_junctions[['x_utm', 'y_utm']].values
        junction_ids = unique_junctions['cluster_id'].values
        tree = cKDTree(junction_coords)
        
        # Query nearest junction for each accident
        accident_coords = year_accidents[['x_utm', 'y_utm']].values
        distances, indices = tree.query(accident_coords, k=1)
        
        # Build result for this year
        year_result = pd.DataFrame({
            'pk_teuna_fikt': year_accidents['pk_teuna_fikt'].values,
            'year': year,
            'cluster_id': [junction_ids[i] for i in indices],
            'distance': distances,
        })
        
        # Filter by max distance
        before = len(year_result)
        year_result = year_result[year_result['distance'] <= MAX_DISTANCE_METERS]
        results.append(year_result)
        
        print(f"  Year {year}: {len(year_result)}/{len(year_accidents)} accidents assigned "
              f"(excluded {before - len(year_result)})")
    
    if not results:
        return pd.DataFrame(columns=['pk_teuna_fikt', 'year', 'cluster_id', 'distance'])
    
    result = pd.concat(results, ignore_index=True)
    
    # Overall stats
    print(f"  Total assigned: {len(result)}")
    if len(result) > 0:
        print(f"  Distance stats: mean={result['distance'].mean():.1f}m, "
              f"median={result['distance'].median():.1f}m, max={result['distance'].max():.1f}m")
    
    return result


def add_history_feature(
    panel_df: pd.DataFrame,
    accidents: pd.DataFrame,
    n_years: int = HISTORY_N_YEARS,
    radius_meters: float = HISTORY_RADIUS_METERS,
) -> pd.DataFrame:
    """
    Add historical accident count for each junction-year.
    
    For a junction in year Y, counts accidents within radius_meters
    in years Y-n to Y-1 (excluding the current year).
    
    Since the same physical junction may have different cluster_ids across years,
    we use spatial matching based on coordinates rather than cluster_id.
    
    The count is scaled (normalized) to account for incomplete history:
    - If n_years=3 but only 2 years of history are available, we scale by 3/2
    - This makes the feature comparable across years with different history availability
    
    NOTE: For future improvement, consider using weighted history where more
    recent years have higher weights (e.g., Y-1 weight=1.0, Y-2 weight=0.5, etc.)
    
    Args:
        panel_df: Panel DataFrame with cluster_id, year, x_utm, y_utm columns
        accidents: Filtered accidents DataFrame with x_utm, y_utm, year columns
        n_years: Number of past years to look back
        radius_meters: Radius for counting nearby accidents
    
    Returns:
        Panel DataFrame with added columns:
        - history_count: Raw count of accidents in past n years within radius
        - history_years: Number of years of history actually available
        - history_scaled: Count scaled to account for incomplete history
    """
    import numpy as np
    
    print(f"\nAdding history feature (n_years={n_years}, radius={radius_meters}m)...")
    
    result = panel_df.copy()
    result['history_count'] = 0
    result['history_years'] = 0
    result['history_scaled'] = 0.0
    
    years = sorted(panel_df['year'].unique())
    
    # Build KD-tree per year for fast spatial queries on accidents
    print("  Building spatial index for accidents by year...")
    accident_trees = {}
    for year in accidents['year'].unique():
        year_acc = accidents[accidents['year'] == year]
        if len(year_acc) > 0:
            coords = year_acc[['x_utm', 'y_utm']].values
            accident_trees[year] = cKDTree(coords)
    
    print(f"  Years with accident data: {sorted(accident_trees.keys())}")
    
    # Process each year
    for year in years:
        # Get junctions for this year
        mask = result['year'] == year
        junction_coords = result.loc[mask, ['x_utm', 'y_utm']].values
        n_junctions = len(junction_coords)
        
        # Determine which past years to include (Y-n to Y-1, excluding current year)
        past_years = [y for y in range(year - n_years, year) if y in accident_trees]
        years_available = len(past_years)
        
        if years_available == 0:
            # No history available (e.g., first year)
            print(f"  Year {year}: No history available (first year), setting to 0")
            continue
        
        # Count accidents in past years within radius for each junction
        history_counts = np.zeros(n_junctions)
        
        for past_year in past_years:
            tree = accident_trees[past_year]
            # query_ball_point returns list of indices for each query point
            nearby_lists = tree.query_ball_point(junction_coords, r=radius_meters)
            for i, nearby in enumerate(nearby_lists):
                history_counts[i] += len(nearby)
        
        # Scale to account for incomplete history
        # e.g., if n_years=3 but only 2 years available, scale by 3/2
        scale_factor = n_years / years_available
        history_scaled = history_counts * scale_factor
        
        # Update result
        result.loc[mask, 'history_count'] = history_counts.astype(int)
        result.loc[mask, 'history_years'] = years_available
        result.loc[mask, 'history_scaled'] = history_scaled
        
        total_history = int(history_counts.sum())
        junctions_with_history = int((history_counts > 0).sum())
        print(f"  Year {year}: {years_available}/{n_years} years available, "
              f"{total_history} total accidents, {junctions_with_history} junctions with history")
    
    # Summary
    print(f"\n  History feature summary:")
    print(f"    Total history_count: {result['history_count'].sum()}")
    print(f"    Junctions with history > 0: {(result['history_count'] > 0).sum()}")
    print(f"    Mean history_scaled: {result['history_scaled'].mean():.2f}")
    
    return result


def main():
    print("=" * 70)
    print("ADDING ACCIDENT LABELS TO JUNCTION PANEL")
    print("=" * 70)
    
    # Load panel dataset
    print(f"\n1. Loading panel: {PANEL_PATH}")
    panel_df = pd.read_csv(PANEL_PATH)
    print(f"   Shape: {panel_df.shape}")
    print(f"   Years: {sorted(panel_df['date'].unique())}")
    
    # Rename 'date' to 'year' for consistency
    panel_df = panel_df.rename(columns={'date': 'year'})
    panel_years = set(panel_df['year'].unique())
    
    # Load accident and vehicle data
    print("\n2. Loading CBS accident data...")
    accident_df = load_accident_files()
    
    print("\n3. Loading CBS vehicle data...")
    vehicle_df = load_vehicle_files()
    
    # Get micromobility accident IDs
    print("\n4. Identifying micromobility accidents...")
    micromobility_ids = get_micromobility_accident_ids(vehicle_df)
    
    # Filter accidents
    print("\n5. Filtering accidents...")
    filtered_accidents = filter_accidents(accident_df, micromobility_ids)
    
    # Convert coordinates from ITM to UTM
    print("\n6. Converting coordinates...")
    filtered_accidents = convert_itm_to_utm(filtered_accidents)
    
    # Add 'year' column from SHNAT_TEUNA for consistency
    filtered_accidents['year'] = filtered_accidents['SHNAT_TEUNA'].astype(int)
    
    # Assign to junctions
    print("\n7. Assigning accidents to junctions...")
    assigned = assign_accidents_to_junctions(filtered_accidents, panel_df)
    
    # Filter to panel years
    assigned = assigned[assigned['year'].isin(panel_years)]
    print(f"   After filtering to panel years: {len(assigned)}")
    
    # Aggregate by junction-year
    print("\n8. Aggregating by junction-year...")
    # Ensure year is int for consistent merging
    assigned['year'] = assigned['year'].astype(int)
    accident_counts = assigned.groupby(['cluster_id', 'year']).size().reset_index(name='accident_count')
    accident_counts['year'] = accident_counts['year'].astype(int)
    print(f"   Junction-years with accidents: {len(accident_counts)}")
    print(f"   Total accidents: {accident_counts['accident_count'].sum()}")
    
    # Debug: Check for mismatches
    print("\n   Debugging merge keys...")
    # Ensure panel year is also int
    panel_df['year'] = panel_df['year'].astype(int)
    panel_cluster_years = set(zip(panel_df['cluster_id'].astype(str), panel_df['year']))
    accident_cluster_years = set(zip(accident_counts['cluster_id'].astype(str), accident_counts['year']))
    missing_in_panel = accident_cluster_years - panel_cluster_years
    if missing_in_panel:
        print(f"   WARNING: {len(missing_in_panel)} accident cluster_id/year combos not in panel!")
        print(f"   Sample missing: {list(missing_in_panel)[:5]}")
        print(f"   These accidents will be lost in the merge!")
    
    # Merge with panel
    print("\n9. Merging with panel...")
    result = panel_df.merge(accident_counts, on=['cluster_id', 'year'], how='left')
    result['accident_count'] = result['accident_count'].fillna(0).astype(int)
    
    # Debug: Check merge result
    matched = result['accident_count'].notna() & (result['accident_count'] > 0)
    print(f"   Rows with accidents after merge: {matched.sum()}")
    print(f"   Total accidents in merged result: {result['accident_count'].sum()}")
    
    # Add history feature
    print("\n10. Adding history feature...")
    result = add_history_feature(
        result,
        filtered_accidents,
        n_years=HISTORY_N_YEARS,
        radius_meters=HISTORY_RADIUS_METERS,
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final panel shape: {result.shape}")
    print(f"Total micromobility accidents (current year): {result['accident_count'].sum()}")
    print(f"Junction-years with >= 1 accident: {(result['accident_count'] > 0).sum()}")
    print(f"Junction-years with 0 accidents: {(result['accident_count'] == 0).sum()}")
    print(f"\nHistory feature (past {HISTORY_N_YEARS} years, {HISTORY_RADIUS_METERS}m radius):")
    print(f"  Total history_count: {result['history_count'].sum()}")
    print(f"  Junction-years with history > 0: {(result['history_count'] > 0).sum()}")
    print(f"  Mean history_scaled: {result['history_scaled'].mean():.2f}")
    
    print("\nAccidents by year:")
    by_year = result.groupby('year')['accident_count'].sum()
    for year, count in by_year.items():
        print(f"  {year}: {count}")
    
    print("\nTop 10 junction-years by accidents:")
    top = result.nlargest(10, 'accident_count')[['cluster_id', 'year', 'x_utm', 'y_utm', 'accident_count', 'history_count', 'history_scaled']]
    print(top.to_string(index=False))
    
    # Reorder columns: features first, label (accident_count) last
    cols = [c for c in result.columns if c != 'accident_count']
    cols.append('accident_count')
    result = result[cols]
    
    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {OUTPUT_PATH}")
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows")
    
    print("\nDone!")
    return result


if __name__ == "__main__":
    main()
