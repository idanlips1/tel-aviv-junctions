"""
Spatial operations for joining accidents to junctions.

This module handles:
- Loading accident data from CSV with coordinates
- Converting Israeli accident data from ITM to WGS84
- Spatial joining accidents to nearest junction
- Temporal filtering to assign accidents to the correct year
- Aggregating accident counts per junction per year
"""

from typing import Optional, Tuple, List, Union
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from tel_aviv_junctions.config import (
    UTM_CRS,
    WGS84_CRS,
    DEFAULT_ACCIDENT_RADIUS_METERS,
    YEARS,
    TEL_AVIV_BBOX,
)

# Israeli Transverse Mercator (ITM) - used by Israeli CBS accident data
ITM_CRS = "EPSG:2039"

# Tel Aviv city code in Israeli CBS data
TEL_AVIV_SEMEL = 5000

# Micromobility vehicle types (SUG_REHEV_LMS codes)
# 15 = Electric bicycle (אופניים חשמליים)
# 21 = Electric scooter (קורקינט חשמלי)  
# 23 = Other micromobility (אחר - רכב תחבורה זעירה)
MICROMOBILITY_VEHICLE_TYPES = [15, 21, 23]


def load_vehicle_data(
    csv_paths: Union[str, List[str]],
) -> pd.DataFrame:
    """
    Load Israeli CBS vehicle data from CSV files.
    
    The vehicle data contains SUG_REHEV_LMS (vehicle type) which is needed
    to filter accidents by vehicle type (e.g., micromobility).
    
    Args:
        csv_paths: Path to CSV file, glob pattern, or list of paths
    
    Returns:
        DataFrame with vehicle records
    """
    # Resolve paths
    if isinstance(csv_paths, str):
        if '*' in csv_paths:
            files = glob.glob(csv_paths)
        else:
            files = [csv_paths]
    else:
        files = csv_paths
    
    if not files:
        raise ValueError(f"No files found matching: {csv_paths}")
    
    print(f"Loading vehicle data from {len(files)} file(s)...")
    
    dfs = []
    for f in files:
        print(f"  Reading: {Path(f).name}")
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total vehicle records: {len(combined_df)}")
    
    return combined_df


def get_micromobility_accident_ids(
    vehicle_csv_paths: Union[str, List[str]],
    vehicle_types: List[int] = None,
) -> set:
    """
    Get accident IDs that involve micromobility vehicles.
    
    Args:
        vehicle_csv_paths: Path(s) to vehicle data CSV files
        vehicle_types: List of SUG_REHEV_LMS codes to include.
                       Defaults to MICROMOBILITY_VEHICLE_TYPES [15, 21, 23]
    
    Returns:
        Set of accident IDs (pk_teuna_fikt) involving specified vehicle types
    """
    if vehicle_types is None:
        vehicle_types = MICROMOBILITY_VEHICLE_TYPES
    
    # Load vehicle data
    veh_df = load_vehicle_data(vehicle_csv_paths)
    
    # Check for required columns
    if 'SUG_REHEV_LMS' not in veh_df.columns:
        raise ValueError("Vehicle data missing SUG_REHEV_LMS column")
    if 'pk_teuna_fikt' not in veh_df.columns:
        raise ValueError("Vehicle data missing pk_teuna_fikt column")
    
    # Filter to micromobility vehicles
    micro_vehicles = veh_df[veh_df['SUG_REHEV_LMS'].isin(vehicle_types)]
    
    # Get unique accident IDs
    accident_ids = set(micro_vehicles['pk_teuna_fikt'].unique())
    
    print(f"  Filtered to vehicle types {vehicle_types}: {len(micro_vehicles)} vehicle records")
    print(f"  Unique accidents involving these vehicles: {len(accident_ids)}")
    
    # Vehicle type breakdown
    type_counts = micro_vehicles['SUG_REHEV_LMS'].value_counts().sort_index()
    print(f"  Vehicle type breakdown:")
    type_names = {15: "Electric bicycle", 21: "Electric scooter", 23: "Other micromobility"}
    for vtype, count in type_counts.items():
        name = type_names.get(vtype, f"Type {vtype}")
        print(f"    {vtype} ({name}): {count} vehicles")
    
    return accident_ids


def load_israeli_accident_data(
    csv_paths: Union[str, List[str]],
    filter_tel_aviv: bool = True,
    filter_bbox: bool = True,
    filter_accurate_location: bool = True,
    accident_ids: Optional[set] = None,
) -> gpd.GeoDataFrame:
    """
    Load Israeli CBS accident data from CSV files and convert to GeoDataFrame.
    
    The Israeli accident data uses:
    - X, Y columns in Israeli Transverse Mercator (ITM/EPSG:2039)
    - SHNAT_TEUNA for year
    - SEMEL_YISHUV for city code (5000 = Tel Aviv)
    - STATUS_IGUN for location accuracy (1 = accurate)
    
    Args:
        csv_paths: Path to CSV file, glob pattern (e.g., "data/*.csv"), 
                   or list of paths
        filter_tel_aviv: If True, filter to Tel Aviv accidents (SEMEL_YISHUV=5000)
        filter_bbox: If True, also filter by Tel Aviv bounding box
        filter_accurate_location: If True, filter to STATUS_IGUN == 1 (accurate location)
        accident_ids: Optional set of accident IDs to include (e.g., from micromobility filter)
    
    Returns:
        GeoDataFrame with accident points in WGS84, including 'year' column
    """
    # Resolve paths
    if isinstance(csv_paths, str):
        if '*' in csv_paths:
            files = glob.glob(csv_paths)
        else:
            files = [csv_paths]
    else:
        files = csv_paths
    
    if not files:
        raise ValueError(f"No files found matching: {csv_paths}")
    
    print(f"Loading Israeli accident data from {len(files)} file(s)...")
    
    # Load and concatenate all CSVs
    dfs = []
    for f in files:
        print(f"  Reading: {Path(f).name}")
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total records loaded: {len(combined_df)}")
    
    # Check for required columns
    required_cols = ['X', 'Y', 'SHNAT_TEUNA']
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter by city code if requested
    if filter_tel_aviv and 'SEMEL_YISHUV' in combined_df.columns:
        before = len(combined_df)
        combined_df = combined_df[combined_df['SEMEL_YISHUV'] == TEL_AVIV_SEMEL].copy()
        print(f"  Filtered to Tel Aviv (SEMEL={TEL_AVIV_SEMEL}): {len(combined_df)} records (from {before})")
    
    # Filter to accurate location only (STATUS_IGUN == 1)
    if filter_accurate_location and 'STATUS_IGUN' in combined_df.columns:
        before = len(combined_df)
        combined_df = combined_df[combined_df['STATUS_IGUN'] == 1].copy()
        print(f"  Filtered to accurate location (STATUS_IGUN=1): {len(combined_df)} records (from {before})")
    
    # Filter by accident IDs (e.g., from micromobility filter)
    if accident_ids is not None:
        before = len(combined_df)
        combined_df = combined_df[combined_df['pk_teuna_fikt'].isin(accident_ids)].copy()
        print(f"  Filtered to specified accident IDs: {len(combined_df)} records (from {before})")
    
    # Drop records with missing coordinates
    valid_coords = combined_df[['X', 'Y']].notna().all(axis=1) & (combined_df['X'] != 0) & (combined_df['Y'] != 0)
    combined_df = combined_df[valid_coords].copy()
    print(f"  Records with valid coordinates: {len(combined_df)}")
    
    if len(combined_df) == 0:
        print("  Warning: No valid records after filtering")
        return gpd.GeoDataFrame(
            columns=['geometry', 'year'],
            crs=WGS84_CRS
        )
    
    # Create geometry from ITM coordinates
    geometry = [Point(x, y) for x, y in zip(combined_df['X'], combined_df['Y'])]
    gdf = gpd.GeoDataFrame(combined_df, geometry=geometry, crs=ITM_CRS)
    
    # Convert to WGS84
    gdf = gdf.to_crs(WGS84_CRS)
    print(f"  Converted coordinates from ITM to WGS84")
    
    # Filter by bounding box if requested
    if filter_bbox:
        min_lon, min_lat, max_lon, max_lat = TEL_AVIV_BBOX
        before = len(gdf)
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat].copy()
        print(f"  Filtered by bounding box: {len(gdf)} records (from {before})")
    
    # Extract year
    gdf['year'] = gdf['SHNAT_TEUNA'].astype(int)
    
    # Print year distribution
    year_counts = gdf['year'].value_counts().sort_index()
    print(f"  Year distribution:")
    for year, count in year_counts.items():
        print(f"    {year}: {count} accidents")
    
    return gdf


def load_micromobility_accidents(
    accident_csv_paths: Union[str, List[str]],
    vehicle_csv_paths: Union[str, List[str]],
    filter_tel_aviv: bool = True,
    filter_accurate_location: bool = True,
    vehicle_types: List[int] = None,
) -> gpd.GeoDataFrame:
    """
    Load only micromobility accidents from Israeli CBS data.
    
    This function joins accident data with vehicle data to filter only
    accidents involving micromobility vehicles (electric scooters, e-bikes, etc.)
    
    Args:
        accident_csv_paths: Path(s) to accident CSV files (H*AccData.csv)
        vehicle_csv_paths: Path(s) to vehicle CSV files (H*VehData.csv)
        filter_tel_aviv: If True, filter to Tel Aviv accidents
        filter_accurate_location: If True, filter to accurate locations only
        vehicle_types: List of SUG_REHEV_LMS codes. Defaults to [15, 21, 23]
                       15 = Electric bicycle
                       21 = Electric scooter
                       23 = Other micromobility
    
    Returns:
        GeoDataFrame of micromobility accidents
    """
    print("=" * 70)
    print("LOADING MICROMOBILITY ACCIDENTS")
    print("=" * 70)
    
    # Get accident IDs involving micromobility
    print("\nStep 1: Identifying micromobility accidents from vehicle data...")
    micromobility_ids = get_micromobility_accident_ids(vehicle_csv_paths, vehicle_types)
    
    # Load accident data filtered to these IDs
    print("\nStep 2: Loading accident data...")
    accidents = load_israeli_accident_data(
        accident_csv_paths,
        filter_tel_aviv=filter_tel_aviv,
        filter_bbox=True,
        filter_accurate_location=filter_accurate_location,
        accident_ids=micromobility_ids,
    )
    
    print(f"\nLoaded {len(accidents)} micromobility accidents in Tel Aviv")
    
    return accidents


def accident_location_to_junction(
    accident_gdf: gpd.GeoDataFrame,
    junctions_gdf: gpd.GeoDataFrame,
    max_distance_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
) -> gpd.GeoDataFrame:
    """
    Transform accident locations to junction assignments.
    
    Each accident is assigned to the nearest junction within the maximum
    distance threshold. This function bridges the gap between raw accident
    coordinates and junction-based analysis.
    
    Args:
        accident_gdf: GeoDataFrame of accidents (from load_israeli_accident_data 
                      or similar). Must have 'geometry' and 'year' columns.
        junctions_gdf: GeoDataFrame of junctions with 'junction_id' column
        max_distance_meters: Maximum distance to assign an accident to a junction.
                             Accidents farther than this are dropped.
    
    Returns:
        GeoDataFrame with original accident data plus:
        - junction_id: ID of nearest junction
        - distance_to_junction: Distance in meters to assigned junction
        Accidents beyond max_distance are excluded.
    """
    print(f"\nTransforming {len(accident_gdf)} accidents to junction assignments...")
    print(f"  Max assignment distance: {max_distance_meters}m")
    
    if len(accident_gdf) == 0:
        print("  No accidents to transform")
        result = accident_gdf.copy()
        result['junction_id'] = None
        result['distance_to_junction'] = None
        return result
    
    if len(junctions_gdf) == 0:
        raise ValueError("No junctions provided")
    
    # Ensure both are in the same CRS
    if accident_gdf.crs != junctions_gdf.crs:
        accident_gdf = accident_gdf.to_crs(junctions_gdf.crs)
    
    # Project to UTM for accurate distance calculation
    accidents_utm = accident_gdf.to_crs(UTM_CRS)
    junctions_utm = junctions_gdf.to_crs(UTM_CRS)
    
    # Build KD-tree for efficient nearest neighbor search
    junction_coords = np.array([
        [geom.x, geom.y] for geom in junctions_utm.geometry
    ])
    junction_ids = junctions_gdf['junction_id'].tolist()
    tree = cKDTree(junction_coords)
    
    # Query nearest junction for each accident
    accident_coords = np.array([
        [geom.x, geom.y] for geom in accidents_utm.geometry
    ])
    distances, indices = tree.query(accident_coords, k=1)
    
    # Assign junction IDs and distances
    result = accident_gdf.copy()
    result['junction_id'] = [junction_ids[i] for i in indices]
    result['distance_to_junction'] = distances
    
    # Filter out accidents too far from any junction
    within_range = result['distance_to_junction'] <= max_distance_meters
    excluded_count = (~within_range).sum()
    result = result[within_range].copy()
    
    print(f"  Assigned: {len(result)} accidents")
    print(f"  Excluded (too far): {excluded_count}")
    
    # Stats on assigned junctions
    unique_junctions = result['junction_id'].nunique()
    print(f"  Unique junctions with accidents: {unique_junctions}")
    
    # Distance stats
    print(f"  Distance stats:")
    print(f"    Mean: {result['distance_to_junction'].mean():.1f}m")
    print(f"    Median: {result['distance_to_junction'].median():.1f}m")
    print(f"    Max: {result['distance_to_junction'].max():.1f}m")
    
    return result


def load_and_transform_israeli_accidents(
    csv_paths: Union[str, List[str]],
    junctions_gdf: gpd.GeoDataFrame,
    max_distance_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
    filter_tel_aviv: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convenience function to load Israeli accident data and transform to junctions.
    
    This is the main entry point for processing Israeli CBS accident data.
    It combines loading, coordinate conversion, and junction assignment.
    
    Args:
        csv_paths: Path(s) to accident CSV files (supports glob patterns)
        junctions_gdf: GeoDataFrame of junctions
        max_distance_meters: Max distance to assign accident to junction
        filter_tel_aviv: Filter to Tel Aviv area only
    
    Returns:
        GeoDataFrame of accidents assigned to junctions
    """
    # Load and convert accident data
    accidents = load_israeli_accident_data(
        csv_paths,
        filter_tel_aviv=filter_tel_aviv,
        filter_bbox=True,
    )
    
    # Transform to junction assignments
    return accident_location_to_junction(
        accidents,
        junctions_gdf,
        max_distance_meters=max_distance_meters,
    )


def load_accidents_csv(
    csv_path: str,
    lat_column: str = 'latitude',
    lon_column: str = 'longitude',
    date_column: Optional[str] = 'date',
    year_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load accident data from a CSV file with coordinates.
    
    Args:
        csv_path: Path to the CSV file
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        date_column: Name of the date column (for extracting year)
        year_column: Name of year column if already extracted (overrides date_column)
    
    Returns:
        GeoDataFrame with accident points and year column
    """
    print(f"Loading accidents from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} accident records")
    
    # Validate required columns
    if lat_column not in df.columns:
        raise ValueError(f"Latitude column '{lat_column}' not found. Available: {list(df.columns)}")
    if lon_column not in df.columns:
        raise ValueError(f"Longitude column '{lon_column}' not found. Available: {list(df.columns)}")
    
    # Drop rows with missing coordinates
    valid_coords = df[[lat_column, lon_column]].notna().all(axis=1)
    df = df[valid_coords].copy()
    print(f"  {len(df)} records with valid coordinates")
    
    # Create geometry
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_column], df[lat_column])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84_CRS)
    
    # Extract or validate year
    if year_column and year_column in gdf.columns:
        gdf['year'] = gdf[year_column].astype(int)
    elif date_column and date_column in gdf.columns:
        # Try to parse date and extract year
        try:
            gdf['year'] = pd.to_datetime(gdf[date_column]).dt.year
            print(f"  Extracted year from '{date_column}' column")
        except Exception as e:
            print(f"  Warning: Could not parse dates from '{date_column}': {e}")
            print(f"  Setting year to None - temporal joining will not work")
            gdf['year'] = None
    else:
        print(f"  Warning: No date/year column found - temporal joining will not work")
        gdf['year'] = None
    
    # Print year distribution
    if gdf['year'].notna().any():
        year_counts = gdf['year'].value_counts().sort_index()
        print(f"  Year distribution:")
        for year, count in year_counts.items():
            print(f"    {int(year)}: {count} accidents")
    
    return gdf


def find_nearest_junction(
    accident_point: Point,
    junctions_tree: cKDTree,
    junctions_coords: np.ndarray,
    junction_ids: List[int],
    max_distance_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the nearest junction to an accident point.
    
    Args:
        accident_point: Point geometry of accident (in UTM)
        junctions_tree: cKDTree of junction coordinates
        junctions_coords: Array of junction coordinates (UTM)
        junction_ids: List of junction IDs corresponding to coordinates
        max_distance_meters: Maximum distance to consider
    
    Returns:
        Tuple of (nearest_junction_id, distance) or (None, None) if too far
    """
    coords = np.array([[accident_point.x, accident_point.y]])
    distance, idx = junctions_tree.query(coords, k=1)
    
    distance = distance[0]
    idx = idx[0]
    
    if distance <= max_distance_meters:
        return junction_ids[idx], distance
    
    return None, None


def join_accidents_to_junctions(
    accidents_gdf: gpd.GeoDataFrame,
    junctions_gdf: gpd.GeoDataFrame,
    max_distance_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
) -> gpd.GeoDataFrame:
    """
    Spatially join accidents to their nearest junction.
    
    Each accident is assigned to the nearest junction within the
    maximum distance threshold.
    
    Args:
        accidents_gdf: GeoDataFrame of accident points
        junctions_gdf: GeoDataFrame of junction points
        max_distance_meters: Maximum distance to assign accident to junction
    
    Returns:
        GeoDataFrame of accidents with junction_id and distance columns
    """
    print(f"Joining {len(accidents_gdf)} accidents to {len(junctions_gdf)} junctions...")
    print(f"  Max distance: {max_distance_meters}m")
    
    if len(accidents_gdf) == 0:
        return accidents_gdf
    
    if len(junctions_gdf) == 0:
        accidents_gdf = accidents_gdf.copy()
        accidents_gdf['junction_id'] = None
        accidents_gdf['distance_to_junction'] = None
        return accidents_gdf
    
    # Project to UTM for accurate distance calculations
    accidents_utm = accidents_gdf.to_crs(UTM_CRS)
    junctions_utm = junctions_gdf.to_crs(UTM_CRS)
    
    # Build KD-tree for junctions
    junction_coords = np.array([
        [geom.x, geom.y] for geom in junctions_utm.geometry
    ])
    junction_ids = junctions_gdf['junction_id'].tolist()
    
    tree = cKDTree(junction_coords)
    
    # Find nearest junction for each accident
    accident_coords = np.array([
        [geom.x, geom.y] for geom in accidents_utm.geometry
    ])
    
    distances, indices = tree.query(accident_coords, k=1)
    
    # Assign junction IDs (None if too far)
    assigned_junctions = []
    assigned_distances = []
    
    for dist, idx in zip(distances, indices):
        if dist <= max_distance_meters:
            assigned_junctions.append(junction_ids[idx])
            assigned_distances.append(dist)
        else:
            assigned_junctions.append(None)
            assigned_distances.append(None)
    
    # Add to output
    result = accidents_gdf.copy()
    result['junction_id'] = assigned_junctions
    result['distance_to_junction'] = assigned_distances
    
    # Stats
    matched = result['junction_id'].notna().sum()
    unmatched = len(result) - matched
    print(f"  Matched: {matched} ({100*matched/len(result):.1f}%)")
    print(f"  Unmatched (too far): {unmatched}")
    
    return result


def aggregate_accidents_by_junction_year(
    accidents_with_junctions: gpd.GeoDataFrame,
    junctions_gdf: gpd.GeoDataFrame,
    years: range = YEARS,
) -> pd.DataFrame:
    """
    Aggregate accident counts by junction and year.
    
    Creates a DataFrame with one row per (junction_id, year) combination,
    containing the accident count for that junction in that year.
    
    Args:
        accidents_with_junctions: GeoDataFrame from join_accidents_to_junctions
        junctions_gdf: GeoDataFrame of junctions
        years: Range of years to include
    
    Returns:
        DataFrame with columns: junction_id, year, accident_count
    """
    print("Aggregating accident counts by junction and year...")
    
    # Filter to matched accidents with valid years
    matched = accidents_with_junctions[
        accidents_with_junctions['junction_id'].notna() &
        accidents_with_junctions['year'].notna()
    ].copy()
    
    if len(matched) == 0:
        print("  Warning: No matched accidents with valid years")
        # Return empty counts for all junction-year combinations
        rows = []
        for jid in junctions_gdf['junction_id']:
            for year in years:
                rows.append({'junction_id': jid, 'year': year, 'accident_count': 0})
        return pd.DataFrame(rows)
    
    # Group by junction and year
    counts = matched.groupby(['junction_id', 'year']).size().reset_index(name='accident_count')
    
    # Create full grid of junction × year combinations
    junction_ids = junctions_gdf['junction_id'].tolist()
    full_grid = pd.DataFrame([
        {'junction_id': jid, 'year': year}
        for jid in junction_ids
        for year in years
    ])
    
    # Merge counts with full grid (fill missing with 0)
    result = full_grid.merge(counts, on=['junction_id', 'year'], how='left')
    result['accident_count'] = result['accident_count'].fillna(0).astype(int)
    
    # Stats
    total_accidents = result['accident_count'].sum()
    junctions_with_accidents = (result['accident_count'] > 0).sum()
    
    print(f"  Total accidents assigned: {total_accidents}")
    print(f"  Junction-years with accidents: {junctions_with_accidents}/{len(result)}")
    
    return result


def join_accidents_temporal(
    panel_df: pd.DataFrame,
    accidents_csv: str,
    junctions_gdf: gpd.GeoDataFrame,
    lat_column: str = 'latitude',
    lon_column: str = 'longitude',
    date_column: Optional[str] = 'date',
    year_column: Optional[str] = None,
    radius_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
) -> pd.DataFrame:
    """
    Join accidents to panel dataset by year and location.
    
    This is the main entry point for adding accident labels to the
    panel dataset. Each accident is assigned to:
    1. The nearest junction within the radius threshold
    2. The year the accident occurred
    
    Args:
        panel_df: Panel DataFrame with junction_id and year columns
        accidents_csv: Path to accidents CSV file
        junctions_gdf: GeoDataFrame of junctions (for spatial join)
        lat_column: Name of latitude column in accidents CSV
        lon_column: Name of longitude column in accidents CSV
        date_column: Name of date column in accidents CSV
        year_column: Name of year column (if pre-extracted)
        radius_meters: Max distance to assign accident to junction
    
    Returns:
        Panel DataFrame with accident_count column added
    """
    print("=" * 60)
    print("JOINING ACCIDENTS TO PANEL DATASET")
    print("=" * 60)
    
    # Load accidents
    accidents_gdf = load_accidents_csv(
        accidents_csv,
        lat_column=lat_column,
        lon_column=lon_column,
        date_column=date_column,
        year_column=year_column,
    )
    
    # Spatial join
    accidents_with_junctions = join_accidents_to_junctions(
        accidents_gdf, junctions_gdf, max_distance_meters=radius_meters
    )
    
    # Get unique years from panel
    years = sorted(panel_df['year'].unique())
    
    # Aggregate counts
    accident_counts = aggregate_accidents_by_junction_year(
        accidents_with_junctions, junctions_gdf, years=range(min(years), max(years) + 1)
    )
    
    # Merge with panel
    result = panel_df.merge(
        accident_counts[['junction_id', 'year', 'accident_count']],
        on=['junction_id', 'year'],
        how='left'
    )
    
    # Fill any missing with 0
    result['accident_count'] = result['accident_count'].fillna(0).astype(int)
    
    print(f"\nFinal panel shape: {result.shape}")
    print(f"Total accidents in dataset: {result['accident_count'].sum()}")
    
    return result


def add_accident_labels_to_panel(
    panel_csv_path: str,
    accident_csv_paths: Union[str, List[str]],
    junctions_gdf: gpd.GeoDataFrame,
    output_path: Optional[str] = None,
    max_distance_meters: float = DEFAULT_ACCIDENT_RADIUS_METERS,
) -> pd.DataFrame:
    """
    Add accident counts to panel dataset from Israeli CBS accident data.
    
    This function:
    1. Loads Israeli accident data (with STATUS_IGUN=1 filter)
    2. Transforms accidents to junction assignments
    3. Aggregates accident counts by junction-year
    4. Joins to the panel dataset
    
    Args:
        panel_csv_path: Path to the panel CSV file (junction × year)
        accident_csv_paths: Path(s) to accident CSV files (supports glob patterns)
        junctions_gdf: GeoDataFrame of junctions (for spatial join)
        output_path: Optional path to save the labeled panel CSV
        max_distance_meters: Max distance to assign accident to junction
    
    Returns:
        Panel DataFrame with 'accident_count' column added
    """
    print("=" * 70)
    print("ADDING ACCIDENT LABELS TO PANEL DATASET")
    print("=" * 70)
    
    # Load panel dataset
    print(f"\n1. Loading panel dataset: {panel_csv_path}")
    panel_df = pd.read_csv(panel_csv_path)
    print(f"   Panel shape: {panel_df.shape}")
    print(f"   Years in panel: {sorted(panel_df['year'].unique())}")
    
    # Load and transform accidents
    print(f"\n2. Loading and transforming accident data...")
    accidents = load_israeli_accident_data(
        accident_csv_paths,
        filter_tel_aviv=True,
        filter_bbox=True,
        filter_accurate_location=True,
    )
    
    # Assign to junctions
    print(f"\n3. Assigning accidents to junctions...")
    accidents_with_junctions = accident_location_to_junction(
        accidents,
        junctions_gdf,
        max_distance_meters=max_distance_meters,
    )
    
    # Filter to years in panel
    panel_years = set(panel_df['year'].unique())
    accidents_filtered = accidents_with_junctions[
        accidents_with_junctions['year'].isin(panel_years)
    ].copy()
    print(f"\n4. Filtered to panel years: {len(accidents_filtered)} accidents")
    
    # Aggregate by junction and year
    print(f"\n5. Aggregating accidents by junction-year...")
    accident_counts = accidents_filtered.groupby(
        ['junction_id', 'year']
    ).size().reset_index(name='accident_count')
    
    print(f"   Junction-years with accidents: {len(accident_counts)}")
    print(f"   Total accidents assigned: {accident_counts['accident_count'].sum()}")
    
    # Merge with panel
    print(f"\n6. Merging with panel dataset...")
    result = panel_df.merge(
        accident_counts,
        on=['junction_id', 'year'],
        how='left'
    )
    
    # Fill missing with 0
    result['accident_count'] = result['accident_count'].fillna(0).astype(int)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final panel shape: {result.shape}")
    print(f"Total accidents in panel: {result['accident_count'].sum()}")
    print(f"Junction-years with >= 1 accident: {(result['accident_count'] > 0).sum()}")
    print(f"Junction-years with 0 accidents: {(result['accident_count'] == 0).sum()}")
    print(f"\nAccident count distribution:")
    print(result['accident_count'].describe())
    
    # By year
    print("\nAccidents by year:")
    by_year = result.groupby('year')['accident_count'].sum()
    for year, count in by_year.items():
        print(f"  {year}: {count}")
    
    # Top junctions
    print("\nTop 10 junction-years by accident count:")
    top = result.nlargest(10, 'accident_count')[['junction_id', 'year', 'latitude', 'longitude', 'accident_count']]
    print(top.to_string(index=False))
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving labeled panel to: {output_path}")
        result.to_csv(output_path, index=False)
        print(f"  Saved {len(result)} rows")
    
    return result

