"""
Spatial operations for joining accidents to junctions.

This module handles:
- Loading accident data from CSV with coordinates
- Spatial joining accidents to nearest junction
- Temporal filtering to assign accidents to the correct year
- Aggregating accident counts per junction per year
"""

from typing import Optional, Tuple, List
from pathlib import Path

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

