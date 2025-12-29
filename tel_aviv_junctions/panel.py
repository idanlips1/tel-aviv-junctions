"""
Panel dataset builder for junction × year data.

This module orchestrates the full pipeline:
1. Extract junctions from current OSM data
2. Extract features from OSMnx edge data (or ohsome API if available)
3. Build panel dataset with one row per junction per year
4. Optionally join accident data as labels

The resulting panel dataset is suitable for ML models that need to
account for temporal changes in infrastructure.
"""

import os
from typing import Optional, Dict, List
from pathlib import Path

import pandas as pd
import geopandas as gpd

from tel_aviv_junctions.config import (
    TEL_AVIV_PLACE_NAME,
    TEL_AVIV_BBOX,
    YEARS,
    OUTPUT_DIR,
    WGS84_CRS,
)
from tel_aviv_junctions.extraction import extract_clustered_junctions
from tel_aviv_junctions.features import extract_features_for_junctions, extract_road_features
from tel_aviv_junctions.spatial import join_accidents_temporal


def extract_features_from_osmnx_edges(
    junctions_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 30.0,
) -> pd.DataFrame:
    """
    Extract features for junctions using OSMnx edge data directly.
    
    This uses the current OSM data from OSMnx instead of ohsome API.
    The features represent the current state of the road network.
    
    Args:
        junctions_gdf: GeoDataFrame of junction points
        edges_gdf: GeoDataFrame of road edges from OSMnx
        buffer_meters: Buffer radius for spatial queries
    
    Returns:
        DataFrame with junction features
    """
    print("Extracting features from OSMnx edge data...")
    
    # Convert edges to a format compatible with our feature extraction
    roads_gdf = _convert_osmnx_edges_to_roads(edges_gdf)
    
    # Extract features
    features_df = extract_features_for_junctions(
        junctions_gdf=junctions_gdf,
        roads_gdf=roads_gdf,
        traffic_signals_gdf=None,  # Not available from OSMnx directly
        crossings_gdf=None,
        buffer_meters=buffer_meters,
    )
    
    return features_df


def _convert_osmnx_edges_to_roads(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert OSMnx edge GeoDataFrame to format expected by feature extraction.
    
    OSMnx edges have multi-index (u, v, key) and various tag columns.
    We flatten it and standardize column names.
    """
    if edges_gdf.empty:
        return gpd.GeoDataFrame(columns=['geometry'], crs=WGS84_CRS)
    
    # Reset index to make it a regular dataframe
    roads = edges_gdf.reset_index().copy()
    
    # Ensure we have a geometry column
    if 'geometry' not in roads.columns:
        return gpd.GeoDataFrame(columns=['geometry'], crs=WGS84_CRS)
    
    # The OSMnx edges already have most tags we need
    # Common columns: highway, maxspeed, lanes, surface, oneway, name, etc.
    
    return gpd.GeoDataFrame(roads, crs=edges_gdf.crs)


def build_panel_dataset_from_osmnx(
    junctions_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    years: range = YEARS,
    buffer_meters: float = 30.0,
) -> pd.DataFrame:
    """
    Build a panel dataset using OSMnx data (current snapshot replicated for all years).
    
    Since we don't have historical data, the features are the same for all years.
    This is still useful for spatial analysis and as a starting point for ML.
    
    Args:
        junctions_gdf: GeoDataFrame of junction points
        edges_gdf: GeoDataFrame of road edges from OSMnx
        years: Range of years to include
        buffer_meters: Buffer radius for feature extraction
    
    Returns:
        DataFrame with panel structure (junction × year)
    """
    print("=" * 60)
    print("BUILDING PANEL DATASET FROM OSMNX DATA")
    print("=" * 60)
    print(f"Junctions: {len(junctions_gdf)}")
    print(f"Edges: {len(edges_gdf)}")
    print(f"Years: {min(years)} - {max(years)}")
    print(f"Total rows: {len(junctions_gdf) * len(years)}")
    print()
    print("Note: Using current OSM data from OSMnx.")
    print("Features are the same for all years (current snapshot).")
    print()
    
    # Extract features from OSMnx edges
    features_df = extract_features_from_osmnx_edges(
        junctions_gdf=junctions_gdf,
        edges_gdf=edges_gdf,
        buffer_meters=buffer_meters,
    )
    
    # Create panel by replicating for each year
    panel_rows = []
    for year in years:
        year_df = features_df.copy()
        year_df['year'] = year
        panel_rows.append(year_df)
    
    panel_df = pd.concat(panel_rows, ignore_index=True)
    
    # Add static junction properties
    static_cols = ['junction_id', 'latitude', 'longitude', 'degree', 'node_count']
    static_data = junctions_gdf[
        [c for c in static_cols if c in junctions_gdf.columns]
    ].copy()
    
    # Merge static properties
    panel_df = panel_df.merge(
        static_data,
        on='junction_id',
        how='left',
        suffixes=('', '_static')
    )
    
    # Reorder columns for clarity
    key_cols = ['junction_id', 'year', 'latitude', 'longitude', 'degree', 'node_count']
    feature_cols = [c for c in panel_df.columns if c not in key_cols]
    panel_df = panel_df[key_cols + feature_cols]
    
    print(f"\nPanel dataset shape: {panel_df.shape}")
    print(f"Columns: {list(panel_df.columns)}")
    
    return panel_df


def build_full_pipeline(
    place_name: str = TEL_AVIV_PLACE_NAME,
    bbox: tuple = TEL_AVIV_BBOX,
    years: range = YEARS,
    accidents_csv: Optional[str] = None,
    accident_lat_col: str = 'latitude',
    accident_lon_col: str = 'longitude',
    accident_date_col: Optional[str] = 'date',
    accident_year_col: Optional[str] = None,
    accident_radius_meters: float = 50.0,
    use_cache: bool = True,
    use_ohsome: bool = False,  # Default to False - use OSMnx data
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline to build an ML-ready panel dataset.
    
    This is the main entry point for the entire workflow:
    1. Extract junctions from OSM
    2. Build features from OSMnx edges (or ohsome if enabled)
    3. Optionally join accident labels
    4. Save to CSV/Parquet
    
    Args:
        place_name: Place name for OSMnx junction extraction
        bbox: Bounding box for ohsome queries (if used)
        years: Range of years for panel structure
        accidents_csv: Optional path to accidents CSV for labels
        accident_lat_col: Latitude column name in accidents CSV
        accident_lon_col: Longitude column name in accidents CSV
        accident_date_col: Date column name in accidents CSV
        accident_year_col: Year column name (if pre-extracted)
        accident_radius_meters: Max distance to assign accidents
        use_cache: Whether to cache ohsome API results (if used)
        use_ohsome: Whether to use ohsome API for historical data
        output_path: Optional path to save the result
    
    Returns:
        Panel DataFrame ready for ML
    """
    print("=" * 60)
    print("TEL AVIV JUNCTIONS - FULL ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Extract junctions (also gets edge data)
    print("\n[STEP 1/4] EXTRACTING JUNCTIONS")
    clustered_junctions, raw_junctions, drive_edges, bike_edges = \
        extract_clustered_junctions(place_name)
    
    print(f"  Clustered junctions: {len(clustered_junctions)}")
    
    # Combine drive and bike edges
    combined_edges = pd.concat([drive_edges, bike_edges], ignore_index=True)
    combined_edges = gpd.GeoDataFrame(combined_edges, crs=drive_edges.crs)
    print(f"  Combined edges: {len(combined_edges)}")
    
    # Step 2: Build panel dataset with features
    print("\n[STEP 2/4] BUILDING PANEL DATASET WITH FEATURES")
    
    if use_ohsome:
        # Try ohsome API for historical data
        from tel_aviv_junctions.ohsome import get_yearly_infrastructure_snapshots
        from tel_aviv_junctions.features import extract_features_for_year
        
        try:
            yearly_infrastructure = get_yearly_infrastructure_snapshots(
                bbox=bbox, years=years, use_cache=use_cache
            )
            
            # Check if we got actual data
            first_year = list(yearly_infrastructure.values())[0]
            if len(first_year.get('roads', [])) == 0:
                print("  ohsome API returned no data, falling back to OSMnx...")
                use_ohsome = False
        except Exception as e:
            print(f"  ohsome API failed: {e}")
            print("  Falling back to OSMnx data...")
            use_ohsome = False
    
    if not use_ohsome:
        # Use OSMnx data (default)
        panel_df = build_panel_dataset_from_osmnx(
            junctions_gdf=clustered_junctions,
            edges_gdf=combined_edges,
            years=years,
            buffer_meters=30.0,
        )
    else:
        # Use ohsome data
        from tel_aviv_junctions.features import extract_features_for_year
        
        yearly_features = []
        for year in years:
            year_features = extract_features_for_year(
                junctions_gdf=clustered_junctions,
                yearly_infrastructure=yearly_infrastructure[year],
                year=year,
                buffer_meters=30.0,
            )
            yearly_features.append(year_features)
        
        panel_df = pd.concat(yearly_features, ignore_index=True)
        
        # Add static junction properties
        static_cols = ['junction_id', 'latitude', 'longitude', 'degree', 'node_count']
        static_data = clustered_junctions[
            [c for c in static_cols if c in clustered_junctions.columns]
        ].copy()
        
        panel_df = panel_df.merge(
            static_data, on='junction_id', how='left', suffixes=('', '_static')
        )
    
    # Step 3: Join accidents (if provided)
    if accidents_csv:
        print("\n[STEP 3/4] JOINING ACCIDENT LABELS")
        panel_df = join_accidents_temporal(
            panel_df=panel_df,
            accidents_csv=accidents_csv,
            junctions_gdf=clustered_junctions,
            lat_column=accident_lat_col,
            lon_column=accident_lon_col,
            date_column=accident_date_col,
            year_column=accident_year_col,
            radius_meters=accident_radius_meters,
        )
    else:
        print("\n[STEP 3/4] SKIPPING ACCIDENT LABELS (no CSV provided)")
    
    # Step 4: Save output
    print("\n[STEP 4/4] SAVING OUTPUT")
    if output_path:
        _save_panel_dataset(panel_df, output_path)
    else:
        default_path = os.path.join(OUTPUT_DIR, "tel_aviv_junctions_panel.csv")
        _save_panel_dataset(panel_df, default_path)
    
    # Summary
    _print_panel_summary(panel_df)
    
    return panel_df


def _save_panel_dataset(panel_df: pd.DataFrame, output_path: str) -> None:
    """Save panel dataset to file (CSV or Parquet based on extension)."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if output_path.endswith('.parquet'):
        panel_df.to_parquet(output_path, index=False)
        print(f"  Saved to: {output_path} (Parquet)")
    else:
        panel_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path} (CSV)")


def _print_panel_summary(panel_df: pd.DataFrame) -> None:
    """Print summary statistics of the panel dataset."""
    print("\n" + "=" * 60)
    print("PANEL DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nShape: {panel_df.shape[0]} rows × {panel_df.shape[1]} columns")
    
    n_junctions = panel_df['junction_id'].nunique()
    n_years = panel_df['year'].nunique()
    print(f"Junctions: {n_junctions}")
    print(f"Years: {sorted(panel_df['year'].unique())}")
    
    # Feature statistics
    print("\nFeature coverage:")
    
    bool_cols = panel_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        pct = 100 * panel_df[col].sum() / len(panel_df)
        print(f"  {col}: {pct:.1f}% True")
    
    # Numeric feature stats
    numeric_cols = ['max_speed', 'total_lanes', 'road_count']
    for col in numeric_cols:
        if col in panel_df.columns:
            valid = panel_df[col].notna()
            if valid.any():
                mean = panel_df.loc[valid, col].mean()
                print(f"  {col}: mean={mean:.1f} ({100*valid.sum()/len(panel_df):.0f}% non-null)")
    
    # Accident stats (if present)
    if 'accident_count' in panel_df.columns:
        total = panel_df['accident_count'].sum()
        has_accidents = (panel_df['accident_count'] > 0).sum()
        print(f"\nAccident labels:")
        print(f"  Total accidents: {total}")
        print(f"  Junction-years with accidents: {has_accidents}/{len(panel_df)} ({100*has_accidents/len(panel_df):.1f}%)")
    
    print("=" * 60)


def load_panel_dataset(path: str) -> pd.DataFrame:
    """
    Load a previously saved panel dataset.
    
    Args:
        path: Path to CSV or Parquet file
    
    Returns:
        Panel DataFrame
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def get_junctions_for_year(panel_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Get junction features for a specific year.
    
    Useful for analyzing a single year's snapshot.
    
    Args:
        panel_df: Full panel dataset
        year: Year to filter
    
    Returns:
        DataFrame with junctions for that year only
    """
    return panel_df[panel_df['year'] == year].copy()


def get_junction_history(panel_df: pd.DataFrame, junction_id: int) -> pd.DataFrame:
    """
    Get the temporal history of a single junction.
    
    Useful for analyzing how a junction's features changed over time.
    
    Args:
        panel_df: Full panel dataset
        junction_id: Junction to query
    
    Returns:
        DataFrame with all years for that junction
    """
    return panel_df[panel_df['junction_id'] == junction_id].sort_values('year').copy()


def compute_temporal_changes(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute which junctions had feature changes over time.
    
    Identifies junctions where time-varying features changed,
    which is useful for analyzing infrastructure improvements.
    
    Args:
        panel_df: Full panel dataset
    
    Returns:
        DataFrame with change statistics per junction
    """
    # Time-varying boolean features
    bool_features = [
        'has_cycleway', 'has_traffic_signal', 'has_crossing',
        'has_traffic_calming', 'lit', 'has_segregated_cycleway'
    ]
    
    existing_features = [f for f in bool_features if f in panel_df.columns]
    
    changes = []
    
    for junction_id, group in panel_df.groupby('junction_id'):
        group = group.sort_values('year')
        
        change_record = {'junction_id': junction_id}
        
        for feature in existing_features:
            values = group[feature].tolist()
            # Check if feature changed at any point
            changed = len(set(values)) > 1
            change_record[f'{feature}_changed'] = changed
            
            # If changed, record when it became True (if applicable)
            if changed and True in values and False in values:
                first_true_idx = values.index(True) if True in values else None
                if first_true_idx is not None and first_true_idx > 0:
                    change_record[f'{feature}_added_year'] = group.iloc[first_true_idx]['year']
        
        changes.append(change_record)
    
    return pd.DataFrame(changes)
