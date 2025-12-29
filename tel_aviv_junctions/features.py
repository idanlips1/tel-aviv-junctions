"""
Feature extraction and aggregation for junctions.

This module handles spatially joining road features to junction locations
and aggregating them into ML-ready features. For each junction, we compute
features from the surrounding road segments.
"""

from typing import Dict, List, Optional, Set, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

from tel_aviv_junctions.config import (
    UTM_CRS,
    WGS84_CRS,
    HIGHWAY_TYPES,
    SURFACE_TYPES,
    CYCLEWAY_TYPES,
    DEFAULT_ACCIDENT_RADIUS_METERS,
)


def _safe_get_tag(row: pd.Series, tag_name: str, default: Any = None) -> Any:
    """Safely get a tag value from a row, handling missing/None values and arrays."""
    val = row.get(tag_name, default)
    
    # Handle array/list values
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return default
        val = val[0]  # Take first element
    
    # Check for NA/None
    try:
        if pd.isna(val):
            return default
    except (ValueError, TypeError):
        # pd.isna() can fail on some types
        if val is None:
            return default
    
    return val


def _parse_speed(speed_str: Any) -> Optional[float]:
    """
    Parse a maxspeed value from OSM format.
    
    Handles formats like:
    - "50" -> 50.0
    - "50 km/h" -> 50.0
    - "30 mph" -> 48.3 (converted)
    """
    if speed_str is None or pd.isna(speed_str):
        return None
    
    speed_str = str(speed_str).strip().lower()
    
    # Handle mph
    if 'mph' in speed_str:
        try:
            num = float(speed_str.replace('mph', '').strip())
            return num * 1.60934  # Convert to km/h
        except ValueError:
            return None
    
    # Remove common suffixes
    for suffix in ['km/h', 'kmh', 'kph']:
        speed_str = speed_str.replace(suffix, '')
    
    try:
        return float(speed_str.strip())
    except ValueError:
        return None


def _parse_lanes(lanes_str: Any) -> Optional[int]:
    """Parse number of lanes from OSM format."""
    if lanes_str is None or pd.isna(lanes_str):
        return None
    
    try:
        return int(float(str(lanes_str)))
    except ValueError:
        return None


def _parse_width(width_str: Any) -> Optional[float]:
    """Parse road width from OSM format (meters)."""
    if width_str is None or pd.isna(width_str):
        return None
    
    width_str = str(width_str).strip().lower()
    
    # Remove 'm' suffix if present
    width_str = width_str.replace('m', '').strip()
    
    try:
        return float(width_str)
    except ValueError:
        return None


def extract_road_features(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract and normalize features from road geometries.
    
    Handles both OSMnx edge format and ohsome API format.
    
    Args:
        roads_gdf: GeoDataFrame of road segments (from OSMnx or ohsome)
    
    Returns:
        GeoDataFrame with normalized feature columns
    """
    if len(roads_gdf) == 0:
        return gpd.GeoDataFrame(columns=['geometry'], crs=WGS84_CRS)
    
    features = []
    
    for idx, row in roads_gdf.iterrows():
        # Handle different ID formats (OSMnx vs ohsome)
        osm_id = _safe_get_tag(row, '@osmId') or _safe_get_tag(row, 'osmid')
        
        # Get highway type - handle list values from OSMnx
        highway_val = _safe_get_tag(row, 'highway')
        if isinstance(highway_val, list):
            highway_val = highway_val[0] if highway_val else None
        
        # Get maxspeed - handle list values
        maxspeed_val = _safe_get_tag(row, 'maxspeed')
        if isinstance(maxspeed_val, list):
            maxspeed_val = maxspeed_val[0] if maxspeed_val else None
        
        # Get lanes - handle list values
        lanes_val = _safe_get_tag(row, 'lanes')
        if isinstance(lanes_val, list):
            lanes_val = lanes_val[0] if lanes_val else None
        
        # Get surface - handle list values
        surface_val = _safe_get_tag(row, 'surface')
        if isinstance(surface_val, list):
            surface_val = surface_val[0] if surface_val else None
        
        # Get oneway - handle various formats
        oneway_val = _safe_get_tag(row, 'oneway')
        if isinstance(oneway_val, list):
            oneway_val = oneway_val[0] if oneway_val else None
        is_oneway = oneway_val in ['yes', 'true', '1', '-1', True]
        
        # Get lit - handle various formats
        lit_val = _safe_get_tag(row, 'lit')
        if isinstance(lit_val, list):
            lit_val = lit_val[0] if lit_val else None
        is_lit = lit_val in ['yes', 'true', '1', True]
        
        # Cycling infrastructure
        cycleway_val = _safe_get_tag(row, 'cycleway')
        if isinstance(cycleway_val, list):
            cycleway_val = cycleway_val[0] if cycleway_val else None
        
        cycleway_left = _safe_get_tag(row, 'cycleway:left')
        cycleway_right = _safe_get_tag(row, 'cycleway:right')
        bicycle_val = _safe_get_tag(row, 'bicycle')
        if isinstance(bicycle_val, list):
            bicycle_val = bicycle_val[0] if bicycle_val else None
        
        segregated_val = _safe_get_tag(row, 'segregated')
        is_segregated = segregated_val in ['yes', 'true', '1', True]
        
        # Traffic calming
        traffic_calming_val = _safe_get_tag(row, 'traffic_calming')
        
        feat = {
            'geometry': row.geometry,
            'osm_id': osm_id,
            'highway': highway_val,
            'maxspeed': _parse_speed(maxspeed_val),
            'lanes': _parse_lanes(lanes_val),
            'width': _parse_width(_safe_get_tag(row, 'width')),
            'surface': surface_val,
            'oneway': is_oneway,
            'lit': is_lit,
            'cycleway': cycleway_val,
            'cycleway_left': cycleway_left,
            'cycleway_right': cycleway_right,
            'bicycle': bicycle_val,
            'segregated': is_segregated,
            'traffic_calming': traffic_calming_val,
        }
        features.append(feat)
    
    return gpd.GeoDataFrame(features, crs=roads_gdf.crs)


def get_roads_near_junction(
    junction_point: Point,
    roads_gdf: gpd.GeoDataFrame,
    buffer_meters: float = 30.0,
) -> gpd.GeoDataFrame:
    """
    Get all road segments that intersect a buffer around a junction.
    
    Args:
        junction_point: Point geometry of the junction (in WGS84)
        roads_gdf: GeoDataFrame of road segments
        buffer_meters: Buffer radius in meters
    
    Returns:
        GeoDataFrame of road segments near the junction
    """
    if len(roads_gdf) == 0:
        return roads_gdf
    
    # Convert to UTM for accurate buffering
    junction_utm = gpd.GeoSeries([junction_point], crs=WGS84_CRS).to_crs(UTM_CRS)[0]
    roads_utm = roads_gdf.to_crs(UTM_CRS)
    
    # Create buffer
    buffer = junction_utm.buffer(buffer_meters)
    
    # Find intersecting roads
    mask = roads_utm.geometry.intersects(buffer)
    nearby_roads = roads_gdf[mask].copy()
    
    return nearby_roads


def aggregate_junction_features(
    junction_id: int,
    junction_point: Point,
    roads_gdf: gpd.GeoDataFrame,
    traffic_signals_gdf: Optional[gpd.GeoDataFrame] = None,
    crossings_gdf: Optional[gpd.GeoDataFrame] = None,
    buffer_meters: float = 30.0,
) -> Dict[str, Any]:
    """
    Aggregate features from surrounding roads for a single junction.
    
    Args:
        junction_id: Unique identifier for the junction
        junction_point: Point geometry of the junction
        roads_gdf: GeoDataFrame of road segments with extracted features
        traffic_signals_gdf: Optional GeoDataFrame of traffic signal points
        crossings_gdf: Optional GeoDataFrame of crossing points
        buffer_meters: Buffer radius for finding nearby features
    
    Returns:
        Dictionary of aggregated features
    """
    # Get nearby roads
    nearby_roads = get_roads_near_junction(junction_point, roads_gdf, buffer_meters)
    
    features = {
        'junction_id': junction_id,
    }
    
    if len(nearby_roads) == 0:
        # No roads found - return default values
        features.update({
            'road_count': 0,
            'max_speed': None,
            'avg_speed': None,
            'total_lanes': None,
            'max_lanes': None,
            'avg_width': None,
            'has_oneway': False,
            'lit': False,
            'has_cycleway': False,
            'cycleway_type': None,
            'has_segregated_cycleway': False,
            'has_traffic_calming': False,
            'dominant_surface': None,
        })
        
        # Highway type flags - all False
        for hw_type in HIGHWAY_TYPES:
            features[f'highway_{hw_type}'] = False
        
    else:
        # Road count
        features['road_count'] = len(nearby_roads)
        
        # Speed aggregation
        speeds = nearby_roads['maxspeed'].dropna()
        features['max_speed'] = speeds.max() if len(speeds) > 0 else None
        features['avg_speed'] = speeds.mean() if len(speeds) > 0 else None
        
        # Lanes aggregation
        lanes = nearby_roads['lanes'].dropna()
        features['total_lanes'] = int(lanes.sum()) if len(lanes) > 0 else None
        features['max_lanes'] = int(lanes.max()) if len(lanes) > 0 else None
        
        # Width aggregation
        widths = nearby_roads['width'].dropna()
        features['avg_width'] = widths.mean() if len(widths) > 0 else None
        
        # Boolean aggregations
        features['has_oneway'] = nearby_roads['oneway'].any()
        features['lit'] = nearby_roads['lit'].any()
        
        # Cycleway detection
        has_cycleway = (
            nearby_roads['cycleway'].notna().any() |
            nearby_roads['cycleway_left'].notna().any() |
            nearby_roads['cycleway_right'].notna().any() |
            (nearby_roads['highway'] == 'cycleway').any() |
            (nearby_roads['bicycle'].isin(['designated', 'yes'])).any()
        )
        features['has_cycleway'] = bool(has_cycleway)
        
        # Get dominant cycleway type
        cycleway_vals = nearby_roads['cycleway'].dropna().tolist()
        cycleway_vals += nearby_roads['cycleway_left'].dropna().tolist()
        cycleway_vals += nearby_roads['cycleway_right'].dropna().tolist()
        
        if cycleway_vals:
            # Get most common cycleway type
            from collections import Counter
            cycleway_counts = Counter(cycleway_vals)
            features['cycleway_type'] = cycleway_counts.most_common(1)[0][0]
        else:
            features['cycleway_type'] = None
        
        features['has_segregated_cycleway'] = nearby_roads['segregated'].any()
        
        # Traffic calming
        features['has_traffic_calming'] = nearby_roads['traffic_calming'].notna().any()
        
        # Dominant surface type
        surfaces = nearby_roads['surface'].dropna()
        if len(surfaces) > 0:
            from collections import Counter
            surface_counts = Counter(surfaces.tolist())
            features['dominant_surface'] = surface_counts.most_common(1)[0][0]
        else:
            features['dominant_surface'] = None
        
        # Highway type flags
        highway_types_present = set(nearby_roads['highway'].dropna().unique())
        for hw_type in HIGHWAY_TYPES:
            features[f'highway_{hw_type}'] = hw_type in highway_types_present
    
    # Check for nearby traffic signals
    if traffic_signals_gdf is not None and len(traffic_signals_gdf) > 0:
        signals_nearby = get_roads_near_junction(
            junction_point, traffic_signals_gdf, buffer_meters
        )
        features['has_traffic_signal'] = len(signals_nearby) > 0
    else:
        features['has_traffic_signal'] = False
    
    # Check for nearby crossings
    if crossings_gdf is not None and len(crossings_gdf) > 0:
        crossings_nearby = get_roads_near_junction(
            junction_point, crossings_gdf, buffer_meters
        )
        features['has_crossing'] = len(crossings_nearby) > 0
    else:
        features['has_crossing'] = False
    
    return features


def extract_features_for_junctions(
    junctions_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    traffic_signals_gdf: Optional[gpd.GeoDataFrame] = None,
    crossings_gdf: Optional[gpd.GeoDataFrame] = None,
    buffer_meters: float = 30.0,
) -> pd.DataFrame:
    """
    Extract aggregated features for all junctions.
    
    Args:
        junctions_gdf: GeoDataFrame of junction points
        roads_gdf: GeoDataFrame of road segments from ohsome
        traffic_signals_gdf: Optional GeoDataFrame of traffic signals
        crossings_gdf: Optional GeoDataFrame of crossings
        buffer_meters: Buffer radius for spatial queries
    
    Returns:
        DataFrame with one row per junction and feature columns
    """
    print(f"Extracting features for {len(junctions_gdf)} junctions...")
    
    # First, extract and normalize road features
    roads_with_features = extract_road_features(roads_gdf)
    
    # Extract features for each junction
    all_features = []
    
    for idx, junction in junctions_gdf.iterrows():
        junction_id = junction.get('junction_id', idx)
        junction_point = junction.geometry
        
        features = aggregate_junction_features(
            junction_id=junction_id,
            junction_point=junction_point,
            roads_gdf=roads_with_features,
            traffic_signals_gdf=traffic_signals_gdf,
            crossings_gdf=crossings_gdf,
            buffer_meters=buffer_meters,
        )
        
        all_features.append(features)
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(junctions_gdf)} junctions")
    
    features_df = pd.DataFrame(all_features)
    print(f"  Extracted {len(features_df.columns)} features per junction")
    
    return features_df


def extract_features_for_year(
    junctions_gdf: gpd.GeoDataFrame,
    yearly_infrastructure: Dict[str, gpd.GeoDataFrame],
    year: int,
    buffer_meters: float = 30.0,
) -> pd.DataFrame:
    """
    Extract features for all junctions for a specific year.
    
    Args:
        junctions_gdf: GeoDataFrame of junction points
        yearly_infrastructure: Dict with 'roads', 'traffic_signals', 'crossings'
        year: Year of the snapshot
        buffer_meters: Buffer radius for spatial queries
    
    Returns:
        DataFrame with junction features for that year
    """
    print(f"\n{'='*40}")
    print(f"Extracting features for year {year}")
    print(f"{'='*40}")
    
    features_df = extract_features_for_junctions(
        junctions_gdf=junctions_gdf,
        roads_gdf=yearly_infrastructure.get('roads', gpd.GeoDataFrame()),
        traffic_signals_gdf=yearly_infrastructure.get('traffic_signals'),
        crossings_gdf=yearly_infrastructure.get('crossings'),
        buffer_meters=buffer_meters,
    )
    
    # Add year column
    features_df['year'] = year
    
    return features_df

