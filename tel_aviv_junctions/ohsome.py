"""
Historical OSM data extraction via ohsome API.

The ohsome API (https://ohsome.org) provides historical OpenStreetMap data
snapshots, allowing us to query what the road network looked like at any
point in time from 2007 to present.

This module handles:
- Querying road/cycleway geometries with tags at specific timestamps
- Caching results locally to avoid repeated API calls
- Building yearly snapshots for temporal analysis
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from tel_aviv_junctions.config import (
    OHSOME_ELEMENTS_GEOMETRY,
    TEL_AVIV_BBOX,
    YEARS,
    SNAPSHOT_DATE_FORMAT,
    CACHE_DIR,
    WGS84_CRS,
)


def _get_cache_path(cache_key: str) -> str:
    """Get the file path for a cached result."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"ohsome_{cache_key}.geojson")


def _compute_cache_key(params: Dict[str, Any]) -> str:
    """Compute a hash-based cache key from query parameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]


def _load_from_cache(cache_key: str) -> Optional[gpd.GeoDataFrame]:
    """Load cached GeoDataFrame if it exists."""
    cache_path = _get_cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            gdf = gpd.read_file(cache_path)
            print(f"  Loaded from cache: {cache_path}")
            return gdf
        except Exception as e:
            print(f"  Cache load failed: {e}")
    return None


def _save_to_cache(gdf: gpd.GeoDataFrame, cache_key: str) -> None:
    """Save GeoDataFrame to cache."""
    cache_path = _get_cache_path(cache_key)
    try:
        gdf.to_file(cache_path, driver="GeoJSON")
        print(f"  Saved to cache: {cache_path}")
    except Exception as e:
        print(f"  Cache save failed: {e}")


def query_ohsome(
    bbox: Tuple[float, float, float, float],
    osm_filter: str,
    timestamp: str,
    properties: str = "tags",
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Query the ohsome API for OSM elements at a specific point in time.
    
    Args:
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        osm_filter: ohsome filter string (e.g., "highway=* and type:way")
        timestamp: ISO date string (e.g., "2020-01-01")
        properties: Which properties to return ("tags", "metadata", or "unclipped")
        use_cache: Whether to use local caching
    
    Returns:
        GeoDataFrame with OSM elements and their tags
    """
    params = {
        "bboxes": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "filter": osm_filter,
        "time": timestamp,
        "properties": properties,
    }
    
    # Check cache first
    cache_key = _compute_cache_key(params)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached
    
    print(f"  Querying ohsome API for {timestamp}...")
    
    try:
        response = requests.post(
            OHSOME_ELEMENTS_GEOMETRY,
            data=params,
            headers={"Accept": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        
        if "features" not in data or len(data["features"]) == 0:
            print(f"  No features returned for {timestamp}")
            return gpd.GeoDataFrame(columns=["geometry"], crs=WGS84_CRS)
        
        # Convert to GeoDataFrame
        features = data["features"]
        geometries = []
        properties_list = []
        
        for feature in features:
            geom = shape(feature["geometry"])
            props = feature.get("properties", {})
            geometries.append(geom)
            properties_list.append(props)
        
        gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs=WGS84_CRS)
        
        print(f"  Retrieved {len(gdf)} features")
        
        # Cache the result
        if use_cache:
            _save_to_cache(gdf, cache_key)
        
        return gdf
        
    except requests.exceptions.RequestException as e:
        print(f"  API request failed: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=WGS84_CRS)
    except json.JSONDecodeError as e:
        print(f"  Failed to parse API response: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=WGS84_CRS)


def get_roads_at_time(
    bbox: Tuple[float, float, float, float] = TEL_AVIV_BBOX,
    timestamp: str = "2024-01-01",
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get all road features at a specific point in time.
    
    Args:
        bbox: Bounding box (default: Tel Aviv)
        timestamp: ISO date string
        use_cache: Whether to use local caching
    
    Returns:
        GeoDataFrame with road geometries and OSM tags
    """
    osm_filter = "highway=* and type:way"
    return query_ohsome(bbox, osm_filter, timestamp, use_cache=use_cache)


def get_traffic_signals_at_time(
    bbox: Tuple[float, float, float, float] = TEL_AVIV_BBOX,
    timestamp: str = "2024-01-01",
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get traffic signal locations at a specific point in time.
    
    Args:
        bbox: Bounding box (default: Tel Aviv)
        timestamp: ISO date string
        use_cache: Whether to use local caching
    
    Returns:
        GeoDataFrame with traffic signal point geometries
    """
    osm_filter = "highway=traffic_signals and type:node"
    return query_ohsome(bbox, osm_filter, timestamp, use_cache=use_cache)


def get_crossings_at_time(
    bbox: Tuple[float, float, float, float] = TEL_AVIV_BBOX,
    timestamp: str = "2024-01-01",
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get pedestrian crossing locations at a specific point in time.
    
    Args:
        bbox: Bounding box (default: Tel Aviv)
        timestamp: ISO date string
        use_cache: Whether to use local caching
    
    Returns:
        GeoDataFrame with crossing point geometries
    """
    osm_filter = "highway=crossing and type:node"
    return query_ohsome(bbox, osm_filter, timestamp, use_cache=use_cache)


def get_yearly_road_snapshots(
    bbox: Tuple[float, float, float, float] = TEL_AVIV_BBOX,
    years: range = YEARS,
    use_cache: bool = True,
) -> Dict[int, gpd.GeoDataFrame]:
    """
    Get road network snapshots for multiple years.
    
    Queries the ohsome API for road features on January 1st of each year.
    Results are cached locally to avoid repeated API calls.
    
    Args:
        bbox: Bounding box (default: Tel Aviv)
        years: Range of years to query (default: 2015-2024)
        use_cache: Whether to use local caching
    
    Returns:
        Dictionary mapping year -> GeoDataFrame of road features
    """
    print("=" * 60)
    print("FETCHING YEARLY ROAD SNAPSHOTS")
    print("=" * 60)
    
    yearly_data = {}
    
    for year in years:
        timestamp = SNAPSHOT_DATE_FORMAT.format(year=year)
        print(f"\nYear {year} ({timestamp}):")
        
        roads = get_roads_at_time(bbox, timestamp, use_cache)
        yearly_data[year] = roads
        
        if len(roads) > 0:
            print(f"  Total road segments: {len(roads)}")
    
    return yearly_data


def get_yearly_infrastructure_snapshots(
    bbox: Tuple[float, float, float, float] = TEL_AVIV_BBOX,
    years: range = YEARS,
    use_cache: bool = True,
) -> Dict[int, Dict[str, gpd.GeoDataFrame]]:
    """
    Get comprehensive infrastructure snapshots for multiple years.
    
    For each year, fetches:
    - Roads (all highway types)
    - Traffic signals
    - Crossings
    
    Args:
        bbox: Bounding box (default: Tel Aviv)
        years: Range of years to query (default: 2015-2024)
        use_cache: Whether to use local caching
    
    Returns:
        Dictionary mapping year -> {
            'roads': GeoDataFrame,
            'traffic_signals': GeoDataFrame,
            'crossings': GeoDataFrame
        }
    """
    print("=" * 60)
    print("FETCHING YEARLY INFRASTRUCTURE SNAPSHOTS")
    print("=" * 60)
    
    yearly_data = {}
    
    for year in years:
        timestamp = SNAPSHOT_DATE_FORMAT.format(year=year)
        print(f"\n{'='*40}")
        print(f"Year {year} ({timestamp})")
        print(f"{'='*40}")
        
        yearly_data[year] = {
            'roads': get_roads_at_time(bbox, timestamp, use_cache),
            'traffic_signals': get_traffic_signals_at_time(bbox, timestamp, use_cache),
            'crossings': get_crossings_at_time(bbox, timestamp, use_cache),
        }
        
        roads_count = len(yearly_data[year]['roads'])
        signals_count = len(yearly_data[year]['traffic_signals'])
        crossings_count = len(yearly_data[year]['crossings'])
        
        print(f"  Summary: {roads_count} roads, {signals_count} signals, {crossings_count} crossings")
    
    return yearly_data


def extract_tags_from_gdf(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract OSM tags from a GeoDataFrame into a flat DataFrame.
    
    The ohsome API returns tags nested in the properties. This function
    flattens them for easier analysis.
    
    Args:
        gdf: GeoDataFrame from ohsome query
    
    Returns:
        DataFrame with flattened tag columns
    """
    if len(gdf) == 0:
        return pd.DataFrame()
    
    # ohsome returns tags in an '@osmId' and '@tags' structure sometimes
    # We need to handle various formats
    
    records = []
    for idx, row in gdf.iterrows():
        record = {'geometry': row.geometry}
        
        # Add all non-geometry columns
        for col in gdf.columns:
            if col != 'geometry':
                val = row[col]
                if isinstance(val, dict):
                    # Flatten nested dict (e.g., tags)
                    for k, v in val.items():
                        record[k] = v
                else:
                    record[col] = val
        
        records.append(record)
    
    result_gdf = gpd.GeoDataFrame(records, crs=gdf.crs)
    return result_gdf

