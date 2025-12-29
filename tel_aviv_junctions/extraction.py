"""
Junction extraction and clustering from OpenStreetMap data.

This module handles downloading road networks via OSMnx and identifying
junction nodes (intersections) using graph degree analysis and DBSCAN clustering.
"""

import os
from typing import Tuple, Dict

import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point

from tel_aviv_junctions.config import (
    UTM_CRS,
    WGS84_CRS,
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
    MIN_JUNCTION_DEGREE,
    TEL_AVIV_PLACE_NAME,
    CACHE_DIR,
)


def download_road_network(
    place_name: str,
    network_type: str,
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Download a road network for a given place using OSMnx.
    
    Args:
        place_name: Name of the place to query (uses administrative boundary)
        network_type: Type of network ('drive', 'bike', 'walk', 'all')
    
    Returns:
        Tuple of (graph, nodes_gdf, edges_gdf)
    """
    print(f"Downloading {network_type} network for: {place_name}")
    print("This may take a few minutes...")
    
    graph = ox.graph_from_place(
        place_name,
        network_type=network_type,
        simplify=True
    )
    
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)
    print(f"Downloaded {len(nodes_gdf)} nodes and {len(edges_gdf)} edges")
    
    return graph, nodes_gdf, edges_gdf


def download_combined_network(
    place_name: str = TEL_AVIV_PLACE_NAME,
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Download both drivable roads and cycleways, then combine them.
    
    Args:
        place_name: Name of the place to query
    
    Returns:
        Tuple of (combined_graph, nodes_gdf, edges_gdf, drive_edges, bike_edges)
    """
    print("=" * 60)
    print("DOWNLOADING NETWORKS")
    print("=" * 60)
    
    drive_graph, drive_nodes, drive_edges = download_road_network(place_name, 'drive')
    drive_edges = drive_edges.copy()
    drive_edges['network_type'] = 'drive'
    
    bike_graph, bike_nodes, bike_edges = download_road_network(place_name, 'bike')
    bike_edges = bike_edges.copy()
    bike_edges['network_type'] = 'bike'
    
    combined_graph = nx.compose(drive_graph, bike_graph)
    combined_nodes, combined_edges = ox.graph_to_gdfs(combined_graph)
    
    drive_edge_set = set(drive_edges.index)
    bike_edge_set = set(bike_edges.index)
    
    def get_network_type(idx):
        in_drive = idx in drive_edge_set
        in_bike = idx in bike_edge_set
        if in_drive and in_bike:
            return 'both'
        elif in_drive:
            return 'drive'
        elif in_bike:
            return 'bike'
        return 'unknown'
    
    combined_edges['network_type'] = [get_network_type(idx) for idx in combined_edges.index]
    
    print(f"\nCombined network: {len(combined_nodes)} nodes, {len(combined_edges)} edges")
    
    return combined_graph, combined_nodes, combined_edges, drive_edges, bike_edges


def calculate_node_degrees(graph: nx.MultiDiGraph) -> Dict[int, int]:
    """
    Calculate the degree (number of connections) for each node in the graph.
    
    Args:
        graph: NetworkX MultiDiGraph from OSMnx
    
    Returns:
        Dictionary mapping node_id to degree
    """
    undirected = graph.to_undirected()
    degrees = dict(undirected.degree())
    return degrees


def extract_junctions(
    graph: nx.MultiDiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    min_degree: int = MIN_JUNCTION_DEGREE,
) -> gpd.GeoDataFrame:
    """
    Extract junction nodes from the road network.
    
    A junction is defined as a node with degree >= min_degree, meaning it
    connects at least that many road segments.
    
    Args:
        graph: NetworkX MultiDiGraph from OSMnx
        nodes_gdf: GeoDataFrame of all nodes
        min_degree: Minimum degree to be considered a junction (default: 3)
    
    Returns:
        GeoDataFrame containing only junction nodes with columns:
        - junction_id: OSM node ID
        - latitude, longitude: Coordinates
        - degree: Number of connecting roads
        - geometry: Point geometry
    """
    print(f"Identifying junctions with degree >= {min_degree}...")
    
    degrees = calculate_node_degrees(graph)
    junction_ids = [node_id for node_id, degree in degrees.items() if degree >= min_degree]
    
    junctions_gdf = nodes_gdf.loc[junction_ids].copy()
    junctions_gdf['degree'] = junctions_gdf.index.map(degrees)
    
    junctions_gdf = junctions_gdf.reset_index()
    junctions_gdf = junctions_gdf.rename(columns={'osmid': 'junction_id'})
    
    junctions_gdf['latitude'] = junctions_gdf.geometry.y
    junctions_gdf['longitude'] = junctions_gdf.geometry.x
    
    output_columns = ['junction_id', 'latitude', 'longitude', 'degree', 'geometry']
    junctions_gdf = junctions_gdf[output_columns]
    
    print(f"Found {len(junctions_gdf)} raw junctions (before clustering)")
    
    return junctions_gdf


def cluster_junctions(
    junctions_gdf: gpd.GeoDataFrame,
    eps_meters: float = DBSCAN_EPS_METERS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> gpd.GeoDataFrame:
    """
    Cluster nearby junctions using DBSCAN to merge multi-node intersections.
    
    Complex intersections in OSM often have multiple nodes representing what is
    effectively a single real-world junction. This function groups nearby nodes
    and creates a single representative point for each cluster.
    
    Args:
        junctions_gdf: GeoDataFrame of raw junction nodes
        eps_meters: Maximum distance (in meters) between points in same cluster.
                    25m works well for typical urban intersections.
        min_samples: Minimum points to form a cluster. Use 1 to keep isolated junctions.
    
    Returns:
        GeoDataFrame with clustered junctions containing:
        - cluster_id: Cluster identifier
        - junction_id: Primary OSM node ID (highest degree node)
        - latitude, longitude: Centroid coordinates
        - degree: Maximum degree in cluster
        - max_degree, sum_degree: Aggregated degree statistics
        - node_count: Number of nodes merged
        - osm_node_ids: Comma-separated list of merged node IDs
        - geometry: Centroid point
    """
    print(f"\nClustering junctions with DBSCAN (eps={eps_meters}m, min_samples={min_samples})...")
    
    if len(junctions_gdf) == 0:
        return junctions_gdf
    
    # Project to UTM for accurate meter-based distance calculations
    junctions_utm = junctions_gdf.to_crs(UTM_CRS)
    
    # Extract coordinates for DBSCAN
    coords = np.array([(geom.x, geom.y) for geom in junctions_utm.geometry])
    
    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps_meters, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(coords)
    
    # Add cluster labels to the dataframe
    junctions_utm['cluster_id'] = cluster_labels
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"  Found {n_clusters} clusters")
    if n_noise > 0:
        print(f"  Noise points (isolated): {n_noise}")
    
    # Aggregate clusters into single representative junctions
    clustered_rows = []
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = junctions_utm['cluster_id'] == cluster_id
        cluster_points = junctions_utm[cluster_mask]
        
        # Calculate centroid of the cluster (in UTM)
        centroid_x = cluster_points.geometry.x.mean()
        centroid_y = cluster_points.geometry.y.mean()
        centroid_utm = Point(centroid_x, centroid_y)
        
        # Aggregate statistics
        max_degree = cluster_points['degree'].max()
        sum_degree = cluster_points['degree'].sum()
        node_count = len(cluster_points)
        osm_node_ids = ','.join(map(str, cluster_points['junction_id'].tolist()))
        
        # Use the junction_id of the highest-degree node as the primary ID
        primary_id = cluster_points.loc[cluster_points['degree'].idxmax(), 'junction_id']
        
        clustered_rows.append({
            'cluster_id': cluster_id if cluster_id >= 0 else f"isolated_{primary_id}",
            'junction_id': primary_id,
            'max_degree': max_degree,
            'sum_degree': sum_degree,
            'node_count': node_count,
            'osm_node_ids': osm_node_ids,
            'geometry': centroid_utm
        })
    
    # Create new GeoDataFrame with clustered junctions
    clustered_gdf = gpd.GeoDataFrame(clustered_rows, crs=UTM_CRS)
    
    # Project back to WGS84
    clustered_gdf = clustered_gdf.to_crs(WGS84_CRS)
    
    # Extract lat/lon from the centroid geometry
    clustered_gdf['latitude'] = clustered_gdf.geometry.y
    clustered_gdf['longitude'] = clustered_gdf.geometry.x
    
    # Use max_degree as the primary "degree" column for compatibility
    clustered_gdf['degree'] = clustered_gdf['max_degree']
    
    print(f"  Reduced {len(junctions_gdf)} raw junctions → {len(clustered_gdf)} clustered junctions")
    
    return clustered_gdf


def extract_clustered_junctions(
    place_name: str = TEL_AVIV_PLACE_NAME,
    eps_meters: float = DBSCAN_EPS_METERS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    min_degree: int = MIN_JUNCTION_DEGREE,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Full pipeline: download network, extract junctions, and cluster.
    
    This is the main entry point for junction extraction.
    
    Args:
        place_name: Name of place to extract junctions from
        eps_meters: DBSCAN epsilon (cluster radius in meters)
        min_samples: DBSCAN minimum samples per cluster
        min_degree: Minimum node degree to be considered a junction
    
    Returns:
        Tuple of:
        - clustered_junctions: GeoDataFrame of clustered junctions
        - raw_junctions: GeoDataFrame of raw (unclustered) junctions
        - drive_edges: GeoDataFrame of drivable road edges
        - bike_edges: GeoDataFrame of cycleway edges
    """
    # Download combined road + bike network
    combined_graph, combined_nodes, combined_edges, drive_edges, bike_edges = \
        download_combined_network(place_name)
    
    # Extract raw junctions
    print("\n" + "=" * 60)
    print("EXTRACTING JUNCTIONS")
    print("=" * 60)
    raw_junctions = extract_junctions(combined_graph, combined_nodes, min_degree)
    
    # Cluster junctions
    print("\n" + "=" * 60)
    print("CLUSTERING JUNCTIONS (DBSCAN)")
    print("=" * 60)
    clustered_junctions = cluster_junctions(raw_junctions, eps_meters, min_samples)
    
    return clustered_junctions, raw_junctions, drive_edges, bike_edges


def get_junctions_dataframe(
    place_name: str = TEL_AVIV_PLACE_NAME,
    clustered: bool = True,
) -> pd.DataFrame:
    """
    Get junctions as a pandas DataFrame (without geometry).
    
    This is a convenience function for ML workflows that don't need
    the full GeoDataFrame.
    
    Args:
        place_name: Name of place to extract junctions from
        clustered: If True, return clustered junctions; otherwise raw
    
    Returns:
        DataFrame with junction data (junction_id, lat, lon, degree, etc.)
    """
    clustered_gdf, raw_gdf, _, _ = extract_clustered_junctions(place_name)
    
    gdf = clustered_gdf if clustered else raw_gdf
    
    # Drop geometry column for plain DataFrame
    df = pd.DataFrame(gdf.drop(columns=['geometry']))
    
    return df


def get_junctions_geodataframe(
    place_name: str = TEL_AVIV_PLACE_NAME,
    clustered: bool = True,
) -> gpd.GeoDataFrame:
    """
    Get junctions as a GeoDataFrame with geometry.
    
    Use this function when you need spatial operations (e.g., joining
    accidents to junctions).
    
    Args:
        place_name: Name of place to extract junctions from
        clustered: If True, return clustered junctions; otherwise raw
    
    Returns:
        GeoDataFrame with junction geometry and attributes
    """
    clustered_gdf, raw_gdf, _, _ = extract_clustered_junctions(place_name)
    
    return clustered_gdf if clustered else raw_gdf


def export_to_csv(junctions_gdf: gpd.GeoDataFrame, output_path: str, clustered: bool = False) -> None:
    """Export junctions to CSV format."""
    if clustered:
        columns = ['junction_id', 'latitude', 'longitude', 'degree', 'node_count', 'osm_node_ids']
    else:
        columns = ['junction_id', 'latitude', 'longitude', 'degree']
    
    csv_df = junctions_gdf[[c for c in columns if c in junctions_gdf.columns]].copy()
    csv_df.to_csv(output_path, index=False)
    print(f"Exported CSV to: {output_path}")


def export_to_geojson(junctions_gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Export junctions to GeoJSON format."""
    gdf = junctions_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Exported GeoJSON to: {output_path}")

