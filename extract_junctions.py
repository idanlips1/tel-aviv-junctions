#!/usr/bin/env python3
"""
Extract road junctions (intersections) from OpenStreetMap data for Tel Aviv-Yafo, Israel.

This script:
1. Downloads the drivable road network AND cycleways for Tel Aviv-Yafo
2. Identifies junctions as nodes with degree >= 3 (excluding dead ends and mid-road nodes)
3. Clusters nearby junctions using DBSCAN to merge multi-node intersections
4. Exports results to CSV, GeoJSON, and interactive HTML map
"""

import os
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from sklearn.cluster import DBSCAN
from shapely.geometry import Point


# UTM Zone 36N for Israel (meters-based CRS for accurate distance calculations)
UTM_CRS = "EPSG:32636"
WGS84_CRS = "EPSG:4326"


def download_road_network(place_name: str, network_type: str) -> tuple:
    """
    Download a road network for a given place using OSMnx.
    
    Args:
        place_name: Name of the place to query (will use administrative boundary)
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


def download_combined_network(place_name: str) -> tuple:
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


def calculate_node_degrees(graph) -> dict:
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


def extract_junctions(graph, nodes_gdf: gpd.GeoDataFrame, min_degree: int = 3) -> gpd.GeoDataFrame:
    """
    Extract junction nodes from the road network.
    
    Args:
        graph: NetworkX MultiDiGraph from OSMnx
        nodes_gdf: GeoDataFrame of all nodes
        min_degree: Minimum degree to be considered a junction (default: 3)
    
    Returns:
        GeoDataFrame containing only junction nodes
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


def cluster_junctions_dbscan(
    junctions_gdf: gpd.GeoDataFrame,
    eps_meters: float = 25.0,
    min_samples: int = 1
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
        GeoDataFrame with clustered junctions, one row per cluster
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


def create_interactive_map(
    junctions_gdf: gpd.GeoDataFrame,
    drive_edges: gpd.GeoDataFrame,
    bike_edges: gpd.GeoDataFrame,
    output_path: str,
    clustered: bool = False
) -> folium.Map:
    """
    Create an interactive Folium map showing junctions and road networks.
    
    Args:
        junctions_gdf: GeoDataFrame of junctions (raw or clustered)
        drive_edges: GeoDataFrame of drivable road edges
        bike_edges: GeoDataFrame of cycleway edges
        output_path: Path to save the HTML file
        clustered: Whether junctions have been clustered (affects popup content)
    
    Returns:
        Folium Map object
    """
    print("Creating interactive map...")
    
    center_lat = junctions_gdf['latitude'].mean()
    center_lon = junctions_gdf['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='CartoDB dark_matter'
    )
    
    folium.TileLayer('CartoDB positron', name='Light').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    
    drive_layer = folium.FeatureGroup(name='🚗 Drivable Roads', show=True)
    bike_layer = folium.FeatureGroup(name='🚴 Cycleways', show=True)
    junction_layer = folium.FeatureGroup(name='🔴 Junctions (All)', show=True)
    high_degree_layer = folium.FeatureGroup(name='⭐ Major Junctions (degree ≥ 4)', show=True)
    
    print("  Adding drivable roads...")
    for idx, row in drive_edges.iterrows():
        if row.geometry is not None:
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(
                coords,
                color='#3388ff',
                weight=2,
                opacity=0.6
            ).add_to(drive_layer)
    
    print("  Adding cycleways...")
    for idx, row in bike_edges.iterrows():
        if row.geometry is not None:
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(
                coords,
                color='#2ecc71',
                weight=3,
                opacity=0.8
            ).add_to(bike_layer)
    
    def get_junction_color(degree):
        if degree >= 6:
            return '#e74c3c'
        elif degree >= 5:
            return '#e67e22'
        elif degree >= 4:
            return '#f1c40f'
        else:
            return '#9b59b6'
    
    print("  Adding junctions...")
    for idx, row in junctions_gdf.iterrows():
        if clustered:
            node_count = row.get('node_count', 1)
            osm_ids = row.get('osm_node_ids', str(row['junction_id']))
            # For clustered junctions, show additional info
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 180px;">
                <b>Clustered Junction</b><br>
                <b>Primary ID:</b> {row['junction_id']}<br>
                <b>Degree:</b> {row['degree']}<br>
                <b>Nodes merged:</b> {node_count}<br>
                <b>Latitude:</b> {row['latitude']:.6f}<br>
                <b>Longitude:</b> {row['longitude']:.6f}<br>
                <b>OSM Node IDs:</b><br>
                <small>{osm_ids}</small><br>
                <a href="https://www.openstreetmap.org/node/{row['junction_id']}" target="_blank">View primary on OSM</a>
            </div>
            """
            tooltip = f"Degree: {row['degree']} ({node_count} nodes)"
        else:
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 150px;">
                <b>Junction ID:</b> {row['junction_id']}<br>
                <b>Degree:</b> {row['degree']}<br>
                <b>Latitude:</b> {row['latitude']:.6f}<br>
                <b>Longitude:</b> {row['longitude']:.6f}<br>
                <a href="https://www.openstreetmap.org/node/{row['junction_id']}" target="_blank">View on OSM</a>
            </div>
            """
            tooltip = f"Degree: {row['degree']}"
        
        color = get_junction_color(row['degree'])
        radius = 4 + (row['degree'] - 3) * 2
        
        # For clustered junctions, make size also reflect node count
        if clustered:
            node_count = row.get('node_count', 1)
            radius = max(radius, 4 + np.log1p(node_count) * 3)
        
        marker = folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip
        )
        marker.add_to(junction_layer)
        
        if row['degree'] >= 4:
            marker_hd = folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius + 2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=tooltip
            )
            marker_hd.add_to(high_degree_layer)
    
    drive_layer.add_to(m)
    bike_layer.add_to(m)
    junction_layer.add_to(m)
    high_degree_layer.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 15px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">Legend</div>
        <div style="margin-bottom: 5px;">
            <span style="background-color: #3388ff; width: 20px; height: 3px; display: inline-block; margin-right: 5px;"></span>
            Drivable Roads
        </div>
        <div style="margin-bottom: 5px;">
            <span style="background-color: #2ecc71; width: 20px; height: 3px; display: inline-block; margin-right: 5px;"></span>
            Cycleways
        </div>
        <div style="margin-bottom: 10px; font-weight: bold;">Junction Degree:</div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: #9b59b6; width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
            Degree 3
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: #f1c40f; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
            Degree 4
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: #e67e22; width: 14px; height: 14px; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
            Degree 5
        </div>
        <div>
            <span style="background-color: #e74c3c; width: 16px; height: 16px; border-radius: 50%; display: inline-block; margin-right: 5px;"></span>
            Degree 6+
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    title_text = "Tel Aviv-Yafo Road Junctions & Cycleways (DBSCAN Clustered)" if clustered else "Tel Aviv-Yafo Road Junctions & Cycleways"
    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        padding: 10px 20px;
        font-family: Arial, sans-serif;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    ">
        {title_text}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_path)
    print(f"Exported interactive map to: {output_path}")
    
    return m


def print_summary(
    junctions_gdf: gpd.GeoDataFrame,
    drive_edges: gpd.GeoDataFrame,
    bike_edges: gpd.GeoDataFrame,
    raw_count: int = None,
    clustered: bool = False
) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if clustered and raw_count:
        print(f"Raw junctions (before clustering): {raw_count}")
        print(f"Clustered junctions: {len(junctions_gdf)}")
        print(f"Reduction: {raw_count - len(junctions_gdf)} junctions merged ({100 * (1 - len(junctions_gdf)/raw_count):.1f}%)")
        
        # Multi-node clusters
        multi_node = junctions_gdf[junctions_gdf['node_count'] > 1]
        print(f"\nMulti-node clusters: {len(multi_node)}")
        if len(multi_node) > 0:
            print(f"  Avg nodes per multi-cluster: {multi_node['node_count'].mean():.1f}")
            print(f"  Max nodes in a cluster: {multi_node['node_count'].max()}")
    else:
        print(f"Total junctions found: {len(junctions_gdf)}")
    
    print(f"\nDrivable road segments: {len(drive_edges)}")
    print(f"Cycleway segments: {len(bike_edges)}")
    
    print(f"\nDegree distribution:")
    degree_counts = junctions_gdf['degree'].value_counts().sort_index()
    for degree, count in degree_counts.items():
        print(f"  Degree {degree}: {count} junctions")
    
    print(f"\nBounding box:")
    print(f"  Latitude:  {junctions_gdf['latitude'].min():.6f} to {junctions_gdf['latitude'].max():.6f}")
    print(f"  Longitude: {junctions_gdf['longitude'].min():.6f} to {junctions_gdf['longitude'].max():.6f}")
    print("=" * 60)


def main():
    """Main execution function."""
    # Configuration
    place_name = "Tel Aviv-Yafo, Israel"
    output_dir = "output"
    
    # DBSCAN parameters
    DBSCAN_EPS_METERS = 25.0  # Max distance between points in same cluster
    DBSCAN_MIN_SAMPLES = 1    # Keep isolated junctions as single-point clusters
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download combined road + bike network
    combined_graph, combined_nodes, combined_edges, drive_edges, bike_edges = download_combined_network(place_name)
    
    # Extract raw junctions from the combined network
    print("\n" + "=" * 60)
    print("EXTRACTING JUNCTIONS")
    print("=" * 60)
    raw_junctions_gdf = extract_junctions(combined_graph, combined_nodes, min_degree=3)
    raw_count = len(raw_junctions_gdf)
    
    # Cluster junctions using DBSCAN
    print("\n" + "=" * 60)
    print("CLUSTERING JUNCTIONS (DBSCAN)")
    print("=" * 60)
    clustered_junctions_gdf = cluster_junctions_dbscan(
        raw_junctions_gdf,
        eps_meters=DBSCAN_EPS_METERS,
        min_samples=DBSCAN_MIN_SAMPLES
    )
    
    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING DATA")
    print("=" * 60)
    
    # Export clustered junctions
    csv_path = os.path.join(output_dir, "tel_aviv_junctions_clustered.csv")
    geojson_path = os.path.join(output_dir, "tel_aviv_junctions_clustered.geojson")
    map_path = os.path.join(output_dir, "tel_aviv_junctions_map.html")
    
    export_to_csv(clustered_junctions_gdf, csv_path, clustered=True)
    export_to_geojson(clustered_junctions_gdf, geojson_path)
    create_interactive_map(clustered_junctions_gdf, drive_edges, bike_edges, map_path, clustered=True)
    
    # Also export raw junctions for reference
    raw_csv_path = os.path.join(output_dir, "tel_aviv_junctions_raw.csv")
    raw_geojson_path = os.path.join(output_dir, "tel_aviv_junctions_raw.geojson")
    export_to_csv(raw_junctions_gdf, raw_csv_path, clustered=False)
    export_to_geojson(raw_junctions_gdf, raw_geojson_path)
    
    # Print summary
    print_summary(clustered_junctions_gdf, drive_edges, bike_edges, raw_count=raw_count, clustered=True)
    
    print("\nSample output - Clustered junctions (first 5 rows):")
    display_cols = ['junction_id', 'latitude', 'longitude', 'degree', 'node_count']
    print(clustered_junctions_gdf[display_cols].head().to_string(index=False))
    
    print(f"\n✅ Done! Open the map in your browser:")
    print(f"   file://{os.path.abspath(map_path)}")
    
    return clustered_junctions_gdf


if __name__ == "__main__":
    junctions = main()
