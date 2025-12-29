#!/usr/bin/env python3
"""
Extract road junctions (intersections) from OpenStreetMap data for Tel Aviv-Yafo, Israel.

This script:
1. Downloads the drivable road network AND cycleways for Tel Aviv-Yafo
2. Identifies junctions as nodes with degree >= 3 (excluding dead ends and mid-road nodes)
3. Clusters nearby junctions using DBSCAN to merge multi-node intersections
4. Exports results to CSV, GeoJSON, and interactive HTML map

This is now a thin CLI wrapper around the tel_aviv_junctions package modules.
"""

import os
import folium

from tel_aviv_junctions.config import (
    TEL_AVIV_PLACE_NAME,
    OUTPUT_DIR,
    DBSCAN_EPS_METERS,
    DBSCAN_MIN_SAMPLES,
)
from tel_aviv_junctions.extraction import (
    extract_clustered_junctions,
    export_to_csv,
    export_to_geojson,
)


def create_interactive_map(
    junctions_gdf,
    drive_edges,
    bike_edges,
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
        
        if clustered:
            node_count = row.get('node_count', 1)
            import numpy as np
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
    junctions_gdf,
    drive_edges,
    bike_edges,
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
    """Main execution function - CLI wrapper around tel_aviv_junctions package."""
    # Configuration
    place_name = TEL_AVIV_PLACE_NAME
    output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract junctions using the new module
    clustered_junctions_gdf, raw_junctions_gdf, drive_edges, bike_edges = \
        extract_clustered_junctions(
            place_name=place_name,
            eps_meters=DBSCAN_EPS_METERS,
            min_samples=DBSCAN_MIN_SAMPLES,
        )
    
    raw_count = len(raw_junctions_gdf)
    
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
