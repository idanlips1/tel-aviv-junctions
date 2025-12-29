"""
Configuration constants for Tel Aviv junction extraction and analysis.
"""

# Coordinate Reference Systems
UTM_CRS = "EPSG:32636"  # UTM Zone 36N for Israel (meters)
WGS84_CRS = "EPSG:4326"  # Standard lat/lon

# Tel Aviv bounding box (approximate municipal boundaries)
TEL_AVIV_BBOX = (34.74, 32.04, 34.82, 32.12)  # (min_lon, min_lat, max_lon, max_lat)
TEL_AVIV_PLACE_NAME = "Tel Aviv-Yafo, Israel"

# Temporal configuration
YEARS = range(2015, 2025)  # 2015 through 2024
SNAPSHOT_DATE_FORMAT = "{year}-01-01"  # January 1st of each year

# DBSCAN clustering parameters
DBSCAN_EPS_METERS = 25.0  # Max distance between points in same cluster
DBSCAN_MIN_SAMPLES = 1     # Keep isolated junctions as single-point clusters

# Junction extraction
MIN_JUNCTION_DEGREE = 3  # Minimum connections to be considered a junction

# Spatial join parameters
DEFAULT_ACCIDENT_RADIUS_METERS = 50.0  # Max distance to assign accident to junction

# ohsome API configuration
OHSOME_API_BASE = "https://api.ohsome.org/v1"
OHSOME_ELEMENTS_GEOMETRY = f"{OHSOME_API_BASE}/elements/geometry"

# OSM filters for ohsome queries
ROAD_FILTER = "highway=* and type:way"
CYCLEWAY_FILTER = "(highway=cycleway or cycleway=*) and type:way"
TRAFFIC_SIGNAL_FILTER = "highway=traffic_signals and type:node"
CROSSING_FILTER = "highway=crossing and type:node"

# Feature extraction - road types to track
HIGHWAY_TYPES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential",
    "unclassified",
    "living_street",
    "pedestrian",
    "cycleway",
    "path",
    "footway",
]

# Surface types
SURFACE_TYPES = [
    "asphalt", "paved", "concrete", "paving_stones",
    "cobblestone", "gravel", "unpaved", "dirt", "sand",
]

# Cycleway types
CYCLEWAY_TYPES = [
    "lane", "track", "shared_lane", "share_busway",
    "opposite", "opposite_lane", "opposite_track",
]

# Output directories
OUTPUT_DIR = "output"
CACHE_DIR = "cache"

