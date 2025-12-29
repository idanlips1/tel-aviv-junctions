"""
Tel Aviv Junctions - OSM junction extraction with temporal feature support.

This package provides tools for:
- Extracting road junctions from OpenStreetMap data
- Querying historical OSM features via ohsome API
- Building panel datasets (junction × year) for ML
- Spatially joining accident data to junctions
"""

from tel_aviv_junctions.config import (
    TEL_AVIV_BBOX,
    TEL_AVIV_PLACE_NAME,
    YEARS,
    UTM_CRS,
    WGS84_CRS,
)

from tel_aviv_junctions.extraction import (
    get_junctions_dataframe,
    get_junctions_geodataframe,
)

from tel_aviv_junctions.spatial import (
    load_israeli_accident_data,
    accident_location_to_junction,
    load_and_transform_israeli_accidents,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "TEL_AVIV_BBOX",
    "TEL_AVIV_PLACE_NAME", 
    "YEARS",
    "UTM_CRS",
    "WGS84_CRS",
    # Extraction functions
    "get_junctions_dataframe",
    "get_junctions_geodataframe",
    # Spatial/accident functions
    "load_israeli_accident_data",
    "accident_location_to_junction",
    "load_and_transform_israeli_accidents",
    # Modules
    "extraction",
    "ohsome",
    "features",
    "spatial",
    "panel",
]

