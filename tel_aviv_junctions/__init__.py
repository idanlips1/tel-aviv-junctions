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

__version__ = "0.1.0"

__all__ = [
    # Config
    "TEL_AVIV_BBOX",
    "TEL_AVIV_PLACE_NAME", 
    "YEARS",
    "UTM_CRS",
    "WGS84_CRS",
    # Modules (will be populated as we add them)
    "extraction",
    "ohsome",
    "features",
    "spatial",
    "panel",
]

