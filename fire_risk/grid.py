"""
Grid Generation Module

Creates geographic grids for Australia at specified resolutions.
Handles coordinate calculations and grid point generation.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Australia bounding box (approximate)
AUSTRALIA_BOUNDS = {
    "min_lat": -44.0,  # Tasmania southern tip
    "max_lat": -10.0,  # Cape York
    "min_lon": 113.0,  # Western Australia
    "max_lon": 154.0,  # Eastern coast
}

# Major Australian cities for reference
AUSTRALIAN_CITIES = {
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Perth": (-31.9505, 115.8605),
    "Adelaide": (-34.9285, 138.6007),
    "Darwin": (-12.4634, 130.8456),
    "Hobart": (-42.8821, 147.3272),
    "Canberra": (-35.2809, 149.1300),
}


@dataclass
class GridPoint:
    """Represents a single point in the grid."""
    lat: float
    lon: float
    index: int


@dataclass
class Grid:
    """Container for grid data."""
    points: List[GridPoint]
    lats: np.ndarray
    lons: np.ndarray
    resolution_km: float
    n_rows: int
    n_cols: int


def km_to_degrees_lat(km: float) -> float:
    """
    Convert kilometers to degrees of latitude.

    1 degree of latitude ≈ 111.32 km (roughly constant)

    Args:
        km: Distance in kilometers

    Returns:
        Equivalent distance in degrees latitude
    """
    return km / 111.32


def km_to_degrees_lon(km: float, latitude: float) -> float:
    """
    Convert kilometers to degrees of longitude at a given latitude.

    Longitude degrees vary with latitude:
    1 degree of longitude = 111.32 * cos(latitude) km

    Args:
        km: Distance in kilometers
        latitude: Latitude in degrees

    Returns:
        Equivalent distance in degrees longitude
    """
    lat_rad = math.radians(abs(latitude))
    km_per_degree = 111.32 * math.cos(lat_rad)
    if km_per_degree < 1:
        km_per_degree = 1  # Avoid division by very small numbers near poles
    return km / km_per_degree


def generate_australia_grid(
    resolution_km: float = 5.0,
    bounds: Optional[dict] = None
) -> Grid:
    """
    Generate a grid of points covering Australia at the specified resolution.

    Args:
        resolution_km: Grid spacing in kilometers (default 5km)
        bounds: Optional custom bounds dict with min_lat, max_lat, min_lon, max_lon

    Returns:
        Grid object containing all grid points
    """
    if bounds is None:
        bounds = AUSTRALIA_BOUNDS

    min_lat = bounds["min_lat"]
    max_lat = bounds["max_lat"]
    min_lon = bounds["min_lon"]
    max_lon = bounds["max_lon"]

    # Calculate step sizes
    lat_step = km_to_degrees_lat(resolution_km)

    # Use average latitude for longitude step calculation
    avg_lat = (min_lat + max_lat) / 2
    lon_step = km_to_degrees_lon(resolution_km, avg_lat)

    # Generate latitude and longitude arrays
    lats = np.arange(min_lat, max_lat + lat_step, lat_step)
    lons = np.arange(min_lon, max_lon + lon_step, lon_step)

    n_rows = len(lats)
    n_cols = len(lons)

    # Generate grid points
    points = []
    index = 0
    for lat in lats:
        for lon in lons:
            points.append(GridPoint(lat=lat, lon=lon, index=index))
            index += 1

    # Create coordinate arrays for all points
    all_lats = np.array([p.lat for p in points])
    all_lons = np.array([p.lon for p in points])

    return Grid(
        points=points,
        lats=all_lats,
        lons=all_lons,
        resolution_km=resolution_km,
        n_rows=n_rows,
        n_cols=n_cols
    )


def generate_subregion_grid(
    center_lat: float,
    center_lon: float,
    width_km: float = 100.0,
    height_km: float = 100.0,
    resolution_km: float = 5.0
) -> Grid:
    """
    Generate a grid for a specific region centered on given coordinates.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        width_km: Width of region in km
        height_km: Height of region in km
        resolution_km: Grid spacing in km

    Returns:
        Grid object for the subregion
    """
    # Calculate bounds from center
    half_height_deg = km_to_degrees_lat(height_km / 2)
    half_width_deg = km_to_degrees_lon(width_km / 2, center_lat)

    bounds = {
        "min_lat": center_lat - half_height_deg,
        "max_lat": center_lat + half_height_deg,
        "min_lon": center_lon - half_width_deg,
        "max_lon": center_lon + half_width_deg,
    }

    return generate_australia_grid(resolution_km=resolution_km, bounds=bounds)


def get_city_grid(
    city_name: str,
    radius_km: float = 50.0,
    resolution_km: float = 5.0
) -> Grid:
    """
    Generate a grid around a major Australian city.

    Args:
        city_name: Name of the city (Sydney, Melbourne, Brisbane, etc.)
        radius_km: Radius around city center in km
        resolution_km: Grid spacing in km

    Returns:
        Grid object for the city region
    """
    if city_name not in AUSTRALIAN_CITIES:
        available = ", ".join(AUSTRALIAN_CITIES.keys())
        raise ValueError(f"Unknown city: {city_name}. Available: {available}")

    lat, lon = AUSTRALIAN_CITIES[city_name]
    return generate_subregion_grid(
        center_lat=lat,
        center_lon=lon,
        width_km=radius_km * 2,
        height_km=radius_km * 2,
        resolution_km=resolution_km
    )


def estimate_grid_size(resolution_km: float, bounds: Optional[dict] = None) -> dict:
    """
    Estimate the number of grid points without generating the full grid.

    Args:
        resolution_km: Grid spacing in kilometers
        bounds: Optional custom bounds

    Returns:
        Dictionary with grid size information
    """
    if bounds is None:
        bounds = AUSTRALIA_BOUNDS

    lat_range = bounds["max_lat"] - bounds["min_lat"]
    lon_range = bounds["max_lon"] - bounds["min_lon"]

    avg_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2

    lat_step = km_to_degrees_lat(resolution_km)
    lon_step = km_to_degrees_lon(resolution_km, avg_lat)

    n_rows = int(lat_range / lat_step) + 1
    n_cols = int(lon_range / lon_step) + 1
    total_points = n_rows * n_cols

    return {
        "resolution_km": resolution_km,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_points": total_points,
        "lat_step_deg": lat_step,
        "lon_step_deg": lon_step,
    }


def chunk_grid_points(
    grid: Grid,
    chunk_size: int = 50
) -> List[List[GridPoint]]:
    """
    Split grid points into chunks for batch API requests.

    Args:
        grid: Grid object
        chunk_size: Maximum points per chunk

    Returns:
        List of point chunks
    """
    points = grid.points
    return [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]
