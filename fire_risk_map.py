#!/usr/bin/env python3
"""
Australia Fire Risk Map - Single File Version (Optimized)

Fast fire risk calculation using Open-Meteo's batch API.
Defaults to Victoria state with 5km resolution.

Usage:
    python fire_risk_map.py --simulate --interactive
    python fire_risk_map.py --city Melbourne --interactive
    python fire_risk_map.py --region australia --resolution 100

Requirements:
    pip install requests numpy matplotlib tqdm folium
"""

import argparse
import csv
import json
import math
import random
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

REQUIRED_DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
]

# State/Region bounds
REGION_BOUNDS = {
    "victoria": {
        "min_lat": -39.2,
        "max_lat": -34.0,
        "min_lon": 141.0,
        "max_lon": 150.0,
    },
    "nsw": {
        "min_lat": -37.5,
        "max_lat": -28.2,
        "min_lon": 141.0,
        "max_lon": 153.6,
    },
    "queensland": {
        "min_lat": -29.0,
        "max_lat": -10.7,
        "min_lon": 138.0,
        "max_lon": 153.5,
    },
    "south_australia": {
        "min_lat": -38.1,
        "max_lat": -26.0,
        "min_lon": 129.0,
        "max_lon": 141.0,
    },
    "western_australia": {
        "min_lat": -35.1,
        "max_lat": -13.7,
        "min_lon": 113.0,
        "max_lon": 129.0,
    },
    "tasmania": {
        "min_lat": -43.6,
        "max_lat": -40.6,
        "min_lon": 144.5,
        "max_lon": 148.5,
    },
    "nt": {
        "min_lat": -26.0,
        "max_lat": -11.0,
        "min_lon": 129.0,
        "max_lon": 138.0,
    },
    "australia": {
        "min_lat": -44.0,
        "max_lat": -10.0,
        "min_lon": 113.0,
        "max_lon": 154.0,
    },
}

AUSTRALIAN_CITIES = {
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Perth": (-31.9505, 115.8605),
    "Adelaide": (-34.9285, 138.6007),
    "Darwin": (-12.4634, 130.8456),
    "Hobart": (-42.8821, 147.3272),
    "Canberra": (-35.2809, 149.1300),
    "Geelong": (-38.1499, 144.3617),
    "Ballarat": (-37.5622, 143.8503),
    "Bendigo": (-36.7570, 144.2794),
}

RISK_COLORS = {
    0: "#4CAF50", 1: "#2196F3", 2: "#FFEB3B",
    3: "#FF9800", 4: "#F44336", 5: "#9C27B0",
}

RISK_LABELS = {
    0: "Low-Moderate", 1: "High", 2: "Very High",
    3: "Severe", 4: "Extreme", 5: "Catastrophic",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GridPoint:
    lat: float
    lon: float
    index: int


@dataclass
class Grid:
    points: List[GridPoint]
    lats: np.ndarray
    lons: np.ndarray
    resolution_km: float
    n_rows: int
    n_cols: int


# =============================================================================
# GRID GENERATION
# =============================================================================

def km_to_degrees_lat(km: float) -> float:
    return km / 111.32


def km_to_degrees_lon(km: float, latitude: float) -> float:
    lat_rad = math.radians(abs(latitude))
    km_per_degree = 111.32 * math.cos(lat_rad)
    return km / max(1, km_per_degree)


def generate_grid(resolution_km: float = 5.0, bounds: dict = None) -> Grid:
    if bounds is None:
        bounds = REGION_BOUNDS["victoria"]

    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]

    lat_step = km_to_degrees_lat(resolution_km)
    avg_lat = (min_lat + max_lat) / 2
    lon_step = km_to_degrees_lon(resolution_km, avg_lat)

    lats = np.arange(min_lat, max_lat + lat_step, lat_step)
    lons = np.arange(min_lon, max_lon + lon_step, lon_step)

    n_rows, n_cols = len(lats), len(lons)

    points = [GridPoint(lat=lat, lon=lon, index=i)
              for i, (lat, lon) in enumerate((lat, lon) for lat in lats for lon in lons)]

    all_lats = np.array([p.lat for p in points])
    all_lons = np.array([p.lon for p in points])

    return Grid(points=points, lats=all_lats, lons=all_lons,
                resolution_km=resolution_km, n_rows=n_rows, n_cols=n_cols)


def get_city_grid(city_name: str, radius_km: float = 50.0, resolution_km: float = 5.0) -> Grid:
    if city_name not in AUSTRALIAN_CITIES:
        raise ValueError(f"Unknown city: {city_name}. Available: {', '.join(AUSTRALIAN_CITIES.keys())}")

    lat, lon = AUSTRALIAN_CITIES[city_name]
    half_h = km_to_degrees_lat(radius_km)
    half_w = km_to_degrees_lon(radius_km, lat)

    bounds = {"min_lat": lat - half_h, "max_lat": lat + half_h,
              "min_lon": lon - half_w, "max_lon": lon + half_w}
    return generate_grid(resolution_km=resolution_km, bounds=bounds)


# =============================================================================
# FIRE RISK CALCULATION (Vectorized for speed)
# =============================================================================

def calculate_ffdi_vectorized(temps: np.ndarray, humidity: np.ndarray,
                               winds: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Vectorized FFDI calculation - much faster than loops."""
    # Estimate drought factor from precipitation
    df = np.where(precip > 10, 2, np.where(precip > 5, 4, np.where(precip > 2, 6, np.where(precip > 0, 8, 10))))
    df = df.astype(float)

    # Clamp values
    temps = np.clip(temps, -10, 50)
    humidity = np.clip(humidity, 5, 100)
    winds = np.clip(winds, 0, 150)
    df = np.clip(df, 0.1, 10)

    # McArthur Mark 5 FFDI formula (vectorized)
    ffdi = 2.0 * np.exp(-0.450 + 0.987 * np.log(df) - 0.0345 * humidity + 0.0338 * temps + 0.0234 * winds)
    return np.maximum(0, ffdi)


def get_risk_levels(ffdi: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Vectorized risk level calculation."""
    levels = np.zeros(len(ffdi), dtype=int)
    levels[ffdi >= 12] = 1
    levels[ffdi >= 25] = 2
    levels[ffdi >= 50] = 3
    levels[ffdi >= 75] = 4
    levels[ffdi >= 100] = 5

    categories = [RISK_LABELS[l] for l in levels]
    return levels, categories


# =============================================================================
# WEATHER DATA FETCHING (Optimized with batch requests)
# =============================================================================

def fetch_weather_batch(lats: List[float], lons: List[float], forecast_days: int = 3) -> Optional[dict]:
    """Fetch weather for multiple locations in ONE API call (much faster)."""
    if not REQUESTS_AVAILABLE:
        return None

    # Open-Meteo accepts comma-separated coordinates
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat in lats),
        "longitude": ",".join(f"{lon:.4f}" for lon in lons),
        "daily": ",".join(REQUIRED_DAILY_VARS),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


def fetch_weather_parallel(grid: Grid, forecast_days: int = 3,
                           batch_size: int = 50, max_workers: int = 8) -> dict:
    """Fetch weather using parallel batch requests for maximum speed."""
    n_points = len(grid.points)
    all_temps = np.zeros(n_points)
    all_humidity = np.zeros(n_points)
    all_winds = np.zeros(n_points)
    all_precip = np.zeros(n_points)
    valid = np.zeros(n_points, dtype=bool)

    # Split into batches
    batches = []
    for i in range(0, n_points, batch_size):
        end = min(i + batch_size, n_points)
        batch_lats = [grid.points[j].lat for j in range(i, end)]
        batch_lons = [grid.points[j].lon for j in range(i, end)]
        batches.append((i, end, batch_lats, batch_lons))

    print(f"Fetching weather in {len(batches)} batches ({batch_size} points each)...")

    def fetch_batch(batch_info):
        start, end, lats, lons = batch_info
        data = fetch_weather_batch(lats, lons, forecast_days)
        return start, end, data

    # Parallel fetch
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, b): b for b in batches}
        pbar = tqdm(total=len(batches), desc="Fetching weather batches")
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)
        pbar.close()

    # Process results
    for start, end, data in results:
        if data is None:
            continue

        # Handle single vs multiple locations response
        if isinstance(data, list):
            for i, item in enumerate(data):
                idx = start + i
                daily = item.get("daily", {})
                if daily.get("temperature_2m_max"):
                    all_temps[idx] = daily["temperature_2m_max"][0] or 25
                    all_humidity[idx] = daily["relative_humidity_2m_min"][0] or 50
                    all_winds[idx] = daily["wind_speed_10m_max"][0] or 10
                    all_precip[idx] = daily["precipitation_sum"][0] or 0
                    valid[idx] = True
        else:
            # Single location or batch response
            daily = data.get("daily", {})
            if daily:
                # Check if it's a batch response (arrays of arrays)
                temps = daily.get("temperature_2m_max", [])
                if temps and isinstance(temps[0], list):
                    # Batch response - each location has its own array
                    for i in range(end - start):
                        idx = start + i
                        all_temps[idx] = temps[i][0] if temps[i] else 25
                        all_humidity[idx] = daily["relative_humidity_2m_min"][i][0] if daily["relative_humidity_2m_min"][i] else 50
                        all_winds[idx] = daily["wind_speed_10m_max"][i][0] if daily["wind_speed_10m_max"][i] else 10
                        all_precip[idx] = daily["precipitation_sum"][i][0] if daily["precipitation_sum"][i] else 0
                        valid[idx] = True
                elif temps:
                    # Single response for single location
                    all_temps[start] = temps[0] or 25
                    all_humidity[start] = daily.get("relative_humidity_2m_min", [50])[0] or 50
                    all_winds[start] = daily.get("wind_speed_10m_max", [10])[0] or 10
                    all_precip[start] = daily.get("precipitation_sum", [0])[0] or 0
                    valid[start] = True

    return {
        "temperatures": all_temps,
        "humidities": all_humidity,
        "wind_speeds": all_winds,
        "precipitations": all_precip,
        "valid_mask": valid,
    }


def generate_simulated_weather_fast(grid: Grid, seed: int = 42) -> dict:
    """Fast vectorized weather simulation."""
    np.random.seed(seed)
    n = len(grid.points)

    # Temperature varies with latitude
    lat_factor = (grid.lats + 44) / 34
    base_temps = 20 + lat_factor * 20
    temps = base_temps + np.random.normal(0, 5, n)

    # Humidity inversely related to latitude factor
    base_humidity = 80 - lat_factor * 30
    humidity = np.clip(base_humidity + np.random.uniform(-20, 10, n), 10, 100)

    # Wind speed
    winds = np.abs(np.random.normal(15, 10, n))

    # Precipitation (mostly dry)
    precip = np.where(np.random.random(n) < 0.2, np.random.uniform(0.1, 20, n), 0)

    return {
        "temperatures": temps,
        "humidities": humidity,
        "wind_speeds": winds,
        "precipitations": precip,
        "valid_mask": np.ones(n, dtype=bool),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_matplotlib_map(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                          ffdi_values: np.ndarray, title: str, output_path: str,
                          bounds: dict) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [RISK_COLORS[i] for i in range(6)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    scatter = ax.scatter(lons, lats, c=risk_levels, cmap=cmap, norm=norm, s=15, alpha=0.8, marker='s')

    ax.set_xlim(bounds["min_lon"] - 0.5, bounds["max_lon"] + 0.5)
    ax.set_ylim(bounds["min_lat"] - 0.5, bounds["max_lat"] + 0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(6), shrink=0.8)
    cbar.set_label("Fire Danger Rating")
    cbar.ax.set_yticklabels([RISK_LABELS[i] for i in range(6)])

    stats = f"FFDI: Min={ffdi_values.min():.1f}, Max={ffdi_values.max():.1f}, Mean={ffdi_values.mean():.1f}"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Map saved: {output_path}")


def create_folium_map(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                      ffdi_values: np.ndarray, risk_categories: List[str],
                      output_path: str, bounds: dict, use_heatmap: bool = False) -> None:
    if not FOLIUM_AVAILABLE:
        print("folium not available. Install: pip install folium")
        return

    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='cartodbpositron')

    # Always use heatmap-style for better visualization
    # Create a feature group for the colored grid
    from folium.plugins import HeatMap

    # Use HeatMap with FFDI values for smooth visualization
    heat_data = [[lat, lon, min(ffdi, 100)] for lat, lon, ffdi in zip(lats, lons, ffdi_values)]

    # Custom gradient matching fire danger colors
    gradient = {
        0.0: '#4CAF50',   # Green - Low
        0.12: '#4CAF50',  # Green - Low
        0.25: '#2196F3',  # Blue - High
        0.50: '#FFEB3B',  # Yellow - Very High
        0.75: '#FF9800',  # Orange - Severe
        0.90: '#F44336',  # Red - Extreme
        1.0: '#9C27B0'    # Purple - Catastrophic
    }

    HeatMap(heat_data, radius=12, blur=8, max_zoom=10, gradient=gradient).add_to(m)

    # Add legend
    legend = '''
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;background:white;
                padding:15px;border:2px solid gray;border-radius:8px;font-family:Arial;">
    <b style="font-size:14px;">Fire Danger Rating</b><br><br>
    '''
    for lvl, lbl in RISK_LABELS.items():
        legend += f'<div style="margin:3px 0;"><span style="background:{RISK_COLORS[lvl]};width:20px;height:14px;display:inline-block;margin-right:8px;border:1px solid #333;"></span>{lbl}</div>'
    legend += '<br><small>FFDI Max: {:.1f}</small>'.format(ffdi_values.max())
    legend += '</div>'
    m.get_root().html.add_child(folium.Element(legend))

    m.save(output_path)
    print(f"Interactive map saved: {output_path}")


def export_csv(lats, lons, risk_levels, ffdi_values, risk_categories, weather, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "ffdi", "risk_level", "risk", "temp_c", "humidity", "wind_kmh", "precip_mm"])
        for i in range(len(lats)):
            w.writerow([f"{lats[i]:.4f}", f"{lons[i]:.4f}", f"{ffdi_values[i]:.2f}",
                        risk_levels[i], risk_categories[i], f"{weather['temperatures'][i]:.1f}",
                        f"{weather['humidities'][i]:.1f}", f"{weather['wind_speeds'][i]:.1f}",
                        f"{weather['precipitations'][i]:.1f}"])
    print(f"CSV saved: {path}")


def export_json(lats, lons, risk_levels, ffdi_values, risk_categories, metadata, path):
    data = {
        "metadata": metadata,
        "stats": {"points": len(lats), "ffdi_min": float(ffdi_values.min()),
                  "ffdi_max": float(ffdi_values.max()), "ffdi_mean": float(ffdi_values.mean())},
        "points": [{"lat": float(lats[i]), "lon": float(lons[i]), "ffdi": float(ffdi_values[i]),
                    "risk": risk_categories[i]} for i in range(len(lats))]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved: {path}")


def print_summary(risk_levels, ffdi_values, title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    print(f"Points: {len(risk_levels)} | FFDI: {ffdi_values.min():.1f}-{ffdi_values.max():.1f} (mean: {ffdi_values.mean():.1f})")
    print("\nRisk Distribution:")
    for lvl in range(6):
        cnt = np.sum(risk_levels == lvl)
        pct = cnt / len(risk_levels) * 100
        print(f"  {RISK_LABELS[lvl]:15s}: {cnt:5d} ({pct:5.1f}%) {'#' * int(pct/2)}")
    if np.any(risk_levels >= 3):
        print(f"\n*** HIGH RISK AREAS: {np.sum(risk_levels >= 3)} points at Severe or above ***")
    print("=" * 50)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast fire risk calculator for Australia (defaults to Victoria)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fire_risk_map.py --simulate --interactive          # Victoria with simulated data
  python fire_risk_map.py --city Melbourne --interactive    # Melbourne region
  python fire_risk_map.py --region nsw --resolution 25      # NSW at 25km
  python fire_risk_map.py --region australia --resolution 100  # All Australia

Regions: victoria, nsw, queensland, south_australia, western_australia, tasmania, nt, australia
Cities: Sydney, Melbourne, Brisbane, Perth, Adelaide, Darwin, Hobart, Canberra, Geelong, Ballarat, Bendigo
        """)

    parser.add_argument("--region", "-r", default="victoria", choices=list(REGION_BOUNDS.keys()),
                        help="Region to analyze (default: victoria)")
    parser.add_argument("--resolution", type=float, default=5.0,
                        help="Grid resolution in km (default: 5km)")
    parser.add_argument("--city", "-c", choices=list(AUSTRALIAN_CITIES.keys()),
                        help="Focus on city region instead")
    parser.add_argument("--radius", type=float, default=50.0,
                        help="Radius around city in km (default: 50)")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--no-map", action="store_true", help="Skip PNG map")
    parser.add_argument("--csv", action="store_true", help="Export CSV")
    parser.add_argument("--json", action="store_true", help="Export JSON")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML map generation")
    parser.add_argument("--heatmap", action="store_true", help="Use heatmap style")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data (no API)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    parser.add_argument("--no-open", action="store_true", help="Don't open HTML in browser")

    args = parser.parse_args()

    print(f"\n{'='*60}\nFIRE RISK MAP - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*60}")

    # Determine bounds
    if args.city:
        region_name = args.city
        grid = get_city_grid(args.city, args.radius, args.resolution)
        bounds = {"min_lat": grid.lats.min(), "max_lat": grid.lats.max(),
                  "min_lon": grid.lons.min(), "max_lon": grid.lons.max()}
    else:
        region_name = args.region.upper()
        bounds = REGION_BOUNDS[args.region]
        grid = generate_grid(args.resolution, bounds)

    print(f"Region: {region_name} | Resolution: {args.resolution}km | Points: {len(grid.points)}")

    # Fetch weather
    print("\n--- Fetching Weather Data ---")
    start_time = time.time()

    if args.simulate:
        print("Using SIMULATED data (demo mode)")
        weather = generate_simulated_weather_fast(grid, args.seed)
    else:
        weather = fetch_weather_parallel(grid, forecast_days=3)

    fetch_time = time.time() - start_time
    valid_pct = np.sum(weather["valid_mask"]) / len(grid.points) * 100
    print(f"Completed in {fetch_time:.1f}s | Valid data: {valid_pct:.1f}%")

    # Calculate fire risk
    print("\n--- Calculating Fire Risk ---")
    ffdi = calculate_ffdi_vectorized(weather["temperatures"], weather["humidities"],
                                      weather["wind_speeds"], weather["precipitations"])
    risk_levels, risk_categories = get_risk_levels(ffdi)

    print_summary(risk_levels, ffdi, f"Fire Risk - {region_name}")

    # Generate outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"fire_risk_{region_name.lower()}_{ts}"

    print("\n--- Generating Outputs ---")

    if not args.no_map:
        create_matplotlib_map(grid.lats, grid.lons, risk_levels, ffdi,
                              f"Fire Risk - {region_name}", str(output_dir / f"{prefix}.png"), bounds)

    html_path = None
    if not args.no_html:
        html_path = str(output_dir / f"{prefix}.html")
        create_folium_map(grid.lats, grid.lons, risk_levels, ffdi, risk_categories,
                          html_path, bounds, args.heatmap)

    if args.csv:
        export_csv(grid.lats, grid.lons, risk_levels, ffdi, risk_categories, weather,
                   str(output_dir / f"{prefix}.csv"))

    if args.json:
        export_json(grid.lats, grid.lons, risk_levels, ffdi, risk_categories,
                    {"region": region_name, "resolution_km": args.resolution,
                     "simulated": args.simulate, "generated": datetime.now().isoformat()},
                    str(output_dir / f"{prefix}.json"))

    # Open HTML in browser
    if html_path and not args.no_open:
        print(f"\nOpening map in browser...")
        webbrowser.open(f"file://{Path(html_path).absolute()}")

    print(f"\n{'='*60}\nDONE! Output: {output_dir.absolute()}\n{'='*60}\n")


if __name__ == "__main__":
    main()
