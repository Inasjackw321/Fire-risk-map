#!/usr/bin/env python3
"""
Australia Fire Risk Map - Single File Version

A Python application that fetches weather data from Open-Meteo and calculates
fire risk across Australia using the McArthur Forest Fire Danger Index (FFDI).

Usage:
    python fire_risk_map.py --simulate --city Sydney --interactive
    python fire_risk_map.py --simulate --resolution 100
    python fire_risk_map.py --city Melbourne --csv --json

Requirements:
    pip install requests numpy pandas matplotlib aiohttp tqdm folium
"""

import argparse
import asyncio
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

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
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
]

AUSTRALIA_BOUNDS = {
    "min_lat": -44.0,
    "max_lat": -10.0,
    "min_lon": 113.0,
    "max_lon": 154.0,
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
}

RISK_COLORS = {
    0: "#4CAF50",  # Low-Moderate - Green
    1: "#2196F3",  # High - Blue
    2: "#FFEB3B",  # Very High - Yellow
    3: "#FF9800",  # Severe - Orange
    4: "#F44336",  # Extreme - Red
    5: "#9C27B0",  # Catastrophic - Purple
}

RISK_LABELS = {
    0: "Low-Moderate",
    1: "High",
    2: "Very High",
    3: "Severe",
    4: "Extreme",
    5: "Catastrophic",
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


@dataclass
class WeatherData:
    lat: float
    lon: float
    dates: List[str]
    temp_max: List[float]
    temp_min: List[float]
    humidity_max: List[float]
    humidity_min: List[float]
    wind_speed_max: List[float]
    precipitation: List[float]


@dataclass
class GridWeatherData:
    grid: Grid
    weather_data: List[WeatherData]
    fetch_date: str


@dataclass
class FireRiskResult:
    ffdi: float
    risk_category: str
    risk_level: int


# =============================================================================
# GRID GENERATION
# =============================================================================

def km_to_degrees_lat(km: float) -> float:
    return km / 111.32


def km_to_degrees_lon(km: float, latitude: float) -> float:
    lat_rad = math.radians(abs(latitude))
    km_per_degree = 111.32 * math.cos(lat_rad)
    if km_per_degree < 1:
        km_per_degree = 1
    return km / km_per_degree


def generate_australia_grid(resolution_km: float = 5.0, bounds: Optional[dict] = None) -> Grid:
    if bounds is None:
        bounds = AUSTRALIA_BOUNDS

    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]

    lat_step = km_to_degrees_lat(resolution_km)
    avg_lat = (min_lat + max_lat) / 2
    lon_step = km_to_degrees_lon(resolution_km, avg_lat)

    lats = np.arange(min_lat, max_lat + lat_step, lat_step)
    lons = np.arange(min_lon, max_lon + lon_step, lon_step)

    n_rows, n_cols = len(lats), len(lons)

    points = []
    index = 0
    for lat in lats:
        for lon in lons:
            points.append(GridPoint(lat=lat, lon=lon, index=index))
            index += 1

    all_lats = np.array([p.lat for p in points])
    all_lons = np.array([p.lon for p in points])

    return Grid(points=points, lats=all_lats, lons=all_lons,
                resolution_km=resolution_km, n_rows=n_rows, n_cols=n_cols)


def get_city_grid(city_name: str, radius_km: float = 50.0, resolution_km: float = 5.0) -> Grid:
    if city_name not in AUSTRALIAN_CITIES:
        available = ", ".join(AUSTRALIAN_CITIES.keys())
        raise ValueError(f"Unknown city: {city_name}. Available: {available}")

    lat, lon = AUSTRALIAN_CITIES[city_name]
    half_height_deg = km_to_degrees_lat(radius_km)
    half_width_deg = km_to_degrees_lon(radius_km, lat)

    bounds = {
        "min_lat": lat - half_height_deg,
        "max_lat": lat + half_height_deg,
        "min_lon": lon - half_width_deg,
        "max_lon": lon + half_width_deg,
    }
    return generate_australia_grid(resolution_km=resolution_km, bounds=bounds)


def estimate_grid_size(resolution_km: float, bounds: Optional[dict] = None) -> dict:
    if bounds is None:
        bounds = AUSTRALIA_BOUNDS

    lat_range = bounds["max_lat"] - bounds["min_lat"]
    lon_range = bounds["max_lon"] - bounds["min_lon"]
    avg_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2

    lat_step = km_to_degrees_lat(resolution_km)
    lon_step = km_to_degrees_lon(resolution_km, avg_lat)

    n_rows = int(lat_range / lat_step) + 1
    n_cols = int(lon_range / lon_step) + 1

    return {"n_rows": n_rows, "n_cols": n_cols, "total_points": n_rows * n_cols}


# =============================================================================
# FIRE RISK CALCULATION
# =============================================================================

def calculate_drought_factor(days_since_rain: int, rainfall_amount: float) -> float:
    if days_since_rain <= 0:
        return 0.0

    if rainfall_amount > 30:
        df = min(10, days_since_rain * 0.3)
    elif rainfall_amount > 15:
        df = min(10, days_since_rain * 0.5)
    elif rainfall_amount > 5:
        df = min(10, days_since_rain * 0.7)
    else:
        df = min(10, days_since_rain * 1.0)

    return max(0, min(10, df))


def calculate_ffdi(temperature: float, relative_humidity: float,
                   wind_speed: float, drought_factor: float) -> float:
    if drought_factor <= 0:
        return 0.0

    temp = max(-10, min(50, temperature))
    rh = max(5, min(100, relative_humidity))
    wind = max(0, min(150, wind_speed))
    df = max(0.1, min(10, drought_factor))

    try:
        ffdi = 2.0 * math.exp(
            -0.450 + 0.987 * math.log(df) - 0.0345 * rh + 0.0338 * temp + 0.0234 * wind
        )
    except (ValueError, OverflowError):
        ffdi = 0.0

    return max(0, ffdi)


def get_risk_category(ffdi: float) -> Tuple[str, int]:
    if ffdi < 12:
        return ("Low-Moderate", 0)
    elif ffdi < 25:
        return ("High", 1)
    elif ffdi < 50:
        return ("Very High", 2)
    elif ffdi < 75:
        return ("Severe", 3)
    elif ffdi < 100:
        return ("Extreme", 4)
    else:
        return ("Catastrophic", 5)


def calculate_fire_risk(temperature: float, relative_humidity: float,
                        wind_speed: float, precipitation_sum: float) -> FireRiskResult:
    if precipitation_sum > 10:
        days_since_rain = 0
    elif precipitation_sum > 5:
        days_since_rain = 2
    elif precipitation_sum > 2:
        days_since_rain = 5
    elif precipitation_sum > 0:
        days_since_rain = 7
    else:
        days_since_rain = 14

    df = calculate_drought_factor(days_since_rain, precipitation_sum)
    ffdi = calculate_ffdi(temperature, relative_humidity, wind_speed, df)
    category, level = get_risk_category(ffdi)

    return FireRiskResult(ffdi=ffdi, risk_category=category, risk_level=level)


def calculate_fire_risk_batch(temperatures: np.ndarray, humidities: np.ndarray,
                              wind_speeds: np.ndarray, precipitations: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n = len(temperatures)
    ffdi_values = np.zeros(n)
    risk_levels = np.zeros(n, dtype=int)
    risk_categories = []

    for i in range(n):
        result = calculate_fire_risk(temperatures[i], humidities[i],
                                     wind_speeds[i], precipitations[i])
        ffdi_values[i] = result.ffdi
        risk_levels[i] = result.risk_level
        risk_categories.append(result.risk_category)

    return ffdi_values, risk_levels, risk_categories


# =============================================================================
# WEATHER DATA FETCHING
# =============================================================================

def generate_simulated_weather(lat: float, lon: float, forecast_days: int = 3,
                               seed: Optional[int] = None) -> WeatherData:
    if seed is not None:
        location_seed = int((lat * 1000 + lon * 100) % 2**31)
        random.seed(location_seed + seed)

    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]

    lat_factor = (lat + 44) / 34
    base_temp = 20 + lat_factor * 20

    temp_max, temp_min, humidity_max, humidity_min = [], [], [], []
    wind_speed, precipitation = [], []

    for _ in range(forecast_days):
        day_temp_max = base_temp + random.gauss(0, 5) + random.uniform(-3, 3)
        day_temp_min = day_temp_max - random.uniform(8, 15)

        base_humidity = 80 - lat_factor * 30
        day_humidity_max = min(100, base_humidity + random.uniform(10, 20))
        day_humidity_min = max(10, base_humidity - random.uniform(20, 40))

        day_wind = abs(random.gauss(15, 10))
        day_precip = random.uniform(0.1, 20) if random.random() < 0.2 else 0.0

        temp_max.append(round(day_temp_max, 1))
        temp_min.append(round(day_temp_min, 1))
        humidity_max.append(round(day_humidity_max, 1))
        humidity_min.append(round(day_humidity_min, 1))
        wind_speed.append(round(day_wind, 1))
        precipitation.append(round(day_precip, 1))

    return WeatherData(lat=lat, lon=lon, dates=dates, temp_max=temp_max,
                       temp_min=temp_min, humidity_max=humidity_max,
                       humidity_min=humidity_min, wind_speed_max=wind_speed,
                       precipitation=precipitation)


def fetch_weather_single(lat: float, lon: float, forecast_days: int = 3) -> Optional[WeatherData]:
    if not REQUESTS_AVAILABLE:
        return None

    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(REQUIRED_DAILY_VARS),
        "forecast_days": forecast_days, "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})

        return WeatherData(
            lat=lat, lon=lon, dates=daily.get("time", []),
            temp_max=daily.get("temperature_2m_max", []),
            temp_min=daily.get("temperature_2m_min", []),
            humidity_max=daily.get("relative_humidity_2m_max", []),
            humidity_min=daily.get("relative_humidity_2m_min", []),
            wind_speed_max=daily.get("wind_speed_10m_max", []),
            precipitation=daily.get("precipitation_sum", []),
        )
    except Exception:
        return None


def fetch_grid_weather(grid: Grid, forecast_days: int = 3, simulate: bool = False,
                       simulation_seed: int = 42, progress_bar: bool = True) -> GridWeatherData:
    print(f"Fetching weather data for {len(grid.points)} grid points...")
    print(f"Forecast days: {forecast_days}")

    if simulate:
        print("Using SIMULATED weather data (demo mode)")
        iterator = tqdm(grid.points, desc="Generating weather") if progress_bar else grid.points
        weather_data = [generate_simulated_weather(p.lat, p.lon, forecast_days, simulation_seed)
                        for p in iterator]
    else:
        iterator = tqdm(grid.points, desc="Fetching weather") if progress_bar else grid.points
        weather_data = []
        for p in iterator:
            weather = fetch_weather_single(p.lat, p.lon, forecast_days)
            weather_data.append(weather)
            time.sleep(0.05)

    valid_weather = [w for w in weather_data if w is not None]
    print(f"Successfully fetched weather for {len(valid_weather)}/{len(grid.points)} points")

    return GridWeatherData(grid=grid, weather_data=weather_data,
                           fetch_date=datetime.now().isoformat())


def extract_day_weather(grid_weather: GridWeatherData, day_index: int = 0) -> Dict[str, np.ndarray]:
    n_points = len(grid_weather.grid.points)

    lats = np.zeros(n_points)
    lons = np.zeros(n_points)
    temps = np.zeros(n_points)
    humidities = np.zeros(n_points)
    winds = np.zeros(n_points)
    precips = np.zeros(n_points)
    valid_mask = np.zeros(n_points, dtype=bool)

    for i, (point, weather) in enumerate(zip(grid_weather.grid.points, grid_weather.weather_data)):
        lats[i], lons[i] = point.lat, point.lon

        if weather is not None and len(weather.temp_max) > day_index:
            temps[i] = weather.temp_max[day_index] if weather.temp_max[day_index] else 25.0
            humidities[i] = weather.humidity_min[day_index] if weather.humidity_min[day_index] else 50.0
            winds[i] = weather.wind_speed_max[day_index] if weather.wind_speed_max[day_index] else 10.0
            precips[i] = weather.precipitation[day_index] if weather.precipitation[day_index] else 0.0
            valid_mask[i] = True
        else:
            temps[i], humidities[i], winds[i], precips[i] = 25.0, 50.0, 10.0, 0.0

    return {"lats": lats, "lons": lons, "temperatures": temps, "humidities": humidities,
            "wind_speeds": winds, "precipitations": precips, "valid_mask": valid_mask}


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_matplotlib_map(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                          ffdi_values: np.ndarray, title: str = "Fire Risk Map",
                          output_path: Optional[str] = None) -> Optional[object]:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = [RISK_COLORS[i] for i in range(6)]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    scatter = ax.scatter(lons, lats, c=risk_levels, cmap=cmap, norm=norm, s=10, alpha=0.8, marker='s')

    ax.set_xlim(AUSTRALIA_BOUNDS["min_lon"] - 1, AUSTRALIA_BOUNDS["max_lon"] + 1)
    ax.set_ylim(AUSTRALIA_BOUNDS["min_lat"] - 1, AUSTRALIA_BOUNDS["max_lat"] + 1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(6), shrink=0.8)
    cbar.set_label("Fire Danger Rating")
    cbar.ax.set_yticklabels([RISK_LABELS[i] for i in range(6)])

    stats_text = f"FFDI Stats:\nMin: {ffdi_values.min():.1f}\nMax: {ffdi_values.max():.1f}\nMean: {ffdi_values.mean():.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Map saved to: {output_path}")

    return fig


def create_folium_map(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                      ffdi_values: np.ndarray, risk_categories: List[str],
                      output_path: str = "fire_risk_map.html", use_heatmap: bool = False) -> Optional[object]:
    if not FOLIUM_AVAILABLE:
        print("folium not available. Install with: pip install folium")
        return None

    center_lat = (AUSTRALIA_BOUNDS["min_lat"] + AUSTRALIA_BOUNDS["max_lat"]) / 2
    center_lon = (AUSTRALIA_BOUNDS["min_lon"] + AUSTRALIA_BOUNDS["max_lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='OpenStreetMap')

    if use_heatmap:
        heat_data = [[lat, lon, ffdi] for lat, lon, ffdi in zip(lats, lons, ffdi_values)]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)
    else:
        for i in range(len(lats)):
            color = RISK_COLORS[risk_levels[i]]
            folium.CircleMarker(
                location=[lats[i], lons[i]], radius=3, color=color, fill=True,
                fillColor=color, fillOpacity=0.7,
                popup=f"FFDI: {ffdi_values[i]:.1f}<br>Risk: {risk_categories[i]}"
            ).add_to(m)

    legend_html = '<div style="position:fixed;bottom:50px;left:50px;z-index:1000;background:white;padding:10px;border:2px solid gray;border-radius:5px;font-size:12px;"><b>Fire Danger Rating</b><br>'
    for level, label in RISK_LABELS.items():
        legend_html += f'<i style="background:{RISK_COLORS[level]};width:15px;height:15px;display:inline-block;margin-right:5px;"></i>{label}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")
    return m


def export_to_csv(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                  ffdi_values: np.ndarray, risk_categories: List[str],
                  weather_data: Optional[Dict] = None, output_path: str = "fire_risk_data.csv") -> None:
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["latitude", "longitude", "ffdi", "risk_level", "risk_category"]
        if weather_data:
            header.extend(["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm"])
        writer.writerow(header)

        for i in range(len(lats)):
            row = [f"{lats[i]:.4f}", f"{lons[i]:.4f}", f"{ffdi_values[i]:.2f}",
                   risk_levels[i], risk_categories[i]]
            if weather_data:
                row.extend([f"{weather_data['temperatures'][i]:.1f}",
                            f"{weather_data['humidities'][i]:.1f}",
                            f"{weather_data['wind_speeds'][i]:.1f}",
                            f"{weather_data['precipitations'][i]:.1f}"])
            writer.writerow(row)
    print(f"Data exported to: {output_path}")


def export_to_json(lats: np.ndarray, lons: np.ndarray, risk_levels: np.ndarray,
                   ffdi_values: np.ndarray, risk_categories: List[str],
                   metadata: Optional[Dict] = None, output_path: str = "fire_risk_data.json") -> None:
    data = {
        "metadata": metadata or {},
        "statistics": {
            "total_points": len(lats),
            "ffdi_min": float(ffdi_values.min()),
            "ffdi_max": float(ffdi_values.max()),
            "ffdi_mean": float(ffdi_values.mean()),
            "risk_distribution": {RISK_LABELS[i]: int(np.sum(risk_levels == i)) for i in range(6)}
        },
        "points": [{"lat": float(lats[i]), "lon": float(lons[i]), "ffdi": float(ffdi_values[i]),
                    "risk_level": int(risk_levels[i]), "risk_category": risk_categories[i]}
                   for i in range(len(lats))]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data exported to: {output_path}")


def print_risk_summary(risk_levels: np.ndarray, ffdi_values: np.ndarray, title: str = "Fire Risk Summary") -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"\nTotal grid points analyzed: {len(risk_levels)}")
    print(f"\nFFDI Statistics:")
    print(f"  Minimum: {ffdi_values.min():.2f}")
    print(f"  Maximum: {ffdi_values.max():.2f}")
    print(f"  Mean: {ffdi_values.mean():.2f}")
    print(f"  Median: {np.median(ffdi_values):.2f}")
    print(f"\nRisk Level Distribution:")
    for level in range(6):
        count = np.sum(risk_levels == level)
        pct = (count / len(risk_levels)) * 100
        bar = "#" * int(pct / 2)
        print(f"  {RISK_LABELS[level]:15s}: {count:6d} ({pct:5.1f}%) {bar}")

    high_risk_mask = risk_levels >= 3
    if np.any(high_risk_mask):
        print(f"\n*** High-risk areas (Severe or above): {np.sum(high_risk_mask)} points ***")
    print("=" * 50 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate fire risk across Australia using Open-Meteo weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fire_risk_map.py --simulate --city Sydney --interactive
  python fire_risk_map.py --simulate --resolution 100
  python fire_risk_map.py --city Melbourne --radius 75 --csv --json

Available cities: Sydney, Melbourne, Brisbane, Perth, Adelaide, Darwin, Hobart, Canberra
        """
    )

    parser.add_argument("--resolution", "-r", type=float, default=50.0,
                        help="Grid resolution in km (default: 50km)")
    parser.add_argument("--city", "-c", type=str, choices=list(AUSTRALIAN_CITIES.keys()),
                        help="Focus on specific city region (uses 5km resolution by default)")
    parser.add_argument("--radius", type=float, default=50.0,
                        help="Radius around city in km (default: 50km)")
    parser.add_argument("--days", "-d", type=int, default=3, choices=range(1, 17),
                        help="Number of forecast days (1-16, default: 3)")
    parser.add_argument("--day-index", type=int, default=0,
                        help="Which day to visualize (0=today, 1=tomorrow, etc.)")
    parser.add_argument("--output-dir", "-o", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--no-map", action="store_true", help="Skip PNG map generation")
    parser.add_argument("--csv", action="store_true", help="Export to CSV")
    parser.add_argument("--json", action="store_true", help="Export to JSON")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML map")
    parser.add_argument("--heatmap", action="store_true", help="Use heatmap style for interactive map")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated weather data (demo mode, no API required)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for simulation (default: 42)")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only estimate grid size without fetching data")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("AUSTRALIA FIRE RISK MAP GENERATOR")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine grid parameters
    if args.city:
        print(f"\nRegion: {args.city} (radius: {args.radius}km)")
        resolution = 5.0 if args.resolution == 50.0 else args.resolution
    else:
        print(f"\nRegion: Full Australia")
        resolution = args.resolution
    print(f"Resolution: {resolution}km")

    # Estimate grid size
    if args.city:
        grid = get_city_grid(args.city, args.radius, resolution)
        estimate = {"total_points": len(grid.points), "n_rows": grid.n_rows, "n_cols": grid.n_cols}
    else:
        estimate = estimate_grid_size(resolution)

    print(f"\nGrid size: {estimate.get('n_rows', 'N/A')} x {estimate.get('n_cols', 'N/A')}")
    print(f"Total points: {estimate['total_points']:,}")

    if args.estimate_only:
        print("\n(Estimate only mode - exiting)")
        return 0

    # Generate grid
    print("\n" + "-" * 40)
    print("Step 1: Generating grid...")
    if args.city:
        grid = get_city_grid(args.city, args.radius, resolution)
    else:
        grid = generate_australia_grid(resolution)
    print(f"Generated {len(grid.points)} grid points")

    # Fetch weather data
    print("\n" + "-" * 40)
    if args.simulate:
        print("Step 2: Generating simulated weather data (demo mode)...")
    else:
        print("Step 2: Fetching weather data from Open-Meteo...")
    grid_weather = fetch_grid_weather(grid, forecast_days=args.days,
                                       simulate=args.simulate, simulation_seed=args.seed)

    # Extract weather for specified day
    print("\n" + "-" * 40)
    print(f"Step 3: Extracting weather for day {args.day_index}...")
    day_weather = extract_day_weather(grid_weather, args.day_index)
    valid_count = np.sum(day_weather["valid_mask"])
    print(f"Valid weather data points: {valid_count}/{len(grid.points)}")

    # Calculate fire risk
    print("\n" + "-" * 40)
    print("Step 4: Calculating fire risk (FFDI)...")
    ffdi_values, risk_levels, risk_categories = calculate_fire_risk_batch(
        day_weather["temperatures"], day_weather["humidities"],
        day_weather["wind_speeds"], day_weather["precipitations"]
    )

    # Print summary
    day_label = ["Today", "Tomorrow", "Day 3"][args.day_index] if args.day_index < 3 else f"Day {args.day_index + 1}"
    title = f"Fire Risk Summary - {args.city or 'Australia'} - {day_label}"
    print_risk_summary(risk_levels, ffdi_values, title)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    region = args.city.lower() if args.city else "australia"

    # Generate outputs
    print("\n" + "-" * 40)
    print("Step 5: Generating outputs...")

    if not args.no_map:
        map_path = output_dir / f"fire_risk_{region}_{timestamp}.png"
        fig = create_matplotlib_map(day_weather["lats"], day_weather["lons"],
                                    risk_levels, ffdi_values,
                                    title=f"Fire Risk Map - {args.city or 'Australia'} - {day_label}",
                                    output_path=str(map_path))
        if fig:
            plt.close(fig)

    if args.interactive:
        html_path = output_dir / f"fire_risk_{region}_{timestamp}.html"
        create_folium_map(day_weather["lats"], day_weather["lons"],
                          risk_levels, ffdi_values, risk_categories,
                          output_path=str(html_path), use_heatmap=args.heatmap)

    if args.csv:
        csv_path = output_dir / f"fire_risk_{region}_{timestamp}.csv"
        export_to_csv(day_weather["lats"], day_weather["lons"],
                      risk_levels, ffdi_values, risk_categories,
                      weather_data=day_weather, output_path=str(csv_path))

    if args.json:
        json_path = output_dir / f"fire_risk_{region}_{timestamp}.json"
        metadata = {
            "region": args.city or "Australia",
            "resolution_km": resolution,
            "forecast_day": args.day_index,
            "generated_at": datetime.now().isoformat(),
            "data_source": "Simulated Data" if args.simulate else "Open-Meteo API",
            "simulated": args.simulate,
        }
        export_to_json(day_weather["lats"], day_weather["lons"],
                       risk_levels, ffdi_values, risk_categories,
                       metadata=metadata, output_path=str(json_path))

    print("\n" + "=" * 60)
    print("FIRE RISK ANALYSIS COMPLETE!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
