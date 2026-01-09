"""
Weather Data Fetching Module

Fetches weather data from Open-Meteo API for fire risk calculations.
Uses async requests for efficient bulk data retrieval.
Includes simulation mode for testing without API access.
"""

import asyncio
import aiohttp
import requests
import time
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .grid import Grid, GridPoint


# Open-Meteo API endpoints
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Weather variables needed for fire risk calculation
REQUIRED_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum",
]


@dataclass
class WeatherData:
    """Container for weather data at a single location."""
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
    """Container for weather data across the entire grid."""
    grid: Grid
    weather_data: List[WeatherData]
    fetch_date: str


def generate_simulated_weather(
    lat: float,
    lon: float,
    forecast_days: int = 3,
    seed: Optional[int] = None
) -> WeatherData:
    """
    Generate simulated weather data for testing.

    The simulation creates realistic weather patterns based on:
    - Latitude (temperature varies with distance from equator)
    - Season simulation (random variation)
    - Spatial correlation (nearby points have similar weather)

    Args:
        lat: Latitude
        lon: Longitude
        forecast_days: Number of forecast days
        seed: Random seed for reproducibility

    Returns:
        WeatherData with simulated values
    """
    if seed is not None:
        # Use location-based seed for spatial consistency
        location_seed = int((lat * 1000 + lon * 100) % 2**31)
        random.seed(location_seed + seed)
        np.random.seed(location_seed + seed)

    from datetime import datetime, timedelta

    # Generate dates
    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]

    # Base temperature varies with latitude (Australia: -10 to -44)
    # Closer to equator (-10) = hotter, further south (-44) = cooler
    lat_factor = (lat + 44) / 34  # 0 at -44, 1 at -10
    base_temp = 20 + lat_factor * 20  # 20-40°C range

    # Add some random variation and daily fluctuation
    temp_max = []
    temp_min = []
    humidity_max = []
    humidity_min = []
    wind_speed = []
    precipitation = []

    for day in range(forecast_days):
        # Temperature with daily variation
        day_temp_max = base_temp + random.gauss(0, 5) + random.uniform(-3, 3)
        day_temp_min = day_temp_max - random.uniform(8, 15)

        # Humidity inversely related to temperature (roughly)
        base_humidity = 80 - lat_factor * 30  # 50-80% range
        day_humidity_max = min(100, base_humidity + random.uniform(10, 20))
        day_humidity_min = max(10, base_humidity - random.uniform(20, 40))

        # Wind speed (km/h)
        day_wind = abs(random.gauss(15, 10))

        # Precipitation (mostly dry with occasional rain)
        if random.random() < 0.2:  # 20% chance of rain
            day_precip = random.uniform(0.1, 20)
        else:
            day_precip = 0.0

        temp_max.append(round(day_temp_max, 1))
        temp_min.append(round(day_temp_min, 1))
        humidity_max.append(round(day_humidity_max, 1))
        humidity_min.append(round(day_humidity_min, 1))
        wind_speed.append(round(day_wind, 1))
        precipitation.append(round(day_precip, 1))

    return WeatherData(
        lat=lat,
        lon=lon,
        dates=dates,
        temp_max=temp_max,
        temp_min=temp_min,
        humidity_max=humidity_max,
        humidity_min=humidity_min,
        wind_speed_max=wind_speed,
        precipitation=precipitation,
    )


def fetch_weather_single(lat: float, lon: float, forecast_days: int = 3) -> Optional[WeatherData]:
    """
    Fetch weather data for a single location using synchronous request.

    Args:
        lat: Latitude
        lon: Longitude
        forecast_days: Number of forecast days (1-16)

    Returns:
        WeatherData object or None if request failed
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(REQUIRED_DAILY_VARS),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})

        return WeatherData(
            lat=lat,
            lon=lon,
            dates=daily.get("time", []),
            temp_max=daily.get("temperature_2m_max", []),
            temp_min=daily.get("temperature_2m_min", []),
            humidity_max=daily.get("relative_humidity_2m_max", []),
            humidity_min=daily.get("relative_humidity_2m_min", []),
            wind_speed_max=daily.get("wind_speed_10m_max", []),
            precipitation=daily.get("precipitation_sum", []),
        )
    except Exception as e:
        return None


async def fetch_weather_async(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    forecast_days: int = 3,
    semaphore: Optional[asyncio.Semaphore] = None,
    retries: int = 3
) -> Optional[WeatherData]:
    """
    Fetch weather data for a single location asynchronously.

    Args:
        session: aiohttp client session
        lat: Latitude
        lon: Longitude
        forecast_days: Number of forecast days
        semaphore: Optional semaphore for rate limiting
        retries: Number of retry attempts

    Returns:
        WeatherData object or None if request failed
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(REQUIRED_DAILY_VARS),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    async def do_fetch():
        for attempt in range(retries):
            try:
                async with session.get(
                    OPEN_METEO_FORECAST_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        daily = data.get("daily", {})

                        return WeatherData(
                            lat=lat,
                            lon=lon,
                            dates=daily.get("time", []),
                            temp_max=daily.get("temperature_2m_max", []),
                            temp_min=daily.get("temperature_2m_min", []),
                            humidity_max=daily.get("relative_humidity_2m_max", []),
                            humidity_min=daily.get("relative_humidity_2m_min", []),
                            wind_speed_max=daily.get("wind_speed_10m_max", []),
                            precipitation=daily.get("precipitation_sum", []),
                        )
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        await asyncio.sleep(1 + attempt)
                    else:
                        return None
            except asyncio.TimeoutError:
                await asyncio.sleep(0.5)
            except Exception:
                await asyncio.sleep(0.5)
        return None

    if semaphore:
        async with semaphore:
            return await do_fetch()
    else:
        return await do_fetch()


def fetch_weather_batch_sync(
    points: List[GridPoint],
    forecast_days: int = 3,
    delay_between_requests: float = 0.05,
    progress_bar: bool = True
) -> List[Optional[WeatherData]]:
    """
    Fetch weather data for multiple points synchronously with rate limiting.

    Args:
        points: List of grid points
        forecast_days: Number of forecast days
        delay_between_requests: Delay in seconds between requests
        progress_bar: Show progress bar

    Returns:
        List of WeatherData objects
    """
    results = []
    iterator = tqdm(points, desc="Fetching weather data") if progress_bar else points

    for point in iterator:
        weather = fetch_weather_single(point.lat, point.lon, forecast_days)
        results.append(weather)
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)

    return results


def fetch_weather_batch_simulated(
    points: List[GridPoint],
    forecast_days: int = 3,
    progress_bar: bool = True,
    seed: int = 42
) -> List[WeatherData]:
    """
    Generate simulated weather data for all points.

    Args:
        points: List of grid points
        forecast_days: Number of forecast days
        progress_bar: Show progress bar
        seed: Random seed for reproducibility

    Returns:
        List of WeatherData objects
    """
    results = []
    iterator = tqdm(points, desc="Generating simulated weather") if progress_bar else points

    for point in iterator:
        weather = generate_simulated_weather(point.lat, point.lon, forecast_days, seed)
        results.append(weather)

    return results


async def _fetch_all_async(
    points: List[GridPoint],
    forecast_days: int,
    max_concurrent: int,
    progress_bar: bool
) -> List[Optional[WeatherData]]:
    """Internal async function to fetch all weather data."""
    semaphore = asyncio.Semaphore(max_concurrent)

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_weather_async(session, p.lat, p.lon, forecast_days, semaphore)
            for p in points
        ]

        if progress_bar:
            results = []
            pbar = tqdm(total=len(tasks), desc="Fetching weather data")

            # Use gather to preserve order
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

            for result in gathered:
                if isinstance(result, Exception):
                    results.append(None)
                else:
                    results.append(result)
                pbar.update(1)
            pbar.close()
            return results
        else:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            return [None if isinstance(r, Exception) else r for r in gathered]


def fetch_grid_weather(
    grid: Grid,
    forecast_days: int = 3,
    use_async: bool = True,
    max_concurrent: int = 10,
    progress_bar: bool = True,
    simulate: bool = False,
    simulation_seed: int = 42
) -> GridWeatherData:
    """
    Fetch weather data for an entire grid.

    Args:
        grid: Grid object
        forecast_days: Number of forecast days (1-16)
        use_async: Use async requests for better performance
        max_concurrent: Max concurrent requests (for async)
        progress_bar: Show progress bar
        simulate: Use simulated weather data instead of API
        simulation_seed: Random seed for simulation reproducibility

    Returns:
        GridWeatherData object
    """
    from datetime import datetime

    print(f"Fetching weather data for {len(grid.points)} grid points...")
    print(f"Forecast days: {forecast_days}")

    if simulate:
        print("Using SIMULATED weather data (demo mode)")
        weather_data = fetch_weather_batch_simulated(
            grid.points,
            forecast_days,
            progress_bar=progress_bar,
            seed=simulation_seed
        )
    elif use_async:
        # Use asyncio.run() for clean event loop handling
        try:
            weather_data = asyncio.run(_fetch_all_async(
                grid.points,
                forecast_days,
                max_concurrent,
                progress_bar
            ))
        except RuntimeError as e:
            # If we're already in an event loop, fall back to sync
            if "running event loop" in str(e):
                print("Falling back to synchronous requests...")
                weather_data = fetch_weather_batch_sync(
                    grid.points,
                    forecast_days,
                    progress_bar=progress_bar
                )
            else:
                raise
    else:
        weather_data = fetch_weather_batch_sync(
            grid.points,
            forecast_days,
            progress_bar=progress_bar
        )

    # Filter out None results and create GridWeatherData
    valid_weather = [w for w in weather_data if w is not None]
    print(f"Successfully fetched weather for {len(valid_weather)}/{len(grid.points)} points")

    return GridWeatherData(
        grid=grid,
        weather_data=weather_data,
        fetch_date=datetime.now().isoformat()
    )


def extract_day_weather(
    grid_weather: GridWeatherData,
    day_index: int = 0
) -> Dict[str, np.ndarray]:
    """
    Extract weather data for a specific day across all grid points.

    Args:
        grid_weather: GridWeatherData object
        day_index: Day index (0 = today, 1 = tomorrow, etc.)

    Returns:
        Dictionary with arrays of weather values
    """
    n_points = len(grid_weather.grid.points)

    lats = np.zeros(n_points)
    lons = np.zeros(n_points)
    temps = np.zeros(n_points)
    humidities = np.zeros(n_points)
    winds = np.zeros(n_points)
    precips = np.zeros(n_points)
    valid_mask = np.zeros(n_points, dtype=bool)

    for i, (point, weather) in enumerate(zip(
        grid_weather.grid.points,
        grid_weather.weather_data
    )):
        lats[i] = point.lat
        lons[i] = point.lon

        if weather is not None and len(weather.temp_max) > day_index:
            temps[i] = weather.temp_max[day_index] if weather.temp_max[day_index] is not None else 25.0
            # Use minimum humidity for fire risk (worst case)
            humidities[i] = weather.humidity_min[day_index] if weather.humidity_min[day_index] is not None else 50.0
            winds[i] = weather.wind_speed_max[day_index] if weather.wind_speed_max[day_index] is not None else 10.0
            precips[i] = weather.precipitation[day_index] if weather.precipitation[day_index] is not None else 0.0
            valid_mask[i] = True
        else:
            # Default values for missing data
            temps[i] = 25.0
            humidities[i] = 50.0
            winds[i] = 10.0
            precips[i] = 0.0
            valid_mask[i] = False

    return {
        "lats": lats,
        "lons": lons,
        "temperatures": temps,
        "humidities": humidities,
        "wind_speeds": winds,
        "precipitations": precips,
        "valid_mask": valid_mask,
    }
