"""
Fire Risk Calculator Module

Implements the McArthur Forest Fire Danger Index (FFDI) and related fire risk calculations.
The FFDI is the standard fire danger rating system used in Australia.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class FireRiskResult:
    """Container for fire risk calculation results."""
    ffdi: float  # Forest Fire Danger Index
    risk_category: str  # Human-readable risk category
    risk_level: int  # 0-5 scale for visualization


def calculate_drought_factor(days_since_rain: int, rainfall_amount: float) -> float:
    """
    Calculate the Drought Factor (DF) based on rainfall history.

    The Drought Factor ranges from 0-10 and represents fuel availability
    based on recent rainfall patterns.

    Args:
        days_since_rain: Number of days since last significant rain (>2mm)
        rainfall_amount: Total rainfall in the last 20 days (mm)

    Returns:
        Drought factor value between 0 and 10
    """
    if days_since_rain <= 0:
        return 0.0

    # Simplified Keetch-Byram Drought Index approximation
    # In practice, this would use soil dryness data
    if rainfall_amount > 30:
        df = min(10, days_since_rain * 0.3)
    elif rainfall_amount > 15:
        df = min(10, days_since_rain * 0.5)
    elif rainfall_amount > 5:
        df = min(10, days_since_rain * 0.7)
    else:
        df = min(10, days_since_rain * 1.0)

    return max(0, min(10, df))


def calculate_ffdi(
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    drought_factor: float
) -> float:
    """
    Calculate the McArthur Forest Fire Danger Index (FFDI).

    The FFDI Mark 5 formula is:
    FFDI = 2 * exp(-0.450 + 0.987*ln(DF) - 0.0345*RH + 0.0338*T + 0.0234*V)

    Where:
        DF = Drought Factor (0-10)
        RH = Relative Humidity (%)
        T = Temperature (°C)
        V = Wind speed (km/h)

    Args:
        temperature: Air temperature in Celsius
        relative_humidity: Relative humidity percentage (0-100)
        wind_speed: Wind speed in km/h
        drought_factor: Drought factor (0-10)

    Returns:
        FFDI value (typically 0-100+, can exceed 100 in extreme conditions)
    """
    # Handle edge cases
    if drought_factor <= 0:
        return 0.0

    # Ensure reasonable bounds
    temp = max(-10, min(50, temperature))
    rh = max(5, min(100, relative_humidity))
    wind = max(0, min(150, wind_speed))
    df = max(0.1, min(10, drought_factor))  # Avoid log(0)

    try:
        # McArthur Mark 5 FFDI formula
        ffdi = 2.0 * math.exp(
            -0.450 +
            0.987 * math.log(df) -
            0.0345 * rh +
            0.0338 * temp +
            0.0234 * wind
        )
    except (ValueError, OverflowError):
        ffdi = 0.0

    return max(0, ffdi)


def get_risk_category(ffdi: float) -> tuple[str, int]:
    """
    Convert FFDI value to risk category and level.

    Australian Fire Danger Rating System categories:
    - 0-11: Low-Moderate (Green)
    - 12-24: High (Blue)
    - 25-49: Very High (Yellow)
    - 50-74: Severe (Orange)
    - 75-99: Extreme (Red)
    - 100+: Catastrophic (Purple/Black)

    Args:
        ffdi: Forest Fire Danger Index value

    Returns:
        Tuple of (category_name, risk_level 0-5)
    """
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


def calculate_fire_risk(
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    precipitation_sum: float,
    days_since_rain: Optional[int] = None
) -> FireRiskResult:
    """
    Calculate comprehensive fire risk from weather parameters.

    Args:
        temperature: Maximum temperature in Celsius
        relative_humidity: Minimum relative humidity (%)
        wind_speed: Maximum wind speed in km/h
        precipitation_sum: Total precipitation over recent period (mm)
        days_since_rain: Days since significant rain (if None, estimated from precipitation)

    Returns:
        FireRiskResult with FFDI, category, and risk level
    """
    # Estimate days since rain if not provided
    if days_since_rain is None:
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

    # Calculate drought factor
    df = calculate_drought_factor(days_since_rain, precipitation_sum)

    # Calculate FFDI
    ffdi = calculate_ffdi(temperature, relative_humidity, wind_speed, df)

    # Get risk category
    category, level = get_risk_category(ffdi)

    return FireRiskResult(ffdi=ffdi, risk_category=category, risk_level=level)


def calculate_fire_risk_batch(
    temperatures: np.ndarray,
    humidities: np.ndarray,
    wind_speeds: np.ndarray,
    precipitations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Calculate fire risk for arrays of weather data (vectorized for performance).

    Args:
        temperatures: Array of temperatures in Celsius
        humidities: Array of relative humidity values (%)
        wind_speeds: Array of wind speeds in km/h
        precipitations: Array of precipitation totals (mm)

    Returns:
        Tuple of (ffdi_array, risk_level_array, risk_categories_list)
    """
    n = len(temperatures)
    ffdi_values = np.zeros(n)
    risk_levels = np.zeros(n, dtype=int)
    risk_categories = []

    for i in range(n):
        result = calculate_fire_risk(
            temperature=temperatures[i],
            relative_humidity=humidities[i],
            wind_speed=wind_speeds[i],
            precipitation_sum=precipitations[i]
        )
        ffdi_values[i] = result.ffdi
        risk_levels[i] = result.risk_level
        risk_categories.append(result.risk_category)

    return ffdi_values, risk_levels, risk_categories
