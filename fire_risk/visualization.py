"""
Visualization Module

Creates visual outputs for fire risk data including:
- Static matplotlib maps
- Interactive folium maps
- CSV/JSON exports
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from .grid import Grid, AUSTRALIA_BOUNDS


# Fire risk color scheme (Australian Fire Danger Rating System)
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


def create_matplotlib_map(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_levels: np.ndarray,
    ffdi_values: np.ndarray,
    title: str = "Australia Fire Risk Map",
    output_path: Optional[str] = None,
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Create a static matplotlib visualization of fire risk.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        risk_levels: Array of risk levels (0-5)
        ffdi_values: Array of FFDI values
        title: Plot title
        output_path: Path to save figure (optional)
        show_colorbar: Whether to show the FFDI colorbar
        figsize: Figure size in inches

    Returns:
        matplotlib Figure object or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap for risk levels
    colors = [RISK_COLORS[i] for i in range(6)]
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Scatter plot of risk levels
    scatter = ax.scatter(
        lons, lats,
        c=risk_levels,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.8,
        marker='s'
    )

    # Add Australia coastline approximation (simple box for now)
    ax.set_xlim(AUSTRALIA_BOUNDS["min_lon"] - 1, AUSTRALIA_BOUNDS["max_lon"] + 1)
    ax.set_ylim(AUSTRALIA_BOUNDS["min_lat"] - 1, AUSTRALIA_BOUNDS["max_lat"] + 1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add colorbar with risk labels
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(6), shrink=0.8)
        cbar.set_label("Fire Danger Rating")
        cbar.ax.set_yticklabels([RISK_LABELS[i] for i in range(6)])

    # Add FFDI statistics text box
    stats_text = (
        f"FFDI Statistics:\n"
        f"Min: {ffdi_values.min():.1f}\n"
        f"Max: {ffdi_values.max():.1f}\n"
        f"Mean: {ffdi_values.mean():.1f}\n"
        f"Points: {len(ffdi_values)}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Map saved to: {output_path}")

    return fig


def create_folium_map(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_levels: np.ndarray,
    ffdi_values: np.ndarray,
    risk_categories: List[str],
    output_path: str = "fire_risk_map.html",
    use_heatmap: bool = False
) -> Optional[object]:
    """
    Create an interactive Folium map of fire risk.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        risk_levels: Array of risk levels (0-5)
        ffdi_values: Array of FFDI values
        risk_categories: List of risk category names
        output_path: Path to save HTML file
        use_heatmap: Use heatmap instead of markers

    Returns:
        Folium Map object or None if folium not available
    """
    if not FOLIUM_AVAILABLE:
        print("folium not available. Install with: pip install folium")
        return None

    # Center map on Australia
    center_lat = (AUSTRALIA_BOUNDS["min_lat"] + AUSTRALIA_BOUNDS["max_lat"]) / 2
    center_lon = (AUSTRALIA_BOUNDS["min_lon"] + AUSTRALIA_BOUNDS["max_lon"]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    if use_heatmap:
        # Create heatmap using FFDI values
        heat_data = [[lat, lon, ffdi] for lat, lon, ffdi in zip(lats, lons, ffdi_values)]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=10).add_to(m)
    else:
        # Add circle markers for each point
        for i in range(len(lats)):
            color = RISK_COLORS[risk_levels[i]]
            folium.CircleMarker(
                location=[lats[i], lons[i]],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"FFDI: {ffdi_values[i]:.1f}<br>Risk: {risk_categories[i]}"
            ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid gray;
                border-radius: 5px; font-size: 12px;">
    <b>Fire Danger Rating</b><br>
    """
    for level, label in RISK_LABELS.items():
        color = RISK_COLORS[level]
        legend_html += f'<i style="background:{color};width:15px;height:15px;display:inline-block;margin-right:5px;"></i>{label}<br>'
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")

    return m


def export_to_csv(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_levels: np.ndarray,
    ffdi_values: np.ndarray,
    risk_categories: List[str],
    weather_data: Optional[Dict] = None,
    output_path: str = "fire_risk_data.csv"
) -> None:
    """
    Export fire risk data to CSV file.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        risk_levels: Array of risk levels
        ffdi_values: Array of FFDI values
        risk_categories: List of risk category names
        weather_data: Optional dictionary with weather arrays
        output_path: Path to save CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ["latitude", "longitude", "ffdi", "risk_level", "risk_category"]
        if weather_data:
            header.extend(["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm"])
        writer.writerow(header)

        # Data rows
        for i in range(len(lats)):
            row = [
                f"{lats[i]:.4f}",
                f"{lons[i]:.4f}",
                f"{ffdi_values[i]:.2f}",
                risk_levels[i],
                risk_categories[i]
            ]
            if weather_data:
                row.extend([
                    f"{weather_data['temperatures'][i]:.1f}",
                    f"{weather_data['humidities'][i]:.1f}",
                    f"{weather_data['wind_speeds'][i]:.1f}",
                    f"{weather_data['precipitations'][i]:.1f}"
                ])
            writer.writerow(row)

    print(f"Data exported to: {output_path}")


def export_to_json(
    lats: np.ndarray,
    lons: np.ndarray,
    risk_levels: np.ndarray,
    ffdi_values: np.ndarray,
    risk_categories: List[str],
    metadata: Optional[Dict] = None,
    output_path: str = "fire_risk_data.json"
) -> None:
    """
    Export fire risk data to JSON file.

    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        risk_levels: Array of risk levels
        ffdi_values: Array of FFDI values
        risk_categories: List of risk category names
        metadata: Optional metadata dictionary
        output_path: Path to save JSON file
    """
    data = {
        "metadata": metadata or {},
        "statistics": {
            "total_points": len(lats),
            "ffdi_min": float(ffdi_values.min()),
            "ffdi_max": float(ffdi_values.max()),
            "ffdi_mean": float(ffdi_values.mean()),
            "risk_distribution": {
                RISK_LABELS[i]: int(np.sum(risk_levels == i))
                for i in range(6)
            }
        },
        "points": [
            {
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "ffdi": float(ffdi_values[i]),
                "risk_level": int(risk_levels[i]),
                "risk_category": risk_categories[i]
            }
            for i in range(len(lats))
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Data exported to: {output_path}")


def print_risk_summary(
    risk_levels: np.ndarray,
    ffdi_values: np.ndarray,
    title: str = "Fire Risk Summary"
) -> None:
    """
    Print a summary of fire risk statistics to console.

    Args:
        risk_levels: Array of risk levels
        ffdi_values: Array of FFDI values
        title: Summary title
    """
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
        bar = "█" * int(pct / 2)
        print(f"  {RISK_LABELS[level]:15s}: {count:6d} ({pct:5.1f}%) {bar}")

    # Identify highest risk areas
    high_risk_mask = risk_levels >= 3  # Severe or above
    if np.any(high_risk_mask):
        print(f"\n⚠️  High-risk areas (Severe or above): {np.sum(high_risk_mask)} points")

    print("=" * 50 + "\n")
