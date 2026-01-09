#!/usr/bin/env python3
"""
Australia Fire Risk Map

A Python application that fetches weather data from Open-Meteo and calculates
fire risk across Australia using the McArthur Forest Fire Danger Index (FFDI).

Usage:
    python main.py                          # Full Australia at 50km resolution
    python main.py --resolution 25          # Full Australia at 25km resolution
    python main.py --city Sydney            # Sydney region at 5km resolution
    python main.py --city Melbourne --radius 100  # Melbourne, 100km radius
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from fire_risk.grid import (
    generate_australia_grid,
    get_city_grid,
    estimate_grid_size,
    AUSTRALIAN_CITIES,
)
from fire_risk.weather import fetch_grid_weather, extract_day_weather
from fire_risk.calculator import calculate_fire_risk_batch
from fire_risk.visualization import (
    create_matplotlib_map,
    create_folium_map,
    export_to_csv,
    export_to_json,
    print_risk_summary,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate fire risk across Australia using Open-Meteo weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full Australia, 50km grid
  python main.py --resolution 25              # Full Australia, 25km grid
  python main.py --city Sydney                # Sydney region, 5km grid
  python main.py --city Melbourne --radius 75 # Melbourne, 75km radius
  python main.py --days 3 --output-dir ./results

Available cities: Sydney, Melbourne, Brisbane, Perth, Adelaide, Darwin, Hobart, Canberra
        """
    )

    parser.add_argument(
        "--resolution", "-r",
        type=float,
        default=50.0,
        help="Grid resolution in kilometers (default: 50km for full Australia)"
    )

    parser.add_argument(
        "--city", "-c",
        type=str,
        choices=list(AUSTRALIAN_CITIES.keys()),
        help="Focus on a specific city region (uses 5km resolution by default)"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Radius around city in km (default: 50km)"
    )

    parser.add_argument(
        "--days", "-d",
        type=int,
        default=3,
        choices=range(1, 17),
        help="Number of forecast days (1-16, default: 3)"
    )

    parser.add_argument(
        "--day-index",
        type=int,
        default=0,
        help="Which day to visualize (0=today, 1=tomorrow, etc.)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Output directory for generated files"
    )

    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip generating map visualizations"
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export data to CSV"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Export data to JSON"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML map (using Folium)"
    )

    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Use heatmap style for interactive map"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)"
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous requests instead of async"
    )

    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate grid size without fetching data"
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated weather data (demo mode, no API required)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation reproducibility (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("🔥 Australia Fire Risk Map Generator")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine grid parameters
    if args.city:
        print(f"\nRegion: {args.city} (radius: {args.radius}km)")
        resolution = 5.0 if args.resolution == 50.0 else args.resolution
        print(f"Resolution: {resolution}km")
    else:
        print(f"\nRegion: Full Australia")
        resolution = args.resolution
        print(f"Resolution: {resolution}km")

    # Estimate grid size
    if args.city:
        grid = get_city_grid(args.city, args.radius, resolution)
        estimate = {
            "total_points": len(grid.points),
            "n_rows": grid.n_rows,
            "n_cols": grid.n_cols,
        }
    else:
        estimate = estimate_grid_size(resolution)

    print(f"\nGrid size: {estimate.get('n_rows', 'N/A')} x {estimate.get('n_cols', 'N/A')}")
    print(f"Total points: {estimate['total_points']:,}")

    if args.estimate_only:
        print("\n(Estimate only mode - exiting)")
        return 0

    # Warning for large grids
    if estimate['total_points'] > 10000:
        print(f"\n⚠️  Warning: Large grid ({estimate['total_points']:,} points)")
        print("This will take a while and make many API requests.")
        print("Consider using a coarser resolution or focusing on a city region.")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1

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
    grid_weather = fetch_grid_weather(
        grid,
        forecast_days=args.days,
        use_async=not args.sync,
        max_concurrent=args.max_concurrent,
        progress_bar=True,
        simulate=args.simulate,
        simulation_seed=args.seed
    )

    # Extract weather for the specified day
    print("\n" + "-" * 40)
    print(f"Step 3: Extracting weather for day {args.day_index}...")
    day_weather = extract_day_weather(grid_weather, args.day_index)

    valid_count = np.sum(day_weather["valid_mask"])
    print(f"Valid weather data points: {valid_count}/{len(grid.points)}")

    # Calculate fire risk
    print("\n" + "-" * 40)
    print("Step 4: Calculating fire risk (FFDI)...")
    ffdi_values, risk_levels, risk_categories = calculate_fire_risk_batch(
        day_weather["temperatures"],
        day_weather["humidities"],
        day_weather["wind_speeds"],
        day_weather["precipitations"]
    )

    # Print summary
    day_label = ["Today", "Tomorrow", "Day 3"][args.day_index] if args.day_index < 3 else f"Day {args.day_index + 1}"
    title = f"Fire Risk Summary - {args.city or 'Australia'} - {day_label}"
    print_risk_summary(risk_levels, ffdi_values, title)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    region = args.city.lower() if args.city else "australia"

    # Generate visualizations
    print("\n" + "-" * 40)
    print("Step 5: Generating outputs...")

    if not args.no_map:
        # Static matplotlib map
        map_path = output_dir / f"fire_risk_{region}_{timestamp}.png"
        fig = create_matplotlib_map(
            day_weather["lats"],
            day_weather["lons"],
            risk_levels,
            ffdi_values,
            title=f"Fire Risk Map - {args.city or 'Australia'} - {day_label}",
            output_path=str(map_path)
        )
        if fig:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    if args.interactive:
        # Interactive Folium map
        html_path = output_dir / f"fire_risk_{region}_{timestamp}.html"
        create_folium_map(
            day_weather["lats"],
            day_weather["lons"],
            risk_levels,
            ffdi_values,
            risk_categories,
            output_path=str(html_path),
            use_heatmap=args.heatmap
        )

    if args.csv:
        csv_path = output_dir / f"fire_risk_{region}_{timestamp}.csv"
        export_to_csv(
            day_weather["lats"],
            day_weather["lons"],
            risk_levels,
            ffdi_values,
            risk_categories,
            weather_data=day_weather,
            output_path=str(csv_path)
        )

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
        export_to_json(
            day_weather["lats"],
            day_weather["lons"],
            risk_levels,
            ffdi_values,
            risk_categories,
            metadata=metadata,
            output_path=str(json_path)
        )

    print("\n" + "=" * 60)
    print("✅ Fire risk analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
