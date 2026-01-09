# Australia Fire Risk Map

A Python application that calculates fire risk across Australia using weather data from the Open-Meteo API. The application uses the McArthur Forest Fire Danger Index (FFDI), the standard fire danger rating system used in Australia.

## Features

- Generates a 5x5km grid (configurable) across Australia or specific city regions
- Fetches 3-day weather forecasts from Open-Meteo API
- Calculates fire risk using the McArthur FFDI formula
- Produces multiple output formats:
  - Static PNG maps (matplotlib)
  - Interactive HTML maps (Folium)
  - CSV data exports
  - JSON data exports
- Includes simulation mode for testing/demos without API access

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Full Australia at 50km resolution (default)
python main.py

# Full Australia at 25km resolution
python main.py --resolution 25

# Sydney region at 5km resolution
python main.py --city Sydney

# Melbourne region with 100km radius
python main.py --city Melbourne --radius 100
```

### Demo Mode (Simulated Data)

If you don't have API access or want to test the application:

```bash
# Use simulated weather data
python main.py --simulate --city Sydney
python main.py --simulate --resolution 100
```

### Output Options

```bash
# Generate all outputs
python main.py --city Sydney --csv --json --interactive

# Generate interactive heatmap
python main.py --city Sydney --interactive --heatmap

# Skip map generation
python main.py --city Sydney --no-map --csv
```

### Available Cities

- Sydney
- Melbourne
- Brisbane
- Perth
- Adelaide
- Darwin
- Hobart
- Canberra

### All Options

```
--resolution, -r    Grid resolution in km (default: 50km)
--city, -c          Focus on specific city region
--radius            Radius around city in km (default: 50km)
--days, -d          Forecast days 1-16 (default: 3)
--day-index         Which day to visualize (0=today)
--output-dir, -o    Output directory (default: ./output)
--csv               Export to CSV
--json              Export to JSON
--interactive       Generate interactive HTML map
--heatmap           Use heatmap style for interactive map
--no-map            Skip PNG map generation
--simulate          Use simulated weather data (demo mode)
--seed              Random seed for simulation (default: 42)
--sync              Use synchronous API requests
--max-concurrent    Max concurrent requests (default: 10)
--estimate-only     Only estimate grid size
```

## Fire Danger Rating System

The application uses the Australian Fire Danger Rating System:

| FFDI Range | Rating | Color |
|------------|--------|-------|
| 0-11 | Low-Moderate | Green |
| 12-24 | High | Blue |
| 25-49 | Very High | Yellow |
| 50-74 | Severe | Orange |
| 75-99 | Extreme | Red |
| 100+ | Catastrophic | Purple |

## How It Works

1. **Grid Generation**: Creates a geographic grid at the specified resolution
2. **Weather Fetching**: Retrieves forecast data from Open-Meteo API (or generates simulated data)
3. **FFDI Calculation**: Applies the McArthur Mark 5 formula using:
   - Maximum temperature
   - Minimum relative humidity
   - Maximum wind speed
   - Precipitation (for drought factor estimation)
4. **Visualization**: Generates maps and data exports

## Output Files

- `fire_risk_{region}_{timestamp}.png` - Static map visualization
- `fire_risk_{region}_{timestamp}.html` - Interactive web map
- `fire_risk_{region}_{timestamp}.csv` - Tabular data with all grid points
- `fire_risk_{region}_{timestamp}.json` - Structured data with metadata

## API Information

This application uses the [Open-Meteo API](https://open-meteo.com/), which provides free weather forecast data. No API key is required.

## License

MIT License
