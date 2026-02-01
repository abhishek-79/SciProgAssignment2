# Kathmandu Valley Environmental Analysis

This repository contains environmental analyses of Kathmandu Valley using satellite remote sensing data.

## Table of Contents

- [NDVI and LST Analysis](#ndvi-and-lst-analysis)
- [LST and Land Use Analysis](#lst-and-land-use-analysis)
- [Installation](#installation)
- [Data Requirements](#data-requirements)

---

## NDVI and LST Analysis

### Overview

Temporal analysis of vegetation (NDVI) and Land Surface Temperature (LST) changes in Kathmandu Valley across 2016, 2020, and 2025. This analysis examines how vegetation cover correlates with surface temperature patterns over a decade.

### Features

- **Multi-temporal Analysis**: Compares three time periods (2016, 2020, 2025)
- **NDVI Calculation**: Derives vegetation indices from Sentinel-2 bands
- **LST Processing**: Processes MODIS 8-day LST composites
- **Temporal Change Detection**: Calculates and visualizes changes between periods
- **Trend Visualization**: Time series plots showing mean NDVI and LST evolution

### Data Sources

- **LST Data**: MODIS 8-Day LST product (MOD11A2) at 1km resolution
- **NDVI Data**: Sentinel-2 MSI (Bands 4 and 8) at 10m resolution
- **Study Area**: Kathmandu Valley boundary shapefile (EPSG:4326)

### Directory Structure

```
sciprogassignment2/
├── ndvi_lst_analysis.py    # Main analysis module
├── main.py                  # Entry point script
├── data/
│   ├── May2016/
│   │   ├── lst_may_2016_1.tif
│   │   ├── lst_may_2016_2.tif
│   │   ├── lst_may_2016_3.tif
│   │   ├── lst_may_2016_4.tif
│   │   ├── sentinel_2_band_04_may_2016.tif
│   │   └── sentinel_2_band_08_may_2016.tif
│   ├── May2020/
│   │   ├── lst_may_2020_1.tif
│   │   ├── lst_may_2020_2.tif
│   │   ├── lst_may_2020_3.tif
│   │   ├── lst_may_2020_4.tif
│   │   ├── sentinel_2_band_04_may_2020.tif
│   │   └── sentinel_2_band_08_may_2020.tif
│   ├── May2025/
│   │   ├── lst_may_2025_1.tif
│   │   ├── lst_may_2025_2.tif
│   │   ├── lst_may_2025_3.tif
│   │   ├── lst_may_2025_4.tif
│   │   ├── sentinel_2_band_04_may_2025.tif
│   │   └── sentinel_2_band_08_may_2025.tif
│   └── ktm_bhktpr_ltpr_shapefile.gpkg
└── README.md
```

### Usage

#### Basic Usage

```python
from pathlib import Path
from ndvi_lst_analysis import run_analysis

data_dir = Path("data")
boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"

# Run complete analysis
lst_cube, ndvi_cube, lst_changes = run_analysis(data_dir, boundary_file)
```

#### Command Line

```bash
python main.py
```

### Key Functions

#### `load_lst_for_year(modis_files, mask_value=0)`

Loads multiple MODIS LST files and returns temporal average in Celsius.

#### `compute_ndvi(red_file, nir_file)`

Computes NDVI from Sentinel-2 Red (Band 4) and NIR (Band 8) bands.

#### `clip_to_boundary(data_list, boundary)`

Clips raster data to study area boundary.

#### `build_datacubes(lst_list, ndvi_list, years, reference_grid)`

Creates aligned temporal datacubes for multi-year analysis.

#### `calculate_changes(cube, years)`

Computes temporal changes between successive years and total change.

#### `run_analysis(data_dir, boundary_file)`

Main workflow that orchestrates the entire analysis pipeline.

### Outputs

The analysis generates two main visualizations:

1. **Spatial Comparison (3×3 grid)**:
   - Row 1: NDVI for 2016, 2020, 2025
   - Row 2: LST for 2016, 2020, 2025
   - Row 3: LST changes (2016→2020, 2020→2025, 2016→2025)

2. **Time Series Plot**:
   - Mean LST trend over time
   - Mean NDVI trend over time

### Requirements

```
rioxarray
xarray
numpy
matplotlib
geopandas
```

Install via Poetry:

```bash
poetry install
```

Or via pip:

```bash
pip install rioxarray xarray numpy matplotlib geopandas
```

### Technical Notes

- **CRS Handling**: All data is reprojected to match the reference grid (2016) to ensure spatial alignment
- **PROJ Database**: Includes workaround for PostgreSQL PROJ conflicts
- **NDVI Masking**: Values outside [-1, 1] are masked as invalid
- **LST Conversion**: MODIS LST data converted from Kelvin to Celsius
- **Resampling**: Uses bilinear interpolation for reprojection

### Known Issues

- PROJ database conflicts may occur if PostgreSQL is installed
- Large `.tif` files may cause slow Git operations (consider Git LFS)

---

## NDVI–LST–Landuse Analysis

### Overview

This module performs a multi-year analysis of NDVI (vegetation) and LST (Land Surface Temperature) in Kathmandu Valley, conditioned on land-use types. The analysis explores the relationships between vegetation, temperature, and urban land-use categories for the years 2020 and 2025.

Key goals:  

- Quantify how LST varies across land-use types  
- Examine the correlation between NDVI and LST globally and per land-use class  
- Attach mean NDVI and LST values to land-use polygons for spatial reporting  

### Features

- **Multi-year Analysis**: 2020 and 2025  
- **Global NDVI–LST Correlation**: Regression slope, R², and RMSE  
- **Land-Use Conditioned Analysis**: Per-class NDVI–LST correlations  
- **Rasterized Land-Use Layer**: Vector-to-raster conversion for pixel-level analysis  
- **GeoPackage Output**: Enriched land-use polygons for mapping  
- **CSV Summary**: Tabular zonal statistics by land-use class  

### Data Sources

- **LST Data**: MODIS 8-day LST composites (May 2020, May 2025)  
- **NDVI Data**: Sentinel-2 MSI bands (Red: Band 4, NIR: Band 8)  
- **Land-Use Data**: Kathmandu Valley land-use polygons (GeoPackage)  
- **Study Area**: Kathmandu Valley boundary shapefile  

### Directory Structure

sciprogassignment2/
├── ndvi_lst_landuse_analysis.py
├── landuse_lst_zonal_stats.py
├── data/
│ ├── May2020/
│ ├── May2025/
│ ├── kathmandu_landuse_osm.gpkg
│ └── ktm_bhktpr_ltpr_shapefile.gpkg
└── README.md


### Usage

#### Python API

```python
from ndvi_lst_landuse_analysis import run_ndvi_lst_landuse_analysis

data_dir = "data"
results_2020, results_2025 = run_ndvi_lst_landuse_analysis(data_dir)
```

#### Command Line
```
python main.py
```
Toggle execution using RUN_NDVI_LST_LANDUSE = TRUE.

### Outputs

1. **CSV Summary** (`data/output/landuse_ndvi_lst_stats_YYYY.csv`)  
   - Regression R², slope, and pixel count per land-use class  

2. **GeoPackage** (`data/output/landuse_ndvi_lst_stats.gpkg`)  
   - Land-use polygons enriched with mean NDVI and LST values  

3. **Global Statistics**  
   - Regression slope, intercept, R², RMSE, pixel count across the study area  

### Requirements

rioxarray
xarray
numpy
geopandas
pandas
scipy
rasterio
rasterstats

Install via Poetry:
```
poetry install
```

Or via pip:
```
pip install rioxarray xarray numpy geopandas pandas scipy rasterio rasterstats
```

### Technical Notes

- CRS Handling: All data reprojected to match NDVI raster CRS  
- NDVI Range Check: Values outside [-1, 1] masked  
- LST Conversion: MODIS LST converted from Kelvin to Celsius  
- Raster-Vector Alignment: Land-use polygons rasterized to NDVI pixel grid  

### Known Issues

- Large raster files may increase memory usage  
- Sparse land-use classes may yield insufficient pixels for regression

---

## Installation

### Prerequisites

- Python 3.8+
- Poetry (recommended) or pip

### Setup

```bash
# Clone repository
git clone <repository-url>
cd sciprogassignment2

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -r requirements.txt
```

## Data Requirements

Download the required satellite imagery and place in the `data/` directory following the structure shown above. Data should include:

- MODIS 8-day LST composites for May 2016, 2020, 2025
- Sentinel-2 Band 4 (Red) and Band 8 (NIR) for May 2016, 2020, 2025
- Kathmandu Valley boundary shapefile

## Contributing

This is a group project with separate analysis modules:

- **NDVI-LST Analysis**: Vegetation and temperature relationship
- **NDVI-LST-Land Use Analysis**: Temperature patterns by land use type

## Authors

- [Meghraj Singh] - NDVI-LST Analysis
- [Abhishek Adhikari] - NDVI-LST-Land Use Analysis
