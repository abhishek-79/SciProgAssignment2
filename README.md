# Kathmandu Valley Environmental Analysis

This repository contains environmental analyses of Kathmandu Valley using satellite remote sensing data with CPU/GPU performance benchmarking.

## Table of Contents

- [NDVI and LST Analysis](#ndvi-and-lst-analysis)
- [LST and Land Use Analysis](#lst-and-land-use-analysis)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)

## NDVI and LST Analysis

### Overview

Temporal analysis of vegetation (NDVI) and Land Surface Temperature (LST) changes in Kathmandu Valley across 2016, 2020, and 2025. This analysis examines how vegetation cover correlates with surface temperature patterns over a decade.

The implementation includes **CPU vs GPU performance benchmarking** to compare traditional NumPy/xarray processing against PyTorch GPU-accelerated computation.

### Features

- **Multi-temporal Analysis**: Compares three time periods (2016, 2020, 2025)
- **NDVI Calculation**: Derives vegetation indices from Sentinel-2 bands
- **LST Processing**: Processes MODIS 8-day LST composites
- **CPU/GPU Benchmarking**: Compares processing performance between CPU (xarray) and GPU (PyTorch tensors)
- **Temporal Change Detection**: Calculates and visualizes changes between periods
- **Trend Visualization**: Time series plots showing mean NDVI and LST evolution

### Performance Benchmarking

The analysis includes timing comparisons for:

- **Per-year LST processing** (CPU vs GPU)
- **Per-year NDVI computation** (CPU vs GPU)
- **Spatial clipping** (CPU)
- **Datacube construction** (CPU)
- **Change calculation** (CPU)
- **Visualization** (CPU)

GPU timings include proper CUDA synchronization for accurate measurements. Note that downstream operations (clipping, datacubes, plotting) use CPU/xarray results as they require geospatial metadata and matplotlib compatibility.

### Data Sources

- **LST Data**: MODIS 8-Day LST product (MOD11A2) at 1km resolution
- **NDVI Data**: Sentinel-2 MSI (Bands 4 and 8) at 10m resolution
- **Study Area**: Kathmandu Valley boundary shapefile (EPSG:4326)

### Directory Structure

```
sciprogassignment2/
├── ndvi_lst_analysis.py    # Main analysis module with CPU/GPU benchmarking
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
├── pyproject.toml
└── README.md
```

## Usage

### Running with Poetry (Recommended)

```bash
# Run the analysis with automatic dependency management
poetry run python main.py
```

Or activate the virtual environment first:

```bash
# Activate poetry shell
poetry shell

# Run the script
python main.py
```

### Basic Usage in Code

```python
from pathlib import Path
from ndvi_lst_analysis import run_analysis

data_dir = Path("data")
boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"

# Run complete analysis with CPU/GPU benchmarking
lst_cube, ndvi_cube, lst_changes = run_analysis(data_dir, boundary_file)
```

### Expected Output

The script will print timing information for each operation:

```
Running on: cuda
CPU LST 2016              0.5432 s
CPU NDVI 2016             0.2103 s
GPU LST 2016              0.1234 s
GPU NDVI 2016             0.0567 s
CPU LST 2020              0.5398 s
CPU NDVI 2020             0.2087 s
GPU LST 2020              0.1189 s
GPU NDVI 2020             0.0542 s
...
CPU Clipping              0.3456 s
CPU Datacubes             0.1234 s
CPU LST Changes           0.0876 s
Plot Spatial Comparison   1.2345 s
Plot Time Series          0.4567 s
```

### Key Functions

#### `load_lst_for_year_cpu(modis_files, mask_value=0)`

Loads multiple MODIS LST files and returns temporal average in Celsius using xarray.

#### `load_lst_for_year_gpu(modis_files, mask_value=0)`

GPU-accelerated version using PyTorch tensors. Used for benchmarking only.

#### `compute_ndvi_cpu(red_file, nir_file)`

Computes NDVI from Sentinel-2 Red (Band 4) and NIR (Band 8) bands using NumPy.

#### `compute_ndvi_gpu(red_file, nir_file)`

GPU-accelerated NDVI computation using PyTorch tensors. Used for benchmarking only.

#### `clip_to_boundary(data_list, boundary)`

Clips raster data to study area boundary.

#### `build_datacubes(lst_list, ndvi_list, years, reference_grid)`

Creates aligned temporal datacubes for multi-year analysis.

#### `calculate_changes(cube, years)`

Computes temporal changes between successive years and total change.

#### `run_analysis(data_dir, boundary_file)`

Main workflow that orchestrates the entire analysis pipeline with CPU/GPU benchmarking.

### Outputs

The analysis generates two main visualizations:

**Spatial Comparison (3×3 grid)**:

- Row 1: NDVI for 2016, 2020, 2025
- Row 2: LST for 2016, 2020, 2025
- Row 3: LST changes (2016→2020, 2020→2025, 2016→2025)

**Time Series Plot**:

- Mean LST trend over time
- Mean NDVI trend over time

## Requirements

### Core Dependencies

- `rioxarray` - Geospatial raster I/O
- `xarray` - Multi-dimensional arrays
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `geopandas` - Vector data handling
- `torch` - GPU-accelerated computing (with CUDA support for GPU benchmarking)

### Installation

#### With Poetry (Recommended)

```bash
poetry install
```

#### With pip

```bash
pip install rioxarray xarray numpy matplotlib geopandas torch
```

#### For GPU Support

Install PyTorch with CUDA support following instructions at [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

The code automatically detects GPU availability and falls back to CPU if CUDA is not available.

## Technical Notes

- **CRS Handling**: All data is reprojected to match the reference grid (2016) to ensure spatial alignment
- **PROJ Database**: Includes workaround for PostgreSQL PROJ conflicts
- **NDVI Masking**: Values outside [-1, 1] are masked as invalid
- **LST Conversion**: MODIS LST data converted from Kelvin to Celsius
- **Resampling**: Uses bilinear interpolation for reprojection
- **GPU Synchronization**: GPU timings include `torch.cuda.synchronize()` for accurate performance measurement
- **Dual Processing**: Data is processed on both CPU (xarray) and GPU (PyTorch) for benchmarking; CPU results are used for final analysis and visualization

## Performance Considerations

- **GPU Advantage**: Most significant for large raster processing (averaging multiple LST files, NDVI computation)
- **CPU Advantage**: Better for geospatial operations requiring metadata (clipping, reprojection, CRS handling)
- **Transfer Overhead**: GPU timings include CPU→GPU data transfer, which can dominate for small datasets
- **Visualization**: Matplotlib requires NumPy arrays, so GPU results aren't used for plotting

## Known Issues

- PROJ database conflicts may occur if PostgreSQL is installed
- Large .tif files may cause slow Git operations (consider Git LFS)
- GPU memory may be insufficient for very high-resolution imagery

## LST and Land Use Analysis

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
```
sciprogassignment2/
├── ndvi_lst_landuse_analysis.py
├── landuse_lst_zonal_stats.py
├── data/
│ ├── May2020/
│ ├── May2025/
│ ├── kathmandu_landuse_osm.gpkg
│ └── ktm_bhktpr_ltpr_shapefile.gpkg
└── README.md
```

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
- CUDA-capable GPU (optional, for GPU benchmarking)

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

- **[Meghraj Singh]** - NDVI-LST Analysis with CPU/GPU Benchmarking
- **[Abhishek Adhikari]** - LST-Land Use Analysis
