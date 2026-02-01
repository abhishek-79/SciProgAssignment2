"""
NDVI-LST Temporal Analysis for Kathmandu Valley
Analyzes vegetation and temperature changes across 2016, 2020, and 2025.
"""

import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import os

os.environ['PROJ_LIB'] = ''


def load_lst_for_year(modis_files, mask_value=0):
    """Load and average multiple MODIS LST files, converting to Celsius."""
    lst_arrays = []
    for f in modis_files:
        da = rxr.open_rasterio(f, masked=False).squeeze()
        da = da.where(da != mask_value) - 273.15
        lst_arrays.append(da)
    return xr.concat(lst_arrays, dim='time').mean(dim='time')


def compute_ndvi(red_file, nir_file):
    """Compute NDVI from Sentinel-2 red and NIR bands."""
    red = rxr.open_rasterio(red_file, masked=True).squeeze()
    nir = rxr.open_rasterio(nir_file, masked=True).squeeze()
    
    ndvi_vals = (nir.values - red.values) / (nir.values + red.values + 1e-10)
    ndvi_vals = np.where((ndvi_vals < -1) | (ndvi_vals > 1), np.nan, ndvi_vals)
    
    ndvi = xr.DataArray(ndvi_vals, coords=nir.coords, dims=nir.dims,
                       attrs={'long_name': 'NDVI', 'units': ''})
    ndvi = ndvi.rio.write_crs(nir.rio.crs)
    ndvi = ndvi.rio.write_transform(nir.rio.transform())
    return ndvi


def load_year_data(data_dir, year, lst_pattern, ndvi_red_file, ndvi_nir_file):
    """Load LST and NDVI data for a single year."""
    lst = load_lst_for_year(lst_pattern)
    ndvi = compute_ndvi(ndvi_red_file, ndvi_nir_file)
    return lst, ndvi


def clip_to_boundary(data_list, boundary):
    """Clip multiple rasters to boundary shapefile."""
    clipped = []
    for data in data_list:
        clipped_data = data.rio.clip(
            boundary.geometry.to_crs(data.rio.crs),
            data.rio.crs, drop=True, all_touched=True
        )
        clipped.append(clipped_data)
    return clipped


def build_datacubes(lst_list, ndvi_list, years, reference_grid):
    """Build temporal datacubes from multiple years, matched to reference grid."""
    lst_matched = [lst_list[0]] + [
        lst.rio.reproject_match(reference_grid) for lst in lst_list[1:]
    ]
    ndvi_matched = [
        ndvi.rio.reproject_match(reference_grid) for ndvi in ndvi_list
    ]
    
    lst_cube = xr.concat(lst_matched, dim=xr.DataArray(years, dims='year', name='year'))
    ndvi_cube = xr.concat(ndvi_matched, dim=xr.DataArray(years, dims='year', name='year'))
    
    return lst_cube, ndvi_cube


def calculate_changes(cube, years):
    """Calculate temporal changes between years."""
    changes = {}
    for i in range(len(years) - 1):
        key = f"{years[i]}_{years[i+1]}"
        changes[key] = cube.sel(year=years[i+1]) - cube.sel(year=years[i])
    changes['total'] = cube.sel(year=years[-1]) - cube.sel(year=years[0])
    return changes


def plot_spatial_comparison(lst_clipped, ndvi_clipped, lst_changes, years):
    """Create 3x3 spatial comparison plot."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    lst_vmin = min(float(lst.min()) for lst in lst_clipped)
    lst_vmax = max(float(lst.max()) for lst in lst_clipped)
    
    # NDVI row
    for i, (ndvi, year) in enumerate(zip(ndvi_clipped, years)):
        ndvi.plot(ax=axes[0, i], cmap='YlGn', vmin=0, vmax=1,
                 cbar_kwargs={'label': 'NDVI'})
        axes[0, i].set_title(f'NDVI {year}', fontsize=12, fontweight='bold')
        axes[0, i].set_aspect('equal')
    
    # LST row
    for i, (lst, year) in enumerate(zip(lst_clipped, years)):
        lst.plot(ax=axes[1, i], cmap='RdYlBu_r', vmin=lst_vmin, vmax=lst_vmax,
                cbar_kwargs={'label': 'Temperature (°C)'})
        axes[1, i].set_title(f'LST {year}', fontsize=12, fontweight='bold')
        axes[1, i].set_aspect('equal')
    
    # Change row
    change_keys = [f"{years[0]}_{years[1]}", f"{years[1]}_{years[2]}", 'total']
    change_titles = [f'{years[0]}→{years[1]}', f'{years[1]}→{years[2]}', 
                    f'{years[0]}→{years[2]} (Total)']
    
    for i, (key, title) in enumerate(zip(change_keys, change_titles)):
        lst_changes[key].plot(ax=axes[2, i], cmap='RdBu_r', center=0,
                            cbar_kwargs={'label': 'LST Change (°C)'})
        axes[2, i].set_title(f'LST Change {title}', fontsize=12, fontweight='bold')
        axes[2, i].set_aspect('equal')
    
    plt.suptitle('NDVI-LST Temporal Analysis: Kathmandu Valley (2016-2025)',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_time_series(lst_cube, ndvi_cube, years):
    """Create time series plot of mean LST and NDVI."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    lst_means = [float(lst_cube.sel(year=y).mean(skipna=True)) for y in years]
    ndvi_means = [float(ndvi_cube.sel(year=y).mean(skipna=True)) for y in years]
    
    ax1.plot(years, lst_means, 'o-', linewidth=2, markersize=8, color='red')
    ax1.set_ylabel('Mean LST (°C)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Temporal Trends in Kathmandu Valley', fontsize=14, fontweight='bold')
    
    ax2.plot(years, ndvi_means, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_ylabel('Mean NDVI', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(years)
    
    plt.tight_layout()
    return fig


def run_analysis(data_dir, boundary_file):
    """Main analysis workflow."""
    data_dir = Path(data_dir)
    years = [2016, 2020, 2025]
    
    # Load boundary
    kathmandu = gpd.read_file(boundary_file)
    
    # Load data for all years
    lst_2016, ndvi_2016 = load_year_data(
        data_dir, 2016,
        [data_dir / f"May2016/lst_may_2016_{i}.tif" for i in range(1, 5)],
        data_dir / "May2016/sentinel_2_band_04_may_2016.tif",
        data_dir / "May2016/sentinel_2_band_08_may_2016.tif"
    )
    
    lst_2020, ndvi_2020 = load_year_data(
        data_dir, 2020,
        [data_dir / f"May2020/lst_may_2020_{i}.tif" for i in range(1, 5)],
        data_dir / "May2020/sentinel_2_band_04_may_2020.tif",
        data_dir / "May2020/sentinel_2_band_08_may_2020.tif"
    )
    
    lst_2025, ndvi_2025 = load_year_data(
        data_dir, 2025,
        [data_dir / f"May2025/lst_may_2025_{i}.tif" for i in range(1, 5)],
        data_dir / "May2025/sentinel_2_band_04_may_2025.tif",
        data_dir / "May2025/sentinel_2_band_08_may_2025.tif"
    )
    
    # Clip to boundary
    lst_clipped = clip_to_boundary([lst_2016, lst_2020, lst_2025], kathmandu)
    ndvi_clipped = clip_to_boundary([ndvi_2016, ndvi_2020, ndvi_2025], kathmandu)
    
    # Build datacubes
    lst_cube, ndvi_cube = build_datacubes(
        lst_clipped, ndvi_clipped, years, lst_clipped[0]
    )
    
    # Calculate changes
    lst_changes = calculate_changes(lst_cube, years)
    
    # Generate plots
    fig_spatial = plot_spatial_comparison(lst_clipped, ndvi_clipped, lst_changes, years)
    fig_timeseries = plot_time_series(lst_cube, ndvi_cube, years)
    
    plt.show()
    
    return lst_cube, ndvi_cube, lst_changes


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"
    
    run_analysis(data_dir, boundary_file)