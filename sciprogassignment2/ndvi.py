"""
NDVI–LST Temporal Analysis for Kathmandu Valley
CPU vs GPU Benchmark
"""

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray as rxr
import torch
import matplotlib.pyplot as plt
import geopandas as gpd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

@contextmanager
def timer(name):
    """Time a code block."""
    start = time.perf_counter()
    yield
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"{name:<25} {end - start:.4f} s")


def load_lst_for_year_cpu(modis_files, mask_value=0):
    """CPU: load and average LST files."""
    arrays = []
    for f in modis_files:
        da = rxr.open_rasterio(f, masked=False).squeeze()
        da = da.where(da != mask_value) - 273.15
        arrays.append(da)
    return xr.concat(arrays, dim="time").mean(dim="time"), da


def compute_ndvi_cpu(red_file, nir_file):
    """CPU: compute NDVI from red/NIR bands."""
    red = rxr.open_rasterio(red_file, masked=True).squeeze()
    nir = rxr.open_rasterio(nir_file, masked=True).squeeze()
    vals = (nir.values - red.values) / (nir.values + red.values + 1e-10)
    vals = np.where((vals < -1) | (vals > 1), np.nan, vals)
    da = xr.DataArray(vals, coords=nir.coords, dims=nir.dims)
    da = da.rio.write_crs(nir.rio.crs).rio.write_transform(nir.rio.transform())
    return da


def to_tensor(arr):
    """Convert NumPy array to torch tensor on DEVICE."""
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)


def load_lst_for_year_gpu(modis_files, mask_value=0):
    """GPU: load and average LST as tensors (per-year only)."""
    tensors = []
    for f in modis_files:
        da = rxr.open_rasterio(f, masked=False).squeeze()
        t = to_tensor(da.values)
        t = torch.where(t == mask_value, torch.nan, t) - 273.15
        tensors.append(t)
    mean_lst = torch.nanmean(torch.stack(tensors), dim=0)
    return mean_lst


def compute_ndvi_gpu(red_file, nir_file):
    """GPU: compute NDVI as tensor (per-year only)."""
    red = to_tensor(rxr.open_rasterio(red_file, masked=True).squeeze().values)
    nir = to_tensor(rxr.open_rasterio(nir_file, masked=True).squeeze().values)
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = torch.where((ndvi < -1) | (ndvi > 1), torch.nan, ndvi)
    return ndvi


def clip_to_boundary(data_list, boundary):
    """Clip rasters to shapefile."""
    clipped = []
    for da in data_list:
        clipped.append(
            da.rio.clip(boundary.geometry.to_crs(da.rio.crs),
                        da.rio.crs, drop=True, all_touched=True)
        )
    return clipped


def build_datacubes(lst_list, ndvi_list, years, reference_grid):
    """CPU: build temporal cubes."""
    lst_matched = [lst_list[0]] + [lst.rio.reproject_match(reference_grid) for lst in lst_list[1:]]
    ndvi_matched = [ndvi.rio.reproject_match(reference_grid) for ndvi in ndvi_list]
    lst_cube = xr.concat(lst_matched, dim=xr.DataArray(years, dims="year", name="year"))
    ndvi_cube = xr.concat(ndvi_matched, dim=xr.DataArray(years, dims="year", name="year"))
    return lst_cube, ndvi_cube


def calculate_changes(cube, years):
    """CPU: compute year-to-year changes."""
    changes = {}
    for i in range(len(years)-1):
        changes[f"{years[i]}_{years[i+1]}"] = cube.sel(year=years[i+1]) - cube.sel(year=years[i])
    changes["total"] = cube.sel(year=years[-1]) - cube.sel(year=years[0])
    return changes


def plot_spatial_comparison(lst_clipped, ndvi_clipped, lst_changes, years):
    """Plot NDVI, LST, and LST changes."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    lst_vmin = min(float(lst.min()) for lst in lst_clipped)
    lst_vmax = max(float(lst.max()) for lst in lst_clipped)

    for i, (ndvi, year) in enumerate(zip(ndvi_clipped, years)):
        ndvi.plot(ax=axes[0, i], cmap="YlGn", vmin=0, vmax=1, cbar_kwargs={"label": "NDVI"})
        axes[0, i].set_title(f"NDVI {year}")
    for i, (lst, year) in enumerate(zip(lst_clipped, years)):
        lst.plot(ax=axes[1, i], cmap="RdYlBu_r", vmin=lst_vmin, vmax=lst_vmax, cbar_kwargs={"label": "LST (°C)"})
        axes[1, i].set_title(f"LST {year}")
    keys = [f"{years[0]}_{years[1]}", f"{years[1]}_{years[2]}", "total"]
    titles = [f"{years[0]}→{years[1]}", f"{years[1]}→{years[2]}", f"{years[0]}→{years[2]}"]
    for i, (k, t) in enumerate(zip(keys, titles)):
        lst_changes[k].plot(ax=axes[2, i], cmap="RdBu_r", center=0, cbar_kwargs={"label": "LST Change"})
        axes[2, i].set_title(f"LST Change {t}")

    plt.tight_layout()
    return fig


def plot_time_series(lst_cube, ndvi_cube, years):
    """Plot mean LST and NDVI over years."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    lst_means = [float(lst_cube.sel(year=y).mean(skipna=True)) for y in years]
    ndvi_means = [float(ndvi_cube.sel(year=y).mean(skipna=True)) for y in years]
    ax1.plot(years, lst_means, "o-", color="red")
    ax2.plot(years, ndvi_means, "o-", color="green")
    ax1.set_ylabel("Mean LST (°C)"); ax2.set_ylabel("Mean NDVI"); ax2.set_xlabel("Year")
    plt.tight_layout()
    return fig


def run_analysis(data_dir, boundary_file):
    """Run workflow: CPU vs GPU per-year timing; CPU datacubes for plotting."""
    data_dir = Path(data_dir)
    years = [2016, 2020, 2025]
    boundary = gpd.read_file(boundary_file)

    lst_cpu_all, ndvi_cpu_all = [], []
    lst_gpu_all, ndvi_gpu_all = [], []

    for year in years:
        lst_files = [data_dir / f"May{year}/lst_may_{year}_{i}.tif" for i in range(1, 5)]
        red_file = data_dir / f"May{year}/sentinel_2_band_04_may_{year}.tif"
        nir_file = data_dir / f"May{year}/sentinel_2_band_08_may_{year}.tif"

        with timer(f"CPU LST {year}"):
            lst_cpu, ref = load_lst_for_year_cpu(lst_files)
        with timer(f"CPU NDVI {year}"):
            ndvi_cpu = compute_ndvi_cpu(red_file, nir_file)

        with timer(f"GPU LST {year}"):
            lst_gpu = load_lst_for_year_gpu(lst_files)
        with timer(f"GPU NDVI {year}"):
            ndvi_gpu = compute_ndvi_gpu(red_file, nir_file)

        lst_cpu_all.append(lst_cpu); ndvi_cpu_all.append(ndvi_cpu)
        lst_gpu_all.append(lst_gpu); ndvi_gpu_all.append(ndvi_gpu)

    with timer("CPU Clipping"):
        lst_cpu_clipped = clip_to_boundary(lst_cpu_all, boundary)
        ndvi_cpu_clipped = clip_to_boundary(ndvi_cpu_all, boundary)

    with timer("CPU Datacubes"):
        lst_cube, ndvi_cube = build_datacubes(lst_cpu_clipped, ndvi_cpu_clipped, years, lst_cpu_clipped[0])

    with timer("CPU LST Changes"):
        lst_changes = calculate_changes(lst_cube, years)

    with timer("Plot Spatial Comparison"):
        plot_spatial_comparison(lst_cpu_clipped, ndvi_cpu_clipped, lst_changes, years)
    with timer("Plot Time Series"):
        plot_time_series(lst_cube, ndvi_cube, years)

    plt.show()
    return lst_cube, ndvi_cube, lst_changes


if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = project_root.parent / "data"
    boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"
    run_analysis(data_dir, boundary_file)
