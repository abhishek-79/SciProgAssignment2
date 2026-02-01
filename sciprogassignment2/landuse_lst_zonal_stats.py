# ============================================================
# landuse_lst_zonal_stats.py
# ============================================================

"""
Module: landuse_lst_zonal_stats

Description:
------------
Provides functionality to calculate zonal statistics of Land Surface
Temperature (LST) by land-use class in Kathmandu Valley, Nepal.

Functions:
----------
- run_zonal_stats(data_dir, landuse_file, lst_file, output_file)
    Computes zonal statistics and saves summary CSV.
"""

import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
from pathlib import Path

def run_zonal_stats(data_dir, landuse_file, lst_file, output_file):
    """
    Compute zonal statistics of LST for each land-use polygon.

    Parameters:
    -----------
    data_dir : str or Path
        Directory containing input data.
    landuse_file : str or Path
        File path to land-use polygons (GeoPackage).
    lst_file : str or Path
        File path to LST raster (GeoTIFF).
    output_file : str or Path
        File path to save summary CSV of zonal statistics.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with mean LST, min, max, std, and pixel count
        aggregated by land-use class.
    """
    
    # Convert to Path
    data_dir = Path(data_dir)
    landuse_file = Path(landuse_file)
    lst_file = Path(lst_file)
    output_file = Path(output_file)

    # --- Load land-use polygons ---
    landuse = gpd.read_file(landuse_file)
    print("Land-use polygons loaded.")
    print("Classes found:", landuse["land_use_class"].unique())

    # --- Load LST raster ---
    with rasterio.open(lst_file) as src:
        lst = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    print("LST raster loaded successfully.")

    # --- Align CRS ---
    if landuse.crs != crs:
        landuse = landuse.to_crs(crs)
        print("Reprojected land-use polygons to match raster CRS.")

    # --- Compute zonal statistics ---
    stats = zonal_stats(
        landuse,
        lst,
        affine=transform,
        nodata=nodata,
        stats=["mean", "min", "max", "std", "count"],
        geojson_out=False
    )
    print("Zonal statistics computed.")

    # --- Build results DataFrame ---
    records = []
    for i, stat in enumerate(stats):
        if stat["mean"] is None:
            continue
        records.append({
            "land_use_class": landuse.iloc[i]["land_use_class"],
            "lst_mean_c": stat["mean"],
            "lst_min_c": stat["min"],
            "lst_max_c": stat["max"],
            "lst_std_c": stat["std"],
            "pixel_count": stat["count"]
        })
    df = pd.DataFrame(records)

    # --- Aggregate by land-use class ---
    summary = df.groupby("land_use_class").mean(numeric_only=True).reset_index()

    # --- Save output ---
    summary.to_csv(output_file, index=False)
    print(f"Zonal statistics saved to: {output_file}")

    return summary
