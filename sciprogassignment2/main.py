"""
Main script for executing geospatial analyses for Kathmandu Valley, Nepal.

Description:
------------
This script orchestrates three geospatial workflows:

1. NDVI Temporal Analysis (via ndvi.py)
2. Land-Use LST Zonal Statistics (via landuse_lst_zonal_stats.py)
3. NDVI–LST–Landuse Analysis (multi-year, via ndvi_lst_landuse_analysis.py)

All workflows read raster and vector data from the `data` directory
and produce CSV/GeoPackage outputs suitable for reporting and mapping.

Usage:
------
Run from the command line or an IDE. Toggle execution of each workflow
via the boolean flags below.
"""

from pathlib import Path

# --- Import your analysis modules ---
from ndvi import run_analysis as run_ndvi
from landuse_lst_zonal_stats import run_zonal_stats as run_lst_zonal_stats
from ndvi_lst_landuse_analysis import run_ndvi_lst_landuse_analysis as run_ndvi_lst

if __name__ == "__main__":
    # --- Setup directories ---
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../data"

    # --- Analysis flags: toggle which workflows to run ---
    RUN_NDVI_TEMPORAL = True
    RUN_LST_ZONAL_STATS = True
    RUN_NDVI_LST_LANDUSE = True

    # --- 1. NDVI Temporal Analysis ---
    if RUN_NDVI_TEMPORAL:
        print("\n===== Starting NDVI Temporal Analysis =====")
        boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"
        lst_cube, ndvi_cube, lst_changes = run_ndvi(data_dir, boundary_file)
        print("NDVI Temporal Analysis completed!\n")

    # --- 2. Land-Use LST Zonal Statistics ---
    if RUN_LST_ZONAL_STATS:
        print("\n===== Starting Land-Use LST Zonal Statistics =====")
        landuse_file = data_dir / "kathmandu_landuse_osm.gpkg"
        lst_file = data_dir / "May2025/lst_may_2025_1.tif"
        output_file = data_dir / "output/lst_may_landuse_zonal_stats.csv"

        summary_df = run_lst_zonal_stats(
            data_dir=data_dir,
            landuse_file=landuse_file,
            lst_file=lst_file,
            output_file=output_file
        )
        print("Land-Use LST Zonal Statistics completed!\n")
        print("Summary (first 5 rows):")
        print(summary_df.head())

    # --- 3. NDVI–LST–Landuse Analysis (multi-year) ---
    if RUN_NDVI_LST_LANDUSE:
        print("\n===== Starting NDVI–LST–Landuse Analysis =====")
        project_root = script_dir.parent
        results_2020, results_2025 = run_ndvi_lst(project_root)
        print("NDVI–LST–Landuse Analysis completed!\n")

    print("All requested analyses completed successfully!")
