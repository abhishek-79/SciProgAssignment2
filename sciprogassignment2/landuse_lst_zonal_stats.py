# ============================================================
# Zonal Statistics of May LST by Land-Use Class
# Study Area: Kathmandu Valley, Nepal
# ============================================================

import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

landuse_fp = "data/kathmandu_landuse_osm.gpkg"
lst_fp = "data/May2025\MODIS_LST_KTM_AVG_MAY_2025_WGS84_Celsius_clipped.tif"
output_csv = "data/lst_may_landuse_zonal_stats.csv"

# ------------------------------------------------------------
# 2. LOAD LAND-USE DATA
# ------------------------------------------------------------

landuse = gpd.read_file(landuse_fp)

print("Land-use classes:")
print(landuse["land_use_class"].unique())

# ------------------------------------------------------------
# 3. OPEN LST RASTER
# ------------------------------------------------------------

with rasterio.open(lst_fp) as src:
    lst = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

print("LST raster loaded (May only).")

# ------------------------------------------------------------
# 4. CRS ALIGNMENT
# ------------------------------------------------------------

if landuse.crs != crs:
    landuse = landuse.to_crs(crs)

# ------------------------------------------------------------
# 5. ZONAL STATISTICS
# ------------------------------------------------------------

stats = zonal_stats(
    landuse,
    lst,
    affine=transform,
    nodata=nodata,
    stats=["mean", "min", "max", "std", "count"],
    geojson_out=False
)

# ------------------------------------------------------------
# 6. BUILD RESULTS TABLE
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 7. AGGREGATE BY LAND-USE CLASS
# ------------------------------------------------------------

summary = (
    df.groupby("land_use_class")
      .mean(numeric_only=True)
      .reset_index()
)

# ------------------------------------------------------------
# 8. SAVE OUTPUT
# ------------------------------------------------------------

summary.to_csv(output_csv, index=False)

print("Zonal statistics saved to:")
print(output_csv)

# ============================================================
# END OF SCRIPT
# ============================================================
