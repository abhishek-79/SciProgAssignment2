# ============================================================
# NDVI–LST Analysis with Land-Use Conditioning
# Study Area: Kathmandu Valley, Nepal
# ============================================================

"""
This module performs a multi-year NDVI–LST analysis for Kathmandu Valley.

It exposes a single function `run_ndvi_lst_landuse_analysis(data_dir)` which:
- Loads LST and NDVI rasters for 2020 and 2025
- Rasterizes land-use polygons
- Computes global and per-land-use NDVI–LST correlations
- Attaches mean NDVI and LST to land-use polygons
- Saves CSV and GeoPackage outputs
"""

# ============================================================
# Imports
# ============================================================

import rioxarray as rxr
import xarray as xr
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy import stats
from rasterio.features import rasterize
import pandas as pd

# ============================================================
# Configuration
# ============================================================

class Config:
    """Configuration container for file paths."""
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_2025 = project_root / "data/May2025"
        self.data_2020 = project_root / "data/May2020"
        self.kathmandu_boundary = project_root / "data/ktm_bhktpr_ltpr_shapefile.gpkg"
        self.landuse_gpkg = project_root / "data/kathmandu_landuse_osm.gpkg"

        self.modis_2025 = sorted(self.data_2025.glob("*.tif"))
        self.red_2025 = self.data_2025 / "sentinel_2_band_04_may_2025.tif"
        self.nir_2025 = self.data_2025 / "sentinel_2_band_08_may_2025.tif"

        self.modis_2020 = sorted(self.data_2020.glob("*.tif"))
        self.red_2020 = self.data_2020 / "sentinel_2_band_04_may_2020.tif"
        self.nir_2020 = self.data_2020 / "sentinel_2_band_08_may_2020.tif"

# ============================================================
# Data Processor
# ============================================================

class DataProcessor:
    @staticmethod
    def load_lst(files):
        """Load one or multiple LST rasters, convert to Celsius, return mean across time."""
        lst_list = []
        for f in files:
            da = rxr.open_rasterio(f, masked=True).squeeze()
            da = da.where(da != 0)
            da = da - 273.15
            da.attrs["units"] = "°C"
            lst_list.append(da)
        return xr.concat(lst_list, dim="time").mean(dim="time", skipna=True)

    @staticmethod
    def load_ndvi(red_fp, nir_fp):
        """Compute NDVI from Sentinel-2 red and NIR bands."""
        red = rxr.open_rasterio(red_fp, masked=True).squeeze()
        nir = rxr.open_rasterio(nir_fp, masked=True).squeeze()
        eps = 1e-10
        ndvi_vals = (nir.values - red.values) / (nir.values + red.values + eps)
        ndvi = xr.DataArray(ndvi_vals, coords=nir.coords, dims=nir.dims, attrs={"long_name": "NDVI"})
        ndvi = ndvi.rio.write_crs(nir.rio.crs)
        ndvi = ndvi.rio.write_transform(nir.rio.transform())
        return ndvi

# ============================================================
# Land-Use Rasterization
# ============================================================

def rasterize_landuse(landuse_gdf, reference_da):
    """Rasterize land-use polygons to match a reference raster."""
    shapes = ((geom, code) for geom, code in zip(landuse_gdf.geometry, landuse_gdf["land_use_code"]))
    raster = rasterize(shapes, out_shape=reference_da.shape, transform=reference_da.rio.transform(), fill=0, dtype="uint8")
    return xr.DataArray(raster, coords=reference_da.coords, dims=reference_da.dims, name="land_use")

# ============================================================
# Analyzer
# ============================================================

class Analyzer:
    @staticmethod
    def global_correlation(lst, ndvi):
        l = lst.values.flatten()
        n = ndvi.values.flatten()
        mask = (~np.isnan(l)) & (~np.isnan(n)) & (l > -273) & (n >= -1) & (n <= 1)
        slope, intercept, r, _, _ = stats.linregress(n[mask], l[mask])
        rmse = np.sqrt(np.mean((l[mask] - (slope * n[mask] + intercept))**2))
        return {"slope": slope, "intercept": intercept, "r_squared": r**2, "rmse": rmse, "n_pixels": mask.sum()}

    @staticmethod
    def correlation_by_landuse(lst, ndvi, landuse_da, class_lookup):
        results = []
        for code, name in class_lookup.items():
            mask = landuse_da.values == code
            l = lst.values[mask]
            n = ndvi.values[mask]
            valid = (~np.isnan(l)) & (~np.isnan(n)) & (l > -273) & (n >= -1) & (n <= 1)
            slope, _, r, _, _ = stats.linregress(n[valid], l[valid]) if valid.sum() > 0 else (np.nan, np.nan, np.nan, np.nan, np.nan)
            results.append({"land_use": name, "r_squared": float(r**2) if valid.sum() > 0 else np.nan,
                            "slope": float(slope) if valid.sum() > 0 else np.nan,
                            "n_pixels": int(valid.sum())})
        return results

# ============================================================
# Workflow
# ============================================================

class Workflow:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kathmandu = gpd.read_file(cfg.kathmandu_boundary)
        self.landuse = gpd.read_file(cfg.landuse_gpkg)
        self.class_lookup = {1: "Agricultural zone", 2: "Residential zone", 3: "Commercial zone",
                             4: "Industrial zone", 5: "Forest zone", 6: "Public use zone",
                             7: "Cultural/archaeological", 8: "Mines and minerals"}
        reverse = {v: k for k, v in self.class_lookup.items()}
        self.landuse["land_use_class"] = self.landuse["land_use_class"].str.strip()
        self.landuse["land_use_code"] = self.landuse["land_use_class"].map(reverse)

    def run_year(self, year, modis, red, nir):
        lst = DataProcessor.load_lst(modis)
        
        # # Ensure vector is in raster CRS
        # self.kathmandu = self.kathmandu.to_crs(lst.rio.crs)

        # # Optional: check overlap before clipping
        # vector_bounds = self.kathmandu.total_bounds
        # raster_bounds = lst.rio.bounds()
        # print("Raster bounds:", raster_bounds)
        # print("Vector bounds:", vector_bounds)

        lst = lst.rio.clip(self.kathmandu.geometry, lst.rio.crs, drop=True, all_touched=True)
        
        ndvi = DataProcessor.load_ndvi(red, nir)
        lst = lst.rio.reproject_match(ndvi)
        landuse = self.landuse.to_crs(ndvi.rio.crs)
        landuse_raster = rasterize_landuse(landuse, ndvi)
        global_stats = Analyzer.global_correlation(lst, ndvi)
        lu_stats = Analyzer.correlation_by_landuse(lst, ndvi, landuse_raster, self.class_lookup)
        landuse_gdf = self.attach_raster_stats_to_landuse(landuse, ndvi, lst, year=year)
        landuse_gdf = landuse_gdf.to_crs("EPSG:4326")  # Transform to WGS84 for output
        return {"lst": lst, "ndvi": ndvi, "global_stats": global_stats, "landuse_stats": lu_stats, "landuse_gdf": landuse_gdf}

    def attach_raster_stats_to_landuse(self, landuse_gdf, ndvi_da, lst_da, year):
        landuse_gdf = landuse_gdf.copy()
        mean_ndvi_list, mean_lst_list = [], []
        for geom in landuse_gdf.geometry:
            mask = rasterize([(geom, 1)], out_shape=ndvi_da.shape, transform=ndvi_da.rio.transform(), fill=0, dtype="uint8").astype(bool)
            ndvi_vals, lst_vals = ndvi_da.values[mask], lst_da.values[mask]
            ndvi_vals, lst_vals = ndvi_vals[~np.isnan(ndvi_vals)], lst_vals[~np.isnan(lst_vals)]
            mean_ndvi_list.append(ndvi_vals.mean() if ndvi_vals.size > 0 else np.nan)
            mean_lst_list.append(lst_vals.mean() if lst_vals.size > 0 else np.nan)
        landuse_gdf["mean_ndvi"] = mean_ndvi_list
        landuse_gdf["mean_lst"] = mean_lst_list
        if year is not None:
            landuse_gdf["land_use_class"] = landuse_gdf["land_use_class"] + f"_{year}"
        return landuse_gdf

    def save_landuse_stats_to_csv(self, results, output_fp):
        records = []
        for year, res in results.items():
            for r in res["landuse_stats"]:
                records.append({"year": year, "land_use_class": r["land_use"], "r_squared": r["r_squared"], "slope": r["slope"], "n_pixels": r["n_pixels"]})
        pd.DataFrame(records).to_csv(output_fp, index=False)

    def save_landuse_gdf(self, results, output_fp, crs="EPSG:4326"):
        gdf_list = []
        for res in results.values():
            gdf = res["landuse_gdf"].copy()
            gdf = gdf.to_crs(crs)  # Transform to common CRS
            gdf_list.append(gdf)

        combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        combined_gdf.to_file(output_fp, driver="GPKG")
    
# ============================================================
# Public Function for main.py
# ============================================================

def run_ndvi_lst_landuse_analysis(data_dir):
    """Run the multi-year NDVI–LST–Landuse workflow (2020 & 2025)."""
    project_root = Path(data_dir)
    cfg = Config(project_root)
    workflow = Workflow(cfg)
    results_2020 = workflow.run_year("2020", cfg.modis_2020, cfg.red_2020, cfg.nir_2020)
    results_2025 = workflow.run_year("2025", cfg.modis_2025, cfg.red_2025, cfg.nir_2025)
    workflow.save_landuse_stats_to_csv({"2020": results_2020}, project_root / "data/output/landuse_ndvi_lst_stats_2020.csv")
    workflow.save_landuse_stats_to_csv({"2025": results_2025}, project_root / "data/output/landuse_ndvi_lst_stats_2025.csv")
    workflow.save_landuse_gdf({"2020": results_2020, "2025": results_2025}, project_root / "data/output/landuse_ndvi_lst_stats.gpkg")
    return results_2020, results_2025
