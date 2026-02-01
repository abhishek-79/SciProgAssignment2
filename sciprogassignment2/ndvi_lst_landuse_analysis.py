"""
NDVI–LST Analysis with Land-Use Conditioning
Kathmandu Valley

This script integrates:
- MODIS LST (May)
- Sentinel-2 NDVI
- OSM-derived land-use zones
- Raster–vector integration
- NumPy, Xarray, and scientific programming best practices
"""

# ============================================================
# Imports
# ============================================================

import rioxarray as rxr
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from rasterio.features import rasterize
import pandas as pd

# ============================================================
# Configuration
# ============================================================

class Config:
    def __init__(self, project_root: Path):
        self.project_root = project_root

        self.data_2025 = project_root / "data/May2025"
        self.data_2020 = project_root / "data/May2020"

        self.kathmandu_boundary = project_root / "data/ktm_bhktpr_ltpr_shapefile.gpkg"
        self.landuse_gpkg = project_root / "data/kathmandu_landuse_osm.gpkg"

        self.modis_2025 = sorted(self.data_2025.glob("reprojected_MODIS*.tif"))
        self.red_2025 = self.data_2025 / "Sentinel2_Band4_Kathmandu_20250509.tif"
        self.nir_2025 = self.data_2025 / "Sentinel2_Band8_Kathmandu_20250509.tif"

        self.modis_2020 = sorted(self.data_2020.glob("lst_may_2020_*.tif"))
        self.red_2020 = self.data_2020 / "sentinel_2_band_04_may_2020.tif"
        self.nir_2020 = self.data_2020 / "sentinel_2_band_08_may_2020.tif"

# ============================================================
# Data Processing
# ============================================================

class DataProcessor:

    @staticmethod
    def load_lst(files):
        lst_list = []
        for f in files:
            da = rxr.open_rasterio(f, masked=True).squeeze()
            da = da.where(da != 0)  # mask invalid values
            da = da - 273.15        # convert Kelvin to Celsius if needed
            da.attrs["units"] = "°C"
            lst_list.append(da)
        # Stack along time if multiple files, then average
        return xr.concat(lst_list, dim="time").mean(dim="time", skipna=True)

    @staticmethod
    def load_ndvi(red_fp, nir_fp):
        red = rxr.open_rasterio(red_fp, masked=True).squeeze()
        nir = rxr.open_rasterio(nir_fp, masked=True).squeeze()

        eps = 1e-10
        ndvi_vals = (nir.values - red.values) / (nir.values + red.values + eps)

        ndvi = xr.DataArray(
            ndvi_vals,
            coords=nir.coords,
            dims=nir.dims,
            attrs={"long_name": "NDVI"}
        )

        ndvi = ndvi.rio.write_crs(nir.rio.crs)
        ndvi = ndvi.rio.write_transform(nir.rio.transform())
        return ndvi

# ============================================================
# Land-Use Rasterization
# ============================================================

def rasterize_landuse(landuse_gdf, reference_da):
    """
    Rasterize land-use polygons using the reference raster (NDVI/LST) grid.
    """
    shapes = (
        (geom, code)
        for geom, code in zip(
            landuse_gdf.geometry,
            landuse_gdf["land_use_code"]
        )
    )

    raster = rasterize(
        shapes=shapes,
        out_shape=reference_da.shape,
        transform=reference_da.rio.transform(),
        fill=0,
        dtype="uint8"
    )

    return xr.DataArray(
        raster,
        coords=reference_da.coords,
        dims=reference_da.dims,
        name="land_use"
    )

# ============================================================
# Analysis
# ============================================================

class Analyzer:

    @staticmethod
    def global_correlation(lst, ndvi):
        l = lst.values.flatten()
        n = ndvi.values.flatten()

        mask = (~np.isnan(l)) & (~np.isnan(n)) & (l > -273) & (n >= -1) & (n <= 1)
        slope, intercept, r, _, _ = stats.linregress(n[mask], l[mask])
        rmse = np.sqrt(np.mean((l[mask] - (slope * n[mask] + intercept)) ** 2))

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r**2,
            "rmse": rmse,
            "n_pixels": mask.sum()
        }

    @staticmethod
    def correlation_by_landuse(lst, ndvi, landuse_da, class_lookup):
        results = []

        for code, name in class_lookup.items():
            mask = landuse_da.values == code
            l = lst.values[mask]
            n = ndvi.values[mask]

            # Remove NaNs and invalid values
            valid = (~np.isnan(l)) & (~np.isnan(n)) & (l > -273) & (n >= -1) & (n <= 1)

            # **No threshold on number of pixels anymore**
            slope, _, r, _, _ = stats.linregress(n[valid], l[valid]) if valid.sum() > 0 else (np.nan, np.nan, np.nan, np.nan, np.nan)

            results.append({
                "land_use": name,
                "r_squared": float(r**2) if valid.sum() > 0 else np.nan,
                "slope": float(slope) if valid.sum() > 0 else np.nan,
                "n_pixels": int(valid.sum())
            })

        return results

# ============================================================
# Workflow
# ============================================================

class Workflow:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kathmandu = gpd.read_file(cfg.kathmandu_boundary)
        self.landuse = gpd.read_file(cfg.landuse_gpkg)

        # Assign numeric codes for rasterization
        self.class_lookup = {
            1: "Agricultural zone",
            2: "Residential zone",
            3: "Commercial zone",
            4: "Industrial zone",
            5: "Forest zone",
            6: "Public use zone",
            7: "Area of cultural and archaeological importance",
            8: "Area of mines and minerals"
        }

        reverse = {v: k for k, v in self.class_lookup.items()}
        # strip whitespace to prevent missing matches
        self.landuse["land_use_class"] = self.landuse["land_use_class"].str.strip()
        self.landuse["land_use_code"] = self.landuse["land_use_class"].map(reverse)

    def run_year(self, year, modis, red, nir):
        print(f"\nProcessing {year}...")

        # Load LST & clip to Kathmandu
        lst = DataProcessor.load_lst(modis)
        lst = lst.rio.clip(self.kathmandu.geometry, self.kathmandu.crs)

        # Load NDVI and reproject LST to NDVI grid
        ndvi = DataProcessor.load_ndvi(red, nir)
        lst = lst.rio.reproject_match(ndvi)

        # Rasterize land-use
        landuse = self.landuse.to_crs(ndvi.rio.crs)
        landuse_raster = rasterize_landuse(landuse, ndvi)

        # Compute correlations
        global_stats = Analyzer.global_correlation(lst, ndvi)
        lu_stats = Analyzer.correlation_by_landuse(lst, ndvi, landuse_raster, self.class_lookup)

        # --- Bidirectional raster-vector interaction ---
        # Attach mean NDVI & LST back to each land-use polygon
        landuse_stats_gdf = self.attach_raster_stats_to_landuse(landuse, ndvi, lst, year=year)
        
        print(f"Global R²: {global_stats['r_squared']:.3f}")
        print("Land-use specific NDVI–LST relationships:")
        for r in lu_stats:
            print(f"  {r['land_use']:<40} R²={r['r_squared']:.3f}  slope={r['slope']:.2f}  pixels={r['n_pixels']}")

        return {
            "lst": lst,
            "ndvi": ndvi,
            "global_stats": global_stats,
            "landuse_stats": lu_stats,
            "landuse_gdf": landuse_stats_gdf
        }

    def attach_raster_stats_to_landuse(self, landuse_gdf, ndvi_da, lst_da, year):
        """
        Compute per-polygon mean NDVI and LST and attach as columns.
        This is the raster → vector direction.
        """
        landuse_gdf = landuse_gdf.copy()
        mean_ndvi_list = []
        mean_lst_list = []

        for geom in landuse_gdf.geometry:
            # Create a mask of the raster pixels inside the polygon
            mask = rasterize(
                [(geom, 1)],
                out_shape=ndvi_da.shape,
                transform=ndvi_da.rio.transform(),
                fill=0,
                dtype="uint8"
            ).astype(bool)

            ndvi_vals = ndvi_da.values[mask]
            lst_vals = lst_da.values[mask]

            # Mask invalid values
            ndvi_vals = ndvi_vals[~np.isnan(ndvi_vals)]
            lst_vals = lst_vals[~np.isnan(lst_vals)]

            mean_ndvi_list.append(ndvi_vals.mean() if ndvi_vals.size > 0 else np.nan)
            mean_lst_list.append(lst_vals.mean() if lst_vals.size > 0 else np.nan)

        landuse_gdf["mean_ndvi"] = mean_ndvi_list
        landuse_gdf["mean_lst"] = mean_lst_list

        if year is not None:
            landuse_gdf["land_use_class"] = landuse_gdf["land_use_class"] + f"_{year}" 

        return landuse_gdf

    def save_landuse_stats_to_csv(self, results, output_fp):
        """
        Save land-use specific NDVI–LST statistics to CSV.
        """
        records = []
        for year, res in results.items():
            for r in res["landuse_stats"]:
                records.append({
                    "year": year,
                    "land_use_class": r["land_use"],
                    "r_squared": r["r_squared"],
                    "slope": r["slope"],
                    "n_pixels": r["n_pixels"]
                })
        df = pd.DataFrame(records)
        df.to_csv(output_fp, index=False)
        print(f"Land-use NDVI–LST stats saved to: {output_fp}")

    def save_landuse_gdf(self, results, output_fp):
        """
        Save land-use GeoDataFrame with attached raster statistics.
        """
        gdf_list = [res["landuse_gdf"] for res in results.values()]
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        combined_gdf.to_file(output_fp, driver="GPKG")
        print(f"Land-use GeoPackage with NDVI/LST stats saved to: {output_fp}")

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    project_root = Path(__file__).parent.parent
    cfg = Config(project_root)

    workflow = Workflow(cfg)

    results_2020 = workflow.run_year("2020", cfg.modis_2020, cfg.red_2020, cfg.nir_2020)
    results_2025 = workflow.run_year("2025", cfg.modis_2025, cfg.red_2025, cfg.nir_2025)

    workflow.save_landuse_stats_to_csv({"2020": results_2020}, project_root / "data/landuse_ndvi_lst_stats_2020.csv")
    workflow.save_landuse_stats_to_csv({"2025": results_2025}, project_root / "data/landuse_ndvi_lst_stats_2025.csv")

    # Save GPKG with raster stats attached
    workflow.save_landuse_gdf({"2020": results_2020, "2025": results_2025}, project_root / "data/landuse_ndvi_lst_stats.gpkg")