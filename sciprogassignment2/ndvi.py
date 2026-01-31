"""
NDVI-LST Temporal Analysis for Kathmandu Valley
Examining the relationship between vegetation (NDVI) and land surface temperature (LST)
across different time periods using MODIS and Sentinel-2 data.
"""

import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from scipy import stats
from typing import List, Tuple
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration for file paths and parameters."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.may2025_data_dir = project_root / "data/May2025"
        self.may2020_data_dir = project_root / "data/May2020"
        self.kathmandu_gpkg = project_root / "data/ktm_bhktpr_ltpr_shapefile.gpkg"
        
        # 2025 files
        self.modis_files_2025 = [
            self.may2025_data_dir / "reprojected_MODIS_Grid_8Day_1km_LST_2025-05-01-2025-05-08.tif",
            self.may2025_data_dir / "reprojected_MODIS_Grid_8Day_1km_LST_2025-05-09-2025-05-16.tif",
            self.may2025_data_dir / "reprojected_MODIS_Grid_8Day_1km_LST_2025-05-17-2025-05-24.tif",
            self.may2025_data_dir / "reprojected_MODIS_Grid_8Day_1km_LST_2025-05-25-2025-06-01.tif",
        ]
        self.red_file_2025 = self.may2025_data_dir / "Sentinel2_Band4_Kathmandu_20250509.tif"
        self.nir_file_2025 = self.may2025_data_dir / "Sentinel2_Band8_Kathmandu_20250509.tif"
        
        # 2020 files
        self.modis_files_2020 = [
            self.may2020_data_dir / "lst_may_2020_1.tif",
            self.may2020_data_dir / "lst_may_2020_2.tif",
            self.may2020_data_dir / "lst_may_2020_3.tif",
            self.may2020_data_dir / "lst_may_2020_4.tif",
        ]
        self.red_file_2020 = self.may2020_data_dir / "sentinel_2_band_04_may_2020.tif"
        self.nir_file_2020 = self.may2020_data_dir / "sentinel_2_band_08_may_2020.tif"


# =========================
# Data Loading & Processing
# =========================
class DataProcessor:
    """Handles all raster data loading and processing operations."""
    
    @staticmethod
    def load_lst_celsius(modis_files: List[Path]) -> List[xr.DataArray]:
        """Load MODIS LST files, mask fill values (0), and convert to Celsius."""
        lst_list = []
        for f in modis_files:
            da = rxr.open_rasterio(f, masked=False).squeeze()
            da_masked = da.where(da != 0)
            da_celsius = da_masked - 273.15
            da_celsius.attrs["units"] = "°C"
            lst_list.append(da_celsius)
        return lst_list
    
    @staticmethod
    def average_lst_list(lst_list: List[xr.DataArray]) -> xr.DataArray:
        """Stack LST arrays along a 'time' dimension and compute pixel-wise mean."""
        lst_stack = xr.concat(lst_list, dim="time")
        avg_lst = lst_stack.mean(dim="time", skipna=True)
        avg_lst.attrs["units"] = "°C"
        return avg_lst
    
    @staticmethod
    def load_sentinel_bands(red_file: Path, nir_file: Path) -> Tuple[xr.DataArray, xr.DataArray]:
        """Load Sentinel-2 Red and NIR bands."""
        red = rxr.open_rasterio(red_file, masked=True).squeeze().load()
        nir = rxr.open_rasterio(nir_file, masked=True).squeeze().load()
        return red, nir
    
    @staticmethod
    def compute_ndvi(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
        """
        Compute NDVI from NIR and Red bands.
        Uses numpy arrays to avoid xarray coordinate alignment issues.
        """
        epsilon = 1e-10
        
        # Do math in pure numpy to avoid coordinate matching issues
        nir_vals = nir.values
        red_vals = red.values
        
        ndvi_vals = (nir_vals - red_vals) / (nir_vals + red_vals + epsilon)
        
        # Reconstruct xarray with same coords/dims as NIR
        ndvi = xr.DataArray(
            ndvi_vals,
            coords=nir.coords,
            dims=nir.dims,
            attrs={"units": "", "long_name": "Normalized Difference Vegetation Index"}
        )
        
        # Copy spatial reference
        ndvi = ndvi.rio.write_crs(nir.rio.crs)
        ndvi = ndvi.rio.write_transform(nir.rio.transform())
        
        return ndvi


# =========================
# Analysis Functions
# =========================
class NDVILSTAnalyzer:
    """Performs statistical analysis on NDVI-LST relationships."""
    
    @staticmethod
    def compute_correlation(lst_da: xr.DataArray, ndvi_da: xr.DataArray) -> dict:
        """Compute simple correlation statistics between NDVI and LST, excluding zeros."""
        lst_vals = lst_da.values.flatten()
        ndvi_vals = ndvi_da.values.flatten()
        
        # CRITICAL: Filter out NaNs AND zeros
        mask = (~np.isnan(lst_vals)) & (~np.isnan(ndvi_vals)) & \
               (ndvi_vals > 0) & (lst_vals > 0)
        
        lst_clean = lst_vals[mask]
        ndvi_clean = ndvi_vals[mask]
        
        # Linear regression
        slope, intercept, r_value, p_reg, std_err = stats.linregress(ndvi_clean, lst_clean)
        
        # Predict LST using the regression line
        lst_predicted = slope * ndvi_clean + intercept
        
        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((lst_clean - lst_predicted)**2))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'rmse': rmse,
            'n_pixels': len(ndvi_clean)
        }
    
    @staticmethod
    def compute_zonal_stats(lst_da: xr.DataArray, ndvi_da: xr.DataArray, 
                           bins: int = 10) -> dict:
        """Compute LST statistics for different NDVI ranges (zonal analysis), excluding zeros."""
        lst_vals = lst_da.values.flatten()
        ndvi_vals = ndvi_da.values.flatten()
        
        # Filter out NaNs AND zeros
        mask = (~np.isnan(lst_vals)) & (~np.isnan(ndvi_vals)) & \
               (ndvi_vals > 0) & (lst_vals > 0)
        
        lst_clean = lst_vals[mask]
        ndvi_clean = ndvi_vals[mask]
        
        # Create NDVI bins
        ndvi_bins = np.linspace(ndvi_clean.min(), ndvi_clean.max(), bins + 1)
        bin_indices = np.digitize(ndvi_clean, ndvi_bins)
        
        stats_by_zone = []
        for i in range(1, bins + 1):
            zone_mask = bin_indices == i
            if zone_mask.sum() > 0:
                zone_lst = lst_clean[zone_mask]
                zone_ndvi = ndvi_clean[zone_mask]
                stats_by_zone.append({
                    'ndvi_range': (ndvi_bins[i-1], ndvi_bins[i]),
                    'ndvi_mean': zone_ndvi.mean(),
                    'lst_mean': zone_lst.mean(),
                    'lst_std': zone_lst.std(),
                    'lst_min': zone_lst.min(),
                    'lst_max': zone_lst.max(),
                    'pixel_count': zone_mask.sum()
                })
        
        return stats_by_zone


# =========================
# Visualization Functions
# =========================
class Visualizer:
    """Creates all visualization outputs."""
    
    @staticmethod
    def identify_hotspots(data_array: xr.DataArray, percentile: float = 90) -> xr.DataArray:
        """Identify hotspots (high values) and coldspots (low values)."""
        vals = data_array.values.flatten()
        vals_clean = vals[~np.isnan(vals)]
        
        high_threshold = np.percentile(vals_clean, percentile)
        low_threshold = np.percentile(vals_clean, 100 - percentile)
        
        hotspots = data_array.where(data_array >= high_threshold)
        coldspots = data_array.where(data_array <= low_threshold)
        
        return hotspots, coldspots, high_threshold, low_threshold
    
    @staticmethod
    def plot_year_comparison(results_dict: dict, save_path: Path = None):
        """Create side-by-side comparison of all years on one page."""
        years = sorted(results_dict.keys())
        n_years = len(years)
        
        # Create figure with 3 rows: NDVI maps, LST maps, Scatter plots
        fig = plt.figure(figsize=(6 * n_years, 15))
        gs = fig.add_gridspec(3, n_years, hspace=0.25, wspace=0.3)
        
        for i, year in enumerate(years):
            result = results_dict[year]
            lst_da = result['lst']
            ndvi_da = result['ndvi']
            stats = result['stats']
            
            # Row 1: NDVI maps with coldspots highlighted
            ax_ndvi = fig.add_subplot(gs[0, i])
            ndvi_da.plot(ax=ax_ndvi, cmap="YlGn", vmin=0, vmax=1,
                        cbar_kwargs={'label': 'NDVI'})
            
            # Highlight NDVI coldspots (low vegetation)
            _, ndvi_coldspots, _, ndvi_low = Visualizer.identify_hotspots(ndvi_da, percentile=90)
            if ndvi_coldspots is not None:
                ndvi_coldspots.plot.contourf(ax=ax_ndvi, levels=[ndvi_low, ndvi_da.min().values],
                                            colors=['none'], hatches=['///'], alpha=0.3, 
                                            add_colorbar=False)
            
            ax_ndvi.set_title(f"NDVI - May {year}", fontsize=12, fontweight='bold')
            ax_ndvi.set_xlabel("Longitude")
            ax_ndvi.set_ylabel("Latitude")
            
            # Row 2: LST maps with hotspots highlighted
            ax_lst = fig.add_subplot(gs[1, i])
            lst_da.plot(ax=ax_lst, cmap="RdYlBu_r",
                       cbar_kwargs={'label': 'Temperature (°C)'})
            
            # Highlight LST hotspots (high temperature areas)
            lst_hotspots, _, lst_high, _ = Visualizer.identify_hotspots(lst_da, percentile=90)
            if lst_hotspots is not None:
                lst_hotspots.plot.contour(ax=ax_lst, levels=5, colors='darkred', 
                                         linewidths=2, alpha=0.8)
            
            ax_lst.set_title(f"LST - May {year}", fontsize=12, fontweight='bold')
            ax_lst.set_xlabel("Longitude")
            ax_lst.set_ylabel("Latitude")
            
            # Row 3: Scatter plots with regression
            ax_scatter = fig.add_subplot(gs[2, i])
            
            lst_vals = lst_da.values.flatten()
            ndvi_vals = ndvi_da.values.flatten()
            
            # CRITICAL: Filter out zeros AND nans
            mask = (~np.isnan(lst_vals)) & (~np.isnan(ndvi_vals)) & \
                   (ndvi_vals > 0) & (lst_vals > 0)
            
            lst_clean = lst_vals[mask]
            ndvi_clean = ndvi_vals[mask]
            
            # Scatter plot with transparency
            ax_scatter.scatter(ndvi_clean, lst_clean, s=1, alpha=0.3, c='steelblue')
            
            # Add regression line
            if len(ndvi_clean) > 0:
                ndvi_range = np.linspace(ndvi_clean.min(), ndvi_clean.max(), 100)
                lst_pred = stats['slope'] * ndvi_range + stats['intercept']
                ax_scatter.plot(ndvi_range, lst_pred, 'r-', linewidth=2.5, 
                              label=f'y = {stats["slope"]:.2f}x + {stats["intercept"]:.2f}')
            
            ax_scatter.set_xlabel("NDVI", fontsize=11)
            ax_scatter.set_ylabel("LST (°C)", fontsize=11)
            ax_scatter.set_title(f"NDVI-LST Relationship ({year})", fontsize=12, fontweight='bold')
            ax_scatter.grid(True, alpha=0.3)
            
            # Add stats text
            stats_text = (f"R² = {stats['r_squared']:.3f}\n"
                         f"RMSE = {stats['rmse']:.2f}°C\n"
                         f"n = {len(ndvi_clean):,}")
            ax_scatter.text(0.05, 0.95, stats_text, transform=ax_scatter.transAxes,
                          verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
            ax_scatter.legend(loc='lower right', fontsize=9)
        
        plt.suptitle(f"NDVI-LST Temporal Analysis: Kathmandu Valley", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# =========================
# Main Workflow
# =========================
class NDVILSTWorkflow:
    """Orchestrates the entire analysis workflow."""
    
    def __init__(self, config: Config):
        self.config = config
        self.kathmandu = gpd.read_file(config.kathmandu_gpkg)
        self.results = {}
    
    def process_year(self, year: str, modis_files: List[Path], 
                    red_file: Path, nir_file: Path) -> dict:
        """Process a single year's data."""
        print(f"\n{'='*50}")
        print(f"Processing {year} data...")
        print(f"{'='*50}")
        
        # 1. Load and average LST
        print("Loading MODIS LST data...")
        lst_list = DataProcessor.load_lst_celsius(modis_files)
        avg_lst = DataProcessor.average_lst_list(lst_list)
        
        # 2. Clip to Kathmandu
        print("Clipping to Kathmandu boundary...")
        avg_lst_clipped = avg_lst.rio.clip(self.kathmandu.geometry, 
                                          self.kathmandu.crs, drop=True)
        
        # 3. Load Sentinel bands and compute NDVI
        print("Loading Sentinel-2 bands...")
        red, nir = DataProcessor.load_sentinel_bands(red_file, nir_file)
        
        print("Computing NDVI...")
        ndvi = DataProcessor.compute_ndvi(nir, red)
        
        # 4. Align LST to NDVI
        print("Aligning LST to NDVI CRS...")
        lst_aligned = avg_lst_clipped.rio.reproject_match(ndvi)
        
        # 5. Compute statistics
        print("Computing correlation statistics...")
        stats = NDVILSTAnalyzer.compute_correlation(lst_aligned, ndvi)
        
        print("Computing zonal statistics...")
        zonal_stats = NDVILSTAnalyzer.compute_zonal_stats(lst_aligned, ndvi)
        
        # Print summary
        print(f"\nResults for {year}:")
        print(f"  R² (goodness of fit): {stats['r_squared']:.3f}")
        print(f"  RMSE: {stats['rmse']:.2f} °C")
        print(f"  Slope: {stats['slope']:.2f} °C per NDVI unit")
        print(f"  Mean NDVI: {np.nanmean(ndvi.values):.3f}")
        print(f"  Mean LST: {np.nanmean(lst_aligned.values):.2f} °C")
        
        return {
            'lst': lst_aligned,
            'ndvi': ndvi,
            'stats': stats,
            'zonal_stats': zonal_stats,
            'year': year
        }
    
    def run_analysis(self, years_to_process: List[str] = None):
        """Run complete analysis for specified years."""
        if years_to_process is None:
            years_to_process = ['2020', '2025']
        
        # Process each year
        if '2020' in years_to_process:
            self.results['2020'] = self.process_year(
                '2020',
                self.config.modis_files_2020,
                self.config.red_file_2020,
                self.config.nir_file_2020
            )
        
        if '2025' in years_to_process:
            self.results['2025'] = self.process_year(
                '2025',
                self.config.modis_files_2025,
                self.config.red_file_2025,
                self.config.nir_file_2025
            )
        
        # Generate single comprehensive visualization
        print(f"\n{'='*50}")
        print("Generating visualization...")
        print(f"{'='*50}")
        
        Visualizer.plot_year_comparison(self.results)
        
        print("\n✅ Analysis complete!")
        return self.results


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent
    config = Config(project_root)
    
    # Run workflow
    workflow = NDVILSTWorkflow(config)
    results = workflow.run_analysis(years_to_process=['2020', '2025'])
    
    # Optional: Save results
    print("\nAnalysis results stored in memory and can be accessed via 'results' variable")