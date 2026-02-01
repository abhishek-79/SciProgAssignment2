"""
Main script for running NDVI-LST-Landuse analysis
"""

from pathlib import Path
from ndvi import run_analysis

if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../data"
    boundary_file = data_dir / "ktm_bhktpr_ltpr_shapefile.gpkg"
    
    # Run analysis
    print("Starting NDVI-LST temporal analysis...")
    lst_cube, ndvi_cube, lst_changes = run_analysis(data_dir, boundary_file)
    print("Analysis complete!")






