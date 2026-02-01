# ============================================================
# Land Use Derivation from OpenStreetMap (OSM)
# Study Area: Kathmandu Valley, Nepal
# ============================================================

"""
This script derives land-use zones for the Kathmandu Valley, Nepal,
using OpenStreetMap (OSM) data. The workflow includes:

1. Loading the study area boundary.
2. Downloading OSM features relevant to land use.
3. Cleaning and filtering OSM data to retain only polygons.
4. Classifying land-use types based on OSM tags.
5. Reprojecting to a projected CRS suitable for area calculation.
6. Dissolving polygons by land-use class.
7. Clipping the final land-use zones to the study area boundary.
8. Saving the results as a GeoPackage file.

Dependencies:
    - geopandas
    - osmnx
"""

import geopandas as gpd
import osmnx as ox

# ============================================================
# 1. LOAD KATHMANDU VALLEY BOUNDARY
# ============================================================

# File path to the Kathmandu Valley shapefile
boundary_fp = "data/ktm_bhktpr_ltpr_shapefile.gpkg"

# Load boundary
boundary = gpd.read_file(boundary_fp)

# Ensure the boundary is in WGS84 (EPSG:4326) for OSM compatibility
boundary = boundary.to_crs(epsg=4326)

# Extract geometry of the study area
study_area = boundary.geometry.iloc[0]


# ============================================================
# 2. DOWNLOAD OSM LAND-USE RELATED FEATURES
# ============================================================

# Define OSM tags to retrieve
tags = {
    "landuse": True,
    "building": True,
    "natural": True,
    "leisure": True,
    "amenity": True,
    "historic": True,
    "tourism": True,
    "man_made": True
}

# Download features from OSM within the study area polygon
osm = ox.features_from_polygon(study_area, tags)

print("OSM features downloaded:", len(osm))


# ============================================================
# 3. CLEAN OSM DATA (KEEP POLYGONS ONLY)
# ============================================================

# Retain only polygonal features for area-based land-use analysis
osm = osm[osm.geometry.type.isin(["Polygon", "MultiPolygon"])]

print("Polygon features retained:", len(osm))


# ============================================================
# 4. LAND-USE CLASSIFICATION FUNCTION
# ============================================================

def classify_landuse(row):
    """
    Classifies a single OSM feature into a land-use zone based on tags.

    Parameters:
        row (pd.Series): A row from the OSM GeoDataFrame.

    Returns:
        str: Land-use classification.
    """

    # Agricultural zone
    if row.get("landuse") in ["farmland", "orchard", "vineyard", "meadow"]:
        return "Agricultural zone"

    # Residential zone
    if row.get("landuse") == "residential":
        return "Residential zone"

    # Commercial zone
    if row.get("landuse") == "commercial" or row.get("building") == "commercial":
        return "Commercial zone"

    # Industrial zone
    if row.get("landuse") == "industrial" or row.get("building") == "industrial":
        return "Industrial zone"

    # Area of mines and minerals
    if row.get("landuse") == "quarry" or row.get("man_made") == "mine":
        return "Area of mines and minerals"

    # Forest zone
    if row.get("landuse") == "forest" or row.get("natural") == "wood":
        return "Forest zone"

    # Public use zone
    if row.get("leisure") in ["park", "playground"]:
        return "Public use zone"
    if row.get("amenity") in ["school", "hospital", "university", "government", "parking"]:
        return "Public use zone"

    # Cultural and archaeological importance
    if row.get("historic") is not None:
        return "Area of cultural and archaeological importance"
    if row.get("tourism") == "attraction":
        return "Area of cultural and archaeological importance"
    if row.get("amenity") == "place_of_worship":
        return "Area of cultural and archaeological importance"

    # Other categories (default)
    return "Other categories"

# Apply land-use classification
osm["land_use_class"] = osm.apply(classify_landuse, axis=1)

print("Land-use classification completed.")


# ============================================================
# 5. REPROJECT TO PROJECTED CRS (FOR AREA / ZONAL ANALYSIS)
# ============================================================

# Use UTM zone for Kathmandu (EPSG:32645) for accurate area calculations
osm = osm.to_crs(epsg=32645)
boundary = boundary.to_crs(epsg=32645)


# ============================================================
# 6. DISSOLVE BY LAND-USE CLASS
# ============================================================

# Aggregate polygons by land-use class
landuse_zones = osm.dissolve(by="land_use_class").reset_index()

# Keep only relevant columns
landuse_zones = landuse_zones[["land_use_class", "geometry"]]

# Calculate area in square meters and square kilometers
landuse_zones["area_m2"] = landuse_zones.area
landuse_zones["area_km2"] = landuse_zones.area / 1e6

print("Dissolved land-use zones created:")
print(landuse_zones.index.tolist())


# ============================================================
# 7. CLIP TO KATHMANDU BOUNDARY (SAFETY STEP)
# ============================================================

# Ensure all zones are clipped to the official study area
landuse_zones = gpd.clip(landuse_zones, boundary)


# ============================================================
# 8. SAVE OUTPUT
# ============================================================

output_fp = "data/kathmandu_landuse_osm.gpkg"

# Save as GeoPackage
landuse_zones.to_file(output_fp, driver="GPKG")

print("Land-use shapefile saved to:")
print(output_fp)


# ============================================================
# END OF SCRIPT
# ============================================================
