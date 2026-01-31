# ============================================================
# Land Use Derivation from OpenStreetMap (OSM)
# Study Area: Kathmandu Valley, Nepal
# ============================================================

import geopandas as gpd
import osmnx as ox

# ------------------------------------------------------------
# 1. LOAD KATHMANDU VALLEY BOUNDARY
# ------------------------------------------------------------

boundary_fp = "data/ktm_bhktpr_ltpr_shapefile.gpkg"

boundary = gpd.read_file(boundary_fp)

# OSM requires WGS84 (EPSG:4326)
boundary = boundary.to_crs(epsg=4326)

study_area = boundary.geometry.iloc[0]


# ------------------------------------------------------------
# 2. DOWNLOAD OSM LAND-USE RELATED FEATURES
# ------------------------------------------------------------

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

osm = ox.features_from_polygon(
    study_area,
    tags
)

print("OSM features downloaded:", len(osm))

# ------------------------------------------------------------
# 3. CLEAN OSM DATA (KEEP POLYGONS ONLY)
# ------------------------------------------------------------

osm = osm[osm.geometry.type.isin(["Polygon", "MultiPolygon"])]

print("Polygon features retained:", len(osm))

# ------------------------------------------------------------
# 4. LAND-USE CLASSIFICATION FUNCTION
# ------------------------------------------------------------

def classify_landuse(row):

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
    if row.get("amenity") in [
        "school", "hospital", "university",
        "government", "parking"
    ]:
        return "Public use zone"

    # Cultural and archaeological importance
    if row.get("historic") is not None:
        return "Area of cultural and archaeological importance"
    if row.get("tourism") == "attraction":
        return "Area of cultural and archaeological importance"
    if row.get("amenity") == "place_of_worship":
        return "Area of cultural and archaeological importance"

    # Other
    return "Other categories"

# Apply classification
osm["land_use_class"] = osm.apply(classify_landuse, axis=1)

print("Land-use classification completed.")

# ------------------------------------------------------------
# 5. REPROJECT TO PROJECTED CRS (FOR AREA / ZONAL ANALYSIS)
# ------------------------------------------------------------

# Use UTM zone for Kathmandu (EPSG:32645)
osm = osm.to_crs(epsg=32645)
boundary = boundary.to_crs(epsg=32645)

# ------------------------------------------------------------
# 6. DISSOLVE BY LAND-USE CLASS
# ------------------------------------------------------------

landuse_zones = osm.dissolve(by="land_use_class")

# Keep only safe columns
landuse_zones = landuse_zones.reset_index()
landuse_zones = landuse_zones[["land_use_class", "geometry"]]

# Calculate area (useful for reporting)
landuse_zones["area_m2"] = landuse_zones.area
landuse_zones["area_km2"] = landuse_zones.area / 1e6

print("Dissolved land-use zones created:")
print(landuse_zones.index.tolist())

# ------------------------------------------------------------
# 7. CLIP TO KATHMANDU BOUNDARY (SAFETY STEP)
# ------------------------------------------------------------

landuse_zones = gpd.clip(landuse_zones, boundary)

# ------------------------------------------------------------
# 8. SAVE OUTPUT
# ------------------------------------------------------------

output_fp = "data/kathmandu_landuse_osm.gpkg"
landuse_zones.to_file(output_fp, driver="GPKG")

print("Land-use shapefile saved to:")
print(output_fp)

# ============================================================
# END OF SCRIPT
# ============================================================
