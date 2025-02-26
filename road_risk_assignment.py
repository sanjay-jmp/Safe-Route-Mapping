import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load crime data
crime_df = pd.read_csv("crime_data_updated.csv")

# Extract only time (HH:MM:SS), removing any date part
crime_df["TIME_BIN"] = pd.to_datetime(crime_df["TIME_BIN"]).dt.strftime("%H:%M:%S")

# Define city
place_name = "Los Angeles, California, USA"

# Download street network
G = ox.graph_from_place(place_name, network_type="drive")

# Convert graph to GeoDataFrame
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# Reset index for easy access to 'u' and 'v'
gdf_edges = gdf_edges.reset_index()

# Define Earth's radius
EARTH_RADIUS_KM = 6371  
RADIUS_KM = 0.2  # 200m search radius

# Precompute KDTree for each TIME_BIN
time_bins = crime_df["TIME_BIN"].unique()
crime_trees = {}
crime_data_by_bin = {}

for time_bin in time_bins:
    crime_filtered = crime_df[crime_df["TIME_BIN"] == time_bin]
    if not crime_filtered.empty:
        crime_locations = np.radians(crime_filtered[['LAT', 'LON']].values)
        crime_trees[time_bin] = cKDTree(crime_locations)
        crime_data_by_bin[time_bin] = crime_filtered["Risk Level_y"].values
    else:
        crime_trees[time_bin] = None
        crime_data_by_bin[time_bin] = None

# Function to compute risk level for an edge
def compute_risk(road_lat, road_lon, time_bin):
    """ Computes risk level for a given road midpoint and time bin """
    if crime_trees[time_bin] is None:
        return 0  # No crimes in this time bin

    # Convert road midpoint to radians
    road_point = np.radians([road_lat, road_lon])

    # Query nearby crimes
    idx = crime_trees[time_bin].query_ball_point(road_point, RADIUS_KM / EARTH_RADIUS_KM)

    if not idx:
        return 0  # No nearby crimes

    # Get risk values for these indices
    risk_values = crime_data_by_bin[time_bin][idx]

    # Compute mode of risk levels
    mode_values = pd.Series(risk_values).mode()
    
    return int(max(mode_values)) if not mode_values.empty else 0

# Compute risk levels for all time bins (Optimized)
for time_bin in time_bins:
    print(f"Processing risk for time bin: {time_bin}...")
    
    risks = []
    for edge in gdf_edges.itertuples(index=False):
        if edge.geometry is None or edge.geometry.is_empty:
            risks.append(0)
            continue

        # Compute road segment midpoint
        road_lat = (edge.geometry.coords[0][1] + edge.geometry.coords[-1][1]) / 2
        road_lon = (edge.geometry.coords[0][0] + edge.geometry.coords[-1][0]) / 2

        # Compute risk level
        risk_level = compute_risk(road_lat, road_lon, time_bin)
        risks.append(risk_level)

    gdf_edges[f"risk_{time_bin}"] = risks  # Assign risk values efficiently

# Convert back to NetworkX graph
gdf_edges = gdf_edges.set_index(["u", "v", "key"])
G_precomputed = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

#Save the graph
ox.save_graphml(G_precomputed, "los_angeles_precomputed_risk1.graphml")

print("✅ Preprocessing complete! Graph saved as 'los_angeles_precomputed_risk.graphml'.")
