import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Load preprocessed crime data
crime_df = pd.read_csv("crime_data_updated.csv")

# Define city or region (Los Angeles)
place_name = "Los Angeles, California, USA"

# Download the street network from OpenStreetMap
G = ox.graph_from_place(place_name, network_type="drive")

# Convert graph to GeoDataFrame
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# Reset index to access `u` and `v` as columns
gdf_edges = gdf_edges.reset_index()

# Extract crime locations as NumPy array for fast lookup
crime_locations = np.radians(crime_df[['LAT', 'LON']].values)  # Convert to radians

# Use KDTree for fast spatial lookup
crime_tree = cKDTree(crime_locations)

# Define Earth's radius (in km)
EARTH_RADIUS_KM = 6371  
RADIUS_KM = 0.2  # 200m search radius

# Function to compute road segment risk
def calculate_risk(edge):
    if edge["geometry"] is None or edge["geometry"].is_empty:  # Ensure geometry exists
        return 0

    # Compute midpoint of the road segment
    road_lat = (edge["geometry"].coords[0][1] + edge["geometry"].coords[-1][1]) / 2
    road_lon = (edge["geometry"].coords[0][0] + edge["geometry"].coords[-1][0]) / 2

    # Convert to radians
    road_point = np.radians([road_lat, road_lon])

    # Query nearby crimes using KDTree
    idx = crime_tree.query_ball_point(road_point, RADIUS_KM / EARTH_RADIUS_KM)  # Radius in radians

    if not idx:
        return 0  # No crimes nearby
    
    # Get mode(s) and select the highest value if there are multiple modes
    risk_values = crime_df.iloc[idx]["Risk Level_y"].values
    mode_values = pd.Series(risk_values).mode()
    return int(max(mode_values))  # Select highest mode value

# Apply optimized function to all edges
gdf_edges["risk_level"] = gdf_edges.apply(calculate_risk, axis=1)

# Convert back to NetworkX graph
gdf_edges = gdf_edges.set_index(["u", "v", "key"])
G_risk = ox.graph_from_gdfs(gdf_nodes, gdf_edges)

# Add risk level as edge weight
for u, v, data in G_risk.edges(data=True):
    data["weight"] = 1 + data["risk_level"]  # Higher risk = Higher weight

# Save graph to disk
ox.save_graphml(G_risk, "los_angeles_risk_graph.graphml")
