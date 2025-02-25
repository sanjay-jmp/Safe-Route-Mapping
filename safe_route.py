import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

def get_time_bin(user_time, available_bins):
    """Find the correct time bin for a given user input time"""
    user_hour = int(user_time.split(":")[0])  # Extract hour
    selected_bin = available_bins[0]  # Default to earliest bin

    for bin_time in sorted(available_bins):
        bin_hour = int(bin_time.split(":")[0])  # Extract hour from time bin
        if user_hour >= bin_hour:
            selected_bin = bin_time

    return selected_bin

def find_safest_route(G, source_lat, source_lon, dest_lat, dest_lon, time_bin):
    """Find the safest route based on crime risk at the determined time bin"""
    risk_attr = f"risk_{time_bin}"

    if risk_attr not in next(iter(G.edges(data=True)))[-1]:
        raise ValueError(f"Time bin '{time_bin}' not found in graph edges.")

    # Convert risk values to numeric (float)
    for u, v, data in G.edges(data=True):
        if risk_attr in data:
            data[risk_attr] = float(data[risk_attr])  # Convert risk values to float

    # Find nearest nodes to source and destination
    start_node = ox.distance.nearest_nodes(G, source_lon, source_lat)
    end_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)

    # Compute the safest path
    safest_path = nx.astar_path(G, start_node, end_node, weight=risk_attr)

    return safest_path, time_bin

# ✅ Load road network graph with precomputed crime risk
G = ox.load_graphml(r"los_angeles_precomputed_risk1.graphml")

# ✅ Extract available time bins from graph edge attributes
sample_edge = next(iter(G.edges(data=True)))[-1]
available_time_bins = [key.replace("risk_", "") for key in sample_edge.keys() if key.startswith("risk_")]

# ✅ Get user input time
user_time = input("Enter time (HH:MM:SS format, e.g., 19:45:00): ").strip()

# ✅ Determine the correct time bin
time_bin = get_time_bin(user_time, available_time_bins)
print(f"Using time bin: {time_bin}")

# Example coordinates (Griffith Observatory to LAX)
source_lat, source_lon = 34.1184, -118.3004
dest_lat, dest_lon = 33.9416, -118.4085

# Find the safest route
safe_route, chosen_time_bin = find_safest_route(G, source_lat, source_lon, dest_lat, dest_lon, time_bin)

# Plot the road network
fig, ax = ox.plot_graph(G, show=False, close=False, edge_color='gray', node_size=0)

# Convert path nodes to coordinates
route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in safe_route]
route_x, route_y = zip(*route_coords)

# Plot the safest route
ax.plot(route_y, route_x, linewidth=3, color='blue', label=f'Safest Route ({chosen_time_bin})')

# Mark start and end points
ax.scatter([source_lon, dest_lon], [source_lat, dest_lat], c='red', s=100, label='Start/End')

plt.legend()
plt.show()
