import networkx as nx
import osmnx as ox
from typing import List, Tuple, Dict
import geopandas as gpd

def map_stops_to_nodes(G_latlon: nx.Graph, stops: gpd.GeoDataFrame) -> List[int]:
    """
    Snap each stop to its nearest OSM node in the geographic (lat/lon) graph.
    """
    stops_ll = stops.to_crs(G_latlon.graph['crs'])
    xs, ys = stops_ll.geometry.x.values, stops_ll.geometry.y.values
    return ox.distance.nearest_nodes(G_latlon, xs, ys)

def build_stop_graph(
    G_latlon: nx.Graph,
    od_counts: Dict[Tuple[int, int], int],
    stops: gpd.GeoDataFrame,
    footfall_col: str
) -> nx.Graph:
    """
    Build a fully connected stop graph with:
      - Node attributes: 'footfall', 'lat', 'lon'
      - Edge attributes: 'length' (shortest-path) and 'demand'
    """
    node_ids = map_stops_to_nodes(G_latlon, stops)
    G = nx.Graph()

    for idx, node in enumerate(node_ids):
        G.add_node(
            node,
            footfall=float(stops.iloc[idx][footfall_col]),
            lat=G_latlon.nodes[node]["y"],
            lon=G_latlon.nodes[node]["x"],
        )

    for i, u in enumerate(node_ids):
        for v in node_ids[i + 1 :]:
            try:
                length = nx.shortest_path_length(G_latlon, u, v, weight="length")
            except nx.NetworkXNoPath:
                continue
            demand = od_counts.get((u, v), 0) + od_counts.get((v, u), 0)
            G.add_edge(u, v, length=length, demand=int(demand))

    return G

def compute_utilities(G: nx.Graph) -> Tuple[Dict, Dict, Dict, Dict]:
    Q, F, M, U = {}, {}, {}, {}

    total_Q = 0.0
    for u, v, data in G.edges(data=True):
        q = data["demand"] / data["length"]
        Q[(u, v)] = q
        total_Q += q
    for e in Q:
        Q[e] /= total_Q

    total_F = 0.0
    for u, v in G.edges():
        fu, fv = G.nodes[u]["footfall"], G.nodes[v]["footfall"]
        F[(u, v)] = (fu + fv) / 2.0
        total_F += F[(u, v)]
    for e in F:
        F[e] /= total_F

    mst_edges = set(nx.minimum_spanning_tree(G, weight="length").edges())
    inv_size = 1.0 / len(mst_edges) if mst_edges else 0.0
    for u, v in G.edges():
        M[(u, v)] = inv_size if ((u, v) in mst_edges or (v, u) in mst_edges) else 0.0

    for u, v in G.edges():
        q = Q.get((u, v), Q.get((v, u), 0.0))
        f = F.get((u, v), F.get((v, u), 0.0))
        m = M.get((u, v), M.get((v, u), 0.0))
        U[(u, v)] = (q + f + m) / 3.0

    return Q, F, M, U
