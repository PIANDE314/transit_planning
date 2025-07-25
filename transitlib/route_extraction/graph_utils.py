import networkx as nx
import osmnx as ox
from typing import List, Tuple, Dict
import geopandas as gpd

"""
§ 3.3 Transit Routes Extraction — build graph and compute utilities (Fig 4A),
   using true simulated link demands D_uv from trip traversal counts.
"""

def map_stops_to_nodes(
    G_latlon: nx.Graph,
    stops: gpd.GeoDataFrame
) -> List[int]:
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
    footfall_col: str = "avg_footfall"
) -> nx.Graph:
    """
    Build a fully connected stop graph with:
      - Node attributes: 'footfall', 'lat', 'lon'
      - Edge attributes: 'length' (shortest-path) and 'demand' (D_uv, possibly zero)

    Args:
        G_latlon: OSM graph in EPSG:4326.
        od_counts: Counter of simulated traversals per (u, v).
        stops: GeoDataFrame of stops, must include footfall_col.
        footfall_col: column name in `stops` containing footfall values.

    Returns:
        G: Undirected NetworkX Graph.
    """
    node_ids = map_stops_to_nodes(G_latlon, stops)
    G = nx.Graph()

    # 1) Add nodes with footfall and coordinates
    for idx, node in enumerate(node_ids):
        G.add_node(
            node,
            footfall=float(stops.iloc[idx][footfall_col]),
            lat=G_latlon.nodes[node]["y"],
            lon=G_latlon.nodes[node]["x"],
        )

    # 2) Fully connect every stop pair, include zero-demand edges
    for i, u in enumerate(node_ids):
        for v in node_ids[i + 1 :]:
            try:
                length = nx.shortest_path_length(G_latlon, u, v, weight="length")
            except nx.NetworkXNoPath:
                continue

            # True link demand: sum of traversals in both directions
            demand = od_counts.get((u, v), 0) + od_counts.get((v, u), 0)

            # Always add the edge, even if demand == 0
            G.add_edge(u, v, length=length, demand=int(demand))

    return G


def compute_utilities(
    G: nx.Graph
) -> Tuple[
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
]:
    """
    Compute per-edge utilities Q_uv, F_uv, M_uv, and U_uv (Fig 4A).

    - Q_uv = normalized (D_uv / L_uv)
    - F_uv = normalized mean footfall of endpoints
    - M_uv = 1/|MST| if edge in MST else 0
    - U_uv = (Q_uv + F_uv + M_uv) / 3

    Returns:
        Q, F, M, U: dicts keyed by edge tuple (u, v).
    """
    Q, F, M, U = {}, {}, {}, {}

    # Q_uv: demand per length, normalized
    total_Q = 0.0
    for u, v, data in G.edges(data=True):
        q = data["demand"] / data["length"]
        Q[(u, v)] = q
        total_Q += q
    for e in Q:
        Q[e] /= total_Q

    # F_uv: mean footfall, normalized
    total_F = 0.0
    for u, v in G.edges():
        fu = G.nodes[u]["footfall"]
        fv = G.nodes[v]["footfall"]
        f = (fu + fv) / 2.0
        F[(u, v)] = f
        total_F += f
    for e in F:
        F[e] /= total_F

    # M_uv: MST membership
    mst = nx.minimum_spanning_tree(G, weight="length")
    mst_edges = set(mst.edges())
    inv_size = 1.0 / len(mst_edges) if mst_edges else 0.0
    for u, v in G.edges():
        M[(u, v)] = inv_size if ((u, v) in mst_edges or (v, u) in mst_edges) else 0.0

    # U_uv: average of Q, F, M
    for u, v in G.edges():
        q = Q.get((u, v), Q.get((v, u), 0.0))
        f = F.get((u, v), F.get((v, u), 0.0))
        m = M.get((u, v), M.get((v, u), 0.0))
        U[(u, v)] = (q + f + m) / 3.0

    return Q, F, M, U
