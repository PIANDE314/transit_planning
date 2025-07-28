import networkx as nx
import osmnx as ox
from typing import List, Tuple, Dict
import geopandas as gpd
import networkit as nk
from networkit import nxadapter
from transitlib.config import Config

cfg = Config()

def _collapse_to_simple(G_multi: nx.MultiDiGraph, weightAttr="length"):
    G = nx.Graph()
    for u, v, data in G_multi.edges(data=True):
        length = data.get(weightAttr)
        if length is None:
            raise KeyError(f"Edge {(u,v)} missing {weightAttr}")
        # if edge already exists, keep the minimum length
        if G.has_edge(u, v):
            G[u][v][weightAttr] = min(G[u][v][weightAttr], length)
        else:
            G.add_edge(u, v, **{weightAttr: length})
    # copy node attributes too
    for n, attr in G_multi.nodes(data=True):
        G.nodes[n].update(attr)
    return G

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
    # 1) Snap stops → nearest OSM nodes
    node_ids = map_stops_to_nodes(G_latlon, stops)
    stops = stops.copy()
    stops['node_id'] = node_ids

    # 2) Collapse duplicates by summing footfall
    agg = (
        stops
        .groupby('node_id')[footfall_col]
        .sum()
        .reset_index()
    )
    # now agg.node_id are unique graph nodes, agg.footfall is total
    unique_nodes = agg['node_id'].tolist()
    footfalls   = dict(zip(agg['node_id'], agg[footfall_col]))
    print("Before NetworKit", flush=True)
    # 1) Build a NetworKit graph and compute APSP all at once
    G_simple = _collapse_to_simple(G_latlon, weightAttr="length")
    G_nk     = nxadapter.nx2nk(G_simple,  weightAttr="length")
    lengths = {}
    for idx, u in enumerate(unique_nodes, start=1):
        print(f"[NK] {idx}/{len(unique_nodes)}: Dijkstra from node {u}", flush=True)
        ssp.run()
        lengths[u] = ssp.getDistances()
    print("After NetworKit", flush=True)
        
    G = nx.Graph()
    for u in unique_nodes:
        G.add_node(u, footfall=float(footfalls[u]),
                      lat=G_latlon.nodes[u]['y'],
                      lon=G_latlon.nodes[u]['x'])

    # 2) Now add edges by simple lookup
    for i, u in enumerate(unique_nodes):
        for v in unique_nodes[i+1:]:
            d = lengths[u][v] if v in lengths[u] else None
            if d is None or d == 0:
                continue
            demand = od_counts.get((u,v),0) + od_counts.get((v,u),0)
            G.add_edge(u, v, length=d, demand=int(demand))

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
