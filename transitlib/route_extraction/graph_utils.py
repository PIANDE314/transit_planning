import networkx as nx
import osmnx as ox
from typing import List, Tuple, Dict
import geopandas as gpd
from joblib import Parallel, delayed
from transitlib.config import Config

cfg = Config()

def _collapse_to_simple(G_multi: nx.MultiDiGraph, weightAttr="length"):
    G = nx.Graph()
    for u, v, data in G_multi.edges(data=True):
        length = data.get(weightAttr)
        if length is None:
            raise KeyError(f"Edge {(u,v)} missing {weightAttr}")
        if G.has_edge(u, v):
            G[u][v][weightAttr] = min(G[u][v][weightAttr], length)
        else:
            G.add_edge(u, v, **{weightAttr: length})
    for n, attr in G_multi.nodes(data=True):
        G.nodes[n].update(attr)
    return G

def map_stops_to_nodes(G_latlon: nx.Graph, stops: gpd.GeoDataFrame) -> List[int]:
    stops_ll = stops.to_crs(G_latlon.graph['crs'])
    xs, ys = stops_ll.geometry.x.values, stops_ll.geometry.y.values
    return ox.distance.nearest_nodes(G_latlon, xs, ys)

def _run_sssp(G: nx.Graph, source: int, targets: List[int], weight: str = "length") -> Tuple[int, Dict[int, float]]:
    dist = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    return source, {t: dist[t] for t in targets if t in dist and t != source}

def build_stop_graph(
    G_latlon: nx.Graph,
    od_counts: Dict[Tuple[int, int], int],
    stops: gpd.GeoDataFrame,
    footfall_col: str
) -> nx.Graph:
    # Step 1: Snap stops → nearest OSM nodes
    node_ids = map_stops_to_nodes(G_latlon, stops)
    stops = stops.copy()
    stops['node_id'] = node_ids

    # Step 2: Collapse duplicates by summing footfall
    agg = stops.groupby('node_id')[footfall_col].sum().reset_index()
    unique_nodes = agg['node_id'].tolist()
    footfalls = dict(zip(agg['node_id'], agg[footfall_col]))

    print(f"[INFO] Running Dijkstra for {len(unique_nodes)} nodes...", flush=True)
    G_simple = _collapse_to_simple(G_latlon, weightAttr="length")

    # Step 3: Parallel Dijkstra
    results = Parallel(n_jobs=cfg.get("n_jobs", 4), backend="threads")(
        delayed(_run_sssp)(G_simple, u, unique_nodes) for u in unique_nodes
    )

    # Step 4: Aggregate pairwise distances
    lengths = {u: dists for u, dists in results}

    print(f"[INFO] Finished computing all pairwise distances.", flush=True)

    # Step 5: Build the simplified graph
    G = nx.Graph()
    for u in unique_nodes:
        G.add_node(u,
                   footfall=float(footfalls[u]),
                   lat=G_latlon.nodes[u]['y'],
                   lon=G_latlon.nodes[u]['x'])

    for i, u in enumerate(unique_nodes):
        for v in unique_nodes[i+1:]:
            d = lengths[u].get(v)
            if d is None or d == 0:
                continue
            demand = od_counts.get((u, v), 0) + od_counts.get((v, u), 0)
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
