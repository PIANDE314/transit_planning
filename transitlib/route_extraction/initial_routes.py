import random
import numpy as np
from typing import List, Tuple, Dict
import networkx as nx
from joblib import Parallel, delayed
from transitlib.config import Config

cfg = Config()

MIN_STOPS = cfg.get("min_stops")
MAX_STOPS = cfg.get("max_stops")

def sample_route_length(node_dist="gam", min_stops=MIN_STOPS, max_stops=MAX_STOPS) -> int:
    rng = np.random.default_rng()  # use numpy's new Generator
    while True:
        if node_dist == "gam":
            val = rng.gamma(shape=3.5, scale=6.0)
        elif node_dist == "norm":
            val = rng.normal(loc=21.0, scale=11.0)
        elif node_dist == "uni":
            val = rng.uniform(2.0, 40.0)
        else:
            raise ValueError(f"Unknown node_dist: {node_dist!r}")

        if val >= 2.0:
            return int(round(val))

def generate_initial_routes(
    G_stop: nx.Graph,
    U: Dict[Tuple[int,int], float],
    node_dist: str = "gam",
    min_stops: int = MIN_STOPS,
    max_stops: int = MAX_STOPS
) -> List[List[int]]:
    num_routes = cfg.get("num_initial_routes")
    routes = []

    adj: Dict[int, List[Tuple[int, float]]] = {}
    for (u, v), w in U.items():
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    def _build_one(_):
        edges = list(U.keys())
        weights = list(U.values())
        
        if sum(weights) == 0:
            u0, v0 = random.choice(edges)
        else:
            u0, v0 = random.choices(edges, weights=weights, k=1)[0]

        route = [u0, v0]
        target_len = sample_route_length(node_dist)

        while len(route) < target_len:
            start, end = route[0], route[-1]
            candidates = []
            for node in (start, end):
                for nbr, wt in adj.get(node, []):
                    if nbr not in route:
                        edge = (node, nbr) if (node, nbr) in U else (nbr, node)
                        candidates.append((edge, U.get(edge, 0.0)))
            edges, weights = zip(*candidates)

            if not candidates:
                break

            weights = [U.get(e, 0.0) for e in candidates]
            if sum(weights) == 0:
                next_edge = random.choice(candidates)
            else:
                next_edge = random.choices(candidates, weights=weights, k=1)[0]

            u1, v1 = next_edge
            if u1 == start:
                route.insert(0, v1)
            elif v1 == start:
                route.insert(0, u1)
            elif u1 == end:
                route.append(v1)
            elif v1 == end:
                route.append(u1)
            else:
                break
        return route
    
    routes = Parallel(n_jobs=cfg.get("n_jobs", 4), backend="threading")(
        delayed(_build_one)(i) for i in range(num_routes)
    )
    
    return routes
