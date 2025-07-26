import random
from typing import List, Tuple, Dict
import networkx as nx
from transitlib.config import Config

cfg = Config()

MIN_STOPS = cfg.get("min_stops")
MAX_STOPS = cfg.get("max_stops")

def sample_route_length(min_stops=MIN_STOPS, max_stops=MAX_STOPS) -> int:
    return random.randint(min_stops, max_stops)

def generate_initial_routes(
    U: Dict[Tuple[int,int], float],
    min_stops: int = MIN_STOPS,
    max_stops: int = MAX_STOPS
) -> List[List[int]]:
    num_routes = cfg.get("num_initial_routes")
    
    edges = list(U.keys())
    weights = [U[e] for e in edges]
    routes = []

    for _ in range(num_routes):
        u0, v0 = random.choices(edges, weights=weights, k=1)[0]
        route = [u0, v0]
        target_len = sample_route_length(min_stops, max_stops)

        while len(route) < target_len:
            start, end = route[0], route[-1]
            candidates = []
            for node in (start, end):
                for nbr in nx.neighbors(nx.Graph(U), node):
                    if nbr not in route:
                        edge = (node, nbr) if (node, nbr) in U else (nbr, node)
                        candidates.append(edge)
            if not candidates:
                break
            next_edge = random.choices(candidates, weights=[U.get(e, 0.0) for e in candidates])[0]
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
        routes.append(route)

    return routes
