import random
from typing import List, Tuple, Dict
import networkx as nx

"""
§ 3.3 Transit Routes Extraction — sample initial routes (Fig 4A)
"""

# defaults per paper
MIN_STOPS = 4
MAX_STOPS = 8

def sample_route_length(
    min_stops: int = MIN_STOPS,
    max_stops: int = MAX_STOPS
) -> int:
    """
    Randomly choose a route length (number of stops) between min_stops and max_stops.

    Returns:
        int: route length.
    """
    return random.randint(min_stops, max_stops)


def generate_initial_routes(
    U: Dict[Tuple[int,int], float],
    num_routes: int = 64,
    min_stops: int = MIN_STOPS,
    max_stops: int = MAX_STOPS
) -> List[List[int]]:
    """
    Generate a list of initial routes by random walks biased by utility U_uv.

    Args:
        U: dict of edge utilities U_uv.
        num_routes: number of routes to sample.
        min_stops: minimum stops per route.
        max_stops: maximum stops per route.

    Returns:
        List of routes, each a list of node IDs.
    """
    edges = list(U.keys())
    weights = [U[e] for e in edges]
    routes: List[List[int]] = []

    for _ in range(num_routes):
        # 1) pick seed edge
        u0, v0 = random.choices(edges, weights=weights, k=1)[0]
        route = [u0, v0]
        target_len = sample_route_length(min_stops, max_stops)

        # 2) grow until reaching target length or dead end
        while len(route) < target_len:
            start, end = route[0], route[-1]
            candidates: List[Tuple[int,int]] = []

            # collect neighboring edges not revisiting nodes
            for node in (start, end):
                for nbr in nx.neighbors(nx.Graph(U), node):
                    if nbr not in route:
                        edge = (node, nbr) if (node, nbr) in U else (nbr, node)
                        candidates.append(edge)

            if not candidates:
                break

            # sample next edge by utility weight
            next_edge = random.choices(candidates, 
                                       weights=[U.get(e, 0.0) for e in candidates], 
                                       k=1)[0]
            u1, v1 = next_edge
            # extend at whichever end matches
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
