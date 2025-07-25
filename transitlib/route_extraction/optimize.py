import random
from typing import List, Tuple, Set, Dict
import networkx as nx
import numpy as np

"""
§ 3.3 Transit Routes Extraction — optimize routes via hill‑climbing (Fig 4B)
"""

# route‐length bounds
MIN_STOPS = 4
MAX_STOPS = 8

# weightings per paper
WEIGHTS: Dict[str, float] = {
    "usage":       0.4,
    "feasibility": 0.2,
    "coverage":    0.2,
    "directness":  0.2
}

def score_usage(
    route: List[int],
    Q: Dict[Tuple[int,int], float],
    F: Dict[Tuple[int,int], float]
) -> float:
    """
    Usage = 0.5 * (mean Q + mean F) over edges in route.
    """
    edges = list(zip(route[:-1], route[1:]))
    q_vals = [Q.get(e, Q.get((e[1], e[0]), 0.0)) for e in edges]
    f_vals = [F.get(e, F.get((e[1], e[0]), 0.0)) for e in edges]
    return 0.5 * (np.mean(q_vals) + np.mean(f_vals)) if edges else 0.0

def score_feasibility(
    route: List[int]
) -> float:
    """
    Feasibility = 1/(MAX_STOPS-MIN_STOPS+1) if len(route) within bounds, else 0.
    """
    L = len(route)
    return (1.0 / (MAX_STOPS - MIN_STOPS + 1)) if MIN_STOPS <= L <= MAX_STOPS else 0.0

def score_directness(
    route: List[int],
    mst_edges: Set[Tuple[int,int]]
) -> float:
    """
    Directness = fraction of route‐edges that lie on the MST.
    """
    edges = list(zip(route[:-1], route[1:]))
    hits = sum(1 for e in edges if e in mst_edges or (e[1], e[0]) in mst_edges)
    return (hits / len(edges)) if edges else 0.0

def score_coverage(
    solution: List[List[int]],
    total_nodes: int
) -> float:
    """
    Coverage = unique stops covered / total stops.
    """
    covered = {n for route in solution for n in route}
    return len(covered) / total_nodes if total_nodes > 0 else 0.0

def route_score(
    route: List[int],
    solution: List[List[int]],
    Q: Dict[Tuple[int,int], float],
    F: Dict[Tuple[int,int], float],
    mst_edges: Set[Tuple[int,int]],
    total_nodes: int
) -> float:
    """
    Combined utility score of a single route within a solution set.
    """
    return (
        WEIGHTS["usage"]       * score_usage(route, Q, F)
      + WEIGHTS["feasibility"] * score_feasibility(route)
      + WEIGHTS["coverage"]    * score_coverage(solution, total_nodes)
      + WEIGHTS["directness"]  * score_directness(route, mst_edges)
    )

# Operators
def insert_node(
    route: List[int],
    position: int,
    node: int
) -> List[int]:
    return route[:position] + [node] + route[position:]

def delete_node(
    route: List[int],
    position: int
) -> List[int]:
    return route[:position] + route[position+1:]

def swap_node(
    route: List[int],
    position: int,
    node: int
) -> List[int]:
    new = route.copy()
    new[position] = node
    return new

def optimize_routes(
    initial_routes: List[List[int]],
    Q: Dict[Tuple[int,int], float],
    F: Dict[Tuple[int,int], float],
    mst_edges: Set[Tuple[int,int]],
    num_nodes: int,
    max_iters: int = 100_000
) -> List[List[int]]:
    """
    Hill‑climb each route by randomly applying insert/delete/swap operators
    and accepting improvements in route_score, for up to max_iters.

    Args:
        initial_routes: starting solution list.
        Q, F: utilities from compute_utilities.
        mst_edges: edges in MST for directness.
        num_nodes: total number of stops (for coverage).
        max_iters: maximum operator trials.

    Returns:
        Optimized list of routes.
    """
    solution = initial_routes.copy()
    for _ in range(max_iters):
        i = random.randrange(len(solution))
        route = solution[i]
        base = route_score(route, solution, Q, F, mst_edges, num_nodes)

        # pick operator
        op = random.choice(["insert", "delete", "swap"])
        if op == "insert":
            # find all valid insertions
            candidates = []
            used = set(route)
            for idx in range(len(route)-1):
                u, v = route[idx], route[idx+1]
                common = set(G.neighbors(u)).intersection(G.neighbors(v))
                for w in common:
                    if w not in used:
                        candidates.append((idx+1, w))
            if not candidates:
                continue
            pos, node = random.choice(candidates)
            new_route = insert_node(route, pos, node)

        elif op == "delete":
            if len(route) <= MIN_STOPS:
                continue
            idx = random.randrange(1, len(route)-1)
            new_route = delete_node(route, idx)

        else:  # swap
            candidates = []
            used = set(route)
            for idx in range(1, len(route)-1):
                u, v = route[idx-1], route[idx+1]
                common = set(G.neighbors(u)).intersection(G.neighbors(v))
                for w in common:
                    if w not in used:
                        candidates.append((idx, w))
            if not candidates:
                continue
            idx, node = random.choice(candidates)
            new_route = swap_node(route, idx, node)

        # evaluate
        score_new = route_score(new_route, solution, Q, F, mst_edges, num_nodes)
        if score_new > base:
            solution[i] = new_route

    return solution
