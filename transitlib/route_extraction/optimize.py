import random
from typing import List, Tuple, Set, Dict
import numpy as np
from transitlib.config import Config

cfg = Config()

MIN_STOPS = cfg.get("min_stops")
MAX_STOPS = cfg.get("max_stops")
WEIGHTS = cfg.get("weights")

def score_usage(route, Q, F):
    edges = list(zip(route[:-1], route[1:]))
    q_vals = [Q.get(e, Q.get((e[1], e[0]), 0.0)) for e in edges]
    f_vals = [F.get(e, F.get((e[1], e[0]), 0.0)) for e in edges]
    return 0.5 * (np.mean(q_vals) + np.mean(f_vals)) if edges else 0.0

def score_feasibility(route):
    L = len(route)
    return 1.0 / (MAX_STOPS - MIN_STOPS + 1) if MIN_STOPS <= L <= MAX_STOPS else 0.0

def score_directness(route, mst_edges):
    edges = list(zip(route[:-1], route[1:]))
    hits = sum(1 for e in edges if e in mst_edges or (e[1], e[0]) in mst_edges)
    return hits / len(edges) if edges else 0.0

def score_coverage(solution, total_nodes):
    covered = {n for route in solution for n in route}
    return len(covered) / total_nodes if total_nodes else 0.0

def route_score(route, solution, Q, F, mst_edges, total_nodes):
    return (
        WEIGHTS["usage"] * score_usage(route, Q, F) +
        WEIGHTS["feasibility"] * score_feasibility(route) +
        WEIGHTS["coverage"] * score_coverage(solution, total_nodes) +
        WEIGHTS["directness"] * score_directness(route, mst_edges)
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
    G_stop: nx.Graph,
    initial_routes: List[List[int]],
    Q: Dict[Tuple[int,int], float],
    F: Dict[Tuple[int,int], float],
    mst_edges: Set[Tuple[int,int]],
    num_nodes: int
) -> List[List[int]]:
    """
    Hill‑climb each route by randomly applying insert/delete/swap operators
    and accepting improvements in route_score, for up to max_iters.
    """
    max_iters = cfg.get("opt_max_iters")
    
    solution = initial_routes.copy()
    for iter in range(max_iters):
        if iter % 1000 == 0:
            print(f"Completed {iter} iterations")
        
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
