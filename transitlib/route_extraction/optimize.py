import random
from typing import List, Tuple, Set, Dict
import numpy as np
import networkx as nx
import math
from scipy.stats import gamma, norm, uniform
from transitlib.config import Config

cfg = Config()

MIN_STOPS = cfg.get("min_stops")
MAX_STOPS = cfg.get("max_stops")
WEIGHTS = cfg.get("weights")

_pmf_cache = {}

def _get_length_pmf(node_dist: str) -> np.ndarray:
    """
    Returns a 1D array pmf[k-2] = P(length=k) for k = 2..MAX_STOPS,
    truncated & normalized so sum = 1.
    """
    key = node_dist
    if key in _pmf_cache:
        return _pmf_cache[key]

    if node_dist == "gam":
        dist = gamma(a=3.5, scale=6.0)
    elif node_dist == "norm":
        dist = norm(loc=21.0, scale=11.0)
    elif node_dist == "uni":
        dist = uniform(loc=2.0, scale=38.0)
    else:
        raise ValueError(f"Unknown node_dist: {node_dist!r}")

    ks = np.arange(2, 1000)
    cdf_high = dist.cdf(ks + 0.5)
    cdf_low  = dist.cdf(ks - 0.5)
    pmf = cdf_high - cdf_low
    pmf = pmf / (1.0 - dist.cdf(1.5))

    _pmf_cache[key] = pmf
    return pmf

def score_usage(route, Q, F):
    edges = list(zip(route[:-1], route[1:]))
    q_vals = [Q.get(e, Q.get((e[1], e[0]), 0.0)) for e in edges]
    f_vals = [F.get(e, F.get((e[1], e[0]), 0.0)) for e in edges]
    return 0.5 * (np.mean(q_vals) + np.mean(f_vals)) if edges else 0.0

def score_feasibility(route, node_dist):
    L = len(route)
    if L < 2 or L > 100:
        return 0.0

    pmf = _get_length_pmf(node_dist)
    return float(pmf[L - 2])

def score_directness(route, mst_edges):
    edges = list(zip(route[:-1], route[1:]))
    hits = sum(1 for e in edges if e in mst_edges or (e[1], e[0]) in mst_edges)
    return hits / len(edges) if edges else 0.0

def score_coverage(solution, total_nodes):
    covered = {n for route in solution for n in route}
    return len(covered) / total_nodes if total_nodes else 0.0

def route_score(route, solution, Q, F, mst_edges, total_nodes, scoring_method, node_dist):
    if scoring_method == "sqrt":
        return (
            WEIGHTS["usage"] * math.sqrt(score_usage(route, Q, F)) +
            WEIGHTS["feasibility"] * math.sqrt(score_feasibility(route, node_dist)) +
            WEIGHTS["coverage"] * math.sqrt(score_coverage(solution, total_nodes)) +
            WEIGHTS["directness"] * math.sqrt(score_directness(route, mst_edges))
        )
    else:
        return (
            WEIGHTS["usage"] * score_usage(route, Q, F) +
            WEIGHTS["feasibility"] * score_feasibility(route, node_dist) +
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
    num_nodes: int,
    node_dist,
    scoring_method,
    search_algorithm
) -> List[List[int]]:
    """
    Hill‑climb each route by randomly applying insert/delete/swap operators
    and accepting improvements in route_score, for up to max_iters.
    """
    max_iters = cfg.get("opt_max_iters")
    T = 0.1
    a  = 0.9999
    use_SA = (search_algorithm == "SA")
    _route_score = route_score
    _Q = Q; _F = F; _mst = set(mst_edges); _G = G_stop
    nbrs_map = {u: set(_G.neighbors(u)) for u in _G.nodes()}

    solution = initial_routes.copy()
    score_trace: List[float] = []
    
    for iter in range(max_iters):
        if iter % 1000 == 0:
            score_trace.append(sum(
                _route_score(r, solution, _Q, _F, _mst, num_nodes, scoring_method, node_dist)
                for r in solution
            ) / len(solution))
            print(f"Completed {iter} iterations")
        
        i = random.randrange(len(solution))
        route = solution[i]
        base = _route_score(route, solution, _Q, _F, _mst, num_nodes, scoring_method, node_dist)

        # pick operator
        op = random.choice(["insert", "delete", "swap"])
        if op == "insert":
            # find all valid insertions
            candidates = []
            used = set(route)
            for idx in range(len(route)-1):
                u, v = route[idx], route[idx+1]
                common = nbrs_map[u] & nbrs_map[v]
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
                common = nbrs_map[u] & nbrs_map[v]
                for w in common:
                    if w not in used:
                        candidates.append((idx, w))
            if not candidates:
                continue
            idx, node = random.choice(candidates)
            new_route = swap_node(route, idx, node)

        # evaluate
        score_new = _route_score(new_route, solution, _Q, _F, _mst, num_nodes, scoring_method, node_dist)
        delta = score_new - base
        if delta > 0 or (use_SA and random.random() < math.exp(delta / T)):
            solution[i] = new_route

        if use_SA:
            T *= a

    score_trace.append(sum(
        _route_score(r, solution, _Q, _F, _mst, num_nodes, scoring_method, node_dist)
        for r in solution
    ) / len(solution))

    return solution, score_trace
