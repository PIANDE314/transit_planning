import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from datetime import datetime, timedelta
from shapely.geometry import Point
from collections import Counter
from typing import Optional, List, Tuple, Dict
from transitlib.config import Config
from joblib import Parallel, delayed

cfg = Config()

def simulate_users(
    G_latlon: nx.Graph,
    days: Optional[List[datetime]] = None,
    use_path_cache: bool = True,
    noise: bool = False
) -> Tuple[gpd.GeoDataFrame, Counter]:
    """
    Simulate pings & record edge traversals in one pass.
    
    If noise=False (developed-region):
      • daily ping count ~ Poisson(avg_pings)
      • ping times sampled by hourly_distribution
      • spatial jitter σ = sigma_deg
    
    If noise=True (developing-region):
      • daily ping count ~ Poisson(avg_pings)
      • ping times scheduled at non-uniform multi-hour intervals:
           intervals ∈ [60,120,240,480] min with weights [0.4,0.3,0.2,0.1]
      • spatial jitter σ = 3 * sigma_deg
    """
    # unpack config
    n_users      = cfg.get("n_users")
    avg_pings    = cfg.get("avg_pings")
    transit_frac = cfg.get("transit_frac")
    sigma_deg    = cfg.get("sigma_deg")
    hourly_pmf   = np.array(cfg.get("hourly_distribution", [1/24]*24), float)
    hourly_pmf  /= hourly_pmf.sum()

    # build list of days if not provided
    if days is None:
        periods = cfg.get("simulation_periods")
        days = []
        for p in periods:
            start = datetime.strptime(p["start"], "%Y-%m-%d")
            end   = datetime.strptime(p["end"],   "%Y-%m-%d")
            cur = start
            while cur <= end:
                days.append(cur)
                cur += timedelta(days=1)

    nodes = list(G_latlon.nodes())

    # optional precompute paths
    if use_path_cache:
        PATH_CACHE: Dict[int, Dict[int, List[int]]] = {
            o: nx.single_source_dijkstra_path(G_latlon, o, weight="length")
            for o in nodes
        }

    def _simulate_one(uid: int):
        od_local = Counter()
        pings_rec: List[Dict] = []

        # helper for each ping timestamp
        def _handle_single_ping(ts):
            if random.random() < transit_frac:
                o, d = random.sample(nodes, 2)
                # find path
                if use_path_cache:
                    path = PATH_CACHE[o].get(d)
                else:
                    try:
                        path = nx.shortest_path(G_latlon, o, d, weight="length")
                    except nx.NetworkXNoPath:
                        path = None
                if path:
                    # record one random stop on path and count all edges
                    node = random.choice(path)
                    for u, v in zip(path, path[1:]):
                        od_local[(u, v)] += 1
                else:
                    node = o
                ptype = "transit"
            else:
                node = random.choice(nodes)
                ptype = "home"

            # spatial jitter
            s = sigma_deg * (3.0 if noise else 1.0)
            lat = G_latlon.nodes[node]["y"] + np.random.normal(0, s)
            lon = G_latlon.nodes[node]["x"] + np.random.normal(0, s)

            pings_rec.append({
                "user_id": uid,
                "timestamp": ts,
                "ping_type": ptype,
                "geometry": Point(lon, lat)
            })

        # daily simulation
        p_counts = np.random.poisson(avg_pings, size=len(days))
        for day, n_pings in zip(days, p_counts):
            if n_pings <= 0:
                continue

            if noise:
                # non-uniform multi-hour intervals
                interval_choices = [60, 120, 240, 480]
                interval_weights = [0.4, 0.3, 0.2, 0.1]
                # sample n_pings intervals
                ints = random.choices(interval_choices, interval_weights, k=n_pings)
                t = day
                for d in ints:
                    t += timedelta(minutes=d)
                    if t >= day + timedelta(days=1):
                        # wrap around within the same day window
                        t = day + (t - (day + timedelta(days=1)))
                    # jitter within ±d/2 minutes
                    ts = t + timedelta(minutes=random.uniform(-d/2, d/2))
                    _handle_single_ping(ts)
            else:
                # Poisson + hourly distribution
                hours = np.random.choice(24, size=n_pings, p=hourly_pmf)
                for hour in hours:
                    ts = day + timedelta(hours=int(hour), minutes=random.uniform(0, 60))
                    _handle_single_ping(ts)

        return pings_rec, od_local

    # parallel simulation
    results = Parallel(n_jobs=cfg.get("n_jobs", 4))(
        delayed(_simulate_one)(uid) for uid in range(n_users)
    )

    # merge all results
    all_pings: List[Dict] = []
    od_counts = Counter()
    for pings_rec, od_local in results:
        all_pings.extend(pings_rec)
        od_counts.update(od_local)

    pings_gdf = gpd.GeoDataFrame(
        pd.DataFrame(all_pings),
        geometry="geometry",
        crs="EPSG:4326"
    )
    return pings_gdf, od_counts
