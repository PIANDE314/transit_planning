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
    days: Optional[List[datetime]] = None
) -> Tuple[gpd.GeoDataFrame, Counter]:
    """
    Simulate pings & record edge traversals in one pass, optimized:
      1) Precompute all‐pairs shortest‐paths from each origin once.
      2) Vectorize Poisson and hour sampling via NumPy.
      3) Parallelize per‑user simulation.
      5) Early bail‑out on unreachable OD pairs.
    """
    # unpack config
    n_users      = cfg.get("n_users")
    avg_pings    = cfg.get("avg_pings")
    transit_frac = cfg.get("transit_frac")
    sigma_deg    = cfg.get("sigma_deg")
    hourly_pmf   = np.array(cfg.get("hourly_distribution", [1/24]*24), float)
    hourly_pmf  /= hourly_pmf.sum()

    # build list of days
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

    # 1) PRECOMPUTE PATH CACHE for every node
    #    paths[o][d] = list of nodes from o→d, or absent if unreachable
    PATH_CACHE: Dict[int, Dict[int, List[int]]] = {
        o: nx.single_source_dijkstra_path(G_latlon, o, weight="length")
        for o in nodes
    }

    # helper for one user
    def _simulate_one(uid: int):
        od_local = Counter()
        pings_rec: List[Dict] = []

        # vectorized Poisson counts: one draw per day
        p_counts = np.random.poisson(avg_pings, size=len(days))
        for day, n_pings in zip(days, p_counts):
            if n_pings <= 0:
                continue
            # 2) vectorized hour picks
            hours = np.random.choice(24, size=n_pings, p=hourly_pmf)
            for hour in hours:
                ts = day + timedelta(hours=int(hour), minutes=random.uniform(0, 60))
                if random.random() < transit_frac:
                    o, d = random.sample(nodes, 2)
                    # 5) early bail‑out if no path cached
                    if d not in PATH_CACHE[o]:
                        node = o
                    else:
                        path = PATH_CACHE[o][d]
                        node = random.choice(path)
                        for u, v in zip(path, path[1:]):
                            od_local[(u, v)] += 1
                    ptype = "transit"
                else:
                    # home ping
                    o = random.choice(nodes)
                    node = o
                    ptype = "home"

                # add noise
                lat = G_latlon.nodes[node]["y"] + np.random.normal(0, sigma_deg)
                lon = G_latlon.nodes[node]["x"] + np.random.normal(0, sigma_deg)
                pings_rec.append({
                    "user_id": uid,
                    "timestamp": ts,
                    "ping_type": ptype,
                    "geometry": Point(lon, lat)
                })

        return pings_rec, od_local

    # 3) PARALLELIZE across users
    results = Parallel(n_jobs=cfg.get("n_jobs", 4))(
        delayed(_simulate_one)(uid) for uid in range(n_users)
    )

    # merge results
    all_pings: List[Dict] = []
    od_counts = Counter()
    for pings_rec, od_local in results:
        all_pings.extend(pings_rec)
        od_counts.update(od_local)

    # build GeoDataFrame
    pings_gdf = gpd.GeoDataFrame(
        pd.DataFrame(all_pings),
        geometry="geometry",
        crs="EPSG:4326"
    )
    return pings_gdf, od_counts
