import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from datetime import datetime, timedelta
from shapely.geometry import Point
from collections import Counter
from typing import Optional, List, Tuple, Dict

"""
§ 3 Passive Datasets — Unified simulation of pings and trips
"""

def simulate_users(
    G_latlon: nx.Graph,
    n_users: int = 1,
    avg_pings: float = 4.5,
    transit_frac: float = 0.30,
    sigma_deg: float = 0.0005,
    days: Optional[List[datetime]] = None
) -> Tuple[gpd.GeoDataFrame, Counter]:
    """
    Simulate mobile‐phone pings and record true link‐traversals in one pass.

    Args:
        G_latlon: OSM graph (EPSG:4326) with node coords and 'length' attribute.
        n_users: Number of users to simulate.
        avg_pings: Mean daily pings per user (Poisson).
        transit_frac: Fraction of pings labeled 'transit' (trip).
        sigma_deg: Gaussian noise std‑dev on ping coords (degrees).
        days: List of datetime objects for simulation days; defaults to Jul & Nov 2019.

    Returns:
        pings_gdf: GeoDataFrame of all simulated pings with columns
            ['user_id','timestamp','ping_type','geometry'] (EPSG:4326).
        od_counts: Counter mapping each traversed edge (u,v) to traversal count D_uv.
    """
    # 1) Build hourly PMF
    bucket_props = {"morning":0.20,"offpeak":0.35,"evening":0.25,"night":0.20}
    mapping = {
        **{h: bucket_props["morning"]/3 for h in range(7,10)},
        **{h: bucket_props["offpeak"]/6 for h in range(10,16)},
        **{h: bucket_props["evening"]/3 for h in range(16,19)},
        **{h: bucket_props["night"]/4 for h in range(19,23)},
        **{h: bucket_props["offpeak"]/16 for h in list(range(0,7))+[23]}
    }
    hourly_pmf = np.array([mapping[h] for h in range(24)])
    hourly_pmf /= hourly_pmf.sum()

    # 2) Default date ranges
    if days is None:
        days = [datetime(2019,7,d) for d in range(1,32)] + \
               [datetime(2019,11,d) for d in range(1,31)]

    ping_records = []
    od_counts: Counter = Counter()
    nodes = list(G_latlon.nodes())

    for uid in range(n_users):
        home = random.choice(nodes)
        for day in days:
            # number of pings this user on this day
            num = np.random.poisson(avg_pings)
            for _ in range(num):
                # sample time
                hour = np.random.choice(24, p=hourly_pmf)
                minute = random.uniform(0,60)
                ts = day + timedelta(hours=hour, minutes=minute)

                if random.random() < transit_frac:
                    # pick a random trip
                    o, d = random.sample(nodes, 2)
                    try:
                        path = nx.shortest_path(G_latlon, o, d, weight="length")
                        # record transit waypoint ping on a random segment node
                        node = random.choice(path)
                        p_type = "transit"
                        # record every traversed edge for D_uv
                        for u, v in zip(path, path[1:]):
                            od_counts[(u, v)] += 1
                    except nx.NetworkXNoPath:
                        # treat as home ping if no path
                        node, p_type = home, "home"
                else:
                    # home ping
                    node, p_type = home, "home"

                # jitter the location
                lat = G_latlon.nodes[node]["y"] + np.random.normal(0, sigma_deg)
                lon = G_latlon.nodes[node]["x"] + np.random.normal(0, sigma_deg)

                ping_records.append({
                    "user_id":   uid,
                    "timestamp": ts,
                    "ping_type": p_type,
                    "geometry":  Point(lon, lat)
                })

    # Build pings GeoDataFrame
    pings_df = pd.DataFrame(ping_records)
    pings_gdf = gpd.GeoDataFrame(pings_df, geometry="geometry", crs="EPSG:4326")

    return pings_gdf, od_counts
