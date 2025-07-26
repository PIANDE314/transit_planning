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

cfg = Config()

def simulate_users(
    G_latlon: nx.Graph,
    days: Optional[List[datetime]] = None
) -> Tuple[gpd.GeoDataFrame, Counter]:
    """
    Simulate pings & record edge traversals in one pass.
    """
    n_users     = cfg.get("n_users")
    avg_pings   = cfg.get("avg_pings")
    transit_frac= cfg.get("transit_frac")
    sigma_deg   = cfg.get("sigma_deg")
    # build hourly PMF
    bucket = cfg.get("hourly_bucket_props", {"morning": .2, "offpeak": .35, "evening": .25, "night": .2})
    # [build mapping & hourly_pmf from bucket]
    periods: List[Dict[str,str]] = cfg.get("simulation_periods")
    days = []
    for p in periods:
        start = datetime.strptime(p["start"], "%Y-%m-%d")
        end   = datetime.strptime(p["end"],   "%Y-%m-%d")
        cur = start
        while cur <= end:
            days.append(cur)
            cur += timedelta(days=1)

    hourly_pmf = np.array(cfg.get("hourly_distribution", [1/24] * 24)).astype(float)
    if len(hourly_pmf) != 24:
        raise ValueError("hourly_distribution must have exactly 24 values.")
    hourly_pmf /= hourly_pmf.sum()

    ping_records = []
    od_counts = Counter()
    nodes = list(G_latlon.nodes())

    for uid in range(n_users):
        home = random.choice(nodes)
        for day in days:
            for _ in range(np.random.poisson(avg_pings)):
                hour = np.random.choice(24, p=hourly_pmf)
                ts = day + timedelta(hours=hour, minutes=random.uniform(0,60))
                if random.random() < transit_frac:
                    o, d = random.sample(nodes, 2)
                    try:
                        path = nx.shortest_path(G_latlon, o, d, weight="length")
                        node = random.choice(path)
                        # record each traversed edge
                        for u, v in zip(path, path[1:]):
                            od_counts[(u, v)] += 1
                    except nx.NetworkXNoPath:
                        node = home
                else:
                    node = home

                lat = G_latlon.nodes[node]["y"] + np.random.normal(0, sigma_deg)
                lon = G_latlon.nodes[node]["x"] + np.random.normal(0, sigma_deg)
                ping_records.append({
                    "user_id": uid, "timestamp": ts,
                    "ping_type": "transit" if node in path else "home",
                    "geometry": Point(lon, lat)
                })

    pings_gdf = gpd.GeoDataFrame(
        pd.DataFrame(ping_records),
        geometry="geometry", crs="EPSG:4326"
    )
    return pings_gdf, od_counts
