import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from datetime import datetime, timedelta
from shapely.geometry import Point
from tqdm import tqdm
from typing import Optional

"""
§ 3 Passive Datasets — Simulate mobile‑phone pings as a proxy for movement data
(using original Poisson & time‑bucket logic) — NOT IN PAPER (exception)
"""

def simulate_pings(
    G_latlon: nx.Graph,
    n_users: int = 1,
    avg_pings: float = 4.5,
    transit_frac: float = 0.30,
    sigma_deg: float = 0.0005,
    days: Optional[list] = None
) -> gpd.GeoDataFrame:
    """
    Simulate user pings over two seasons using Poisson arrivals and a time‑of‑day PMF.

    Args:
        G_latlon: OSM graph in EPSG:4326 with node coords.
        n_users: Number of users.
        avg_pings: Mean daily pings per user.
        transit_frac: Fraction labeled 'transit'.
        sigma_deg: Gaussian noise (degrees).
        days: Optional list of datetime.date for simulation.

    Returns:
        GeoDataFrame with columns:
          user_id, timestamp, ping_type ('home'/'transit'), geometry (Point, EPSG:4326)
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

    # 2) Default dates (July & November 2019)
    if days is None:
        days = [datetime(2019,7,d) for d in range(1,32)] + \
               [datetime(2019,11,d) for d in range(1,31)]

    records = []
    nodes = list(G_latlon.nodes())

    for uid in tqdm(range(n_users), desc="Simulating users"):
        home = random.choice(nodes)
        for day in days:
            for _ in range(np.random.poisson(avg_pings)):
                hour = np.random.choice(24, p=hourly_pmf)
                minute = random.uniform(0,60)
                ts = day + timedelta(hours=hour, minutes=minute)

                if random.random() < transit_frac:
                    o, d = random.sample(nodes, 2)
                    try:
                        path = nx.shortest_path(G_latlon, o, d, weight="length")
                        node = random.choice(path)
                        p_type = "transit"
                    except nx.NetworkXNoPath:
                        node, p_type = home, "home"
                else:
                    node, p_type = home, "home"

                lat = G_latlon.nodes[node]["y"] + np.random.normal(0, sigma_deg)
                lon = G_latlon.nodes[node]["x"] + np.random.normal(0, sigma_deg)

                records.append({
                    "user_id":   uid,
                    "timestamp": ts,
                    "ping_type": p_type,
                    "geometry":  Point(lon, lat)
                })

    df = pd.DataFrame(records)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
