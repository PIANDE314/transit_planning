import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from typing import Tuple
from datetime import date

"""
§ 3.2 Stop Location Extraction — cluster pings to identify high‑footfall stops :contentReference[oaicite:8]{index=8}
"""

def extract_candidate_stops(
    segments_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame,
    pois_gdf: gpd.GeoDataFrame,
    final_label: str = 'final_viable',
    ping_buffer: float = 100.0,
    db_eps: float = 200.0,
    db_min_samples: int = 1,
    top_frac: float = 0.10
) -> gpd.GeoDataFrame:
    """
    1. Buffer all segments labeled `final_label` by `ping_buffer` (100 m).
    2. Select pings within that union.
    3. Cluster with DBSCAN(eps=db_eps, min_samples=db_min_samples).
    4. For each cluster, compute average daily unique users on weekdays.
    5. Select top `top_frac` clusters by footfall.
    6. Compute centroids of those clusters.
    7. Append existing OSM POI stops.
    8. Dedupe geometries.

    Returns:
        GeoDataFrame with columns:
          - geometry: Point locations
          - type: 'extracted' or 'osm'
    """
    # 1) Filter viable segments and build buffer union
    viable = segments_gdf[segments_gdf[final_label] == 1]
    union_area = viable.geometry.buffer(ping_buffer).unary_union

    # 2) Select pings near viable roads
    pings_sel = pings_gdf[pings_gdf.geometry.within(union_area)].copy()

    # 3) DBSCAN clustering
    coords = np.vstack([pings_sel.geometry.x, pings_sel.geometry.y]).T
    db = DBSCAN(eps=db_eps, min_samples=db_min_samples)
    pings_sel['cluster'] = db.fit_predict(coords)

    # 4) Compute average daily unique users on weekdays
    pings_sel['date'] = pings_sel['timestamp'].dt.date
    pings_sel['weekday'] = pings_sel['timestamp'].dt.weekday < 5
    wkday = pings_sel[pings_sel['weekday']]
    # unique users per (cluster, date)
    uday = wkday.groupby(['cluster', 'date'])['user_id'].nunique()
    avg_footfall = uday.groupby('cluster').mean()

    # 5) Select top clusters
    n_top = max(int(len(avg_footfall) * top_frac), 1)
    top_clusters = avg_footfall.nlargest(n_top).index

    # 6) Centroid of each top cluster
    centroids = (
        pings_sel[pings_sel['cluster'].isin(top_clusters)]
        .groupby('cluster').geometry.apply(lambda geoms: Point(
            np.mean([pt.x for pt in geoms]),
            np.mean([pt.y for pt in geoms])
        ))
    )
    extracted = gpd.GeoDataFrame(
        {'type': 'extracted'},
        geometry=centroids,
        crs=segments_gdf.crs
    )

    # 7) Existing OSM POI stops
    osm = pois_gdf.copy().rename_geometry('geometry')
    osm['type'] = 'osm'
    osm = osm[['geometry', 'type']]

    # 8) Combine and dedupe
    all_stops = pd.concat([extracted, osm], ignore_index=True)
    all_stops = all_stops.drop_duplicates(subset='geometry')

    return all_stops
