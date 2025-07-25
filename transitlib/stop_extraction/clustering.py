import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from transit_planner.config import Config

cfg = Config()

def extract_candidate_stops(
    segments_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame,
    pois_gdf: gpd.GeoDataFrame,
    final_label: str = 'final_viable'
) -> gpd.GeoDataFrame:
    """
    Cluster pings within viable road buffers; extract centroid stops.
    """
    buf = cfg.get("buffer_poi")
    eps = cfg.get("db_eps")
    min_samples = cfg.get("db_min_samples")
    top_frac = cfg.get("top_frac_stops")

    viable = segments_gdf[segments_gdf[final_label] == 1]
    union_area = viable.geometry.buffer(buf).unary_union

    pings_sel = pings_gdf[pings_gdf.geometry.within(union_area)].copy()

    coords = np.vstack([pings_sel.geometry.x, pings_sel.geometry.y]).T
    db = DBSCAN(eps=eps, min_samples=min_samples)
    pings_sel['cluster'] = db.fit_predict(coords)

    pings_sel['date'] = pings_sel['timestamp'].dt.date
    pings_sel['weekday'] = pings_sel['timestamp'].dt.weekday < 5
    wkday = pings_sel[pings_sel['weekday']]
    uday = wkday.groupby(['cluster', 'date'])['user_id'].nunique()
    avg_footfall = uday.groupby('cluster').mean()

    n_top = max(int(len(avg_footfall) * top_frac), 1)
    top_clusters = avg_footfall.nlargest(n_top).index

    centroids = (
        pings_sel[pings_sel['cluster'].isin(top_clusters)]
        .groupby('cluster').geometry.apply(lambda geoms: Point(
            np.mean([pt.x for pt in geoms]), np.mean([pt.y for pt in geoms])
        ))
    )
    extracted = gpd.GeoDataFrame({'type': 'extracted'}, geometry=centroids, crs=segments_gdf.crs)

    osm = pois_gdf.copy().rename_geometry('geometry')
    osm['type'] = 'osm'
    osm = osm[['geometry', 'type']]

    all_stops = pd.concat([extracted, osm], ignore_index=True).drop_duplicates(subset='geometry')
    return all_stops
