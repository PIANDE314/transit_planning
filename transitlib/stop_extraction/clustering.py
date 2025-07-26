import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from transitlib.config import Config

cfg = Config()

def extract_candidate_stops(
    segments_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame,
    pois_gdf: gpd.GeoDataFrame,
    final_label: str = 'final_viable'
) -> gpd.GeoDataFrame:
    """
    Cluster pings within viable road buffers; extract centroid stops with footfall data.
    """
    buf = cfg.get("ping_buffer")
    eps = cfg.get("db_eps")
    min_samples = cfg.get("db_min_samples")
    top_frac = cfg.get("top_frac_stops")

    # 1. Select pings within viable road segments
    viable = segments_gdf[segments_gdf[final_label] == 1]
    union_area = viable.geometry.buffer(buf).unary_union

    pings_gdf = pings_gdf.to_crs(segments_gdf.crs)
    pings_sel = pings_gdf[pings_gdf.geometry.within(union_area)].copy()

    # 2. DBSCAN clustering
    coords = np.vstack([pings_sel.geometry.x, pings_sel.geometry.y]).T
    db = DBSCAN(eps=eps, min_samples=min_samples)
    pings_sel['cluster'] = db.fit_predict(coords)

    # 3. Compute weekday footfall per cluster
    pings_sel['date'] = pings_sel['timestamp'].dt.date
    pings_sel['weekday'] = pings_sel['timestamp'].dt.weekday < 5
    wkday = pings_sel[pings_sel['weekday']]
    uday = wkday.groupby(['cluster', 'date'])['user_id'].nunique()
    avg_footfall = uday.groupby('cluster').mean()

    # 4. Top clusters
    n_top = max(int(len(avg_footfall) * top_frac), 1)
    top_clusters = avg_footfall.nlargest(n_top).index

    # 5. Centroids for top clusters
    selected = pings_sel[pings_sel['cluster'].isin(top_clusters)]
    centroids = (
        selected.groupby('cluster').geometry
        .apply(lambda geoms: Point(np.mean([pt.x for pt in geoms]), np.mean([pt.y for pt in geoms])))
    )

    # 6. Include footfall values
    footfall_vals = avg_footfall.loc[centroids.index]

    extracted = gpd.GeoDataFrame({
        'geometry': centroids.values,
        'type': ['extracted'] * len(centroids),
        'footfall': footfall_vals.values
    }, geometry='geometry', crs=segments_gdf.crs)

    # 7. Prepare POI stops (with zero footfall)
    osm = pois_gdf.copy()
    osm['type'] = 'osm'
    osm['footfall'] = 0.0
    osm = osm[['geometry', 'type', 'footfall']]

    # 8. Combine
    all_stops = pd.concat([extracted, osm], ignore_index=True).drop_duplicates(subset='geometry')

    return all_stops
