import numpy as np
import pandas as pd
import geopandas as gpd
import hdbscan
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from transitlib.config import Config

cfg = Config()

def extract_candidate_stops(
    segments_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame,
    pois_gdf: gpd.GeoDataFrame,
    final_label: str = 'final_viable'
    use_HDBSCAN: bool = False
) -> gpd.GeoDataFrame:
    """
    Cluster pings within viable road buffers; extract centroid stops with footfall data.
    Optimizations:
      1) Pre-filter pings by bounding box before .within().
      2) Vectorized centroid computation via groupby.mean().
      3) Skip DBSCAN & grouping if no pings selected.
    """
    # configuration
    buf         = cfg.get("ping_buffer")
    eps         = cfg.get("db_eps")
    min_samples = cfg.get("db_min_samples")
    top_frac    = cfg.get("top_frac_stops")

    # 1) Build union of viable buffers
    viable     = segments_gdf[segments_gdf[final_label] == 1]
    buffers    = viable.geometry.buffer(buf)
    union_area = buffers.unary_union
    if union_area.is_empty:
        # no viable area → return only POI stops
        return pois_gdf.rename(columns={'geometry':'geometry'}).assign(type='osm', footfall=0.0)[['geometry','type','footfall']]

    # 2) Fast bbox pre-filter on pings (project once)
    pings_proj = pings_gdf.to_crs(segments_gdf.crs)
    minx, miny, maxx, maxy = union_area.bounds
    pings_roi = pings_proj.cx[minx:maxx, miny:maxy]

    # exact within()
    pings_sel = pings_roi[pings_roi.geometry.within(union_area)].copy()
    if pings_sel.empty:
        # fallback if no pings in viable area
        extracted = gpd.GeoDataFrame([], columns=['geometry','type','footfall'], geometry='geometry', crs=segments_gdf.crs)
        osm = pois_gdf.copy()
        osm['type'] = 'osm'
        osm['footfall'] = 0.0
        osm = osm[['geometry','type','footfall']]
        return pd.concat([extracted, osm], ignore_index=True)

    # 3) DBSCAN clustering
    coords = np.vstack((pings_sel.geometry.x.values, pings_sel.geometry.y.values)).T

    if use_HDBSCAN:
        min_cluster_size = cfg.get("hdb_min_cluster_size", min_samples)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
        )
        clusters = clusterer.fit_predict(coords)
    else:  
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        
    pings_sel['cluster'] = clusters

    # 4) Compute average weekday footfall per cluster
    pings_sel['date']    = pings_sel['timestamp'].dt.date
    pings_sel['weekday'] = pings_sel['timestamp'].dt.weekday < 5
    wkday = pings_sel[pings_sel['weekday']]
    # unique users per (cluster, date)
    uday = wkday.groupby(['cluster','date'])['user_id'].nunique()
    avg_footfall = uday.groupby('cluster').mean()

    # 5) Pick top clusters
    n_top = max(int(len(avg_footfall)*top_frac), 1)
    top_clusters = avg_footfall.nlargest(n_top).index

    # 6) Vectorized centroid calculation
    sel = pings_sel[pings_sel['cluster'].isin(top_clusters)].copy()
    sel['x'] = sel.geometry.x
    sel['y'] = sel.geometry.y
    cent_df = sel.groupby('cluster')[['x','y']].mean()
    centroids = gpd.GeoSeries(
        [Point(xy) for xy in cent_df.values],
        index=cent_df.index,
        crs=segments_gdf.crs
    )

    # 7) Build extracted stops with footfall
    extracted = gpd.GeoDataFrame({
        'geometry': centroids.values,
        'type': ['extracted'] * len(centroids),
        'footfall': avg_footfall.loc[centroids.index].values
    }, geometry='geometry', crs=segments_gdf.crs)

    # 8) POI stops
    osm = pois_gdf.copy()
    osm['type'] = 'osm'
    osm['footfall'] = 0.0
    osm = osm[['geometry','type','footfall']]

    # 9) Combine and dedupe by rounded coords
    all_stops = pd.concat([extracted, osm], ignore_index=True)
    all_stops['xr'] = all_stops.geometry.x.round(3)
    all_stops['yr'] = all_stops.geometry.y.round(3)
    idx = (
        all_stops
        .reset_index()   # preserve original row index
        .groupby(['xr','yr'])['footfall']
        .idxmax()
    )
    result = all_stops.loc[idx.values].drop(columns=['xr','yr']).reset_index(drop=True)

    return result
