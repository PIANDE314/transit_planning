import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import LineString
from sklearn.preprocessing import MinMaxScaler
from rasterstats import zonal_stats
from typing import Tuple
from transitlib.config import Config

cfg = Config()

# highway → ordinal
TYPE_MAP = {
    'tertiary': 0, 'secondary': 1,
    'primary_link': 2, 'primary': 3,
    'trunk_link': 4, 'trunk': 5
}

def extract_road_segments(
    G_proj: nx.Graph
) -> gpd.GeoDataFrame:
    buf = cfg.get("buffer_viability")
    rows = []
    for idx, (u, v, data) in enumerate(G_proj.edges(data=True)):
        hw = data.get("highway")
        hw_list = hw if isinstance(hw, list) else [hw]
        if not any(isinstance(h, str) and h in TYPE_MAP for h in hw_list):
            continue
        geom = LineString([
            (G_proj.nodes[u]['x'], G_proj.nodes[u]['y']),
            (G_proj.nodes[v]['x'], G_proj.nodes[v]['y'])
        ])
        rows.append({
            "segment_id": idx,
            "u": u, "v": v,
            "length_m": data.get("length", 0.0),
            "highway": hw,
            "geometry": geom
        })

    segs = gpd.GeoDataFrame(rows, crs="EPSG:3857")
    segs["buffer"]   = segs.geometry.buffer(buf)
    segs["area_km2"] = segs["buffer"].area / 1e6
    return segs


def compute_segment_features(
    segs: gpd.GeoDataFrame,
    worldpop_path: str,
    pois_gdf: gpd.GeoDataFrame,
    wealth_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Compute all Table 2 features using batched operations:
      • pop_density via zonal_stats
      • poi_density and all ping‐based densities via two SJOINs + groupby
      • wealth_mean via single SJOIN
      • road_density via STRtree + id→index map
      • type_ord
      • finally MinMax scaling
    """

    # 1) Population density (do one buffer at a time to keep memory low)
    pops = []
    with rasterio.open(worldpop_path) as src:
        for buf in segs["buffer"].to_crs(src.crs):
            # read only the window covering this buffer
            out_image, out_transform = mask(src, [buf], crop=True, all_touched=False)
            data = out_image[0]
            # treat nodata as zero
            data = np.where(data == src.nodata, 0, data)
            pops.append(data.sum())
    segs["pop_density"] = np.array(pops) / segs["area_km2"]

    # 2) poi_density
    #    one join, then group
    poi_joined = gpd.sjoin(
        pois_gdf, segs.set_geometry("buffer"),
        predicate="within", how="inner"
    )
    poi_counts = poi_joined.groupby("segment_id").size()
    segs["poi_density"] = poi_counts.reindex(index=segs.segment_id, fill_value=0) / segs["area_km2"]

    # 3) wealth_mean
    wj = gpd.sjoin(
        wealth_gdf, segs.set_geometry("buffer"),
        predicate="within", how="inner"
    )
    wm = wj.groupby("segment_id")["rwi"].mean()
    segs["wealth_mean"] = wm.reindex(index=segs.segment_id).fillna(wm.median())

    # 4) Pull all pings inside buffers in one go
    p = pings_gdf.copy()
    p["is_weekend"] = p.timestamp.dt.weekday >= 5
    # bucket time of day
    def _bucket(h):
        if 7 <= h < 10: return "MP"
        if 10 <= h < 16: return "OP"
        if 16 <= h < 19: return "EP"
        if 19 <= h < 23: return "Night"
        return "OP"
    p["bucket"] = p.timestamp.dt.hour.map(_bucket)
    p.loc[p.is_weekend, "bucket"] = "Weekend"

    pj = gpd.sjoin(
        p, segs.set_geometry("buffer"),
        predicate="within", how="inner"
    )

    # 5) From pj compute:
    #    - MP/OP/EP/Night/Weekend densities
    #    - conn_weekday / conn_weekend = unique users per segment
    #    - transit_wp and transit_wp_conn
    areas = segs.set_index("segment_id")["area_km2"]
    feat = {}
    # time‐bucket densities
    for b in ["MP","OP","EP","Night","Weekend"]:
        c = pj[pj.bucket == b].groupby("segment_id")["user_id"].count()
        feat[f"{b}_dens"] = c.reindex(index=segs.segment_id, fill_value=0) / areas

    # connected: unique users per segment per weekday/weekend
    for label, col in [(False,"conn_weekday_dens"), (True,"conn_weekend_dens")]:
        sub = pj[pj.is_weekend == label]
        u = sub.groupby("segment_id")["user_id"].nunique()
        feat[col] = u.reindex(index=segs.segment_id, fill_value=0) / areas

    # transit waypoints
    trans = pj[pj.ping_type == "transit"]
    tcnt = trans.groupby("segment_id")["user_id"].count()
    feat["transit_wp_dens"] = tcnt.reindex(index=segs.segment_id, fill_value=0) / areas

    # transit_wp_conn = count of transit pings in neighbors’ buffers
    # build neighbor map
    bufs = list(segs["buffer"])
    ids  = list(segs["segment_id"])
    tree = nx.strtree.STRtree(bufs)
    # precompute neighbor‐lists once
    nbr_map = {
        sid: [
            ids[j] for j, g in enumerate(bufs)
            if sid!=ids[j] and bufs[ids.index(sid)].intersects(g)
        ]
        for sid in ids
    }
    # now count for each sid
    twc = []
    for sid in ids:
        own_ids = nbr_map[sid]
        # number of transit pings in sid’s buffer that also lie in any neighbor’s buffer
        pts = list(trans.loc[trans.index].geometry.values)
        # but faster is to filter trans to this seg only, then test neighbor buffers
        seg_pts = trans[trans.segment_id == sid]
        cnt = 0
        for nb in own_ids:
            buf_nb = bufs[ids.index(nb)]
            cnt += seg_pts.geometry.apply(buf_nb.contains).sum()
        twc.append(cnt / areas[sid])
    feat["transit_wp_conn_dens"] = pd.Series(twc, index=segs.segment_id)

    # 6) Road density via STRtree + id→index
    geoms = segs.geometry.values
    lens  = segs.length_m.values
    tree2 = nx.strtree.STRtree(geoms)
    geom_idx = {id(g): i for i,g in enumerate(geoms)}
    rd = []
    for i, buf in enumerate(bufs):
        hits = tree2.query(buf)
        total = sum(
            lens[geom_idx[id(g)]]
            for g in hits
            if geom_idx[id(g)] != i and buf.intersects(g)
        )
        rd.append(total / areas.iloc[i])
    feat["road_density"] = pd.Series(rd, index=segs.segment_id)

    # 7) Road type ordinal
    segs["type_ord"] = segs.highway.map(
        lambda hw: max((TYPE_MAP.get(h,-1) for h in (hw if isinstance(hw,list) else [hw])))
    ).astype(int)

    # assemble X and scale
    feat_df = pd.DataFrame(feat)
    X = feat_df.fillna(0.0)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # attach raw segs (with pop/poi/wealth) and return
    segs = pd.concat([segs.set_index("segment_id"), feat_df], axis=1).reset_index()
    return segs, X_scaled, scaler
