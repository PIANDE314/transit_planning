import networkx as nx
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.strtree import STRtree
from sklearn.preprocessing import MinMaxScaler
from rasterio import features
from rasterio.mask import mask
from typing import Tuple
from transit_planner.config import Config

cfg = Config()

TYPE_MAP = {
    'tertiary': 0, 'secondary': 1,
    'primary_link': 2, 'primary': 3,
    'trunk_link': 4, 'trunk': 5
}

def extract_road_segments(G_proj: nx.Graph) -> gpd.GeoDataFrame:
    """
    Extract OSM edges of 6 types; buffer each and compute area.
    """
    buf = cfg.get("buffer_viability")
    rows = []
    for idx, (u, v, data) in enumerate(G_proj.edges(data=True)):
        hw = data.get("highway")
        hw_list = hw if isinstance(hw, list) else [hw]
        if not any(h in TYPE_MAP for h in hw_list if isinstance(h, str)):
            continue
        geom = LineString([
            (G_proj.nodes[u]['x'], G_proj.nodes[u]['y']),
            (G_proj.nodes[v]['x'], G_proj.nodes[v]['y'])
        ])
        rows.append({
            "segment_id": idx, "u": u, "v": v,
            "length_m": data.get("length", 0.0),
            "highway": hw, "geometry": geom
        })

    segs = gpd.GeoDataFrame(rows, crs="EPSG:3857")
    segs["buffer"] = segs.geometry.buffer(buf)
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
    Compute Table 2 features for each road segment (within buffer).
    """
    # 1) Population density
    with rasterio.open(worldpop_path) as src:
        buf_proj = segs["buffer"].to_crs(src.crs)
        union = buf_proj.unary_union
        img, tf = mask(src, [union], crop=True)
        band = img[0]
        shapes = ((geom, i) for i, geom in enumerate(buf_proj))
        mask_arr = features.rasterize(shapes, band.shape, transform=tf,
                                      fill=-1, dtype=np.int32)
        pops = np.array([band[mask_arr == i].sum() for i in range(len(segs))])
    segs["pop_density"] = pops / segs["area_km2"]

    # 2) POI density
    poi_j = gpd.sjoin(pois_gdf, segs.set_geometry("buffer"), predicate="within", how="inner")
    counts = poi_j.groupby("segment_id").size().reindex(segs.segment_id, fill_value=0)
    segs["poi_density"] = counts.values / segs["area_km2"]

    # 3) Wealth mean
    wj = gpd.sjoin(wealth_gdf, segs.set_geometry("buffer"), predicate="within", how="inner")
    wm = wj.groupby("segment_id")["rwi"].mean().reindex(segs.segment_id)
    segs["wealth_mean"] = wm.fillna(wm.median())

    # 4) Road density
    geoms, lens = segs.geometry.values, segs.length_m.values
    tree = STRtree(geoms)
    rd = []
    for i, buf in enumerate(segs["buffer"]):
        total = sum(lens[j] for g in tree.query(buf)
                    for j, gg in enumerate(geoms)
                    if id(gg) == id(g) and j != i and buf.intersects(gg))
        rd.append(total / segs.at[i, "area_km2"])
    segs["road_density"] = rd

    # 5) Road type ordinal
    def map_type(hw):
        if isinstance(hw, list):
            return max(TYPE_MAP.get(h, -1) for h in hw)
        return TYPE_MAP.get(hw, -1)
    segs["type_ord"] = segs.highway.apply(map_type).astype(int)

    # 6) Ping densities
    def bucket(ts):
        h = ts.hour
        if 7 <= h < 10: return "MP"
        if 10 <= h < 16: return "OP"
        if 16 <= h < 19: return "EP"
        if 19 <= h < 23: return "Night"
        return "OP"
    p = pings_gdf.copy()
    p["is_weekend"] = p.timestamp.dt.weekday >= 5
    p["bucket"] = p.apply(lambda r: "Weekend" if r.is_weekend else bucket(r.timestamp), axis=1)
    sb = segs[['segment_id', 'buffer']].set_geometry('buffer')
    for b in ["MP", "OP", "EP", "Night", "Weekend"]:
        sub = p[p.bucket == b]
        j = gpd.sjoin(sub, sb, predicate='within', how='inner')
        c = j.groupby('segment_id').size().reindex(segs.segment_id, fill_value=0)
        segs[f"{b}_dens"] = c.values / segs["area_km2"]

    # 7) Connected pings
    bufs = list(segs.buffer); ids = list(segs.segment_id)
    tree_b = STRtree(bufs)
    wday = p[~p.is_weekend]; wend = p[p.is_weekend]
    t_wday, t_wend = STRtree(wday.geometry), STRtree(wend.geometry)
    cw, cn = [], []
    for sid, buf in zip(ids, bufs):
        wd_cnt = sum(1 for pt in t_wday.query(buf) if buf.contains(pt))
        we_cnt = sum(1 for pt in t_wend.query(buf) if buf.contains(pt))
        area = segs.loc[segs.segment_id == sid, 'area_km2'].iat[0]
        cw.append(wd_cnt / area); cn.append(we_cnt / area)
    segs["conn_weekday_dens"], segs["conn_weekend_dens"] = cw, cn

    # 8) Transit waypoint density
    trans = p[p.ping_type == "transit"]
    t_tree = STRtree(trans.geometry)
    tw, twc = [], []
    for sid, buf in zip(ids, bufs):
        pts = [pt for pt in t_tree.query(buf) if buf.contains(pt)]
        area = segs.loc[segs.segment_id == sid, 'area_km2'].iat[0]
        tw.append(len(pts) / area)
        nbr_bufs = [bufs[k] for k in range(len(bufs)) if ids[k] != sid and buf.intersects(bufs[k])]
        cnt = sum(1 for pt in t_tree.query(buf) for nb in nbr_bufs if nb.contains(pt))
        twc.append(cnt / area)
    segs["transit_wp_dens"], segs["transit_wp_conn_dens"] = tw, twc

    # 9) Normalize
    feat_cols = [
        'pop_density', 'poi_density', 'wealth_mean', 'road_density', 'type_ord',
        'MP_dens', 'OP_dens', 'EP_dens', 'Night_dens', 'Weekend_dens',
        'conn_weekday_dens', 'conn_weekend_dens',
        'transit_wp_dens', 'transit_wp_conn_dens'
    ]
    X = segs[feat_cols].fillna(0.0)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols, index=segs.segment_id)
    return segs, X_scaled, scaler
