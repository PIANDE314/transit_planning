import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import LineString
from shapely.strtree import STRtree
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


def batch_zonal_stats(
    buffers: gpd.GeoSeries,
    raster_path: str,
    stats: str = "sum",
    all_touched: bool = False,
    nodata=None,
    batch_size: int = 100
) -> np.ndarray:
    """Compute zonal_stats in batches, treating nodata as zero."""
    out = []
    with rasterio.open(raster_path) as src:
        src_nodata = nodata if nodata is not None else src.nodata
        for i in range(0, len(buffers), batch_size):
            chunk = buffers.iloc[i : i + batch_size]
            zs = zonal_stats(
                chunk.geometry,
                raster_path,
                stats=stats,
                all_touched=all_touched,
                nodata=src_nodata
            )
            # zonal_stats sum ignores nodata cells (i.e. treats them like zeros)
            out.extend(zs)
    return np.array([z[stats] for z in out])


def compute_segment_features(
    segs: gpd.GeoDataFrame,
    worldpop_path: str,
    pois_gdf: gpd.GeoDataFrame,
    wealth_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame,
    cfg,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, MinMaxScaler]:
    # --- 0) prep raster & geometries ---
    src = rasterio.open(worldpop_path)
    src_nodata = src.nodata

    # buffers in both CRS
    buffers_3857 = segs.set_geometry("buffer")
    buffers_raster = buffers_3857.to_crs(src.crs)

    # line‐segments DF for road‐density
    lines = segs[["segment_id", "geometry", "length_m"]].rename_geometry("geometry")

    # --- 1) Population density (sum over masked window) ---
    pop_sums = batch_zonal_stats(
        buffers_raster,
        worldpop_path,
        stats="sum",
        all_touched=False,
        nodata=src_nodata,
        batch_size=100
    )
    segs["pop_density"] = pop_sums / segs["area_km2"].values

    # --- 2) POI density (exact same join) ---
    poi_joined = gpd.sjoin(
        pois_gdf,
        buffers_3857,
        predicate="within",
        how="inner"
    )
    poi_counts = poi_joined.groupby("segment_id").size()
    segs["poi_density"] = poi_counts.reindex(segs.segment_id, fill_value=0) / segs["area_km2"]

    # --- 3) Wealth mean (exact same join) ---
    wj = gpd.sjoin(
        wealth_gdf,
        buffers_3857,
        predicate="within",
        how="inner"
    )
    wm = wj.groupby("segment_id")["rwi"].mean()
    segs["wealth_mean"] = wm.reindex(segs.segment_id).fillna(wm.median())

    # --- 4) Prepare pings & one big sjoin for all buffers ---
    p = pings_gdf.copy()
    p["is_weekend"] = p.timestamp.dt.weekday >= 5

    def _bucket(h):
        if  7 <= h < 10:   return "MP"
        if 10 <= h < 16:   return "OP"
        if 16 <= h < 19:   return "EP"
        if 19 <= h < 23:   return "Night"
        return "OP"

    p["bucket"] = p.timestamp.dt.hour.map(_bucket)
    p.loc[p.is_weekend, "bucket"] = "Weekend"

    p_proj = p.to_crs(buffers_3857.crs)
    pj = gpd.sjoin(
        p_proj,
        buffers_3857,
        predicate="within",
        how="inner"
    )

    # --- 5) Compute the ping‐based features just like before ---
    areas = segs.set_index("segment_id")["area_km2"]
    feat: Dict[str, pd.Series] = {}

    for b in ["MP","OP","EP","Night","Weekend"]:
        c = pj[pj.bucket == b].groupby("segment_id").size()
        feat[f"{b}_dens"] = c.reindex(segs.segment_id, 0) / areas

    for weekend_flag, name in [(False, "conn_weekday_dens"), (True, "conn_weekend_dens")]:
        u = (
            pj[pj.is_weekend == weekend_flag]
            .groupby("segment_id")["user_id"]
            .nunique()
        )
        feat[name] = u.reindex(segs.segment_id, 0) / areas

    trans = pj[pj.ping_type == "transit"]

    # transit waypoint density
    tcnt = trans.groupby("segment_id").size()
    feat["transit_wp_dens"] = tcnt.reindex(segs.segment_id, 0) / areas

    # --- 6) Neighbor‐pairs for transit‐wp‐connectivity ---
    buf_orig = buffers_3857[["segment_id", "buffer"]].rename_geometry("buffer")
    buf_nbr  = buf_orig.rename(columns={"buffer": "geometry"})
    buf_pairs = (
        gpd.sjoin(
            buf_orig,
            buf_nbr,
            predicate="intersects",
            how="inner",
            lsuffix="_orig",
            rsuffix="_nbr"
        )
        .query("segment_id_orig != segment_id_nbr")
    )

    # only pings that were in the segment to start with
    trans_orig = trans[["user_id", "geometry", "segment_id"]].rename(
        columns={"segment_id": "orig_sid"}
    ).set_geometry("geometry")

    # join those onto the neighbor buffers
    tp = gpd.sjoin(
        trans_orig,
        buf_pairs.rename(columns={"buffer_nbr": "geometry"}).set_geometry("geometry"),
        predicate="within",
        how="inner"
    )
    twc = tp.groupby("orig_sid")["user_id"].count()
    feat["transit_wp_conn_dens"] = twc.reindex(segs.segment_id, 0) / areas

    # --- 7) Road density via one spatial‐join, with proper suffixes ---
    buf_for_rd = buffers_3857.rename(columns={"buffer": "geometry"}).set_geometry("geometry")
    li = (
        gpd.sjoin(
            lines.rename_geometry("geometry"),
            buf_for_rd,
            predicate="intersects",
            how="inner",
            lsuffix="",
            rsuffix="_nbr"
        )
        .query("segment_id != segment_id_nbr")
        .merge(
            lines[["segment_id", "length_m"]]
            .rename(columns={"segment_id": "segment_id_nbr"}),
            on="segment_id_nbr"
        )
    )
    rd = li.groupby("segment_id")["length_m"].sum().reindex(segs.segment_id, 0)
    feat["road_density"] = rd / areas

    # --- 8) Highway type ordinal & assemble matrix ---
    segs["type_ord"] = segs.highway.map(
        lambda hw: max(
            TYPE_MAP.get(h, -1)
            for h in (hw if isinstance(hw, list) else [hw])
        )
    ).astype(int)

    feat_df = pd.DataFrame(feat)
    X = feat_df.fillna(0.0)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index, columns=X.columns
    )

    segs_out = pd.concat(
        [segs.set_index("segment_id"), feat_df],
        axis=1
    ).reset_index()

    return segs_out, X_scaled, scaler
