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
from typing import Tuple, Dict, List
from transitlib.config import Config
from joblib import Parallel, delayed
from shapely.strtree import STRtree

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
    total_edges = G_proj.number_of_edges()
    kept = 0
    rows = []
    for idx, (u, v, data) in enumerate(G_proj.edges(data=True)):
        hw = data.get("highway")
        hw_list = hw if isinstance(hw, list) else [hw]
        if not any(isinstance(h, str) and h in TYPE_MAP for h in hw_list):
            continue
        kept += 1
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
    # 1) Parallelize the per‐chunk zonal_stats
    with rasterio.open(raster_path) as src:
        src_nodata = nodata if nodata is not None else src.nodata
    # split into chunks
    chunks = [
        buffers.iloc[i : i + batch_size]
        for i in range(0, len(buffers), batch_size)
    ]

    def _zs_chunk(chunk: gpd.GeoSeries) -> List[float]:
        zs = zonal_stats(
            chunk.geometry,
            raster_path,
            stats=stats,
            all_touched=all_touched,
            nodata=src_nodata
        )
        return [z[stats] for z in zs]

    results = Parallel(n_jobs=cfg.get("n_jobs", 4))(
        delayed(_zs_chunk)(chunk) for chunk in chunks
    )
    # flatten
    flat = [val for sub in results for val in sub]
    return np.array(flat)


def compute_segment_features(
    segs: gpd.GeoDataFrame,
    worldpop_path: str,
    pois_gdf: gpd.GeoDataFrame,
    wealth_gdf: gpd.GeoDataFrame,
    pings_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, MinMaxScaler]:
    # --- 0) prep raster & geometries ---
    src = rasterio.open(worldpop_path)
    src_nodata = src.nodata

    # buffers in both CRS
    buffers_3857 = segs.set_geometry("buffer")
    buffers_raster = buffers_3857.to_crs(src.crs)

    # line‐segments DF for road‐density
    lines = segs[["segment_id", "geometry", "length_m"]].set_geometry("geometry")

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

    # --- 2) POI density (make sure POIs are in the same CRS) ---
    pois_proj = pois_gdf.to_crs(buffers_3857.crs)

    # spatial join
    poi_joined = gpd.sjoin(
        pois_proj,
        buffers_3857,
        predicate="within",
        how="inner"
    )
    
    poi_counts = poi_joined.groupby("segment_id").size()
    segs["poi_density"] = poi_counts.reindex(segs.segment_id, fill_value=0) / segs["area_km2"]

    # --- 3) Wealth mean (exact same join) ---
    wealth_proj = wealth_gdf.to_crs(buffers_3857.crs)
    wj = gpd.sjoin(
        wealth_proj,
        buffers_3857,
        predicate="within",
        how="inner"
    )
    wm = wj.groupby("segment_id")["rwi"].mean()
    segs["wealth_mean"] = wm.reindex(segs.segment_id).fillna(wm.median())

    # --- 4) Prepare pings & one big sjoin for all buffers ---
    p = pings_gdf.copy()
    p["is_weekend"] = p.timestamp.dt.weekday >= 5

    hour2bucket = np.array([
        "Night",  # 0
        "Night",  # 1
        "Night",  # 2
        "Night",  # 3
        "Night",  # 4
        "Night",  # 5
        "Night",  # 6
        "MP",     # 7
        "MP",     # 8
        "MP",     # 9
        "OP",     # 10
        "OP",     # 11
        "OP",     # 12
        "OP",     # 13
        "OP",     # 14
        "OP",     # 15
        "EP",     # 16
        "EP",     # 17
        "EP",     # 18
        "Night",  # 19
        "Night",  # 20
        "Night",  # 21
        "Night",  # 22
        "Night",  # 23
    ], dtype="U8")

    hrs = p.timestamp.dt.hour.values
    p["bucket"] = hour2bucket[hrs]
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

    # bucket density
    dens = pj.pivot_table(
        index="segment_id", columns="bucket",
        values="user_id", aggfunc="size", fill_value=0
    )
    for b in dens.columns:
        feat[f"{b}_dens"] = dens[b].reindex(segs.segment_id, fill_value=0) / areas

    # unique user connectivity
    uniq = pj.pivot_table(
        index="segment_id", columns="is_weekend",
        values="user_id", aggfunc=lambda x: x.nunique(), fill_value=0
    )
    feat["conn_weekday_dens"] = uniq.get(False, pd.Series(0, index=uniq.index)).reindex(segs.segment_id, fill_value=0) / areas
    feat["conn_weekend_dens"] = uniq.get(True, pd.Series(0, index=uniq.index)).reindex(segs.segment_id, fill_value=0) / areas

    trans = pj[pj.ping_type == "transit"]

    # transit waypoint density
    tcnt = trans.groupby("segment_id").size()
    feat["transit_wp_dens"] = tcnt.reindex(segs.segment_id, fill_value=0) / areas

    # --- Neighbor‐pairs for transit‐wp‐connectivity ---
    buffers_3857 = buffers_3857.set_geometry("buffer")

    # — 1) Compute all buffer‐neighbor pairs via a GeoPandas self‐join —
    bufs = buffers_3857[["segment_id", "buffer"]]
    bufs = bufs.set_geometry("buffer")
    pairs = gpd.sjoin(
        bufs, bufs,
        predicate="intersects",
        how="inner",
        lsuffix="orig", rsuffix="nbr"
    ).reset_index(drop=True)
    
    # Drop self‐joins & duplicates A–B vs B–A
    pairs = pairs[pairs.segment_id_orig != pairs.segment_id_nbr].copy()
    pairs["pair_key"] = pairs.apply(
        lambda r: frozenset({r.segment_id_orig, r.segment_id_nbr}), axis=1
    )
    pairs = pairs.drop_duplicates("pair_key")
    
    # — 2) transit_wp_conn_dens —
    # a) isolate transit pings and rename their segment column
    trans = pj[pj.ping_type == "transit"]
    trans_orig = (
        trans.rename(columns={"segment_id": "orig_sid"})
             [["user_id","geometry","orig_sid"]]
             .set_geometry("geometry")
    )
    
    # b) build a GeoDataFrame of neighbor buffers keyed by nbr_sid
    nbr_geoms = buffers_3857.set_index("segment_id").geometry.loc[
        pairs.segment_id_nbr
    ].values
    nbr_buf = gpd.GeoDataFrame({
        "nbr_sid": pairs.segment_id_nbr
    }, geometry=nbr_geoms, crs=buffers_3857.crs)
    
    # c) spatial join pings → neighbor buffers
    #    we’ll keep trans_orig.orig_sid and nbr_buf.nbr_sid separate
    tp = gpd.sjoin(
        trans_orig, nbr_buf,
        predicate="within",
        how="inner",
        lsuffix="orig", rsuffix="nbr"
    )
    
    # d) group by orig_sid (now safe) and normalize
    twc = tp.groupby("orig_sid")["user_id"].size()
    feat["transit_wp_conn_dens"] = twc.reindex(segs.segment_id, fill_value=0) / areas
    
    # — 3) road_density —
    rd_df = pairs[["segment_id_orig","segment_id_nbr"]].rename(
        columns={"segment_id_orig":"orig","segment_id_nbr":"nbr"}
    )
    # join on nbr → length_m
    rd_df = rd_df.merge(
        lines[["segment_id","length_m"]]
             .rename(columns={"segment_id":"nbr"}),
        on="nbr", how="left"
    )
    rd_sum = rd_df.groupby("orig")["length_m"].sum()
    feat["road_density"] = rd_sum.reindex(segs.segment_id, fill_value=0) / areas

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
