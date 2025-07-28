import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
from transitlib.config import Config
from transitlib.data.download import (
    download_file, fetch_worldpop_url, extract_worldpop_tif,
    fetch_hdx_rwi_url
)
from transitlib.data.osm import load_osm_network, load_osm_pois
from transitlib.data.preprocess import load_rwi_csv, points_to_gdf
from transitlib.simulation.users import simulate_users
from transitlib.viability.segments import extract_road_segments, compute_segment_features
from transitlib.viability.model import run_self_training
from transitlib.stop_extraction.clustering import extract_candidate_stops
from transitlib.route_extraction.graph_utils import build_stop_graph, compute_utilities
from transitlib.route_extraction.initial_routes import generate_initial_routes
from transitlib.route_extraction.optimize import score_usage, optimize_routes
from transitlib.gtfs_generation.writer import write_gtfs
from transitlib.accessibility.compare import compare_accessibility

# — CONFIG & DIRS —  
cfg_path = Path(__file__).parent / "maputo_config.yml"
cfg      = Config(path=cfg_path)

raw_dir  = Path(cfg.get("raw_data_dir"));      raw_dir.mkdir(parents=True, exist_ok=True)
int_dir  = Path(cfg.get("intermediate_dir"));  int_dir.mkdir(parents=True, exist_ok=True)
gtfs_dir = Path(cfg.get("gtfs_output_dir"));  gtfs_dir.mkdir(parents=True, exist_ok=True)
outputs_dir = Path("output"); outputs_dir.mkdir(parents=True, exist_ok=True)

# — 1) Stage functions return only their “delta” dict —  
def step_download(ctx):
    wp_url  = fetch_worldpop_url(cfg.get("country_code"))
    wp_zip  = download_file(wp_url, raw_dir / "worldpop.zip", overwrite=False)
    wp_tif  = extract_worldpop_tif(wp_zip, raw_dir)

    rwi_url    = fetch_hdx_rwi_url(cfg.get("country_name"), cfg.get("country_code"))
    rwi_csv    = download_file(rwi_url, raw_dir / "rwi.csv", overwrite=False)
    wealth_df  = load_rwi_csv(str(rwi_csv))
    wealth_gdf = points_to_gdf(wealth_df)

    return {"wp_tif": wp_tif, "wealth_gdf": wealth_gdf}

def step_osm(ctx):
    place = cfg.get("place_name")
    G_latlon, G_proj = load_osm_network(place)
    pois             = load_osm_pois(place)
    return {"G_latlon": G_latlon, "G_proj": G_proj, "pois": pois}

def step_simulation(ctx):
    pings_gdf, od_counts = simulate_users(ctx["G_latlon"], use_path_cache=False)
    return {"pings_gdf": pings_gdf, "od_counts": od_counts}

def step_viability(ctx):
    segs = extract_road_segments(ctx["G_proj"])
    segs_feat, feat_mat, _ = compute_segment_features(
        segs, str(ctx["wp_tif"]), ctx["pois"], ctx["wealth_gdf"], ctx["pings_gdf"]
    )
    final_labels = run_self_training(segs, feat_mat, ctx["pois"])
    segs["final_viable"] = (
        segs_feat["segment_id"].map(final_labels).fillna(0).astype(int)
    )
    return {"segs": segs}

def step_stops(ctx):
    stops = extract_candidate_stops(
        ctx["segs"], ctx["pings_gdf"], ctx["pois"], final_label="final_viable"
    )
    return {"stops": stops}

def step_routes(ctx):
    G_stop = build_stop_graph(
        ctx["G_latlon"], ctx["od_counts"], ctx["stops"], footfall_col="footfall"
    )
    Q, F, M, U = compute_utilities(G_stop)
    init_routes = generate_initial_routes(G_stop, U)
    optimized   = optimize_routes(G_stop, init_routes, Q, F, set(M.keys()), len(G_stop.nodes))
    return {"G_stop": G_stop, "Q": Q, "F": F, "optimized": optimized}

def step_gtfs(ctx):
    out_dir = ctx["gtfs_out_dir"]
    write_gtfs(ctx["G_stop"], ctx["optimized"], lambda r: score_usage(r, ctx["Q"], ctx["F"]),
               output_dir=out_dir)
    return {"gtfs_dir": out_dir}

def step_compare(ctx):
    """operational_gtfs = str(gtfs_dir)
    zones = gpd.read_file(raw_dir / "gadm41_MOZ_3.json")
    stops = gpd.read_file(int_dir / "maputo_stops.geojson")
    stops = stops.to_crs(zones.crs)    
    stops["stop_id"] = stops.index.astype(str)
    stops_with_zone = gpd.sjoin(stops, zones, how="inner", predicate="within")
    zone_id_col = "GID_3"
    zone_map_df = pd.DataFrame({
        "zone_id": stops_with_zone[zone_id_col],
        "stop_id": stops_with_zone["stop_id"]
    })
    zone_map_df.to_csv("maputo_zone_map.csv", index=False)
    zone_map = pd.read_csv("maputo_zone_map.csv")
    # phone_density = pd.read_csv("path/to/phone_density.csv")
    acc_df, comp_stats, bias_stats = compare_accessibility(
        str(gtfs_dir), operational_gtfs, zone_map, wealth_df["rwi"] #phone_density["density"]
    )
    acc_df.to_csv(int_dir / "accessibility_comparison.csv", index=False)
    print("Accessibility comparison stats:", comp_stats, bias_stats)"""
    return {}

# — 2) Stage definitions (all single‐choice for now) —  
stages = [
    {"name": "download",   "choices": ["once"], "fn": step_download},
    {"name": "osm_load",   "choices": ["once"], "fn": step_osm},
    {"name": "simulation", "choices": ["once"], "fn": step_simulation},
    {"name": "viability",  "choices": ["once"], "fn": step_viability},
    {"name": "stops",      "choices": ["once"], "fn": step_stops},
    {"name": "routes",     "choices": ["once"], "fn": step_routes},
    {"name": "gtfs",       "choices": ["once"], "fn": step_gtfs},
    {"name": "compare",    "choices": ["once"], "fn": step_compare},
]

SKIP_CACHE = {"download", "osm_load", "gtfs", "compare"}

# — 3) Caching & DFS —  
def cache_path(stage_name, choice):
    """
    Each stage gets its own file.
    If it branches, we’ll append '__choice'.
    """
    if len(next(s for s in stages if s["name"] == stage_name)["choices"]) > 1:
        return int_dir / f"{stage_name}__{choice}.pkl"
    else:
        return int_dir / f"{stage_name}.pkl"

def run_cached(stage_name, choice, ctx):
    """
    If stage is in SKIP_CACHE, always re-run its fn (no pickle load or dump).
    Otherwise, behave exactly as before.
    """
    stage = next(s for s in stages if s["name"] == stage_name)
    stage_fn = stage["fn"]

    if stage_name in SKIP_CACHE:
        if stage_name == "gtfs":
            labels = ctx.get("labels", [])
            if labels:
                dirname = "_".join(labels) + "_gtfs"
            else:
                dirname = "gtfs"
            out_dir   = outputs_dir / dirname
            out_dir.mkdir(parents=True, exist_ok=True)

            ctx = {**ctx, "gtfs_out_dir": out_dir}
            return stage_fn(ctx)
        return stage_fn(ctx)

    cache_file = cache_path(stage_name, choice)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    delta = stage_fn(ctx)
    with open(cache_file, "wb") as f:
        pickle.dump(delta, f)
    return delta

final_results = []

def dfs(idx, ctx):
    if idx == len(stages):
        final_results.append(ctx.copy())
        return
    
    stage = stages[idx]
    multi = len(stage["choices"]) > 1
    
    for choice in stage["choices"]:
        new_ctx = ctx.copy()
        
        if multi:
            new_ctx["labels"] = ctx.get("labels", []) + [choice]
        else:
            new_ctx["labels"] = ctx.get("labels", []).copy()
        
        delta = run_cached(stage["name"], choice, new_ctx)
        new_ctx.update(delta)
        dfs(idx + 1, new_ctx)

if __name__ == "__main__":
    dfs(0, {"labels": []})  # start with empty context
    print(f"All done!  Generated {len(final_results)} outputs.")
    for out in final_results:
        print("  gtfs_dir:", out.get("gtfs_dir"))
