import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
from transitlib.config import Config
from transitlib.data.download import download_file, fetch_worldpop_url, extract_worldpop_tif, fetch_hdx_rwi_url
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

# — 1) Stage functions return only their “delta” dict —  
def step_download(ctx):
    """Download worldpop + RWI, stash in ctx."""
    # WorldPop
    wp_url  = fetch_worldpop_url(cfg.get("country_code"))
    wp_zip  = download_file(wp_url, raw_dir / "worldpop.zip", overwrite=False)
    wp_tif  = extract_worldpop_tif(wp_zip, raw_dir)

    # RWI
    rwi_url    = fetch_hdx_rwi_url(cfg.get("country_name"), cfg.get("country_code"))
    rwi_csv    = download_file(rwi_url, raw_dir / "rwi.csv", overwrite=False)
    wealth_df  = load_rwi_csv(str(rwi_csv))
    wealth_gdf = points_to_gdf(wealth_df)

    return {"wp_tif": wp_tif, "wealth_gdf": wealth_gdf}

def step_osm(ctx):
    """Load OSM network + POIs, stash in ctx."""
    place = cfg.get("place_name")
    G_latlon, G_proj = load_osm_network(place)
    pois             = load_osm_pois(place)
    return {"G_latlon": G_latlon, "G_proj": G_proj, "pois": pois}

def step_simulation(ctx):
    """Simulate users, stash in ctx."""
    G_latlon = ctx["G_latlon"]
    pings_gdf, od_counts = simulate_users(G_latlon, use_path_cache=False)
    return {"pings_gdf": pings_gdf, "od_counts": od_counts}

def step_viability(ctx):
    """Extract & featurize segments + self-train, stash 'segs'."""
    G_proj     = ctx["G_proj"]
    wp_tif     = ctx["wp_tif"]
    pois       = ctx["pois"]
    wealth_gdf = ctx["wealth_gdf"]
    pings_gdf  = ctx["pings_gdf"]

    segs = extract_road_segments(G_proj)
    segs_feat, feat_mat, scaler = compute_segment_features(
        segs, str(wp_tif), pois, wealth_gdf, pings_gdf
    )
    final_labels = run_self_training(segs, feat_mat, pois)
    segs["final_viable"] = (
        segs_feat["segment_id"].map(final_labels).fillna(0).astype(int)
    )
    return {"segs": segs}

def step_stops(ctx):
    """Extract candidate stops, stash 'stops'."""
    segs      = ctx["segs"]
    pings_gdf = ctx["pings_gdf"]
    pois      = ctx["pois"]

    stops = extract_candidate_stops(segs, pings_gdf, pois, final_label="final_viable")
    stops.to_file(int_dir / "stops.geojson", driver="GeoJSON")
    return {"stops": stops}

def step_routes(ctx):
    """Build stop graph, compute & optimize routes, stash G_stop/Q/F/optimized."""
    G_latlon  = ctx["G_latlon"]
    od_counts = ctx["od_counts"]
    stops     = ctx["stops"]

    G_stop = build_stop_graph(G_latlon, od_counts, stops, footfall_col="footfall")
    Q, F, M, U = compute_utilities(G_stop)
    init_routes = generate_initial_routes(G_stop, U)
    optimized   = optimize_routes(G_stop, init_routes, Q, F, set(M.keys()), len(G_stop.nodes))

    return {"G_stop": G_stop, "Q": Q, "F": F, "optimized": optimized}

def step_gtfs(ctx):
    """Write GTFS feed, stash 'gtfs_dir'."""
    G_stop    = ctx["G_stop"]
    optimized = ctx["optimized"]
    Q         = ctx["Q"]
    F         = ctx["F"]

    write_gtfs(G_stop, optimized, lambda r: score_usage(r, Q, F), output_dir=gtfs_dir)
    return {"gtfs_dir": gtfs_dir}

def step_compare(ctx):
    """(Optional) Compare accessibility; stash 'acc_df'."""
    # Uncomment and adapt if you want to run this:
    # operational_gtfs = str(ctx["gtfs_dir"])
    # zones = gpd.read_file(raw_dir / "gadm41_MOZ_3.json")
    # stops = gpd.read_file(int_dir / "stops.geojson")
    # stops = stops.to_crs(zones.crs)
    # stops["stop_id"] = stops.index.astype(str)
    # stops_with_zone = gpd.sjoin(stops, zones, how="inner", predicate="within")
    # zone_id_col = "GID_3"
    # zone_map_df = pd.DataFrame({
    #     "zone_id": stops_with_zone[zone_id_col],
    #     "stop_id": stops_with_zone["stop_id"]
    # })
    # zone_map_df.to_csv("maputo_zone_map.csv", index=False)
    # zone_map = pd.read_csv("maputo_zone_map.csv")
    # acc_df, comp_stats, bias_stats = compare_accessibility(
    #     operational_gtfs, operational_gtfs, zone_map,
    #     ctx["wealth_gdf"]["rwi"]
    # )
    # acc_df.to_csv(int_dir / "accessibility_comparison.csv", index=False)
    # print("Accessibility comparison stats:", comp_stats, bias_stats)
    return {}

# — 2) Define our stages (all single-choice for now) —  
stages = [
    {"name": "download",    "choices": ["once"], "function_for": {"once": step_download}},
    {"name": "osm_load",    "choices": ["once"], "function_for": {"once": step_osm}},
    {"name": "simulation",  "choices": ["once"], "function_for": {"once": step_simulation}},
    {"name": "viability",   "choices": ["once"], "function_for": {"once": step_viability}},
    {"name": "stops",       "choices": ["once"], "function_for": {"once": step_stops}},
    {"name": "routes",      "choices": ["once"], "function_for": {"once": step_routes}},
]

# — 3) Caching & DFS machinery —  
def cache_path(intermediate_dir: Path, prefix_choices):
    key_parts = []
    for stage_name, choice in prefix_choices:
        stage = next(s for s in stages if s["name"] == stage_name)
        if len(stage["choices"]) > 1:
            key_parts.append(f"{stage_name}-{choice}")
    key = "__".join(key_parts) if key_parts else "base"
    return intermediate_dir / f"{key}.pkl"

def run_cached(fn, cache_file: Path, ctx):
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    delta = fn(ctx)
    with open(cache_file, "wb") as f:
        pickle.dump(delta, f)
    return delta

final_results = []

def dfs(stage_idx, ctx, prefix_choices):
    if stage_idx >= len(stages):
        final_results.append(ctx.copy())
        return

    stage = stages[stage_idx]
    for choice in stage["choices"]:
        new_prefix = prefix_choices + [(stage["name"], choice)]
        cache_file = cache_path(int_dir, new_prefix)
        delta = run_cached(stage["function_for"][choice], cache_file, ctx)
        # merge only the new outputs into a fresh context
        new_ctx = ctx.copy()
        new_ctx.update(delta)
        dfs(stage_idx + 1, new_ctx, new_prefix)

if __name__ == "__main__":
    # start with empty context
    initial_ctx = {}
    dfs(0, initial_ctx, [])
    print("All done!  Generated", len(final_results), "pipeline outputs.")
    for ctx in final_results:
        print("  gtfs_dir:", ctx.get("gtfs_dir"))
