import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from transitlib.config import Config

# §3: Data Ingestion
from transitlib.data.download import download_file, fetch_worldpop_url, extract_worldpop_tif, fetch_hdx_rwi_url
from transitlib.data.osm import load_osm_network, load_osm_pois
from transitlib.data.preprocess import clip_raster_to_region, load_rwi_csv, points_to_gdf

# §3: Simulation
from transitlib.simulation.users import simulate_users

# §3.1: Viability
from transitlib.viability.segments import extract_road_segments, compute_segment_features
from transitlib.viability.model import run_self_training

# §3.2: Stop Location Extraction
from transitlib.stop_extraction.clustering import extract_candidate_stops

# §3.3: Route Extraction
from transitlib.route_extraction.graph_utils import build_stop_graph, compute_utilities
from transitlib.route_extraction.initial_routes import generate_initial_routes
from transitlib.route_extraction.optimize import score_usage, optimize_routes

# §3.4: GTFS Generation
from transitlib.gtfs_generation.writer import write_gtfs

# §3.5: Accessibility Comparison
from transitlib.accessibility.compare import compare_accessibility

def main():
    # 1) Load config
    cfg_path = Path(__file__).parent / "maputo_config.yml"
    cfg = Config(path=cfg_path)

    raw_dir = Path(cfg.get("raw_data_dir"))
    int_dir = Path(cfg.get("intermediate_dir"))
    gtfs_dir = Path(cfg.get("gtfs_output_dir"))

    raw_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    gtfs_dir.mkdir(parents=True, exist_ok=True)

    # 2) Download & unpack WorldPop
    wp_url = fetch_worldpop_url(cfg.get("country_code"))
    wp_zip = download_file(wp_url, raw_dir / "worldpop.zip", overwrite=False)
    wp_tif = extract_worldpop_tif(wp_zip, raw_dir)

    # 3) Download & load HDX RWI CSV
    rwi_url = fetch_hdx_rwi_url(cfg.get("country_name"), cfg.get("country_code"))
    rwi_csv = download_file(rwi_url, raw_dir / "rwi.csv", overwrite=False)
    wealth_df = load_rwi_csv(str(rwi_csv))
    wealth_gdf = points_to_gdf(wealth_df)

    # 4) Load OSM network & POIs
    place = cfg.get("place_name")
    G_latlon, G_proj = load_osm_network(place)
    pois = load_osm_pois(place)

    # 5) Simulate users & OD counts
    pings_gdf, od_counts = simulate_users(G_latlon, use_path_cache=False)

    # 6) Viability: extract segments & compute features
    segs = extract_road_segments(G_proj)
    segs_feat, feat_mat, scaler = compute_segment_features(
        segs, str(wp_tif), pois, wealth_gdf, pings_gdf
    )

    # 7) Self‑train to label viable vs non‑viable
    final_labels = run_self_training(segs, feat_mat, pois)
    segs["final_viable"] = segs_feat["segment_id"].map(final_labels).fillna(0).astype(int) #CHANGE THE ZERO LOGIC

    # 8) Stop extraction
    stops = extract_candidate_stops(segs, pings_gdf, pois, final_label="final_viable")
    # (optionally save)
    stops.to_file(int_dir / "maputo_stops.geojson", driver="GeoJSON")

    # 9) Build stop graph & utilities
    G_stop = build_stop_graph(G_latlon, od_counts, stops, footfall_col="footfall")
    Q, F, M, U = compute_utilities(G_stop)

    # 10) Generate & optimize routes
    init_routes = generate_initial_routes(G_stop, U)
    optimized = optimize_routes(G_stop, init_routes, Q, F, set(M.keys()), len(G_stop.nodes))

    # 11) Write GTFS feed
    write_gtfs(G_stop, optimized, lambda r: score_usage(r, Q, F))

    # 12) Accessibility comparison
    operational_gtfs = str(gtfs_dir)
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
    print("Accessibility comparison stats:", comp_stats, bias_stats)

    print(f"All done!  GTFS written to {gtfs_dir.resolve()}")

if __name__ == "__main__":
    main()
