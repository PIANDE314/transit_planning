import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# Default parameters, organized by pipeline stage
# =============================================================================
DEFAULTS: Dict[str, Any] = {
    # === Data ingestion (§ 3) ===
    "download_timeout":  10,                # HTTP timeout (s)
    "pop_version":       "v2.1",            # WorldPop version

    # HDX RWI
    "hdx_search_api":    "https://data.humdata.org/api/3/action/package_search",
    "hdx_show_api":      "https://data.humdata.org/api/3/action/package_show",

    # OSM
    "osm_network_type":  "drive",
    "osm_poi_tags": {
        "highway": ["bus_stop"],
        "public_transport": ["platform", "station"],
        "amenity": ["bus_station"]
    },

    # === Simulation (§ 3 “Passive Datasets”) ===
    "n_users":           1,
    "avg_pings":         4.5,
    "transit_frac":      0.30,
    "sigma_deg":         0.0005,
    "simulation_periods": [
        {"start": "2019-07-01", "end": "2019-07-31"},
        {"start": "2019-11-01", "end": "2019-11-30"}
    ],
    "hourly_distribution": [1/24.0] * 24,

    # === Viability (§ 3.1, Table 2) ===
    "buffer_poi":        100.0,   # m buffer for POI stop‑infrastructure
    "buffer_viability":  500.0,   # m buffer for feature extraction
    "neg_percentile":    15.0,    # bottom X% for negative seeds
    "K_pos":             5,       # max new positive seeds per iter
    "K_neg":             5,       # max new negative seeds per iter
    "pos_thresh":        0.95,    # probability threshold for positives
    "neg_thresh":        0.80,    # probability threshold for negatives
    "logreg_neg_tresh":  0.90,    # treshold for label-0 segments
    "noise_thresh_frac": 0.5,     # trigger for noise injection
    "self_max_iters":    200,     # max self‑training iterations
    "self_test_size":    0.2,     # validation split for initial model
    "self_runs":         100,     # number of self-training runs
    "random_state":      0,       # global RNG seed

    # === Stop extraction (§ 3.2) ===
    "ping_buffer":       100.0,   # m buffer for selecting pings
    "db_eps":            200.0,   # DBSCAN eps (m)
    "db_min_samples":    1,       # DBSCAN min_samples
    "top_frac_stops":    0.10,    # fraction of clusters to keep

    # === Route extraction (§ 3.3) ===
    "num_initial_routes": 64,     # number of initial candidate routes
    "min_stops":          4,      # minimum stops per route
    "max_stops":          8,      # maximum stops per route
    "opt_max_iters":      100_000,# hill‑climbing iterations
    "weights": {                  # objective weights
        "usage":       0.4,
        "feasibility": 0.2,
        "coverage":    0.2,
        "directness":  0.2
    },

    # === GTFS generation (§ 3.4) ===
    "gtfs_output_dir":   "gtfs_feed",
    "service_id":        "S1",
    "agency_id":         "A1",
    "agency_name":       "Simulated Transit Agency",
    "agency_url":        "http://example.com",
    "agency_timezone":   "UTC",
    "start_time":        "16:00:00",
    "end_time":          "19:00:00",
    "start_date":        "20250101",
    "end_date":          "20251231",
    "bus_speed_kmh":     20.0,    # assumed bus speed

    # headway mapping by usage decile (1–10 → seconds)
    "decile_headways": {
        1: 3600, 2: 2700, 3: 1800, 4: 1500, 5: 1200,
        6: 900,  7: 600,  8: 300,  9: 180, 10: 120
    },

    # === Accessibility (§ 3.5) ===
    "max_travel_min":    30.0,    # reachable within 30 min
    "access_window_min": 60,      # evaluation window length (minutes)
    "access_fraction":   0.5      # must be reachable ≥50% of intervals
}


class Config:
    """
    Centralized configuration loader.

    Loads user overrides from a YAML/JSON file (path can be passed in or
    set via TRANSIT_PLANNING_CONFIG env var), and fills in any missing
    parameters from DEFAULTS above.
    """
    def __init__(self, path: Optional[Path] = None):
        # 1) Determine path: explicit > env var > error
        if path is None:
            env = os.getenv("TRANSIT_PLANNING_CONFIG")
            if env:
                path = Path(env)
            else:
                raise RuntimeError(
                    "No config path provided and TRANSIT_PLANNING_CONFIG not set"
                )
        self.path = Path(path)
        # 2) Load file (YAML or JSON)
        text = self.path.read_text()
        user_cfg = yaml.safe_load(text) or {}
        # 3) Merge: user overrides take precedence
        self._cfg: Dict[str, Any] = {**DEFAULTS, **user_cfg}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.
        If not specified in the file or DEFAULTS, return the provided default.
        """
        return self._cfg.get(key, default)
