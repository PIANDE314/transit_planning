import yaml
from pathlib import Path
from typing import Any, Dict

# =============================================================================
# Default parameters, organized by pipeline stage
# =============================================================================
DEFAULTS: Dict[str, Any] = {
    # === Data ingestion (§ 3) ===
    "pop_version": "v2.1",            # WorldPop version
    "buffer_pop": 500.0,              # m buffer for WorldPop around segments
    "buffer_poi": 100.0,              # m buffer for POI stop‐infrastructure
    # === Viability (§ 3.1) ===
    "neg_percentile": 15.0,           # bottom X% for negative seeds
    "buffer_viability": 500.0,        # m buffer for feature extraction
    "K_pos": 5,                       # max new positive seeds per iter
    "K_neg": 5,                       # max new negative seeds per iter
    "pos_thresh": 0.95,               # probability threshold for positives
    "neg_thresh": 0.80,               # probability threshold for negatives
    "self_max_iters": 200,            # max self‐training iterations
    "self_test_size": 0.2,            # validation split for initial model
    "random_state": 42,               # global RNG seed
    # === Stop extraction (§ 3.2) ===
    "ping_buffer": 100.0,             # m buffer for selecting pings
    "db_eps": 200.0,                  # DBSCAN eps (m)
    "db_min_samples": 1,              # DBSCAN min_samples
    "top_frac_stops": 0.10,           # fraction of clusters to keep
    # === Route extraction (§ 3.3) ===
    "num_initial_routes": 64,         # number of initial candidate routes
    "min_stops": 4,                   # minimum stops per route
    "max_stops": 8,                   # maximum stops per route
    "opt_max_iters": 100_000,         # hill‑climbing iterations
    "weights": {                      # objective weights (usage, feasibility, coverage, directness)
        "usage":       0.4,
        "feasibility": 0.2,
        "coverage":    0.2,
        "directness":  0.2
    },
    # === GTFS generation (§ 3.4) ===
    "service_id":    "S1",
    "agency_id":     "A1",
    "agency_name":   "Simulated Transit Agency",
    "agency_url":    "http://example.com",
    "agency_timezone":"UTC",
    "start_time":    "16:00:00",
    "end_time":      "19:00:00",
    "start_date":    "20250101",
    "end_date":      "20251231",
    "bus_speed_kmh": 20.0,            # assumed bus speed
    # headway mapping by usage decile (1–10 → seconds)
    "decile_headways": {
        1: 3600, 2: 2700, 3: 1800, 4: 1500, 5: 1200,
        6: 900,  7: 600,  8: 300,  9: 180, 10: 120
    },
    # === Accessibility (§ 3.5) ===
    "max_travel_min": 30,             # reachable within 30 min
    "access_window_min": 60,          # evaluation window length (minutes)
    "access_fraction": 0.5            # must be reachable ≥50% of intervals
}

class Config:
    """
    Centralized configuration loader.  
    Loads user overrides from a YAML/JSON file and fills in any missing
    parameters from the defaults above.
    """
    def __init__(self, path: Path):
        # Load file (YAML or JSON)
        text = path.read_text()
        cfg_file = yaml.safe_load(text) or {}
        # Merge: user overrides take precedence
        self._cfg: Dict[str, Any] = {**DEFAULTS, **cfg_file}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.
        If not specified in the file or DEFAULTS, return the provided default.
        """
        return self._cfg.get(key, default)
