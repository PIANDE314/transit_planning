import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Tuple
from threading import Lock
from scipy.stats import linregress, pearsonr
from joblib import Parallel, delayed
from transitlib.config import Config

cfg = Config()

# -------------------- Global cache for reachability -------------------- #
_reachable_cache: Dict[Tuple[str, int, str], Set[str]] = {}
_cache_lock = Lock()

def reachable_from_cached(
    origin: str,
    dep_minute: int,
    zone_key: str,
    max_travel: int,
    trip_schedules: Dict[str, List[Tuple[str, int]]],
    stop_departures: Dict[str, List[Tuple[int, str, int]]]
) -> Set[str]:
    cache_key = (origin, dep_minute, zone_key)
    with _cache_lock:
        if cache_key in _reachable_cache:
            return _reachable_cache[cache_key]
    result = _reachable_from(origin, dep_minute, max_travel, trip_schedules, stop_departures)
    with _cache_lock:
        _reachable_cache[cache_key] = result
    return result

# -------------------- Load GTFS -------------------- #
def _load_gtfs_schedules(gtfs_dir: str) -> Tuple[
    Dict[str, List[Tuple[str, int]]],
    Dict[str, List[Tuple[int, str, int]]]
]:
    st = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))

    hms = st["departure_time"].str.split(":", expand=True).astype(int)
    st["dep_min"] = hms[0] * 60 + hms[1]

    hms = st["arrival_time"].str.split(":", expand=True).astype(int)
    st["arr_min"] = hms[0] * 60 + hms[1]

    trip_schedules = {
        trip_id: list(zip(df.sort_values("stop_sequence")["stop_id"], df.sort_values("stop_sequence")["arr_min"]))
        for trip_id, df in st.groupby("trip_id")
    }

    stop_departures: Dict[str, List[Tuple[int, str, int]]] = {}
    for trip_id, df in st.groupby("trip_id"):
        df_sorted = df.sort_values("departure_time")
        for seq_idx, (_, row) in enumerate(df_sorted.iterrows()):
            stop_departures.setdefault(row["stop_id"], []).append(
                (row["dep_min"], trip_id, seq_idx)
            )
    for stops in stop_departures.values():
        stops.sort(key=lambda x: x[0])

    return trip_schedules, stop_departures

# -------------------- Accessibility Comparison -------------------- #
def compare_accessibility(
    extracted_gtfs: str,
    operational_gtfs: str,
    zone_map: pd.DataFrame,
    wealth: pd.Series
  # phone_density: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]]]:
    start_time = cfg.get("start_time")
    window = cfg.get("access_window_min")
    max_travel = cfg.get("max_travel_min")
    n_jobs = cfg.get("n_jobs", 4)

    h0, m0, _ = map(int, start_time.split(":"))
    base_min = h0 * 60 + m0

    sched_ext, deps_ext = _load_gtfs_schedules(extracted_gtfs)
    sched_op,  deps_op  = _load_gtfs_schedules(operational_gtfs)

    zone_to_stops: Dict[str, Set[str]] = {}
    for _, row in zone_map.iterrows():
        zone_to_stops.setdefault(str(row["zone_id"]), set()).add(str(row["stop_id"]))

    def _per_zone_access(sched, deps, zone_key: str) -> pd.Series:
        def compute_access(zone: str) -> Tuple[str, float]:
            hits = 0
            targets = zone_to_stops[zone]
            for offset in range(window):
                dep_min = base_min + offset
                reachable = set()
                for o in targets:
                    reachable |= reachable_from_cached(o, dep_min, zone_key, max_travel, sched, deps)
                if reachable & targets:
                    hits += 1
            return zone, hits / window

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_access)(zone) for zone in zone_to_stops
        )
        return pd.Series(dict(results), name="accessibility")

    acc_ext = _per_zone_access(sched_ext, deps_ext, "ext")
    acc_op  = _per_zone_access(sched_op,  deps_op, "op")

    df = pd.concat([acc_op, acc_ext], axis=1, keys=["operational", "extracted"]).dropna()
    x, y = df["operational"].values, df["extracted"].values
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    mse = np.mean((y - x) ** 2)
    comp_stats = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "mse": mse
    }

    mse_zone = (acc_ext - acc_op).pow(2).rename("mse")
    dfb = pd.concat([mse_zone, wealth], axis=1).dropna()  # phone_density
    bias_stats = {
        "wealth": pearsonr(dfb.iloc[:, 1], dfb["mse"])
      # "phone":  pearsonr(dfb.iloc[:,2], dfb["mse"])
    }

    df_acc = pd.concat([
        acc_ext.rename("access_extracted"),
        acc_op.rename("access_operational")
    ], axis=1)

    return df_acc, comp_stats, bias_stats

# -------------------- Reachability -------------------- #
def _reachable_from(
    origin: str,
    dep_minute: int,
    max_travel: int,
    trip_schedules: Dict[str, List[Tuple[str, int]]],
    stop_departures: Dict[str, List[Tuple[int, str, int]]]
) -> Set[str]:
    visited: Set[Tuple[str, int]] = set()
    reachable: Set[str] = set()
    queue: List[Tuple[str, int]] = [(origin, dep_minute)]

    while queue:
        stop_id, t0 = queue.pop()
        for dep_time, trip_id, seq_idx in stop_departures.get(stop_id, []):
            if dep_time < t0 or dep_time > dep_minute + max_travel:
                continue
            schedule = trip_schedules[trip_id]
            for s_id, arr_min in schedule[seq_idx:]:
                travel = arr_min - dep_minute
                if travel > max_travel:
                    break
                reachable.add(s_id)
                state = (s_id, arr_min)
                if state not in visited:
                    visited.add(state)
                    queue.append(state)

    return reachable
