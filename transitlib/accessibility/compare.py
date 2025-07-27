import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Tuple
from functools import lru_cache
from scipy.stats import linregress, pearsonr
from joblib import Parallel, delayed
from transitlib.config import Config

cfg = Config()

def _load_gtfs_schedules(gtfs_dir: str) -> Tuple[
    Dict[str, List[Tuple[str, int]]],
    Dict[str, List[Tuple[int, str, int]]]
]:
    """
    Parse stop_times.txt into:
      1) trip_schedules: trip_id → list of (stop_id, arrival_minute)
      2) stop_departures: stop_id → sorted list of (dep_minute, trip_id, seq_idx)
    """
    st = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    
    # Vectorized time conversion to minutes since midnight
    hms = st["departure_time"].str.split(":", expand=True).astype(int)
    st["dep_min"] = hms[0] * 60 + hms[1]

    hms = st["arrival_time"].str.split(":", expand=True).astype(int)
    st["arr_min"] = hms[0] * 60 + hms[1]

    # 1) trip_schedules
    trip_schedules = {
        trip_id: list(zip(df.sort_values("stop_sequence")["stop_id"], df.sort_values("stop_sequence")["arr_min"]))
        for trip_id, df in st.groupby("trip_id")
    }

    # 2) stop_departures
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

def compare_accessibility(
    extracted_gtfs: str,
    operational_gtfs: str,
    zone_map: pd.DataFrame,
    wealth: pd.Series
    #phone_density: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    1) Build GTFS schedules/departure indexes
    2) For each minute, compute per-zone accessibility fraction
    3) Compare extracted vs. operational (slope, r, mse)
    4) Bias vs. wealth
    """
    start_time = cfg.get("start_time")
    window = cfg.get("access_window_min")
    max_travel = cfg.get("max_travel_min")
    n_jobs = cfg.get("n_jobs", 4)

    h0, m0, _ = map(int, start_time.split(":"))
    base_min = h0 * 60 + m0

    sched_ext, deps_ext = _load_gtfs_schedules(extracted_gtfs)
    sched_op,  deps_op  = _load_gtfs_schedules(operational_gtfs)

    # zone → set of stop_ids
    zone_to_stops: Dict[str, Set[str]] = {}
    for _, row in zone_map.iterrows():
        zone_to_stops.setdefault(str(row["zone_id"]), set()).add(str(row["stop_id"]))

    # LRU-cached reachability to reduce recomputation
    @lru_cache(maxsize=None)
    def _reachable_from_cached(origin, dep_minute, key):
        return _reachable_from(origin, dep_minute, max_travel, key[0], key[1])

    def _per_zone_access(sched, deps) -> pd.Series:
        def compute_access(zone: str) -> Tuple[str, float]:
            hits = 0
            targets = zone_to_stops[zone]
            for offset in range(window):
                dep_min = base_min + offset
                reachable = set()
                for o in targets:
                    reachable |= _reachable_from_cached(o, dep_min, (sched, deps))
                if reachable & targets:
                    hits += 1
            return zone, hits / window

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_access)(zone) for zone in zone_to_stops
        )
        return pd.Series(dict(results), name="accessibility")

    acc_ext = _per_zone_access(sched_ext, deps_ext)
    acc_op  = _per_zone_access(sched_op,  deps_op)

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
    dfb = pd.concat([mse_zone, wealth], axis=1).dropna() # phone_density
    bias_stats = {
        "wealth": pearsonr(dfb.iloc[:, 1], dfb["mse"])
        # "phone":  pearsonr(dfb.iloc[:,2], dfb["mse"])
    }

    df_acc = pd.concat([
        acc_ext.rename("access_extracted"),
        acc_op.rename("access_operational")
    ], axis=1)

    return df_acc, comp_stats, bias_stats

def _reachable_from(
    origin: str,
    dep_minute: int,
    max_travel: int,
    trip_schedules: Dict[str, List[Tuple[str, int]]],
    stop_departures: Dict[str, List[Tuple[int, str, int]]]
) -> Set[str]:
    """
    BFS over same-stop transfers only.
    Returns set of stop_ids reachable from `origin` when
    boarding exactly at `dep_minute`, within `max_travel` minutes.
    """
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
