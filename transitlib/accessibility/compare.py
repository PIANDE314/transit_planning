import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from scipy.stats import linregress, pearsonr
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
    # convert HH:MM:SS → minutes since midnight
    def to_min(t: str) -> int:
        h, m, s = map(int, t.split(":"))
        return h * 60 + m

    st["dep_min"] = st["departure_time"].apply(to_min)
    st["arr_min"] = st["arrival_time"].apply(to_min)

    # 1) trip_schedules
    trip_schedules: Dict[str, List[Tuple[str, int]]] = {}
    for trip_id, df in st.groupby("trip_id"):
        # sort by stop_sequence
        df = df.sort_values("stop_sequence")
        trip_schedules[trip_id] = list(zip(df["stop_id"], df["arr_min"]))

    # 2) stop_departures
    stop_departures: Dict[str, List[Tuple[int, str, int]]] = {}
    for trip_id, df in st.groupby("trip_id"):
        for seq_idx, (_, row) in enumerate(df.sort_values("departure_time").iterrows()):
            stop_departures.setdefault(row["stop_id"], []).append(
                (row["dep_min"], trip_id, seq_idx)
            )
    # sort each list by dep_min
    for stops in stop_departures.values():
        stops.sort(key=lambda x: x[0])

    return trip_schedules, stop_departures


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
    visited: Set[Tuple[str, int]] = set()  # (stop_id, time_minute)
    reachable: Set[str] = set()
    queue: List[Tuple[str, int]] = [(origin, dep_minute)]

    while queue:
        stop_id, t0 = queue.pop()
        # look up all departures from stop at or after t0
        for dep_time, trip_id, seq_idx in stop_departures.get(stop_id, []):
            if dep_time < t0 or dep_time > dep_minute + max_travel:
                continue
            # follow that trip from seq_idx onward
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


def compare_accessibility(
    extracted_gtfs: str,
    operational_gtfs: str,
    zone_map: pd.DataFrame,
    wealth: pd.Series,
    phone_density: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    1) Build GTFS schedules/departure indexes
    2) For each minute, compute per-zone accessibility fraction
    3) Compare extracted vs. operational (slope, r, mse)
    4) Bias vs. wealth & phone-density
    """
    # config
    service_date = cfg.get("start_date")       # not used directly here
    start_time = cfg.get("start_time")         # used for HH:MM:SS → base minute
    window = cfg.get("access_window_min")
    max_travel = cfg.get("max_travel_min")
    threshold = cfg.get("access_fraction")

    # parse start_time into base minute
    h0, m0, _ = map(int, start_time.split(":"))
    base_min = h0 * 60 + m0

    # load schedules
    sched_ext, deps_ext = _load_gtfs_schedules(extracted_gtfs)
    sched_op,  deps_op  = _load_gtfs_schedules(operational_gtfs)

    # build zone → stops map
    zone_to_stops: Dict[str, Set[str]] = {}
    for _, row in zone_map.iterrows():
        zone_to_stops.setdefault(str(row["zone_id"]), set()).add(str(row["stop_id"]))

    # compute per-zone, per-minute accessibility
    def _per_zone_access(sched, deps) -> pd.Series:
        acc: Dict[str, int] = {z: 0 for z in zone_to_stops}
        for minute_offset in range(window):
            dep_min = base_min + minute_offset
            # precompute reachable stops from each origin stop
            origin_cache: Dict[str, Set[str]] = {}
            for zone, origins in zone_to_stops.items():
                # check if any origin in zone can reach zone's stops >= threshold
                reachable_union: Set[str] = set()
                for o in origins:
                    if o not in origin_cache:
                        origin_cache[o] = _reachable_from(o, dep_min, max_travel, sched, deps)
                    reachable_union |= origin_cache[o]
                # how many of this zone's targets are reached?
                hits = len(reachable_union & zone_to_stops[zone])
                if hits > 0:
                    acc[zone] += 1
        # fraction of minutes where accessible ≥ threshold
        frac = {z: cnt / window for z, cnt in acc.items()}
        return pd.Series(frac, name="accessibility")

    acc_ext = _per_zone_access(sched_ext, deps_ext)
    acc_op  = _per_zone_access(sched_op,  deps_op)

    # comparison stats
    df = pd.concat([acc_op, acc_ext], axis=1, keys=["operational","extracted"]).dropna()
    x, y = df["operational"].values, df["extracted"].values
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    mse = np.mean((y - x)**2)
    comp_stats = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "mse": mse
    }

    # bias analysis
    mse_zone = (acc_ext - acc_op).pow(2).rename("mse")
    dfb = pd.concat([mse_zone, wealth, phone_density], axis=1).dropna()
    bias_stats = {
        "wealth": pearsonr(dfb.iloc[:,1], dfb["mse"]),
        "phone":  pearsonr(dfb.iloc[:,2], dfb["mse"])
    }

    # final DataFrame
    df_acc = pd.concat([
        acc_ext.rename("access_extracted"),
        acc_op.rename("access_operational")
    ], axis=1)

    return df_acc, comp_stats, bias_stats
