import os
import numpy as np
import pandas as pd
from threading import Lock
from joblib import Parallel, delayed
from typing import Dict, List, Set, Tuple
from transitlib.config import Config

cfg = Config()

# —————————————————————————————————————————————————————————————————————
# 1) Load GTFS via frequencies only
# —————————————————————————————————————————————————————————————————————

def _load_gtfs_frequencies(gtfs_dir: str
    ) -> Tuple[
        Dict[str, List[Tuple[str, int]]],
        Dict[str, List[Tuple[str, int, int, int, int]]]
    ]:
    """
    Returns:
      trip_schedules:
        trip_id → list of (stop_id, arrival_min) in stop_sequence order
      stop_frequencies:
        stop_id → list of (trip_id, seq_idx, start_min, end_min, headway_min)
    """
    # 1a) Read stop_times to build trip_schedules
    st = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    hms = st["arrival_time"].str.split(":", expand=True).astype(int)
    st["arr_min"] = hms[0]*60 + hms[1]

    trip_schedules: Dict[str, List[Tuple[str, int]]] = {
        trip_id: list(zip(
            df.sort_values("stop_sequence")["stop_id"],
            df.sort_values("stop_sequence")["arr_min"]
        ))
        for trip_id, df in st.groupby("trip_id")
    }

    # 1b) Read frequencies.txt
    freq_path = os.path.join(gtfs_dir, "frequencies.txt")
    freq = pd.read_csv(freq_path)
    hms = freq["start_time"].str.split(":", expand=True).astype(int)
    freq["start_min"] = hms[0]*60 + hms[1]
    hms = freq["end_time"].str.split(":", expand=True).astype(int)
    freq["end_min"] = hms[0]*60 + hms[1]
    freq["headway_min"] = (freq["headway_secs"] // 60)

    stop_frequencies: Dict[str, List[Tuple[str, int, int, int, int]]] = {}
    for _, row in freq.iterrows():
        trip_id    = row["trip_id"]
        start_min  = row["start_min"]
        end_min    = row["end_min"]
        headway    = row["headway_min"]
        # assign this frequency rule to each stop on that trip
        sched = trip_schedules[trip_id]
        for seq_idx, (stop_id, _) in enumerate(sched):
            stop_frequencies.setdefault(stop_id, []).append(
                (trip_id, seq_idx, start_min, end_min, headway)
            )

    return trip_schedules, stop_frequencies


# —————————————————————————————————————————————————————————————————————
# 2) Reachability using only headways
# —————————————————————————————————————————————————————————————————————

_reachable_cache: Dict[Tuple[str, int, str, int], Set[str]] = {}
_cache_lock = Lock()

def reachable_from_frequencies(
    origin: str,
    dep_minute: int,                   # base_min + offset
    cache_key_id: str,                 # unique id (e.g. gtfs_dir)
    max_travel: int,
    trip_schedules: Dict[str, List[Tuple[str, int]]],
    stop_frequencies: Dict[str, List[Tuple[str, int, int, int, int]]]
) -> Set[str]:
    """
    BFS over (stop, arrival_time) states, boarding via headway rules only.
    """
    cache_key = (origin, dep_minute, cache_key_id, max_travel)
    with _cache_lock:
        if cache_key in _reachable_cache:
            return _reachable_cache[cache_key]

    visited: Set[Tuple[str,int]] = set()
    reachable: Set[str] = set()
    queue: List[Tuple[str,int]] = [(origin, dep_minute)]

    while queue:
        stop_id, t0 = queue.pop()
        # for each frequency rule at this stop
        for trip_id, seq_idx, start_min, end_min, headway in \
                stop_frequencies.get(stop_id, []):

            # skip if you're after the end of service or beyond max travel
            if t0 > end_min or t0 > dep_minute + max_travel:
                continue

            # compute next possible departure ≥ t0
            first = max(t0, start_min)
            delta = (first - start_min) % headway
            next_dep = first if delta == 0 else first + (headway - delta)

            if next_dep > end_min or next_dep > dep_minute + max_travel:
                continue

            # board and ride through the rest of the trip
            for s_id, arr_min in trip_schedules[trip_id][seq_idx:]:
                # total travel from original start
                total_travel = arr_min - dep_minute
                if total_travel > max_travel:
                    break
                reachable.add(s_id)
                state = (s_id, arr_min)
                if state not in visited:
                    visited.add(state)
                    queue.append((s_id, arr_min))

    with _cache_lock:
        _reachable_cache[cache_key] = reachable
    return reachable


# —————————————————————————————————————————————————————————————————————
# 3) Calculate a single accessibility score for one GTFS
# —————————————————————————————————————————————————————————————————————

def calculate_accessibility(gtfs_dir: str) -> float:
    """
    Returns the average fraction of all stops reachable,
    across every stop-origin and every minute in the access window.
    """
    # load settings
    start_time = cfg.get("start_time")        # e.g. "06:00:00"
    window     = cfg.get("access_window_min") # e.g. 60
    max_travel = cfg.get("max_travel_min")    # e.g. 90
    n_jobs     = cfg.get("n_jobs", 4)

    # parse base minute
    h0, m0, _ = map(int, start_time.split(":"))
    base_min = h0 * 60 + m0

    # load GTFS
    sched, freqs = _load_gtfs_frequencies(gtfs_dir)
    all_stops   = list(freqs.keys())
    total_stops = len(all_stops)

    # fraction reachable from one origin at one time offset
    def frac_reachable(origin: str, offset: int) -> float:
        dep_min = base_min + offset
        reached = reachable_from_frequencies(
            origin, dep_min, gtfs_dir, max_travel, sched, freqs
        )
        # exclude self
        return len(reached) / (total_stops - 1)

    # build all tasks
    tasks = [(origin, off) for origin in all_stops for off in range(window)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda o, off: frac_reachable(o, off))(o, off)
        for o, off in tasks
    )

    # return overall mean accessibility (0.0–1.0)
    return float(np.mean(results))
