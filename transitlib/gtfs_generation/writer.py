import os
from collections import defaultdict
from typing import Callable, Dict, List
import networkx as nx
import pandas as pd
from pathlib import Path
from transitlib.config import Config

cfg = Config()

def write_gtfs(
    G: nx.Graph,
    optimized_routes: List[List[int]],
    score_usage_fn: Callable[[List[int]], float]
) -> None:
    """
    Write a GTFS feed from the optimized routes and stop graph.
    """
    output_dir = cfg.get("gtfs_output_dir", "gtfs_feed")
    os.makedirs(output_dir, exist_ok=True)

    speed_mps = cfg.get("bus_speed_kmh") * 1000.0 / 3600.0

    start_time = cfg.get("start_time")
    end_time = cfg.get("end_time")
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")

    service_id = cfg.get("service_id")
    agency_id = cfg.get("agency_id")
    agency_name = cfg.get("agency_name")
    agency_url = cfg.get("agency_url")
    agency_tz = cfg.get("agency_timezone")

    headway_map = cfg.get("decile_headways")

    # 1. stops.txt
    stops = [
        {
            "stop_id": str(n),
            "stop_name": G.nodes[n].get("stop_name", f"Stop {n}"),
            "stop_lat": G.nodes[n]["lat"],
            "stop_lon": G.nodes[n]["lon"]
        }
        for n in G.nodes
    ]
    pd.DataFrame(stops).to_csv(os.path.join(output_dir, "stops.txt"), index=False)

    # 2. stop_times.txt & 3. shapes.txt
    stop_times, shapes = [], []
    for i, route in enumerate(optimized_routes, start=1):
        trip_id = f"T{i}"
        shape_id = f"S{i}"

        cum_dist = 0.0
        for seq, stop in enumerate(route, start=1):
            shapes.append({
                "shape_id": shape_id,
                "shape_pt_lat": G.nodes[stop]["lat"],
                "shape_pt_lon": G.nodes[stop]["lon"],
                "shape_pt_sequence": seq - 1,
                "shape_dist_traveled": round(cum_dist)
            })
            if seq < len(route):
                cum_dist += nx.shortest_path_length(G, route[seq - 1], route[seq], weight="length")

        h0, m0, s0 = map(int, start_time.split(":"))
        t_sec = h0 * 3600 + m0 * 60 + s0

        for seq, stop in enumerate(route, start=1):
            if seq > 1:
                prev = route[seq - 2]
                dist = nx.shortest_path_length(G, prev, stop, weight="length")
                t_sec += dist / speed_mps
            hh, rem = divmod(int(t_sec), 3600)
            mm, ss = divmod(rem, 60)
            timestr = f"{hh:02d}:{mm:02d}:{ss:02d}"
            stop_times.append({
                "trip_id": trip_id,
                "arrival_time": timestr,
                "departure_time": timestr,
                "stop_id": str(stop),
                "stop_sequence": seq
            })

    pd.DataFrame(stop_times).to_csv(os.path.join(output_dir, "stop_times.txt"), index=False)
    pd.DataFrame(shapes).to_csv(os.path.join(output_dir, "shapes.txt"), index=False)

    # 4. frequencies.txt
    usage_scores = [score_usage_fn(r) for r in optimized_routes]
    sorted_scores = sorted(usage_scores)
    freqs = []
    import bisect
    for idx, usage in enumerate(usage_scores, start=1):
        rank = min(10, bisect.bisect_right(sorted_scores, usage) * 10 // len(sorted_scores) or 1)
        freqs.append({
            "trip_id": f"T{idx}",
            "start_time": start_time,
            "end_time": end_time,
            "headway_secs": headway_map[rank],
            "exact_times": 0
        })
    pd.DataFrame(freqs).to_csv(os.path.join(output_dir, "frequencies.txt"), index=False)

    # 5. trips.txt
    trips = [
        {
            "trip_id": f"T{i}",
            "route_id": f"R{i}",
            "service_id": service_id,
            "shape_id": f"S{i}"
        }
        for i in range(1, len(optimized_routes) + 1)
    ]
    pd.DataFrame(trips).to_csv(os.path.join(output_dir, "trips.txt"), index=False)

    # 6. routes.txt
    routes = [
        {
            "route_id": f"R{i}",
            "agency_id": agency_id,
            "route_short_name": str(i),
            "route_long_name": f"Route {i}",
            "route_type": 3
        }
        for i in range(1, len(optimized_routes) + 1)
    ]
    pd.DataFrame(routes).to_csv(os.path.join(output_dir, "routes.txt"), index=False)

    # 7. agency.txt
    agency = [{
        "agency_id": agency_id,
        "agency_name": agency_name,
        "agency_url": agency_url,
        "agency_timezone": agency_tz
    }]
    pd.DataFrame(agency).to_csv(os.path.join(output_dir, "agency.txt"), index=False)

    # 8. calendar.txt
    calendar = [{
        "service_id": service_id,
        "monday": 1, "tuesday": 1, "wednesday": 1,
        "thursday": 1, "friday": 1,
        "saturday": 1, "sunday": 1,
        "start_date": start_date,
        "end_date": end_date
    }]
    pd.DataFrame(calendar).to_csv(os.path.join(output_dir, "calendar.txt"), index=False)
