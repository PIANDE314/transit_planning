import os
from collections import defaultdict
from typing import Callable, Dict, List
import networkx as nx
import pandas as pd

# Headway mapping by decile rank 1–10
_DECILE_HEADWAYS = {
    1: 3600, 2: 2700, 3: 1800, 4: 1500, 5: 1200,
    6: 900, 7: 600, 8: 300, 9: 180, 10: 120
}

def write_gtfs(
    G: nx.Graph,
    optimized_routes: List[List[int]],
    score_usage_fn: Callable[[List[int]], float],
    output_dir: str = "gtfs_feed",
    service_id: str = "S1",
    agency_id: str = "A1",
    agency_name: str = "Simulated Transit Agency",
    agency_url: str = "http://example.com",
    agency_tz: str = "UTC",
    start_time: str = "16:00:00",
    end_time: str = "19:00:00",
    start_date: str = "20250101",
    end_date: str = "20251231",
    bus_speed_kmh: float = 20.0
) -> None:
    """
    Write a GTFS feed with stops, stop_times, shapes (with shape_dist_traveled),
    frequencies (by usage decile), trips, routes, agency, and calendar.

    Args:
        G: Graph of stops as nodes with 'lat','lon'; edges with 'length' in meters.
        optimized_routes: list of routes (list of node IDs).
        score_usage_fn: function mapping a route to a usage score.
        output_dir: directory to write GTFS files.
        service_id: GTFS service_id for calendar.
        agency_id: GTFS agency_id.
        agency_name: name of the agency.
        agency_url: agency website URL.
        agency_tz: agency timezone (IANA).
        start_time: service window start time.
        end_time: service window end time.
        start_date: calendar start date (YYYYMMDD).
        end_date: calendar end date (YYYYMMDD).
        bus_speed_kmh: assumed bus speed for travel time (km/h).
    """
    os.makedirs(output_dir, exist_ok=True)
    speed_mps = (bus_speed_kmh * 1000.0) / 3600.0  # m/s

    # 1) stops.txt
    stops = []
    for stop_id, data in G.nodes(data=True):
        stops.append({
            "stop_id": str(stop_id),
            "stop_name": data.get("stop_name", f"Stop {stop_id}"),
            "stop_lat": data["lat"],
            "stop_lon": data["lon"],
        })
    pd.DataFrame(stops).to_csv(os.path.join(output_dir, "stops.txt"), index=False)

    # 2) stop_times.txt & 3) shapes.txt (with shape_dist_traveled)
    stop_times = []
    shapes = []
    for rt_idx, route in enumerate(optimized_routes, start=1):
        trip_id = f"T{rt_idx}"
        shape_id = f"S{rt_idx}"

        # compute shape points and cumulative distance
        cum_dist = 0.0
        for seq, stop in enumerate(route, start=1):
            lat = G.nodes[stop]["lat"]
            lon = G.nodes[stop]["lon"]
            shapes.append({
                "shape_id": shape_id,
                "shape_pt_lat": lat,
                "shape_pt_lon": lon,
                "shape_pt_sequence": seq,
                "shape_dist_traveled": round(cum_dist, 1)
            })
            if seq < len(route):
                nxt = route[seq]  # next stop
                dist = nx.shortest_path_length(
                    G, stop, nxt, weight="length"
                )
                cum_dist += dist

        # compute stop_times
        # start at start_time
        h0, m0, s0 = map(int, start_time.split(":"))
        t_sec = h0 * 3600 + m0 * 60 + s0
        for seq, stop in enumerate(route, start=1):
            if seq > 1:
                prev = route[seq - 2]
                dist = nx.shortest_path_length(
                    G, prev, stop, weight="length"
                )
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
    pd.DataFrame(stop_times).to_csv(
        os.path.join(output_dir, "stop_times.txt"), index=False
    )
    pd.DataFrame(shapes).to_csv(
        os.path.join(output_dir, "shapes.txt"), index=False
    )

    # 4) frequencies.txt (map usage decile → headway_secs)
    usage_scores = [score_usage_fn(r) for r in optimized_routes]
    # assign decile rank 1–10
    sorted_scores = sorted(usage_scores)
    freqs = []
    import bisect
    for idx, usage in enumerate(usage_scores, start=1):
        # determine decile: find insertion index in 10 buckets
        rank = min(
            10,
            bisect.bisect_right(
                sorted_scores,
                usage
            ) * 10 // len(sorted_scores) or 1
        )
        headway = _DECILE_HEADWAYS[rank]
        freqs.append({
            "trip_id": f"T{idx}",
            "start_time": start_time,
            "end_time": end_time,
            "headway_secs": headway,
            "exact_times": 0
        })
    pd.DataFrame(freqs).to_csv(
        os.path.join(output_dir, "frequencies.txt"), index=False
    )

    # 5) trips.txt
    trips = []
    for idx, _ in enumerate(optimized_routes, start=1):
        trips.append({
            "route_id": f"R{idx}",
            "service_id": service_id,
            "trip_id": f"T{idx}",
            "shape_id": f"S{idx}"
        })
    pd.DataFrame(trips).to_csv(
        os.path.join(output_dir, "trips.txt"), index=False
    )

    # 6) routes.txt
    routes = []
    for idx, _ in enumerate(optimized_routes, start=1):
        routes.append({
            "route_id": f"R{idx}",
            "agency_id": agency_id,
            "route_short_name": str(idx),
            "route_long_name": f"Route {idx}",
            "route_type": 3  # bus
        })
    pd.DataFrame(routes).to_csv(
        os.path.join(output_dir, "routes.txt"), index=False
    )

    # 7) agency.txt
    agency = [{
        "agency_id": agency_id,
        "agency_name": agency_name,
        "agency_url": agency_url,
        "agency_timezone": agency_tz
    }]
    pd.DataFrame(agency).to_csv(
        os.path.join(output_dir, "agency.txt"), index=False
    )

    # 8) calendar.txt
    calendar = [{
        "service_id": service_id,
        "monday": 1, "tuesday": 1, "wednesday": 1,
        "thursday": 1, "friday": 1,
        "saturday": 1, "sunday": 1,
        "start_date": start_date,
        "end_date": end_date
    }]
    pd.DataFrame(calendar).to_csv(
        os.path.join(output_dir, "calendar.txt"), index=False
    )
