import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from scipy.stats import linregress, pearsonr

"""
§ 3.5 Accessibility Comparisons — compare extracted vs. operational GTFS,
compute bias against wealth and phone-data representativeness.
"""

# ----------------------------------------------------------------------
# 1. Load GTFS travel‐time matrix
# ----------------------------------------------------------------------
def load_gtfs_times(
    gtfs_dir: str,
    service_date: str,
    start_time: str,
    window_minutes: int = 60
) -> pd.DataFrame:
    """
    For each departure minute in a service‐window, compute cumulative travel times
    (in minutes) to each stop via stop_times.txt.
    Returns a long DataFrame with columns ['minute','stop_id','travel_min'].
    """
    # Load stop_times and cumulative travel time per trip
    st = pd.read_csv(
        os.path.join(gtfs_dir, "stop_times.txt"),
        parse_dates=["arrival_time", "departure_time"]
    )
    st["cum_min"] = (
        (st["arrival_time"] - st.groupby("trip_id")["arrival_time"].transform("first"))
        .dt.total_seconds() / 60.0
    )
    # Build lookup by departure_time
    h0, m0, s0 = map(int, start_time.split(":"))
    t0 = datetime.strptime(service_date + start_time, "%Y%m%d%H:%M:%S")
    records = []
    for minute in range(window_minutes):
        dep = (t0 + timedelta(minutes=minute)).time()
        df = st[st["departure_time"].dt.time == dep]
        if not df.empty:
            df2 = df[["stop_id", "cum_min"]].copy()
            df2["minute"] = minute
            records.append(df2.rename(columns={"cum_min": "travel_min"}))
    return pd.concat(records, ignore_index=True)


# ----------------------------------------------------------------------
# 2. Compute per‐zone accessibility fraction
# ----------------------------------------------------------------------
def compute_accessibility(
    times: pd.DataFrame,
    zone_map: pd.DataFrame,
    max_travel_min: float = 30.0
) -> pd.Series:
    """
    Given `times` with ['minute','stop_id','travel_min'] and a zone_map
    with ['zone_id','stop_id'], returns Series zone_id→fraction of minutes
    where any stop in zone reachable within max_travel_min.
    """
    df = times.merge(zone_map, on="stop_id", how="inner")
    df["accessible"] = df["travel_min"] <= max_travel_min
    agg = (
        df.groupby(["zone_id", "minute"])["accessible"]
          .any()
          .reset_index()
    )
    frac = agg.groupby("zone_id")["accessible"].mean()
    return frac.rename("accessibility")


# ----------------------------------------------------------------------
# 3. Compute comparison statistics between two accessibility Series
# ----------------------------------------------------------------------
def compute_comparison_stats(
    ext: pd.Series,
    op: pd.Series
) -> Dict[str, float]:
    """
    Given extracted vs. operational accessibility (aligned indices),
    returns slope, intercept, r_value, p_value, mse.
    """
    df = pd.concat([op, ext], axis=1, keys=["operational","extracted"]).dropna()
    x = df["operational"].values
    y = df["extracted"].values

    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    mse = np.mean((y - x) ** 2)
    return {
        "slope":     slope,
        "intercept": intercept,
        "r_value":   r_value,
        "p_value":   p_value,
        "mse":       mse
    }


# ----------------------------------------------------------------------
# 4. Compute bias correlations vs. wealth & phone‐density
# ----------------------------------------------------------------------
def compute_bias_analysis(
    mse_by_zone: pd.Series,
    wealth: pd.Series,
    phone_density: pd.Series
) -> Dict[str, Tuple[float,float]]:
    """
    Correlate per-zone MSE with wealth index and phone-data density.
    Returns dict with keys 'wealth','phone', values (r,p).
    """
    # align indices
    df = pd.concat([mse_by_zone, wealth, phone_density], axis=1).dropna()
    results = {}
    for name, series in [("wealth", df.iloc[:,1]), ("phone", df.iloc[:,2])]:
        r, p = pearsonr(df[name], df["mse"])
        results[name] = (r, p)
    return results


# ----------------------------------------------------------------------
# 5. Main comparison function
# ----------------------------------------------------------------------
def compare_accessibility(
    extracted_gtfs: str,
    operational_gtfs: str,
    zone_map: pd.DataFrame,
    wealth: pd.Series,
    phone_density: pd.Series,
    service_date: str = "20250101",
    start_time: str = "16:00:00"
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float,float]]]:
    """
    Full pipeline:
      1) Load travel times for each feed
      2) Compute accessibility fractions per zone
      3) Compare stats (slope, r, mse)
      4) Compute bias correlations vs. wealth & phone_density

    Args:
        extracted_gtfs: dir of generated GTFS feed.
        operational_gtfs: dir of real GTFS feed.
        zone_map: DataFrame with ['zone_id','stop_id'] linking stops→zones.
        wealth: Series mapping zone_id→wealth index.
        phone_density: Series mapping zone_id→phone‐ping density.
        service_date: YYYYMMDD of service.
        start_time: HH:MM:SS window start.

    Returns:
        df_acc: DataFrame with columns ['access_extracted','access_operational'].
        comp_stats: dict of slope, intercept, r_value, p_value, mse.
        bias_stats: dict of ('wealth':(r,p),'phone':(r,p)).
    """
    # 1. Travel‐time matrices
    times_ext = load_gtfs_times(extracted_gtfs, service_date, start_time)
    times_op  = load_gtfs_times(operational_gtfs, service_date, start_time)

    # 2. Accessibility fractions
    acc_ext = compute_accessibility(times_ext, zone_map)
    acc_op  = compute_accessibility(times_op,  zone_map)

    # 3. Comparison stats
    comp_stats = compute_comparison_stats(acc_ext, acc_op)

    # 4. Bias analysis
    # use per‐zone MSE = (ext - op)^2
    mse_zone = (acc_ext - acc_op).pow(2).rename("mse")
    bias_stats = compute_bias_analysis(mse_zone, wealth, phone_density)

    # 5. Combined DataFrame
    df_acc = pd.concat(
        [acc_ext.rename("access_extracted"), acc_op.rename("access_operational")],
        axis=1
    )

    return df_acc, comp_stats, bias_stats
