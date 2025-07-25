import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime, timedelta
from scipy.stats import linregress, pearsonr
from transit_planner.config import Config

cfg = Config()

def load_gtfs_times(gtfs_dir: str, service_date: str, start_time: str, window_minutes: int) -> pd.DataFrame:
    st = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    st["arrival_time"] = pd.to_timedelta(st["arrival_time"] + ":00")
    st["cum_min"] = (st["arrival_time"] - st.groupby("trip_id")["arrival_time"].transform("first")).dt.total_seconds() / 60.0

    h0, m0, s0 = map(int, start_time.split(":"))
    t0 = datetime.strptime(f"{service_date}{start_time}", "%Y%m%d%H:%M:%S")
    records = []

    for minute in range(window_minutes):
        dep = (t0 + timedelta(minutes=minute)).time()
        match = st[st["arrival_time"].dt.seconds == dep.hour * 3600 + dep.minute * 60 + dep.second]
        if not match.empty:
            df2 = match[["stop_id", "cum_min"]].copy()
            df2["minute"] = minute
            df2.rename(columns={"cum_min": "travel_min"}, inplace=True)
            records.append(df2)
    return pd.concat(records, ignore_index=True)

def compute_accessibility(times: pd.DataFrame, zone_map: pd.DataFrame, max_travel_min: float) -> pd.Series:
    df = times.merge(zone_map, on="stop_id", how="inner")
    df["accessible"] = df["travel_min"] <= max_travel_min
    agg = df.groupby(["zone_id", "minute"])["accessible"].any().reset_index()
    return agg.groupby("zone_id")["accessible"].mean().rename("accessibility")

def compute_comparison_stats(ext: pd.Series, op: pd.Series) -> Dict[str, float]:
    df = pd.concat([op, ext], axis=1, keys=["operational", "extracted"]).dropna()
    x, y = df["operational"].values, df["extracted"].values
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    mse = np.mean((y - x) ** 2)
    return {"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "mse": mse}

def compute_bias_analysis(mse_zone: pd.Series, wealth: pd.Series, phone_density: pd.Series) -> Dict[str, Tuple[float, float]]:
    df = pd.concat([mse_zone.rename("mse"), wealth, phone_density], axis=1).dropna()
    return {
        "wealth": pearsonr(df.iloc[:, 1], df["mse"]),
        "phone":  pearsonr(df.iloc[:, 2], df["mse"])
    }

def compare_accessibility(
    extracted_gtfs: str,
    operational_gtfs: str,
    zone_map: pd.DataFrame,
    wealth: pd.Series,
    phone_density: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]]]:

    service_date = cfg.get("start_date")
    start_time = cfg.get("start_time")
    window_minutes = cfg.get("access_window_min")
    max_travel_min = cfg.get("max_travel_min")

    times_ext = load_gtfs_times(extracted_gtfs, service_date, start_time, window_minutes)
    times_op = load_gtfs_times(operational_gtfs, service_date, start_time, window_minutes)

    acc_ext = compute_accessibility(times_ext, zone_map, max_travel_min)
    acc_op = compute_accessibility(times_op, zone_map, max_travel_min)

    comp_stats = compute_comparison_stats(acc_ext, acc_op)
    mse_zone = (acc_ext - acc_op).pow(2).rename("mse")
    bias_stats = compute_bias_analysis(mse_zone, wealth, phone_density)

    df_acc = pd.concat([
        acc_ext.rename("access_extracted"),
        acc_op.rename("access_operational")
    ], axis=1)

    return df_acc, comp_stats, bias_stats
