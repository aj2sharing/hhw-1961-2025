import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from event_pipeline_utils import (
    YEARS,
    finalize_parquet_writer,
    load_event_table,
    open_parquet_writer,
    open_z_precip,
    open_z_twb,
    to_pydate,
    write_grid_metrics,
    write_parquet_rows,
)


HHW_ANNUAL_DIR = Path(r"J:\HHW_events_1961-2025\annual")
EP_ANNUAL_DIR = Path(r"J:\EP_events_1961-2025\annual")
OUT_DIR = Path(r"J:\EP_HHW_1961-2025_vectors_v2")
MAX_LAG = 7
PARQUET_BATCH_ROWS = 100000

EVENT_COLUMNS = [
    "lon",
    "lat",
    "ep_start_date",
    "ep_end_date",
    "ep_duration_days",
    "ep_peak_date",
    "hhw_start_date",
    "hhw_end_date",
    "hhw_duration_days",
    "hhw_peak_date",
    "lag_days",
    "distance",
    "speed",
    "heating",
    "angle_rad",
    "drying",
    "z_twb_at_ep_peak",
    "z_precip_ep_peak",
    "z_twb_hhw_peak",
    "z_precip_at_hhw_peak",
]

EVENT_SCHEMA = pa.schema([
    ("lon", pa.float32()),
    ("lat", pa.float32()),
    ("ep_start_date", pa.date32()),
    ("ep_end_date", pa.date32()),
    ("ep_duration_days", pa.int32()),
    ("ep_peak_date", pa.date32()),
    ("hhw_start_date", pa.date32()),
    ("hhw_end_date", pa.date32()),
    ("hhw_duration_days", pa.int32()),
    ("hhw_peak_date", pa.date32()),
    ("lag_days", pa.int32()),
    ("distance", pa.float32()),
    ("speed", pa.float32()),
    ("heating", pa.float32()),
    ("angle_rad", pa.float32()),
    ("drying", pa.float32()),
    ("z_twb_at_ep_peak", pa.float32()),
    ("z_precip_ep_peak", pa.float32()),
    ("z_twb_hhw_peak", pa.float32()),
    ("z_precip_at_hhw_peak", pa.float32()),
])


def load_grid():
    ds, da = open_z_twb(YEARS[0])
    try:
        lon = da["longitude"].values
        lat = da["latitude"].values
    finally:
        ds.close()
    return lon, lat


def build_indexer(values):
    start = float(values[0])
    delta = float(values[1] - values[0])
    return start, delta


def coord_to_index(value, start, delta, n):
    idx = int(round((float(value) - start) / delta))
    return max(0, min(n - 1, idx))


def load_year_array(kind, year, cache, limit=6):
    key = (kind, year)
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    if kind == "twb":
        ds, da = open_z_twb(year)
    else:
        ds, da = open_z_precip(year)
    try:
        time0 = da["time"].values[0].astype("datetime64[D]")
        arr = da.values.astype(np.float32)
    finally:
        ds.close()
    cache[key] = (time0, arr)
    while len(cache) > limit:
        cache.popitem(last=False)
    return cache[key]


def sample_z(kind, date_value, j, i, cache):
    year = int(str(date_value)[:4])
    time0, arr = load_year_array(kind, year, cache)
    idx = int((date_value - time0).astype(int))
    return float(arr[idx, j, i])


def to_day_values(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[D]")


def build_ep_groups(ep: pd.DataFrame):
    groups = {}
    for key, grp in ep.groupby(["lon", "lat"], sort=False):
        peak_dates = to_day_values(grp["peak_date"])
        order = np.argsort(peak_dates)
        groups[key] = {
            "start_date": to_day_values(grp["start_date"])[order],
            "end_date": to_day_values(grp["end_date"])[order],
            "duration_days": grp["duration_days"].to_numpy(dtype=np.int32)[order],
            "peak_date": peak_dates[order],
            "peak_z_precip": grp["peak_z_precip"].to_numpy(dtype=np.float32)[order],
        }
    return groups


def build_hhw_groups(hhw: pd.DataFrame):
    groups = {}
    for key, grp in hhw.groupby(["lon", "lat"], sort=False):
        groups[key] = {
            "start_date": to_day_values(grp["start_date"]),
            "end_date": to_day_values(grp["end_date"]),
            "duration_days": grp["duration_days"].to_numpy(dtype=np.int32),
            "peak_date": to_day_values(grp["peak_date"]),
            "peak_z_twb": grp["peak_z_twb"].to_numpy(dtype=np.float32),
        }
    return groups


def select_best_ep_candidate(ep_group, hhw_peak: np.datetime64, hhw_start: np.datetime64):
    peak_dates = ep_group["peak_date"]
    left = np.searchsorted(peak_dates, hhw_peak - np.timedelta64(MAX_LAG, "D"), side="left")
    right = np.searchsorted(peak_dates, hhw_peak, side="right")
    if right <= left:
        return None

    end_dates = ep_group["end_date"][left:right]
    valid = end_dates < hhw_start
    if not np.any(valid):
        return None

    rel_idx = np.flatnonzero(valid)
    zvals = ep_group["peak_z_precip"][left:right][valid]
    best_z = np.nanmax(zvals)
    top = rel_idx[zvals == best_z]
    best_rel = int(top[-1])
    idx = left + best_rel
    return {
        "start_date": ep_group["start_date"][idx],
        "end_date": ep_group["end_date"][idx],
        "duration_days": int(ep_group["duration_days"][idx]),
        "peak_date": ep_group["peak_date"][idx],
        "peak_z_precip": float(ep_group["peak_z_precip"][idx]),
    }


def identify(overwrite=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    event_parquet = OUT_DIR / "EP_HHW_Event_List_1961-2025_vectors.parquet"
    grid_nc = OUT_DIR / "EP_HHW_Gridded_Metrics_1961-2025_vectors.nc"
    if event_parquet.exists() and not overwrite:
        raise FileExistsError(event_parquet)
    writer, tmp_parquet = open_parquet_writer(event_parquet, EVENT_SCHEMA)

    lon_vals, lat_vals = load_grid()
    lon0, dlon = build_indexer(lon_vals)
    lat0, dlat = build_indexer(lat_vals)

    z_cache = OrderedDict()
    counts = np.zeros((lon_vals.size, lat_vals.size), dtype=np.int32)
    sums = [np.zeros((lon_vals.size, lat_vals.size), dtype=np.float64) for _ in range(6)]
    total_events = 0

    for year in YEARS:
        hhw = load_event_table(HHW_ANNUAL_DIR / f"HHW_Events_{year}.parquet")
        if hhw.empty:
            continue
        ep_frames = [load_event_table(EP_ANNUAL_DIR / f"EP_Events_{year}.parquet")]
        if year > YEARS[0]:
            ep_frames.append(load_event_table(EP_ANNUAL_DIR / f"EP_Events_{year - 1}.parquet"))
        ep = pd.concat([df for df in ep_frames if not df.empty], ignore_index=True) if any(not df.empty for df in ep_frames) else pd.DataFrame()
        if ep.empty:
            continue

        ep_groups = build_ep_groups(ep)
        hhw_groups = build_hhw_groups(hhw)
        year_events = 0
        buffer_rows = []

        for key, hhw_group in hhw_groups.items():
            ep_group = ep_groups.get(key)
            if ep_group is None:
                continue

            lon, lat = key
            i = coord_to_index(lon, lon0, dlon, lon_vals.size)
            j = coord_to_index(lat, lat0, dlat, lat_vals.size)
            for idx_evt in range(hhw_group["peak_date"].size):
                hhw_peak = hhw_group["peak_date"][idx_evt]
                hhw_start = hhw_group["start_date"][idx_evt]
                best = select_best_ep_candidate(ep_group, hhw_peak, hhw_start)
                if best is None:
                    continue

                ep_peak = best["peak_date"]
                lag = int((hhw_peak - ep_peak).astype(int))
                if lag > MAX_LAG:
                    continue

                z_precip_ep = best["peak_z_precip"]
                z_twb_hhw = float(hhw_group["peak_z_twb"][idx_evt])
                z_twb_ep = sample_z("twb", ep_peak, j, i, z_cache)
                z_precip_hhw = sample_z("precip", hhw_peak, j, i, z_cache)
                if not all(np.isfinite(v) for v in [z_precip_ep, z_twb_hhw, z_twb_ep, z_precip_hhw]):
                    continue

                heating = z_twb_hhw - z_twb_ep
                drying = z_precip_ep - z_precip_hhw
                distance = float(np.sqrt(heating * heating + drying * drying))
                speed = distance / (lag + 1.0)
                angle = float(np.arctan2(drying, heating))

                buffer_rows.append([
                    float(lon),
                    float(lat),
                    to_pydate(best["start_date"]),
                    to_pydate(best["end_date"]),
                    best["duration_days"],
                    to_pydate(best["peak_date"]),
                    to_pydate(hhw_group["start_date"][idx_evt]),
                    to_pydate(hhw_group["end_date"][idx_evt]),
                    int(hhw_group["duration_days"][idx_evt]),
                    to_pydate(hhw_group["peak_date"][idx_evt]),
                    lag,
                    distance,
                    speed,
                    heating,
                    angle,
                    drying,
                    z_twb_ep,
                    z_precip_ep,
                    z_twb_hhw,
                    z_precip_hhw,
                ])
                vals = [lag, distance, speed, heating, angle, drying]
                for k, val in enumerate(vals):
                    sums[k][i, j] += val
                counts[i, j] += 1
                year_events += 1

            if len(buffer_rows) >= PARQUET_BATCH_ROWS:
                write_parquet_rows(writer, EVENT_COLUMNS, buffer_rows, EVENT_SCHEMA)
                buffer_rows.clear()

        if buffer_rows:
            write_parquet_rows(writer, EVENT_COLUMNS, buffer_rows, EVENT_SCHEMA)

        total_events += year_events
        print(f"[pair-ep-hhw] {year} events: {year_events}", flush=True)

    finalize_parquet_writer(writer, tmp_parquet, event_parquet)

    write_grid_metrics(
        grid_nc,
        lon_vals,
        lat_vals,
        counts,
        sums,
        ["lag", "distance", "speed", "heating", "angle_rad", "drying"],
        "EP-HHW vector metrics, 1961-2025",
        "Paired from independent HHW and EP annual event tables; EP peak precedes HHW peak by 0..7 days and the two events do not overlap.",
    )
    print(f"[pair-ep-hhw] total events: {total_events}", flush=True)
    print(f"[pair-ep-hhw] saved {event_parquet}", flush=True)
    print(f"[pair-ep-hhw] saved {grid_nc}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    identify(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
