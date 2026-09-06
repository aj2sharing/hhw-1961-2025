import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


WETBULB_DIR = Path(r"J:\wetbulb_global")
CLIM_OUT_DIR = WETBULB_DIR / "climatology_1961-1990"
P95_OUT_DIR = WETBULB_DIR / "climatology_1961-1990"
P95_FILE = P95_OUT_DIR / "climatology_P95_1961-1990.nc"
MASK_OUT_DIR = Path(r"J:\twb_23\heatwave_masks_1961-2025_python")

YEARS = list(range(1961, 2026))
CLIM_BASELINE_YEARS = list(range(1961, 1991))
P95_BASELINE_YEARS = list(range(1961, 1991))
MIN_TWB_THRESHOLD_C = 23.0
MIN_DURATION = 3
PAD_DAYS = MIN_DURATION - 1
CHUNK_DAYS = 15
P95_LAT_CHUNK = 90


def wetbulb_file(year):
    return WETBULB_DIR / f"wetbulb_daily_{year}.nc"


def normalize_twb(da):
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "lon" in da.dims:
        da = da.rename({"lon": "longitude"})
    if "lat" in da.dims:
        da = da.rename({"lat": "latitude"})
    return da.transpose("time", "latitude", "longitude")


def drop_feb29(da):
    time = da["time"]
    keep = ~((time.dt.month == 2) & (time.dt.day == 29))
    return da.sel(time=keep)


def open_twb(year):
    path = wetbulb_file(year)
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    return ds, normalize_twb(ds["twb"])


def write_climatology(out_file, var_name, data_lon_lat_day, lon, lat, description):
    if out_file.exists():
        out_file.unlink()
    with Dataset(out_file, "w", format="NETCDF4") as nc:
        nc.createDimension("longitude", lon.size)
        nc.createDimension("latitude", lat.size)
        nc.createDimension("dayofyear", data_lon_lat_day.shape[2])

        lon_var = nc.createVariable("longitude", "f8", ("longitude",))
        lat_var = nc.createVariable("latitude", "f8", ("latitude",))
        day_var = nc.createVariable("dayofyear", "i2", ("dayofyear",))
        data_var = nc.createVariable(
            var_name,
            "f4",
            ("longitude", "latitude", "dayofyear"),
            zlib=True,
            complevel=3,
            shuffle=True,
            fill_value=np.float32(np.nan),
        )

        lon_var[:] = lon
        lat_var[:] = lat
        day_var[:] = np.arange(1, data_lon_lat_day.shape[2] + 1, dtype=np.int16)
        data_var[:, :, :] = data_lon_lat_day.astype(np.float32)

        lon_var.units = "degrees_east"
        lat_var.units = "degrees_north"
        day_var.description = "No-leap day of year; Feb 29 removed"
        data_var.description = description
        data_var.units = "C"
        nc.title = "Daily wet-bulb climatology, 1961-1990"
        nc.years = "1961-1990"
        nc.note = "Leap day removed before daily climatology calculation."


def compute_daily_climatology(overwrite=False):
    CLIM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_file = CLIM_OUT_DIR / "clim_mean_twb_1961-1990.nc"
    std_file = CLIM_OUT_DIR / "clim_std_twb_1961-1990.nc"
    if mean_file.exists() and std_file.exists() and not overwrite:
        print(f"climatology exists, skip: {CLIM_OUT_DIR}", flush=True)
        return

    with open_twb(YEARS[0])[0] as ds0:
        da0 = normalize_twb(ds0["twb"])
        lat = da0["latitude"].values
        lon = da0["longitude"].values
        nlat = lat.size
        nlon = lon.size

    sum_x = np.zeros((365, nlat, nlon), dtype=np.float32)
    sum_sq = np.zeros((365, nlat, nlon), dtype=np.float32)
    count = np.zeros(365, dtype=np.float32)

    for year in CLIM_BASELINE_YEARS:
        tic = time.time()
        ds, da = open_twb(year)
        with ds:
            da = drop_feb29(da)
            nt = da.sizes["time"]
            if nt != 365:
                raise ValueError(f"{year} has {nt} no-leap days, expected 365")
            for t0 in range(0, 365, CHUNK_DAYS):
                t1 = min(t0 + CHUNK_DAYS, 365)
                arr = da.isel(time=slice(t0, t1)).values.astype(np.float32)
                sum_x[t0:t1] += arr
                sum_sq[t0:t1] += arr * arr
            count += 1
        print(f"[clim] {year} done in {time.time() - tic:.1f}s", flush=True)

    n = count[:, None, None]
    mean_t_lat_lon = sum_x / n
    var_t_lat_lon = np.maximum(sum_sq / n - mean_t_lat_lon * mean_t_lat_lon, 0.0)
    std_t_lat_lon = np.sqrt(var_t_lat_lon).astype(np.float32)

    mean_lon_lat_day = np.transpose(mean_t_lat_lon, (2, 1, 0))
    std_lon_lat_day = np.transpose(std_t_lat_lon, (2, 1, 0))
    write_climatology(mean_file, "clim_mean", mean_lon_lat_day, lon, lat, "Daily mean wet-bulb temperature")
    write_climatology(std_file, "clim_std", std_lon_lat_day, lon, lat, "Daily standard deviation of wet-bulb temperature")
    print(f"[clim] saved {mean_file}", flush=True)
    print(f"[clim] saved {std_file}", flush=True)


def write_p95(out_file, p95_lat_lon, lon, lat):
    if out_file.exists():
        out_file.unlink()
    with Dataset(out_file, "w", format="NETCDF4") as nc:
        nc.createDimension("latitude", lat.size)
        nc.createDimension("longitude", lon.size)

        lat_var = nc.createVariable("latitude", "f8", ("latitude",))
        lon_var = nc.createVariable("longitude", "f8", ("longitude",))
        q_var = nc.createVariable("quantile", "f8")
        p95_var = nc.createVariable(
            "twb_p95",
            "f4",
            ("latitude", "longitude"),
            zlib=True,
            complevel=3,
            shuffle=True,
            fill_value=np.float32(np.nan),
        )

        lat_var[:] = lat
        lon_var[:] = lon
        q_var[...] = np.float64(0.95)
        p95_var[:, :] = p95_lat_lon.astype(np.float32)

        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        p95_var.units = "C"
        p95_var.description = "Grid-cell 95th percentile of daily wet-bulb temperature"
        nc.title = "Wet-bulb 95th percentile climatology"
        nc.base_period = "1961-1990"
        nc.note = "Computed from all daily Twb values in 1961-1990; Feb 29 removed."


def compute_p95_baseline(overwrite=False):
    P95_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if P95_FILE.exists() and not overwrite:
        print(f"[p95] exists, skip: {P95_FILE}", flush=True)
        return

    handles = []
    arrays = []
    try:
        for year in P95_BASELINE_YEARS:
            ds, da = open_twb(year)
            da = drop_feb29(da)
            if da.sizes["time"] != 365:
                raise ValueError(f"{year} has {da.sizes['time']} no-leap days, expected 365")
            handles.append(ds)
            arrays.append(da)

        lat = arrays[0]["latitude"].values
        lon = arrays[0]["longitude"].values
        nlat = lat.size
        nlon = lon.size
        p95 = np.empty((nlat, nlon), dtype=np.float32)

        for j0 in range(0, nlat, P95_LAT_CHUNK):
            j1 = min(j0 + P95_LAT_CHUNK, nlat)
            tic = time.time()
            pieces = [
                da.isel(latitude=slice(j0, j1)).values.astype(np.float32)
                for da in arrays
            ]
            block = np.concatenate(pieces, axis=0)
            p95[j0:j1, :] = np.nanpercentile(block, 95, axis=0).astype(np.float32)
            print(f"[p95] lat rows {j0 + 1:03d}-{j1:03d}/{nlat:03d} done in {time.time() - tic:.1f}s", flush=True)

        write_p95(P95_FILE, p95, lon, lat)
        print(f"[p95] saved {P95_FILE}", flush=True)
    finally:
        for ds in handles:
            ds.close()


def load_threshold():
    if not P95_FILE.exists():
        compute_p95_baseline(overwrite=False)
    with xr.open_dataset(P95_FILE) as ds:
        p95 = ds["twb_p95"].load()
    if "lon" in p95.dims:
        p95 = p95.rename({"lon": "longitude"})
    if "lat" in p95.dims:
        p95 = p95.rename({"lat": "latitude"})
    p95 = p95.transpose("latitude", "longitude")
    return np.maximum(p95.values.astype(np.float32), np.float32(MIN_TWB_THRESHOLD_C)), p95["longitude"].values, p95["latitude"].values


def year_time_values(da):
    values = da["time"].values
    if np.issubdtype(values.dtype, np.datetime64):
        return values
    year_match = re.search(r"(\d{4})", str(da.encoding.get("source", "")))
    year = int(year_match.group(1)) if year_match else None
    if year is None:
        return np.arange(1, da.sizes["time"] + 1, dtype=np.int32)
    return np.array([np.datetime64(f"{year}-01-01") + np.timedelta64(i, "D") for i in range(da.sizes["time"])])


def read_exceed_slice(year, start=None, stop=None, threshold=None):
    ds, da = open_twb(year)
    with ds:
        if start is not None or stop is not None:
            da = da.isel(time=slice(start, stop))
        arr = da.values.astype(np.float32)
        return arr > threshold


def write_mask(out_file, mask_time_lat_lon, lon, lat, time_values, year):
    tmp_file = out_file.with_suffix(".tmp.nc")
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()

    nt, nlat, nlon = mask_time_lat_lon.shape
    with Dataset(tmp_file, "w", format="NETCDF4") as nc:
        nc.createDimension("longitude", nlon)
        nc.createDimension("latitude", nlat)
        nc.createDimension("time", nt)

        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        time_var = nc.createVariable("time", "i4", ("time",))
        mask_var = nc.createVariable(
            "heatwave_mask",
            "i1",
            ("longitude", "latitude", "time"),
            zlib=True,
            complevel=5,
            shuffle=True,
        )

        lon_var[:] = lon.astype(np.float32)
        lat_var[:] = lat.astype(np.float32)
        time_var[:] = np.arange(1, nt + 1, dtype=np.int32)
        mask_var[:, :, :] = np.transpose(mask_time_lat_lon.astype(np.int8), (2, 1, 0))

        lon_var.units = "degrees_east"
        lat_var.units = "degrees_north"
        time_var.description = f"Day index within {year}"
        mask_var.description = "1 = day belongs to HHW event (>=3 consecutive days, Twb > max(P95_1961-1990, 23C))"
        nc.title = f"Wet-bulb humid heatwave mask {year}"
        nc.heatwave_definition = "At least 3 consecutive days with daily Twb > max(P95_1961-1990, 23C)"
        nc.base_period_p95 = "1961-1990"
        nc.data_years = "1961-2025"

    os.replace(tmp_file, out_file)


def compute_hhw_masks(overwrite=False):
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold, lon, lat = load_threshold()

    for year in YEARS:
        out_file = MASK_OUT_DIR / f"heatwave_mask_{year}.nc"
        if out_file.exists() and not overwrite:
            print(f"[mask] {year} exists, skip", flush=True)
            continue

        tic = time.time()
        ds, da = open_twb(year)
        with ds:
            nt = da.sizes["time"]
            time_values = year_time_values(da)
            cur = da.values.astype(np.float32) > threshold

        if year > YEARS[0]:
            prev_ds, prev_da = open_twb(year - 1)
            with prev_ds:
                prev = (prev_da.isel(time=slice(-PAD_DAYS, None)).values.astype(np.float32) > threshold)
        else:
            prev = np.zeros((PAD_DAYS, lat.size, lon.size), dtype=bool)

        if year < YEARS[-1]:
            next_ds, next_da = open_twb(year + 1)
            with next_ds:
                next_ = (next_da.isel(time=slice(0, PAD_DAYS)).values.astype(np.float32) > threshold)
        else:
            next_ = np.zeros((PAD_DAYS, lat.size, lon.size), dtype=bool)

        exceed_ext = np.concatenate([prev, cur, next_], axis=0)
        triplets = exceed_ext[:-2] & exceed_ext[1:-1] & exceed_ext[2:]
        hw_ext = np.zeros_like(exceed_ext, dtype=bool)
        hw_ext[:-2] |= triplets
        hw_ext[1:-1] |= triplets
        hw_ext[2:] |= triplets
        mask_year = hw_ext[PAD_DAYS:PAD_DAYS + nt]

        write_mask(out_file, mask_year, lon, lat, time_values, year)
        print(f"[mask] {year} saved in {time.time() - tic:.1f}s: {out_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "clim", "p95", "mask"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.only in ("all", "clim"):
        compute_daily_climatology(overwrite=args.overwrite)
    if args.only in ("all", "p95"):
        compute_p95_baseline(overwrite=args.overwrite)
    if args.only in ("all", "mask"):
        compute_hhw_masks(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
