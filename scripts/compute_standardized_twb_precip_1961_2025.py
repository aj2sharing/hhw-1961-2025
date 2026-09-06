import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


YEARS = list(range(1961, 2026))
CHUNK_DAYS = 60
REF_PERIOD_LABEL = "1961-1990"

TWB_DIR = Path(r"J:\wetbulb_global")
TWB_CLIM_DIR = TWB_DIR / f"climatology_{REF_PERIOD_LABEL}"
TWB_OUT_DIR = TWB_DIR / "standardized_1961-2025"

PRECIP_DIR = Path(r"K:\era5_daily_pre")
PRECIP_CLIM_DIR = PRECIP_DIR / f"climatology_{REF_PERIOD_LABEL}"
PRECIP_OUT_DIR = PRECIP_DIR / "standardized_1961-2025"
PRECIP_M_TO_MM = np.float32(1000.0)
PRECIP_STD_FLOOR = np.float32(0.1)


def normalize_lon_lat_time(da):
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "lon" in da.dims:
        da = da.rename({"lon": "longitude"})
    if "lat" in da.dims:
        da = da.rename({"lat": "latitude"})
    lon = da["longitude"].values
    lon0360 = np.mod(lon, 360.0)
    if np.any(np.diff(lon0360) < 0):
        order = np.argsort(lon0360)
        da = da.isel(longitude=order)
        lon0360 = lon0360[order]
    da = da.assign_coords(longitude=lon0360)
    return da.transpose("time", "latitude", "longitude")


def noleap_indices_and_weights(time_values):
    dates = time_values.astype("datetime64[D]")
    indices = []
    weights = []
    for d in dates:
        year = int(str(d)[:4])
        month = int(str(d)[5:7])
        day = int(str(d)[8:10])
        day_of_year = int((d - np.datetime64(f"{year}-01-01")).astype(int)) + 1
        if month == 2 and day == 29:
            indices.append((58, 59))
            weights.append((0.5, 0.5))
        else:
            idx = day_of_year - 1
            is_leap = (np.datetime64(f"{year}-03-01") - np.datetime64(f"{year}-02-28")).astype(int) == 2
            if is_leap and (month > 2):
                idx -= 1
            indices.append((idx,))
            weights.append((1.0,))
    return indices, weights


def load_clim_arrays(mean_file, std_file):
    with xr.open_dataset(mean_file) as mean_ds, xr.open_dataset(std_file) as std_ds:
        mean = mean_ds["clim_mean"].transpose("dayofyear", "latitude", "longitude").values.astype(np.float32)
        std = std_ds["clim_std"].transpose("dayofyear", "latitude", "longitude").values.astype(np.float32)
    return mean, std


def read_clim_chunk(mean_arr, std_arr, indices, weights):
    mean_parts = []
    std_parts = []
    for idxs, ws in zip(indices, weights):
        if len(idxs) == 1:
            mean_parts.append(mean_arr[idxs[0]])
            std_parts.append(std_arr[idxs[0]])
        else:
            m = sum(np.float32(w) * mean_arr[i] for i, w in zip(idxs, ws))
            s = sum(np.float32(w) * std_arr[i] for i, w in zip(idxs, ws))
            mean_parts.append(m)
            std_parts.append(s)
    return np.stack(mean_parts, axis=0).astype(np.float32), np.stack(std_parts, axis=0).astype(np.float32)


def create_z_file(out_file, var_name, lon, lat, time_values, attrs):
    tmp_file = out_file.with_suffix(".tmp.nc")
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()
    nc = Dataset(tmp_file, "w", format="NETCDF4")
    nc.createDimension("longitude", lon.size)
    nc.createDimension("latitude", lat.size)
    nc.createDimension("time", len(time_values))
    lon_var = nc.createVariable("longitude", "f8", ("longitude",))
    lat_var = nc.createVariable("latitude", "f8", ("latitude",))
    time_var = nc.createVariable("time", "f8", ("time",))
    z_var = nc.createVariable(
        var_name,
        "f4",
        ("longitude", "latitude", "time"),
        zlib=True,
        complevel=3,
        shuffle=True,
        fill_value=np.float32(np.nan),
    )
    lon_var[:] = lon
    lat_var[:] = lat
    if np.issubdtype(time_values.dtype, np.datetime64):
        days = (time_values.astype("datetime64[ns]") - np.datetime64("1970-01-01T00:00:00", "ns")) / np.timedelta64(1, "D")
        time_var[:] = days.astype(np.float64)
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "proleptic_gregorian"
    else:
        time_var[:] = np.arange(1, len(time_values) + 1)
        time_var.units = "day index"
    lon_var.units = "degrees_east"
    lat_var.units = "degrees_north"
    for k, v in attrs.items():
        setattr(z_var, k, v)
    return nc, z_var, tmp_file


def standardize_twb(overwrite=False):
    TWB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_arr, std_arr = load_clim_arrays(
        TWB_CLIM_DIR / f"clim_mean_twb_{REF_PERIOD_LABEL}.nc",
        TWB_CLIM_DIR / f"clim_std_twb_{REF_PERIOD_LABEL}.nc",
    )
    for year in YEARS:
        out_file = TWB_OUT_DIR / f"z_twb_{year}.nc"
        if out_file.exists() and not overwrite:
            print(f"[z_twb] {year} exists, skip", flush=True)
            continue
        with xr.open_dataset(TWB_DIR / f"wetbulb_daily_{year}.nc") as ds:
            da = normalize_lon_lat_time(ds["twb"])
            lon = da["longitude"].values
            lat = da["latitude"].values
            time_values = da["time"].values
            day_index, day_weights = noleap_indices_and_weights(time_values)
            nc, z_var, tmp_file = create_z_file(
                out_file,
                "z_twb",
                lon,
                lat,
                time_values,
                {
                    "long_name": "Standardized wet-bulb temperature",
                    "description": f"z_twb = (Twb - daily_mean_{REF_PERIOD_LABEL}) / daily_std_{REF_PERIOD_LABEL}; Feb 29 uses average climatology of Feb 28 and Mar 1",
                },
            )
            try:
                for t0 in range(0, da.sizes["time"], CHUNK_DAYS):
                    t1 = min(t0 + CHUNK_DAYS, da.sizes["time"])
                    raw = da.isel(time=slice(t0, t1)).values.astype(np.float32)
                    mean, std = read_clim_chunk(mean_arr, std_arr, day_index[t0:t1], day_weights[t0:t1])
                    z = (raw - mean) / std
                    z[~np.isfinite(z)] = np.nan
                    z_var[:, :, t0:t1] = np.transpose(z, (2, 1, 0)).astype(np.float32)
                print(f"[z_twb] {year} saved {out_file}", flush=True)
            finally:
                nc.close()
            os.replace(tmp_file, out_file)


def standardize_precip(overwrite=False):
    PRECIP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_arr, std_arr = load_clim_arrays(
        PRECIP_CLIM_DIR / f"clim_mean_precip_log_{REF_PERIOD_LABEL}.nc",
        PRECIP_CLIM_DIR / f"clim_std_precip_log_{REF_PERIOD_LABEL}.nc",
    )
    for year in YEARS:
        out_file = PRECIP_OUT_DIR / f"z_precip_{year}.nc"
        if out_file.exists() and not overwrite:
            print(f"[z_precip] {year} exists, skip", flush=True)
            continue
        with xr.open_dataset(PRECIP_DIR / f"era5_total_precipitation_{year}.nc") as ds:
            var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
            da = normalize_lon_lat_time(ds[var])
            lon = da["longitude"].values
            lat = da["latitude"].values
            time_values = da["time"].values
            day_index, day_weights = noleap_indices_and_weights(time_values)
            nc, z_var, tmp_file = create_z_file(
                out_file,
                "z_precip",
                lon,
                lat,
                time_values,
                {
                    "long_name": "Standardized log precipitation",
                    "description": (
                        f"z_precip = (log(tp_mm + 1) - daily_mean_{REF_PERIOD_LABEL}) / "
                        f"daily_std_{REF_PERIOD_LABEL}, where tp_mm = tp_m * 1000; "
                        "Feb 29 uses average climatology of Feb 28 and Mar 1"
                    ),
                },
            )
            try:
                for t0 in range(0, da.sizes["time"], CHUNK_DAYS):
                    t1 = min(t0 + CHUNK_DAYS, da.sizes["time"])
                    raw = da.isel(time=slice(t0, t1)).values.astype(np.float32)
                    raw = np.log(raw * PRECIP_M_TO_MM + np.float32(1.0))
                    mean, std = read_clim_chunk(mean_arr, std_arr, day_index[t0:t1], day_weights[t0:t1])
                    std = np.maximum(std, PRECIP_STD_FLOOR)
                    z = (raw - mean) / std
                    z[~np.isfinite(z)] = np.nan
                    z_var[:, :, t0:t1] = np.transpose(z, (2, 1, 0)).astype(np.float32)
                print(f"[z_precip] {year} saved {out_file}", flush=True)
            finally:
                nc.close()
            os.replace(tmp_file, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", choices=["twb", "precip", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.var in ("twb", "both"):
        standardize_twb(overwrite=args.overwrite)
    if args.var in ("precip", "both"):
        standardize_precip(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
