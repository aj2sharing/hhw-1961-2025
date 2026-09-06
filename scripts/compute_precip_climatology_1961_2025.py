import argparse
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


PRECIP_DIR = Path(r"K:\era5_daily_pre")
OUT_DIR = PRECIP_DIR / "climatology_1961-1990"
YEARS = list(range(1961, 2026))
BASELINE_YEARS = list(range(1961, 1991))
CHUNK_DAYS = 15
PRECIP_M_TO_MM = np.float32(1000.0)
PRECIP_STD_FLOOR = np.float32(0.1)


def precip_file(year):
    return PRECIP_DIR / f"era5_total_precipitation_{year}.nc"


def normalize_precip(da):
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


def drop_feb29(da):
    time = da["time"]
    keep = ~((time.dt.month == 2) & (time.dt.day == 29))
    return da.sel(time=keep)


def open_precip(year):
    path = precip_file(year)
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
    return ds, normalize_precip(ds[var])


def write_climatology(out_file, var_name, data_lon_lat_day, lon, lat, description):
    tmp_file = out_file.with_suffix(".tmp.nc")
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()

    with Dataset(tmp_file, "w", format="NETCDF4") as nc:
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
        data_var.units = "log(mm + 1)"
        nc.title = "Daily precipitation climatology, 1961-1990"
        nc.years = "1961-1990"
        nc.transform = "log(tp_mm + 1), where tp_mm = tp_m * 1000"
        nc.note = (
            "Input annual files from K:\\era5_daily_pre. Longitudes normalized to 0..359.75. "
            "Leap day removed. ERA5 total precipitation is converted from m to mm before log transform."
        )
        nc.precip_std_floor_log_mm_plus_1 = float(PRECIP_STD_FLOOR)

    os.replace(tmp_file, out_file)


def compute(overwrite=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean_file = OUT_DIR / "clim_mean_precip_log_1961-1990.nc"
    std_file = OUT_DIR / "clim_std_precip_log_1961-1990.nc"
    if mean_file.exists() and std_file.exists() and not overwrite:
        print(f"precip climatology exists, skip: {OUT_DIR}", flush=True)
        return

    with open_precip(YEARS[0])[0] as ds0:
        da0 = normalize_precip(ds0["tp"] if "tp" in ds0.data_vars else ds0[list(ds0.data_vars)[0]])
        lat = da0["latitude"].values
        lon = da0["longitude"].values
        nlat = lat.size
        nlon = lon.size

    sum_x = np.zeros((365, nlat, nlon), dtype=np.float32)
    sum_sq = np.zeros((365, nlat, nlon), dtype=np.float32)
    count = np.zeros(365, dtype=np.float32)

    for year in BASELINE_YEARS:
        tic = time.time()
        ds, da = open_precip(year)
        with ds:
            da = drop_feb29(da)
            nt = da.sizes["time"]
            if nt != 365:
                raise ValueError(f"{year} has {nt} no-leap days, expected 365")
            for t0 in range(0, 365, CHUNK_DAYS):
                t1 = min(t0 + CHUNK_DAYS, 365)
                arr = da.isel(time=slice(t0, t1)).values.astype(np.float32)
                arr = np.log(arr * PRECIP_M_TO_MM + np.float32(1.0))
                sum_x[t0:t1] += arr
                sum_sq[t0:t1] += arr * arr
            count += 1
        print(f"[precip-clim] {year} done in {time.time() - tic:.1f}s", flush=True)

    n = count[:, None, None]
    mean_t_lat_lon = sum_x / n
    var_t_lat_lon = np.maximum(sum_sq / n - mean_t_lat_lon * mean_t_lat_lon, 0.0)
    std_t_lat_lon = np.sqrt(var_t_lat_lon).astype(np.float32)
    std_t_lat_lon = np.maximum(std_t_lat_lon, PRECIP_STD_FLOOR)

    write_climatology(
        mean_file,
        "clim_mean",
        np.transpose(mean_t_lat_lon, (2, 1, 0)),
        lon,
        lat,
        "Daily mean of log-transformed precipitation after converting ERA5 tp from m to mm",
    )
    write_climatology(
        std_file,
        "clim_std",
        np.transpose(std_t_lat_lon, (2, 1, 0)),
        lon,
        lat,
        (
            "Daily standard deviation of log-transformed precipitation after converting ERA5 tp from m to mm; "
            f"values are floored at {float(PRECIP_STD_FLOOR):g} to stabilize hyper-arid grid cells"
        ),
    )
    print(f"[precip-clim] saved {mean_file}", flush=True)
    print(f"[precip-clim] saved {std_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compute(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
