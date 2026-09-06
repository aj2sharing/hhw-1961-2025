import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


DEFAULT_PYCHARM_PROJECT = r"D:\pycharm_projects\pythonProject1"
if DEFAULT_PYCHARM_PROJECT not in sys.path:
    sys.path.insert(0, DEFAULT_PYCHARM_PROJECT)

from speedywetbulb import WetBulb_par


T2M_DIR = Path(r"G:\ERA5_daidown\wangwang2014041\t2m")
D2M_DIR = Path(r"G:\ERA5_daidown\wangwang2014041\d2m")
SP_DIR = Path(r"K:\ERA5_daily\surface_pressure")
OUT_DIR = Path(r"J:\wetbulb_global")


def parse_years(text):
    years = []
    for part in text.replace(",", " ").split():
        if ":" in part:
            a, b = part.split(":", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return years


def find_file(base_dir, year, names):
    for name in names:
        path = base_dir / name.format(year=year)
        if path.exists():
            return path
    hits = sorted(base_dir.glob(f"*{year}*.nc"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No NetCDF file for {year} in {base_dir}")


def to_0360_longitudes(ds):
    lon = ds["longitude"].values
    lon0360 = np.mod(lon, 360.0)
    if np.any(np.diff(lon0360) < 0):
        order = np.argsort(lon0360)
        ds = ds.isel(longitude=order)
        lon0360 = lon0360[order]
    return ds.assign_coords(longitude=lon0360)


def dewpoint_to_rh(temp_c, dew_c):
    alpha = np.float32(17.67)
    beta = np.float32(243.5)
    e = np.float32(611.2) * np.exp((alpha * dew_c) / (dew_c + beta))
    es = np.float32(611.2) * np.exp((alpha * temp_c) / (temp_c + beta))
    rh = np.float32(100.0) * (e / es)
    return np.clip(rh, np.float32(0.0), np.float32(100.0))


def compute_year(year, chunk_days, overwrite):
    t_file = find_file(T2M_DIR, year, ["era5_2m_temperature_{year}.nc"])
    d_file = find_file(
        D2M_DIR,
        year,
        ["2m_dewpoint_temperature_{year}.nc", "era5_2m_dewpoint_temperature_{year}.nc"],
    )
    p_file = find_file(SP_DIR, year, ["surface_pressure_{year}.nc"])

    out_file = OUT_DIR / f"wetbulb_daily_{year}.nc"
    tmp_file = OUT_DIR / f"wetbulb_daily_{year}.tmp.nc"
    if out_file.exists() and not overwrite:
        print(f"[{year}] exists, skip: {out_file}", flush=True)
        return
    if tmp_file.exists():
        tmp_file.unlink()

    print(f"\n[{year}] t2m={t_file}", flush=True)
    print(f"[{year}] d2m={d_file}", flush=True)
    print(f"[{year}] sp ={p_file}", flush=True)

    with xr.open_dataset(t_file) as t_src, xr.open_dataset(d_file) as d_src, xr.open_dataset(p_file) as p_src:
        t_ds = to_0360_longitudes(t_src)
        d_ds = to_0360_longitudes(d_src)
        p_ds = to_0360_longitudes(p_src)

        target_lon = t_ds["longitude"]
        target_lat = t_ds["latitude"]
        target_time = t_ds["valid_time"]

        if not np.array_equal(d_ds["longitude"].values, target_lon.values):
            d_ds = d_ds.reindex(longitude=target_lon)
        if not np.array_equal(p_ds["longitude"].values, target_lon.values):
            p_ds = p_ds.reindex(longitude=target_lon)
        if not np.array_equal(d_ds["latitude"].values, target_lat.values):
            d_ds = d_ds.reindex(latitude=target_lat)
        if not np.array_equal(p_ds["latitude"].values, target_lat.values):
            p_ds = p_ds.reindex(latitude=target_lat)
        if not np.array_equal(d_ds["valid_time"].values, target_time.values):
            d_ds = d_ds.reindex(valid_time=target_time)
        if not np.array_equal(p_ds["valid_time"].values, target_time.values):
            p_ds = p_ds.reindex(valid_time=target_time)

        n_time = t_ds.sizes["valid_time"]
        with Dataset(tmp_file, "w", format="NETCDF4") as nc:
            nc.createDimension("valid_time", n_time)
            nc.createDimension("latitude", target_lat.size)
            nc.createDimension("longitude", target_lon.size)

            number_var = nc.createVariable("number", "i8")
            time_var = nc.createVariable("valid_time", "f8", ("valid_time",))
            lat_var = nc.createVariable("latitude", "f8", ("latitude",))
            lon_var = nc.createVariable("longitude", "f8", ("longitude",))
            twb_var = nc.createVariable(
                "twb",
                "f4",
                ("valid_time", "latitude", "longitude"),
                zlib=True,
                complevel=5,
                shuffle=True,
                fill_value=np.float32(-9999.0),
            )

            number_var[...] = np.int64(0)
            time_values = target_time.values.astype("datetime64[ns]")
            time_var[:] = (time_values - np.datetime64("1970-01-01T00:00:00", "ns")) / np.timedelta64(1, "D")
            time_var.units = "days since 1970-01-01 00:00:00"
            time_var.calendar = "proleptic_gregorian"
            lat_var[:] = target_lat.values
            lon_var[:] = target_lon.values
            lat_var.units = "degrees_north"
            lon_var.units = "degrees_east"
            twb_var.long_name = "2-meter wet-bulb temperature"
            twb_var.units = "C"
            twb_var.calculation_method = "Davies-Jones 2008 via speedywetbulb.py"

            for t0 in range(0, n_time, chunk_days):
                t1 = min(t0 + chunk_days, n_time)
                tic = time.time()
                temp_c = (t_ds["t2m"].isel(valid_time=slice(t0, t1)).values.astype(np.float32) - np.float32(273.15))
                dew_c = (d_ds["d2m"].isel(valid_time=slice(t0, t1)).values.astype(np.float32) - np.float32(273.15))
                pressure_pa = p_ds["sp"].isel(valid_time=slice(t0, t1)).values.astype(np.float32)

                rh = dewpoint_to_rh(temp_c, dew_c)
                twb, _, _ = WetBulb_par(temp_c, pressure_pa, rh, 1)
                twb_var[t0:t1, :, :] = twb.astype(np.float32)

                print(f"[{year}] days {t0 + 1:03d}-{t1:03d}/{n_time:03d} done in {time.time() - tic:.1f}s", flush=True)

    os.replace(tmp_file, out_file)
    print(f"[{year}] saved {out_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compute added ERA5 wet-bulb years with longitude alignment.")
    parser.add_argument("--years", default="1961:1989 2025")
    parser.add_argument("--chunk-days", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in parse_years(args.years):
        compute_year(year, args.chunk_days, args.overwrite)


if __name__ == "__main__":
    main()
