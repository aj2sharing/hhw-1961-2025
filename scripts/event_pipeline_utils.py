import csv
import os
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


YEARS = list(range(1961, 2026))
DEFAULT_PAD_DAYS = 31
WET_DAY_THRESHOLD_M = np.float32(0.001)

TWB_DIR = Path(r"J:\wetbulb_global")
TWB_P95_FILE = TWB_DIR / "climatology_1961-1990" / "climatology_P95_1961-1990.nc"
ZT_DIR = TWB_DIR / "standardized_1961-2025"

PRECIP_DIR = Path(r"K:\era5_daily_pre")
PRECIP_P90_DIR = PRECIP_DIR / "climatology_1961-1990"
PRECIP_P90_FILE = PRECIP_P90_DIR / "precip_p90_wet_days_1961-1990.nc"
ZP_DIR = PRECIP_DIR / "standardized_1961-2025"
PRECIP_P90_LAT_CHUNK = 20


def normalize_time_lat_lon(da):
    rename = {}
    if "valid_time" in da.dims:
        rename["valid_time"] = "time"
    if "lon" in da.dims:
        rename["lon"] = "longitude"
    if "lat" in da.dims:
        rename["lat"] = "latitude"
    if rename:
        da = da.rename(rename)

    lon = da["longitude"].values
    lon0360 = np.mod(lon, 360.0)
    if np.any(np.diff(lon0360) < 0):
        order = np.argsort(lon0360)
        da = da.isel(longitude=order)
        lon0360 = lon0360[order]
    da = da.assign_coords(longitude=lon0360)
    return da.transpose("time", "latitude", "longitude")


def iso_date(value):
    return str(value.astype("datetime64[D]"))


def to_pydate(value) -> date:
    return pd.Timestamp(value).date()


def open_twb(year):
    ds = xr.open_dataset(TWB_DIR / f"wetbulb_daily_{year}.nc")
    return ds, normalize_time_lat_lon(ds["twb"])


def open_z_twb(year):
    ds = xr.open_dataset(ZT_DIR / f"z_twb_{year}.nc")
    return ds, normalize_time_lat_lon(ds["z_twb"])


def open_precip(year):
    ds = xr.open_dataset(PRECIP_DIR / f"era5_total_precipitation_{year}.nc")
    var_name = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
    return ds, normalize_time_lat_lon(ds[var_name])


def open_z_precip(year):
    ds = xr.open_dataset(ZP_DIR / f"z_precip_{year}.nc")
    return ds, normalize_time_lat_lon(ds["z_precip"])


def pad_times_before(first_date, pad_days):
    return np.array(
        [first_date - np.timedelta64(pad_days - i, "D") for i in range(pad_days)],
        dtype="datetime64[D]",
    )


def pad_times_after(last_date, pad_days):
    return np.array(
        [last_date + np.timedelta64(i + 1, "D") for i in range(pad_days)],
        dtype="datetime64[D]",
    )


def concat_year_with_padding(year, opener, pad_before=DEFAULT_PAD_DAYS, pad_after=DEFAULT_PAD_DAYS):
    handles = []
    arrays = []
    time_arrays = []

    if year > YEARS[0]:
        ds, da = opener(year - 1)
        handles.append(ds)
        n_prev = min(pad_before, da.sizes["time"])
        start_prev = da.sizes["time"] - n_prev
        arrays.append(da.isel(time=slice(start_prev, da.sizes["time"])).values)
        time_arrays.append(da["time"].values[start_prev:da.sizes["time"]].astype("datetime64[D]"))
        if n_prev < pad_before:
            pad_shape = (pad_before - n_prev, da.sizes["latitude"], da.sizes["longitude"])
            fill = np.nan if da.dtype.kind != "b" else False
            arrays.insert(0, np.full(pad_shape, fill, dtype=np.float32 if da.dtype.kind != "b" else bool))
            time_arrays.insert(0, pad_times_before(da["time"].values[0].astype("datetime64[D]"), pad_before - n_prev))
    else:
        ds, da = opener(year)
        handles.append(ds)
        pad_shape = (pad_before, da.sizes["latitude"], da.sizes["longitude"])
        fill = np.nan if da.dtype.kind != "b" else False
        arrays.append(np.full(pad_shape, fill, dtype=np.float32 if da.dtype.kind != "b" else bool))
        time_arrays.append(pad_times_before(np.datetime64(f"{year}-01-01"), pad_before))

    ds, da = opener(year)
    handles.append(ds)
    cur_len = da.sizes["time"]
    arrays.append(da.values)
    time_arrays.append(da["time"].values.astype("datetime64[D]"))
    lon = da["longitude"].values
    lat = da["latitude"].values

    if year < YEARS[-1]:
        ds, da = opener(year + 1)
        handles.append(ds)
        n_next = min(pad_after, da.sizes["time"])
        arrays.append(da.isel(time=slice(0, n_next)).values)
        time_arrays.append(da["time"].values[:n_next].astype("datetime64[D]"))
        if n_next < pad_after:
            pad_shape = (pad_after - n_next, lat.size, lon.size)
            fill = np.nan if arrays[1].dtype.kind != "b" else False
            arrays.append(np.full(pad_shape, fill, dtype=np.float32 if arrays[1].dtype.kind != "b" else bool))
            time_arrays.append(pad_times_after(da["time"].values[-1].astype("datetime64[D]"), pad_after - n_next))
    else:
        pad_shape = (pad_after, lat.size, lon.size)
        fill = np.nan if arrays[1].dtype.kind != "b" else False
        arrays.append(np.full(pad_shape, fill, dtype=np.float32 if arrays[1].dtype.kind != "b" else bool))
        time_arrays.append(pad_times_after(np.datetime64(f"{year}-12-31"), pad_after))

    return np.concatenate(arrays, axis=0), np.concatenate(time_arrays), handles, cur_len, lon, lat


def find_event_starts(event_mask_ext, year_start, year_end):
    prev = np.concatenate([np.zeros_like(event_mask_ext[:1], dtype=bool), event_mask_ext[:-1]], axis=0)
    starts = event_mask_ext & ~prev
    starts[:year_start] = False
    starts[year_end + 1:] = False
    return np.where(starts)


def event_end(mask_ts, start_idx):
    end_idx = start_idx
    while end_idx + 1 < mask_ts.shape[0] and mask_ts[end_idx + 1]:
        end_idx += 1
    return end_idx


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def require_pyarrow():
    if pa is None or pq is None:
        raise ImportError("pyarrow is required for parquet event tables")


def open_parquet_writer(out_file, schema, compression="zstd"):
    require_pyarrow()
    ensure_parent(out_file)
    tmp_file = out_file.with_name(out_file.stem + ".tmp" + out_file.suffix)
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()
    writer = pq.ParquetWriter(tmp_file, schema=schema, compression=compression)
    return writer, tmp_file


def write_parquet_rows(writer, columns, rows, schema):
    if not rows:
        return
    data = {name: [row[idx] for row in rows] for idx, name in enumerate(columns)}
    table = pa.Table.from_pydict(data, schema=schema)
    writer.write_table(table)


def finalize_parquet_writer(writer, tmp_file, out_file):
    writer.close()
    if out_file.exists():
        out_file.unlink()
    os.replace(tmp_file, out_file)


def load_event_table(path: Path):
    if not path.exists():
        return pd.DataFrame()
    require_pyarrow()
    df = pd.read_parquet(path)
    for col in ("start_date", "end_date", "peak_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def write_csv_rows(out_file, columns, rows):
    ensure_parent(out_file)
    tmp_file = out_file.with_suffix(".tmp.csv")
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()
    with open(tmp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp_file, out_file)


def load_twb_threshold():
    with xr.open_dataset(TWB_P95_FILE) as ds:
        p95 = ds["twb_p95"].load()
    if "lon" in p95.dims:
        p95 = p95.rename({"lon": "longitude"})
    if "lat" in p95.dims:
        p95 = p95.rename({"lat": "latitude"})
    p95 = p95.transpose("latitude", "longitude")
    thr = np.maximum(p95.values.astype(np.float32), np.float32(23.0))
    return thr, p95["longitude"].values, p95["latitude"].values


def compute_precip_p90_file(
    out_file=PRECIP_P90_FILE,
    baseline_years=range(1961, 1991),
    wet_day_threshold=WET_DAY_THRESHOLD_M,
    overwrite=False,
):
    ensure_parent(out_file)
    if out_file.exists() and not overwrite:
        return out_file

    handles = []
    arrays = []
    try:
        for year in baseline_years:
            ds, da = open_precip(year)
            handles.append(ds)
            arrays.append(da)

        lat_vals = arrays[0]["latitude"].values
        lon_vals = arrays[0]["longitude"].values
        nlat = lat_vals.size
        nlon = lon_vals.size
        p90 = np.empty((nlat, nlon), dtype=np.float32)

        for j0 in range(0, nlat, PRECIP_P90_LAT_CHUNK):
            j1 = min(j0 + PRECIP_P90_LAT_CHUNK, nlat)
            pieces = []
            for da in arrays:
                block = da.isel(latitude=slice(j0, j1)).values.astype(np.float32)
                block[block <= wet_day_threshold] = np.nan
                pieces.append(block)
            full = np.concatenate(pieces, axis=0)
            p90[j0:j1, :] = np.nanpercentile(full, 90.0, axis=0).astype(np.float32)
            print(
                f"[precip-p90] lat rows {j0 + 1:03d}-{j1:03d}/{nlat:03d} done",
                flush=True,
            )

        with xr.open_dataset(PRECIP_DIR / f"era5_total_precipitation_{baseline_years[0]}.nc") as demo_ds:
            demo_var = "tp" if "tp" in demo_ds.data_vars else list(demo_ds.data_vars)[0]
            demo_da = normalize_time_lat_lon(demo_ds[demo_var])
            lat_vals = demo_da["latitude"].values
            lon_vals = demo_da["longitude"].values

        tmp_file = out_file.with_suffix(".tmp.nc")
        if tmp_file.exists():
            tmp_file.unlink()
        if out_file.exists():
            out_file.unlink()
        with Dataset(tmp_file, "w", format="NETCDF4") as nc:
            nc.createDimension("longitude", nlon)
            nc.createDimension("latitude", nlat)
            lon_var = nc.createVariable("longitude", "f8", ("longitude",))
            lat_var = nc.createVariable("latitude", "f8", ("latitude",))
            p90_var = nc.createVariable(
                "precip_p90",
                "f4",
                ("latitude", "longitude"),
                zlib=True,
                complevel=3,
                shuffle=True,
                fill_value=np.float32(np.nan),
            )
            lon_var[:] = lon_vals
            lat_var[:] = lat_vals
            p90_var[:, :] = p90
            lon_var.units = "degrees_east"
            lat_var.units = "degrees_north"
            p90_var.units = "m"
            p90_var.description = "Grid-cell 90th percentile of wet-day daily precipitation"
            nc.base_period = f"{min(baseline_years)}-{max(baseline_years)}"
            nc.wet_day_threshold_m = float(wet_day_threshold)
        os.replace(tmp_file, out_file)
        return out_file
    finally:
        for ds in handles:
            ds.close()


def load_precip_p90():
    if not PRECIP_P90_FILE.exists():
        compute_precip_p90_file()
    with xr.open_dataset(PRECIP_P90_FILE) as ds:
        p90 = ds["precip_p90"].load()
    if "lon" in p90.dims:
        p90 = p90.rename({"lon": "longitude"})
    if "lat" in p90.dims:
        p90 = p90.rename({"lat": "latitude"})
    p90 = p90.transpose("latitude", "longitude")
    return p90.values.astype(np.float32), p90["longitude"].values, p90["latitude"].values


def write_grid_metrics(out_file, lon, lat, counts, sums, metric_names, title, definition):
    ensure_parent(out_file)
    tmp_file = out_file.with_suffix(".tmp.nc")
    if tmp_file.exists():
        tmp_file.unlink()
    if out_file.exists():
        out_file.unlink()
    with Dataset(tmp_file, "w", format="NETCDF4") as nc:
        nc.createDimension("longitude", lon.size)
        nc.createDimension("latitude", lat.size)
        lon_var = nc.createVariable("longitude", "f8", ("longitude",))
        lat_var = nc.createVariable("latitude", "f8", ("latitude",))
        cnt_var = nc.createVariable("event_count", "i4", ("longitude", "latitude"), zlib=True, complevel=3)
        lon_var[:] = lon
        lat_var[:] = lat
        cnt_var[:, :] = counts.astype(np.int32)
        lon_var.units = "degrees_east"
        lat_var.units = "degrees_north"
        for name, total in zip(metric_names, sums):
            var = nc.createVariable(
                f"avg_{name}",
                "f4",
                ("longitude", "latitude"),
                zlib=True,
                complevel=3,
                fill_value=np.float32(np.nan),
            )
            out = np.full(counts.shape, np.nan, dtype=np.float32)
            mask = counts > 0
            out[mask] = (total[mask] / counts[mask]).astype(np.float32)
            var[:, :] = out
        nc.title = title
        nc.definition = definition
    os.replace(tmp_file, out_file)
