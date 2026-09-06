import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa

from event_pipeline_utils import (
    DEFAULT_PAD_DAYS,
    YEARS,
    concat_year_with_padding,
    finalize_parquet_writer,
    load_twb_threshold,
    open_parquet_writer,
    open_twb,
    open_z_twb,
    to_pydate,
    write_parquet_rows,
)


OUT_DIR = Path(r"J:\HHW_events_1961-2025")
ANNUAL_DIR = OUT_DIR / "annual"
MIN_DURATION = 3
CELL_CHUNK = 32768
PARQUET_BATCH_ROWS = 100000

EVENT_COLUMNS = [
    "lon",
    "lat",
    "start_date",
    "end_date",
    "duration_days",
    "peak_date",
    "peak_twb",
    "peak_z_twb",
]

EVENT_SCHEMA = pa.schema([
    ("lon", pa.float32()),
    ("lat", pa.float32()),
    ("start_date", pa.date32()),
    ("end_date", pa.date32()),
    ("duration_days", pa.int32()),
    ("peak_date", pa.date32()),
    ("peak_twb", pa.float32()),
    ("peak_z_twb", pa.float32()),
])


def iter_event_bounds(mask_ext: np.ndarray, year_start: int, year_end: int, nlon: int):
    nt, nlat, _ = mask_ext.shape
    flat = mask_ext.reshape(nt, nlat * nlon)
    ncell = flat.shape[1]
    zeros = None

    for c0 in range(0, ncell, CELL_CHUNK):
        c1 = min(c0 + CELL_CHUNK, ncell)
        sub = flat[:, c0:c1]
        if zeros is None or zeros.shape[1] != sub.shape[1]:
            zeros = np.zeros((1, sub.shape[1]), dtype=bool)
        start_mask = sub & ~np.concatenate([zeros, sub[:-1]], axis=0)
        end_mask = sub & ~np.concatenate([sub[1:], zeros], axis=0)
        active = np.flatnonzero(np.any(start_mask[year_start:year_end + 1], axis=0))
        for local_idx in active:
            starts_all = np.flatnonzero(start_mask[:, local_idx])
            ends_all = np.flatnonzero(end_mask[:, local_idx])
            sel = (starts_all >= year_start) & (starts_all <= year_end)
            starts = starts_all[sel]
            ends = ends_all[sel]
            if starts.size == 0:
                continue
            cell = c0 + int(local_idx)
            j, i = divmod(cell, nlon)
            yield j, i, starts, ends


def identify(overwrite=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNUAL_DIR.mkdir(parents=True, exist_ok=True)

    threshold, _, _ = load_twb_threshold()
    total_events = 0

    for year in YEARS:
        out_file = ANNUAL_DIR / f"HHW_Events_{year}.parquet"
        if out_file.exists() and not overwrite:
            print(f"[hhw-events] {year} exists, skip", flush=True)
            continue
        writer, tmp_file = open_parquet_writer(out_file, EVENT_SCHEMA)

        twb_ext, time_ext, twb_handles, cur_len, lon, lat = concat_year_with_padding(
            year,
            open_twb,
            pad_before=DEFAULT_PAD_DAYS,
            pad_after=DEFAULT_PAD_DAYS,
        )
        zt_ext, _, zt_handles, _, _, _ = concat_year_with_padding(
            year,
            open_z_twb,
            pad_before=DEFAULT_PAD_DAYS,
            pad_after=DEFAULT_PAD_DAYS,
        )
        try:
            exceed_ext = twb_ext >= threshold[None, :, :]
            year_start = DEFAULT_PAD_DAYS
            year_end = DEFAULT_PAD_DAYS + cur_len - 1
            year_events = 0
            buffer_rows = []

            for j, i, starts, ends in iter_event_bounds(exceed_ext, year_start, year_end, lon.size):
                for start_idx, end_idx in zip(starts, ends):
                    duration = end_idx - start_idx + 1
                    if duration < MIN_DURATION:
                        continue

                    twb_segment = twb_ext[start_idx:end_idx + 1, j, i]
                    if np.all(np.isnan(twb_segment)):
                        continue
                    peak_offset = int(np.nanargmax(twb_segment))
                    peak_idx = start_idx + peak_offset
                    peak_twb = float(twb_ext[peak_idx, j, i])
                    peak_z_twb = float(zt_ext[peak_idx, j, i])
                    if not np.isfinite(peak_twb) or not np.isfinite(peak_z_twb):
                        continue

                    buffer_rows.append([
                        float(lon[i]),
                        float(lat[j]),
                        to_pydate(time_ext[start_idx]),
                        to_pydate(time_ext[end_idx]),
                        duration,
                        to_pydate(time_ext[peak_idx]),
                        peak_twb,
                        peak_z_twb,
                    ])
                    year_events += 1

                if len(buffer_rows) >= PARQUET_BATCH_ROWS:
                    write_parquet_rows(writer, EVENT_COLUMNS, buffer_rows, EVENT_SCHEMA)
                    buffer_rows.clear()

            if buffer_rows:
                write_parquet_rows(writer, EVENT_COLUMNS, buffer_rows, EVENT_SCHEMA)

            total_events += year_events
            print(f"[hhw-events] {year} events: {year_events}", flush=True)
            finalize_parquet_writer(writer, tmp_file, out_file)
        finally:
            for ds in twb_handles + zt_handles:
                ds.close()

    print(f"[hhw-events] total events: {total_events}", flush=True)
    print(f"[hhw-events] saved annual files to {ANNUAL_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    identify(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
