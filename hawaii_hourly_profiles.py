"""Build hourly Hawaii40 DC-OPF inputs. Requires powerworld_aux.py."""

import argparse
import json
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from powerworld_aux import load_grid

HST = "Pacific/Honolulu"
PUDL = "pudl.catalyst.coop/nightly/core_ferc714__hourly_planning_area_demand.parquet"
FIELDS = ["demand_mw", "solar_cf", "wind_cf"]
HOUR = pd.Timedelta(hours=1)


def read_heco_demand(path, shift_hours):
    import pyarrow.parquet as pq
    from pyarrow.fs import S3FileSystem

    filesystem = None if path else S3FileSystem(anonymous=True, region="us-west-2")
    frame = pq.read_table(
        path or PUDL,
        filesystem=filesystem,
        columns=["datetime_utc", "demand_mwh"],
        filters=[("respondent_id_ferc714_csv", "=", 178)],
    ).to_pandas()
    index = pd.DatetimeIndex(pd.to_datetime(frame["datetime_utc"], utc=True))
    index = index + shift_hours * HOUR
    # MWh over a one-hour interval is numerically equal to its average MW.
    return pd.DataFrame({"demand_mw": frame["demand_mwh"].to_numpy()}, index=index)


def select_week(frame, start):
    if frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError("Input timestamps must be present and unique.")
    if not (frame.index == frame.index.floor("h")).all():
        raise ValueError("Input timestamps must fall on whole hours.")
    if start is None:
        valid = frame.loc[np.isfinite(frame).all(axis=1) & (frame["demand_mw"] > 0)]
        daily = valid.sort_index().tz_convert(HST).resample("D").size()
        complete = daily.eq(24).rolling(7).sum().eq(7)
        if not complete.any():
            raise ValueError("No complete seven-day window in the input.")
        start = (complete[complete].index[-1] - pd.Timedelta(days=6)).date()
    first = pd.Timestamp(start).tz_localize(HST).normalize()
    index = pd.date_range(first, periods=168, freq="h").tz_convert("UTC")
    week = frame.reindex(index)
    if not np.isfinite(week.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Missing or invalid input hours in the week starting {first.date()}."
        )
    return week


def weather_fractions(index):
    local = index.tz_convert(HST)
    query = urlencode(
        {
            "latitude": 21.58,
            "longitude": -158.03,
            "start_date": local[0].date(),
            "end_date": (local[-1] + HOUR).date(),
            "timezone": HST,
            "models": "era5",
            "wind_speed_unit": "ms",
            "hourly": "shortwave_radiation,wind_speed_100m",
        }
    )
    with urlopen(
        "https://archive-api.open-meteo.com/v1/archive?" + query, timeout=45
    ) as response:
        data = json.load(response)["hourly"]

    times = pd.DatetimeIndex(pd.to_datetime(data["time"])).tz_localize(HST)
    times = times.tz_convert("UTC")
    # Open-Meteo labels mean irradiance at the END of the preceding hour.
    ghi = pd.Series(data["shortwave_radiation"], index=times - HOUR).reindex(index)
    speed = pd.Series(data["wind_speed_100m"], index=times).reindex(index)
    ghi, speed = ghi.to_numpy(dtype=float), speed.to_numpy(dtype=float)
    if not np.isfinite([ghi, speed]).all():
        raise ValueError("Weather API returned missing hours for the selected week.")

    solar_cf = np.clip(ghi / 1000, 0, 1)
    ramp = (speed**3 - 3**3) / (12**3 - 3**3)
    wind_cf = np.where(speed < 25, np.clip(ramp, 0, 1), 0)
    return solar_cf, wind_cf


def demo_week(start):
    first = pd.Timestamp(start or "2025-07-07", tz=HST)
    index = pd.date_range(first, periods=168, freq="h").tz_convert("UTC")
    hour = index.tz_convert(HST).hour.to_numpy()
    return pd.DataFrame(
        {
            "demand_mw": 690
            + 70 * np.exp(-(((hour - 12) / 4) ** 2))
            + 240 * np.exp(-(((hour - 19) / 3) ** 2)),
            "solar_cf": 0.8 * np.maximum(0, np.sin(np.pi * (hour - 6) / 12)),
            "wind_cf": 0.28 + 0.15 * np.sin(2 * np.pi * np.arange(168) / 59),
        },
        index=index,
    )


def build_arrays(grid, week, peak_mw):
    demand, solar_cf, wind_cf = week[FIELDS].to_numpy(dtype=float).T
    if np.any(demand <= 0) or not np.isfinite(demand).all():
        raise ValueError("Total hourly demand must be finite and positive.")
    fractions = np.stack([solar_cf, wind_cf])
    if not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("Solar and wind availability must be between 0 and 1.")
    if peak_mw is not None:
        if not np.isfinite(peak_mw) or peak_mw <= 0:
            raise ValueError("--peak-mw must be finite and positive.")
        demand = demand * peak_mw / demand.max()

    dc = grid.dc_inputs()
    bus_load = dc["demand_mw"]
    if np.any(bus_load < 0) or bus_load.sum() <= 0:
        raise ValueError("Expected nonnegative bus loads and a positive system load.")
    keys = dc["generator_keys"]
    fuels = {(g["bus"], g["id"]): g["fuel"].strip().upper() for g in grid.generators}
    fuel = np.array([fuels[key] for key in keys])
    solar = np.isin(fuel, ["SUN", "SOLAR"])
    wind = np.isin(fuel, ["WND", "WIND"])
    if not solar.any() or not wind.any():
        raise ValueError("Expected online wind and solar units in the Hawaii40 case.")

    shape = (len(week), len(keys))
    pmin = np.broadcast_to(dc["pmin_mw"], shape).copy()
    pmax = np.broadcast_to(dc["pmax_mw"], shape).copy()
    pmin[:, solar | wind] = 0
    pmax[:, solar] *= solar_cf[:, None]
    pmax[:, wind] *= wind_cf[:, None]
    return {
        "timestamp_utc": np.array([t.isoformat() for t in week.index]),
        "bus_ids": dc["bus_ids"],
        "generator_bus_ids": np.array([bus for bus, _ in keys]),
        "generator_ids": np.array([unit for _, unit in keys], dtype=str),
        "demand_mw": demand[:, None] * (bus_load / bus_load.sum()),
        "pmin_mw": pmin,
        "pmax_mw": pmax,
        "solar_cf": solar_cf,
        "wind_cf": wind_cf,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("-o", "--output", default="hawaii_week.npz")
    parser.add_argument("--start", help="first day in Hawaiʻi local time, YYYY-MM-DD")
    parser.add_argument(
        "--peak-mw", type=float, help="optional rescaling of weekly peak"
    )
    parser.add_argument(
        "--save-csv", help="save reusable source profiles before scaling"
    )
    parser.add_argument("--pudl-shift-hours", type=int, choices=[-1, 0, 1], default=0)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--parquet", help="local copy of the PUDL demand table")
    source.add_argument("--csv", help="local demand and renewable profiles CSV")
    source.add_argument("--demo", action="store_true", help="synthetic offline example")
    args = parser.parse_args()

    try:
        if args.demo:
            week = demo_week(args.start)
            provenance = "Synthetic demo"
        elif args.csv:
            frame = pd.read_csv(args.csv, usecols=["timestamp_utc", *FIELDS])
            frame.index = pd.to_datetime(
                frame.pop("timestamp_utc"), utc=True, format="mixed"
            )
            week = select_week(frame, args.start)
            provenance = f"Local CSV: {args.csv}"
        else:
            demand = read_heco_demand(args.parquet, args.pudl_shift_hours)
            week = select_week(demand, args.start)
            week["solar_cf"], week["wind_cf"] = weather_fractions(week.index)
            provenance = (
                f"HECO Form 714 via {args.parquet or PUDL}; "
                f"timestamp shift {args.pudl_shift_hours:+d} h; Open-Meteo ERA5 proxy"
            )

        arrays = build_arrays(load_grid(args.case), week, args.peak_mw)
        np.savez_compressed(args.output, **arrays, source=np.array(provenance))
        if args.save_csv:
            week.to_csv(args.save_csv, index_label="timestamp_utc")
    except (OSError, ValueError, ImportError, KeyError) as exc:
        parser.error(str(exc))

    total = arrays["demand_mw"].sum(axis=1)
    short = total > arrays["pmax_mw"].sum(axis=1) + 1e-6
    surplus = total < arrays["pmin_mw"].sum(axis=1) - 1e-6
    print(f"{provenance}\nWeek starting {week.index[0].tz_convert(HST).date()}")
    print(f"{len(week)} hours, {total.max():.1f} MW peak demand")
    print(f"Hours outside aggregate generation bounds: {int((short | surplus).sum())}")


if __name__ == "__main__":
    main()
