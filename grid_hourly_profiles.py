"""Build hourly DC-OPF profiles for a PowerWorld AUX or ZIP.

Requires Python 3.10+, numpy, scipy, pandas 2+, and powerworld_aux.py.

Bus mapping CSV: bus,zone. Every bus with nonzero snapshot demand needs a
zone. Omit the map to treat all buses as one zone named "system".

Demand CSV: timestamp_utc plus one MW column per zone. The single-zone
column may also be named demand_mw, as in hawaii_hourly_profiles.py.
Timestamps are interval starts in UTC. Optional solar_cf and wind_cf
columns apply to all corresponding generators and preserve compatibility
with the Hawaii script's CSV.

For EIA mode, use EIA balancing authority codes as zone names (e.g. TVA)
or PARENT:SUBBA (from the EIA subregion code list). Set EIA_API_KEY in
your environment. EIA hour-ending observations are converted to starts.

Optional renewable CSV: timestamp_utc,bus,id,capacity_factor. If supplied,
provide every online wind and solar unit for every selected hour. IDs are
strings, including leading zeros. These data override the common fractions.

Example using local files:
  python grid_hourly_profiles.py case.zip --bus-zones bus_zones.csv --demand demand.csv --renewables availability.csv -o profiles.npz

Example for one EIA zone with all buses sharing its demand shape:
  python grid_hourly_profiles.py ACTIVSg500.zip --eia --bus-zones bus_zones.csv --start 2025-07-01T00:00Z --scale-zone-peaks -o profiles.npz
Here bus_zones.csv would map the loaded buses to DUK or another chosen proxy.
By default, demand is in absolute MW. --scale-zone-peaks uses EIA or CSV
shapes and sets each zone's peak over the selected window to its case load.

Output arrays are (hour, bus) or (hour, online generator), in dc_inputs()
order. In CVXPY use demand_mw[t], pmin_mw[t], and pmax_mw[t] for hour t.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from powerworld_aux import load_grid

HOUR = pd.Timedelta(hours=1)
FUEL_PROFILE = {
    "SUN": "solar_cf",
    "SOLAR": "solar_cf",
    "PV": "solar_cf",
    "WND": "wind_cf",
    "WIND": "wind_cf",
}


def _timestamps(values):
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="mixed"))
    if index.hasnans or not (index == index.floor("h")).all():
        raise ValueError("Timestamps must be whole UTC hours.")
    return index


def _read_demand(path):
    frame = pd.read_csv(path)
    frame.index = _timestamps(frame.pop("timestamp_utc"))
    if frame.index.has_duplicates:
        raise ValueError("Demand timestamps must be unique.")
    return frame.sort_index()


def read_bus_zones(path, dc):
    if path is None:
        return {int(bus): "system" for bus in dc["bus_ids"]}
    frame = pd.read_csv(path, usecols=["bus", "zone"], dtype={"bus": int, "zone": str})
    zones = frame["zone"].str.strip()
    if frame["bus"].duplicated().any() or zones.isna().any() or zones.eq("").any():
        raise ValueError(
            "Bus-to-zone rows need a unique bus number and a nonempty zone."
        )
    mapped = dict(zip(frame["bus"], zones))
    unknown = mapped.keys() - dc["bus_index"].keys()
    missing = set(dc["bus_ids"][dc["demand_mw"] != 0]) - mapped.keys()
    if unknown or missing:
        raise ValueError(
            f"Unknown mapped buses: {sorted(unknown)[:8]}. "
            f"Unmapped buses with demand: {sorted(missing)[:8]}."
        )
    return mapped


def _eia_series(zone, hours, api_key):
    parts = zone.split(":")
    if len(parts) > 2 or any(not part.isalnum() for part in parts):
        raise ValueError(f"EIA zone must be BA or BA:SUBBA: {zone!r}")
    if len(parts) == 2:
        route = "region-sub-ba-data"
        filters = {"facets[parent][]": parts[0], "facets[subba][]": parts[1]}
    else:
        route = "region-data"
        filters = {"facets[respondent][]": parts[0], "facets[type][]": "D"}
    # Query hour-ending observations; retain exactly the requested starts.
    end_labels = hours + HOUR
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "start": end_labels[0].strftime("%Y-%m-%dT%H"),
        "end": end_labels[-1].strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
        **filters,
    }
    url = f"https://api.eia.gov/v2/electricity/rto/{route}/data/?{urlencode(params)}"
    try:
        with urlopen(url, timeout=30) as response:
            rows = json.load(response)["response"]["data"]
    except (OSError, KeyError, ValueError):
        raise ValueError(
            f"EIA request failed for {zone}. Check the API key, "
            "zone code, or connection, or use --demand."
        ) from None
    index = _timestamps([row["period"] for row in rows]) - HOUR
    if index.has_duplicates:
        raise ValueError(f"EIA returned multiple observations per hour for {zone}.")
    values = pd.to_numeric(pd.Series([row["value"] for row in rows], index=index))
    return values.reindex(hours)


def allocate_demand(dc, bus_zones, demand, scale_zone_peaks=False):
    """Distribute each zone's MW among its buses by snapshot demand shares."""
    base = dc["demand_mw"]
    if np.any(base < 0):
        raise ValueError("Bus allocation requires nonnegative snapshot bus demand.")
    bus_demand = np.zeros((len(demand), len(base)))
    for zone in sorted(set(bus_zones.values())):
        columns = np.array([bus_zones.get(bus) == zone for bus in dc["bus_ids"]])
        total = base[columns].sum()
        if total == 0:
            continue
        name = "demand_mw" if zone == "system" and zone not in demand else zone
        if name not in demand:
            raise ValueError(f"Missing demand column {name!r}.")
        series = pd.to_numeric(demand[name]).to_numpy(dtype=float)
        bad = np.flatnonzero(~np.isfinite(series) | (series < 0))
        if len(bad):
            raise ValueError(
                f"Invalid or missing demand for {zone} at {demand.index[bad[0]]}."
            )
        if scale_zone_peaks:
            if series.max() <= 0:
                raise ValueError(f"Cannot rescale a zero demand profile for {zone}.")
            series = series * (total / series.max())
        bus_demand[:, columns] = series[:, None] * (base[columns] / total)
    return bus_demand


def generator_bounds(grid, dc, demand, renewable_path):
    """Apply generator-level availability or shared solar/wind fractions."""
    keys = dc["generator_keys"]
    fuels = {
        (row["bus"], row["id"]): row["fuel"].upper().strip() for row in grid.generators
    }
    shape = (len(demand), len(keys))
    pmin = np.broadcast_to(dc["pmin_mw"], shape).copy()
    pmax = np.broadcast_to(dc["pmax_mw"], shape).copy()
    profiles = {}
    if renewable_path is not None:
        table = pd.read_csv(
            renewable_path,
            usecols=["timestamp_utc", "bus", "id", "capacity_factor"],
            dtype={"bus": int, "id": str},
            keep_default_na=False,
        )
        table["timestamp_utc"] = _timestamps(table["timestamp_utc"])
        table["id"] = table["id"].str.strip()
        for key, rows in table.groupby(["bus", "id"]):
            if fuels.get(key) not in FUEL_PROFILE:
                raise ValueError(f"Unknown or non-wind/solar generator {key}.")
            values = rows.set_index("timestamp_utc")["capacity_factor"]
            if values.index.has_duplicates:
                raise ValueError(f"Duplicate renewable timestamp for generator {key}.")
            profiles[key] = values.reindex(demand.index)

    static_units = 0
    for i, key in enumerate(keys):
        field = FUEL_PROFILE.get(fuels[key])
        if field is None:
            continue
        series = profiles.get(key) if renewable_path is not None else demand.get(field)
        if series is None:
            if renewable_path is not None:
                raise ValueError(f"Missing availability for online generator {key}.")
            static_units += 1
            continue
        fractions = pd.to_numeric(series).to_numpy(dtype=float)
        if not np.isfinite(fractions).all() or np.any(
            (fractions < 0) | (fractions > 1)
        ):
            raise ValueError(
                f"Missing or invalid capacity fractions for generator {key}."
            )
        pmin[:, i] = 0
        pmax[:, i] *= fractions
    return pmin, pmax, static_units


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case")
    parser.add_argument("-o", "--output", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demand", type=Path, help="local demand profiles CSV")
    source.add_argument("--eia", action="store_true", help="download EIA-930 demand")
    parser.add_argument("--bus-zones", type=Path)
    parser.add_argument("--renewables", type=Path)
    parser.add_argument("--start", help="first interval start, in UTC")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--scale-zone-peaks", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.hours <= 5000:
        parser.error("--hours must lie between 1 and 5000")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    try:
        grid = load_grid(args.case)
        dc = grid.dc_inputs()
        zones = read_bus_zones(args.bus_zones, dc)
        if args.eia:
            key = os.environ.get("EIA_API_KEY")
            if not key or args.start is None or args.bus_zones is None:
                raise ValueError("--eia requires EIA_API_KEY, --start and --bus-zones.")
            hours = pd.date_range(
                _timestamps([args.start])[0], periods=args.hours, freq="h"
            )
            active_zones = {
                zones[bus] for bus, mw in zip(dc["bus_ids"], dc["demand_mw"]) if mw != 0
            }
            demand = pd.DataFrame(
                {zone: _eia_series(zone, hours, key) for zone in sorted(active_zones)},
                index=hours,
            )
        else:
            demand = _read_demand(args.demand)
            if demand.empty:
                raise ValueError("Demand CSV is empty.")
            start = _timestamps([args.start])[0] if args.start else demand.index[0]
            hours = pd.date_range(start, periods=args.hours, freq="h")
            demand = demand.reindex(hours)

        bus_demand = allocate_demand(dc, zones, demand, args.scale_zone_peaks)
        pmin, pmax, static = generator_bounds(grid, dc, demand, args.renewables)
        np.savez_compressed(
            args.output,
            timestamp_utc=np.array([time.isoformat() for time in hours]),
            hours_per_step=np.array(1.0),
            bus_ids=dc["bus_ids"],
            bus_zones=np.array([zones.get(bus, "") for bus in dc["bus_ids"]]),
            generator_bus_ids=np.array([bus for bus, _ in dc["generator_keys"]]),
            generator_ids=np.array(
                [unit for _, unit in dc["generator_keys"]], dtype=str
            ),
            demand_mw=bus_demand,
            pmin_mw=pmin,
            pmax_mw=pmax,
            demand_source=np.array("EIA-930" if args.eia else str(args.demand)),
            renewable_source=np.array(
                str(args.renewables)
                if args.renewables
                else "Shared demand CSV fractions where supplied; "
                "otherwise static AUX limits"
            ),
            scaled_to_snapshot_zone_peaks=np.array(args.scale_zone_peaks),
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))

    total = bus_demand.sum(axis=1)
    infeasible = (total < pmin.sum(axis=1) - 1e-6) | (total > pmax.sum(axis=1) + 1e-6)
    print(
        f"{args.output}: {len(hours)} hours, {bus_demand.shape[1]} buses, "
        f"{pmax.shape[1]} generators. Peak demand {total.max():.1f} MW."
    )
    print(f"Hours outside system generation bounds: {int(infeasible.sum())}.")
    if static:
        print(
            f"Using static AUX MW limits for {static} wind/solar units without profiles."
        )


if __name__ == "__main__":
    main()
