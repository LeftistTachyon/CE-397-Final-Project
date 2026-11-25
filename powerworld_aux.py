"""Import a PowerWorld AUX network export into JSON or NumPy/SciPy.

Requires Python 3.10+, numpy and scipy. See --help for archive and cost options.

    grid = load_grid("ACTIVSg500.zip")
    dc = grid.dc_inputs()
    # Angles in radians and f in MW, oriented from each branch's from bus:
    # incidence @ theta - shift_rad == mw_to_angle * f
    # incidence.T @ f == generator_map @ pg + battery_injection - demand_mw
    # theta[reference_indices] == 0 (one reference per connected component)
    # Apply abs(f) <= rate_a_mva only where a positive rate A is present.
    # apply pg bounds from pmin_mw/pmax_mw. See cost_c2_c1_c0 for costs.

Demand is the online ZIP load evaluated at 1 p.u., minus online embedded
generation. DC inputs omit AC losses, shunts, reactive power, and interfaces.
An AUX file is read as data. SCRIPT and unrelated object tables are ignored.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import warnings
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# A quoted string may contain whitespace, doubled quotes, or literal '//'.
_TOKEN = re.compile(r'"(?:[^"]|"")*"|//[^\n]*|[()[\]{},]|[^\s"()[\]{},/]+|/')
_TABLES = {"bus", "branch", "gen", "load", "substation", "sim_solution_options_value"}
_EXTRA_NETWORK = {"dctransmissionline", "vscdcline", "mtdcrecord", "3wxformer", "facts"}


def _unquote(token):
    return token[1:-1].replace('""', '"').strip() if token.startswith('"') else token


def _decode(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252")


def _blocks(text):
    """Yield supported blocks, with rows split on physical lines."""
    tokens = [
        (match[0], i)
        for i, line in enumerate(text.splitlines(), 1)
        for match in _TOKEN.finditer(line)
        if not match[0].startswith("//")
    ]
    cursor = 0
    while cursor < len(tokens):
        start = cursor
        cursor += 1
        kind = tokens[start][0].lower()
        if cursor < len(tokens) and tokens[cursor][0] == "(":
            close = cursor + 1
            depth = 1
            while close < len(tokens):
                token = tokens[close][0]
                depth += (token == "(") - (token == ")")
                if depth == 0:
                    break
                close += 1
            header = [token for token, _ in tokens[cursor + 1 : close]]
            cursor = close + 1
        else:
            header = []
        if cursor >= len(tokens) or tokens[cursor][0] != "{":
            continue
        cursor += 1
        body = cursor
        depth = 1
        while cursor < len(tokens) and depth:
            token = tokens[cursor][0]
            depth += (token == "{") - (token == "}")
            cursor += 1
        if depth:
            raise ValueError(f"Unclosed AUX block at line {tokens[start][1]}")
        if kind == "data":
            if not header:
                raise ValueError(f"Empty DATA header at line {tokens[start][1]}")
            kind = _unquote(header[0]).lower()
            if kind not in _TABLES | _EXTRA_NETWORK:
                continue
            if "[" not in header or "]" not in header:
                raise ValueError(f"Expected DATA field list at line {tokens[start][1]}")
            header = header[header.index("[") + 1 : header.index("]")]
        if kind not in _TABLES | _EXTRA_NETWORK:
            continue
        fields = [_unquote(token).lower() for token in header if token != ","]
        lines = defaultdict(list)
        for token, line in tokens[body : cursor - 1]:
            lines[line].append(token)
        rows = []
        subdata = False
        for line, values in lines.items():
            first = values[0].upper()
            if first.startswith("<SUBDATA"):
                subdata = True
            if subdata:
                if first.startswith("</SUBDATA"):
                    subdata = False
                continue
            if len(fields) != len(values):
                raise ValueError(
                    f"AUX line {line}, {kind}: expected {len(fields)} values, "
                    f"found {len(values)}"
                )
            rows.append(dict(zip(fields, map(_unquote, values))))
        yield kind, rows


def _get(row, *keys):
    return next((row[k] for k in keys if k in row), "")


def _num(row, *keys, default=None):
    raw = _get(row, *keys)
    if raw == "":
        return default
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {keys[0]} value {raw!r}")
    return value


def _need(row, *keys):
    value = _num(row, *keys)
    if value is None:
        raise ValueError(f"Missing {keys[0]} field in AUX row")
    return value


def _id(row, *keys):
    value = _need(row, *keys)
    if not value.is_integer():
        raise ValueError(f"{keys[0]} must be an integer")
    return int(value)


def _on(row, *keys):
    status = _get(row, *keys).lower()
    if status in {"closed", "yes", "1", "in service"}:
        return True
    if status in {"open", "no", "0", "out of service"}:
        return False
    raise ValueError(f"Unknown or missing {keys[0]} value {status!r}")


def _unique(rows, keys, kind):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in index:
            raise ValueError(f"Duplicate {kind} key {key}")
        index[key] = row
    return index


def _read_case(path, member):
    if path.suffix.lower() != ".zip":
        if member:
            raise ValueError("--member requires a ZIP archive")
        if path.suffix.lower() != ".aux":
            raise ValueError("Input must be a PowerWorld .aux export or a .zip archive")
        return path.read_bytes(), path.name
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".aux")]
        if member is None:
            by_stem = [
                name
                for name in names
                if PurePosixPath(name).stem.lower() == path.stem.lower()
            ]
            candidates = by_stem or [
                name
                for name in names
                if not PurePosixPath(name).stem.lower().endswith("_dynamics")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Use --member to choose a network AUX file from {names}"
                )
            member = candidates[0]
        if member not in names:
            raise ValueError(f"AUX member {member!r} not found. Available: {names}")
        return archive.read(member), member


def _aux_cost(row):
    """Hourly polynomial ordered c3, c2, c1, c0. Empty model stays unknown."""
    model = _get(row, "gencostmodel", "costmodel").casefold()
    fuel = _num(row, "genfuelcost", "fuelcost")
    if model != "cubic" or fuel is None:
        return None
    c3 = fuel * _num(row, "geniod", "iod", default=0)
    c2 = fuel * _num(row, "genioc", "ioc", default=0)
    c1 = fuel * _num(row, "geniob", "iob", default=0)
    c1 += _num(row, "genvariableom", "variableom", default=0)
    total_fixed = _num(row, "gentotalfixedcosts", "totalfixedcosts")
    if total_fixed is None:
        total_fixed = _num(row, "genfixedcost", "fixedcost", default=0)
        total_fixed += fuel * _num(row, "genioa", "ioa", default=0)
    return [c3, c2, c1, total_fixed]


def _branch(row, buses, base_mva):
    a = _id(row, "busnumfrom", "busnum")
    b = _id(row, "busnumto", "busnum:1")
    circuit = _get(row, "circuit", "linecircuit")
    label = f"branch {a}-{b} circuit {circuit!r}"
    if a not in buses or b not in buses or a == b or not circuit:
        raise ValueError(f"Invalid endpoint or circuit ID on {label}")
    kind = _get(row, "branchdevicetype")
    shift = _num(row, "phase", "linephase", default=0)
    resistance, reactance = _num(row, "r", "liner"), _num(row, "x", "linex")
    tap = _num(row, "tap", "linetap", default=1)
    if _num(row, "xxfbase", "linex:1") is not None:
        # PowerWorld's conversion from transformer to system bases.
        xf_base = _need(row, "xfmvabase")
        xf_from = _need(row, "xfnomkvbasefrom", "xfnominalkv")
        xf_to = _need(row, "xfnomkvbaseto", "xfnominalkv:1")
        fixed_from = _num(row, "tapfixedfrom", "xffixedtap", default=1)
        fixed_to = _num(row, "tapfixedto", "xffixedtap:1", default=1)
        xf_tap = _need(row, "tapxfbase", "linetap:1")
        from_kv, to_kv = buses[a]["nominal_kv"], buses[b]["nominal_kv"]
        if (
            min(xf_base, xf_from, xf_to, from_kv, to_kv, fixed_from, fixed_to, xf_tap)
            <= 0
        ):
            raise ValueError(f"Invalid transformer base or tap on {label}")
        to_scale = fixed_to * xf_to / to_kv
        impedance_scale = base_mva / xf_base * to_scale**2
        tap = (xf_tap + fixed_from - 1) * xf_from / from_kv / to_scale
        xf_resistance = _num(row, "rxfbase", "liner:1")
        resistance = None if xf_resistance is None else xf_resistance * impedance_scale
        reactance = _need(row, "xxfbase", "linex:1") * impedance_scale
    if reactance is None or tap < 0:
        raise ValueError(f"Missing reactance or invalid tap on {label}")
    tap = tap or 1.0  # A system-base zero tap indicates no off-nominal tap.
    rate = _num(row, "limitmvaa", "lineamva")
    if rate is not None and rate < 0:
        raise ValueError(f"Negative rate A on {label}")
    return dict(
        from_bus=a,
        to_bus=b,
        circuit=circuit,
        device_type=kind,
        online=_on(row, "status", "linestatus"),
        r_pu=resistance,
        x_pu=reactance,
        tap=tap,
        shift_deg=shift,
        rate_a_mva=rate,
        bypassed=_get(row, "bypass", "seriescapstatus").lower() == "yes",
        impedance_correction_table=_num(row, "impcorrtable", "xftablenum", default=0),
    )


@dataclass
class Grid:
    source: str
    base_mva: float
    buses: list[dict]
    branches: list[dict]
    generators: list[dict]
    loads: list[dict]
    substations: list[dict]
    unsupported_network_objects: list[str]

    def dc_inputs(self):
        """Return ordered NumPy data and sparse matrices for a DC model.

        cost_c2_c1_c0 has NaNs for missing, cubic, or nonconvex cost curves.
        Check those entries before forming a CVXPY quadratic objective.
        """
        branches = [row for row in self.branches if row["online"]]
        generators = [row for row in self.generators if row["online"]]
        if self.unsupported_network_objects:
            raise ValueError(
                f"Unsupported AUX network tables: {self.unsupported_network_objects}"
            )
        for row in branches:
            if (
                row["bypassed"]
                or row["impedance_correction_table"]
                or row["device_type"].lower()
                not in {
                    "",
                    "line",
                    "transformer",
                    "breaker",
                    "switch",
                    "disconnect",
                    "load break disconnect",
                    "fuse",
                    "zbr",
                    "jumper",
                }
            ):
                raise ValueError(f"Branch needs special DC treatment: {row}")
        bus_ids = [row["bus"] for row in self.buses]
        bus_index = {bus: i for i, bus in enumerate(bus_ids)}
        f = [bus_index[row["from_bus"]] for row in branches]
        t = [bus_index[row["to_bus"]] for row in branches]
        n, m = len(bus_ids), len(branches)
        incidence = coo_matrix(
            (np.r_[np.ones(m), -np.ones(m)], (np.tile(np.arange(m), 2), f + t)),
            shape=(m, n),
        ).tocsr()
        generator_map = coo_matrix(
            (
                np.ones(len(generators)),
                (
                    [bus_index[row["bus"]] for row in generators],
                    np.arange(len(generators)),
                ),
            ),
            shape=(n, len(generators)),
        ).tocsr()
        graph = coo_matrix((np.ones(m), (f, t)), shape=(n, n)).tocsr()
        n_components, components = connected_components(graph, directed=False)
        references = np.full(n_components, -1, dtype=int)
        for i, component in enumerate(components):
            if references[component] == -1 or (
                self.buses[i]["slack"]
                and not self.buses[references[component]]["slack"]
            ):
                references[component] = i
        demand = np.zeros(n)
        for row in self.loads:
            if row["online"]:
                demand[bus_index[row["bus"]]] += row["net_mw_at_1_pu"]
        cost = np.full((len(generators), 3), np.nan)
        for i, row in enumerate(generators):
            coefficients = row["cost_c3_c2_c1_c0"]
            if (
                coefficients is not None
                and coefficients[0] == 0
                and coefficients[1] >= 0
            ):
                cost[i, :] = coefficients[1:]
        return dict(
            bus_ids=np.array(bus_ids),
            bus_index=bus_index,
            incidence=incidence,
            generator_map=generator_map,
            reference_indices=references,
            components=components,
            demand_mw=demand,
            branch_keys=[
                (row["from_bus"], row["to_bus"], row["circuit"]) for row in branches
            ],
            generator_keys=[(row["bus"], row["id"]) for row in generators],
            mw_to_angle=np.array(
                [row["x_pu"] * row["tap"] / self.base_mva for row in branches]
            ),
            shift_rad=np.deg2rad([row["shift_deg"] for row in branches]),
            rate_a_mva=np.array([row["rate_a_mva"] for row in branches], dtype=float),
            pmin_mw=np.array([row["pmin_mw"] for row in generators]),
            pmax_mw=np.array([row["pmax_mw"] for row in generators]),
            cost_c2_c1_c0=cost,
        )


def load_grid(path, *, member=None, base_mva=None, matpower=None):
    """Read a network AUX export or an archive containing one.

    Pass matpower=True to read cost curves from the companion .m in the ZIP,
    or pass a path to a separate .m file. This assumes the two exports preserve
    generator order, including at buses with several units. Bus, status and
    upper MW bounds are checked row by row. AUX dispatch limits are retained.
    """
    data, source = _read_case(Path(path), member)
    tables = defaultdict(list)
    for kind, rows in _blocks(_decode(data)):
        tables[kind].extend(rows)
    if not tables["bus"]:
        raise ValueError(
            f"{source} contains no Bus records. Choose the network export."
        )
    if base_mva is None:
        base_mva = next(
            (
                _need(row, "value", "valuefield")
                for row in tables["sim_solution_options_value"]
                if _get(row, "option", "variablename").lower() == "mvabase"
            ),
            None,
        )
    if base_mva is None or not math.isfinite(base_mva) or base_mva <= 0:
        raise ValueError(
            "Missing or invalid system MVA base. Supply --base-mva if needed."
        )
    buses = [
        dict(
            bus=_id(row, "number", "busnum", "bus"),
            name=_get(row, "name", "busname"),
            nominal_kv=_need(row, "nomkv", "busnomvolt"),
            slack=_on(row, "slack", "busslack"),
            substation=_num(row, "subnumber", "subnum"),
            latitude=_num(row, "latitude"),
            longitude=_num(row, "longitude"),
        )
        for row in tables["bus"]
    ]
    bus_index = {key[0]: row for key, row in _unique(buses, ("bus",), "bus").items()}
    branches = [_branch(row, bus_index, base_mva) for row in tables["branch"]]
    _unique(branches, ("from_bus", "to_bus", "circuit"), "branch")
    generators = []
    for row in tables["gen"]:
        if not any(k in row for k in ("mwmax", "genmwmax", "mwmin", "genmwmin")):
            continue  # Dynamics and reporting blocks repeat the same generator keys.
        item = dict(
            bus=_id(row, "busnum"),
            id=_get(row, "id", "genid"),
            online=_on(row, "status", "genstatus"),
            pmin_mw=_need(row, "mwmin", "genmwmin"),
            pmax_mw=_need(row, "mwmax", "genmwmax"),
            setpoint_mw=_num(row, "mwsetpoint", "genmwsetpoint"),
            fuel=_get(row, "fueltypecode", "genfueltype", "fueltype"),
            cost_model=_get(row, "gencostmodel", "costmodel"),
            cost_c3_c2_c1_c0=_aux_cost(row),
            cost_source=None,
        )
        if item["pmin_mw"] > item["pmax_mw"]:
            raise ValueError(f"Generator minimum exceeds maximum: {item}")
        if item["cost_c3_c2_c1_c0"] is not None:
            item["cost_source"] = "AUX"
        generators.append(item)
    _unique(generators, ("bus", "id"), "generator")
    loads = []
    for row in tables["load"]:
        if not any(k in row for k in ("smw", "loadsmw")):
            continue
        components = [
            _num(row, *aliases, default=0)
            for aliases in [("smw", "loadsmw"), ("imw", "loadimw"), ("zmw", "loadzmw")]
        ]
        dist = _num(row, "distmwinput", default=0)
        online_dist = dist != 0 and _on(row, "diststatus")
        loads.append(
            dict(
                bus=_id(row, "busnum"),
                id=_get(row, "id", "loadid"),
                online=_on(row, "status", "loadstatus"),
                constant_power_mw=components[0],
                constant_current_mw=components[1],
                constant_impedance_mw=components[2],
                embedded_generation_mw=dist if online_dist else 0.0,
                net_mw_at_1_pu=sum(components) - (dist if online_dist else 0.0),
            )
        )
    _unique(loads, ("bus", "id"), "load")
    for row in generators + loads:
        if row["bus"] not in bus_index or not row["id"]:
            raise ValueError(f"Unknown bus or missing device ID: {row}")
    substations = [
        dict(
            number=_id(row, "number", "subnum"),
            name=_get(row, "name", "subname"),
            latitude=_num(row, "latitude"),
            longitude=_num(row, "longitude"),
        )
        for row in tables["substation"]
    ]
    if matpower is True:
        if Path(path).suffix.lower() != ".zip":
            raise ValueError("Automatic MATPOWER selection requires a ZIP archive")
        stem = PurePosixPath(source).stem.lower()
        with ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).name.lower() in {f"{stem}.m", f"case_{stem}.m"}
            ]
            if len(names) != 1:
                raise ValueError(
                    "Could not choose one companion MATPOWER file. "
                    "Supply its path with --matpower FILE."
                )
            _add_matpower_costs(archive.read(names[0]), names[0], generators)
    elif matpower is not None:
        cost_file = Path(matpower)
        _add_matpower_costs(cost_file.read_bytes(), cost_file.name, generators)
    return Grid(
        source,
        base_mva,
        buses,
        branches,
        generators,
        loads,
        substations,
        sorted(kind for kind in _EXTRA_NETWORK if tables[kind]),
    )


def _matpower_rows(text, name):
    """Parse literal exported matrices without running MATLAB code."""
    pattern = rf"\bmpc\.{name}\s*=\s*\[([^\]]*)\]"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"MATPOWER file has no literal mpc.{name} matrix")
    rows = [
        [float(value) for value in line.split()]
        for line in re.split(r"[;\n]", match[1])
        if line.strip()
    ]
    if not rows or not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"MATPOWER mpc.{name} is empty or contains nonfinite values")
    return rows


def _add_matpower_costs(data, name, generators):
    source = re.sub(r"%[^\n]*", "", _decode(data))
    gen = _matpower_rows(source, "gen")
    cost = _matpower_rows(source, "gencost")
    if len(gen) != len(generators) or len(cost) not in {len(gen), 2 * len(gen)}:
        raise ValueError("AUX and MATPOWER generator counts or cost counts differ")
    min_mismatches = []
    for i, (item, row, coeff) in enumerate(zip(generators, gen, cost), 1):
        if (
            len(row) < 10
            or row[0] != item["bus"]
            or row[7] != int(item["online"])
            or (abs(row[8] - item["pmax_mw"]) > 0.011)
        ):
            raise ValueError(
                f"MATPOWER generator row {i} does not align with "
                f"AUX generator {(item['bus'], item['id'])}"
            )
        if abs(row[9] - item["pmin_mw"]) > 0.011:
            min_mismatches.append((item["bus"], item["id"]))
        if len(coeff) < 4:
            raise ValueError(f"Invalid MATPOWER gencost row {i}")
        if coeff[0] != 2 or coeff[3] not in {1, 2, 3, 4}:
            warnings.warn(
                f"MATPOWER gencost row {i} has an unsupported cost model or degree"
            )
            continue
        n = int(coeff[3])
        if len(coeff) < 4 + n:
            raise ValueError(f"Truncated MATPOWER gencost row {i}")
        item["cost_c3_c2_c1_c0"] = [0.0] * (4 - n) + coeff[4 : 4 + n]
        item["cost_source"] = f"MATPOWER:{name}"
    if min_mismatches:
        warnings.warn(
            f"MATPOWER and AUX minimum MW differ for {len(min_mismatches)} "
            f"generator(s). Retaining AUX limits. First keys: {min_mismatches[:5]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Read a PowerWorld AUX case into JSON."
    )
    parser.add_argument("path", type=Path, help="PowerWorld AUX file or case ZIP")
    parser.add_argument(
        "--member", help="exact path to the network AUX member in the ZIP"
    )
    parser.add_argument("--base-mva", type=float, help="system MVA base override")
    parser.add_argument(
        "--matpower",
        nargs="?",
        const=True,
        metavar="FILE",
        help="import costs from companion .m inside ZIP or from FILE. "
        "Assumes matching generator order.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write normalized tables as JSON"
    )
    args = parser.parse_args()
    try:
        grid = load_grid(
            args.path,
            member=args.member,
            base_mva=args.base_mva,
            matpower=args.matpower,
        )
        if args.output:
            args.output.write_text(
                json.dumps(asdict(grid), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        online = [row for row in grid.generators if row["online"]]
        costs = sum(row["cost_c3_c2_c1_c0"] is not None for row in online)
        print(
            f"{grid.source}: {len(grid.buses)} buses, {len(grid.branches)} branches, "
            f"{len(grid.generators)} generators, {len(grid.loads)} loads, "
            f"{grid.base_mva:g} MVA base. Polynomial costs for "
            f"{costs}/{len(online)} online generators."
        )
    except (OSError, ValueError, BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
