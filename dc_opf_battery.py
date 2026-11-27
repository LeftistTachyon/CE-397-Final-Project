"""DC optimal power flow and optional battery siting for the PowerWorld workflow.

Requires numpy, scipy, and powerworld_aux.py.

    python dc_opf.py ACTIVSg500.zip profiles.npz -o dispatch.npz --summary hourly.csv
    python dc_opf.py ACTIVSg500.zip profiles.npz --battery battery.json -o dispatch.npz

The first argument may also be the JSON exported by powerworld_aux.py or a
standalone AUX file. With a case ZIP, --matpower loads companion generator
costs where needed (e.g., Hawaii40 and Memphis). --matpower FILE uses a
separate MATPOWER export. The ACTIVSg500 AUX already contains costs.

The baseline solves each hour independently, with lossless DC power balance,
fixed generator commitment, profile MW bounds, and branch rating A as a
proxy for the MW flow limit. Costs are in USD/hour. A sparse HiGHS LP
outer approximation refines convex quadratic generator costs until the
feasible cost and lower bound differ by at most --cost-tolerance.

NPZ result arrays are indexed (hour, device). Branch flows are positive
from branch_from_bus toward branch_to_bus. An optional summary CSV has
one row per hour. Failed hours have NaN dispatch and cost. The command
exits nonzero if any hour fails, after writing the results.

Battery mode also writes PREFIX_sites.csv and PREFIX_battery.npz for the site
search and the best solved battery, where PREFIX is the -o path without .npz.
Generation, charge, discharge, and flows have one row per interval. Stored
energy includes the initial value and all interval ends. Baseline LMPs
come from the final hourly LP approximations. Site wholesale reports use
conditional repricing with the battery schedule fixed.

The battery JSON accepts battery, dispatch, candidates, search and retail.
See parse_study() for fields and defaults. Site period_cost_usd is charged
once over the input horizon. The default battery relaxation permits
fractional use of the charge and discharge modes within an interval.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from powerworld_aux import Grid, load_grid


def read_case(path, member=None, matpower=None, base_mva=None):
    """Load a grid with the same importer used to build the profile IDs."""
    if path.suffix.lower() == ".json":
        if member or matpower is not None or base_mva is not None:
            raise ValueError(
                "For JSON, set AUX or MATPOWER options when exporting the grid."
            )
        with path.open(encoding="utf-8") as stream:
            return Grid(**json.load(stream))
    return load_grid(path, member=member, matpower=matpower, base_mva=base_mva)


def read_profiles(path, dc):
    """Validate profile dimensions and exact bus and generator column order."""
    fields = (
        "timestamp_utc",
        "bus_ids",
        "generator_bus_ids",
        "generator_ids",
        "demand_mw",
        "pmin_mw",
        "pmax_mw",
    )
    with np.load(path, allow_pickle=False) as archive:
        profiles = {field: archive[field] for field in fields}
        # The earlier Hawaii-only builder omitted this field and used one-hour steps.
        profiles["hours_per_step"] = (
            float(archive["hours_per_step"].item())
            if "hours_per_step" in archive
            else 1.0
        )
    keys = dc["generator_keys"]
    if not np.array_equal(profiles["bus_ids"], dc["bus_ids"]):
        raise ValueError(
            "Profile bus IDs or column order differ from the grid. "
            "Regenerate profiles from this case."
        )
    if not np.array_equal(
        profiles["generator_bus_ids"], [bus for bus, _ in keys]
    ) or not np.array_equal(profiles["generator_ids"], [unit for _, unit in keys]):
        raise ValueError(
            "Profile generator IDs or column order differ from the grid. "
            "Regenerate profiles from this case."
        )
    time = profiles["timestamp_utc"]
    if time.ndim != 1 or len(time) == 0 or len(np.unique(time)) != len(time):
        raise ValueError("Expected a nonempty list of unique hourly timestamps.")
    dt = profiles["hours_per_step"]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("hours_per_step must be finite and positive.")
    for field, width in (
        ("demand_mw", len(dc["bus_ids"])),
        ("pmin_mw", len(keys)),
        ("pmax_mw", len(keys)),
    ):
        values = np.asarray(profiles[field], dtype=float)
        if values.shape != (len(time), width) or not np.isfinite(values).all():
            raise ValueError(
                f"{field} must contain finite values with shape {(len(time), width)}."
            )
        profiles[field] = values
    invalid = np.argwhere(profiles["pmin_mw"] > profiles["pmax_mw"])
    if len(invalid):
        hour, generator = invalid[0]
        raise ValueError(
            f"Generator minimum exceeds maximum for {keys[generator]} at {time[hour]}."
        )
    return profiles


class DCOPF:
    """Reusable network constraints for independent dispatch intervals."""

    def __init__(self, dc):
        self.a = dc["incidence"]
        self.g = dc["generator_map"]
        self.nbranch, self.nbus = self.a.shape
        self.ngen = self.g.shape[1]
        self.shift = dc["shift_rad"]
        self.x = dc["mw_to_angle"]
        self.rating = dc["rate_a_mva"]
        self.rated = np.isfinite(self.rating) & (self.rating > 0)
        self.c2, self.c1, c0 = dc["cost_c2_c1_c0"].T
        missing = ~np.isfinite(dc["cost_c2_c1_c0"]).all(axis=1) | (self.c2 < 0)
        if missing.any():
            keys = [key for key, bad in zip(dc["generator_keys"], missing) if bad]
            raise ValueError(
                f"No supported convex quadratic cost for {len(keys)} online "
                f"generators. First keys: {keys[:8]}. "
                "Check case costs or supply --matpower if available."
            )
        self.constant_cost = float(c0.sum())
        self.quad = np.flatnonzero(self.c2 > 0)
        nq = len(self.quad)
        # Variables are theta, flow, generation, and quadratic cost epigraphs.
        self.gen_start = self.nbus + self.nbranch
        self.epigraph_start = self.gen_start + self.ngen
        self.objective = np.r_[np.zeros(self.gen_start), self.c1, np.ones(nq)]
        self.equality = sparse.bmat(
            [
                [
                    self.a,
                    -sparse.diags(self.x),
                    sparse.csr_matrix((self.nbranch, self.ngen)),
                    sparse.csr_matrix((self.nbranch, nq)),
                ],
                [
                    sparse.csr_matrix((self.nbus, self.nbus)),
                    self.a.T,
                    -self.g,
                    sparse.csr_matrix((self.nbus, nq)),
                ],
            ],
            format="csr",
        )
        angle_bounds = [(None, None)] * self.nbus
        for i in dc["reference_indices"]:
            angle_bounds[i] = (0.0, 0.0)
        branch_bounds = [
            (-rate, rate) if limit else (None, None)
            for rate, limit in zip(self.rating, self.rated)
        ]
        self.network_bounds = angle_bounds + branch_bounds

    def _tangents(self, points):
        """Form 2*c2*p0*pg - epigraph <= c2*p0**2 for each tangent point p0."""
        if not points.size:
            return None, None
        nq = len(self.quad)
        repetitions = len(points)
        rows = np.arange(points.size)
        slopes = (2 * self.c2[self.quad] * points).ravel()
        gen_cols = self.gen_start + np.tile(self.quad, repetitions)
        epi_cols = self.epigraph_start + np.tile(np.arange(nq), repetitions)
        matrix = sparse.coo_matrix(
            (
                np.r_[slopes, -np.ones(points.size)],
                (np.r_[rows, rows], np.r_[gen_cols, epi_cols]),
            ),
            shape=(points.size, len(self.objective)),
        ).tocsr()
        return matrix, (self.c2[self.quad] * points**2).ravel()

    def solve(self, demand, pmin, pmax, tolerance, max_refinements):
        if demand.sum() < pmin.sum() - 1e-7 or demand.sum() > pmax.sum() + 1e-7:
            return (
                "infeasible",
                "Total demand is outside aggregate generator bounds.",
                None,
            )
        rhs = np.r_[self.shift, -demand]
        bounds = (
            self.network_bounds + list(zip(pmin, pmax)) + [(0, None)] * len(self.quad)
        )
        points = np.linspace(pmin[self.quad], pmax[self.quad], num=9)
        best = None
        gap = 0.0
        for _ in range(max_refinements + 1):
            tangent_matrix, tangent_rhs = self._tangents(points)
            sol = linprog(
                self.objective,
                A_eq=self.equality,
                b_eq=rhs,
                A_ub=tangent_matrix,
                b_ub=tangent_rhs,
                bounds=bounds,
                method="highs",
            )
            if not sol.success:
                status = "infeasible" if sol.status == 2 else "solver_failed"
                return status, sol.message, None
            theta = sol.x[: self.nbus]
            flow = sol.x[self.nbus : self.gen_start]
            pg = sol.x[self.gen_start : self.epigraph_start]
            balance_error = np.max(
                np.abs(self.a.T @ flow - self.g @ pg + demand), initial=0.0
            )
            flow_error = np.max(
                np.abs(self.a @ theta - self.x * flow - self.shift), initial=0.0
            )
            limit_violation = max(
                np.max(np.abs(flow[self.rated]) - self.rating[self.rated], initial=0.0),
                np.max(pmin - pg, initial=0.0),
                np.max(pg - pmax, initial=0.0),
            )
            if balance_error > 1e-5 or limit_violation > 1e-5 or flow_error > 1e-6:
                return (
                    "solver_failed",
                    "Solution exceeds the network or device residual tolerance.",
                    None,
                )
            variable_cost = float(np.dot(self.c2, pg**2) + np.dot(self.c1, pg))
            if best is None or variable_cost < best[0]:
                best = (variable_cost, theta, flow, pg)
            gap = max(0.0, best[0] - sol.fun)
            if gap <= tolerance:
                return (
                    "optimal",
                    "",
                    dict(
                        theta=best[1],
                        flow=best[2],
                        pg=best[3],
                        cost=best[0] + self.constant_cost,
                        gap=gap,
                        lower_bound=sol.fun + self.constant_cost,
                        lmp=-sol.eqlin.marginals[self.nbranch :],
                    ),
                )
            points = np.vstack((points, pg[self.quad]))
        return (
            "cost_tolerance",
            f"Cost gap {gap:.6g} USD/hour exceeds target {tolerance:g}.",
            None,
        )


def solve_profiles(dc, profiles, tolerance=0.05, max_refinements=24):
    """Solve all intervals and return results with the source device IDs."""
    model = DCOPF(dc)
    time = profiles["timestamp_utc"]
    steps = len(time)
    pg = np.full((steps, model.ngen), np.nan)
    flow = np.full((steps, model.nbranch), np.nan)
    theta = np.full((steps, model.nbus), np.nan)
    cost = np.full(steps, np.nan)
    gap = np.full(steps, np.nan)
    lower_bound = np.full(steps, np.nan)
    lmp = np.full((steps, model.nbus), np.nan)
    status, messages = [], []
    demand = profiles["demand_mw"]
    for i in range(steps):
        state, message, result = model.solve(
            demand[i],
            profiles["pmin_mw"][i],
            profiles["pmax_mw"][i],
            tolerance,
            max_refinements,
        )
        status.append(state)
        messages.append(message)
        if result is not None:
            pg[i], flow[i], theta[i] = result["pg"], result["flow"], result["theta"]
            cost[i], gap[i] = result["cost"], result["gap"]
            lower_bound[i], lmp[i] = result["lower_bound"], result["lmp"]

    duration = profiles["hours_per_step"]
    served_mw = demand.sum(axis=1)
    # Positive finite rate A is modeled as an MW limit in the lossless DC approximation.
    loading = np.abs(flow[:, model.rated]) / model.rating[model.rated]
    binding = np.sum(loading >= 1 - 1e-5, axis=1)
    branch_keys = dc["branch_keys"]
    return dict(
        timestamp_utc=time,
        hours_per_step=np.array(duration),
        status=np.array(status),
        message=np.array(messages),
        bus_ids=dc["bus_ids"],
        generator_bus_ids=profiles["generator_bus_ids"],
        generator_ids=profiles["generator_ids"],
        branch_from_bus=np.array([a for a, _, _ in branch_keys], dtype=int),
        branch_to_bus=np.array([b for _, b, _ in branch_keys], dtype=int),
        branch_circuit=np.array([c for _, _, c in branch_keys], dtype=str),
        branch_rate_a_mva=model.rating,
        branch_flow_mw=flow,
        generation_mw=pg,
        bus_angle_rad=theta,
        demand_mw=demand,
        system_demand_mw=served_mw,
        production_cost_usd_per_h=cost,
        production_cost_usd=cost * duration,
        cost_gap_usd_per_h=gap,
        demand_mwh=served_mw * duration,
        cost_lower_bound_usd_per_h=lower_bound,
        lmp_usd_per_mwh=lmp,
        binding_branch_count=np.where(np.isfinite(cost), binding, -1),
    )


def write_summary(path, results):
    columns = [
        "timestamp_utc",
        "status",
        "system_demand_mw",
        "demand_mwh",
        "production_cost_usd",
        "cost_gap_usd_per_h",
        "binding_branch_count",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in zip(*(results[field] for field in columns)):
            writer.writerow(
                "" if isinstance(value, float) and not np.isfinite(value) else value
                for value in row
            )


@dataclass(frozen=True)
class Battery:
    energy_capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float
    standing_loss_fraction_per_hour: float
    auxiliary_load_mw: float
    degradation_usd_per_mwh_discharged: float
    exclusive_charge_discharge: bool


@dataclass(frozen=True)
class Site:
    bus: int
    max_charge_mw: float
    max_discharge_mw: float
    period_cost_usd: float = 0.0


@dataclass(frozen=True)
class BatteryStudy:
    battery: Battery
    sites: list[Site]
    max_full_solves: int | None
    ranking_tolerance_usd: float
    report_wholesale: bool
    pass_through_fraction: float | None
    retail_energy_kwh: float | None


def _section(data, fields, label):
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = data.keys() - set(fields.split())
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}.")
    return data


def _number(data, key, default=None, minimum=0.0, maximum=np.inf, positive=False):
    value = data.get(key, default)
    if (
        type(value) not in (int, float)
        or not np.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{key} must be a finite number between {minimum:g} and {maximum:g}."
        )
    if positive and value <= 0:
        raise ValueError(f"{key} must be positive.")
    return float(value)


def _flag(data, key, default=False):
    value = data.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be true or false.")
    return value


def parse_study(data, bus_index):
    """Validate JSON data from the CLI or a future API request.

    battery: required ratings and efficiencies; SOC bounds default to 0 and 1.
    dispatch: initial SOC defaults to 0.5; terminal SOC equals the initial SOC.
    candidates: list of bus numbers or objects with bus, max_charge_mw,
        max_discharge_mw, and period_cost_usd. Omit to use every grid bus.
    search: max_full_solves (default 20, null for unlimited),
        ranking_tolerance_usd (default 10), report_wholesale (default false).
    retail: pass_through_fraction and optional retail_energy_kwh.
    """
    root = _section(data, "battery dispatch candidates search retail", "study")
    b = _section(
        root.get("battery"),
        "energy_capacity_mwh max_charge_mw max_discharge_mw "
        "charge_efficiency discharge_efficiency min_soc_fraction max_soc_fraction "
        "standing_loss_fraction_per_hour auxiliary_load_mw "
        "degradation_usd_per_mwh_discharged",
        "battery",
    )
    dispatch = _section(
        root.get("dispatch", {}),
        "initial_soc_fraction terminal_soc_policy " "exclusive_charge_discharge",
        "dispatch",
    )
    if dispatch.get("terminal_soc_policy", "equal_initial") != "equal_initial":
        raise ValueError("terminal_soc_policy must be equal_initial.")
    minimum = _number(b, "min_soc_fraction", 0, maximum=1)
    maximum = _number(b, "max_soc_fraction", 1, maximum=1)
    if minimum >= maximum:
        raise ValueError("min_soc_fraction must be less than max_soc_fraction.")
    battery = Battery(
        energy_capacity_mwh=_number(b, "energy_capacity_mwh", positive=True),
        max_charge_mw=_number(b, "max_charge_mw", positive=True),
        max_discharge_mw=_number(b, "max_discharge_mw", positive=True),
        charge_efficiency=_number(b, "charge_efficiency", maximum=1, positive=True),
        discharge_efficiency=_number(
            b, "discharge_efficiency", maximum=1, positive=True
        ),
        min_soc_fraction=minimum,
        max_soc_fraction=maximum,
        initial_soc_fraction=_number(
            dispatch, "initial_soc_fraction", 0.5, minimum, maximum
        ),
        standing_loss_fraction_per_hour=_number(
            b, "standing_loss_fraction_per_hour", 0, maximum=1
        ),
        auxiliary_load_mw=_number(b, "auxiliary_load_mw", 0),
        degradation_usd_per_mwh_discharged=_number(
            b, "degradation_usd_per_mwh_discharged", 0
        ),
        exclusive_charge_discharge=_flag(dispatch, "exclusive_charge_discharge"),
    )
    if battery.standing_loss_fraction_per_hour == 1:
        raise ValueError("standing_loss_fraction_per_hour must be less than 1.")

    entries = root.get("candidates", list(map(int, bus_index)))
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "candidates must be a nonempty list of bus numbers or site objects."
        )
    sites, seen = [], set()
    for entry in entries:
        site = (
            {"bus": entry}
            if type(entry) is int
            else _section(
                entry, "bus max_charge_mw max_discharge_mw period_cost_usd", "candidate"
            )
        )
        bus = site.get("bus")
        if type(bus) is not int or bus not in bus_index or bus in seen:
            raise ValueError(f"Unknown, invalid, or duplicate candidate bus {bus!r}.")
        sites.append(
            Site(
                bus,
                min(
                    battery.max_charge_mw,
                    _number(site, "max_charge_mw", battery.max_charge_mw),
                ),
                min(
                    battery.max_discharge_mw,
                    _number(site, "max_discharge_mw", battery.max_discharge_mw),
                ),
                _number(site, "period_cost_usd", 0),
            )
        )
        seen.add(bus)
    search = _section(
        root.get("search", {}),
        "max_full_solves ranking_tolerance_usd " "report_wholesale",
        "search",
    )
    limit = search.get("max_full_solves", 20)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("max_full_solves must be a positive integer or null.")
    retail = _section(
        root.get("retail", {}), "pass_through_fraction retail_energy_kwh", "retail"
    )
    return BatteryStudy(
        battery,
        sites,
        limit,
        _number(search, "ranking_tolerance_usd", 10),
        _flag(search, "report_wholesale"),
        _number(retail, "pass_through_fraction", maximum=1) if retail else None,
        (
            _number(retail, "retail_energy_kwh", positive=True)
            if "retail_energy_kwh" in retail
            else None
        ),
    )


class BatteryDispatch:
    """Linear constraints on charge, discharge, and stored energy."""

    def __init__(self, battery, site, steps, dt, exclusive=False):
        self.steps = steps
        self.exclusive = exclusive
        self.battery = battery
        self.dt = dt
        ident = sparse.eye(steps, format="csr")
        retention = (1 - battery.standing_loss_fraction_per_hour) ** dt
        energy_change = sparse.diags([-retention, 1], [0, 1], shape=(steps, steps + 1))
        self.balance = sparse.hstack(
            [
                -battery.charge_efficiency * dt * ident,
                dt / battery.discharge_efficiency * ident,
                energy_change,
            ],
            format="csr",
        )
        self.injection = sparse.hstack(
            [-ident, ident, sparse.csr_matrix((steps, steps + 1))], format="csr"
        )
        self.objective = np.zeros(3 * steps + 1)
        self.objective[steps : 2 * steps] = (
            dt * battery.degradation_usd_per_mwh_discharged
        )
        capacity = battery.energy_capacity_mwh
        initial = capacity * battery.initial_soc_fraction
        energy_bounds = (
            capacity * battery.min_soc_fraction,
            capacity * battery.max_soc_fraction,
        )
        ch, dis = site.max_charge_mw, site.max_discharge_mw
        self.bounds = (
            [(0, ch)] * steps
            + [(0, dis)] * steps
            + [(initial, initial)]
            + [energy_bounds] * (steps - 1)
            + [(initial, initial)]
        )
        self.integrality = None
        if exclusive:
            self.balance = sparse.hstack(
                [self.balance, sparse.csr_matrix((steps, steps))]
            )
            self.injection = sparse.hstack(
                [self.injection, sparse.csr_matrix((steps, steps))]
            )
            self.inequality = sparse.bmat(
                [
                    [ident, None, sparse.csr_matrix((steps, steps + 1)), -ch * ident],
                    [None, ident, sparse.csr_matrix((steps, steps + 1)), dis * ident],
                ],
                format="csr",
            )
            self.inequality_rhs = np.r_[np.zeros(steps), np.full(steps, dis)]
            self.bounds += [(0, 1)] * steps
            self.objective = np.r_[self.objective, np.zeros(steps)]
            self.integrality = np.r_[
                np.zeros(3 * steps + 1, dtype=int), np.ones(steps, dtype=int)
            ]
        else:
            self.inequality = sparse.hstack(
                [
                    ident / ch if ch else 0 * ident,
                    ident / dis if dis else 0 * ident,
                    sparse.csr_matrix((steps, steps + 1)),
                ],
                format="csr",
            )
            self.inequality_rhs = np.ones(steps)

    def screen(self, prices):
        """Optimistic value at fixed baseline prices, using continuous operation."""
        result = linprog(
            self.objective - self.dt * (self.injection.T @ prices),
            A_eq=self.balance,
            b_eq=np.zeros(self.steps),
            A_ub=self.inequality,
            b_ub=self.inequality_rhs,
            bounds=self.bounds,
            method="highs",
        )
        if not result.success:
            return None, "infeasible" if result.status == 2 else "screen_failed"
        score = -result.fun - self.dt * self.battery.auxiliary_load_mw * prices.sum()
        return float(score), "screened"

    def schedule(self, point):
        t = self.steps
        return point[:t], point[t : 2 * t], point[2 * t : 3 * t + 1]


class HorizonOPF:
    """Sparse multi-interval solve using the baseline OPF's network and costs."""

    def __init__(self, model, profiles, baseline):
        self.model = model
        self.profiles = profiles
        self.steps = steps = len(profiles["timestamp_utc"])
        self.dt = profiles["hours_per_step"]
        self.width = len(model.objective)
        self.network_size = steps * self.width
        self.equality = sparse.kron(sparse.eye(steps), model.equality, format="csr")
        self.rhs = np.hstack(
            [
                np.broadcast_to(model.shift, (steps, model.nbranch)),
                -profiles["demand_mw"],
            ]
        ).ravel()
        self.bounds = [
            bound
            for low, high in zip(profiles["pmin_mw"], profiles["pmax_mw"])
            for bound in model.network_bounds
            + list(zip(low, high))
            + [(0, None)] * len(model.quad)
        ]
        self.objective = np.tile(model.objective, steps) * self.dt
        self.constant_cost = model.constant_cost * steps * self.dt
        self.initial_points = np.concatenate(
            [
                np.linspace(
                    profiles["pmin_mw"][:, model.quad],
                    profiles["pmax_mw"][:, model.quad],
                    5,
                ),
                baseline["generation_mw"][None, :, model.quad],
            ]
        )

    def _tangents(self, points):
        # points has axes (cut, interval, quadratic generator).
        model = self.model
        rows = np.arange(points.size)
        offset = np.arange(self.steps)[:, None] * self.width
        gen_cols = np.broadcast_to(
            offset + model.gen_start + model.quad, points.shape
        ).ravel()
        aux_cols = np.broadcast_to(
            offset + model.epigraph_start + np.arange(len(model.quad)), points.shape
        ).ravel()
        c2 = model.c2[model.quad]
        matrix = sparse.coo_matrix(
            (
                np.r_[(2 * c2 * points).ravel(), -np.ones(points.size)],
                (np.r_[rows, rows], np.r_[gen_cols, aux_cols]),
            ),
            shape=(points.size, self.network_size),
        ).tocsr()
        return matrix, (c2 * points**2).ravel()

    def solve(self, bus_index, battery, site, tolerance, max_refinements):
        model, steps = self.model, self.steps
        storage = BatteryDispatch(
            battery, site, steps, self.dt, exclusive=battery.exclusive_charge_discharge
        )
        rows = (
            np.arange(steps) * (model.nbranch + model.nbus) + model.nbranch + bus_index
        )
        connect = sparse.coo_matrix(
            (np.ones(steps), (rows, np.arange(steps))), shape=(len(self.rhs), steps)
        ).tocsr()
        a_eq = sparse.bmat(
            [[self.equality, -connect @ storage.injection], [None, storage.balance]],
            format="csr",
        )
        rhs = self.rhs.copy()
        rhs[rows] -= battery.auxiliary_load_mw
        rhs = np.r_[rhs, np.zeros(steps)]
        objective = np.r_[self.objective, storage.objective]
        bounds = self.bounds + storage.bounds
        lb, ub = np.array(bounds, dtype=float).T
        lb = np.where(np.isnan(lb), -np.inf, lb)
        ub = np.where(np.isnan(ub), np.inf, ub)
        options = {}
        if storage.exclusive:
            options["integrality"] = np.r_[
                np.zeros(self.network_size, dtype=int), storage.integrality
            ]
            options["options"] = {"mip_rel_gap": 0.0}

        points = self.initial_points
        best, lower = None, -np.inf
        for _ in range(max_refinements + 1):
            cuts, limits = self._tangents(points)
            a_ub = sparse.block_diag([cuts, storage.inequality], format="csr")
            result = linprog(
                objective,
                A_eq=a_eq,
                b_eq=rhs,
                A_ub=a_ub,
                b_ub=np.r_[limits, storage.inequality_rhs],
                bounds=bounds,
                method="highs",
                **options,
            )
            if not result.success:
                status = "infeasible" if result.status == 2 else "solver_failed"
                return status, result.message, best
            if np.max(np.abs(a_eq @ result.x - rhs), initial=0) > 1e-5:
                return (
                    "solver_failed",
                    "Network or storage balance residual exceeds tolerance.",
                    best,
                )
            violation = max(
                np.max(lb - result.x),
                np.max(result.x - ub),
                np.max(a_ub @ result.x - np.r_[limits, storage.inequality_rhs]),
            )
            if violation > 1e-5:
                return (
                    "solver_failed",
                    "Network or storage limit residual exceeds tolerance.",
                    best,
                )
            matrix = result.x[: self.network_size].reshape(steps, self.width)
            pg = matrix[:, model.gen_start : model.epigraph_start]
            point = result.x[self.network_size :]
            cost_hour = pg**2 @ model.c2 + pg @ model.c1 + model.constant_cost
            wear = float(storage.objective @ point)
            cost = float(cost_hour.sum() * self.dt + wear + site.period_cost_usd)
            lp_bound = (
                getattr(result, "mip_dual_bound", None)
                if storage.exclusive
                else result.fun
            )
            if lp_bound is None:
                lp_bound = (
                    result.fun
                )  # A presolved continuous optimum may omit the MIP field.
            lower = max(lower, lp_bound + self.constant_cost + site.period_cost_usd)
            if best is None or cost < best["total_cost_usd"]:
                charge, discharge, energy = storage.schedule(point)
                best = dict(
                    generation_mw=pg,
                    branch_flow_mw=matrix[:, model.nbus : model.gen_start],
                    bus_angle_rad=matrix[:, : model.nbus],
                    charge_mw=charge,
                    discharge_mw=discharge,
                    stored_energy_mwh=energy,
                    production_cost_usd_per_h=cost_hour,
                    production_cost_usd=float(cost_hour.sum() * self.dt),
                    battery_wear_usd=wear,
                    site_period_cost_usd=site.period_cost_usd,
                    total_cost_usd=cost,
                    simultaneous_intervals=int(
                        np.count_nonzero((charge > 1e-5) & (discharge > 1e-5))
                    ),
                )
            best["lower_bound_usd"] = min(lower, best["total_cost_usd"])
            best["cost_gap_usd"] = best["total_cost_usd"] - best["lower_bound_usd"]
            if best["cost_gap_usd"] <= tolerance:
                return "optimal", "", best
            points = np.concatenate([points, pg[None, :, model.quad]])
        return "cost_tolerance", "Site reached the cost refinement limit.", best


def verify_horizon(profiles):
    try:
        time = [
            datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            for t in profiles["timestamp_utc"]
        ]
        step = timedelta(hours=profiles["hours_per_step"])
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "Battery dispatch needs ISO timestamps and a valid step length."
        ) from exc
    if step <= timedelta(0) or any(b - a != step for a, b in zip(time, time[1:])):
        raise ValueError(
            "Battery dispatch needs consecutive timestamps matching hours_per_step."
        )


def _wholesale_payment(
    model, profiles, bus_index, net_injection, tolerance, refinements
):
    """Reprice the network dispatch with the battery schedule held fixed."""
    payment = 0.0
    for i, gross_load in enumerate(profiles["demand_mw"]):
        net_load = gross_load.copy()
        net_load[bus_index] -= net_injection[i]
        status, _, result = model.solve(
            net_load,
            profiles["pmin_mw"][i],
            profiles["pmax_mw"][i],
            tolerance,
            refinements,
        )
        if status != "optimal":
            return None
        payment += float(gross_load @ result["lmp"])
    return payment * profiles["hours_per_step"]


def search_sites(dc, profiles, baseline, study, tolerance=0.05, max_refinements=24):
    """Screen, solve, and rank sites for minimum production, wear, and site cost."""
    verify_horizon(profiles)
    if not np.all(baseline["status"] == "optimal"):
        raise ValueError(
            "Battery screening requires a solved baseline in every interval."
        )
    battery = study.battery
    dt = profiles["hours_per_step"]
    cost0 = float(baseline["production_cost_usd"].sum())
    lb0 = float(baseline["cost_lower_bound_usd_per_h"].sum() * dt)
    served_kwh = float(baseline["demand_mwh"].sum() * 1000)
    retail_kwh = study.retail_energy_kwh if study.retail_energy_kwh else served_kwh
    if study.pass_through_fraction is not None and retail_kwh <= 0:
        raise ValueError(
            "Retail pass-through needs positive retail sales or modeled served energy."
        )
    prices = baseline["lmp_usd_per_mwh"]
    payment0 = (
        float(np.sum(prices * profiles["demand_mw"]) * dt)
        if study.report_wholesale
        else None
    )
    rows, cache, pending = [], {}, []
    for site in study.sites:
        bus = dc["bus_index"][site.bus]
        trace = prices[:, bus]
        cache_key = (site.max_charge_mw, site.max_discharge_mw, trace.tobytes())
        if cache_key not in cache:
            battery_only = BatteryDispatch(battery, site, len(trace), dt)
            cache[cache_key] = battery_only.screen(trace)
        score, status = cache[cache_key]
        # A baseline LP dual defines a supporting affine lower bound for the
        # quadratic operating cost at any feasible demand shifted by storage.
        lower = lb0 - score + site.period_cost_usd if score is not None else None
        row = dict(
            bus=site.bus,
            status=status,
            screen_value_usd=score,
            lower_bound_usd=lower,
            site_period_cost_usd=site.period_cost_usd,
        )
        rows.append(row)
        if status != "infeasible":
            pending.append(
                (float("-inf") if lower is None else lower, site.bus, site, row)
            )

    pending.sort(key=lambda candidate: candidate[:2])
    model = DCOPF(dc)
    horizon = None
    best, attempts = None, 0
    horizon_tolerance = tolerance * len(prices) * dt
    numerical_margin = 1e-8 * max(1, abs(cost0))
    for lower, _, site, row in pending:
        if best is not None and lower >= (
            best["total_cost_usd"] - study.ranking_tolerance_usd + numerical_margin
        ):
            row["status"] = "pruned_by_bound"
            continue
        if study.max_full_solves is not None and attempts >= study.max_full_solves:
            row["status"] = "search_budget"
            continue
        if horizon is None:
            horizon = HorizonOPF(model, profiles, baseline)
        attempts += 1
        bus_index = dc["bus_index"][site.bus]
        status, message, solution = horizon.solve(
            bus_index,
            battery,
            site,
            horizon_tolerance,
            max_refinements,
        )
        row.update(status=status, message=message)
        if solution is None:
            continue
        total = solution["total_cost_usd"]
        row.update(
            lower_bound_usd=min(total, max(lower, solution["lower_bound_usd"])),
            production_cost_usd=solution["production_cost_usd"],
            battery_wear_usd=solution["battery_wear_usd"],
            total_cost_usd=total,
            net_savings_usd=cost0 - total,
            simultaneous_intervals=solution["simultaneous_intervals"],
            production_cost_usd_per_kwh=(
                solution["production_cost_usd"] / served_kwh if served_kwh > 0 else None
            ),
            total_cost_usd_per_kwh=total / served_kwh if served_kwh > 0 else None,
        )
        if study.pass_through_fraction is not None:
            row["estimated_bill_savings_usd_per_kwh"] = (
                study.pass_through_fraction * (cost0 - total) / retail_kwh
            )
        net = (
            solution["discharge_mw"] - solution["charge_mw"] - battery.auxiliary_load_mw
        )
        if study.report_wholesale:
            payment = _wholesale_payment(
                model, profiles, bus_index, net, tolerance, max_refinements
            )
            row["wholesale_load_payment_usd"] = payment
            row["wholesale_load_payment_savings_usd"] = (
                payment0 - payment if payment is not None else None
            )
        if best is None or total < best["total_cost_usd"]:
            best = dict(
                solution,
                selected_bus=site.bus,
                net_grid_injection_mw=net,
                site_cost_lower_bound_usd=row["lower_bound_usd"],
                baseline_production_cost_usd=cost0,
                net_savings_usd=cost0 - total,
                auxiliary_load_mw=battery.auxiliary_load_mw,
            )
            for field in (
                "estimated_bill_savings_usd_per_kwh",
                "wholesale_load_payment_usd",
                "wholesale_load_payment_savings_usd",
                "production_cost_usd_per_kwh",
                "total_cost_usd_per_kwh",
            ):
                if row.get(field) is not None:
                    best[field] = row[field]

    if best is not None:
        bounds = [r["lower_bound_usd"] for r in rows if r["status"] != "infeasible"]
        if all(value is not None for value in bounds):
            best["site_search_gap_usd"] = max(0.0, best["total_cost_usd"] - min(bounds))
        else:
            best["site_search_gap_usd"] = np.nan
        best["rank_certified_within_tolerance"] = (
            best["site_search_gap_usd"] <= study.ranking_tolerance_usd
        )
        best["screened_bus_count"] = len(rows)
        best["full_solve_count"] = attempts
        if study.pass_through_fraction is not None:
            best["retail_energy_kwh"] = retail_kwh
            best["pass_through_fraction"] = study.pass_through_fraction
        if payment0 is not None:
            best["baseline_wholesale_load_payment_usd"] = payment0
    rows.sort(
        key=lambda row: (
            row.get("total_cost_usd", np.inf),
            row["lower_bound_usd"] if row["lower_bound_usd"] is not None else -np.inf,
            row["bus"],
        )
    )
    return rows, best


def write_sites(path, rows):
    fields = [
        "bus",
        "status",
        "screen_value_usd",
        "lower_bound_usd",
        "production_cost_usd",
        "battery_wear_usd",
        "site_period_cost_usd",
        "total_cost_usd",
        "net_savings_usd",
        "production_cost_usd_per_kwh",
        "total_cost_usd_per_kwh",
        "estimated_bill_savings_usd_per_kwh",
        "wholesale_load_payment_usd",
        "wholesale_load_payment_savings_usd",
        "simultaneous_intervals",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "case", type=Path, help="PowerWorld ZIP, AUX, or exported grid JSON"
    )
    parser.add_argument("profiles", type=Path, help="input profiles.npz")
    parser.add_argument("-o", "--output", type=Path, default=Path("dispatch.npz"))
    parser.add_argument("--summary", type=Path, help="optional hourly summary CSV")
    parser.add_argument(
        "--battery", type=Path, help="optional JSON battery siting study"
    )
    parser.add_argument("--member", help="exact network AUX name within a ZIP")
    parser.add_argument(
        "--base-mva", type=float, help="override the system base when importing AUX"
    )
    parser.add_argument(
        "--matpower",
        nargs="?",
        const=True,
        metavar="FILE",
        help="import companion MATPOWER costs from ZIP or FILE",
    )
    parser.add_argument(
        "--cost-tolerance", type=float, default=0.05, metavar="USD_PER_H"
    )
    parser.add_argument("--max-refinements", type=int, default=24)
    args = parser.parse_args()
    if (
        not np.isfinite(args.cost_tolerance)
        or args.cost_tolerance <= 0
        or args.max_refinements < 0
    ):
        parser.error(
            "Use a positive finite cost tolerance and nonnegative maximum refinements."
        )
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    outputs = [args.output.resolve()]
    if args.summary is not None:
        outputs.append(args.summary.resolve())
    inputs = [args.case, args.profiles]
    if isinstance(args.matpower, str):
        inputs.append(Path(args.matpower))
    if args.battery is not None:
        inputs.append(args.battery)
        sites_path = args.output.with_name(args.output.stem + "_sites.csv")
        battery_path = args.output.with_name(args.output.stem + "_battery.npz")
        outputs.extend([sites_path.resolve(), battery_path.resolve()])
    if len(set(outputs)) != len(outputs) or any(
        path.resolve() in outputs for path in inputs
    ):
        parser.error("Choose distinct output paths away from the input files.")
    try:
        grid = read_case(args.case, args.member, args.matpower, args.base_mva)
        dc = grid.dc_inputs()
        profiles = read_profiles(args.profiles, dc)
        study = None
        if args.battery is not None:
            with args.battery.open(encoding="utf-8") as stream:
                study = parse_study(json.load(stream), dc["bus_index"])
            verify_horizon(profiles)
        results = solve_profiles(
            dc, profiles, args.cost_tolerance, args.max_refinements
        )
        np.savez_compressed(args.output, **results)
        if args.summary is not None:
            write_summary(args.summary, results)
    except (OSError, BadZipFile, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    solved = results["status"] == "optimal"
    print(
        f"{int(solved.sum())}/{len(solved)} hours solved. Dispatch saved to {args.output}."
    )
    if not solved.all():
        for index in np.flatnonzero(~solved)[:5]:
            print(
                f"{results['timestamp_utc'][index]} {results['status'][index]}: "
                f"{results['message'][index]}"
            )
        return 1
    total = results["production_cost_usd"].sum()
    energy = results["demand_mwh"].sum()
    print(f"Production cost: USD {total:,.2f}. Energy served: {energy:,.2f} MWh.")
    if energy > 0:
        print(f"Production cost per delivered kWh: USD {total / (1000 * energy):.5f}.")
    if study is not None:
        try:
            rows, best = search_sites(
                dc, profiles, results, study, args.cost_tolerance, args.max_refinements
            )
            write_sites(sites_path, rows)
            print(f"Site search saved to {sites_path}.")
            if best is None:
                print(
                    "No feasible battery schedule was obtained. See the site search statuses."
                )
                return 1
            labels = {
                key: results[key]
                for key in (
                    "timestamp_utc",
                    "hours_per_step",
                    "bus_ids",
                    "generator_bus_ids",
                    "generator_ids",
                    "branch_from_bus",
                    "branch_to_bus",
                    "branch_circuit",
                    "branch_rate_a_mva",
                )
            }
            np.savez_compressed(battery_path, **labels, **best)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(str(exc))
        print(
            f"Best solved bus: {best['selected_bus']}. "
            f"Net savings: USD {best['net_savings_usd']:,.2f}. "
            f"Full solves: {best['full_solve_count']}/{best['screened_bus_count']}."
        )
        if best["rank_certified_within_tolerance"]:
            print(
                f"Site ranking gap: USD {best['site_search_gap_usd']:,.2f} "
                f"(target USD {study.ranking_tolerance_usd:,.2f})."
            )
        else:
            print(
                f"Site ranking target unverified. "
                f"Bound gap in USD: {best['site_search_gap_usd']:,.2f}."
            )
        if best["simultaneous_intervals"]:
            print(
                f"Best relaxed schedule charges and discharges in "
                f"{best['simultaneous_intervals']} interval(s). "
                "Use dispatch.exclusive_charge_discharge for mutually exclusive operation."
            )
        print(f"Battery dispatch saved to {battery_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
