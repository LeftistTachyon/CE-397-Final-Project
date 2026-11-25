"""Hourly, no-battery DC optimal power flow for the PowerWorld profile workflow.

Requires numpy, scipy, and powerworld_aux.py.

    python dc_opf.py ACTIVSg500.zip profiles.npz -o dispatch.npz --summary hourly.csv

The first argument may also be the JSON exported by powerworld_aux.py or a
standalone AUX file. With a case ZIP, --matpower loads companion generator
costs where needed (e.g., Hawaii40 and Memphis). --matpower FILE uses a
separate MATPOWER export. The ACTIVSg500 AUX already contains costs.

The model solves each hour independently, with lossless DC power balance,
fixed generator commitment, profile MW bounds, and branch rating A as a
proxy for the MW flow limit. Costs are in USD/hour. A sparse HiGHS LP
outer approximation refines convex quadratic generator costs until the
feasible cost and lower bound differ by at most --cost-tolerance.

NPZ result arrays are indexed (hour, device). Branch flows are positive
from branch_from_bus toward branch_to_bus. An optional summary CSV has
one row per hour. Failed hours have NaN dispatch and cost. The command
exits nonzero if any hour fails, after writing the results.
"""
from __future__ import annotations

import argparse
import csv
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
            raise ValueError("For JSON, set AUX or MATPOWER options when exporting the grid.")
        with path.open(encoding="utf-8") as stream:
            return Grid(**json.load(stream))
    return load_grid(path, member=member, matpower=matpower, base_mva=base_mva)


def read_profiles(path, dc):
    """Validate profile dimensions and exact bus and generator column order."""
    fields = ("timestamp_utc", "bus_ids", "generator_bus_ids", "generator_ids",
              "demand_mw", "pmin_mw", "pmax_mw")
    with np.load(path, allow_pickle=False) as archive:
        profiles = {field: archive[field] for field in fields}
        # The earlier Hawaii-only builder omitted this field and used one-hour steps.
        profiles["hours_per_step"] = float(archive["hours_per_step"].item()) \
            if "hours_per_step" in archive else 1.0
    keys = dc["generator_keys"]
    if not np.array_equal(profiles["bus_ids"], dc["bus_ids"]):
        raise ValueError("Profile bus IDs or column order differ from the grid. "
                         "Regenerate profiles from this case.")
    if (not np.array_equal(profiles["generator_bus_ids"], [bus for bus, _ in keys])
            or not np.array_equal(profiles["generator_ids"], [unit for _, unit in keys])):
        raise ValueError("Profile generator IDs or column order differ from the grid. "
                         "Regenerate profiles from this case.")
    time = profiles["timestamp_utc"]
    if time.ndim != 1 or len(time) == 0 or len(np.unique(time)) != len(time):
        raise ValueError("Expected a nonempty list of unique hourly timestamps.")
    dt = profiles["hours_per_step"]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("hours_per_step must be finite and positive.")
    for field, width in (("demand_mw", len(dc["bus_ids"])), ("pmin_mw", len(keys)),
                         ("pmax_mw", len(keys))):
        values = np.asarray(profiles[field], dtype=float)
        if values.shape != (len(time), width) or not np.isfinite(values).all():
            raise ValueError(f"{field} must contain finite values with shape {(len(time), width)}.")
        profiles[field] = values
    invalid = np.argwhere(profiles["pmin_mw"] > profiles["pmax_mw"])
    if len(invalid):
        hour, generator = invalid[0]
        raise ValueError(f"Generator minimum exceeds maximum for {keys[generator]} at {time[hour]}.")
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
            raise ValueError(f"No supported convex quadratic cost for {len(keys)} online "
                             f"generators. First keys: {keys[:8]}. "
                             "Check case costs or supply --matpower if available.")
        self.constant_cost = float(c0.sum())
        self.quad = np.flatnonzero(self.c2 > 0)
        nq = len(self.quad)
        # Variables are theta, flow, generation, and quadratic cost epigraphs.
        self.gen_start = self.nbus + self.nbranch
        self.epigraph_start = self.gen_start + self.ngen
        self.objective = np.r_[np.zeros(self.gen_start), self.c1, np.ones(nq)]
        self.equality = sparse.bmat([
            [self.a, -sparse.diags(self.x), sparse.csr_matrix((self.nbranch, self.ngen)),
             sparse.csr_matrix((self.nbranch, nq))],
            [sparse.csr_matrix((self.nbus, self.nbus)), self.a.T, -self.g,
             sparse.csr_matrix((self.nbus, nq))],
        ], format="csr")
        angle_bounds = [(None, None)] * self.nbus
        for i in dc["reference_indices"]:
            angle_bounds[i] = (0.0, 0.0)
        branch_bounds = [(-rate, rate) if limit else (None, None)
                         for rate, limit in zip(self.rating, self.rated)]
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
            (np.r_[slopes, -np.ones(points.size)], (np.r_[rows, rows], np.r_[gen_cols, epi_cols])),
            shape=(points.size, len(self.objective)),
        ).tocsr()
        return matrix, (self.c2[self.quad] * points**2).ravel()

    def solve(self, demand, pmin, pmax, tolerance, max_refinements):
        if demand.sum() < pmin.sum() - 1e-7 or demand.sum() > pmax.sum() + 1e-7:
            return "infeasible", "Total demand is outside aggregate generator bounds.", None
        rhs = np.r_[self.shift, -demand]
        bounds = self.network_bounds + list(zip(pmin, pmax)) + [(0, None)] * len(self.quad)
        points = np.linspace(pmin[self.quad], pmax[self.quad], num=9)
        best = None
        for _ in range(max_refinements + 1):
            tangent_matrix, tangent_rhs = self._tangents(points)
            sol = linprog(self.objective, A_eq=self.equality, b_eq=rhs,
                          A_ub=tangent_matrix, b_ub=tangent_rhs, bounds=bounds, method="highs")
            if not sol.success:
                status = "infeasible" if sol.status == 2 else "solver_failed"
                return status, sol.message, None
            theta = sol.x[:self.nbus]
            flow = sol.x[self.nbus:self.gen_start]
            pg = sol.x[self.gen_start:self.epigraph_start]
            balance_error = np.max(np.abs(self.a.T @ flow - self.g @ pg + demand), initial=0.0)
            flow_error = np.max(np.abs(self.a @ theta - self.x * flow - self.shift), initial=0.0)
            limit_violation = max(
                np.max(np.abs(flow[self.rated]) - self.rating[self.rated], initial=0.0),
                np.max(pmin - pg, initial=0.0), np.max(pg - pmax, initial=0.0),
            )
            if balance_error > 1e-5 or limit_violation > 1e-5 or flow_error > 1e-6:
                return "solver_failed", "Solution exceeds the network or device residual tolerance.", None
            variable_cost = float(np.dot(self.c2, pg**2) + np.dot(self.c1, pg))
            if best is None or variable_cost < best[0]:
                best = (variable_cost, theta, flow, pg)
            gap = max(0.0, best[0] - sol.fun)
            if gap <= tolerance:
                return "optimal", "", dict(
                    theta=best[1], flow=best[2], pg=best[3],
                    cost=best[0] + self.constant_cost, gap=gap,
                )
            points = np.vstack((points, pg[self.quad]))
        return "cost_tolerance", f"Cost gap {gap:.6g} USD/hour exceeds target {tolerance:g}.", None


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
    status, messages = [], []
    demand = profiles["demand_mw"]
    for i in range(steps):
        state, message, result = model.solve(
            demand[i], profiles["pmin_mw"][i], profiles["pmax_mw"][i], tolerance, max_refinements,
        )
        status.append(state)
        messages.append(message)
        if result is not None:
            pg[i], flow[i], theta[i] = result["pg"], result["flow"], result["theta"]
            cost[i], gap[i] = result["cost"], result["gap"]

    duration = profiles["hours_per_step"]
    served_mw = demand.sum(axis=1)
    # Positive finite rate A is modeled as an MW limit in the lossless DC approximation.
    loading = np.abs(flow[:, model.rated]) / model.rating[model.rated]
    binding = np.sum(loading >= 1 - 1e-5, axis=1)
    branch_keys = dc["branch_keys"]
    return dict(
        timestamp_utc=time, hours_per_step=np.array(duration),
        status=np.array(status), message=np.array(messages),
        bus_ids=dc["bus_ids"], generator_bus_ids=profiles["generator_bus_ids"],
        generator_ids=profiles["generator_ids"],
        branch_from_bus=np.array([a for a, _, _ in branch_keys], dtype=int),
        branch_to_bus=np.array([b for _, b, _ in branch_keys], dtype=int),
        branch_circuit=np.array([c for _, _, c in branch_keys], dtype=str),
        branch_rate_a_mva=model.rating, branch_flow_mw=flow,
        generation_mw=pg, bus_angle_rad=theta,
        demand_mw=demand, system_demand_mw=served_mw,
        production_cost_usd_per_h=cost, production_cost_usd=cost * duration,
        cost_gap_usd_per_h=gap, demand_mwh=served_mw * duration,
        binding_branch_count=np.where(np.isfinite(cost), binding, -1),
    )


def write_summary(path, results):
    columns = ["timestamp_utc", "status", "system_demand_mw", "demand_mwh",
               "production_cost_usd", "cost_gap_usd_per_h", "binding_branch_count", "message"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in zip(*(results[field] for field in columns)):
            writer.writerow("" if isinstance(value, float) and not np.isfinite(value) else value
                            for value in row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="PowerWorld ZIP, AUX, or exported grid JSON")
    parser.add_argument("profiles", type=Path, help="input profiles.npz")
    parser.add_argument("-o", "--output", type=Path, default=Path("dispatch.npz"))
    parser.add_argument("--summary", type=Path, help="optional hourly summary CSV")
    parser.add_argument("--member", help="exact network AUX name within a ZIP")
    parser.add_argument("--base-mva", type=float, help="override the system base when importing AUX")
    parser.add_argument("--matpower", nargs="?", const=True, metavar="FILE",
                        help="import companion MATPOWER costs from ZIP or FILE")
    parser.add_argument("--cost-tolerance", type=float, default=0.05, metavar="USD_PER_H")
    parser.add_argument("--max-refinements", type=int, default=24)
    args = parser.parse_args()
    if not np.isfinite(args.cost_tolerance) or args.cost_tolerance <= 0 or args.max_refinements < 0:
        parser.error("Use a positive finite cost tolerance and nonnegative maximum refinements.")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    outputs = [args.output.resolve()]
    if args.summary is not None:
        outputs.append(args.summary.resolve())
    if len(set(outputs)) != len(outputs) or any(
            path.resolve() in outputs for path in (args.case, args.profiles)):
        parser.error("Choose distinct output paths away from the input files.")
    try:
        grid = read_case(args.case, args.member, args.matpower, args.base_mva)
        dc = grid.dc_inputs()
        profiles = read_profiles(args.profiles, dc)
        results = solve_profiles(dc, profiles, args.cost_tolerance, args.max_refinements)
        np.savez_compressed(args.output, **results)
        if args.summary is not None:
            write_summary(args.summary, results)
    except (OSError, BadZipFile, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))

    solved = results["status"] == "optimal"
    print(f"{int(solved.sum())}/{len(solved)} hours solved. Dispatch saved to {args.output}.")
    if not solved.all():
        for index in np.flatnonzero(~solved)[:5]:
            print(f"{results['timestamp_utc'][index]} {results['status'][index]}: "
                  f"{results['message'][index]}")
        return 1
    total = results["production_cost_usd"].sum()
    energy = results["demand_mwh"].sum()
    print(f"Production cost: USD {total:,.2f}. Energy served: {energy:,.2f} MWh.")
    if energy > 0:
        print(f"Production cost per delivered kWh: USD {total / (1000 * energy):.5f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
