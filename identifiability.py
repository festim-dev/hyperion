"""Identifiability of the FLiBe transport parameters from steady-state flux.

The inversion in `para_swap_pure.py` prescribes the FLiBe diffusivity D and
recovers a single scalar per experimental point.  This script asks whether that
is a limitation: given only a steady-state downstream flux, can D and the Henry
solubility K_H be recovered separately, or is the permeability Phi = D * K_H the
only combination the measurement constrains?

Four tests, all run on the same 2-D axisymmetric FESTIM model and the same
experimental points as the production inversion:

  A. re-inversion    For each D multiplier in D_MULT, prescribe D and invert
                     K_H against the measured flux, then report Phi = D * K_H.
                     If Phi is invariant while K_H tracks 1/D, the two are
                     individually unidentifiable and Phi is the estimand.

  B. forward         Hold Phi at the value recovered at D_MULT == 1 and sweep D
                     over four decades.  Any drift of the modelled flux is
                     numerical, not physical, so this bounds the discretisation
                     error of the claim made by test A.

  C. landscape       Map the modelled flux over a 2-D grid in (D, K_H) around
                     the recovered point.  The misfit valley is the geometric
                     statement of what the data do and do not constrain.

  D. bypass floor    Make the salt effectively impermeable and re-solve.  The
                     residual downstream flux is the Ni-envelope bypass: the
                     part of the measured signal that never crossed the salt.
                     Its size sets how strongly an error in the measured flux is
                     amplified into an error in Phi, reported here as the
                     sensitivity dln(J)/dln(Phi).

Outputs (results/):
    identifiability_inversion.csv   test A, one row per (case, run, T, D mult)
    identifiability_forward.csv     test B
    identifiability_landscape.csv   test C
    logs/identifiability.log        progress log

Run from the repository root:  python identifiability.py
"""

from __future__ import annotations

import csv
import hashlib
import math
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

# ── Constants shared with the production pipeline ────────────────────────────

KB_EV = 8.617333262e-5

# Nominal FLiBe diffusivity (exp_data.D_flibe).  Test A perturbs the
# pre-exponential around this value; the activation energy is held fixed so that
# a multiplier scales D by the same factor at every temperature.
D_FLIBE_0 = 2.5e-7   # m^2/s
E_D_FLIBE = 0.24     # eV

D_MULT = [0.1, 0.3, 1.0, 3.0, 10.0]

# Wider sweep for the forward test, where no root find is needed.
D_MULT_FORWARD = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Grid for the misfit landscape, as multipliers on the recovered (D, K_H).
GRID_MULT = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# Phi that makes the salt effectively impermeable (7 decades below nominal).
PHI_BLOCKED = 1e6

# Root-find bracket, fixed in K_H so that it carries no dependence on the
# prescribed D (see _invert_KH), and a tolerance tight enough that the spread of
# the recovered Phi measures the solver rather than the bisection step.
K_H_LO, K_H_HI = 1e15, 1e23
TOL_DECADES = 1e-5

T2K = {Tc: Tc + 273.15 for Tc in (500.0, 550.0, 600.0, 650.0, 700.0)}
Y_FT_BY_TC = {500.0: 0.02914, 550.0: 0.02919, 600.0: 0.02925,
              650.0: 0.02930, 700.0: 0.02936}

# Points used for the landscape and forward tests: one low, one mid, one high
# temperature, for each boundary-condition envelope.
REPRESENTATIVE = [(500.0, "Run 1"), (600.0, "Run 1"), (700.0, "Run 1")]

OUTDIR = Path("results")
LOGDIR = OUTDIR / "logs"

# This box also runs the editor's WSL server and each worker peaks near 3.4 GB,
# so on a 15 GB machine the pool size is a memory budget, not a CPU one.  Six
# workers is fine on a dedicated box; the reruns here set HYPERION_WORKERS=1.
N_WORKERS = int(os.environ.get("HYPERION_WORKERS", "6"))
# A child is retired after this many tasks, which is what actually returns its
# memory.  At one worker, retiring after every task keeps the peak at one child.
MAX_TASKS_PER_CHILD = int(os.environ.get("HYPERION_MAXTASKS", "3"))


def D_at(T_K: float, mult: float = 1.0) -> float:
    """FLiBe diffusivity [m^2/s] at temperature T_K, scaled by `mult`."""
    return mult * D_FLIBE_0 * math.exp(-E_D_FLIBE / (KB_EV * T_K))


# ── Child-side solve ─────────────────────────────────────────────────────────
# Everything below `_flux` runs inside a spawned worker.  FESTIM allocates a new
# MPI communicator per mesh read, so the heavy imports stay inside the worker and
# each worker is retired after a few tasks (maxtasksperchild).

_NI_CACHE: dict[str, object] = {}


def _ni_solubility(bc_type: str):
    if bc_type not in _NI_CACHE:
        from para_swap_pure import _ni_solubility_for_bc
        _NI_CACHE[bc_type] = _ni_solubility_for_bc(bc_type)
    return _NI_CACHE[bc_type]


def _flux(point: dict, phi_T: float, d_mult: float) -> float:
    """Total downstream flux [H/s] for a prescribed (D, Phi) at this point.

    `phi_T` is the permeability at the point's temperature.  It is passed to the
    solver as a temperature-independent Permeability, so `make_materials` derives
    K_H = phi_T / D(T) -- the solver therefore sees exactly the (D, K_H) pair
    this call intends.
    """
    import h_transport_materials as htm
    from para_swap_pure import run_once

    D_obj = htm.Diffusivity(D_0=d_mult * D_FLIBE_0, E_D=E_D_FLIBE)
    perm = htm.Permeability(pre_exp=float(phi_T), act_energy=0.0, law="henry")

    out_bc = ({"type": "sieverts", "pressure": point["P_gb"]}
              if point["P_gb"] is not None else {"type": "particle_flux_zero"})

    return float(run_once(
        point["T_K"], point["P_up"], point["P_down"],
        D_obj, perm, _ni_solubility(point["bc_type"]),
        out_bc, point["y_ft"],
    ))


def _init_worker() -> None:
    from dolfinx.log import LogLevel, set_log_level
    set_log_level(LogLevel.WARNING)


# ── Test A: prescribe D, invert K_H ──────────────────────────────────────────


def _invert_KH(point: dict, d_mult: float) -> dict:
    """Bisect on log10(K_H) until the modelled flux matches the measured one.

    The bracket is fixed in K_H, deliberately.  An earlier version set it to
    [1e10, 1e15]/D and then multiplied by D again to form Phi, which cancelled:
    every prescribed D probed the *same* sequence of Phi, so the inversions could
    not disagree and the recovered Phi was bit-identical by construction rather
    than by result.  With the bracket fixed in K_H the probed Phi = D*K_H scales
    with D, so each D follows its own sequence of solves and branch decisions,
    and the agreement between the recovered Phi values is something the run has
    to earn.
    """
    D_T = D_at(point["T_K"], d_mult)
    target = float(point["J_exp"])
    n_solves = 0

    def J_of_logKH(log_KH: float) -> float:
        nonlocal n_solves
        n_solves += 1
        return _flux(point, D_T * 10.0 ** log_KH, d_mult)

    # K_H spans ~2e17 to 2.5e20 over the temperatures and D multipliers used
    # here; this bracket covers that with room to spare, and does not depend on D.
    #
    # The bracket is then jittered by a deterministic per-search amount.  A fixed
    # bracket is not enough on its own: bisection lands on a dyadic lattice, and a
    # D multiplier of exactly ten shifts the target by exactly one eighth of an
    # eight-decade bracket, which maps that lattice onto itself -- so the searches
    # would again agree for arithmetic reasons.  Jittering gives every search its
    # own lattice, and any agreement between them has to be earned.
    key = f"{point['case']}|{point['run']}|{point['T_C']}|{d_mult}".encode()
    u = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    lo = math.log10(K_H_LO) - 0.41 * u
    hi = math.log10(K_H_HI) + 0.29 * (1.0 - u)
    J_lo, J_hi = J_of_logKH(lo), J_of_logKH(hi)

    for _ in range(4):
        if (J_lo - target) * (J_hi - target) <= 0.0:
            break
        if J_hi < target:
            hi += 1.0
            J_hi = J_of_logKH(hi)
        else:
            lo -= 1.0
            J_lo = J_of_logKH(lo)
    else:
        return {"ok": False, "reason": "no bracket", "n_solves": n_solves}

    # Tight enough that the residual disagreement between prescribed D values
    # reflects the finite-element solve, not the quantisation of the root find.
    while hi - lo > TOL_DECADES:
        mid = 0.5 * (lo + hi)
        J_mid = J_of_logKH(mid)
        if (J_lo - target) * (J_mid - target) <= 0.0:
            hi, J_hi = mid, J_mid
        else:
            lo, J_lo = mid, J_mid

    log_KH = 0.5 * (lo + hi)
    K_H = 10.0 ** log_KH
    phi = D_T * K_H
    J_fit = J_of_logKH(log_KH)

    # Local sensitivity dln(J)/dln(Phi): how much of the measured flux actually
    # responds to the salt.  1 / this factor is the amplification of a relative
    # flux error into a relative Phi error.
    eps = 0.02
    J_p = _flux(point, phi * (1.0 + eps), d_mult)
    J_m = _flux(point, phi * (1.0 - eps), d_mult)
    n_solves += 2
    dlnJ_dlnPhi = (math.log(J_p) - math.log(J_m)) / (2.0 * math.log1p(eps))

    # Bypass floor: downstream flux with the salt made impermeable.
    J_floor = _flux(point, PHI_BLOCKED, d_mult)
    n_solves += 1

    return {
        "ok": True, "D_T": D_T, "K_H": K_H, "phi": phi,
        "J_fit": J_fit, "rel_err": J_fit / target - 1.0,
        "dlnJ_dlnPhi": dlnJ_dlnPhi, "J_floor": J_floor,
        "bypass_frac": J_floor / target, "n_solves": n_solves,
    }


def _task_invert(args) -> dict:
    point, d_mult = args
    t0 = time.time()
    res = _invert_KH(point, d_mult)
    row = {k: point[k] for k in ("case", "run", "T_C", "T_K", "J_exp")}
    row["D_mult"] = d_mult
    row.update(res)
    row["wall_s"] = time.time() - t0
    return row


def _task_forward(args) -> dict:
    point, d_mult, phi_ref = args
    t0 = time.time()
    J = _flux(point, phi_ref, d_mult)
    row = {k: point[k] for k in ("case", "run", "T_C", "T_K", "J_exp")}
    row.update({"D_mult": d_mult, "D_T": D_at(point["T_K"], d_mult),
                "phi_ref": phi_ref, "K_H": phi_ref / D_at(point["T_K"], d_mult),
                "J": J, "wall_s": time.time() - t0})
    return row


def _task_landscape(args) -> dict:
    point, a, b, D_ref, KH_ref = args
    t0 = time.time()
    # D_ref is D at the nominal multiplier, so scaling D by `a` is multiplier `a`.
    phi = (a * D_ref) * (b * KH_ref)
    J = _flux(point, phi, a)
    row = {k: point[k] for k in ("case", "run", "T_C", "T_K", "J_exp")}
    row.update({"a_D": a, "b_KH": b, "D_T": a * D_ref, "K_H": b * KH_ref,
                "phi": phi, "J": J, "rel_err": J / point["J_exp"] - 1.0,
                "wall_s": time.time() - t0})
    return row


# ── Experimental points ──────────────────────────────────────────────────────


def build_points() -> list[dict]:
    """The same SWAP points the production inversion uses, H runs only."""
    from exp_data import swap_infinite, swap_transparent, swap_flux_err

    tables = {"swap_infinite": swap_infinite, "swap_transparent": swap_transparent}
    pts = []
    for case, table in tables.items():
        for Tc, block in table.items():
            for run, cond in block["runs"].items():
                if run == "Run 3":       # D2 measurements, excluded from H fits
                    continue
                err = swap_flux_err[case].get(float(Tc), {}).get("runs", {}).get(run)
                pts.append({
                    "case": case, "run": run, "T_C": float(Tc),
                    "T_K": T2K[float(Tc)], "P_up": float(cond["P_up"]),
                    "P_down": float(cond["P_down"]),
                    "P_gb": float(cond["P_gb"]) if "P_gb" in cond else None,
                    "bc_type": ("sieverts" if "P_gb" in cond
                                else "particle_flux_zero"),
                    "y_ft": float(f"{Y_FT_BY_TC[float(Tc)]:.5f}"),
                    "J_exp": float(cond["J_exp"]),
                    # stored k=2, convert to 1-sigma
                    "sigma_J": (float(err) / 2.0) if err else float("nan"),
                })
    pts.sort(key=lambda p: (p["case"], p["T_C"], p["run"]))
    return pts


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path}  ({len(rows)} rows)", flush=True)


def _run(pool, fn, tasks, label, key_of_task, key_of_row):
    """Run `tasks`, checkpointing each result so an interruption costs one task.

    Results are appended to a JSON-lines file as they arrive -- JSON rather than
    CSV because the round trip has to preserve types, and the rows carry bools
    and NaNs.  On a restart the finished tasks are read back and skipped, so the
    study survives the machine going down mid-run.
    """
    stage = label.split("/")[0].lower()
    partial = LOGDIR / f"partial_{stage}.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)

    cached = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cached[tuple(key_of_row(r))] = r

    wanted = [tuple(key_of_task(t)) for t in tasks]
    todo = [t for t, k in zip(tasks, wanted) if k not in cached]
    rows = [cached[k] for k in wanted if k in cached]

    print(f"\n=== {label}: {len(tasks)} tasks, {len(rows)} cached, "
          f"{len(todo)} to run ===", flush=True)
    t0 = time.time()
    with partial.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(fn, todo), 1):
            rows.append(row)
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            print(f"  [{i}/{len(todo)}] {label} done "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    return rows


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    points = build_points()
    print(f"{len(points)} experimental points "
          f"({len({p['case'] for p in points})} BC envelopes)", flush=True)

    ctx = mp.get_context("spawn")
    print(f"pool: {N_WORKERS} worker(s), child retired every "
          f"{MAX_TASKS_PER_CHILD} task(s)", flush=True)
    with ctx.Pool(N_WORKERS, initializer=_init_worker,
                  maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        # -- Test A ----------------------------------------------------------
        tasks_a = [(p, m) for p in points for m in D_MULT]
        rows_a = _run(pool, _task_invert, tasks_a, "A/re-inversion",
                      lambda t: (t[0]["case"], t[0]["run"], t[0]["T_C"], t[1]),
                      lambda r: (r["case"], r["run"], r["T_C"], r["D_mult"]))
        rows_a.sort(key=lambda r: (r["case"], r["T_C"], r["run"], r["D_mult"]))
        _write_csv(OUTDIR / "identifiability_inversion.csv", rows_a)

        # Reference (D, K_H, Phi) at the nominal diffusivity, per point.
        ref = {(r["case"], r["run"], r["T_C"]): r
               for r in rows_a if r["D_mult"] == 1.0 and r.get("ok")}

        rep = [p for p in points if (p["T_C"], p["run"]) in REPRESENTATIVE]

        # -- Test B ----------------------------------------------------------
        tasks_b = [(p, m, ref[(p["case"], p["run"], p["T_C"])]["phi"])
                   for p in rep for m in D_MULT_FORWARD
                   if (p["case"], p["run"], p["T_C"]) in ref]
        rows_b = _run(pool, _task_forward, tasks_b, "B/forward",
                      lambda t: (t[0]["case"], t[0]["run"], t[0]["T_C"], t[1]),
                      lambda r: (r["case"], r["run"], r["T_C"], r["D_mult"]))
        rows_b.sort(key=lambda r: (r["case"], r["T_C"], r["run"], r["D_mult"]))
        _write_csv(OUTDIR / "identifiability_forward.csv", rows_b)

        # -- Test C ----------------------------------------------------------
        tasks_c = []
        for p in rep:
            key = (p["case"], p["run"], p["T_C"])
            if key not in ref or p["T_C"] != 600.0:
                continue          # landscape on the mid-range point only
            r = ref[key]
            for a in GRID_MULT:
                for b in GRID_MULT:
                    tasks_c.append((p, a, b, r["D_T"], r["K_H"]))
        rows_c = _run(pool, _task_landscape, tasks_c, "C/landscape",
                      lambda t: (t[0]["case"], t[0]["run"], t[0]["T_C"],
                                 t[1], t[2]),
                      lambda r: (r["case"], r["run"], r["T_C"],
                                 r["a_D"], r["b_KH"]))
        rows_c.sort(key=lambda r: (r["case"], r["a_D"], r["b_KH"]))
        _write_csv(OUTDIR / "identifiability_landscape.csv", rows_c)

    summarise(rows_a)


def summarise(rows_a: list[dict]) -> None:
    """Print the headline number: spread of Phi across the D sweep."""
    print("\n" + "=" * 78)
    print("TEST A -- Phi = D * K_H recovered at each prescribed D")
    print("=" * 78)
    by_point: dict[tuple, list[dict]] = {}
    for r in rows_a:
        if r.get("ok"):
            by_point.setdefault((r["case"], r["run"], r["T_C"]), []).append(r)

    worst = 0.0
    for key in sorted(by_point):
        group = sorted(by_point[key], key=lambda r: r["D_mult"])
        phis = np.array([g["phi"] for g in group])
        spread = phis.max() / phis.min() - 1.0
        worst = max(worst, spread)
        case, run, Tc = key
        print(f"\n{case} | {run} | {Tc:.0f} C   "
              f"bypass={group[0]['bypass_frac'] * 100:.1f}% of J_exp   "
              f"dlnJ/dlnPhi={group[0]['dlnJ_dlnPhi']:.3f}")
        print(f"  {'D/D_0':>7} {'D [m2/s]':>11} {'K_H':>12} {'Phi':>12} "
              f"{'Phi/Phi_ref-1':>14}")
        ref_phi = next((g["phi"] for g in group if g["D_mult"] == 1.0),
                       float(np.median(phis)))
        for g in group:
            print(f"  {g['D_mult']:>7.2f} {g['D_T']:>11.3e} {g['K_H']:>12.4e} "
                  f"{g['phi']:>12.5e} {g['phi'] / ref_phi - 1.0:>13.2e}")
        print(f"  -> Phi varies by {spread * 100:.3f}% over a "
              f"{max(D_MULT) / min(D_MULT):.0f}x range in D")

    print(f"\nWorst-case Phi spread across all points: {worst * 100:.3f}%")


if __name__ == "__main__":
    main()
