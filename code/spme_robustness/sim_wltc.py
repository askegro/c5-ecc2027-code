import sys, json, time
from pathlib import Path
import numpy as np
import simlib
PEAK_A, DT, REST_S = 28.6, 0.05, 600.0
CHUNK = 2000
def wltc_current():
    d = np.loadtxt(Path(__file__).resolve().parent / 'WLTC.csv', delimiter=',', comments='#')
    t, p = d[:, 0], d[:, 1]
    assert abs(t[1] - t[0] - 0.1) < 1e-9 and abs(t[-1] - 1800.0) < 1e-9
    tg = np.arange(0.0, 1800.0, DT)
    idx = np.minimum(10 * np.floor(tg + 1e-9).astype(int), len(p) - 1)
    cur = p[idx] / p[::10].max() * PEAK_A
    return np.concatenate([cur, np.zeros(int(round(REST_S / DT)))])

def simulate_light(cell, current_a, dt_s, soc0, temperature_c, soh):
    import pybamm, campaign_v4_backend_n6 as B
    current = np.asarray(current_a, dtype=float); n = len(current); t_eval = np.arange(n) * float(dt_s)
    p = cell._params_for_state(temperature_c, soh)
    model = pybamm.lithium_ion.SPMe(options=B._sim_options())
    z0 = float(np.clip(soc0, 0.03, 0.97)); cap = cell.capacity_ah(soh)
    V, Q, O = [], [], []; prev = None
    pieces = []
    for start, end, is_fast in B._classify_fast_regions(current, dt_s):
        dt_max = dt_s if is_fast else min(0.25, max(dt_s, t_eval[end - 1] - t_eval[start]))
        pieces += [(a, min(a + CHUNK, end), dt_max) for a in range(start, end, CHUNK)]
    for idx, (start, end, dt_max) in enumerate(pieces):
        seg = t_eval[start:end]
        solver = pybamm.CasadiSolver(mode="safe", dt_max=dt_max)
        lo, hi = max(0, start - 1), min(n, end + 1)
        p["Current function [A]"] = B._zero_order_hold_interpolant(t_eval[lo:hi], current[lo:hi])
        sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
        if idx == 0:
            sol = sim.solve(t_eval=seg, initial_soc=z0)
        else:
            rel = np.concatenate(([0.0], seg - prev.t[-1]))
            sol = sim.step(dt=float(rel[-1]), solver=solver, t_eval=rel, starting_solution=prev)
        if sol.termination != 'final time' or not np.allclose(sol.t[-len(seg):], seg, rtol=0, atol=1e-9):
            raise RuntimeError('continuation did not reach the requested times')
        V.append(np.asarray(sol["Terminal voltage [V]"].data, float)[-len(seg):]); Q.append(np.asarray(sol["Discharge capacity [A.h]"].data, float)[-len(seg):])
        O.append(np.asarray(sol["Bulk open-circuit voltage [V]"].data, float)[-len(seg):])
        prev = sol.last_state
    v, q, o = np.concatenate(V), np.concatenate(Q), np.concatenate(O)
    return {"voltage_v": v, "soc": np.clip(z0 - q / cap, 0.0, 1.0), "ocv_bulk_v": o}

if __name__ == '__main__':
    seed_base, cond, pack_id, ci, outdir = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    fn = Path(outdir) / f"{cond}_p{pack_id:02d}_c{ci}.wltc.npz"; fn.parent.mkdir(parents=True, exist_ok=True)
    if fn.exists(): sys.exit(0)
    cfg = simlib.load_cfg()
    seed, pack, soc0, temp0, soh0 = simlib.pack_state(seed_base, cond, pack_id, cfg)
    meta = dict(spread=simlib.SPREAD['name'], profile='wltc', cond=cond, pack=pack_id, cell=ci, seed=seed, soc0=float(soc0[ci]), temp0=float(temp0[ci]), soh0=float(soh0[ci]))
    t0 = time.time()
    vcur = wltc_current(); vmeas = vcur + np.random.default_rng(seed + 9_000_001 + 10_007 * ci).normal(0.0, .02, len(vcur))
    vt = simulate_light(pack[ci], vcur, DT, soc0[ci], temp0[ci], soh0[ci])
    np.savez_compressed(fn, vcur=vcur, vmeas=vmeas, v_val_truth=vt["voltage_v"], ocv_val=vt["ocv_bulk_v"], soc_val=vt["soc"], meta=json.dumps(meta))
    print(fn.name, f"{time.time() - t0:.1f}s", flush=True)
