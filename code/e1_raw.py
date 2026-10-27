import sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, designs
from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
from pseudoeis.signals import synthetic_drive_current, validation_current_profile
from pseudoeis import estimation as est
from sup_raw import fit_raw
cfg = ev.cfg
p = ECMTruthModel(parameters_from_prior_config("NMC", cfg)).effective_parameters(25.0, 0.90, 0.50)
R0, R, TAU = float(p.r0_ohm), np.asarray(p.rc_resistances_ohm, float), np.asarray(p.time_constants_s, float)
def truth_drop(i, dt): return est._simulate_drop(np.asarray(i, float), dt, R0, R, TAU, np.zeros(3), np.zeros(1))
VCUR = validation_current_profile(180.0, 0.05, 20.0, np.random.default_rng(20261001)); VTRUE = truth_drop(VCUR, 0.05)
q = np.load('seqs/s4786.npy'); D = designs.all_designs()
OUT = Path('sup/e1_raw.json'); rows = json.loads(OUT.read_text()) if OUT.exists() else []; done = {r['rep'] for r in rows}
for rep in range(5001, 5001 + int(sys.argv[1])):
    if rep in done: continue
    ip = synthetic_drive_current(265.23, 0.01, 20.0, np.random.default_rng(400_000 + rep), minimum_current_a=10.0)
    noise = np.random.default_rng(500_000 + rep).normal(0, 1e-3, 26523)
    i = 0.5 * (1 + q) * ip[:len(q)]; y = truth_drop(i, 0.01) + noise[:len(q)]
    t0 = time.time(); fit = fit_raw(i, y, D['s4786'], 600_000 + rep); sec = time.time() - t0
    rows.append(dict(rep=rep, rmse_mv=ev.rmse_mv(est.predict_polarization_drop(VCUR, 0.05, fit) - VTRUE), order=int(fit.order), sec=sec))
    OUT.write_text(json.dumps(rows)); print(rep, rows[-1], flush=True)
