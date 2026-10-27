import json, numpy as np
from pathlib import Path
import evallib as ev
from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
from pseudoeis.signals import synthetic_drive_current, validation_current_profile
from pseudoeis import estimation as est
p = ECMTruthModel(parameters_from_prior_config('NMC', ev.cfg)).effective_parameters(25.0, 0.90, 0.50)
R0, R, TAU = float(p.r0_ohm), np.asarray(p.rc_resistances_ohm, float), np.asarray(p.time_constants_s, float)
td = lambda i, dt: est._simulate_drop(np.asarray(i, float), dt, R0, R, TAU, np.zeros(3), np.zeros(1))
meta = dict(cond='smoke', pack=0, cell=0, seed=1, soc0=0.5, temp0=25.0, soh0=1.0, orientation=1); cap = ev.bms.capacity_ah(1.0) * 3600.0
vrng = np.random.default_rng(1 + 7_000_001); vcur = validation_current_profile(180.0, .05, 20.0, vrng); vnoise = vrng.normal(0.0, .02, len(vcur))
soc_val = ev.coulomb(0.5, vcur, .05, cap); ocv_val = ev.bms.ocv(soc_val)
out = Path('traces_smoke'); out.mkdir(exist_ok=True)
np.savez(out / 'smoke_p00_c0.val.npz', vcur=vcur, vmeas=vcur + vnoise, v_val_truth=ocv_val - td(vcur, .05), ocv_val=ocv_val, soc_val=soc_val, meta=json.dumps(meta))
q = np.load('seqs/s4786.npy'); rng = np.random.default_rng(77)
ip = synthetic_drive_current(265.23, 0.01, 20.0, rng, minimum_current_a=10.0); i = 0.5 * (1 + q) * ip[:len(q)]
soc = ev.coulomb(0.5, i, .01, cap); ocv = ev.bms.ocv(soc); v = ocv - td(i, .01)
np.savez(out / 'smoke_p00_c0.s4786.npz', i_true=i, i_meas=i + rng.normal(0, .02, len(i)), v_true=v, v_meas=v + rng.normal(0, 5e-4, len(i)), ocv=ocv, soc=soc, q=q, meta=json.dumps(meta))
print('smoke traces written', R0, R, TAU)
