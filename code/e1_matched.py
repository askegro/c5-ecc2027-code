import sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, designs, cvx_estimator as ce
from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
from pseudoeis.signals import synthetic_drive_current, validation_current_profile
from pseudoeis import estimation as est
from pseudoeis.estimation import FitSegment, _compress_segment

cfg = ev.cfg
p = ECMTruthModel(parameters_from_prior_config("NMC", cfg)).effective_parameters(25.0, 0.90, 0.50)
R0, R, TAU = float(p.r0_ohm), np.asarray(p.rc_resistances_ohm, float), np.asarray(p.time_constants_s, float)
assert np.allclose(R0, 0.000646875) and np.allclose(TAU, [0.014, 8.629, 52.308])


def truth_drop(i, dt):
    return est._simulate_drop(np.asarray(i, float), dt, R0, R, TAU, np.zeros(3), np.zeros(1))


VCUR = validation_current_profile(180.0, 0.05, 20.0, np.random.default_rng(20261001)); VTRUE = truth_drop(VCUR, 0.05)
SEQ = {n: np.load(f'seqs/{n}.npy') for n in ('s4786', 's9588', 'short74', 'passive')}
mode = sys.argv[1]
import os
out = Path(os.environ.get('V3_E1_OUT', 'results_e1')); out.mkdir(parents=True, exist_ok=True)
rows = json.loads((out / f'e1_{mode}.json').read_text()) if (out / f'e1_{mode}.json').exists() else []
done = {r['rep'] for r in rows}
reps = range(5001, 5201) if mode == 'v3' else range(5001, 5051)
D = designs.all_designs(); ident = cfg['identification']; vf = float(ident['blocked_validation_fraction'])
for rep in reps:
    if rep in done: continue
    ip = synthetic_drive_current(265.23, 0.01, 20.0, np.random.default_rng(400_000 + rep), minimum_current_a=10.0)
    noise = np.random.default_rng(500_000 + rep).normal(0, 1e-3, 26523)
    row = dict(rep=rep)
    for n, q in SEQ.items():
        i = 0.5 * (1 + q) * ip[:len(q)]; y = truth_drop(i, 0.01) + noise[:len(q)]
        if mode == 'v3':
            t0 = time.time(); m = ce.fit(i, y, 0.01, offset=False)
            row[n] = dict(rmse_mv=ev.rmse_mv(m.predict_drop(VCUR, 0.05) - VTRUE), kkt=m.kkt_violation, sec=time.time() - t0,
                          r_inst=m.r_inst_small_signal, n_active=m.n_active_branches)
        elif n in ('s4786', 's9588'):
            d = D[n]; segments, start = [], 0
            for segment in d.segments:
                stop = start + len(segment.q); x, yy = i[start:stop], y[start:stop]; start = stop
                target_dt = min(segment.bit_period_s / 4.0, max(d.dt_s, 0.25))
                if segment.bit_period_s <= 0.05: target_dt = d.dt_s
                n_train = min(max(8, int(np.floor((1.0 - vf) * len(x)))), len(x))
                xt, yt, dtc = _compress_segment(x[:n_train], yy[:n_train], d.dt_s, target_dt)
                xh, yh, _ = _compress_segment(x[n_train:], yy[n_train:], d.dt_s, target_dt)
                mask = np.zeros(len(xt) + len(xh), bool); mask[:len(xt)] = True
                segments.append(FitSegment(np.concatenate([xt, xh]), np.concatenate([yt, yh]), dtc, mask, segment.name))
            t0 = time.time(); fit, _ = est.select_model_order(segments, ident['candidate_rc_orders'], cfg, np.random.default_rng(600_000 + rep))
            row[n] = dict(rmse_mv=ev.rmse_mv(est.predict_polarization_drop(VCUR, 0.05, fit) - VTRUE), order=int(fit.order), sec=time.time() - t0,
                          refit='slow-branch' in fit.message)
    rows.append(row)
    (out / f'e1_{mode}.json').write_text(json.dumps(rows))
    if rep % 10 == 0: print(rep, {k: round(v['rmse_mv'], 4) for k, v in row.items() if k != 'rep'}, flush=True)
print('E1_DONE', mode)
