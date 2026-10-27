import sys, json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce
TD = sys.argv[1]; CONDS = ('cold', 'nominal', 'hot', 'aged')
tags = sorted(p.name[:-len('.s4786.npz')] for p in Path(TD).glob('*.s4786.npz')); R = {}
for t in tags:
    cell = ev.load_cell(TD, t); Z = np.load(Path(TD) / f'{t}.s4786.npz'); drop = Z['ocv'] - Z['v_meas']; tgt = cell['target']['C1'] - cell['v_val']
    rows = []
    for k, i in [(0, Z['i_meas'])] + [(s, Z['i_true'] + np.random.default_rng(s).normal(0, .02, len(Z['i_true']))) for s in range(1, 6)]:
        m = ce.fit(i, drop, .01, offset=False, capacity_ah=60.0)
        rows.append(dict(seed=k, rmse=ev.rmse_mv(tgt - m.predict_drop(cell['vmeas'], .05)), r0=m.r0, rho=m.rho.tolist(), rss=m.r_inst_small_signal))
    R[t] = dict(cond=cell['meta']['cond'], fits=rows, scales=m.scales.tolist())
sp = {c: [max(f['rmse'] for f in R[t]['fits']) - min(f['rmse'] for f in R[t]['fits']) for t in tags if R[t]['cond'] == c] for c in CONDS}
print('RMSE spread over the 6 noise realizations per unit (mV): ' + '  '.join(f'{c} median {np.median(v):.4f} max {max(v):.3f}' for c, v in sp.items()))
sc = R[tags[0]]['scales']
for c in CONDS:
    ts = [t for t in tags if R[t]['cond'] == c]
    sd = lambda g: np.median([np.std([g(f) for f in R[t]['fits']]) for t in ts])
    print(f'{c:8s} median SD over realizations (micro-ohm): r0 {1e6*sd(lambda f: f["r0"]):7.1f}  ' + '  '.join(f'rho@{s:g}A {1e6*sd(lambda f, j=j: f["rho"][j]):7.1f}' for j, s in enumerate(sc)) + f'  r_ss {1e6*sd(lambda f: f["rss"]):6.1f}')
json.dump(dict(units=R, spread=sp), open('noise_resample.json', 'w'))
