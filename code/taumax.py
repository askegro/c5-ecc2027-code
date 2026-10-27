import sys, json, glob, time, numpy as np
sys.path.insert(0, '.')
import evallib as ev, cvx_estimator as ce
E2D = sys.argv[1]; OUT = sys.argv[2]; TM = (50.0, 90.0, 265.0, 500.0, 2000.0)
tags = sorted(p.split('/')[-1][:-8] for p in glob.glob('traces_confirm/*.val.npz'))
try: res = json.load(open(OUT))
except Exception: res = {}
t0 = time.time()
for t in tags:
    if t in res: continue
    if time.time() - t0 > 150: break
    cell = ev.load_cell('traces_confirm', t); seq = ev.load_seq('traces_confirm', t, 's4786', cell)
    zr = json.load(open(f'{E2D}/{t}.json'))['zref']['z']; r = dict(cond=cell['meta']['cond'], pack=cell['meta']['pack'])
    for tm in TM:
        m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0, taus=ce.tau_grid(tau_max=tm))
        rm = ev.rmse_mv(cell['target']['C1'] - m.predict_drop(cell['vmeas'], .05) - cell['v_val'])
        z = {f: abs(complex(*zr[f]['drop_bulk']) - m.impedance([float(f)])[0]) / abs(complex(*zr[f]['drop_bulk'])) * 100 for f in ('0.008', '0.05', '1.0')}
        r[str(tm)] = dict(rmse=rm, z=z, nb=len(ce.tau_grid(tau_max=tm)))
    res[t] = r
json.dump(res, open(OUT, 'w')); print(len(res), 'of', len(tags))
