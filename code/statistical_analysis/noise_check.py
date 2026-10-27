import sys, json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce
TD, E2D = sys.argv[1], sys.argv[2]; CONDS = ('cold', 'nominal', 'hot', 'aged')
tags = sorted(p.name[:-len('.s4786.npz')] for p in Path(TD).glob('*.s4786.npz')); assert len(tags) == 96
R = {}
for t in tags:
    cell = ev.load_cell(TD, t); Z = np.load(Path(TD) / f'{t}.s4786.npz'); V = np.load(Path(TD) / f'{t}.val.npz')
    drop = Z['ocv'] - Z['v_meas']; tgt = cell['target']['C1'] - cell['v_val']; r = dict(cond=cell['meta']['cond'])
    for name, kw in (('cge', dict(capacity_ah=60.0)), ('lin', dict(scales=np.zeros(0)))):
        mA = ce.fit(Z['i_meas'], drop, .01, offset=False, **kw); mB = ce.fit(Z['i_true'], drop, .01, offset=False, **kw)
        r[name] = dict(A=ev.rmse_mv(tgt - mA.predict_drop(V['vmeas'], .05)), B=ev.rmse_mv(tgt - mB.predict_drop(V['vmeas'], .05)),
                       C=ev.rmse_mv(tgt - mB.predict_drop(V['vcur'], .05)), rss_A=mA.r_inst_small_signal, rss_B=mB.r_inst_small_signal)
    st = json.loads((Path(E2D) / f'{t}.json').read_text())['seq']['s4786']['C1']['v3']['rmse_mv']; assert abs(r['cge']['A'] - st) < 1e-9, (t, r['cge']['A'], st)
    R[t] = r
out = {}
for e in ('cge', 'lin'):
    for k in ('A', 'B', 'C'):
        out[f'{e}.{k}'] = {c: float(np.median([R[t][e][k] for t in tags if R[t]['cond'] == c])) for c in CONDS}
    out[f'{e}.maxunit.B-A'] = max(abs(R[t][e]['B'] - R[t][e]['A']) for t in tags); out[f'{e}.maxunit.C-A'] = max(abs(R[t][e]['C'] - R[t][e]['A']) for t in tags)
    out[f'{e}.maxrel.rss'] = max(abs(R[t][e]['rss_B'] / R[t][e]['rss_A'] - 1) for t in tags)
json.dump(dict(summary=out, units=R), open('noise_check.json', 'w'), indent=1)
for k, v in out.items(): print(k, {c: round(x, 4) for c, x in v.items()} if isinstance(v, dict) else '%.2e' % v)
