import json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce

TD = Path('traces_confirm')
VAR = {
    'default': {},
    'grid 4/decade': dict(taus=ce.tau_grid(per_decade=4)),
    'grid 12/decade': dict(taus=ce.tau_grid(per_decade=12)),
    'tau_max 500 s': dict(taus=ce.tau_grid(tau_max=500.0)),
    'tau_min 20 ms': dict(taus=ce.tau_grid(tau_min=0.02)),
    'scales 10, 20, 40 A': dict(scales=np.array([10.0, 20.0, 40.0])),
    'scales 2.5 ... 160 A': dict(scales=np.array([2.5, 5, 10, 20, 40, 80, 160.0])),
    'eps = 0': dict(eps=0.0),
    'eps = 1e-6': dict(eps=1e-6),
}
tags = sorted(f.name.split('.')[0] for f in TD.glob('*.zref.json'))
rows = {v: {} for v in VAR}
for tag in tags:
    cell = ev.load_cell(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell)
    for v, kw in VAR.items():
        kw = dict(kw); eps = kw.pop('eps', None); old = ce.EPS
        if eps is not None: ce.EPS = eps
        for arm in ('C1', 'C2'):
            m = ce.fit(seq['i'], seq['drop'][arm], .01, offset=(arm == 'C2'), **kw)
            rows[v][(tag, arm)] = (ev.rmse_mv(cell['target'][arm] - m.predict_drop(cell['vmeas'], .05) - cell['v_val']), m.kkt_violation)
        ce.EPS = old
out = {}
for v in VAR:
    o = dict(max_kkt=float(max(x[1] for x in rows[v].values())))
    for arm in ('C1', 'C2'):
        for cond in ('cold', 'nominal', 'hot', 'aged'):
            o[f'{cond}_{arm}_median'] = float(np.median([rows[v][(t, arm)][0] for t in tags if t.startswith(cond)]))
        o[f'{arm}_max_abs_change_vs_default'] = float(max(abs(rows[v][(t, arm)][0] - rows['default'][(t, arm)][0]) for t in tags))
    out[v] = o
Path('results_sensitivity.json').write_text(json.dumps(out, indent=1))
print(f"{'variant':22s} " + " ".join(f"{c[:3]}-{a}" for a in ('C1', 'C2') for c in ('cold', 'nominal', 'hot', 'aged')) + "  maxchg C1  maxchg C2  max KKT")
for v, o in out.items():
    print(f"{v:22s} " + " ".join(f"{o[f'{c}_{a}_median']:6.3f}" for a in ('C1', 'C2') for c in ('cold', 'nominal', 'hot', 'aged')) + f"  {o['C1_max_abs_change_vs_default']:8.3f}  {o['C2_max_abs_change_vs_default']:8.3f}  {o['max_kkt']:.1e}")
