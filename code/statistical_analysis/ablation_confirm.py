import json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce

TD = Path('traces_confirm'); RC = Path('results_confirm')
rows = []
for tag in sorted(f.name.split('.')[0] for f in TD.glob('*.zref.json')):
    cell = ev.load_cell(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell); st = json.loads((RC / f'{tag}.json').read_text())
    for arm in ('C1', 'C2'):
        lin = ce.fit(seq['i'], seq['drop'][arm], .01, offset=(arm == 'C2'), scales=np.array([]))
        e_lin = ev.rmse_mv(cell['target'][arm] - lin.predict_drop(cell['vmeas'], .05) - cell['v_val'])
        r = dict(tag=tag, cond=cell['meta']['cond'], pack=cell['meta']['pack'], arm=arm, v2=st['seq']['s4786'][arm]['v2']['rmse_mv'],
                 linear=e_lin, full=st['seq']['s4786'][arm]['v3']['rmse_mv'])
        if arm == 'C2':
            r['linear_persist'] = ev.rmse_mv(cell['target']['C2p'] - lin.predict_drop(cell['vmeas'], .05) - cell['v_val'])
            r['linear_offset_discarded'] = ev.rmse_mv(cell['target'][arm] - (lin.predict_drop(cell['vmeas'], .05) - lin.b) - cell['v_val'])
            r['full_offset_discarded'] = st['seq']['s4786'][arm]['v3']['rmse_nooffset_pred_mv']
        rows.append(r)
out = {}
for cond in ('cold', 'nominal', 'hot', 'aged'):
    for arm in ('C1', 'C2'):
        sub = [r for r in rows if r['cond'] == cond and r['arm'] == arm]
        o = {k: float(np.median([r[k] for r in sub])) for k in sub[0] if k not in ('tag', 'cond', 'pack', 'arm')}
        o['n_linear_below_v2'] = int(sum(r['linear'] < r['v2'] for r in sub)); o['n_full_below_linear'] = int(sum(r['full'] < r['linear'] for r in sub)); o['n'] = len(sub)
        out[f'{cond}_{arm}'] = o
Path('results_ablation.json').write_text(json.dumps(out, indent=1))
for k, o in out.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in o.items()})
