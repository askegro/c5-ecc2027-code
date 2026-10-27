import os, sys, json, subprocess
from pathlib import Path
import numpy as np
import evallib as ev, designs, cvx_estimator as ce
variant, tag = sys.argv[1], sys.argv[2]
SEED_BASE = 9202610010
E2 = Path('traces_confirm'); TD = Path('traces_mm') / variant; OUT = Path('posthoc_mm') / variant; OUT.mkdir(parents=True, exist_ok=True)
f = OUT / f'{tag}.json'
if f.exists(): sys.exit(0)
cond, p, c = tag.split('_'); pack, ci = int(p[1:]), int(c[1:])
need = ['passive_eq'] if variant == 'base' else ['s4786', 'passive', 'passive_eq', 'val']
for name in need:
    if not (TD / f'{tag}.{name}.npz').exists():
        subprocess.run([sys.executable, '-B', 'sim_mm.py', variant, cond, str(pack), str(ci), name, str(TD)], check=True)
cell = ev.load_cell(E2 if variant == 'base' else TD, tag)
if variant != 'base':
    E = np.load(E2 / f'{tag}.val.npz'); assert np.allclose(E['vmeas'], cell['vmeas'])
out = dict(tag=tag, variant=variant, cond=cond, pack=pack, cell=ci, rec={}, linear={}, v2={})
recs = [('passive_eq', TD)] if variant == 'base' else [('s4786', TD), ('passive', TD), ('passive_eq', TD)]
for name, d in recs:
    seq = ev.load_seq(d, tag, name, cell); Z = np.load(d / f'{tag}.{name}.npz'); out['rec'][name] = {'rms_a': float(np.sqrt(np.mean(Z['i_true'] ** 2)))}
    for arm in ('C1', 'C2'):
        r = ev.eval_new(cell, seq, arm); out['rec'][name][arm] = dict(rmse_mv=r['rmse_mv'], train_rmse_mv=r['train_rmse_mv'], kkt=r['kkt'], b=r['b'])
        if name == 's4786' and variant != 'base':
            lin = ce.fit(seq['i'], seq['drop'][arm], .01, offset=(arm == 'C2'), scales=np.array([]))
            out['linear'][arm] = ev.rmse_mv(cell['target'][arm] - lin.predict_drop(cell['vmeas'], .05) - cell['v_val'])
if variant != 'base':
    D = designs.all_designs(); seq = ev.load_seq(TD, tag, 's4786', cell)
    for arm in ('C1', 'C2'):
        a = ev.eval_frozen(cell, seq, arm, D['s4786'], cell['meta']['seed'] + 50_000 + cell['meta']['cell'])
        out['v2'][arm] = dict(rmse_mv=a['rmse_mv'], order=a['order'], refit=a['refit'])
    V = np.load(TD / f'{tag}.val.npz'); out['val_ocv_range_mv'] = 1e3 * float(np.ptp(V['ocv_val']))
f.write_text(json.dumps(out, indent=1))
print(variant, tag, {k: ({a: round(v[a]['rmse_mv'], 3) for a in ('C1', 'C2')} if 'C1' in v else v) for k, v in out['rec'].items()}, flush=True)
