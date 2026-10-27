import os, sys, json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce, ms_estimator as ms
N_DRIVE = 36000; TAU_END = 265.0
def load_cell_wltc(trace_dir, tag):
    V = np.load(Path(trace_dir) / f"{tag}.wltc.npz"); meta = json.loads(str(V['meta'])); seed, ci = meta['seed'], meta['cell']
    cap = ev.bms.capacity_ah(meta['soh0']) * 3600.0
    z0_val = float(np.clip(meta['soc0'] + np.random.default_rng(seed + 9_000_003 + 10_007 * ci).normal(0.0, ev.SOC_BIAS_SD), 0, 1))
    z0_id = float(np.clip(meta['soc0'] + np.random.default_rng(seed + 303 * (ci + 1)).normal(0.0, ev.SOC_BIAS_SD), 0, 1))
    return dict(meta=meta, cap=cap, vmeas=V['vmeas'], v_val=V['v_val_truth'], z0_id=z0_id, z0_val=z0_val,
                target=dict(C1=V['ocv_val'], C2=ev.bms.ocv(ev.coulomb(z0_val, V['vmeas'], .05, cap)), C2p=ev.bms.ocv(ev.coulomb(z0_id, V['vmeas'], .05, cap))))
tag = sys.argv[1]; TD = Path(os.environ['TRACES']); ROOT = Path(os.environ['NEWOUT'])
cell = load_cell_wltc(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell)
o = json.loads(str(np.load(TD / f'{tag}.s4786.npz')['meta']))['orientation']; assert o == (1 if cell['meta']['cell'] == 0 else -1), f'{tag}: orientation {o}'
rec_len = len(seq['i']) * 0.01; assert abs(rec_len - TAU_END) < 0.5, rec_len
def score(pred): e = cell['target']['C1'] - pred - cell['v_val']; return dict(all=ev.rmse_mv(e), drive=ev.rmse_mv(e[:N_DRIVE]), rest=ev.rmse_mv(e[N_DRIVE:]))
r = dict(tag=tag, cond=cell['meta']['cond'], pack=cell['meta']['pack'], cell=cell['meta']['cell'], record_s=rec_len)
for name, tm, sc in (('cge|265', TAU_END, None), ('lin|265', TAU_END, np.zeros(0)), ('cge|2000', 2000.0, None)):
    m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0, taus=ce.tau_grid(tau_max=tm), scales=sc)
    r[name] = dict(score(m.predict_drop(cell['vmeas'], .05)), kkt=m.kkt_violation)
m = ms.fit(seq['i'], seq['drop']['C1'], .01, ce.tau_grid(tau_max=TAU_END), 1.0, cdiff=False, offset=False)
r['drt|265'] = score(m.predict(cell['vmeas'], .05))
(ROOT / 'g265').mkdir(parents=True, exist_ok=True); (ROOT / 'g265' / f'{tag}.json').write_text(json.dumps(r))
print('g265', tag, {k: round(v['all'], 3) for k, v in r.items() if '|' in k}, flush=True)
