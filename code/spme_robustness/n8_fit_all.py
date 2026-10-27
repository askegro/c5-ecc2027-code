import os, sys, json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce
N_DRIVE = 36000
def load_cell_wltc(trace_dir, tag):
    V = np.load(Path(trace_dir) / f"{tag}.wltc.npz"); meta = json.loads(str(V['meta'])); seed, ci = meta['seed'], meta['cell']
    cap = ev.bms.capacity_ah(meta['soh0']) * 3600.0
    z0_val = float(np.clip(meta['soc0'] + np.random.default_rng(seed + 9_000_003 + 10_007 * ci).normal(0.0, ev.SOC_BIAS_SD), 0, 1))
    z0_id = float(np.clip(meta['soc0'] + np.random.default_rng(seed + 303 * (ci + 1)).normal(0.0, ev.SOC_BIAS_SD), 0, 1))
    return dict(meta=meta, cap=cap, vmeas=V['vmeas'], v_val=V['v_val_truth'], z0_id=z0_id, z0_val=z0_val,
                target=dict(C1=V['ocv_val'], C2=ev.bms.ocv(ev.coulomb(z0_val, V['vmeas'], .05, cap)), C2p=ev.bms.ocv(ev.coulomb(z0_id, V['vmeas'], .05, cap))))
ev.load_cell = load_cell_wltc
import n6_fit, n2_ms, n1_ecm
tag = sys.argv[1]; TD = Path(os.environ['TRACES']); ROOT = Path(os.environ['NEWOUT'])
n6_fit.run(tag)
n2_ms.run(tag)
n1_ecm.run('nodrift', tag, n1_ecm.setup('nodrift'))
cell = load_cell_wltc(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell); out = {}
for name, sc in (('cge', None), ('lin', np.zeros(0))):
    m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0, scales=sc)
    e = cell['target']['C1'] - m.predict_drop(cell['vmeas'], .05) - cell['v_val']
    out[name] = dict(drive=ev.rmse_mv(e[:N_DRIVE]), rest=ev.rmse_mv(e[N_DRIVE:]), all=ev.rmse_mv(e))
s = cell['meta']; out['soc_change_pct'] = float(abs(np.load(TD / f'{tag}.wltc.npz')['soc_val'][-1] - s['soc0']) * 100)
(ROOT / 'split').mkdir(parents=True, exist_ok=True); (ROOT / 'split' / f'{tag}.json').write_text(json.dumps(dict(tag=tag, cond=s['cond'], pack=s['pack'], **out)))
print('split', tag, {k: (round(v['drive'], 3), round(v['rest'], 3)) for k, v in out.items() if isinstance(v, dict)}, flush=True)
