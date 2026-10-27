import sys, json, time
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
TD = Path(sys.argv[1] if len(sys.argv) > 1 else 'n8out/orig/traces'); OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'n9out/taumax.json')
TAU_MAX = (50.0, 90.0, 265.0, 500.0, 2000.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
res = json.loads(OUT.read_text()) if OUT.exists() else {}
tags = sorted(p.name[:-len('.wltc.npz')] for p in TD.glob('*.wltc.npz'))
for t in tags:
    if t in res: continue
    cell = load_cell_wltc(TD, t); seq = ev.load_seq(TD, t, 's4786', cell)
    o = json.loads(str(np.load(TD / f'{t}.s4786.npz')['meta']))['orientation']
    assert o == (1 if cell['meta']['cell'] == 0 else -1), f'{t}: orientation {o}'
    r = dict(cond=cell['meta']['cond'], pack=cell['meta']['pack'], cell=cell['meta']['cell'])
    for tm in TAU_MAX:
        for name, sc in (('cge', None), ('lin', np.zeros(0))):
            m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0, taus=ce.tau_grid(tau_max=tm), scales=sc)
            e = cell['target']['C1'] - m.predict_drop(cell['vmeas'], .05) - cell['v_val']
            r[f'{name}|{tm:g}'] = dict(all=ev.rmse_mv(e), drive=ev.rmse_mv(e[:N_DRIVE]), rest=ev.rmse_mv(e[N_DRIVE:]), kkt=m.kkt_violation)
    res[t] = r; OUT.write_text(json.dumps(res))
    print(t, {k: round(v['all'], 3) for k, v in r.items() if '|' in k}, flush=True)
print('N9_DONE', len(res), 'of', len(tags))
