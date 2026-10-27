import os, sys, json
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce
TD = Path(os.environ.get('TRACES', 'traces_confirm')); ROOT = Path(os.environ.get('NEWOUT', 'new')); FREQS = (0.008, 0.05, 1.0)
def run(tag):
    out_dir = ROOT / 'imp'; out_dir.mkdir(parents=True, exist_ok=True); f = out_dir / f'{tag}.json'
    if f.exists(): return
    cell = ev.load_cell(TD, tag); meta = cell['meta']; seq = ev.load_seq(TD, tag, 's4786', cell); out = dict(tag=tag, cond=meta['cond'], pack=meta['pack'], cell=meta['cell'])
    for arm in ('C1', 'C2'):
        m = ce.fit(seq['i'], seq['drop'][arm], .01, offset=(arm == 'C2'), capacity_ah=60.0); pred = m.predict_drop(cell['vmeas'], .05)
        out[arm] = dict(rmse_mv=ev.rmse_mv(cell['target'][arm] - pred - cell['v_val']), z={str(fr): [float(v.real), float(v.imag)] for fr, v in zip(FREQS, m.impedance(FREQS))})
    f.write_text(json.dumps(out, indent=1)); print(tag, round(out['C1']['rmse_mv'], 3), flush=True)
if __name__ == '__main__':
    for t in sys.argv[1:]: run(t)
