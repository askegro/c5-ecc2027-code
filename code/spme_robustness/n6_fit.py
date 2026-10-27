import os, sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce, designs
TD = Path(os.environ.get('TRACES', 'traces')); ROOT = Path(os.environ.get('NEWOUT', 'out')); FREQS = (0.008, 0.05, 1.0)
D = designs.all_designs()
def zc(m): return {str(f): [float(v.real), float(v.imag)] for f, v in zip(FREQS, m.impedance(FREQS))}
def run(tag):
    out_dir = ROOT / 'fit'; out_dir.mkdir(parents=True, exist_ok=True); f = out_dir / f'{tag}.json'
    if f.exists(): return
    cell = ev.load_cell(TD, tag); meta = cell['meta']; seq = ev.load_seq(TD, tag, 's4786', cell)
    o = json.loads(str(np.load(TD / f'{tag}.s4786.npz')['meta']))['orientation']
    assert o == (1 if meta['cell'] == 0 else -1), f'{tag}: orientation {o}, the paper uses +1 for unit 0 and -1 for unit 1'
    t0 = time.time(); a = ev.eval_new(cell, seq, 'C1'); a['sec'] = time.time() - t0
    a['repeat_identical'] = bool(a['rmse_mv'] == ev.eval_new(cell, seq, 'C1')['rmse_mv'])
    m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0); a['z'] = zc(m)
    lin = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, scales=np.zeros(0))
    linear = dict(rmse_mv=ev.rmse_mv(cell['target']['C1'] - lin.predict_drop(cell['vmeas'], .05) - cell['v_val']), kkt=lin.kkt_violation, z=zc(lin))
    v2 = ev.eval_frozen(cell, seq, 'C1', D['s4786'], meta['seed'] + 50_000 + meta['cell'])
    res = dict(meta=meta, zref=json.loads((TD / f'{tag}.zref.json').read_text()), seq=dict(s4786=dict(C1=dict(v3=a, v2=v2))), linear=dict(C1=linear))
    f.write_text(json.dumps(res, indent=1))
    print(tag, meta.get('spread'), 'cge', round(a['rmse_mv'], 3), 'linear', round(linear['rmse_mv'], 3), 'conv', round(v2['rmse_mv'], 3), flush=True)
if __name__ == '__main__':
    for t in sys.argv[1:]: run(t)
