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
        rows.append(r)
    print(tag, flush=True)
Path('posthoc_linear_rows.json').write_text(json.dumps(rows, indent=1))
