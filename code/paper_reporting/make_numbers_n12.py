import sys, json, tarfile
from pathlib import Path
import numpy as np
MSKEY = 'cge|1|C1'; RUN = ('cold', 'nominal'); EST = ('conv', 'ecm', 'drt', 'lin', 'cge')
def rows(tar, out):
    J = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            if m.isfile() and p.suffix == '.json' and out in m.name and p.parent.name in ('fit', 'nodrift', 'ms'):
                J.setdefault(p.parent.name, {})[p.stem] = json.load(tf.extractfile(m))
    R = {}
    for t, r in J['fit'].items():
        s = r['seq']['s4786']['C1']
        R[t] = dict(cond=r['meta']['cond'], rmse=dict(conv=s['v2']['rmse_mv'], ecm=J['nodrift'][t]['C1']['rmse_mv'], drt=J['ms'][t][MSKEY]['rmse_mv'],
                                                      lin=r['linear']['C1']['rmse_mv'], cge=s['v3']['rmse_mv']))
    return R
A, B = rows(sys.argv[1], 'n11out/orig/'), rows(sys.argv[2], 'n12out/tight/')
assert set(A) == set(B) and len(A) == 48
med = lambda R, c, e: float(np.median([r['rmse'][e] for r in R.values() if r['cond'] == c]))
ch = max(abs(med(B, c, e) - med(A, c, e)) for c in RUN for e in EST)
out = {'n12.maxchange': '%.3f' % ch}
with open('numbers_n12.tex', 'w') as f:
    for k, v in out.items(): f.write('\\expandafter\\def\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(out, ch)
