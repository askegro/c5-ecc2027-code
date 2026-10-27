import sys, json, tarfile
from math import comb
from pathlib import Path
import numpy as np
J = {}
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        p = Path(m.name)
        if m.isfile() and p.suffix == '.json' and 'n15out/orig/' in m.name and p.parent.name in ('fit', 'nodrift', 'split', 'g265'):
            J.setdefault(p.parent.name, {})[p.stem] = json.load(tf.extractfile(m))
CONDS = ('cold', 'nominal', 'hot', 'aged'); T = sorted(J['g265']); assert len(T) == 96
R = {}
for t in T:
    g = J['g265'][t]; assert abs(g['cge|2000']['all'] - J['split'][t]['cge']['all']) < 1e-9
    R[t] = dict(cond=g['cond'], pack=g['pack'], cge=g['cge|265']['all'], cge2000=g['cge|2000']['all'], ecm=J['nodrift'][t]['C1']['rmse_mv'],
                conv=J['fit'][t]['seq']['s4786']['C1']['v2']['rmse_mv'], drt=g['drt|265']['all'], lin=g['lin|265']['all'])
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
med = lambda c, k: float(np.median([r[k] for r in R.values() if r['cond'] == c]))
rng = lambda v: f'{min(v):.2f}--{max(v):.2f}'
nl, hp = [], []
for e in ('ecm', 'conv', 'drt', 'lin'):
    ps = []
    for c in CONDS:
        d = {}
        for r in R.values():
            if r['cond'] == c: d.setdefault(r['pack'], []).append(r['cge'] - r[e])
        dd = np.array([np.mean(v) for v in d.values()]); assert len(dd) == 12; nl.append(int((dd < 0).sum())); ps.append(sign_p(dd))
    hp += list(holm(np.array(ps)))
M = {'n15.cge.range': rng([med(c, 'cge') for c in CONDS]), 'n15.cge2000.range': rng([med(c, 'cge2000') for c in CONDS]),
     'n15.oth.range': rng([med(c, e) for c in CONDS for e in ('ecm', 'conv', 'drt', 'lin')]),
     'n15.nlowmin': str(min(nl)), 'n15.holm.max': f'{max(hp):.3f}', 'n15.n': str(len(R))}
open('numbers_n15.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items())); print(M)
