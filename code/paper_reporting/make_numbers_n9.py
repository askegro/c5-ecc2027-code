import sys, json, tarfile, numpy as np
from math import comb
R, DR, N8 = None, None, {'fit': {}, 'nodrift': {}, 'ms': {}}
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        p = m.name.split('/')
        if m.name == 'n9out/taumax.json': R = json.load(tf.extractfile(m))
        elif m.name == 'n9out/drt_taumax.json': DR = json.load(tf.extractfile(m))
        elif len(p) == 4 and p[0] == 'n8out' and p[1] == 'orig' and p[2] in N8 and p[3].endswith('.json'): N8[p[2]][p[3][:-5]] = json.load(tf.extractfile(m))
T = sorted(R); assert len(T) == 96
assert all(abs(R[t]['cge|2000']['all'] - N8['fit'][t]['seq']['s4786']['C1']['v3']['rmse_mv']) < 1e-9 for t in T)
assert sorted(DR) == T and all(abs(DR[t]['drt|2000']['all'] - N8['ms'][t]['cge|1|C1']['rmse_mv']) < 1e-9 for t in T)
CONDS = ('cold', 'nominal', 'hot', 'aged'); M = {}
oth = dict(ecm=lambda t: N8['nodrift'][t]['C1']['rmse_mv'], drt265=lambda t: DR[t]['drt|265']['all'], lin265=lambda t: R[t]['lin|265']['all'])
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def med(key, part, c): return float(np.median([R[t][key][part] for t in T if R[t]['cond'] == c]))
def rng(v): return f'{min(v):.2f}--{max(v):.2f}'
for tm in ('50', '90', '265', '500'):
    M[f'n9.cge{tm}.range'] = rng([med(f'cge|{tm}', 'all', c) for c in CONDS])
M['n9.cge265.rest.range'] = rng([med('cge|265', 'rest', c) for c in CONDS]); M['n9.lin265.range'] = rng([med('lin|265', 'all', c) for c in CONDS])
M['n9.drt265.range'] = rng([float(np.median([DR[t]['drt|265']['all'] for t in T if R[t]['cond'] == c])) for c in CONDS])
nl, hp = [], []
for e, g in oth.items():
    ps = []
    for c in CONDS:
        d = {}
        for t in T:
            if R[t]['cond'] == c: d.setdefault(R[t]['pack'], []).append(R[t]['cge|265']['all'] - g(t))
        dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); nl.append(int((dd < 0).sum())); ps.append(sign_p(dd))
    hp += list(holm(np.array(ps)))
M['n9.nlowmin'] = str(min(nl)); M['n9.holm.max'] = f'{max(hp):.3f}'
M['n9.vslin265'] = str(sum(R[t]['cge|265']['all'] < R[t]['lin|265']['all'] for t in T)); M['n9.n'] = str(len(T))
open('numbers_n9.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items()))
for k, v in M.items(): print(k, v)
