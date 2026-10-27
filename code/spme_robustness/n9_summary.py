import sys, json, numpy as np
from math import comb
from pathlib import Path
R = json.load(open(sys.argv[1])); N8 = Path(sys.argv[2])
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted((N8 / d).glob('*.json'))}
F, N, M = load('fit'), load('nodrift'), load('ms')
oth = dict(ecm=lambda t: N[t]['C1']['rmse_mv'], drt=lambda t: M[t]['cge|1|C1']['rmse_mv'], lin2000=lambda t: F[t]['linear']['C1']['rmse_mv'])
CONDS = ('cold', 'nominal', 'hot', 'aged'); TM = ('50', '90', '265', '500', '2000')
assert all(abs(R[t]['cge|2000']['all'] - F[t]['seq']['s4786']['C1']['v3']['rmse_mv']) < 1e-9 for t in R), 'grid 2000 s differs from N8'
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
print(f'{len(R)} units. Median RMSE (mV) over units: all [drive / rest]')
for tm in TM:
    print(f'tau_max {tm:>4} s  ' + '  '.join(f"{c[:3]} CGE {np.median([R[t][f'cge|{tm}']['all'] for t in R if R[t]['cond'] == c]):.2f} [{np.median([R[t][f'cge|{tm}']['drive'] for t in R if R[t]['cond'] == c]):.2f}/{np.median([R[t][f'cge|{tm}']['rest'] for t in R if R[t]['cond'] == c]):.2f}] lin {np.median([R[t][f'lin|{tm}']['all'] for t in R if R[t]['cond'] == c]):.2f}" for c in CONDS))
print('Pairs in which the CGE (each grid end) has the lower error than the N8 estimators (of 12), Holm-adjusted p:')
for tm in TM:
    row = []
    for e, g in oth.items():
        nl, ps = [], []
        for c in CONDS:
            d = {}
            for t in R:
                if R[t]['cond'] == c: d.setdefault(R[t]['pack'], []).append(R[t][f'cge|{tm}']['all'] - g(t))
            dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); nl.append(int((dd < 0).sum())); ps.append(sign_p(dd))
        h = holm(np.array(ps)); row.append(f"{e}: " + ' '.join(f'{n}({p:.3f})' for n, p in zip(nl, h)))
    print(f'tau_max {tm:>4} s  ' + ' | '.join(row))
