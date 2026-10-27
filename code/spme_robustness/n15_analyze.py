import sys, json
from math import comb
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); CONDS = ('cold', 'nominal', 'hot', 'aged')
def load(sub): return {p.stem: json.loads(p.read_text()) for p in sorted((D / sub).glob('*.json'))}
F, N, S, G = load('fit'), load('nodrift'), load('split'), load('g265')
T = sorted(set(F) & set(N) & set(S) & set(G)); print(f'complete units: {len(T)} of 96')
R = {}
for t in T:
    g = G[t]; assert abs(g['cge|2000']['all'] - S[t]['cge']['all']) < 1e-9, (t, 'cge|2000 differs from n8_fit_all')
    R[t] = dict(cond=g['cond'], pack=g['pack'], cge=g['cge|265']['all'], cge2000=g['cge|2000']['all'], ecm=N[t]['C1']['rmse_mv'],
                conv=F[t]['seq']['s4786']['C1']['v2']['rmse_mv'], drt=g['drt|265']['all'], lin=g['lin|265']['all'],
                cge_drive=g['cge|265']['drive'], cge_rest=g['cge|265']['rest'])
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def pairs(c, f):
    d = {}
    for r in R.values():
        if r['cond'] == c: d.setdefault(r['pack'], []).append(f(r))
    return np.array([np.mean(v) for k, v in sorted(d.items()) if len(v) == 2])
med = lambda c, k: float(np.median([r[k] for r in R.values() if r['cond'] == c])) if any(r['cond'] == c for r in R.values()) else float('nan')
out = dict(n_units=len(R), medians={k: {c: med(c, k) for c in CONDS} for k in ('cge', 'cge2000', 'ecm', 'conv', 'drt', 'lin', 'cge_drive', 'cge_rest')}, contrasts={})
ok = len(R) == 96
for e in ('ecm', 'conv', 'drt', 'lin'):
    ps, rows = [], {}
    for c in CONDS:
        dd = pairs(c, lambda r, e=e: r['cge'] - r[e])
        rows[c] = dict(n=len(dd), median=float(np.median(dd)) if len(dd) else float('nan'), n_lower=int((dd < 0).sum()), p=sign_p(dd)); ps.append(rows[c]['p'])
    for c, h in zip(CONDS, holm(np.array(ps))): rows[c]['p_holm'] = float(h); ok &= rows[c]['n'] == 12 and rows[c]['n_lower'] >= 11 and h <= 0.025
    out['contrasts'][e] = rows
print('median RMSE (mV) on the WLTC validation:')
for k, v in out['medians'].items(): print(f'  {k:10s} ' + '  '.join(f'{c[:3]} {x:6.2f}' for c, x in v.items()))
for e, rows in out['contrasts'].items(): print(f'  CGE(265) - {e:4s}: ' + '  '.join(f"{c[:3]} {r['median']:+.2f} [{r['n_lower']}/12, Holm {r['p_holm']:.3f}]" for c, r in rows.items()))
out['criterion_met'] = bool(ok); print('N15 CRITERION', 'MET' if ok else 'NOT MET')
json.dump(out, open('n15_result.json', 'w'), indent=1)
