import sys, json, numpy as np
from math import comb
from pathlib import Path
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))}
E2, MS, LIN = load(sys.argv[1]), load(Path(sys.argv[2]) / 'ms'), json.load(open(sys.argv[3]))
CONDS = ('cold', 'nominal', 'hot', 'aged'); KEY = dict(new='rmse_mv', same='rmse_persist_mv'); M = {}
def get(t, e, case):
    k = KEY[case]; s = E2[t]['seq']['s4786']['C2']
    return dict(conv=lambda: s['v2'][k], drt=lambda: MS[t]['cge|1|C2'][k], lin=lambda: LIN[t]['C2'][k], cge=lambda: s['v3'][k], drtv=lambda: MS[t]['cge|1|V']['rmse_mv'])[e]()
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def contrast(ea, ca, eb, cb):
    n_low, ps = [], []
    for c in CONDS:
        d = {}
        for t, r in E2.items():
            if r['meta']['cond'] == c: d.setdefault(r['meta']['pack'], []).append(get(t, ea, ca) - get(t, eb, cb))
        dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); n_low.append(int((dd < 0).sum())); ps.append(sign_p(dd))
    return dict(zip(CONDS, n_low)), dict(zip(CONDS, holm(np.array(ps))))
med = {(e, case): {c: float(np.median([get(t, e, case) for t, r in E2.items() if r['meta']['cond'] == c])) for c in CONDS} for e in ('conv', 'drt', 'lin', 'cge', 'drtv') for case in KEY}
def rng(v): return f'{min(v):.2f}--{max(v):.2f}'
for case in KEY:
    M[f'dep.cge.{case}.range'] = rng(med[('cge', case)].values())
    M[f'dep.oth.{case}.range'] = rng([med[(e, case)][c] for e in ('conv', 'drt', 'lin') for c in CONDS])
M['dep.v.range'] = rng(med[('drtv', 'new')].values())
nl, hp = contrast('cge', 'new', 'conv', 'new')
for c in CONDS: M[f'dep.conv.nlow.{c}'] = str(nl[c]); M[f'dep.conv.holm.{c}'] = f'{hp[c]:.3f}'
x = [contrast('cge', 'new', e, 'new') for e in ('drt', 'lin')]
M['dep.lin.nlowmin'] = str(min(min(a.values()) for a, _ in x)); M['dep.lin.holm.max'] = f'{max(max(b.values()) for _, b in x):.3f}'
x = [contrast('cge', 'same', e, 'same') for e in ('conv', 'drt', 'lin')] + [contrast('cge', 'same', 'drtv', 'new')]
M['dep.same.nlowmin'] = str(min(min(a.values()) for a, _ in x)); M['dep.same.holm.max'] = f'{max(max(b.values()) for _, b in x):.3f}'
nl, hp = contrast('drtv', 'new', 'cge', 'new')
for c in CONDS: M[f'dep.v.nlow.{c}'] = str(nl[c]); M[f'dep.v.holm.{c}'] = f'{hp[c]:.3f}'
M['dep.v.cold.cgelow'] = str(12 - nl['cold'])
M['dep.n'] = str(len(E2))
open('numbers_dep.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items()))
rows = []
for case, lab in (('new', 'New'), ('same', 'Same')):
    for i, c in enumerate(CONDS):
        rows.append(f"{lab if i == 0 else ''} & {c.capitalize()} & " + ' & '.join(f"{med[(e, case)][c]:.2f}" for e in ('conv', 'drt', 'lin', 'cge')) + r'\\' + (r'\hline' if i == 3 else ''))
open('table_dep.tex', 'w').write('\\begin{tabular}{@{}llrrrr@{}}\\hline\nSOC Error & Condition & Conv. & DRT & Linear & \\cge{}\\\\\\hline\n' + '\n'.join(rows) + '\n\\end{tabular}\n')
for k, v in M.items(): print(k, v)
print(open('table_dep.tex').read())
