import sys, json, tarfile, numpy as np
from math import comb
R = {'fit': {}, 'nodrift': {}, 'ms': {}, 'split': {}}
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        p = m.name.split('/')
        if len(p) == 4 and p[0] == 'n8out' and p[1] == 'orig' and p[2] in R and p[3].endswith('.json'):
            R[p[2]][p[3][:-5]] = json.load(tf.extractfile(m))
T = sorted(R['fit']); assert len(T) == 96 and all(t in R[k] for k in R for t in T)
CONDS = ('cold', 'nominal', 'hot', 'aged'); WARM = CONDS[1:]; M = {}
cond = {t: R['fit'][t]['meta']['cond'] for t in T}; pack = {t: R['fit'][t]['meta']['pack'] for t in T}
def rmse(t, e):
    s = R['fit'][t]['seq']['s4786']['C1']
    return dict(conv=s['v2']['rmse_mv'], ecm=R['nodrift'][t]['C1']['rmse_mv'], drt=R['ms'][t]['cge|1|C1']['rmse_mv'], lin=R['fit'][t]['linear']['C1']['rmse_mv'], cge=s['v3']['rmse_mv'])[e]
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def contrast(ga, gb):
    nl, ps = {}, []
    for c in CONDS:
        d = {}
        for t in T:
            if cond[t] == c: d.setdefault(pack[t], []).append(ga(t) - gb(t))
        dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); nl[c] = int((dd < 0).sum()); ps.append(sign_p(dd))
    return nl, dict(zip(CONDS, holm(np.array(ps))))
def med(g, c): return float(np.median([g(t) for t in T if cond[t] == c]))
def rng(v, nd=2): return f'{min(v):.{nd}f}--{max(v):.{nd}f}'
um = {e: {c: med(lambda t, e=e: rmse(t, e), c) for c in CONDS} for e in ('conv', 'ecm', 'drt', 'lin', 'cge')}
M['wl.cge.range'] = rng(um['cge'].values()); M['wl.cge.cold'] = f"{um['cge']['cold']:.2f}"
M['wl.cge.warm.range'] = rng([um['cge'][c] for c in WARM]); M['wl.conv.range'] = rng(um['conv'].values())
M['wl.lin3.cold.range'] = rng([um[e]['cold'] for e in ('ecm', 'drt', 'lin')]); M['wl.lin3.warm.range'] = rng([um[e][c] for e in ('ecm', 'drt', 'lin') for c in WARM])
nl, hp = contrast(lambda t: rmse(t, 'cge'), lambda t: rmse(t, 'conv'))
M['wl.cgeconv.nlowmin'] = str(min(nl.values())); M['wl.cgeconv.holm.max'] = f'{max(hp.values()):.3f}'
x = [contrast(lambda t: rmse(t, 'cge'), lambda t, e=e: rmse(t, e)) for e in ('ecm', 'drt', 'lin')]
M['wl.cold.nlowmin'] = str(min(a['cold'] for a, _ in x)); M['wl.cold.holm.max'] = f"{max(b['cold'] for _, b in x):.3f}"
y = [contrast(lambda t, e=e: rmse(t, e), lambda t: rmse(t, 'cge')) for e in ('ecm', 'drt', 'lin')]
M['wl.warm.othlow.min'] = str(min(a[c] for a, _ in y for c in WARM)); M['wl.warm.othlow.max'] = str(max(a[c] for a, _ in y for c in WARM))
M['wl.warm.othlow.holm.min'] = f"{min(b[c] for _, b in y for c in WARM):.3f}"; M['wl.warm.othlow.holm.max'] = f"{max(b[c] for _, b in y for c in WARM):.3f}"
S = R['split']
for e in ('cge', 'lin'):
    for part in ('drive', 'rest'):
        M[f'wl.{e}.{part}.range'] = rng([float(np.median([S[t][e][part] for t in T if cond[t] == c])) for c in CONDS])
M['wl.soc.range'] = rng([float(np.median([S[t]['soc_change_pct'] for t in T if cond[t] == c])) for c in CONDS], 1)
open('numbers_wltc.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items()))
for k, v in M.items(): print(k, v)
print({e: {c: round(v, 2) for c, v in um[e].items()} for e in um})
for e, (a, b) in zip(('ecm', 'drt', 'lin'), y): print('others lower than CGE', e, a, {c: round(v, 3) for c, v in b.items()})
