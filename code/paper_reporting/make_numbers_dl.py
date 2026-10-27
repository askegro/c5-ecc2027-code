import sys, json, tarfile, numpy as np
from math import comb
R = {'fit': {}, 'nodrift': {}, 'ms': {}}
with tarfile.open(sys.argv[1]) as tf:
    for m in tf.getmembers():
        p = m.name.split('/')
        if len(p) == 4 and p[0] == 'n6out' and p[1] == 'dl' and p[2] in R and p[3].endswith('.json'):
            R[p[2]][p[3][:-5]] = json.load(tf.extractfile(m))
CONDS = ('cold', 'nominal', 'hot', 'aged'); FREQS = ('0.008', '0.05', '1.0'); M = {}
T = sorted(R['fit']); assert len(T) == 96 and all(t in R['nodrift'] and t in R['ms'] for t in T)
cond = {t: R['fit'][t]['meta']['cond'] for t in T}; pack = {t: R['fit'][t]['meta']['pack'] for t in T}
assert all(R['fit'][t]['meta']['spread'] == 'dl' for t in T)
def zc(v): return complex(*v)
def zconv(v, f): w = 2 * np.pi * float(f); return v['r0'] + sum(r / (1 + 1j * w * tau) for r, tau in zip(v['r'], v['tau']))
def rmse(t, e):
    s = R['fit'][t]['seq']['s4786']['C1']
    return dict(conv=s['v2']['rmse_mv'], ecm=R['nodrift'][t]['C1']['rmse_mv'], drt=R['ms'][t]['cge|1|C1']['rmse_mv'], lin=R['fit'][t]['linear']['C1']['rmse_mv'], cge=s['v3']['rmse_mv'])[e]
def z(t, e, f):
    s = R['fit'][t]['seq']['s4786']['C1']
    return dict(conv=lambda: zconv(s['v2'], f), ecm=lambda: zc(R['nodrift'][t]['C1']['z'][f]), drt=lambda: zc(R['ms'][t]['cge|1|C1']['z'][f]), lin=lambda: zc(R['fit'][t]['linear']['C1']['z'][f]), cge=lambda: zc(s['v3']['z'][f]))[e]()
def zref(t, f): return zc(R['fit'][t]['zref']['z'][f]['drop_bulk'])
def zerr(t, e, f, part='abs'):
    d = (z(t, e, f) - zref(t, f)) / abs(zref(t, f)) * 100; return abs(d) if part == 'abs' else (d.real if part == 're' else d.imag)
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
M['dl.cge.range'] = rng(um['cge'].values()); M['dl.cge.cold'] = f"{um['cge']['cold']:.2f}"
M['dl.cge.warm.range'] = rng([um['cge'][c] for c in ('nominal', 'hot', 'aged')])
M['dl.lin3.range'] = rng([um[e][c] for e in ('ecm', 'drt', 'lin') for c in CONDS]); M['dl.conv.range'] = rng(um['conv'].values())
x = [contrast(lambda t: rmse(t, 'cge'), lambda t, e=e: rmse(t, e)) for e in ('conv', 'ecm', 'drt', 'lin')]
M['dl.cge.nlowmin'] = str(min(min(a.values()) for a, _ in x)); M['dl.cge.holm.max'] = f'{max(max(b.values()) for _, b in x):.3f}'
zm = {(e, f): {c: med(lambda t, e=e, f=f: zerr(t, e, f), c) for c in CONDS} for e in ('conv', 'ecm', 'drt', 'lin', 'cge') for f in FREQS}
M['dl.z.cge.low.range'] = rng([zm[('cge', f)][c] for f in FREQS[:2] for c in CONDS], 1)
M['dl.z.oth.low.range'] = rng([zm[(e, f)][c] for e in ('conv', 'ecm', 'drt', 'lin') for f in FREQS[:2] for c in CONDS], 1)
xl = [contrast(lambda t, f=f: zerr(t, 'cge', f), lambda t, e=e, f=f: zerr(t, e, f)) for e in ('conv', 'ecm', 'drt', 'lin') for f in FREQS[:2]]
M['dl.z.low.nlowmin'] = str(min(min(a.values()) for a, _ in xl)); M['dl.z.low.holm.max'] = f'{max(max(b.values()) for _, b in xl):.3f}'
M['dl.z.cge.one.range'] = rng(zm[('cge', '1.0')].values(), 1); M['dl.z.lin3.one.range'] = rng([zm[(e, '1.0')][c] for e in ('ecm', 'drt', 'lin') for c in CONDS], 1)
nl, hp = contrast(lambda t: zerr(t, 'drt', '1.0'), lambda t: zerr(t, 'cge', '1.0'))
for c in CONDS: M[f'dl.z.drtlow.{c}'] = str(nl[c]); M[f'dl.z.drtlow.holm.{c}'] = f'{hp[c]:.3f}'
im = {c: med(lambda t: zerr(t, 'cge', '1.0', 'im'), c) for c in CONDS}; M['dl.z.cge.im.one.range'] = rng(im.values(), 1)
M['dl.n'] = str(len(T))
open('numbers_dl.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items()))
for k, v in M.items(): print(k, v)
print({e: {c: round(v, 2) for c, v in um[e].items()} for e in um})
