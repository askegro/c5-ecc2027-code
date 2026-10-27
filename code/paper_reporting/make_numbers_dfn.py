import sys, json, tarfile
from math import comb
from pathlib import Path
import numpy as np
TARS, (E2D, NEW, LINF) = sys.argv[1:3], sys.argv[3:6]
CONDS = ('cold', 'nominal', 'hot', 'aged'); RUN = CONDS_RUN = ('cold', 'nominal', 'hot', 'aged'); FREQS = ('0.008', '0.05', '1.0'); EST = ('conv', 'ecm', 'drt', 'lin', 'cge'); MSKEY = 'cge|1|C1'
def zv(p): return complex(*p)
def zconv(v, f): w = 2 * np.pi * float(f); return v['r0'] + sum(r / (1 + 1j * w * tau) for r, tau in zip(v['r'], v['tau']))
J = {}
for TAR, OUT in zip(TARS, ('n11out/orig/', 'n13out/orig/')):
    with tarfile.open(TAR) as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            if m.isfile() and p.suffix == '.json' and OUT in m.name and p.parent.name in ('fit', 'nodrift', 'ms'):
                assert p.stem not in J.get(p.parent.name, {}), p.stem
                J.setdefault(p.parent.name, {})[p.stem] = json.load(tf.extractfile(m))
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))}
D = {}
for t, r in J['fit'].items():
    s = r['seq']['s4786']['C1']; N, M = J['nodrift'][t], J['ms'][t]
    D[t] = dict(cond=r['meta']['cond'], pack=r['meta']['pack'],
                rmse=dict(conv=s['v2']['rmse_mv'], ecm=N['C1']['rmse_mv'], drt=M[MSKEY]['rmse_mv'], lin=r['linear']['C1']['rmse_mv'], cge=s['v3']['rmse_mv']),
                z={f: dict(ref=zv(r['zref']['z'][f]['drop_bulk']), conv=zconv(s['v2'], f), ecm=zv(N['C1']['z'][f]), drt=zv(M[MSKEY]['z'][f]),
                           lin=zv(r['linear']['C1']['z'][f]), cge=zv(s['v3']['z'][f])) for f in FREQS})
E2, NOD, MS = load(E2D), load(Path(NEW) / 'nodrift'), load(Path(NEW) / 'ms')
LIN = {r['tag']: r['linear'] for r in json.load(open(LINF)) if r['arm'] == 'C1'}
S = {t: dict(cond=E2[t]['meta']['cond'], rmse=dict(conv=E2[t]['seq']['s4786']['C1']['v2']['rmse_mv'], ecm=NOD[t]['C1']['rmse_mv'], drt=MS[t][MSKEY]['rmse_mv'],
                                                   lin=LIN[t], cge=E2[t]['seq']['s4786']['C1']['v3']['rmse_mv'])) for t in D}
assert len(D) == 96 and {D[t]['cond'] for t in D} == set(RUN)
def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum()); n = pos + neg; k = min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def pairs(c, g):
    d = {}
    for t, r in D.items():
        if r['cond'] == c: d.setdefault(r['pack'], []).append(g(r))
    return np.array([np.mean(v) for k, v in sorted(d.items()) if len(v) == 2])
med = lambda R, c, e: float(np.median([r['rmse'][e] for r in R.values() if r['cond'] == c]))
zerr = lambda r, e, f: abs(r['z'][f][e] - r['z'][f]['ref']) / abs(r['z'][f]['ref']) * 100
nlow, hmax = [], []
for e in EST[:-1]:
    ps, nl = [], []
    for c in CONDS:
        if c not in RUN: ps.append(1.0); continue
        dd = pairs(c, lambda r: r['rmse']['cge']) - pairs(c, lambda r, e=e: r['rmse'][e]); assert len(dd) == 12
        ps.append(sign_p(dd)); nl.append(int((dd < 0).sum()))
    nlow += nl; hmax.append(max(holm(np.array(ps))))
zc = [float(np.median([zerr(r, 'cge', f) for r in D.values() if r['cond'] == c])) for c in RUN for f in FREQS]
zo = [float(np.median([zerr(r, e, f) for r in D.values() if r['cond'] == c])) for c in RUN for f in FREQS for e in EST[:-1]]
chg = [abs(med(D, c, e) - med(S, c, e)) for c in RUN for e in EST]
out = {'dfn.cge.range': '%.2f--%.2f' % (min(med(D, c, 'cge') for c in RUN), max(med(D, c, 'cge') for c in RUN)),
       'dfn.nlowmin': str(min(nlow)), 'dfn.holm.max': '%.3f' % max(hmax), 'dfn.maxchange': '%.2f' % max(chg),
       'dfn.maxchange.cge': '%.2f' % max(abs(med(D, c, 'cge') - med(S, c, 'cge')) for c in RUN),
       'dfn.maxchange.oth': '%.2f' % max(abs(med(D, c, e) - med(S, c, e)) for c in RUN for e in EST[:-1]),
       'dfn.z.cge.max': '%.1f' % max(zc), 'dfn.z.oth.min': '%.1f' % min(zo), 'dfn.n': str(len(D))}
with open('numbers_dfn.tex', 'w') as f:
    for k, v in out.items(): f.write('\\expandafter\\def\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(out); print('largest median change (mV):', max(chg), [(c, e, round(med(D, c, e) - med(S, c, e), 4)) for c in RUN for e in EST])
