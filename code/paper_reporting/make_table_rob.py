import sys, json, tarfile
from pathlib import Path
import numpy as np
E2D, NEW, LINF, CVXF, MMD, T6, T11, T12, T13, T15 = sys.argv[1:11]
CONDS = ('cold', 'nominal', 'hot', 'aged'); MS = 'cge|1|C1'
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))}
def from_tar(tar, prefix):
    J = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            if m.isfile() and p.suffix == '.json' and m.name.startswith(prefix) and p.parent.name in ('fit', 'nodrift', 'ms') and p.parent.parent.as_posix() == prefix.rstrip('/'):
                J.setdefault(p.parent.name, {})[p.stem] = json.load(tf.extractfile(m))
    return J
def arm(J):
    R = {}
    for t, r in J['fit'].items():
        s = r['seq']['s4786']['C1']
        R[t] = dict(cond=r['meta']['cond'], pack=r['meta']['pack'], cge=s['v3']['rmse_mv'], conv=s['v2']['rmse_mv'],
                    ecm=J['nodrift'][t]['C1']['rmse_mv'], drt=J['ms'][t][MS]['rmse_mv'], lin=r['linear']['C1']['rmse_mv'])
    return R
E2, NOD, MSD = load(E2D), load(Path(NEW) / 'nodrift'), load(Path(NEW) / 'ms')
LIN1 = {r['tag']: r['linear'] for r in json.load(open(LINF)) if r['arm'] == 'C1'}; CVX = json.load(open(CVXF))
base = {t: dict(cond=r['meta']['cond'], pack=r['meta']['pack'], cge=r['seq']['s4786']['C1']['v3']['rmse_mv'], conv=r['seq']['s4786']['C1']['v2']['rmse_mv'],
                ecm=NOD[t]['C1']['rmse_mv'], drt=MSD[t][MS]['rmse_mv'], lin=LIN1[t]) for t, r in E2.items()}
def dep(case):
    k = dict(new='rmse_mv', same='rmse_persist_mv')[case]; out = {}
    for t, r in E2.items():
        s = r['seq']['s4786']['C2']
        out[t] = dict(cond=r['meta']['cond'], pack=r['meta']['pack'], cge=s['v3'][k], conv=s['v2'][k], drt=MSD[t]['cge|1|C2'][k], lin=CVX[t]['C2'][k],
                      drtv=MSD[t]['cge|1|V']['rmse_mv'])
    return out
MM = load(MMD)
asym = {t: dict(cond=r['cond'], pack=r['pack'], cge=r['rec']['s4786']['C1']['rmse_mv'], conv=r['v2']['C1']['rmse_mv'], lin=r['linear']['C1']) for t, r in MM.items()}
J6 = {a: from_tar(T6, f'n6out/{a}/') for a in ('lit', 'zero', 'dl')}
J8 = from_tar(T6, 'n8out/orig/'); w8 = arm(J8)
with tarfile.open(T6) as tf:
    N9 = json.load(tf.extractfile('n9out/taumax.json')); D9 = json.load(tf.extractfile('n9out/drt_taumax.json'))
g265 = {t: dict(cond=w8[t]['cond'], pack=w8[t]['pack'], cge=N9[t]['cge|265']['all'], conv=w8[t]['conv'], ecm=w8[t]['ecm'], drt=D9[t]['drt|265']['all'], lin=N9[t]['lin|265']['all']) for t in w8}
for t in w8: assert abs(N9[t]['cge|2000']['all'] - w8[t]['cge']) < 1e-9
def from_tar15(tar):
    J = {}
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            if m.isfile() and p.suffix == '.json' and m.name.startswith('n15out/orig/') and p.parent.name in ('fit', 'nodrift', 'g265'):
                J.setdefault(p.parent.name, {})[p.stem] = json.load(tf.extractfile(m))
    return {t: dict(cond=g['cond'], pack=g['pack'], cge=g['cge|265']['all'], conv=J['fit'][t]['seq']['s4786']['C1']['v2']['rmse_mv'], ecm=J['nodrift'][t]['C1']['rmse_mv'],
                    drt=g['drt|265']['all'], lin=g['lin|265']['all']) for t, g in J['g265'].items()}
n15 = from_tar15(T15)
dfn = arm(from_tar(T11, 'n11out/orig/')); dfn.update(arm(from_tar(T13, 'n13out/orig/'))); tight = arm(from_tar(T12, 'n12out/tight/'))
def row(R, others):
    conds = [c for c in CONDS if any(r['cond'] == c for r in R.values())]
    med = lambda c, e: float(np.median([r[e] for r in R.values() if r['cond'] == c]))
    cg = [med(c, 'cge') for c in conds]; bo = [min(med(c, e) for e in others) for c in conds]; nl = []
    for c in conds:
        for e in others:
            d = {}
            for r in R.values():
                if r['cond'] == c: d.setdefault(r['pack'], []).append(r['cge'] - r[e])
            dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); assert len(dd) == 12, (c, e, len(dd)); nl.append(int((dd < 0).sum()))
    return f'{min(cg):.2f}--{max(cg):.2f}', f'{min(bo):.2f}--{max(bo):.2f}', str(min(nl)), len(R)
FIVE = ('conv', 'ecm', 'drt', 'lin')
rows = [('SPMe, Table~II', base, FIVE), ('Measured spreads', arm(J6['lit']), FIVE), ('No spreads', arm(J6['zero']), FIVE),
        ('DFN', dfn, FIVE), ('DFN, tolerance $10^{-9}$\\textsuperscript{a}', tight, FIVE), ('Double layer', arm(J6['dl']), FIVE),
        ('Asymmetric kinetics\\textsuperscript{b}', asym, ('conv', 'lin')), ('2400~s drive cycle', w8, FIVE),
        ('\\quad grid to 265~s\\textsuperscript{c}', g265, FIVE), ('\\quad grid to 265~s, new units\\textsuperscript{c}', n15, FIVE),
        ('OCV table, same SOC error', dep('same'), ('conv', 'drt', 'lin', 'drtv')),
        ('OCV table, new SOC error', dep('new'), ('conv', 'drt', 'lin', 'drtv'))]
res = {}
for lab, R, oth in rows:
    a, b, n, nu = row(R, oth); res[lab] = (a, b, n); print(f'{lab:45s} units {nu:3d}  CGE {a:12s} best other {b:12s} min pairs {n}')
assert res['Measured spreads'] == res['No spreads'], (res['Measured spreads'], res['No spreads'])
res['Measured or no spreads'] = res['Measured spreads']
TABLE = ['Measured or no spreads', 'DFN', 'DFN, tolerance $10^{-9}$\\textsuperscript{a}', 'Double layer',
         'Asymmetric kinetics\\textsuperscript{b}', '2400~s drive cycle', '\\quad grid to 265~s\\textsuperscript{c}',
         '\\quad grid to 265~s, new units\\textsuperscript{c}']
lines = [f'{lab} & {res[lab][0]} & {res[lab][1]} & {res[lab][2]}\\\\' for lab in TABLE]
open('table_rob.tex', 'w').write('\\begin{tabular}{@{}lccc@{}}\\hline\nTest & \\cge{} & Best Other & Pairs\\\\\\hline\n' + '\n'.join(lines) + '\\hline\n\\end{tabular}\n')
