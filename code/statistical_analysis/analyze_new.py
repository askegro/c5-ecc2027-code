import sys, json, numpy as np
from pathlib import Path
from stats_helpers import boot_ci_median, sign_p, holm
A = dict(e2='results_confirm', raw='sup/raw', lin='posthoc_linear_rows.json', new='new', e1raw='sup/e1_raw.json')
for k, v in zip(sys.argv[1::2], sys.argv[2::2]): A[k.lstrip('-')] = v
CONDS = ('cold', 'nominal', 'hot', 'aged'); FREQS = ('0.008', '0.05', '1.0'); NEW = Path(A['new']); MSKEY = 'cge|1'
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))} if Path(d).is_dir() else {}
E2 = load(A['e2']); RAW = load(A['raw']); RICH = load(NEW / 'rich'); NOD = load(NEW / 'nodrift'); MS = load(NEW / 'ms'); IMP = load(NEW / 'imp')
LIN = {(r['tag'], r['arm']): r['linear'] for r in json.load(open(A['lin']))} if Path(A['lin']).exists() else {}
def e2(tag, arm, est, key='rmse_mv'): return E2[tag]['seq']['s4786'][arm][est][key]
def meta(tag):
    for S in (RICH, NOD, MS, IMP, RAW):
        if tag in S: return S[tag]['cond'], S[tag]['pack']
def contrast(tags, a, b):
    res, ps = {}, []
    for c in CONDS:
        d = {}
        for t in tags:
            cond, pack = meta(t)
            if cond != c: continue
            try: d.setdefault(pack, []).append((float(a(t)), float(b(t))))
            except KeyError: pass
        d = {k: v for k, v in d.items() if len(v) == 2}
        x = np.array([np.mean([u for u, _ in v]) for _, v in sorted(d.items())]); y = np.array([np.mean([w for _, w in v]) for _, v in sorted(d.items())]); dd = x - y
        if len(dd) == 0: res[c] = dict(n=0); ps.append(1.0); continue
        res[c] = dict(n=len(dd), median=float(np.median(dd)), ci95=boot_ci_median(dd), n_lower=int((dd < 0).sum()), n_higher=int((dd > 0).sum()), p=sign_p(dd), med_a=float(np.median(x)), med_b=float(np.median(y))); ps.append(res[c]['p'])
    for c, h in zip(CONDS, holm(ps)):
        if res[c]['n']: res[c]['p_holm'] = h
    return res
def show(name, res):
    print(name)
    for c, r in res.items():
        print(f"  {c:8s} n={r['n']:2d} diff {r['median']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] lower {r['n_lower']}/higher {r['n_higher']} holm={r['p_holm']:.3g} (a {r['med_a']:.3f}, b {r['med_b']:.3f})" if r['n'] else f'  {c:8s} n=0')
def unit_medians(tags, getters):
    out = {}
    for c in CONDS:
        row = {}
        for k, g in getters.items():
            v = []
            for t in tags:
                if meta(t)[0] != c: continue
                try: v.append(float(g(t)))
                except KeyError: pass
            if v: row[k] = float(np.median(v)); row[k + '_n'] = len(v)
        out[c] = row
    return out
out = {}
def block(name, tags, families, medians=None):
    if not tags: print(f'== {name}: no results'); return
    out[name] = dict(n_units=len(tags), contrasts={k: contrast(tags, a, b) for k, (a, b) in families.items()})
    if medians: out[name]['unit_medians'] = unit_medians(tags, medians)
    print(f'== {name}: {len(tags)} units')
    for k, r in out[name]['contrasts'].items(): show(k, r)
for arm in ('C1', 'C2'):
    T = [t for t in RICH if arm in RICH[t]]; g = lambda t, arm=arm: RICH[t][arm]['rmse_mv']
    fam = {'rich_vs_cge': (g, lambda t, arm=arm: e2(t, arm, 'v3')), 'rich_vs_raw': (g, lambda t, arm=arm: RAW[t][arm]['rmse_mv']), 'rich_vs_conv': (g, lambda t, arm=arm: e2(t, arm, 'v2')), 'rich_vs_linear': (g, lambda t, arm=arm: LIN[(t, arm)])}
    block(f'rich_{arm}', T, fam, dict(rich=g, cge=fam['rich_vs_cge'][1], raw=fam['rich_vs_raw'][1], conv=fam['rich_vs_conv'][1]))
    if T: out[f'rich_{arm}'].update(order_hist={str(o): sum(RICH[t][arm]['order'] == o for t in T) for o in range(1, 9)}, fail=sum(not RICH[t][arm]['success'] for t in T), median_sec=float(np.median([RICH[t][arm]['sec'] for t in T])))
T = [t for t in NOD if 'C1' in NOD[t]]; g = lambda t: NOD[t]['C1']['rmse_mv']
fam = {'nodrift_vs_raw': (g, lambda t: RAW[t]['C1']['rmse_mv']), 'nodrift_vs_cge': (g, lambda t: e2(t, 'C1', 'v3')), 'nodrift_vs_linear': (g, lambda t: LIN[(t, 'C1')]), 'nodrift_vs_conv': (g, lambda t: e2(t, 'C1', 'v2'))}
block('nodrift_C1', T, fam, dict(nodrift=g, raw=fam['nodrift_vs_raw'][1], cge=fam['nodrift_vs_cge'][1]))
if T: out['nodrift_C1'].update(order_hist={str(o): sum(NOD[t]['C1']['order'] == o for t in T) for o in range(1, 5)}, fail=sum(not NOD[t]['C1']['success'] for t in T))
T = list(MS); m = lambda arm, key='rmse_mv': (lambda t: MS[t][f'{MSKEY}|{arm}'][key])
fam = {'ms_vs_cge_C1': (m('C1'), lambda t: e2(t, 'C1', 'v3')), 'ms_vs_linear_C1': (m('C1'), lambda t: LIN[(t, 'C1')]), 'ms_vs_conv_C1': (m('C1'), lambda t: e2(t, 'C1', 'v2')),
       'ms_vs_cge_C2': (m('C2'), lambda t: e2(t, 'C2', 'v3')), 'ms_vs_conv_C2': (m('C2'), lambda t: e2(t, 'C2', 'v2')), 'ms_vs_cge_C2shared': (m('C2', 'rmse_persist_mv'), lambda t: e2(t, 'C2', 'v3', 'rmse_persist_mv')),
       'msV_vs_cge_C2': (m('V'), lambda t: e2(t, 'C2', 'v3')), 'msV_vs_cge_C1': (m('V'), lambda t: e2(t, 'C1', 'v3'))}
block('ms', T, fam)
if T: out['ms']['sensitivity_unit_medians'] = unit_medians(T, {k: (lambda t, k=k: MS[t][k]['rmse_mv']) for k in MS[T[0]] if '|' in k})
def zref(t, f): return complex(*E2[t]['zref']['z'][f]['drop_bulk'])
def zconv(t, arm, f):
    v = E2[t]['seq']['s4786'][arm]['v2']; w = 2 * np.pi * float(f); return v['r0'] + sum(r / (1 + 1j * w * tau) for r, tau in zip(v['r'], v['tau']))
ZG = dict(cge=lambda t, arm, f: complex(*IMP[t][arm]['z'][f]), conv=zconv, rich=lambda t, arm, f: complex(*RICH[t][arm]['z'][f]), nodrift=lambda t, arm, f: complex(*NOD[t][arm]['z'][f]), ms=lambda t, arm, f: complex(*MS[t][f'{MSKEY}|{arm}']['z'][f]))
def zerr(est, arm, f, part='abs'):
    def g(t):
        e = (ZG[est](t, arm, f) - zref(t, f)) / abs(zref(t, f)) * 100
        return abs(e) if part == 'abs' else (e.real if part == 're' else e.imag)
    return g
T = [t for t in IMP if t in E2]
if T:
    out['impedance'] = {}
    for arm in ('C1', 'C2'):
        for f in FREQS:
            fam = {'cge_vs_conv': (zerr('cge', arm, f), zerr('conv', arm, f))}
            if arm == 'C1': fam.update(cge_vs_ms=(zerr('cge', arm, f), zerr('ms', arm, f)), cge_vs_rich=(zerr('cge', arm, f), zerr('rich', arm, f)), cge_vs_nodrift=(zerr('cge', arm, f), zerr('nodrift', arm, f)))
            key = f'{arm}_{f}Hz'; out['impedance'][key] = dict(contrasts={k: contrast(T, a, b) for k, (a, b) in fam.items()},
                unit_medians=unit_medians(T, {f'{e}_{p}': zerr(e, arm, f, p) for e in ZG for p in ('abs', 're', 'im')}))
            print(f'== impedance {key}')
            for k, r in out['impedance'][key]['contrasts'].items(): show(k, r)
def rows(p): return {r['rep']: r for r in json.loads(Path(p).read_text())} if Path(p).exists() else {}
E1M, E1N, E1R = rows(NEW / 'ms_e1.json'), rows(NEW / 'e1_nodrift.json'), rows(A['e1raw'])
if E1M:
    cge = {k: v['cge'] for k, v in E1M.items()}; series = dict(raw={k: v['rmse_mv'] for k, v in E1R.items()}, nodrift={k: v['rmse_mv'] for k, v in E1N.items()}, **{f'ms|{k}': {r: v[k] for r, v in E1M.items()} for k in next(iter(E1M.values())) if '|' in k})
    out['e1'] = dict(cge_median=float(np.median(list(cge.values()))), n=len(cge))
    for name, s in series.items():
        reps = sorted(set(s) & set(cge))
        if not reps: continue
        d = np.array([s[r] - cge[r] for r in reps]); out['e1'][name] = dict(n=len(reps), median=float(np.median([s[r] for r in reps])), cge_median_same=float(np.median([cge[r] for r in reps])), diff_median=float(np.median(d)), ci95=boot_ci_median(d), n_cge_lower=int((d > 0).sum()), p=sign_p(d))
    print('== E1', json.dumps(out['e1'], indent=0)[:1500])
json.dump(out, open('new_contrast.json', 'w'), indent=1); print('-> new_contrast.json')
