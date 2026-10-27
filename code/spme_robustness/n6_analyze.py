import sys, json, numpy as np
from pathlib import Path
from stats_helpers import boot_ci_median, sign_p, holm
CONDS = ('cold', 'nominal', 'hot', 'aged'); FREQS = ('0.008', '0.05', '1.0'); EST = ('conv', 'ecm', 'drt', 'lin', 'cge'); MSKEY = 'cge|1|C1'
args = {'arm': []}
for k, v in zip(sys.argv[1::2], sys.argv[2::2]):
    k = k.lstrip('-'); (args['arm'].append(v) if k == 'arm' else args.__setitem__(k, v))
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))} if Path(d).is_dir() else {}
def zv(p): return complex(*p)
def zconv(v, f): w = 2 * np.pi * float(f); return v['r0'] + sum(r / (1 + 1j * w * tau) for r, tau in zip(v['r'], v['tau']))
def rows_arm(d):
    d = Path(d); FIT, NOD, MS = load(d / 'fit'), load(d / 'nodrift'), load(d / 'ms'); out = {}
    for t, r in FIT.items():
        if t not in NOD or t not in MS or 'C1' not in NOD[t]: continue
        s = r['seq']['s4786']['C1']; m = r['meta']
        out[t] = dict(cond=m['cond'], pack=m['pack'], spread=m.get('spread'),
                      rmse=dict(conv=s['v2']['rmse_mv'], ecm=NOD[t]['C1']['rmse_mv'], drt=MS[t][MSKEY]['rmse_mv'], lin=r['linear']['C1']['rmse_mv'], cge=s['v3']['rmse_mv']),
                      z={f: dict(ref=zv(r['zref']['z'][f]['drop_bulk']), conv=zconv(s['v2'], f), ecm=zv(NOD[t]['C1']['z'][f]), drt=zv(MS[t][MSKEY]['z'][f]),
                                 lin=zv(r['linear']['C1']['z'][f]), cge=zv(s['v3']['z'][f])) for f in FREQS},
                      ok=dict(cge_repeat=s['v3']['repeat_identical'], conv_success=s['v2']['success'], ecm_success=NOD[t]['C1']['success']))
    return out
def rows_orig(e2, new, lin):
    E2, NOD, MS, IMP = load(e2), load(Path(new) / 'nodrift'), load(Path(new) / 'ms'), load(Path(new) / 'imp')
    LIN = {r['tag']: r['linear'] for r in json.load(open(lin)) if r['arm'] == 'C1'}; out = {}
    for t, r in E2.items():
        if t not in NOD or t not in MS or t not in IMP or t not in LIN: continue
        s = r['seq']['s4786']['C1']; m = r['meta']
        out[t] = dict(cond=m['cond'], pack=m['pack'], spread='paper',
                      rmse=dict(conv=s['v2']['rmse_mv'], ecm=NOD[t]['C1']['rmse_mv'], drt=MS[t][MSKEY]['rmse_mv'], lin=LIN[t], cge=s['v3']['rmse_mv']),
                      z={f: dict(ref=zv(r['zref']['z'][f]['drop_bulk']), conv=zconv(s['v2'], f), ecm=zv(NOD[t]['C1']['z'][f]), drt=zv(MS[t][MSKEY]['z'][f]),
                                 cge=zv(IMP[t]['C1']['z'][f])) for f in FREQS})
    return out
def zerr(row, e, f): z = row['z'][f]; return abs(z[e] - z['ref']) / abs(z['ref']) * 100
def packs(R, cond, g):
    d = {}
    for t, r in R.items():
        if r['cond'] != cond: continue
        try: d.setdefault(r['pack'], []).append(float(g(t, r)))
        except KeyError: pass
    return {k: float(np.mean(v)) for k, v in d.items() if len(v) == 2}
def contrast(Ra, ga, Rb, gb):
    res, ps = {}, []
    for c in CONDS:
        a, b = packs(Ra, c, ga), packs(Rb, c, gb); k = sorted(set(a) & set(b)); dd = np.array([a[i] - b[i] for i in k])
        if len(dd) == 0: res[c] = dict(n=0); ps.append(1.0); continue
        res[c] = dict(n=len(dd), median=float(np.median(dd)), ci95=boot_ci_median(dd), n_lower=int((dd < 0).sum()), n_higher=int((dd > 0).sum()), p=sign_p(dd)); ps.append(res[c]['p'])
    for c, h in zip(CONDS, holm(ps)):
        if res[c]['n']: res[c]['p_holm'] = h
    return res
def unit_med(R, g): return {c: (float(np.median(v)) if (v := [g(t, r) for t, r in R.items() if r['cond'] == c]) else float('nan')) for c in CONDS}
def summary(R):
    S = dict(n_units=len(R), rmse_unit_median={e: unit_med(R, lambda t, r, e=e: r['rmse'][e]) for e in EST})
    S['cge_minus'] = {e: contrast(R, lambda t, r: r['rmse']['cge'], R, lambda t, r, e=e: r['rmse'][e]) for e in EST[:-1]}
    S['agree'] = {k: contrast(R, lambda t, r, a=a: r['rmse'][a], R, lambda t, r, b=b: r['rmse'][b]) for k, (a, b) in dict(ecm_lin=('ecm', 'lin'), drt_lin=('drt', 'lin'), ecm_drt=('ecm', 'drt')).items()}
    S['agree_max_abs_median'] = max((abs(v[c]['median']) for v in S['agree'].values() for c in CONDS if v[c]['n']), default=float('nan'))
    zest = [e for e in EST if all(e in r['z'][FREQS[0]] for r in R.values())]
    S['z_unit_median'] = {f: {e: unit_med(R, lambda t, r, e=e, f=f: zerr(r, e, f)) for e in zest} for f in FREQS}
    S['z_cge_minus'] = {f: {e: contrast(R, lambda t, r, f=f: zerr(r, 'cge', f), R, lambda t, r, e=e, f=f: zerr(r, e, f)) for e in zest if e != 'cge'} for f in FREQS}
    return S
def show(name, S):
    print(f'== {name}: {S["n_units"]} units. Median RMSE over units (mV), known OCV')
    print('   ' + ' '.join(f'{e:>7s}' for e in EST) + '   CGE-ECM [95% CI]        CGE lower than conv/ecm/drt/lin (of 12)')
    for c in CONDS:
        ce = S['cge_minus']['ecm'][c]
        if not ce['n']: print(f'{c[:3]}  no complete pack'); continue
        print(f'{c[:3]} ' + ' '.join(f'{S["rmse_unit_median"][e][c]:7.2f}' for e in EST) + f'   {ce["median"]:+.2f} [{ce["ci95"][0]:+.2f},{ce["ci95"][1]:+.2f}]   ' + ' / '.join(str(S['cge_minus'][e][c]['n_lower']) for e in EST[:-1]))
    print(f'   max Holm p (CGE vs each) {max((S["cge_minus"][e][c]["p_holm"] for e in EST[:-1] for c in CONDS if S["cge_minus"][e][c]["n"]), default=float("nan")):.4f}; linear estimators agree within {S["agree_max_abs_median"]:.3f} mV (max |median pair difference|)')
    for f in FREQS:
        if not any(S['z_cge_minus'][f][e][c]['n'] for e in S['z_cge_minus'][f] for c in CONDS): continue
        um = S['z_unit_median'][f]; others = [um[e][c] for e in um if e != 'cge' for c in CONDS if um[e][c] == um[e][c]]; cg = [v for v in um['cge'].values() if v == v]
        print(f'   |Z error| % at {f} Hz: CGE {min(cg):.1f}-{max(cg):.1f}, others {min(others):.1f}-{max(others):.1f}; CGE lower in min '
              f'{min(S["z_cge_minus"][f][e][c]["n_lower"] for e in S["z_cge_minus"][f] for c in CONDS if S["z_cge_minus"][f][e][c]["n"])} of the packs')
OUT = {}
ORIG = rows_orig(args['e2'], args['new'], args['lin']) if all(k in args for k in ('e2', 'new', 'lin')) else {}
if ORIG: OUT['paper'] = summary(ORIG); show('paper spreads (stored results)', OUT['paper'])
for d in args['arm']:
    R = rows_arm(d); name = Path(d).name; spreads = {r['spread'] for r in R.values()}
    assert spreads <= {name}, f'{d}: traces of arm {spreads}, folder {name}'
    bad = {k: sum(not r['ok'][k] for r in R.values()) for k in ('cge_repeat', 'conv_success', 'ecm_success')}
    OUT[name] = summary(R); OUT[name]['checks'] = bad; show(f'arm {name}', OUT[name]); print(f'   checks (count of failures): {bad}')
    if ORIG:
        OUT[name]['vs_paper'] = {e: contrast(R, lambda t, r, e=e: r['rmse'][e], ORIG, lambda t, r, e=e: r['rmse'][e]) for e in EST}
        OUT[name]['gain_change'] = {e: contrast(R, lambda t, r, e=e: r['rmse'][e] - r['rmse']['cge'], ORIG, lambda t, r, e=e: r['rmse'][e] - r['rmse']['cge']) for e in EST[:-1]}
        print(f'   arm {name} minus paper spreads, same packs and seeds: median pack difference (mV) [lower/higher of 12]')
        for e in EST: print(f'     {e:5s} ' + '  '.join(f"{c[:3]} {v['median']:+.3f} [{v['n_lower']}/{v['n_higher']}]" if v['n'] else f'{c[:3]} n/a' for c, v in OUT[name]['vs_paper'][e].items()))
        print('   change of the CGE gain (X - CGE), arm minus paper spreads:')
        for e in EST[:-1]: print(f'     {e:5s} ' + '  '.join(f"{c[:3]} {v['median']:+.3f}" if v['n'] else f'{c[:3]} n/a' for c, v in OUT[name]['gain_change'][e].items()))
json.dump(OUT, open('n6_contrast.json', 'w'), indent=1); print('-> n6_contrast.json')
