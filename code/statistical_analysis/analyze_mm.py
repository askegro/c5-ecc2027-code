import sys, json, io, contextlib, numpy as np
from pathlib import Path
_argv = sys.argv; sys.argv = [sys.argv[0]]
with contextlib.redirect_stdout(io.StringIO()):
    from analyze_confirm import boot_ci_median, sign_p, holm
sys.argv = _argv
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'posthoc_mm'); CONDS = ('cold', 'nominal', 'hot', 'aged')
E2 = {}
def e2(tag):
    if tag not in E2: E2[tag] = json.loads((Path('results_confirm') / f'{tag}.json').read_text())['seq']
    return E2[tag]
def getter(variant):
    def g(r, arm, k):
        t = r['tag']
        if k == 'cge_s4786': return e2(t)['s4786'][arm]['v3']['rmse_mv'] if variant == 'base' else r['rec']['s4786'][arm]['rmse_mv']
        if k == 'cge_passive': return e2(t)['passive'][arm]['v3']['rmse_mv'] if variant == 'base' else r['rec']['passive'][arm]['rmse_mv']
        if k == 'cge_passive_eq': return r['rec']['passive_eq'][arm]['rmse_mv']
        if k == 'linear': return r['linear'][arm]
        if k == 'v2': return r['v2'][arm]['rmse_mv']
        raise KeyError(k)
    return g
def contrast(R, g, arm, a, b):
    res, ps = {}, []
    for c in CONDS:
        d = {}
        for r in R:
            if r['cond'] == c: d.setdefault(r['pack'], []).append((g(r, arm, a), g(r, arm, b)))
        x = np.array([np.mean([u for u, _ in v]) for _, v in sorted(d.items())]); y = np.array([np.mean([w for _, w in v]) for _, v in sorted(d.items())]); dd = x - y
        res[c] = dict(n=len(dd), median=float(np.median(dd)), ci95=boot_ci_median(dd), n_lower=int((dd < 0).sum()), n_higher=int((dd > 0).sum()), p=sign_p(dd), med_a=float(np.median(x)), med_b=float(np.median(y))); ps.append(res[c]['p'])
    for c, h in zip(CONDS, holm(ps)): res[c]['p_holm'] = h
    return res
out = {}
for variant in ('hyst', 'asym', 'base'):
    R = [json.loads(p.read_text()) for p in sorted((ROOT / variant).glob('*.json'))]
    if not R: continue
    g = getter(variant)
    pairs = dict(cge_vs_passive=('cge_s4786', 'cge_passive'), cge_vs_passive_eq=('cge_s4786', 'cge_passive_eq'))
    if variant != 'base': pairs.update(cge_vs_v2=('cge_s4786', 'v2'), cge_vs_linear=('cge_s4786', 'linear'))
    if variant == 'base': pairs = dict(cge_vs_passive_eq=pairs['cge_vs_passive_eq'], passive_eq_vs_passive=('cge_passive_eq', 'cge_passive'))
    out[variant] = dict(n_units=len(R), contrasts={f'{k}_{arm}': contrast(R, g, arm, a, b) for arm in ('C1', 'C2') for k, (a, b) in pairs.items()})
    print(f'== {variant}: {len(R)} units')
    for arm in ('C1', 'C2'):
        for c in CONDS:
            cs = [r for r in R if r['cond'] == c]
            ks = ['cge_s4786', 'cge_passive', 'cge_passive_eq'] + (['linear', 'v2'] if variant != 'base' else [])
            print(arm, c, 'unit medians', {k: round(float(np.median([g(r, arm, k) for r in cs])), 3) for k in ks})
    for k, res in out[variant]['contrasts'].items():
        print(k)
        for c, r in res.items(): print(f"  {c:8s} diff {r['median']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] lower {r['n_lower']}/higher {r['n_higher']} holm={r['p_holm']:.3g} (a {r['med_a']:.3f}, b {r['med_b']:.3f})")
json.dump(out, open('posthoc_mm_contrast.json', 'w'), indent=1)
