import sys, json, io, contextlib, numpy as np
from pathlib import Path
_argv = sys.argv; sys.argv = [sys.argv[0]]
with contextlib.redirect_stdout(io.StringIO()):
    from analyze_confirm import boot_ci_median, sign_p, holm
sys.argv = _argv
CONDS = ('cold', 'nominal', 'hot', 'aged')
lin_rows = {(r['tag'], r['arm']): r for r in json.load(open('posthoc_linear_rows.json'))}
def e2(tag): return json.loads((Path('results_confirm') / f'{tag}.json').read_text())['seq']['s4786']
def load(d): return [json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))]
def contrast(R, a, b):
    res, ps = {}, []
    for c in CONDS:
        d = {}
        for r in R:
            if r['cond'] == c: d.setdefault(r['pack'], []).append((a(r), b(r)))
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
        if r['n']: print(f"  {c:8s} n={r['n']:2d} diff {r['median']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] lower {r['n_lower']}/higher {r['n_higher']} holm={r['p_holm']:.3g} (a {r['med_a']:.3f}, b {r['med_b']:.3f})")
        else: print(f'  {c:8s} n=0')
out = {}
S = load('sup/soc1')
if S:
    out['soc1'] = dict(n_units=len(S), contrasts={
        'cge_vs_v2': contrast(S, lambda r: r['full'], lambda r: r['v2']), 'cge_vs_linear': contrast(S, lambda r: r['full'], lambda r: r['linear']),
        'cge_vs_nooffset': contrast(S, lambda r: r['full'], lambda r: r['nooffset']), 'nooffset_vs_v2': contrast(S, lambda r: r['nooffset'], lambda r: r['v2'])},
        unit_medians={c: {k: float(np.median([r[k] for r in S if r['cond'] == c])) for k in ('full', 'linear', 'v2', 'nooffset')} for c in CONDS if any(r['cond'] == c for r in S)},
        v2_order_hist={str(o): sum(r['v2_order'] == o for r in S) for o in (1, 2, 3, 4)}, v2_fail=sum(not r['v2_success'] for r in S))
    print(f"== soc1: {len(S)} units"); print('unit medians', json.dumps(out['soc1']['unit_medians']))
    for k, res in out['soc1']['contrasts'].items(): show(k, res)
for arm in ('C1', 'C2'):
    R = [r for r in load('sup/raw') if arm in r]
    if not R: continue
    for r in R: r['_arm'] = arm
    g = dict(raw=lambda r: r[arm]['rmse_mv'], cge=lambda r: e2(r['tag'])[arm]['v3']['rmse_mv'], v2=lambda r: e2(r['tag'])[arm]['v2']['rmse_mv'], linear=lambda r: lin_rows[(r['tag'], arm)]['linear'])
    out[f'raw_{arm}'] = dict(n_units=len(R), contrasts={f'raw_vs_{k}': contrast(R, g['raw'], g[k]) for k in ('cge', 'v2', 'linear')},
        unit_medians={c: {k: float(np.median([g[k](r) for r in R if r['cond'] == c])) for k in g} for c in CONDS if any(r['cond'] == c for r in R)},
        order_hist={str(o): sum(r[arm]['order'] == o for r in R) for o in (1, 2, 3, 4)}, fail=sum(not r[arm]['success'] for r in R), median_sec=float(np.median([r[arm]['sec'] for r in R])))
    print(f"== raw {arm}: {len(R)} units; orders {out[f'raw_{arm}']['order_hist']}, failures {out[f'raw_{arm}']['fail']}, median sec {out[f'raw_{arm}']['median_sec']:.0f}"); print('unit medians', json.dumps(out[f'raw_{arm}']['unit_medians']))
    for k, res in out[f'raw_{arm}']['contrasts'].items(): show(f'{arm} {k}', res)
json.dump(out, open('sup_contrast.json', 'w'), indent=1)
