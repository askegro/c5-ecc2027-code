import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

RC = Path(sys.argv[1] if len(sys.argv) > 1 else 'results_confirm'); RE = Path(sys.argv[2] if len(sys.argv) > 2 else 'results_e1')
CONDS = ('cold', 'nominal', 'hot', 'aged'); SEQS = ('s4786', 's9588', 'short74', 'passive')
NBOOT = 10_000


def boot_ci_median(x, seed=20261001):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(NBOOT, len(x)))
    return [float(v) for v in np.quantile(np.median(x[idx], axis=1), [0.025, 0.975])]


def sign_p(d):
    d = np.asarray(d, float); neg, pos = int(np.sum(d < 0)), int(np.sum(d > 0))
    return 1.0 if neg + pos == 0 else float(binomtest(neg, neg + pos, 0.5).pvalue)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, k in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[k]); adj[k] = min(1.0, run)
    return adj.tolist()


def contrast(d):
    d = np.asarray(d, float)
    return dict(n=int(len(d)), median=float(np.median(d)), ci95=boot_ci_median(d), n_lower=int(np.sum(d < 0)), n_higher=int(np.sum(d > 0)), sign_p=sign_p(d))


cells = [json.loads(p.read_text()) for p in sorted(RC.glob('*.json'))]
out = dict(E2=dict(n_cells=len(cells)), E1={})
by = {}
for c in cells:
    by.setdefault((c['meta']['cond'], c['meta']['pack']), []).append(c)
out['E2']['packs_per_condition'] = {cond: len([k for k in by if k[0] == cond]) for cond in CONDS}
out['E2']['cells_per_condition'] = {cond: sum(len(v) for k, v in by.items() if k[0] == cond) for cond in CONDS}


def val(c, seq, arm, est, key='rmse_mv'):
    return c['seq'][seq][arm][est][key]


def pack_values(cond, fn):
    return np.array([np.mean([fn(c) for c in by[k]]) for k in sorted(by) if k[0] == cond])


med = {}
for cond in CONDS:
    cs = [c for c in cells if c['meta']['cond'] == cond]
    if not cs: continue
    med[cond] = {}
    for arm in ('C1', 'C2'):
        m = {}
        for s in SEQS:
            m[f'v3_{s}'] = float(np.median([val(c, s, arm, 'v3') for c in cs]))
            if 'v2' in cs[0]['seq'][s][arm]:
                m[f'v2_{s}'] = float(np.median([val(c, s, arm, 'v2') for c in cs]))
        m['v3_long'] = m['v3_s4786']; m['v2_long'] = m['v2_s4786']
        if arm == 'C2':
            m['v3_long_persist'] = float(np.median([val(c, 's4786', arm, 'v3', 'rmse_persist_mv') for c in cs]))
            m['v2_long_persist'] = float(np.median([val(c, 's4786', arm, 'v2', 'rmse_persist_mv') for c in cs]))
            m['v3_long_offset_discarded'] = float(np.median([val(c, 's4786', arm, 'v3', 'rmse_nooffset_pred_mv') for c in cs]))
            m['offset_b_mv'] = float(np.median([1e3 * val(c, 's4786', arm, 'v3', 'b') for c in cs]))
        med[cond][arm] = m
out['E2']['cell_medians_mv'] = med

fam = {
    'F1_estimator_C2': lambda c: val(c, 's4786', 'C2', 'v3') - val(c, 's4786', 'C2', 'v2'),
    'F2_estimator_C1': lambda c: val(c, 's4786', 'C1', 'v3') - val(c, 's4786', 'C1', 'v2'),
    'F3_excitation_C1': lambda c: val(c, 's4786', 'C1', 'v3') - val(c, 'passive', 'C1', 'v3'),
    'F4_selection_C1': lambda c: val(c, 's4786', 'C1', 'v3') - val(c, 's9588', 'C1', 'v3'),
    'F5_duration_C1': lambda c: val(c, 's4786', 'C1', 'v3') - val(c, 'short74', 'C1', 'v3'),
}
sec = {
    'S3_excitation_C2': lambda c: val(c, 's4786', 'C2', 'v3') - val(c, 'passive', 'C2', 'v3'),
    'S4_selection_C2': lambda c: val(c, 's4786', 'C2', 'v3') - val(c, 's9588', 'C2', 'v3'),
    'S5_duration_C2': lambda c: val(c, 's4786', 'C2', 'v3') - val(c, 'short74', 'C2', 'v3'),
    'S1_estimator_C2_persistent_soc': lambda c: val(c, 's4786', 'C2', 'v3', 'rmse_persist_mv') - val(c, 's4786', 'C2', 'v2', 'rmse_persist_mv'),
}
for group, defs in (('primary', fam), ('secondary', sec)):
    out['E2'][group] = {}
    for name, fn in defs.items():
        res = {cond: contrast(pack_values(cond, fn)) for cond in CONDS if any(k[0] == cond for k in by)}
        adj = holm([res[c]['sign_p'] for c in res])
        for c, a in zip(res, adj): res[c]['holm_p'] = a
        res['_pack_values'] = {cond: pack_values(cond, fn).tolist() for cond in CONDS if any(k[0] == cond for k in by)}
        out['E2'][group][name] = res


byunit = {}
for name, other in (('selection', 's9588'), ('duration', 'short74'), ('excitation', 'passive')):
    byunit[name] = {}
    for cond in CONDS:
        for u in (0, 1):
            d = [val(c, 's4786', 'C1', 'v3') - val(c, other, 'C1', 'v3') for c in cells if c['meta']['cond'] == cond and c['meta']['cell'] == u]
            if d: byunit[name][f'{cond}_unit{u}'] = contrast(d)
    for u in (0, 1):
        d = [val(c, 's4786', 'C1', 'v3') - val(c, other, 'C1', 'v3') for c in cells if c['meta']['cell'] == u]
        byunit[name][f'all_unit{u}'] = contrast(d)
    d = [np.mean([val(c, 's4786', 'C1', 'v3') - val(c, other, 'C1', 'v3') for c in by[k]]) for k in sorted(by)]
    byunit[name]['all_packs'] = contrast(d)
out['E2']['by_unit'] = byunit
out['E2']['p90_mv'] = {cond: {s: float(np.quantile([val(c, s, 'C1', 'v3') for c in cells if c['meta']['cond'] == cond], 0.9)) for s in SEQS} for cond in CONDS if any(c['meta']['cond'] == cond for c in cells)}
out['E2']['max_mv'] = {cond: {s: float(np.max([val(c, s, 'C1', 'v3') for c in cells if c['meta']['cond'] == cond])) for s in SEQS} for cond in CONDS if any(c['meta']['cond'] == cond for c in cells)}

imp = {}
for cond in CONDS:
    cs = [c for c in cells if c['meta']['cond'] == cond]
    if not cs: continue
    imp[cond] = {}
    for f in ('0.008', '0.05', '1.0'):
        row = {}
        for est, ss in (('v3', ('s4786', 's9588')), ('v2', ('s4786',))):
            for s in ss:
                e = [100 * (c['seq'][s]['C1'][est]['rez'][f] - c['zref']['z'][f]['drop_bulk'][0]) / c['zref']['z'][f]['drop_bulk'][0] for c in cs]
                row[f'{est}_{s}_median_err_pct'] = float(np.median(e)); row[f'{est}_{s}_median_abs_err_pct'] = float(np.median(np.abs(e)))
        e = [100 * (c['zref']['pybamm_eis'][['0.008', '0.05', '1.0'].index(f)][0] - c['zref']['z'][f]['drop_bulk'][0]) / c['zref']['z'][f]['drop_bulk'][0] for c in cs]
        row['pybamm_eis_vs_time_domain_median_pct'] = float(np.median(e))
        imp[cond][f] = row
    for a in ('0.5', '20.0'):
        e = [100 * (c['seq']['s4786']['C1']['v3']['chord'][a] - c['zref']['chord_inst_ohm'][a]) / c['zref']['chord_inst_ohm'][a] for c in cs]
        imp[cond][f'v3_chord_{a}A_median_err_pct'] = float(np.median(e))
    imp[cond]['truth_chord_ratio_20A_over_small_signal'] = float(np.median([c['zref']['chord_inst_ohm']['20.0'] / c['zref']['chord_inst_ohm']['0.5'] for c in cs]))
    imp[cond]['truth_small_signal_inst_mohm'] = float(np.median([1e3 * c['zref']['chord_inst_ohm']['0.5'] for c in cs]))
out['E2']['impedance'] = imp

kk = [c['seq'][s][arm]['v3']['kkt'] for c in cells for s in SEQS for arm in ('C1', 'C2')]
rep = [c['seq'][s][arm]['v3']['repeat_identical'] for c in cells for s in SEQS for arm in ('C1', 'C2')]
sec3 = [c['seq'][s][arm]['v3']['sec'] for c in cells for s in SEQS for arm in ('C1', 'C2')]
v2 = [c['seq']['s4786'][arm]['v2'] for c in cells for arm in ('C1', 'C2')]
out['E2']['reliability'] = dict(v3_fits=len(kk), v3_max_kkt=float(np.max(kk)), v3_all_certified=bool(np.max(kk) < 1e-9), v3_all_repeat_identical=bool(all(rep)),
                                v3_median_sec=float(np.median(sec3)), v2_fits=len(v2), v2_median_sec=float(np.median([x['sec'] for x in v2])),
                                v2_slow_branch_refits=int(sum(x['refit'] for x in v2)), v2_orders={str(o): int(sum(x['order'] == o for x in v2)) for o in (1, 2, 3, 4)},
                                v3_wins_vs_v2=int(sum(c['seq']['s4786'][arm]['v3']['rmse_mv'] < c['seq']['s4786'][arm]['v2']['rmse_mv'] for c in cells for arm in ('C1', 'C2'))),
                                v3_median_active_branches=float(np.median([c['seq'][s][arm]['v3']['n_active'] for c in cells for s in SEQS for arm in ('C1', 'C2')])),
                                v3_median_train_rmse_mv=float(np.median([c['seq'][s]['C1']['v3']['train_rmse_mv'] for c in cells for s in ('s4786', 's9588')])))

if (RE / 'e1_v3.json').exists():
    r3 = json.loads((RE / 'e1_v3.json').read_text())
    e = {s: np.array([r[s]['rmse_mv'] for r in r3]) for s in SEQS}
    E1 = dict(n_v3=len(r3), v3_median_mv={s: float(np.median(e[s])) for s in SEQS},
              v3_p10_p90_mv={s: [float(v) for v in np.quantile(e[s], [0.1, 0.9])] for s in SEQS},
              a_v3_4786_minus_9588=contrast(e['s4786'] - e['s9588']), c_v3_4786_minus_passive=contrast(e['s4786'] - e['passive']),
              d_v3_4786_minus_short=contrast(e['s4786'] - e['short74']), v3_max_kkt=float(max(r[s]['kkt'] for r in r3 for s in SEQS)))
    if (RE / 'e1_v2.json').exists():
        r2 = json.loads((RE / 'e1_v2.json').read_text()); n2 = len(r2)
        e2 = {s: np.array([r[s]['rmse_mv'] for r in r2]) for s in ('s4786', 's9588')}
        e3 = {s: e[s][:n2] for s in ('s4786', 's9588')}
        E1.update(n_v2=n2, v2_median_mv={s: float(np.median(e2[s])) for s in e2},
                  b_v3_minus_v2=contrast(0.5 * (e3['s4786'] + e3['s9588']) - 0.5 * (e2['s4786'] + e2['s9588'])),
                  v2_4786_minus_9588=contrast(e2['s4786'] - e2['s9588']), v2_orders={str(o): int(sum(r[s]['order'] == o for r in r2 for s in ('s4786', 's9588'))) for o in (1, 2, 3, 4)},
                  v2_refits=int(sum(r[s]['refit'] for r in r2 for s in ('s4786', 's9588'))))
    out['E1'] = E1

Path('results_summary.json').write_text(json.dumps(out, indent=1))

print(f"E2: {len(cells)} cells; packs per condition {out['E2']['packs_per_condition']}")
print("\nCell medians (mV): v2 / v3 on the information sequence")
for cond in med:
    for arm in ('C1', 'C2'):
        m = med[cond][arm]; extra = f" | persistent SOC error: v2 {m['v2_long_persist']:.2f} v3 {m['v3_long_persist']:.2f}; v3 offset discarded {m['v3_long_offset_discarded']:.2f}; b {m['offset_b_mv']:.2f} mV" if arm == 'C2' else ''
        print(f"  {cond:8s} {arm}: v2 {m['v2_long']:.3f}  v3 {m['v3_long']:.3f} | v3 by sequence: " + ", ".join(f"{s} {m['v3_' + s]:.3f}" for s in SEQS) + extra)
for group in ('primary', 'secondary'):
    print(f"\n{group.upper()} contrasts (pack level): median [95% CI], packs lower/higher, sign p, Holm p")
    for name, res in out['E2'][group].items():
        for cond in CONDS:
            if cond in res:
                r = res[cond]
                print(f"  {name:32s} {cond:8s} {r['median']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  {r['n_lower']}/{r['n_higher']} of {r['n']}  p={r['sign_p']:.4f}  Holm={r['holm_p']:.4f}")
print("\nBy unit (descriptive), information minus comparator, convex, known OCV:")
for name, d in out['E2']['by_unit'].items():
    for k, r in d.items():
        if k.startswith('all'):
            print(f"  {name:10s} {k:12s} {r['median']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  lower {r['n_lower']}/{r['n']}  p={r['sign_p']:.4f}")
print("p90 over units (mV):", json.dumps({c: {s: round(v, 3) for s, v in d.items()} for c, d in out['E2']['p90_mv'].items()}))
print("\nImpedance (median signed error of small-signal Re Z, %, C1 fits):")
for cond, d in imp.items():
    print(f"  {cond:8s} " + "; ".join(f"{f} Hz: v3 {d[f]['v3_s4786_median_err_pct']:+.2f} v2 {d[f]['v2_s4786_median_err_pct']:+.2f} (PyBaMM EIS vs time domain {d[f]['pybamm_eis_vs_time_domain_median_pct']:+.2f})" for f in ('0.008', '0.05', '1.0'))
          + f" | chord err 0.5 A {d['v3_chord_0.5A_median_err_pct']:+.2f}%, 20 A {d['v3_chord_20.0A_median_err_pct']:+.2f}%; truth chord(20A)/small-signal {d['truth_chord_ratio_20A_over_small_signal']:.3f}")
print("\nReliability:", json.dumps(out['E2']['reliability']))
if out['E1']:
    print("\nE1:", json.dumps(out['E1'], indent=1))
