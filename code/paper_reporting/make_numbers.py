import json
import warnings
from pathlib import Path
import numpy as np
import cvx_estimator as ce

S = json.loads(Path('results_summary.json').read_text())
Dg = json.loads(Path('results_diagnosis.json').read_text())
OUT = Path('manuscript'); OUT.mkdir(exist_ok=True)
N = {}


def put(key, value):
    assert key not in N, key
    N[key] = value


def f(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return s.replace('-', r'\ensuremath{-}').replace('+', r'\ensuremath{+}')


def p_fmt(p):
    if p > 0.9995:
        return "1"
    return f"{p:.3f}" if p >= 0.001 else r"\ensuremath{<}0.001"


CONDS = ('cold', 'nominal', 'hot', 'aged'); LAB = dict(cold='Cold', nominal='Nominal', hot='Hot', aged='Aged')
SH = dict(cold='C', nominal='N', hot='H', aged='A')
E2, E1 = S['E2'], S['E1']
put('J', str(len(ce.tau_grid())))
put('E2.ncells', str(E2['n_cells'])); put('E2.npacks', str(sum(E2['packs_per_condition'].values())))
put('E2.packspercond', str(min(E2['packs_per_condition'].values())))

put('E1.n3', str(E1['n_v3'])); put('E1.n2', str(E1['n_v2']))
for s, k in (('s4786', 'info'), ('s9588', 'cov'), ('short74', 'short'), ('passive', 'fixed')):
    put(f'E1.v3.{k}', f(E1['v3_median_mv'][s], 3))
put('E1.v3.long', f(0.5 * (E1['v3_median_mv']['s4786'] + E1['v3_median_mv']['s9588']), 3))
put('E1.v2.info', f(E1['v2_median_mv']['s4786'], 2)); put('E1.v2.cov', f(E1['v2_median_mv']['s9588'], 2))
put('E1.v2.long', f(0.5 * (E1['v2_median_mv']['s4786'] + E1['v2_median_mv']['s9588']), 2))
for key, name in (('a_v3_4786_minus_9588', 'a'), ('c_v3_4786_minus_passive', 'c'), ('d_v3_4786_minus_short', 'd'), ('b_v3_minus_v2', 'b'), ('v2_4786_minus_9588', 'v2sel')):
    c = E1[key]; d = 2 if name == 'b' else 4
    put(f'E1.{name}.med', f(c['median'], d, True)); put(f'E1.{name}.lo', f(c['ci95'][0], d, True)); put(f'E1.{name}.hi', f(c['ci95'][1], d, True))
    put(f'E1.{name}.nlow', str(c['n_lower'])); put(f'E1.{name}.n', str(c['n'])); put(f'E1.{name}.p', p_fmt(c['sign_p']))
put('E1.ratio', f"{0.5 * (E1['v2_median_mv']['s4786'] + E1['v2_median_mv']['s9588']) / (0.5 * (E1['v3_median_mv']['s4786'] + E1['v3_median_mv']['s9588'])):.0f}")
put('E1.selpct', f"{100 * (1 - E1['v3_median_mv']['s4786'] / E1['v3_median_mv']['s9588']):.0f}")
put('E1.fixedratio', f"{E1['v3_median_mv']['passive'] / E1['v3_median_mv']['s4786']:.1f}")
put('E1.shortratio', f"{E1['v3_median_mv']['short74'] / E1['v3_median_mv']['s4786']:.1f}")

M = E2['cell_medians_mv']
for cond in CONDS:
    for arm in ('C1', 'C2'):
        m = M[cond][arm]
        put(f'E2.{cond}.{arm}.v2', f(m['v2_long'], 2)); put(f'E2.{cond}.{arm}.v3', f(m['v3_long'], 2))
        put(f'E2.{cond}.{arm}.ratio', f"{m['v2_long'] / m['v3_long']:.1f}")
        for s, k in (('s4786', 'info'), ('s9588', 'cov'), ('short74', 'short'), ('passive', 'fixed')):
            put(f'E2.{cond}.{arm}.v3.{k}', f(m['v3_' + s], 2))
    m = M[cond]['C2']
    put(f'E2.{cond}.persist.v2', f(m['v2_long_persist'], 2)); put(f'E2.{cond}.persist.v3', f(m['v3_long_persist'], 2))
    put(f'E2.{cond}.offsetdiscarded', f(m['v3_long_offset_discarded'], 2)); put(f'E2.{cond}.b', f(m['offset_b_mv'], 1, True))
for group in ('primary', 'secondary'):
    for name, res in E2[group].items():
        short = name.split('_')[0]
        for cond in CONDS:
            r = res[cond]; d = 3 if short in ('F4', 'S4', 'S6') else 2
            put(f'{short}.{cond}.med', f(r['median'], d, True)); put(f'{short}.{cond}.lo', f(r['ci95'][0], d, True)); put(f'{short}.{cond}.hi', f(r['ci95'][1], d, True))
            put(f'{short}.{cond}.nlow', str(r['n_lower'])); put(f'{short}.{cond}.n', str(r['n'])); put(f'{short}.{cond}.p', p_fmt(r['sign_p'])); put(f'{short}.{cond}.holm', p_fmt(r['holm_p']))
        put(f'{short}.maxholm', p_fmt(max(res[c]['holm_p'] for c in CONDS))); put(f'{short}.minlow', str(min(res[c]['n_lower'] for c in CONDS)))
        put(f'{short}.nreject', str(sum(res[c]['holm_p'] < 0.05 for c in CONDS)))


def rng_txt(vals, d=2):
    return f"{min(vals):.{d}f}--{max(vals):.{d}f}"


for arm in ('C1', 'C2'):
    put(f'E2.range.{arm}.v2', rng_txt([M[c][arm]['v2_long'] for c in CONDS])); put(f'E2.range.{arm}.v3', rng_txt([M[c][arm]['v3_long'] for c in CONDS]))
    put(f'E2.ratiorange.{arm}', rng_txt([M[c][arm]['v2_long'] / M[c][arm]['v3_long'] for c in CONDS], 1))
put('E2.range.persist.v3', rng_txt([M[c]['C2']['v3_long_persist'] for c in CONDS]))
put('E2.fixedratio.range', rng_txt([M[c]['C1']['v3_passive'] / M[c]['C1']['v3_s4786'] for c in CONDS], 1))
put('E2.shortratio.range', rng_txt([M[c]['C1']['v3_short74'] / M[c]['C1']['v3_s4786'] for c in CONDS], 1))
put('E2.selpct.range', "--".join(f"{v:.0f}" for v in sorted([100 * (1 - M[c]['C1']['v3_s4786'] / M[c]['C1']['v3_s9588']) for c in CONDS])[::len(CONDS) - 1]))

I = E2['impedance']
for cond in CONDS:
    for fq, k in (('0.008', 'lf'), ('0.05', 'mf'), ('1.0', 'hf')):
        put(f'imp.{cond}.{k}.v3', f(I[cond][fq]['v3_s4786_median_err_pct'], 1, True)); put(f'imp.{cond}.{k}.v2', f(I[cond][fq]['v2_s4786_median_err_pct'], 1, True))
        put(f'imp.{cond}.{k}.eis', f(I[cond][fq]['pybamm_eis_vs_time_domain_median_pct'], 1, True))
    put(f'imp.{cond}.chordratio', f(I[cond]['truth_chord_ratio_20A_over_small_signal'], 2))
    put(f'imp.{cond}.chord20', f(I[cond]['v3_chord_20.0A_median_err_pct'], 1, True)); put(f'imp.{cond}.chord05', f(I[cond]['v3_chord_0.5A_median_err_pct'], 1, True))
    put(f'imp.{cond}.rss', f(I[cond]['truth_small_signal_inst_mohm'], 2))
allv3 = [abs(I[c][fq]['v3_s4786_median_err_pct']) for c in CONDS for fq in ('0.008', '0.05', '1.0')]
allv2 = [abs(I[c][fq]['v2_s4786_median_err_pct']) for c in CONDS for fq in ('0.008', '0.05', '1.0')]
alleis = [abs(I[c][fq]['pybamm_eis_vs_time_domain_median_pct']) for c in CONDS for fq in ('0.008', '0.05', '1.0')]
put('imp.v3.maxabs', f(max(allv3), 1)); put('imp.v2.maxabs', f(max(allv2), 1)); put('imp.v2.minabs', f(min(allv2), 1)); put('imp.v3.minabs', f(min(allv3), 1))
put('imp.eis.range', rng_txt(alleis, 1))
put('imp.lf.v3.range', rng_txt([abs(I[c]['0.008']['v3_s4786_median_err_pct']) for c in CONDS], 1)); put('imp.lf.v2.range', rng_txt([abs(I[c]['0.008']['v2_s4786_median_err_pct']) for c in CONDS], 1))
put('imp.chord20.maxabs', f(max(abs(I[c]['v3_chord_20.0A_median_err_pct']) for c in CONDS), 1))

Rr = E2['reliability']
put('rel.v3fits', str(Rr['v3_fits'])); put('rel.kkt', f"{Rr['v3_max_kkt']:.0e}".replace('e-', r'\times10^{-').replace('{-0', '{-') + '}')
put('rel.v3sec', f(Rr['v3_median_sec'], 2)); put('rel.v2sec', f(Rr['v2_median_sec'], 0)); put('rel.v2fits', str(Rr['v2_fits']))
put('rel.v2refits', str(Rr['v2_slow_branch_refits'])); put('rel.wins', str(Rr['v3_wins_vs_v2']))
put('rel.nactive', f"{Rr['v3_median_active_branches']:.0f}"); put('rel.train', f(Rr['v3_median_train_rmse_mv'], 2))
put('rel.v2order1', str(Rr['v2_orders']['1'])); put('rel.identical', 'all' if Rr['v3_all_repeat_identical'] else 'NOT ALL')
put('rel.certified', 'all' if Rr['v3_all_certified'] else 'NOT ALL')
put('rel.totalfits', str(Rr['v3_fits'] + 4 * E1['n_v3']))

A = Dg['A_forward_consistency']
put('diag.forward', f(np.mean([A[s]['v2_lowband_residual_at_truth_mv'] for s in A], ), 1))
put('diag.forwardnofast', f(np.mean([A[s]['with_fast_branch_folded_into_r0_mv'] for s in A]), 1))
C = {(r['cond'], r['arm']): r for r in Dg['C_summary']}
for cond in CONDS:
    put(f'diag.lin.{cond}', f(C[(cond, 'C1')]['v3_linear'], 2)); put(f'diag.full.{cond}', f(C[(cond, 'C1')]['v3'], 2))
    put(f'diag.nob.{cond}', f(C[(cond, 'C2')]['v3_offset_discarded'], 2)); put(f'diag.b.{cond}', f(C[(cond, 'C2')]['v3'], 2))
put('diag.bhot', f(C[('hot', 'C2')]['b_mv'], 1, True))
B = Dg['B_summary']
put('diag.seedcount', str(B['n_spread_above_0p1'])); put('diag.seedn', str(B['n'])); put('diag.seedmax', f(B['spread_mv_max'], 2)); put('diag.seedorder', str(B['n_order_differs']))


BU = E2['by_unit']
for name, k in (('selection', 'sel'), ('duration', 'dur'), ('excitation', 'exc')):
    for u in ('all_unit0', 'all_unit1', 'all_packs'):
        r = BU[name][u]; d = 3 if k == 'sel' else 2; kk = u.replace('all_', '')
        put(f'BU.{k}.{kk}.med', f(r['median'], d, True)); put(f'BU.{k}.{kk}.lo', f(r['ci95'][0], d, True)); put(f'BU.{k}.{kk}.hi', f(r['ci95'][1], d, True))
        put(f'BU.{k}.{kk}.nlow', str(r['n_lower'])); put(f'BU.{k}.{kk}.nhigh', str(r['n_higher'])); put(f'BU.{k}.{kk}.n', str(r['n']))
P90 = E2['p90_mv']
for cond in CONDS:
    for s_, k in (('s4786', 'info'), ('s9588', 'cov'), ('short74', 'short'), ('passive', 'fixed')):
        put(f'p90.{cond}.{k}', f(P90[cond][s_], 2))
put('p90.info.range', rng_txt([P90[c]['s4786'] for c in CONDS])); put('p90.short.range', rng_txt([P90[c]['short74'] for c in CONDS]))
put('p90.shortratio.range', rng_txt([P90[c]['short74'] / P90[c]['s4786'] for c in CONDS], 1))
put('F3.absrange', rng_txt([abs(E2['primary']['F3_excitation_C1'][c]['median']) for c in CONDS]))
put('F5.absrange', rng_txt([abs(E2['primary']['F5_duration_C1'][c]['median']) for c in CONDS]))
put('F4.absmax', f(max(abs(E2['primary']['F4_selection_C1'][c]['median']) for c in CONDS), 3))
put('E2.range.persist.v2', rng_txt([M[c]['C2']['v2_long_persist'] for c in CONDS]))
put('rel.winspct', f"{100 * Rr['v3_wins_vs_v2'] / Rr['v2_fits']:.0f}")


put('imp.lf.v3.max', f(max(abs(I[c]['0.008']['v3_s4786_median_err_pct']) for c in CONDS), 1))
put('imp.lf.v2.max', f(max(abs(I[c]['0.008']['v2_s4786_median_err_pct']) for c in CONDS), 1))
put('offset.cost', f(max(M[c]['C2']['v3_long'] - M[c]['C2']['v3_long_offset_discarded'] for c in CONDS), 2))
T = json.loads(Path('timing_idle.json').read_text())
put('rel.v3sec.idle', f(T['v3_median_sec'], 2))
Sn = json.loads(Path('results_sensitivity.json').read_text())
structural = [v for v in Sn if v != 'default' and not v.startswith('eps')]
put('sens.maxmedian', f(max(abs(Sn[v][f'{c}_C1_median'] - Sn['default'][f'{c}_C1_median']) for v in structural for c in CONDS), 3))
put('sens.maxmedian.depl', f(max(abs(Sn[v][f'{c}_C2_median'] - Sn['default'][f'{c}_C2_median']) for v in structural for c in CONDS), 2))
put('sens.maxunit', f(max(Sn[v]['C1_max_abs_change_vs_default'] for v in structural), 2))


Ab = json.loads(Path('results_ablation.json').read_text())
for cond in CONDS:
    put(f'abl.{cond}.lin', f(Ab[f'{cond}_C1']['linear'], 2)); put(f'abl.{cond}.lin.depl', f(Ab[f'{cond}_C2']['linear'], 2))


BS_path = Path('results_baseline_summary.json')
if BS_path.exists():
    BS = json.loads(BS_path.read_text()); BLm = BS['medians_mv']
    put('BL.nunits', str(BS['n_units'])); put('BL.margin', f(BS['margin_mv'], 2))
    KEY = dict(v2='v2', nls_lin='nlslin', cvx_lin='cvxlin', nls_stat='nlsstat', v3='v3')
    for cond in CONDS:
        for arm in ('C1', 'C2'):
            for e, k in KEY.items():
                put(f'BL.{cond}.{arm}.{k}', f(BLm[cond][arm][e], 2))
            put(f'BL.{cond}.{arm}.ratio', f"{BLm[cond][arm]['nls_lin'] / BLm[cond][arm]['v3']:.1f}")
        for e, k in (('nls_lin', 'nlslin'), ('nls_stat', 'nlsstat')):
            put(f'BL.{cond}.persist.{k}', f(BLm[cond]['C2_persist'][e], 2))
    for arm in ('C1', 'C2'):
        put(f'BL.range.{arm}.nlslin', rng_txt([BLm[c][arm]['nls_lin'] for c in CONDS])); put(f'BL.range.{arm}.nlsstat', rng_txt([BLm[c][arm]['nls_stat'] for c in CONDS]))
        put(f'BL.ratiorange.{arm}', rng_txt([BLm[c][arm]['nls_lin'] / BLm[c][arm]['v3'] for c in CONDS], 1))
    put('BL.range.persist.nlslin', rng_txt([BLm[c]['C2_persist']['nls_lin'] for c in CONDS]))
    for name, res in BS['families'].items():
        short = name.split('_')[0] + name[-2:]
        for cond in CONDS:
            r = res[cond]; d = 3 if short[:2] in ('G3', 'G4') else 2
            put(f'{short}.{cond}.med', f(r['median'], d, True)); put(f'{short}.{cond}.lo', f(r['ci95'][0], d, True)); put(f'{short}.{cond}.hi', f(r['ci95'][1], d, True))
            put(f'{short}.{cond}.nlow', str(r['n_lower'])); put(f'{short}.{cond}.n', str(r['n'])); put(f'{short}.{cond}.holm', p_fmt(r['holm_p']))
        put(f'{short}.maxholm', p_fmt(max(res[c]['holm_p'] for c in CONDS))); put(f'{short}.minlow', str(min(res[c]['n_lower'] for c in CONDS)))
        put(f'{short}.nreject', str(sum(res[c]['holm_p'] < 0.05 for c in CONDS)))
        put(f'{short}.maxabsmed', f(max(abs(res[c]['median']) for c in CONDS), 3)); put(f'{short}.maxabsci', f(max(max(abs(v) for v in res[c]['ci95']) for c in CONDS), 3))
        if name in BS['equal_accuracy']:
            put(f'{short}.nequal', str(sum(BS['equal_accuracy'][name][c] for c in CONDS)))
    for e, k in (('nls_lin', 'nlslin'), ('nls_stat', 'nlsstat')):
        Dd = BS['descriptive'][e]
        put(f'BL.{k}.nfits', str(Dd['n_fits'])); put(f'BL.{k}.sec', f(Dd['median_sec'], 0)); put(f'BL.{k}.multiopt', str(Dd['fits_with_more_than_one_optimum']))
        put(f'BL.{k}.maxspread', f(Dd['max_start_spread_mv'], 1)); put(f'BL.{k}.atlimit', str(Dd['fits_with_tau_at_limit']))
        put(f'BL.{k}.orders', ", ".join(f"{n_} of order {o}" for o, n_ in Dd['orders'].items()))
        put(f'BL.{k}.modeorder', max(Dd['orders'], key=Dd['orders'].get))
    baseline_dir = Path('results_baseline')
    baseline_files = sorted(baseline_dir.glob('*.json'))
    if baseline_files:
        conf = {q.stem: json.loads(q.read_text()) for q in Path('results_confirm').glob('*.json')}
        base = {q.stem: json.loads(q.read_text()) for q in baseline_files}
        missing = set(conf) - set(base)
        if missing:
            raise FileNotFoundError(f'Incomplete results_baseline/: {len(missing)} comparison units are missing')
        cvl = json.loads(Path('results_cvxlin.json').read_text())
        for cond in CONDS:
            ts = [t for t in base if conf[t]['meta']['cond'] == cond]
            if not ts:
                raise ValueError(f'No baseline records for condition {cond}')
            for fq, k in (('0.008', 'lf'), ('1.0', 'hf')):
                ref = lambda t: conf[t]['zref']['z'][fq]['drop_bulk'][0]
                for e, kk in (('nls_lin', 'nlslin'), ('nls_stat', 'nlsstat')):
                    put(f'imp.{cond}.{k}.{kk}', f(float(np.median([100 * (base[t]['arms']['C1'][e]['rez'][fq] - ref(t)) / ref(t) for t in ts])), 1, True))
                put(f'imp.{cond}.{k}.cvxlin', f(float(np.median([100 * (cvl[t]['C1']['rez'][fq] - ref(t)) / ref(t) for t in ts])), 1, True))
        put('imp.lf.nlslin.max', f(max(abs(float(N[f'imp.{c}.lf.nlslin'].replace(r'\ensuremath{-}', '-').replace(r'\ensuremath{+}', ''))) for c in CONDS), 1))
    else:
        warnings.warn('results_baseline/ was not supplied; omitting optional raw-baseline impedance macros', stacklevel=1)
    put('BL.speedratio', f"{BS['descriptive']['nls_lin']['median_sec'] / T['v3_median_sec']:.0f}")
RP_path = Path('results_repro.json')
if RP_path.exists():
    RP = json.loads(RP_path.read_text())
    put('repro.v3fits', str(RP['v3_fits'])); put('repro.v2fits', str(RP['v2_fits'])); put('repro.v2differ', {1: 'one'}.get(RP['v2_n_above_0p01'], str(RP['v2_n_above_0p01'])))
    put('repro.v3max', f"{RP['v3_max_abs_diff_mv']:.0e}".replace('e-', r'\times10^{-').replace('{-0', '{-') + '}'); put('repro.v2max', f(RP['v2_max_abs_diff_mv'], 2))
E1N_path = Path('results_e1/e1_nls.json')
if E1N_path.exists():
    rn = json.loads(E1N_path.read_text()); r3 = json.loads(Path('results_e1/e1_v3.json').read_text())
    assert [x['rep'] for x in rn] == [x['rep'] for x in r3[:len(rn)]]
    en = np.array([x['s4786']['rmse_mv'] for x in rn]); e3 = np.array([x['s4786']['rmse_mv'] for x in r3[:len(rn)]])
    put('E1.nls.n', str(len(rn))); put('E1.nls.info', f(float(np.median(en)), 3)); put('E1.v3.info.same', f(float(np.median(e3)), 3))
    put('E1.nls.diff', f(float(np.median(e3 - en)), 3, True)); put('E1.nls.nlow', str(int(np.sum(en < e3))))
    put('E1.nls.order3', str(sum(x['s4786']['order'] == 3 for x in rn)))

R3_path = Path('results_rob_summary.json')
if R3_path.exists():
    R3 = json.loads(R3_path.read_text()); PM = R3['pack_medians_mv']
    VK = {'dl0.pre0': 'rest', 'dl1.pre0': 'dl', 'dl0.pre1': 'pre', 'dl1.pre1': 'both'}
    put('E3.npacks', str(R3['n_packs'])); put('E3.nunits', str(R3['n_units']))
    for c in CONDS:
        for v, vk in VK.items():
            for arm in ('C1', 'C2'):
                for e in ('v30', 'v31', 'nls_lin', 'v31b'):
                    if e in PM[c][v][arm]: put(f'E3.{c}.{vk}.{arm}.{e}', f(PM[c][v][arm][e], 2))
    rb = R3['robustness']['C1']
    put('E3.maxupper', f(max(r['ci95'][1] for v in rb.values() for r in v.values()), 2))
    put('E3.nrobust', str(sum(r['below_threshold'] for v in rb.values() for r in v.values())) )
    for k, key in (('R1', 'R1_lagged_kinetics_under_double_layer_C1'), ('R2', 'R2_history_on_preloaded_units_C1'),
                   ('R3', 'R3_v31_minus_conventional_C1'), ('R4', 'R4_v31_minus_conventional_C2')):
        fam = R3['primary'][key]
        for c in CONDS:
            put(f'E3.{k}.{c}.med', f(fam[c]['median'], 2, True)); put(f'E3.{k}.{c}.nlow', f"{fam[c]['n_lower']}/{fam[c]['n']}"); put(f'E3.{k}.{c}.holm', p_fmt(fam[c]['holm_p']))
    put('E3.R1.hot.ci', f"[{f(R3['primary']['R1_lagged_kinetics_under_double_layer_C1']['hot']['ci95'][0], 3, True)},{f(R3['primary']['R1_lagged_kinetics_under_double_layer_C1']['hot']['ci95'][1], 3, True)}]")
    q = [fam[c] for fam in R3['equal_accuracy'].values() for c in CONDS]
    put('E3.nequal', str(sum(v['within_margin'] for v in q))); put('E3.nequalall', str(len(q)))
    put('E3.q.maxabs', f(max(abs(x) for v in q for x in (v['ci95'][0], v['ci95'][1])), 3))
    D = R3['descriptive']
    put('E3.nls.sec', f(D['nls_lin']['median_sec'], 1)); put('E3.nls.maxsec', f(D['nls_lin']['max_sec'], 0)); put('E3.v31.sec', f(D['v31']['median_sec'], 2))
    put('E3.nls.multi', f"{D['nls_lin']['fits_with_more_than_one_optimum']}"); put('E3.nls.nfits', str(D['nls_lin']['n_fits']))
    put('E3.nls.multipct', f"{100 * D['nls_lin']['fits_with_more_than_one_optimum'] / D['nls_lin']['n_fits']:.0f}")
    put('E3.cvx.cert', f"{sum(D[e]['n_certified'] for e in ('v30','v31','v31v','v31b','lin'))}/{sum(D[e]['n_fits'] for e in ('v30','v31','v31v','v31b','lin'))}")
    c2 = R3['pack_medians_mv']
    put('E3.c2.rest.min', f(min(c2[c]['dl0.pre0']['C2']['v31'] for c in CONDS), 1)); put('E3.c2.rest.max', f(max(c2[c]['dl0.pre0']['C2']['v31'] for c in CONDS), 1))
    put('E3.c2.pre.min', f(min(c2[c]['dl0.pre1']['C2']['v31'] for c in CONDS), 1)); put('E3.c2.pre.max', f(max(c2[c]['dl0.pre1']['C2']['v31'] for c in CONDS), 1))
    put('E3.c2.v30pre.min', f(min(c2[c]['dl0.pre1']['C2']['v30'] for c in CONDS), 0)); put('E3.c2.v30pre.max', f(max(c2[c]['dl0.pre1']['C2']['v30'] for c in CONDS), 0))
    put('E3.c2.v31b.max', f(max(c2[c][v]['C2']['v31b'] for c in CONDS for v in VK), 0))
    put('E3.c1.v30pre.min', f(min(c2[c]['dl0.pre1']['C1']['v30'] for c in CONDS), 1)); put('E3.c1.v30pre.max', f(max(c2[c]['dl0.pre1']['C1']['v30'] for c in CONDS), 1))
    put('E3.c1.v31pre.min', f(min(c2[c]['dl0.pre1']['C1']['v31'] for c in CONDS), 2)); put('E3.c1.v31pre.max', f(max(c2[c]['dl0.pre1']['C1']['v31'] for c in CONDS), 2))
    put('E3.c1.v31rest.min', f(min(c2[c]['dl0.pre0']['C1']['v31'] for c in CONDS), 2)); put('E3.c1.v31rest.max', f(max(c2[c]['dl0.pre0']['C1']['v31'] for c in CONDS), 2))
    rows = [r"\begin{tabular}{@{}llrrrr@{}}\hline", r"Truth & Estimator & Cold & Nom. & Hot & Aged\\\hline"]
    for v, vk in VK.items():
        if vk in ('rest', 'both'): continue
        lab = {'rest': 'Rested', 'dl': 'Double layer', 'pre': 'Pre-loaded', 'both': 'Both'}[vk]
        for j, (e, el) in enumerate((('v30', 'v3.0'), ('v31', 'v3.1'), ('nls_lin', 'ECM'))):
            rows.append((lab if j == 0 else '') + f" & {el} & " + " & ".join(f(PM[c][v]['C1'][e], 2) for c in CONDS) + r"\\")
    rows += [r"\hline", r"\end{tabular}"]
    (OUT / 'table_rob.tex').write_text("\n".join(rows) + "\n")

if BS_path.exists():
    rows = [r"\begin{tabular}{@{}lrrrrr@{}}\hline", r" & \multicolumn{2}{c}{Linear ECM} & \multicolumn{2}{c}{With static term} & Earlier\\",
            r"Cond. & NLS & Convex & NLS & Convex & pipeline\\\hline"]
    for arm, title in (('C1', 'Known OCV'), ('C2', 'Deployment')):
        rows.append(r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\")
        for cond in CONDS:
            m = BLm[cond][arm]
            rows.append(f"{SH[cond]} & " + " & ".join(f(m[e], 2) for e in ('nls_lin', 'cvx_lin', 'nls_stat', 'v3', 'v2')) + r"\\")
    rows += [r"\hline", r"\end{tabular}"]
else:
    rows = [r"\begin{tabular}{@{}lrrcrr@{}}\hline", r"Cond. & Conv. & Convex & Pack difference & Packs & $p$\\\hline"]
    for arm, fam, title in (('C1', 'F2_estimator_C1', 'Known OCV'), ('C2', 'F1_estimator_C2', 'Deployment')):
        rows.append(r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\")
        for cond in CONDS:
            r = E2['primary'][fam][cond]; m = M[cond][arm]
            rows.append(f"{SH[cond]} & {f(m['v2_long'], 2)} & {f(m['v3_long'], 2)} & {f(r['median'], 2, True)} $[{f(r['ci95'][0], 2, True)},{f(r['ci95'][1], 2, True)}]$ & {r['n_lower']}/{r['n']} & {p_fmt(r['holm_p'])}\\\\")
    rows += [r"\hline", r"\end{tabular}"]
(OUT / 'table_estimator.tex').write_text("\n".join(rows) + "\n")
rows = [r"\begin{tabular}{@{}lcccc@{}}\hline", r"Cond. & Information & Coverage & Short & Fixed\\\hline"]
for arm, title in (('C1', 'Known OCV'), ('C2', 'Deployment')):
    rows.append(r"\multicolumn{5}{@{}l}{\emph{" + title + r"}}\\")
    for cond in CONDS:
        m = M[cond][arm]
        rows.append(f"{SH[cond]} & " + " & ".join(f(m['v3_' + s], 2) for s in ('s4786', 's9588', 'short74', 'passive')) + r"\\")
rows += [r"\hline", r"\end{tabular}"]
(OUT / 'table_excitation.tex').write_text("\n".join(rows) + "\n")

lines = ["% GENERATED by make_numbers.py from results_summary.json and results_diagnosis.json. Do not edit by hand."]
for k, v in N.items():
    lines.append(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}")
(OUT / 'numbers.tex').write_text("\n".join(lines) + "\n")
Path('numbers.json').write_text(json.dumps(N, indent=1))
print(len(N), 'macros written')
