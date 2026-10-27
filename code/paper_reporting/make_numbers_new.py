import sys, json, numpy as np
from pathlib import Path
C = json.load(open(sys.argv[1])); NEW = Path(sys.argv[2]); E2D = Path(sys.argv[3]); LIN = {(r['tag'], r['arm']): r['linear'] for r in json.load(open(sys.argv[4]))}; E1R = json.load(open(sys.argv[5]))
CONDS = ('cold', 'nominal', 'hot', 'aged'); AB = dict(cold='C', nominal='N', hot='H', aged='A'); FREQS = ('0.008', '0.05', '1.0'); M = {}
def put(k, v): M[k] = v
def sgn(x, nd=2): return (r'\ensuremath{-}' if x < 0 else r'\ensuremath{+}') + f'{abs(x):.{nd}f}'
def load(d): return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob('*.json'))}
E2 = load(E2D); NOD = load(NEW / 'nodrift'); MS = load(NEW / 'ms')
unit = dict(conv=lambda t: E2[t]['seq']['s4786']['C1']['v2']['rmse_mv'], cge=lambda t: E2[t]['seq']['s4786']['C1']['v3']['rmse_mv'], ecm=lambda t: NOD[t]['C1']['rmse_mv'], ms=lambda t: MS[t]['cge|1|C1']['rmse_mv'], lin=lambda t: LIN[(t, 'C1')])
med = {e: {c: float(np.median([g(t) for t in NOD if NOD[t]['cond'] == c])) for c in CONDS} for e, g in unit.items()}
for e in med:
    for c in CONDS: put(f'nw.{e}.{c}', f'{med[e][c]:.2f}')
    put(f'nw.{e}.range', f'{min(med[e].values()):.2f}--{max(med[e].values()):.2f}')
ct = C['nodrift_C1']['contrasts']; ce = ct['nodrift_vs_cge']; cr = ct['nodrift_vs_raw']
for c in CONDS:
    put(f'nw.cgeecm.diff.{c}', sgn(-ce[c]['median'])); put(f'nw.cgeecm.lo.{c}', sgn(-ce[c]['ci95'][1])); put(f'nw.cgeecm.hi.{c}', sgn(-ce[c]['ci95'][0])); put(f'nw.cgeecm.nlow.{c}', str(ce[c]['n_higher']))
    put(f'nw.drift.{c}', f"{-cr[c]['median']:.2f}"); put(f'nw.drift.nlow.{c}', str(cr[c]['n_lower'])); put(f'nw.drift.holm.{c}', f"{cr[c]['p_holm']:.3f}")
put('nw.cgeecm.nlowmin', str(min(ce[c]['n_higher'] for c in CONDS))); put('nw.cgeecm.holm.max', f"{max(ce[c]['p_holm'] for c in CONDS):.3f}")
put('nw.drift.range', f"{min(-cr[c]['median'] for c in CONDS):.2f}--{max(-cr[c]['median'] for c in CONDS):.2f}"); put('nw.drift.nlowmin', str(min(cr[c]['n_lower'] for c in CONDS))); put('nw.drift.holm.max', f"{max(cr[c]['p_holm'] for c in CONDS):.3f}")
cm = C['ms']['contrasts']['ms_vs_cge_C1']; put('nw.cgems.nlowmin', str(min(cm[c]['n_higher'] for c in CONDS))); put('nw.cgems.holm.max', f"{max(cm[c]['p_holm'] for c in CONDS):.3f}")
agree = max(max(abs(ct['nodrift_vs_linear'][c]['median']), abs(C['ms']['contrasts']['ms_vs_linear_C1'][c]['median'])) for c in CONDS); put('nw.agree', f'{agree:.2f}')
gain = {c: med['lin'][c] - med['cge'][c] for c in CONDS}
rr = C['rich_C1']['contrasts']['rich_vs_raw']; put('nw.rich.maxabs', f"{max(abs(rr[c]['median']) for c in CONDS):.2f}"); oh = C['rich_C1']['order_hist']; put('nw.rich.order12', str(oh['1'] + oh['2'])); put('nw.rich.n', str(C['rich_C1']['n_units']))
rc = C['rich_C1']['contrasts']['rich_vs_cge']; put('nw.cgerich.nlowmin', str(min(rc[c]['n_higher'] for c in CONDS)))
Z = {}; allcge, alllin = [], []
for f in FREQS:
    um = C['impedance'][f'C1_{f}Hz']['unit_medians']
    for c in CONDS:
        for e, k in (('cge', 'cge_abs'), ('conv', 'conv_abs'), ('ecm', 'nodrift_abs'), ('ms', 'ms_abs')):
            Z[(e, f, c)] = um[c][k]; put(f'nw.z.{e}.{f}.{c}', f'{um[c][k]:.1f}'); (allcge if e == 'cge' else alllin).append(um[c][k])
put('nw.z.cge.range', f'{min(allcge):.1f}--{max(allcge):.1f}'); put('nw.z.lin.range', f'{min(alllin):.1f}--{max(alllin):.1f}')
for c in CONDS:
    v = [Z[(e, f, c)] for e in ('conv', 'ecm', 'ms') for f in FREQS]; put(f'nw.z.lin.{c}.range', f'{min(v):.0f}--{max(v):.0f}')
nl, hm = [], []
for f in FREQS:
    for k in ('cge_vs_conv', 'cge_vs_ms', 'cge_vs_nodrift'):
        for c in CONDS: r = C['impedance'][f'C1_{f}Hz']['contrasts'][k][c]; nl.append(r['n_lower']); hm.append(r['p_holm'])
put('nw.z.nlowmin', str(min(nl))); put('nw.z.holm.max', f'{max(hm):.3f}')
ims = [abs(C['impedance'][f'C1_{f}Hz']['unit_medians'][c][f'{e}_im']) for f in FREQS for c in CONDS for e in ('cge', 'conv', 'nodrift', 'ms')]; put('nw.z.im.max', f'{max(ims):.1f}')
e1 = C['e1']; put('nw.e1.n', str(e1['n'])); put('nw.e1.cge', f"{e1['cge_median']:.3f}"); put('nw.e1.ecm', f"{e1['nodrift']['median']:.3f}"); put('nw.e1.ecm.nlow', str(e1['nodrift']['n'] - e1['nodrift']['n_cge_lower']))
put('nw.e1.ecm.diff', f"{abs(e1['nodrift']['diff_median']):.3f}"); put('nw.e1.ms', f"{e1['ms|cge|1']['median']:.2f}"); put('nw.e1.raw', f"{e1['raw']['median']:.2f}"); put('nw.e1.raw.cgelow', str(e1['raw']['n_cge_lower']))
x = [r['rmse_mv'] for r in E1R]; put('nw.e1.raw.min', f'{min(x):.2f}'); put('nw.e1.raw.max', f'{max(x):.2f}')
v = [med[e][c] for e in ('ecm', 'ms', 'lin') for c in CONDS]; put('nw.linall.range', f'{min(v):.2f}--{max(v):.2f}')
S = C['ms']['sensitivity_unit_medians']; sp = max(max(x for k, x in S[c].items() if k.endswith('|C1')) - min(x for k, x in S[c].items() if k.endswith('|C1')) for c in CONDS); put('nw.ms.sens', f'{np.ceil(sp * 100) / 100:.2f}')
open('numbers_new.tex', 'w').write(''.join(f'\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n' for k, v in M.items()))
rows = [f"{c.capitalize()} & {med['conv'][c]:.2f} & {med['ecm'][c]:.2f} & {med['ms'][c]:.2f} & {med['lin'][c]:.2f} & {med['cge'][c]:.2f} & {M[f'nw.cgeecm.diff.{c}']} $[{M[f'nw.cgeecm.lo.{c}']},{M[f'nw.cgeecm.hi.{c}']}]$\\\\" for c in CONDS]
open('table_pred.tex', 'w').write('\\begin{tabular}{@{}lrrrrrc@{}}\\hline\nCondition & Conv. & ECM & DRT & Linear & \\cge{} & \\cge{} $-$ ECM [95\\% Interval]\\\\\\hline\n' + '\n'.join(rows) + '\n\\hline\n\\end{tabular}\n')
rows = []
for f in FREQS:
    for i, c in enumerate(CONDS):
        rows.append((f"{float(f):g}" if i == 0 else '') + f" & {AB[c]} & {Z[('conv', f, c)]:.1f} & {Z[('ecm', f, c)]:.1f} & {Z[('ms', f, c)]:.1f} & {Z[('cge', f, c)]:.1f}\\\\" + ('\\hline' if i == 3 else ''))
open('table_imp.tex', 'w').write('\\begin{tabular}{@{}llrrrr@{}}\\hline\nHz & Cond. & Conv. & ECM & DRT & \\cge{}\\\\\\hline\n' + '\n'.join(rows) + '\n\\end{tabular}\n')
print(len(M), 'macros'); print({k: v for k, v in M.items() if not k.startswith('nw.z.') or 'range' in k or 'min' in k or 'max' in k})
print('static-term gain lin - cge', {c: round(g, 2) for c, g in gain.items()})
