import sys, json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import evallib as ev, cvx_estimator as ce, ms_estimator as ms
from pseudoeis import estimation as est
E2D, NEW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); TD = Path('traces_confirm')
CONDS = ('cold', 'nominal', 'hot', 'aged'); LAB = dict(cold='Cold', nominal='Nominal', hot='Hot', aged='Aged')
COL = dict(conv='#eb6834', ecm='#1baf7a', ms='#eda100', lin='#e87ba4', cge='#2a78d6'); MK = dict(conv='s', ecm='^', ms='D', lin='v', cge='o')
NAME = dict(conv='conventional pipeline', ecm='ECM fit', ms='DRT baseline', lin='linear variant', cge='CGE'); INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdbd6'; SHORT = dict(conv='Conv.', ecm='ECM', ms='DRT', lin='Linear', cge='CGE')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'axes.titlesize': 8, 'axes.titlelocation': 'left'})
def J(p): return json.loads(Path(p).read_text())
E2 = {p.stem: J(p) for p in sorted(E2D.glob('*.json'))}; NOD = {p.stem: J(p) for p in sorted((NEW / 'nodrift').glob('*.json'))}; MSR = {p.stem: J(p) for p in sorted((NEW / 'ms').glob('*.json'))}; IMP = {p.stem: J(p) for p in sorted((NEW / 'imp').glob('*.json'))}
LIN = {(r['tag'], r['arm']): r['linear'] for r in J(sys.argv[4])}
def ecm_drop(i, dt, r0, r, tau): return est._simulate_drop(np.asarray(i, float), dt, float(r0), np.asarray(r, float), np.asarray(tau, float), np.zeros(len(r)), np.zeros(1))
fig = plt.figure(figsize=(7.1, 2.25)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.75], height_ratios=[1, 3.2], hspace=0.12, wspace=0.22, left=0.065, right=0.985, top=0.90, bottom=0.17)
ax = fig.add_subplot(gs[:, 0]); I = np.linspace(0.3, 30, 200)
for c in CONDS:
    tag = f'{c}_p00_c0'; cell = ev.load_cell(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell); m = ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0)
    stored = E2[tag]['seq']['s4786']['C1']['v3']; assert abs(m.chord_resistance(20.0) - stored['chord']['20.0']) < 1e-12, 'CGE refit differs from the stored fit'
    ax.plot(I, [1e3 * m.chord_resistance(a) for a in I], color=INK, lw=1.1)
    ch = E2[tag]['zref']['chord_inst_ohm']; ax.plot([float(k) for k in ch], [1e3 * v for v in ch.values()], ls='none', marker='o', ms=4.5, mfc='white', mec=COL['conv'], mew=1.1)
    ax.text(30.6, 1e3 * m.chord_resistance(30.0), LAB[c], va='center', ha='left', fontsize=7.5, color=INK)
ax.plot([], [], color=INK, lw=1.1, label='fitted static term'); ax.plot([], [], ls='none', marker='o', ms=4.5, mfc='white', mec=COL['conv'], mew=1.1, label='SPMe'); ax.legend(loc='upper right', handlelength=1.6)
ax.set_xlim(0, 37); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel('Current (A)'); ax.set_ylabel('Chord resistance (m$\\Omega$)'); ax.set_title('(a)'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
tag = 'cold_p00_c0'; cell = ev.load_cell(TD, tag); seq = ev.load_seq(TD, tag, 's4786', cell); t = np.arange(len(cell['vmeas'])) * 0.05; tgt = cell['target']['C1'] - cell['v_val']
v2 = E2[tag]['seq']['s4786']['C1']['v2']; nd = NOD[tag]['C1']
pred = dict(conv=ecm_drop(cell['vmeas'], .05, v2['r0'], v2['r'], v2['tau']), ecm=ecm_drop(cell['vmeas'], .05, nd['r0'], nd['r'], nd['tau']),
            ms=ms.fit(seq['i'], seq['drop']['C1'], .01, ce.tau_grid(), 1.0).predict(cell['vmeas'], .05), cge=ce.fit(seq['i'], seq['drop']['C1'], .01, offset=False, capacity_ah=60.0).predict_drop(cell['vmeas'], .05))
stored = dict(conv=v2['rmse_mv'], ecm=nd['rmse_mv'], ms=MSR[tag]['cge|1|C1']['rmse_mv'], cge=E2[tag]['seq']['s4786']['C1']['v3']['rmse_mv']); rm = {}
for k in pred:
    rm[k] = ev.rmse_mv(tgt - pred[k]); assert abs(rm[k] - stored[k]) < 1e-6, (k, rm[k], stored[k])
print('Fig. 2b unit', tag, {k: round(v, 3) for k, v in rm.items()})
a1 = fig.add_subplot(gs[0, 1]); a1.plot(t, cell['vmeas'], color=MUTED, lw=0.7); a1.axhline(0, color=GRID, lw=0.5); a1.set_ylabel('Current\n(A)', fontsize=7); a1.set_xlim(0, 180); a1.tick_params(labelbottom=False); a1.set_yticks([-20, 0, 20]); a1.set_title('(b)')
a2 = fig.add_subplot(gs[1, 1], sharex=a1); a2.axhline(0, color=MUTED, lw=0.5)
for k in ('conv', 'ecm', 'ms', 'cge'): a2.plot(t, 1e3 * (pred[k] - tgt) * -1, color=COL[k], lw=1.0 if k != 'cge' else 1.3, label=f'{NAME[k]} ({rm[k]:.2f} mV)', zorder=3 if k == 'cge' else 2)
a2.set_xlabel('Time on the validation profile (s)'); a2.set_ylabel('Error (mV)'); a2.grid(axis='y', color=GRID, lw=0.5); a2.set_axisbelow(True); h, l = a2.get_legend_handles_labels(); a2.legend([h[i] for i in (0, 2, 1, 3)], [l[i] for i in (0, 2, 1, 3)], loc='upper right', ncol=2, handlelength=1.6, columnspacing=1.2)
lo, hi = a2.get_ylim(); a2.set_ylim(lo, hi + 0.62 * (hi - lo)); a2.set_yticks([-10, 0, 10])
fig.savefig(OUT / 'fig_static.pdf'); plt.close(fig)
unit = dict(conv=lambda g: E2[g]['seq']['s4786']['C1']['v2']['rmse_mv'], ecm=lambda g: NOD[g]['C1']['rmse_mv'], ms=lambda g: MSR[g]['cge|1|C1']['rmse_mv'], lin=lambda g: LIN[(g, 'C1')], cge=lambda g: E2[g]['seq']['s4786']['C1']['v3']['rmse_mv'])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.25), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.24, left=0.065, right=0.985, top=0.90, bottom=0.17))
order = ('conv', 'ecm', 'ms', 'lin', 'cge'); rng = np.random.default_rng(0)
for ci, c in enumerate(CONDS):
    packs = sorted({E2[g]['meta']['pack'] for g in E2 if E2[g]['meta']['cond'] == c})
    for ei, e in enumerate(order):
        y = [np.mean([unit[e](f'{c}_p{p:02d}_c{u}') for u in (0, 1)]) for p in packs]; x = ci + (ei - 2) * 0.16 + rng.uniform(-0.045, 0.045, len(y))
        ax.plot(x, y, ls='none', marker=MK[e], ms=3.6, mfc=COL[e], mec='white', mew=0.4, label=SHORT[e] if ci == 0 else None)
    if ci: ax.axvline(ci - 0.5, color=GRID, lw=0.5)
ax.set_yscale('log'); ax.set_xticks(range(4)); ax.set_xticklabels([LAB[c] for c in CONDS]); ax.tick_params(axis='x', length=0, labelsize=8, labelcolor=INK); ax.set_xlim(-0.55, 3.5); ax.set_ylabel('RMSE of each pair (mV)'); ax.set_title('(a)')
ax.grid(axis='y', which='major', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0.15, 30); ax.set_yticks([0.2, 0.5, 1, 2, 5, 10]); ax.set_yticklabels(['0.2', '0.5', '1', '2', '5', '10']); ax.minorticks_off()
ax.legend(loc='upper center', ncol=5, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.2, markerscale=1.3)
ZL = {}
for g in sorted(IMP):
    cell_g = ev.load_cell(TD, g); seq_g = ev.load_seq(TD, g, 's4786', cell_g); ml = ce.fit(seq_g['i'], seq_g['drop']['C1'], .01, offset=False, scales=np.array([]))
    assert abs(ev.rmse_mv(cell_g['target']['C1'] - ml.predict_drop(cell_g['vmeas'], .05) - cell_g['v_val']) - LIN[(g, 'C1')]) < 1e-6, ('linear variant refit differs', g)
    ZL[g] = {fr: complex(v) for fr, v in zip(('0.008', '0.05', '1.0'), ml.impedance((0.008, 0.05, 1.0)))}
EST = ('conv', 'ecm', 'ms', 'lin', 'cge'); F = '1.0'; xs, ys = {k: [] for k in EST}, {k: [] for k in EST}
def zc(g, f): v = E2[g]['seq']['s4786']['C1']['v2']; w = 2 * np.pi * float(f); return v['r0'] + sum(r / (1 + 1j * w * ta) for r, ta in zip(v['r'], v['tau']))
ZF = dict(conv=zc, ecm=lambda g, f: complex(*NOD[g]['C1']['z'][f]), ms=lambda g, f: complex(*MSR[g]['cge|1|C1']['z'][f]), lin=lambda g, f: ZL[g][f], cge=lambda g, f: complex(*IMP[g]['C1']['z'][f]))
ZG = {k: (lambda g, k=k: ZF[k](g, F)) for k in EST}
for g in sorted(IMP):
    ch = E2[g]['zref']['chord_inst_ohm']; x = 100 * (1 - ch['20.0'] / ch['0.5']); zr = complex(*E2[g]['zref']['z'][F]['drop_bulk'])
    for k in xs: xs[k].append(x); ys[k].append(100 * (ZG[k](g).real - zr.real) / abs(zr))
m = max(max(xs['cge']), 1); bx.plot([0, 28], [0, -28], color=MUTED, lw=0.7, ls=(0, (4, 3))); bx.text(12.6, -10.6, 'error equal\nto the drop', fontsize=7, color=MUTED, ha='left', va='bottom'); bx.axhline(0, color=MUTED, lw=0.5)
for k in EST: bx.plot(xs[k], ys[k], ls='none', marker=MK[k], ms=3.4, mfc=COL[k], mec='white', mew=0.35, label=NAME[k], zorder=3 if k == 'cge' else 2)
bx.set_xlabel('Drop of the chord resistance at 20 A (%)'); bx.set_ylabel('Error of Re $Z$ (%)'); bx.set_title('(b)'); bx.grid(color=GRID, lw=0.5); bx.set_axisbelow(True); bx.set_xlim(0, 28); bx.set_ylim(-31, 4)
fig.savefig(OUT / 'fig_results.pdf'); plt.close(fig)
lin3 = ('conv', 'ecm', 'ms', 'lin'); cmin = min(abs(np.corrcoef(xs[k], ys[k])[0, 1]) for k in lin3); off = [abs(np.median(np.array(ys[k]) + np.array(xs[k]))) for k in lin3]
open(OUT / 'numbers_fig.tex', 'w').write('\\expandafter\\def\\csname num@fg.corr.min\\endcsname{%.2f}\n\\expandafter\\def\\csname num@fg.off.range\\endcsname{%.1f--%.1f}\n\\expandafter\\def\\csname num@fg.cold.cge\\endcsname{%.2f}\n\\expandafter\\def\\csname num@fg.cold.lin\\endcsname{%.2f--%.2f}\n' % (np.floor(cmin * 100) / 100, min(off), max(off), rm['cge'], min(rm[k] for k in ('conv', 'ecm', 'ms')), max(rm[k] for k in ('conv', 'ecm', 'ms'))))
for k in xs: print('Fig. 3b', k, 'corr with -x %.3f' % np.corrcoef(xs[k], ys[k])[0, 1], 'median y+x %.2f' % np.median(np.array(ys[k]) + np.array(xs[k])), 'max|y+x| %.2f' % np.max(np.abs(np.array(ys[k]) + np.array(xs[k]))))
print('x range', round(min(xs['cge']), 2), round(max(xs['cge']), 2))

from stats_helpers import sign_p, holm
def relerr(k, g, f, part='abs'):
    zr = complex(*E2[g]['zref']['z'][f]['drop_bulk']); e = 100 * (ZF[k](g, f) - zr) / abs(zr); return abs(e) if part == 'abs' else (e.real if part == 're' else e.imag)
FREQS3 = ('0.008', '0.05', '1.0'); tags = sorted(IMP); cond = {g: IMP[g]['cond'] for g in tags}; pack = {g: IMP[g]['pack'] for g in tags}; med = {}
for k in EST:
    for f in FREQS3:
        for c in CONDS: med[(k, f, c)] = float(np.median([relerr(k, g, f) for g in tags if cond[g] == c]))
nl, hp = [], []
for k in lin3:
    for f in FREQS3:
        ps = []
        for c in CONDS:
            d = {}
            for g in tags:
                if cond[g] == c: d.setdefault(pack[g], []).append(relerr('cge', g, f) - relerr(k, g, f))
            dd = np.array([np.mean(v) for v in d.values() if len(v) == 2]); nl.append(int((dd < 0).sum())); ps.append(sign_p(dd))
        hp += holm(ps)
imx = max(abs(float(np.median([relerr(k, g, f, 'im') for g in tags if cond[g] == c]))) for k in EST for f in FREQS3 for c in CONDS)
cg = [med[('cge', f, c)] for f in FREQS3 for c in CONDS]; ot = [med[(k, f, c)] for k in lin3 for f in FREQS3 for c in CONDS]
L = ['\\expandafter\\def\\csname num@fg.z.cge.range\\endcsname{%.1f--%.1f}' % (min(cg), max(cg)), '\\expandafter\\def\\csname num@fg.z.cge.max\\endcsname{%.1f}' % max(cg),
     '\\expandafter\\def\\csname num@fg.z.oth.range\\endcsname{%.1f--%.1f}' % (min(ot), max(ot)), '\\expandafter\\def\\csname num@fg.z.oth.max\\endcsname{%.1f}' % max(ot),
     '\\expandafter\\def\\csname num@fg.z.nlowmin\\endcsname{%d}' % min(nl), '\\expandafter\\def\\csname num@fg.z.holm.max\\endcsname{%.3f}' % max(hp), '\\expandafter\\def\\csname num@fg.z.im.max\\endcsname{%.1f}' % imx]
for c in CONDS:
    v = [med[(k, f, c)] for k in lin3 for f in FREQS3]; L.append('\\expandafter\\def\\csname num@fg.z.oth.%s.range\\endcsname{%.0f--%.0f}' % (c, min(v), max(v)))
open(OUT / 'numbers_fig.tex', 'a').write('\n'.join(L) + '\n'); print('impedance: cge', min(cg), max(cg), 'others', min(ot), max(ot), 'nlow min', min(nl), 'holm max', max(hp), 'im max', imx)
print({c: (min(med[(k, f, c)] for k in lin3 for f in FREQS3), max(med[(k, f, c)] for k in lin3 for f in FREQS3)) for c in CONDS})
