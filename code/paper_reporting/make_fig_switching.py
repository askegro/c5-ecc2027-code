import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})
BLUE, GREY, RED = '#1f5f8b', '#444444', '#c0502d'
fig = plt.figure(figsize=(3.125, 3.05))
ax = fig.add_axes([0.0, 0.62, 1.0, 0.38]); ax.set_xlim(0, 10); ax.set_ylim(-0.4, 5); ax.axis('off')
ax.text(0.0, 4.95, '(a) Four-unit pack with two complementary pairs', va='top', ha='left', fontsize=8.2, transform=ax.transData)
ax.plot([0.9, 9.6], [4.0, 4.0], color='#999999', lw=0.6)
ax.text(0.0, 4.0, r'$i_p\rightarrow$', ha='left', va='center', fontsize=8)
xs = [1.9, 4.1, 6.1, 8.3]; cols = [BLUE, BLUE, GREY, GREY]
for k, (x, c) in enumerate(zip(xs, cols)):
    ax.plot([x - 0.35, x + 0.35], [3.55, 3.55], color='black', lw=1.0)
    ax.plot([x - 0.35, x + 0.35], [3.55, 3.55], 'o', color='black', ms=2)
    ax.add_patch(FancyBboxPatch((x - 0.8, 2.15), 1.6, 0.85, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=c, lw=1.0))
    ax.text(x, 2.57, f'unit {k}', color=c, ha='center', va='center', fontsize=8)
    ax.plot([x - 0.3, x + 0.3], [1.8, 1.45], color='black', lw=1.0); ax.plot([x - 0.3, x + 0.3], [1.8, 1.45], 'o', color='black', ms=2)
ax.text(0.0, 3.2, 'insert', fontsize=6.5, color='#555555', va='center')
ax.text(0.0, 1.2, 'bypass', fontsize=6.5, color='#555555', va='center')
ax.add_patch(FancyBboxPatch((1.0, 0.1), 4.1, 0.85, boxstyle='round,pad=0.02,rounding_size=0.08', fc='none', ec=BLUE, lw=1.0, ls='--'))
ax.text(3.05, 0.52, r'pair 1: $s_0+s_1=1$', color=BLUE, ha='center', va='center', fontsize=7.5)
ax.add_patch(FancyBboxPatch((5.2, 0.1), 4.1, 0.85, boxstyle='round,pad=0.02,rounding_size=0.08', fc='none', ec=GREY, lw=1.0, ls='--'))
ax.text(7.25, 0.52, r'pair 2: $s_2+s_3=1$', color=GREY, ha='center', va='center', fontsize=7.5)

bx = fig.add_axes([0.14, 0.15, 0.80, 0.42]); n = 25
q = np.array([1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,-1])
s0 = (1 + q) / 2; s1 = (1 - q) / 2
def step(y, off, c, lw=1.1, h=0.7, **kw): bx.step(np.arange(n), off + h * y, where='post', color=c, lw=lw, **kw)
step((q + 1) / 2, 3.2, 'black'); step(s0, 2.1, BLUE); step(s0, 1.0, RED, lw=1.3); step(s1, -0.1, RED, lw=1.3)
bx.text(-1.0, 3.55, r'$q[k]$', ha='right', va='center'); bx.text(-1.0, 2.45, r'$s_0[k]$', ha='right', va='center')
bx.text(-1.0, 1.35, r'$i_0[k]$', ha='right', va='center'); bx.text(-1.0, 0.25, r'$i_1[k]$', ha='right', va='center')
bx.set_xlim(-0.5, n + 0.5); bx.set_ylim(-0.4, 4.2); bx.set_yticks([])
for sp in ('top', 'right', 'left'): bx.spines[sp].set_visible(False)
bx.set_xticks([0, 6, 12, 18, 24]); bx.set_xlabel(r'Sample $k$')
fig.text(0.0, 0.595, '(b) One pair: sequence, insertion states, unit currents', ha='left', va='center', fontsize=8.2)
fig.savefig('fig_switching.pdf'); fig.savefig('fig_switching_check.png', dpi=160)
