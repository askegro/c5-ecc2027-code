import sys, json, glob, os
import numpy as np
E2, MM, SRC = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
out = {}
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(E2, '*.json')))]
v3 = [r['seq']['s4786'][a]['v3']['sec'] for r in R for a in ('C1', 'C2')]; v2 = [r['seq']['s4786'][a]['v2']['sec'] for r in R for a in ('C1', 'C2')]
out['rev.v3sec'] = f'{np.median(v3):.1f}'; out['rev.v2sec'] = f'{np.median(v2):.0f}'; out['rev.nfits'] = str(len(v3)); assert len(v2) == len(v3)
C = json.load(open(MM))['asym']['contrasts']['cge_vs_v2_C1']['cold']; out['mm.asym.v2.cold.holm'] = f"{C['p_holm']:.3f}"
sys.path.insert(0, SRC); cwd = os.getcwd(); os.chdir(SRC)
import simlib
cfg = simlib.load_cfg(); mx, fl = [], []
for c, off in simlib.COND_OFFSET.items():
    for p in range(12):
        ip = simlib.pack_current(9202610010 + off + p * 100003, 26523, 0.01, cfg); mx.append(ip.max()); fl.append(np.mean(ip <= 10.0 + 1e-12))
os.chdir(cwd)
out['env.max'] = f'{max(mx):.1f}'; out['env.floor'] = f'{100 * max(fl):.1f}'
with open('numbers_rev.tex', 'w') as f:
    for k, v in out.items(): f.write('\\expandafter\\def\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(out)
