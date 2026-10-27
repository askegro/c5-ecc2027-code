import json, sys, numpy as np
tag = 'cold_p00_c0'; A, B = 'n15out/orig', 'chk_n15'; ok = True
for name, keys in (('s4786', ('i_meas', 'v_meas', 'v_true', 'ocv')), ('wltc', ('vmeas', 'v_val_truth', 'ocv_val'))):
    a = np.load(f'{A}/traces/{tag}.{name}.npz'); b = np.load(f'{B}/traces/{tag}.{name}.npz')
    for k in keys:
        dd = a[k] - b[k]; d = float(np.max(np.abs(dd))); r = float(np.sqrt(np.mean(dd ** 2)))
        good = (d <= 5e-4 and r <= 2e-5) if k in ('v_meas', 'v_true', 'v_val_truth') else d <= 1e-6
        ok &= good; print(f'record {name:5s} {k:11s} max |difference| {d:.3g}, rms {r:.3g}  {"ok" if good else "DIFFERS"}')
def ld(root, d): return json.load(open(f'{root}/{d}/{tag}.json'))
def vals(root):
    F, g = ld(root, 'fit'), ld(root, 'g265')
    return {'conv': F['seq']['s4786']['C1']['v2']['rmse_mv'], 'ecm': ld(root, 'nodrift')['C1']['rmse_mv'], 'cge|2000': F['seq']['s4786']['C1']['v3']['rmse_mv'],
            **{k: g[k]['all'] for k in ('cge|265', 'lin|265', 'drt|265')}}
x, y = vals(A), vals(B)
for k in x:
    good = abs(x[k] - y[k]) <= 1e-2; ok &= good; print(f'{k:9s} {x[k]:.9f} workspace {y[k]:.9f}  difference {x[k] - y[k]:+.2e} mV  {"ok" if good else "DIFFERS"}')
print('N15_CHECK', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)
