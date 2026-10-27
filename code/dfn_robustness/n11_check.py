import json, sys, numpy as np
tag = 'cold_p00_c0'; A, B = 'n11out/orig', 'chk_dfn'; ok = True
def ld(root, d): return json.load(open(f'{root}/{d}/{tag}.json'))
for name, keys in (('s4786', ('i_meas', 'v_meas', 'v_true', 'ocv')), ('val', ('vmeas', 'v_val_truth', 'ocv_val'))):
    a = np.load(f'{A}/traces/{tag}.{name}.npz'); b = np.load(f'{B}/traces/{tag}.{name}.npz')
    for k in keys:
        dd = a[k] - b[k]; d = float(np.max(np.abs(dd))); r = float(np.sqrt(np.mean(dd ** 2)))
        if k in ('v_meas', 'v_true', 'v_val_truth'): good = d <= 5e-4 and r <= 2e-5
        else: good = d <= 1e-6
        ok &= good; print(f'record {name:5s} {k:11s} max |difference| {d:.3g} V, rms {r:.3g} V  {"ok" if good else "DIFFERS"}')
F, G = ld(A, 'fit'), ld(B, 'fit'); s, o = F['seq']['s4786']['C1'], G['seq']['s4786']['C1']
za = json.load(open(f'{A}/traces/{tag}.zref.json')); zb = json.load(open(f'{B}/traces/{tag}.zref.json'))
rel = {f'zref {f} {p}': (za['z'][f]['drop_bulk'][i], zb['z'][f]['drop_bulk'][i]) for f in ('0.008', '0.05', '1.0') for i, p in enumerate(('re', 'im'))}
for k, (x, y) in rel.items():
    good = abs(x - y) <= 1e-6 * max(abs(y), 1e-12); ok &= good; print(f'{k:16s} {x:.12g} workspace {y:.12g}  {"ok" if good else "DIFFERS"}')
fits = dict(cge=(s['v3']['rmse_mv'], o['v3']['rmse_mv']), lin=(F['linear']['C1']['rmse_mv'], G['linear']['C1']['rmse_mv']),
            drt=(ld(A, 'ms')['cge|1|C1']['rmse_mv'], ld(B, 'ms')['cge|1|C1']['rmse_mv']), conv=(s['v2']['rmse_mv'], o['v2']['rmse_mv']),
            ecm=(ld(A, 'nodrift')['C1']['rmse_mv'], ld(B, 'nodrift')['C1']['rmse_mv']))
for k, (x, y) in fits.items():
    good = abs(x - y) <= 1e-2; ok &= good; print(f'{k:16s} {x:.12g} workspace {y:.12g}  difference {x - y:+.2e} mV  {"ok" if good else "DIFFERS"}')
print('DFN_CHECK', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)
