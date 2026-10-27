import json, sys, numpy as np
STORED = dict(nominal_p00_c0=dict(lin=1.1710188051314752, ecm=1.1569599226744927, drt=1.162151374497735),
              cold_p03_c1=dict(lin=7.56963898361842, ecm=7.536333668844757, drt=7.564199476718548))
ok = True
for tag, st in STORED.items():
    print('==', tag)
    a = np.load(f'n6out/orig/traces/{tag}.s4786.npz'); b = np.load(f'chk/{tag}.s4786.npz')
    v = np.load(f'n6out/orig/traces/{tag}.val.npz'); w = np.load(f'chk/{tag}.val.npz')
    for k in ('i_meas', 'v_meas', 'v_true', 'ocv'):
        d = float(np.max(np.abs(a[k] - b[k]))); print(f'record s4786 {k:7s} max |difference| {d:.3g}'); ok &= d <= 1e-9
    for k in ('vmeas', 'v_val_truth', 'ocv_val'):
        d = float(np.max(np.abs(v[k] - w[k]))); print(f'record val   {k:11s} max |difference| {d:.3g}'); ok &= d <= 1e-9
    E2 = json.load(open(f'chk/{tag}.json')); F = json.load(open(f'n6out/orig/fit/{tag}.json'))
    N = json.load(open(f'n6out/orig/nodrift/{tag}.json')); M = json.load(open(f'n6out/orig/ms/{tag}.json'))
    z = json.load(open(f'n6out/orig/traces/{tag}.zref.json')); s, o = F['seq']['s4786']['C1'], E2['seq']['s4786']['C1']
    exact = dict(cge=(s['v3']['rmse_mv'], o['v3']['rmse_mv']), lin=(F['linear']['C1']['rmse_mv'], st['lin']), drt=(M['cge|1|C1']['rmse_mv'], st['drt']),
                 zref=(z['z']['0.008']['drop_bulk'][0], E2['zref']['z']['0.008']['drop_bulk'][0]))
    for k, (x, y) in exact.items():
        good = abs(x - y) <= 1e-9 * max(1.0, abs(y)); ok &= good; print(f'{k:5s} {x:.12g} stored {y:.12g}  {"ok" if good else "DIFFERS"}')
    for k, (x, y) in dict(conv=(s['v2']['rmse_mv'], o['v2']['rmse_mv']), ecm=(N['C1']['rmse_mv'], st['ecm'])).items():
        good = abs(x - y) <= 1e-3; ok &= good; print(f'{k:5s} {x:.12g} stored {y:.12g}  difference {x - y:+.2e} mV  {"ok (tolerance 1e-3 mV)" if good else "DIFFERS"}')
print('ORIG_CHECK', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)
