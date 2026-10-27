import os, sys, importlib, json
import numpy as np
os.environ['N6_SPREAD'] = 'orig'
import simlib, spread_arms
import campaign_v4_backend_ocv as old, campaign_v4_backend_n6 as new
cfg = simlib.load_cfg()
KEYS = ('Number of electrodes connected in parallel to make a cell', 'Contact resistance [Ohm]')
def state(arm, cond, pack):
    os.environ['N6_SPREAD'] = arm; importlib.reload(simlib)
    seed, pk, soc0, temp0, soh0 = simlib.pack_state(9202610010, cond, pack, cfg)
    return seed, pk, soc0, temp0, soh0, simlib.pack_current(seed, 2000, 0.01, cfg)
n = 0
for cond in simlib.CONDS:
    for pack in range(12):
        seed = 9202610010 + simlib.COND_OFFSET[cond] + pack * 100003
        pk_old = old.build_pack('NMC', 4, cfg, np.random.default_rng(seed))
        s = state('orig', cond, pack)
        for a, b in zip(pk_old, s[1]):
            for k in KEYS: assert a.parameter_values[k] == b.parameter_values[k], (cond, pack, k)
            assert a.capacity_ah_base == b.capacity_ah_base
        for arm in ('lit', 'zero'):
            t = state(arm, cond, pack)
            assert np.array_equal(s[2], t[2]) and np.array_equal(s[3], t[3]) and np.array_equal(s[5], t[5]), (arm, cond, pack)
            if arm != 'orig': assert np.all(t[4] == simlib.CONDS[cond]['soh'])
        n += 1
print(f'1, 2: {n} packs: arm orig identical to the original backend; SOC, temperature and pack current identical in every arm')
os.environ['N6_SPREAD'] = 'lit'
cap, rc = [], []
for k in range(5000):
    for c in new.build_pack('NMC', 4, cfg, np.random.default_rng(k)):
        cap.append(c.capacity_ah_base); rc.append(c.parameter_values['Contact resistance [Ohm]'])
cap, rc = np.array(cap), np.array(rc)
print(f'3: lit, {len(cap)} units: capacity SD {np.std(cap) / np.mean(cap) * 100:.3f} % (target 0.28); contact resistance mean {rc.mean() * 1e3:.4f} mOhm, '
      f'SD {rc.std() * 1e3:.4f} mOhm (target {spread_arms.RC_SD * 1e3:.4f}), min {rc.min() * 1e3:.4f}; options {new._sim_options()}')
R = spread_arms.R_REF; tot = np.sqrt((np.std(1 / (cap / np.mean(cap))) * R) ** 2 + np.var(rc)) / (R + rc.mean())
print(f'   implied impedance spread at 0.008 Hz (nominal): {tot * 100:.3f} % of {(R + rc.mean()) * 1e3:.3f} mOhm (literature 0.72 % of the cell value)')
if '--sim' in sys.argv:
    import subprocess, tempfile
    d = tempfile.mkdtemp(); env = dict(os.environ, N6_SPREAD='orig')
    subprocess.run([sys.executable, 'sim_seq.py', '9202610010', 'nominal', '0', '0', 'val', 'val', d], env=env, check=True, capture_output=True)
    a = np.load(f'{d}/nominal_p00_c0.val.npz'); b = np.load('chk/nominal_p00_c0.val.npz')
    for k in ('vcur', 'vmeas', 'v_val_truth', 'ocv_val'): assert np.array_equal(a[k], b[k]), k
    print('--sim: arm orig reproduces the stored validation trace of nominal_p00_c0 bit for bit')
print('ALL N6 CHECKS PASSED')
