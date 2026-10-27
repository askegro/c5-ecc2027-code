import os, sys, json, time
variant, cond, pack_id, ci, name, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
os.environ['TRUTH_VARIANT'] = variant; os.environ['TRUTH_BACKEND'] = 'campaign_v4_backend_mm'
import numpy as np
import simlib
from pathlib import Path
SEED_BASE = 9202610010
out = Path(outdir); out.mkdir(parents=True, exist_ok=True); fn = out / f"{cond}_p{pack_id:02d}_c{ci}.{name}.npz"
if fn.exists(): sys.exit(0)
cfg = simlib.load_cfg(); seed, pack, soc0, temp0, soh0 = simlib.pack_state(SEED_BASE, cond, pack_id, cfg)
ORI = 1 if ci == 0 else -1
meta = dict(cond=cond, pack=pack_id, cell=ci, seed=seed, soc0=float(soc0[ci]), temp0=float(temp0[ci]), soh0=float(soh0[ci]), orientation=ORI if name == 's4786' else 1, variant=variant)
t0 = time.time()
if name == 'val':
    from pseudoeis.signals import validation_current_profile
    vrng = np.random.default_rng(seed + 7_000_001 + 10_007 * ci)
    vcur = validation_current_profile(180.0, .05, 20.0, vrng); vmeas = vcur + vrng.normal(0.0, .02, len(vcur))
    vt = pack[ci].simulate(vcur, .05, soc0[ci], temp0[ci], soh0[ci])
    np.savez_compressed(fn, vcur=vcur, vmeas=vmeas, v_val_truth=vt["voltage_v"], ocv_val=vt["ocv_bulk_v"], soc_val=vt["soc"], meta=json.dumps(meta))
else:
    NMAX = 26523
    q4 = np.load('seqs/s4786.npy').astype(float)
    q = q4 if name == 's4786' else np.load('seqs/passive.npy').astype(float)
    ip = simlib.pack_current(seed, NMAX, 0.01, cfg)[:len(q)]
    nrng = np.random.default_rng(seed + 101 * (ci + 1)); v_noise = nrng.normal(0, 5e-4, NMAX)[:len(q)]; i_noise = nrng.normal(0, .02, NMAX)[:len(q)]
    if name == 's4786':
        i_cell = 0.5 * (1.0 + ORI * q) * ip
    elif name == 'passive':
        i_cell = 0.5 * (1.0 + q) * ip
    elif name == 'passive_eq':
        i_sw = 0.5 * (1.0 + ORI * q4) * simlib.pack_current(seed, NMAX, 0.01, cfg)[:len(q4)]
        scale = float(np.sqrt(np.mean(i_sw ** 2) / np.mean(ip ** 2))); meta['rms_scale'] = scale
        i_cell = scale * 0.5 * (1.0 + q) * ip
    else:
        raise ValueError(name)
    sim = pack[ci].simulate(i_cell, 0.01, soc0[ci], temp0[ci], soh0[ci])
    np.savez_compressed(fn, i_true=i_cell, i_meas=i_cell + i_noise, v_true=sim["voltage_v"], v_meas=sim["voltage_v"] + v_noise,
                        ocv=sim["ocv_bulk_v"], soc=sim["soc"], q=q, meta=json.dumps(meta))
print(variant, fn.name, f"{time.time()-t0:.1f}s", flush=True)
