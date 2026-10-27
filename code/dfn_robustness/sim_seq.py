import sys, json, time
import numpy as np
import simlib

seed_base, cond, pack_id, ci = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
seqfile, name, outdir = sys.argv[5], sys.argv[6], sys.argv[7]
orient = int(sys.argv[8]) if len(sys.argv) > 8 else 1
from pathlib import Path
out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
fn = out / f"{cond}_p{pack_id:02d}_c{ci}.{name}.npz"
if fn.exists(): sys.exit(0)
cfg = simlib.load_cfg()
seed, pack, soc0, temp0, soh0 = simlib.pack_state(seed_base, cond, pack_id, cfg)
meta = dict(spread=simlib.SPREAD['name'], cond=cond, pack=pack_id, cell=ci, seed=seed, soc0=float(soc0[ci]), temp0=float(temp0[ci]), soh0=float(soh0[ci]), orientation=orient)
t0 = time.time()
if seqfile == 'val':
    from pseudoeis.signals import validation_current_profile
    vrng = np.random.default_rng(seed + 7_000_001 + 10_007 * ci)
    vcur = validation_current_profile(180.0, .05, 20.0, vrng); vmeas = vcur + vrng.normal(0.0, .02, len(vcur))
    vt = pack[ci].simulate(vcur, .05, soc0[ci], temp0[ci], soh0[ci])
    np.savez_compressed(fn, vcur=vcur, vmeas=vmeas, v_val_truth=vt["voltage_v"], ocv_val=vt["ocv_bulk_v"], soc_val=vt["soc"], meta=json.dumps(meta))
else:
    q = np.load(seqfile).astype(float)
    NMAX = 26523
    ip = simlib.pack_current(seed, NMAX, 0.01, cfg)[:len(q)]
    nrng = np.random.default_rng(seed + 101 * (ci + 1))
    v_noise = nrng.normal(0, 5e-4, NMAX)[:len(q)]; i_noise = nrng.normal(0, .02, NMAX)[:len(q)]
    i_cell = 0.5 * (1.0 + orient * q) * ip
    sim = pack[ci].simulate(i_cell, 0.01, soc0[ci], temp0[ci], soh0[ci])
    np.savez_compressed(fn, i_true=i_cell, i_meas=i_cell + i_noise, v_true=sim["voltage_v"], v_meas=sim["voltage_v"] + v_noise,
                        ocv=sim["ocv_bulk_v"], soc=sim["soc"], q=q, meta=json.dumps(meta))
print(fn.name, f"{time.time()-t0:.1f}s", flush=True)
