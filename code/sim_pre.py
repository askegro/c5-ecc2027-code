import sys, json, time
import numpy as np
import simlib
from pathlib import Path
seed_base, cond, pack_id, ci, T_PRE, outdir = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
out = Path(outdir); out.mkdir(parents=True, exist_ok=True); fn = out / f"{cond}_p{pack_id:02d}_c{ci}.s4786.npz"
if fn.exists(): sys.exit(0)
ORI = 1 if ci == 0 else -1
cfg = simlib.load_cfg(); seed, pack, soc0, temp0, soh0 = simlib.pack_state(seed_base, cond, pack_id, cfg)
meta = dict(cond=cond, pack=pack_id, cell=ci, seed=seed, soc0=float(soc0[ci]), temp0=float(temp0[ci]), soh0=float(soh0[ci]), orientation=ORI, t_pre=T_PRE)
t0 = time.time(); q = np.load('seqs/s4786.npy').astype(float); NMAX = 26523
ip = simlib.pack_current(seed, NMAX, 0.01, cfg)[:len(q)]
nrng = np.random.default_rng(seed + 101 * (ci + 1)); v_noise = nrng.normal(0, 5e-4, NMAX)[:len(q)]; i_noise = nrng.normal(0, .02, NMAX)[:len(q)]
i_rec = 0.5 * (1.0 + ORI * q) * ip
npre = int(round(T_PRE / 0.01)); ipre_pack = np.full(npre, float(cfg['study']['baseline_current_a']))
sgn = 1.0 if (pack_id + ci) % 2 == 0 else -1.0
s = np.concatenate([np.full(npre // 2, sgn), np.full(npre - npre // 2, -sgn)])
i_all = np.concatenate([s * ipre_pack, i_rec])
sim = pack[ci].simulate(i_all, 0.01, soc0[ci], temp0[ci], soh0[ci])
sl = slice(npre, None)
np.savez_compressed(fn, i_true=i_rec, i_meas=i_rec + i_noise, v_true=sim["voltage_v"][sl], v_meas=sim["voltage_v"][sl] + v_noise, ocv=sim["ocv_bulk_v"][sl], soc=sim["soc"][sl], q=q,
                    soc_start=float(sim["soc"][npre]), meta=json.dumps(meta))
print(fn.name, f"{time.time()-t0:.1f}s", 'soc0', soc0[ci], 'soc_start', float(sim['soc'][npre]), flush=True)
