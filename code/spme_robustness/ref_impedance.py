import sys, json, time
import numpy as np
import simlib
from pathlib import Path

FREQS = (0.008, 0.05, 1.0)
AMP = 0.5

def td_impedance(cell, f, soc, temp, soh, amp=AMP, periods=4):
    dt = min(0.25, 1.0 / (f * 40)); dt = max(0.0025, round(dt / 0.0025) * 0.0025)
    n_per = int(round(1.0 / (f * dt))); n = n_per * periods
    t = np.arange(n) * dt
    w = 2 * np.pi * f
    cur = amp * np.sin(w * (t + 0.5 * dt))
    s = cell.simulate(cur, dt, soc, temp, soh)
    sl = slice(n_per, n)
    A = np.column_stack([np.sin(w * (t[sl] + 0.5 * dt)), np.cos(w * (t[sl] + 0.5 * dt)), np.ones(n - n_per)])
    out = {}
    for name, y in (('drop_bulk', s['ocv_bulk_v'] - s['voltage_v']), ('terminal', s['voltage_v'][0] - s['voltage_v'])):
        c = np.linalg.lstsq(A, y[sl], rcond=None)[0]
        out[name] = [float(c[0] / amp), float(c[1] / amp)]
    return out

if __name__ == '__main__':
    seed_base, cond, pack_id, ci, outdir = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    fn = Path(outdir) / f"{cond}_p{pack_id:02d}_c{ci}.zref.json"
    if fn.exists(): sys.exit(0)
    cfg = simlib.load_cfg()
    seed, pack, soc0, temp0, soh0 = simlib.pack_state(seed_base, cond, pack_id, cfg)
    t0 = time.time()
    res = dict(spread=simlib.SPREAD['name'], freqs=list(FREQS), amp_a=AMP, z={})
    for f in FREQS:
        res['z'][str(f)] = td_impedance(pack[ci], f, soc0[ci], temp0[ci], soh0[ci])
    chord = {}
    for amp in (0.5, 10.0, 20.0):
        cur = np.concatenate([np.zeros(3), np.full(5, amp)])
        s = pack[ci].simulate(cur, 0.01, soc0[ci], temp0[ci], soh0[ci])
        chord[str(amp)] = float((s['ocv_bulk_v'][3] - s['voltage_v'][3]) / amp)
    res['chord_inst_ohm'] = chord
    eis = pack[ci].impedance(np.asarray(FREQS), temp0[ci], soh0[ci], soc0[ci])
    res['pybamm_eis'] = [[float(z.real), float(z.imag)] for z in eis]
    fn.write_text(json.dumps(res, indent=1))
    print(fn.name, f"{time.time()-t0:.1f}s", flush=True)
