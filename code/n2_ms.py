import os, sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, cvx_estimator as ce, ms_estimator as ms
TD = Path(os.environ.get('TRACES', 'traces_confirm')); ROOT = Path(os.environ.get('NEWOUT', 'new')); FREQS = (0.008, 0.05, 1.0)
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)
GRIDS = dict(cge=ce.tau_grid(), ten=np.geomspace(0.005, 2000.0, int(round(np.log10(2000.0 / 0.005) * 10)) + 1))
def one(i, y, cell, arm, taus, lam, v_id=None):
    if arm == 'V':
        m = ms.fit(i, -v_id, .01, taus, lam, cdiff=True, offset=True); vhat = -m.predict(cell['vmeas'], .05)
        return m, dict(rmse_mv=ev.rmse_mv(vhat - cell['v_val']), u0=-m.b, inv_c=m.inv_c)
    m = ms.fit(i, y, .01, taus, lam, cdiff=False, offset=(arm == 'C2')); pred = m.predict(cell['vmeas'], .05)
    o = dict(rmse_mv=ev.rmse_mv(cell['target'][arm] - pred - cell['v_val']))
    if arm == 'C2': o.update(rmse_persist_mv=ev.rmse_mv(cell['target']['C2p'] - pred - cell['v_val']), b=m.b)
    return m, o
def run(tag):
    out_dir = ROOT / 'ms'; out_dir.mkdir(parents=True, exist_ok=True); f = out_dir / f'{tag}.json'
    if f.exists(): return
    cell = ev.load_cell(TD, tag); meta = cell['meta']; seq = ev.load_seq(TD, tag, 's4786', cell); v_id = np.load(TD / f'{tag}.s4786.npz')['v_meas']
    out = dict(tag=tag, cond=meta['cond'], pack=meta['pack'], cell=meta['cell']); t0 = time.time()
    for g, taus in GRIDS.items():
        for lam in LAMS:
            for arm in ('C1', 'C2', 'V'):
                m, o = one(seq['i'], seq['drop'].get(arm), cell, arm, taus, lam, v_id)
                o.update(train_rmse_mv=m.train_rmse_mv, n_active=int(np.sum(m.r > 0)), r0=m.r0, z={str(fr): [float(v.real), float(v.imag)] for fr, v in zip(FREQS, m.impedance(FREQS))})
                out[f'{g}|{lam:g}|{arm}'] = o
    out['sec'] = time.time() - t0; f.write_text(json.dumps(out, indent=1)); print(tag, {a: round(out[f'cge|1|{a}']['rmse_mv'], 3) for a in ('C1', 'C2', 'V')}, round(out['sec'], 1), flush=True)
def e1(nrep, first=5001):
    from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
    from pseudoeis.signals import synthetic_drive_current, validation_current_profile
    from pseudoeis import estimation as est
    p = ECMTruthModel(parameters_from_prior_config('NMC', ev.cfg)).effective_parameters(25.0, 0.90, 0.50)
    R0, R, TAU = float(p.r0_ohm), np.asarray(p.rc_resistances_ohm, float), np.asarray(p.time_constants_s, float)
    td = lambda i, dt: est._simulate_drop(np.asarray(i, float), dt, R0, R, TAU, np.zeros(3), np.zeros(1))
    VCUR = validation_current_profile(180.0, 0.05, 20.0, np.random.default_rng(20261001)); VTRUE = td(VCUR, 0.05); q = np.load('seqs/s4786.npy')
    ROOT.mkdir(parents=True, exist_ok=True); OUT = ROOT / 'ms_e1.json'; rows = json.loads(OUT.read_text()) if OUT.exists() else []; done = {r['rep'] for r in rows}
    for rep in range(first, first + nrep):
        if rep in done: continue
        ip = synthetic_drive_current(265.23, 0.01, 20.0, np.random.default_rng(400_000 + rep), minimum_current_a=10.0)
        noise = np.random.default_rng(500_000 + rep).normal(0, 1e-3, 26523); i = 0.5 * (1 + q) * ip[:len(q)]; y = td(i, 0.01) + noise[:len(q)]
        row = dict(rep=rep, cge=ev.rmse_mv(ce.fit(i, y, .01, offset=False).predict_drop(VCUR, .05) - VTRUE))
        for g, taus in GRIDS.items():
            for lam in LAMS: row[f'{g}|{lam:g}'] = ev.rmse_mv(ms.fit(i, y, .01, taus, lam).predict(VCUR, .05) - VTRUE)
        rows.append(row); OUT.write_text(json.dumps(rows)); print(rep, {k: round(v, 4) for k, v in row.items() if k != 'rep'}, flush=True)
if __name__ == '__main__':
    if sys.argv[1] == 'e1': e1(int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
    else:
        for t in sys.argv[2:]: run(t)
