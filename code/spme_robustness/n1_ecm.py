import os, sys, json, time, copy
from pathlib import Path
import numpy as np
import evallib as ev, designs
from pseudoeis import estimation as est
from pseudoeis.estimation import FitSegment, _compress_segment
TD = Path(os.environ.get('TRACES', 'traces_confirm')); ROOT = Path(os.environ.get('NEWOUT', 'new')); FREQS = (0.008, 0.05, 1.0)
MODES = dict(rich=dict(orders=[1, 2, 3, 4, 5, 6, 7, 8], arms=('C1', 'C2')), nodrift=dict(orders=[1, 2, 3, 4], arms=('C1',)))
D = designs.all_designs()
def setup(mode):
    cfg = copy.deepcopy(ev.cfg); idn = cfg['identification']; idn['candidate_rc_orders'] = MODES[mode]['orders']
    if os.environ.get('SMOKE_NFEV'): idn['optimizer_max_nfev'] = int(os.environ['SMOKE_NFEV'])
    if mode == 'nodrift':
        idn['transient_initial_states'] = False; idn['drift_polynomial_order'] = 0
        orig = est._unpack_parameters
        def unpack(vector, order, n_segments, drift_order, transient_initial_states):
            r0, r, tau, init, drift = orig(vector, order, n_segments, drift_order, transient_initial_states)
            return r0, r, tau, [np.zeros_like(s) for s in init], [np.zeros_like(d) for d in drift]
        est._unpack_parameters = unpack
    return cfg
def fit_native(i, drop, design, seed, cfg, one_segment=False):
    idn = cfg['identification']; vf = float(idn['blocked_validation_fraction']); segments, start = [], 0
    if one_segment:
        n_train = int(np.floor((1.0 - vf) * len(i))); mask = np.zeros(len(i), bool); mask[:n_train] = True
        fit, _ = est.select_model_order([FitSegment(np.asarray(i, float), np.asarray(drop, float), design.dt_s, mask, 'record')], idn['candidate_rc_orders'], cfg, np.random.default_rng(seed))
        return fit
    for segment in design.segments:
        stop = start + len(segment.q); x, y = i[start:stop], drop[start:stop]; start = stop
        n_train = min(max(8, int(np.floor((1.0 - vf) * len(x)))), len(x))
        xt, yt, dtc = _compress_segment(x[:n_train], y[:n_train], design.dt_s, design.dt_s)
        xh, yh, _ = _compress_segment(x[n_train:], y[n_train:], design.dt_s, design.dt_s)
        mask = np.zeros(len(xt) + len(xh), bool); mask[:len(xt)] = True
        segments.append(FitSegment(np.concatenate([xt, xh]), np.concatenate([yt, yh]), dtc, mask, segment.name))
    fit, _ = est.select_model_order(segments, idn['candidate_rc_orders'], cfg, np.random.default_rng(seed))
    return fit
def run(mode, tag, cfg):
    out_dir = ROOT / mode; out_dir.mkdir(parents=True, exist_ok=True); f = out_dir / f'{tag}.json'
    out = json.loads(f.read_text()) if f.exists() else {}
    cell = ev.load_cell(TD, tag); meta = cell['meta']; seq = ev.load_seq(TD, tag, 's4786', cell); out.update(tag=tag, cond=meta['cond'], pack=meta['pack'], cell=meta['cell'])
    for arm in MODES[mode]['arms']:
        if arm in out: continue
        t0 = time.time(); fit = fit_native(seq['i'], seq['drop'][arm], D['s4786'], meta['seed'] + 50_000 + meta['cell'], cfg, one_segment=(mode == 'nodrift')); sec = time.time() - t0
        pred = est.predict_polarization_drop(cell['vmeas'], .05, fit); z = fit.impedance(np.array(FREQS))
        out[arm] = dict(rmse_mv=ev.rmse_mv(cell['target'][arm] - pred - cell['v_val']), order=int(fit.order), success=bool(fit.success), refit='slow-branch' in fit.message, sec=sec,
                        r0=float(fit.r0_ohm), r=np.asarray(fit.rc_resistances_ohm).tolist(), tau=np.asarray(fit.time_constants_s).tolist(),
                        z={str(fr): [float(v.real), float(v.imag)] for fr, v in zip(FREQS, z)})
        if arm == 'C2': out[arm]['rmse_persist_mv'] = ev.rmse_mv(cell['target']['C2p'] - pred - cell['v_val'])
        f.write_text(json.dumps(out, indent=1)); print(mode, tag, arm, 'sec', round(sec, 1), 'order', out[arm]['order'], 'ok', out[arm]['success'], flush=True)
def e1(nrep, first=5001):
    from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
    from pseudoeis.signals import synthetic_drive_current, validation_current_profile
    cfg = setup('nodrift'); p = ECMTruthModel(parameters_from_prior_config('NMC', ev.cfg)).effective_parameters(25.0, 0.90, 0.50)
    R0, R, TAU = float(p.r0_ohm), np.asarray(p.rc_resistances_ohm, float), np.asarray(p.time_constants_s, float)
    orig = est._simulate_drop; td = lambda i, dt: orig(np.asarray(i, float), dt, R0, R, TAU, np.zeros(3), np.zeros(1))
    VCUR = validation_current_profile(180.0, 0.05, 20.0, np.random.default_rng(20261001)); VTRUE = td(VCUR, 0.05); q = np.load('seqs/s4786.npy')
    ROOT.mkdir(parents=True, exist_ok=True); OUT = ROOT / 'e1_nodrift.json'; rows = json.loads(OUT.read_text()) if OUT.exists() else []; done = {r['rep'] for r in rows}
    for rep in range(first, first + nrep):
        if rep in done: continue
        ip = synthetic_drive_current(265.23, 0.01, 20.0, np.random.default_rng(400_000 + rep), minimum_current_a=10.0)
        noise = np.random.default_rng(500_000 + rep).normal(0, 1e-3, 26523); i = 0.5 * (1 + q) * ip[:len(q)]; y = td(i, 0.01) + noise[:len(q)]
        t0 = time.time(); fit = fit_native(i, y, D['s4786'], 600_000 + rep, cfg, one_segment=True); sec = time.time() - t0
        rows.append(dict(rep=rep, rmse_mv=ev.rmse_mv(est.predict_polarization_drop(VCUR, 0.05, fit) - VTRUE), order=int(fit.order), success=bool(fit.success), sec=sec))
        OUT.write_text(json.dumps(rows)); print(rep, rows[-1], flush=True)
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'e1nodrift': e1(int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
    else:
        cfg = setup(mode)
        for t in sys.argv[2:]: run(mode, t, cfg)
