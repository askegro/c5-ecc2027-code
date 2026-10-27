import os, sys, json
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(k, "1")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pseudoeis.models import ECMTruthModel, parameters_from_prior_config
from pseudoeis.signals import validation_current_profile
import cvx_estimator as ce

cfg = json.loads((HERE / "config.json").read_text())
bms = ECMTruthModel(parameters_from_prior_config("NMC", cfg))
SOC_BIAS_SD = 0.002


def coulomb(z0, i, dt, cap_as):
    z = np.empty(len(i)); z[0] = z0
    z[1:] = z0 - np.cumsum(i[:-1]) * dt / cap_as
    return np.clip(z, 0, 1)


def rmse_mv(a):
    return 1e3 * float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def load_cell(trace_dir, tag):
    V = np.load(Path(trace_dir) / f"{tag}.val.npz"); meta = json.loads(str(V['meta'])); seed, ci = meta['seed'], meta['cell']
    cap = bms.capacity_ah(meta['soh0']) * 3600.0
    vrng = np.random.default_rng(seed + 7_000_001 + 10_007 * ci)
    vcur = validation_current_profile(180.0, .05, 20.0, vrng); vnoise = vrng.normal(0.0, .02, len(vcur))
    assert np.allclose(vcur + vnoise, V['vmeas'])
    z0_val = float(np.clip(meta['soc0'] + vrng.normal(0.0, SOC_BIAS_SD), 0, 1))
    z0_id = float(np.clip(meta['soc0'] + np.random.default_rng(seed + 303 * (ci + 1)).normal(0.0, SOC_BIAS_SD), 0, 1))
    return dict(meta=meta, cap=cap, vmeas=V['vmeas'], v_val=V['v_val_truth'], z0_id=z0_id, z0_val=z0_val,
                target=dict(C1=V['ocv_val'], C2=bms.ocv(coulomb(z0_val, V['vmeas'], .05, cap)),
                            C2p=bms.ocv(coulomb(z0_id, V['vmeas'], .05, cap))))


def load_seq(trace_dir, tag, name, cell):
    Z = np.load(Path(trace_dir) / f"{tag}.{name}.npz")
    i = Z['i_meas']; v = Z['v_meas']
    return dict(i=i, drop=dict(C1=Z['ocv'] - v, C2=bms.ocv(coulomb(cell['z0_id'], i, .01, cell['cap'])) - v), q=Z['q'])


def eval_new(cell, seq, arm):
    m = ce.fit(seq['i'], seq['drop'][arm], .01, offset=(arm == 'C2'), capacity_ah=60.0)
    pred = m.predict_drop(cell['vmeas'], .05)
    out = dict(rmse_mv=rmse_mv(cell['target'][arm] - pred - cell['v_val']), train_rmse_mv=m.train_rmse_mv, kkt=m.kkt_violation,
               n_active=m.n_active_branches, r_inst_ss=m.r_inst_small_signal, b=m.b,
               rez={str(f): float(m.impedance([f])[0].real) for f in (0.008, 0.05, 1.0)},
               chord={str(a): m.chord_resistance(a) for a in (0.5, 10.0, 20.0)})
    if arm == 'C2':
        out['rmse_persist_mv'] = rmse_mv(cell['target']['C2p'] - pred - cell['v_val'])
        out['rmse_nooffset_pred_mv'] = rmse_mv(cell['target'][arm] - (pred - m.b) - cell['v_val'])
    return out


def eval_frozen(cell, seq, arm, design, rng_seed):
    from pseudoeis import estimation as est
    from pseudoeis.estimation import FitSegment, _compress_segment
    ident = cfg['identification']; vf = float(ident['blocked_validation_fraction'])
    i, drop = seq['i'], seq['drop'][arm]
    segments, start = [], 0
    for segment in design.segments:
        stop = start + len(segment.q); x, y = i[start:stop], drop[start:stop]; start = stop
        target_dt = min(segment.bit_period_s / 4.0, max(design.dt_s, 0.25))
        if segment.bit_period_s <= 0.05:
            target_dt = design.dt_s
        n_train = min(max(8, int(np.floor((1.0 - vf) * len(x)))), len(x))
        xt, yt, dtc = _compress_segment(x[:n_train], y[:n_train], design.dt_s, target_dt)
        xh, yh, _ = _compress_segment(x[n_train:], y[n_train:], design.dt_s, target_dt)
        mask = np.zeros(len(xt) + len(xh), bool); mask[:len(xt)] = True
        segments.append(FitSegment(np.concatenate([xt, xh]), np.concatenate([yt, yh]), dtc, mask, segment.name))
    import time
    t0 = time.time()
    fit, _ = est.select_model_order(segments, ident['candidate_rc_orders'], cfg, np.random.default_rng(rng_seed))
    sec = time.time() - t0
    pred = est.predict_polarization_drop(cell['vmeas'], .05, fit)
    out = dict(rmse_mv=rmse_mv(cell['target'][arm] - pred - cell['v_val']), order=int(fit.order), success=bool(fit.success),
               r0=float(fit.r0_ohm), r=np.asarray(fit.rc_resistances_ohm).tolist(), tau=np.asarray(fit.time_constants_s).tolist(),
               refit='slow-branch' in fit.message, sec=sec,
               rez={str(f): float(np.real(fit.impedance(np.array([f]))[0])) for f in (0.008, 0.05, 1.0)})
    if arm == 'C2':
        out['rmse_persist_mv'] = rmse_mv(cell['target']['C2p'] - pred - cell['v_val'])
    return out
