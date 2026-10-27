import sys, json, time
from pathlib import Path
import numpy as np
import evallib as ev, designs
from evallib import cfg
from pseudoeis import estimation as est
from pseudoeis.estimation import FitSegment, _compress_segment
TD = Path('traces_confirm'); OUT = Path('sup/raw'); OUT.mkdir(parents=True, exist_ok=True)
D = designs.all_designs()
def fit_raw(i, drop, design, seed):
    ident = cfg['identification']; vf = float(ident['blocked_validation_fraction']); segments, start = [], 0
    for segment in design.segments:
        stop = start + len(segment.q); x, y = i[start:stop], drop[start:stop]; start = stop
        n_train = min(max(8, int(np.floor((1.0 - vf) * len(x)))), len(x))
        xt, yt, dtc = _compress_segment(x[:n_train], y[:n_train], design.dt_s, design.dt_s)
        xh, yh, _ = _compress_segment(x[n_train:], y[n_train:], design.dt_s, design.dt_s)
        mask = np.zeros(len(xt) + len(xh), bool); mask[:len(xt)] = True
        segments.append(FitSegment(np.concatenate([xt, xh]), np.concatenate([yt, yh]), dtc, mask, segment.name))
    fit, _ = est.select_model_order(segments, ident['candidate_rc_orders'], cfg, np.random.default_rng(seed))
    return fit
def run(tag, arms=('C1', 'C2')):
    f = OUT / f'{tag}.json'
    prev = json.loads(f.read_text()) if f.exists() else {}
    cell = ev.load_cell(TD, tag); meta = cell['meta']; seq = ev.load_seq(TD, tag, 's4786', cell); out = dict(tag=tag, cond=meta['cond'], pack=meta['pack'], cell=meta['cell']); out.update(prev)
    for arm in arms:
        if arm in out: continue
        t0 = time.time(); fit = fit_raw(seq['i'], seq['drop'][arm], D['s4786'], meta['seed'] + 50_000 + meta['cell']); sec = time.time() - t0
        pred = est.predict_polarization_drop(cell['vmeas'], .05, fit)
        out[arm] = dict(rmse_mv=ev.rmse_mv(cell['target'][arm] - pred - cell['v_val']), order=int(fit.order), success=bool(fit.success), refit='slow-branch' in fit.message, sec=sec)
        f.write_text(json.dumps(out, indent=1)); print(tag, arm, 'sec', round(sec, 1), 'order', out[arm]['order'], 'ok', out[arm]['success'], flush=True)
if __name__ == '__main__':
    for t in sys.argv[1:]: run(t)
