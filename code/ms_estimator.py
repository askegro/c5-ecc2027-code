import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter
def kernels(current, dt, taus):
    current = np.asarray(current, float); K = np.empty((len(current), len(taus)))
    for j, tau in enumerate(taus):
        a = 2.0 * tau / dt; al = 1.0 / (1.0 + a); be = (1.0 - a) / (1.0 + a)
        K[:, j] = lfilter([al, al], [1.0, be], current)
    return K
def integrator(current, dt):
    current = np.asarray(current, float); kc = np.zeros(len(current)); kc[1:] = np.cumsum(0.5 * (current[:-1] + current[1:]) * dt); return kc
def design(current, dt, taus, cdiff, offset):
    cols = [kernels(current, dt, taus)]
    if cdiff: cols.append(integrator(current, dt)[:, None])
    cols.append(np.asarray(current, float)[:, None])
    if offset: cols.append(np.ones((len(current), 1)))
    return np.hstack(cols)
class MSModel:
    def __init__(s, taus, r, inv_c, r0, b, cdiff, offset, lam, train_rmse_mv): s.taus, s.r, s.inv_c, s.r0, s.b, s.cdiff, s.offset, s.lam, s.train_rmse_mv = taus, r, inv_c, r0, b, cdiff, offset, lam, train_rmse_mv
    def predict(s, current, dt):
        return s.b + s.r0 * np.asarray(current, float) + kernels(current, dt, s.taus) @ s.r + (s.inv_c * integrator(current, dt) if s.cdiff else 0.0)
    def impedance(s, f):
        w = 2 * np.pi * np.atleast_1d(np.asarray(f, float))[:, None]; return s.r0 + np.sum(s.r[None, :] / (1 + 1j * w * s.taus[None, :]), axis=1)
def fit(current, y, dt, taus, lam=1.0, cdiff=False, offset=False):
    current = np.asarray(current, float); y = np.asarray(y, float); taus = np.asarray(taus, float); J = len(taus)
    A = design(current, dt, taus, cdiff, offset); n, p = A.shape
    col = np.sqrt(np.mean(A ** 2, axis=0)); col[col == 0] = 1.0
    Q, R = np.linalg.qr(A / col, mode='reduced')
    D2 = np.zeros((J - 2, p)); idx = np.arange(J - 2); D2[idx, idx] = 1.0; D2[idx, idx + 1] = -2.0; D2[idx, idx + 2] = 1.0
    Ra = np.vstack([R, lam * D2 / col]); ya = np.concatenate([Q.T @ y, np.zeros(J - 2)])
    if offset:
        u = Ra[:, -1] / np.linalg.norm(Ra[:, -1]); M = Ra[:, :-1] - np.outer(u, u @ Ra[:, :-1]); yy = ya - u * (u @ ya)
        x, _ = nnls(M, yy, maxiter=200 * p); bs = float(Ra[:, -1] @ (ya - Ra[:, :-1] @ x) / (Ra[:, -1] @ Ra[:, -1])); xs = np.concatenate([x, [bs]])
    else:
        xs, _ = nnls(Ra, ya, maxiter=200 * p)
    x = xs / col; k = J
    inv_c = float(x[k]) if cdiff else 0.0; k += int(cdiff); r0 = float(x[k]); b = float(x[k + 1]) if offset else 0.0
    return MSModel(taus, x[:J].copy(), inv_c, r0, b, cdiff, offset, lam, 1e3 * float(np.sqrt(np.mean((A @ x - y) ** 2))))
