from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.signal import lfilter

__version__ = "3.0.0"

TAU_MIN_S = 0.005
TAU_MAX_S = 2000.0
GRID_PER_DECADE = 8
BV_SCALES_C_RATE = (1 / 12, 1 / 6, 1 / 3, 2 / 3, 4 / 3)
KKT_TOL = 1e-9
EPS = 1e-8


def tau_grid(tau_min: float = TAU_MIN_S, tau_max: float = TAU_MAX_S, per_decade: int = GRID_PER_DECADE) -> np.ndarray:
    n = int(round(np.log10(tau_max / tau_min) * per_decade)) + 1
    return np.geomspace(tau_min, tau_max, n)


def bv_scales(capacity_ah: float = 60.0) -> np.ndarray:
    return np.asarray(BV_SCALES_C_RATE) * float(capacity_ah)


def rc_states(current: np.ndarray, dt: float, taus: np.ndarray) -> np.ndarray:
    current = np.asarray(current, float)
    out = np.empty((len(current), len(taus)))
    for j, tau in enumerate(taus):
        a = float(np.exp(-dt / tau))
        out[:, j] = lfilter([0.0, 1.0 - a], [1.0, -a], current)
    return out


def bv_basis(current: np.ndarray, scales: np.ndarray) -> np.ndarray:
    current = np.asarray(current, float)
    if len(scales) == 0:
        return np.empty((len(current), 0))
    return np.column_stack([s * np.arcsinh(current / s) for s in scales])


def design_matrix(current: np.ndarray, dt: float, taus: np.ndarray, scales: np.ndarray, offset: bool) -> np.ndarray:
    current = np.asarray(current, float)
    first = [np.ones(len(current))] if offset else []
    return np.hstack([np.column_stack(first + [current]), bv_basis(current, scales), rc_states(current, dt, taus)])


@dataclass
class LocalModel:
    taus: np.ndarray
    scales: np.ndarray
    b: float
    r0: float
    rho: np.ndarray
    r: np.ndarray
    offset: bool
    n_samples: int
    train_rmse_mv: float
    kkt_violation: float
    version: str = __version__

    @property
    def certified(self) -> bool:
        return bool(self.kkt_violation < KKT_TOL)

    def static_drop(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, float)
        return self.r0 * current + bv_basis(current, self.scales) @ self.rho

    def predict_drop(self, current: np.ndarray, dt: float) -> np.ndarray:
        current = np.asarray(current, float)
        return self.b + self.static_drop(current) + rc_states(current, dt, self.taus) @ self.r

    @property
    def r_inst_small_signal(self) -> float:
        return float(self.r0 + self.rho.sum())

    def chord_resistance(self, current_a: float) -> float:
        return float(self.static_drop(np.array([current_a]))[0] / current_a)

    def impedance(self, f_hz) -> np.ndarray:
        w = 2 * np.pi * np.atleast_1d(np.asarray(f_hz, float))[:, None]
        return self.r_inst_small_signal + np.sum(self.r[None, :] / (1 + 1j * w * self.taus[None, :]), axis=1)

    @property
    def n_active_branches(self) -> int:
        return int(np.sum(self.r > 0))

    def to_dict(self) -> dict:
        return dict(version=self.version, offset=self.offset, b=self.b, r0=self.r0, rho=self.rho.tolist(), scales=self.scales.tolist(),
                    r=self.r.tolist(), taus=self.taus.tolist(), n_samples=self.n_samples, train_rmse_mv=self.train_rmse_mv,
                    kkt_violation=self.kkt_violation, certified=self.certified, n_active_branches=self.n_active_branches)


def fit(current: np.ndarray, drop: np.ndarray, dt: float, *, offset: bool, capacity_ah: float = 60.0,
        taus: np.ndarray | None = None, scales: np.ndarray | None = None) -> LocalModel:
    current = np.asarray(current, float)
    drop = np.asarray(drop, float)
    if current.ndim != 1 or current.shape != drop.shape or len(current) < 100:
        raise ValueError("current and drop must be one-dimensional records of equal length (at least 100 samples).")
    if not (np.all(np.isfinite(current)) and np.all(np.isfinite(drop))):
        raise ValueError("current and drop must be finite.")
    taus = tau_grid() if taus is None else np.asarray(taus, float)
    scales = bv_scales(capacity_ah) if scales is None else np.asarray(scales, float)
    X = design_matrix(current, dt, taus, scales, offset)
    n, p = X.shape
    k0 = 1 if offset else 0
    col = np.sqrt(np.mean(X ** 2, axis=0))
    col[col == 0] = 1.0
    Q, R = np.linalg.qr(X / col, mode="reduced")
    Ra = np.vstack([R, np.sqrt(EPS * n) * np.eye(p)])
    ya = np.concatenate([Q.T @ drop, np.zeros(p)])
    if offset:
        u = Ra[:, 0] / np.linalg.norm(Ra[:, 0])
        A = Ra[:, 1:] - np.outer(u, u @ Ra[:, 1:])
        yy = ya - u * (u @ ya)
    else:
        A, yy = Ra, ya
    x, _ = nnls(A, yy, maxiter=200 * p)
    if offset:
        b_s = float(Ra[:, 0] @ (ya - Ra[:, 1:] @ x) / (Ra[:, 0] @ Ra[:, 0]))
        ts = np.concatenate([[b_s], x])
    else:
        ts = x
    g = Ra.T @ (Ra @ ts - ya)
    gscale = float(np.linalg.norm(Ra.T @ ya, np.inf)) or 1.0
    at_bound = np.ones(p, bool)
    at_bound[k0:] = ts[k0:] <= 0
    if offset:
        at_bound[0] = False
    viol = np.where(at_bound, np.maximum(-g, 0.0), np.abs(g))
    theta = ts / col
    res = drop - X @ theta
    return LocalModel(taus=taus, scales=scales, b=float(theta[0]) if offset else 0.0, r0=float(theta[k0]),
                      rho=np.asarray(theta[k0 + 1:k0 + 1 + len(scales)], float), r=np.asarray(theta[k0 + 1 + len(scales):], float),
                      offset=offset, n_samples=n, train_rmse_mv=1e3 * float(np.sqrt(np.mean(res ** 2))),
                      kkt_violation=float(np.max(viol) / gscale))
