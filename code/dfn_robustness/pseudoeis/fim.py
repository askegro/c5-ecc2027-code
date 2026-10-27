from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import ECMParameters
from .signals import ExcitationSegment, spectrum_energy

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FIMResult:
    matrix: FloatArray
    logdet: float
    min_eigenvalue: float
    condition_number: float
    score: float


def impedance_jacobian(
    frequencies_hz: FloatArray,
    nominal: ECMParameters,
) -> ComplexArray:
    f = np.asarray(frequencies_hz, dtype=float)
    r0 = nominal.r0_ohm
    r = nominal.rc_resistances_ohm
    tau = nominal.time_constants_s
    jw_tau = 1j * 2.0 * np.pi * f[:, None] * tau[None, :]
    denom = 1.0 + jw_tau
    d_log_r0 = np.full((len(f), 1), r0, dtype=complex)
    d_log_r = r[None, :] / denom
    d_log_tau = -r[None, :] * jw_tau / (denom**2)
    return np.concatenate([d_log_r0, d_log_r, d_log_tau], axis=1)


def fisher_information(
    q: FloatArray,
    dt_s: float,
    target_frequencies_hz: FloatArray,
    nominal: ECMParameters,
    voltage_noise_std_v: float,
    current_amplitude_a: float,
    low_frequency_weight_power: float = 0.35,
    regularization: float = 1e-12,
    objective: str = "D",
) -> FIMResult:
    if voltage_noise_std_v <= 0 or current_amplitude_a <= 0:
        raise ValueError("Noise standard deviation and current amplitude must be positive.")
    f = np.asarray(target_frequencies_hz, dtype=float)
    nyquist = 0.5 / dt_s
    valid = (f > 0) & (f < 0.95 * nyquist)
    f = f[valid]
    if len(f) == 0:
        n_par = 1 + 2 * nominal.n_rc
        return FIMResult(np.zeros((n_par, n_par)), -np.inf, 0.0, np.inf, -np.inf)
    energy = spectrum_energy(q, dt_s, f) * current_amplitude_a**2
    jac = impedance_jacobian(f, nominal)
    weights = (np.maximum(f, 1e-12) / np.min(f)) ** (-low_frequency_weight_power)
    fim = np.zeros((jac.shape[1], jac.shape[1]), dtype=float)
    for k in range(len(f)):
        j = jac[k]
        scale = weights[k] * energy[k] / (voltage_noise_std_v**2 + 1e-30)
        fim += scale * np.real(np.outer(np.conjugate(j), j))
    fim += regularization * np.eye(fim.shape[0])
    sign, logdet = np.linalg.slogdet(fim)
    eigenvalues = np.linalg.eigvalsh(fim)
    min_eig = float(np.min(eigenvalues))
    cond = float(np.max(eigenvalues) / max(min_eig, regularization))
    if sign <= 0:
        logdet = -np.inf
    obj = objective.upper()
    if obj == "D":
        score = float(logdet)
    elif obj == "E":
        score = float(np.log(max(min_eig, regularization)))
    elif obj == "A":
        score = -float(np.trace(np.linalg.pinv(fim)))
    else:
        raise ValueError(f"Unsupported FIM objective: {objective}")
    score -= 0.02 * np.log1p(cond)
    return FIMResult(fim, float(logdet), min_eig, cond, score)


def select_information_optimal_segment(
    candidates: list[ExcitationSegment],
    target_frequencies_hz: FloatArray,
    nominal: ECMParameters,
    voltage_noise_std_v: float,
    current_amplitude_a: float,
    low_frequency_weight_power: float,
    regularization: float,
    objective: str,
) -> tuple[ExcitationSegment, FIMResult]:
    if not candidates:
        raise ValueError("Candidate list is empty.")
    evaluated = [
        (
            candidate,
            fisher_information(
                candidate.q,
                candidate.dt_s,
                target_frequencies_hz,
                nominal,
                voltage_noise_std_v,
                current_amplitude_a,
                low_frequency_weight_power,
                regularization,
                objective,
            ),
        )
        for candidate in candidates
    ]
    return max(evaluated, key=lambda item: item[1].score)
