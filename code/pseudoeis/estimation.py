from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import coherence, csd, lfilter, lfiltic, resample_poly, welch

from .models import ECMTruthModel, ecm_impedance
from .signals import ExcitationDesign
from .simulation import CellMeasurement

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SpectralEstimate:
    frequency_hz: FloatArray
    impedance_ohm: ComplexArray
    coherence: FloatArray
    input_snr_db: FloatArray
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class FitSegment:
    current_a: FloatArray
    voltage_drop_v: FloatArray
    dt_s: float
    train_mask: NDArray[np.bool_]
    name: str


@dataclass(frozen=True)
class ECMFitResult:
    order: int
    r0_ohm: float
    rc_resistances_ohm: FloatArray
    time_constants_s: FloatArray
    initial_states_v: tuple[FloatArray, ...]
    drift_coefficients_v: tuple[FloatArray, ...]
    train_rmse_mv: float
    validation_rmse_mv: float
    bic_per_sample: float
    selection_score: float
    success: bool
    message: str
    nfev: int
    solve_time_s: float
    residuals_v: tuple[FloatArray, ...]

    def impedance(self, frequencies_hz: Sequence[float]) -> ComplexArray:
        return ecm_impedance(
            frequencies_hz,
            self.r0_ohm,
            self.rc_resistances_ohm,
            self.time_constants_s,
        )


@dataclass(frozen=True)
class BootstrapUncertainty:
    frequency_hz: FloatArray
    impedance_mean_ohm: ComplexArray
    impedance_std_ohm: FloatArray
    lower_real_ohm: FloatArray
    upper_real_ohm: FloatArray
    lower_imag_ohm: FloatArray
    upper_imag_ohm: FloatArray
    successful_replicates: int


def estimate_ocv_corrected_drop(measurement: CellMeasurement, nominal_model: ECMTruthModel) -> FloatArray:
    ocv_est = nominal_model.ocv(measurement.estimated_soc)
    return np.asarray(ocv_est - measurement.measured_voltage_v, dtype=float)


def spectral_impedance(
    measurement: CellMeasurement,
    design: ExcitationDesign,
    nominal_model: ECMTruthModel,
    coherence_threshold: float,
    minimum_excitation_snr_db: float,
) -> SpectralEstimate:
    drop = estimate_ocv_corrected_drop(measurement, nominal_model)
    all_f: list[FloatArray] = []
    all_z: list[ComplexArray] = []
    all_c: list[FloatArray] = []
    all_snr: list[FloatArray] = []

    for segment, slc in zip(design.segments, measurement.segment_slices, strict=True):
        x = measurement.measured_current_a[slc]
        y = drop[slc]
        n = len(x)
        if n < 16:
            continue
        nperseg = max(16, min(n, n // 2))
        noverlap = nperseg // 2
        f, sii = welch(x, fs=1.0 / design.dt_s, nperseg=nperseg, noverlap=noverlap, detrend="linear")
        _, siy = csd(x, y, fs=1.0 / design.dt_s, nperseg=nperseg, noverlap=noverlap, detrend="linear")
        _, coh = coherence(x, y, fs=1.0 / design.dt_s, nperseg=nperseg, noverlap=noverlap, detrend="linear")
        z = siy / np.maximum(sii, 1e-30)
        positive = sii[1:]
        noise_floor = np.percentile(positive, 25) if len(positive) else 1e-30
        snr = 10.0 * np.log10(np.maximum(sii, 1e-30) / max(noise_floor, 1e-30))
        keep = f > 0
        all_f.append(f[keep])
        all_z.append(z[keep])
        all_c.append(coh[keep])
        all_snr.append(snr[keep])

    if not all_f:
        return SpectralEstimate(np.array([]), np.array([], dtype=complex), np.array([]), np.array([]), np.array([], dtype=bool))
    f = np.concatenate(all_f)
    z = np.concatenate(all_z)
    coh = np.concatenate(all_c)
    snr = np.concatenate(all_snr)
    order = np.argsort(f)
    f, z, coh, snr = f[order], z[order], coh[order], snr[order]
    valid = (coh >= coherence_threshold) & (snr >= minimum_excitation_snr_db)
    return SpectralEstimate(f, z, coh, snr, valid)


def target_frequency_coverage(spectral: SpectralEstimate, target_frequencies_hz: FloatArray) -> float:
    if len(spectral.frequency_hz) == 0 or len(target_frequencies_hz) == 0:
        return 0.0
    hits = [
        bool(spectral.valid[int(np.argmin(np.abs(spectral.frequency_hz - f)))])
        for f in target_frequencies_hz
    ]
    return float(np.mean(hits))


def _compress_segment(x: FloatArray, y: FloatArray, original_dt_s: float, target_dt_s: float) -> tuple[FloatArray, FloatArray, float]:
    if target_dt_s <= original_dt_s * (1.0 + 1e-12):
        return x.copy(), y.copy(), original_dt_s
    factor = int(round(target_dt_s / original_dt_s))
    if factor <= 1:
        return x.copy(), y.copy(), original_dt_s
    return (
        np.asarray(resample_poly(x, up=1, down=factor), dtype=float),
        np.asarray(resample_poly(y, up=1, down=factor), dtype=float),
        original_dt_s * factor,
    )


def prepare_fit_segments(
    measurement: CellMeasurement,
    design: ExcitationDesign,
    nominal_model: ECMTruthModel,
    validation_fraction: float,
) -> list[FitSegment]:
    drop = estimate_ocv_corrected_drop(measurement, nominal_model)
    segments: list[FitSegment] = []
    for segment, slc in zip(design.segments, measurement.segment_slices, strict=True):
        x = measurement.measured_current_a[slc]
        y = drop[slc]
        target_dt = min(segment.bit_period_s / 4.0, max(design.dt_s, 0.25))
        if segment.bit_period_s <= 0.05:
            target_dt = design.dt_s
        n_train_raw = max(8, int(np.floor((1.0 - validation_fraction) * len(x))))
        n_train_raw = min(n_train_raw, len(x))
        x_train_c, y_train_c, dt_c = _compress_segment(
            x[:n_train_raw], y[:n_train_raw], design.dt_s, target_dt
        )
        x_hold_c, y_hold_c, dt_c_hold = _compress_segment(
            x[n_train_raw:], y[n_train_raw:], design.dt_s, target_dt
        )
        assert dt_c == dt_c_hold, "train/holdout compression must agree on sample period"
        x_c = np.concatenate([x_train_c, x_hold_c])
        y_c = np.concatenate([y_train_c, y_hold_c])
        mask = np.zeros(len(x_c), dtype=bool)
        mask[: len(x_train_c)] = True
        segments.append(FitSegment(x_c, y_c, dt_c, mask, segment.name))
    return segments


def _soft_bounds_for_tau(order: int, min_dt: float, max_duration: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    low = max(2.0 * min_dt, 1e-3)
    high = max(low * 5.0, 2.0 * max_duration)
    centers = np.geomspace(low, high, order)
    edges = np.sqrt(centers[:-1] * centers[1:]) if order > 1 else np.array([])
    lower = np.concatenate([[low / 5.0], edges])
    upper = np.concatenate([edges, [high * 5.0]])
    return np.log(centers), np.log(lower), np.log(upper)


def _unpack_parameters(
    vector: FloatArray,
    order: int,
    n_segments: int,
    drift_order: int,
    transient_initial_states: bool,
) -> tuple[float, FloatArray, FloatArray, list[FloatArray], list[FloatArray]]:
    cursor = 0
    r0 = float(np.exp(vector[cursor])); cursor += 1
    r = np.exp(vector[cursor : cursor + order]); cursor += order
    tau = np.exp(vector[cursor : cursor + order]); cursor += order
    initial_states: list[FloatArray] = []
    drift: list[FloatArray] = []
    for _ in range(n_segments):
        if transient_initial_states:
            initial_states.append(vector[cursor : cursor + order].copy()); cursor += order
        else:
            initial_states.append(np.zeros(order, dtype=float))
        drift.append(vector[cursor : cursor + drift_order + 1].copy()); cursor += drift_order + 1
    return r0, r, tau, initial_states, drift


def _simulate_drop(
    current: FloatArray,
    dt_s: float,
    r0: float,
    r: FloatArray,
    tau: FloatArray,
    initial_state: FloatArray,
    drift_coefficients: FloatArray,
) -> FloatArray:
    n = len(current)
    current = np.asarray(current, dtype=float)
    t_norm = np.linspace(-1.0, 1.0, n)
    drift = np.zeros(n, dtype=float)
    for p, coefficient in enumerate(drift_coefficients):
        drift += coefficient * t_norm**p
    predicted = r0 * current + drift
    alpha = np.exp(-dt_s / tau)
    for j in range(len(r)):
        aj = float(alpha[j])
        b = [float(r[j] * (1.0 - aj))]
        a = [1.0, -aj]
        zi = lfiltic(b, a, [float(initial_state[j])])
        branch, _ = lfilter(b, a, current, zi=zi)
        contribution = np.empty(n, dtype=float)
        contribution[0] = initial_state[j]
        contribution[1:] = branch[:-1]
        predicted += contribution
    return predicted


def fit_ecm_order(
    segments: list[FitSegment],
    order: int,
    drift_order: int,
    transient_initial_states: bool,
    max_nfev: int,
    bic_weight: float,
    voltage_weight: float,
    rng: np.random.Generator,
    *,
    fixed: bool = False,
) -> ECMFitResult:
    if order < 1:
        raise ValueError("ECM order must be positive.")
    n_segments = len(segments)
    min_dt = min(s.dt_s for s in segments)
    max_duration = max(len(s.current_a) * s.dt_s for s in segments)
    tau0, tau_lb, tau_ub = _soft_bounds_for_tau(order, min_dt, max_duration)
    ratios = []
    for segment in segments:
        denom = np.std(segment.current_a[segment.train_mask]) if fixed else np.std(segment.current_a)
        if denom > 1e-9:
            ratios.append((np.std(segment.voltage_drop_v[segment.train_mask]) if fixed else np.std(segment.voltage_drop_v)) / denom)
    total_r_guess = float(np.clip(np.median(ratios) if ratios else 0.005, 1e-4, 0.05))
    r0_guess = max(0.25 * total_r_guess, 1e-5)
    r_guess = np.full(order, max(0.75 * total_r_guess / order, 1e-5))

    x0_parts = [np.log(r0_guess), *np.log(r_guess), *tau0]
    lb_parts = [np.log(1e-6), *np.full(order, np.log(1e-7)), *tau_lb]
    ub_parts = [np.log(0.2), *np.full(order, np.log(0.2)), *tau_ub]
    for segment in segments:
        if transient_initial_states:
            x0_parts.extend(np.zeros(order))
            lb_parts.extend(np.full(order, -0.5))
            ub_parts.extend(np.full(order, 0.5))
        offset_guess = float(np.median((segment.voltage_drop_v - r0_guess * segment.current_a)[segment.train_mask])) if fixed else float(np.median(segment.voltage_drop_v - r0_guess * segment.current_a))
        x0_parts.extend([offset_guess] + [0.0] * drift_order)
        lb_parts.extend(np.full(drift_order + 1, -0.5))
        ub_parts.extend(np.full(drift_order + 1, 0.5))
    x0 = np.asarray(x0_parts, dtype=float)
    lb = np.asarray(lb_parts, dtype=float)
    ub = np.asarray(ub_parts, dtype=float)

    scales = [max(np.std(s.voltage_drop_v[s.train_mask]), 1e-4) for s in segments]

    def residual(vector: FloatArray, use_train: bool = True) -> FloatArray:
        r0, r, tau, initial, drift = _unpack_parameters(
            vector, order, n_segments, drift_order, transient_initial_states
        )
        chunks = []
        for idx, segment in enumerate(segments):
            pred = _simulate_drop(segment.current_a, segment.dt_s, r0, r, tau, initial[idx], drift[idx])
            mask = segment.train_mask if use_train else ~segment.train_mask
            if not np.any(mask):
                continue
            chunks.append((pred[mask] - segment.voltage_drop_v[mask]) / scales[idx])
        return np.concatenate(chunks) if chunks else np.array([1e6])

    starts = [x0]
    for _ in range(2):
        jitter = rng.normal(0.0, 0.15, size=len(x0))
        candidate = np.clip(x0 + jitter, lb + 1e-9, ub - 1e-9)
        starts.append(candidate)

    best = None
    start_time = perf_counter()
    for start in starts:
        try:
            fitted = least_squares(
                residual,
                start,
                bounds=(lb, ub),
                method="trf",
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=max_nfev,
                x_scale="jac",
            )
        except Exception as exc:
            continue
        if fixed:
            if not fitted.success or not np.all(np.isfinite(fitted.x)) or not np.isfinite(fitted.cost):
                continue
            cost = float(fitted.cost)
        else:
            cost = float(np.sum(residual(fitted.x, use_train=True) ** 2))
        if best is None or cost < best[0]:
            best = (cost, fitted)
    solve_time = perf_counter() - start_time
    if best is None:
        nan = np.full(order, np.nan)
        return ECMFitResult(order, np.nan, nan, nan, tuple(), tuple(), np.inf, np.inf, np.inf, np.inf, False, "Optimization failed", 0, solve_time, tuple())

    _, fitted = best
    r0, r, tau, initial, drift = _unpack_parameters(
        fitted.x, order, n_segments, drift_order, transient_initial_states
    )
    train_residual_v: list[FloatArray] = []
    val_residual_v: list[FloatArray] = []
    full_residual_v: list[FloatArray] = []
    for idx, segment in enumerate(segments):
        pred = _simulate_drop(segment.current_a, segment.dt_s, r0, r, tau, initial[idx], drift[idx])
        res = pred - segment.voltage_drop_v
        full_residual_v.append(res)
        train_residual_v.append(res[segment.train_mask])
        val_residual_v.append(res[~segment.train_mask])
    train_concat = np.concatenate(train_residual_v)
    val_concat = np.concatenate([r for r in val_residual_v if len(r)]) if any(len(r) for r in val_residual_v) else train_concat
    train_rmse_mv = 1e3 * float(np.sqrt(np.mean(train_concat**2)))
    val_rmse_mv = 1e3 * float(np.sqrt(np.mean(val_concat**2)))
    n_train = len(train_concat)
    k_params = len(fitted.x)
    rss = float(np.sum(train_concat**2))
    bic_per_sample = float(np.log(max(rss / max(n_train, 1), 1e-30)) + k_params * np.log(max(n_train, 2)) / max(n_train, 1))
    score = float(voltage_weight * val_rmse_mv + bic_weight * bic_per_sample)
    return ECMFitResult(
        order=order,
        r0_ohm=r0,
        rc_resistances_ohm=r,
        time_constants_s=tau,
        initial_states_v=tuple(initial),
        drift_coefficients_v=tuple(drift),
        train_rmse_mv=train_rmse_mv,
        validation_rmse_mv=val_rmse_mv,
        bic_per_sample=bic_per_sample,
        selection_score=score,
        success=bool(fitted.success),
        message=str(fitted.message),
        nfev=int(fitted.nfev),
        solve_time_s=float(solve_time),
        residuals_v=tuple(full_residual_v),
    )


def _select_model_order_single(
    segments: list[FitSegment],
    candidate_orders: Sequence[int],
    cfg: dict,
    rng: np.random.Generator,
    *,
    fixed: bool = False,
) -> tuple[ECMFitResult, list[ECMFitResult]]:
    ident = cfg["identification"]
    fits = [
        fit_ecm_order(
            segments,
            int(order),
            int(ident["drift_polynomial_order"]),
            bool(ident["transient_initial_states"]),
            int(ident["optimizer_max_nfev"]),
            float(ident["model_selection_bic_weight"]),
            float(ident["model_selection_voltage_weight"]),
            rng,
            fixed=fixed,
        )
        for order in candidate_orders
    ]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.selection_score)]
    if not valid:
        return replace(fits[0], success=False, message="No converged candidate with a finite selection score"), fits
    return min(valid, key=lambda fit: fit.selection_score), fits


SLOW_BRANCH_RATIO_THRESHOLD = 3.0
SLOW_BRANCH_REFERENCE_HZ = 0.008
SLOW_BRANCH_REFIT_ORDERS = (1, 2)
SLOW_BRANCH_REFIT_DRIFT_ORDER = 0


def slow_branch_ratio(fit: ECMFitResult) -> float:
    if not fit.success or not np.all(np.isfinite(fit.rc_resistances_ohm)) or not np.isfinite(fit.r0_ohm):
        return float("nan")
    extrapolated = float(fit.r0_ohm + np.sum(fit.rc_resistances_ohm))
    reference = float(np.real(fit.impedance(np.array([SLOW_BRANCH_REFERENCE_HZ]))[0]))
    return extrapolated / reference if reference > 0 else float("inf")


def select_model_order(
    segments: list[FitSegment],
    candidate_orders: Sequence[int],
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[ECMFitResult, list[ECMFitResult]]:
    start_state = copy.deepcopy(rng.bit_generator.state)
    selected, fits = _select_model_order_single(segments, candidate_orders, cfg, rng)
    ratio = slow_branch_ratio(selected)
    if not (ratio > SLOW_BRANCH_RATIO_THRESHOLD):
        return selected, fits
    refit_rng = np.random.default_rng()
    refit_rng.bit_generator.state = copy.deepcopy(start_state)
    refit_cfg = dict(cfg)
    refit_cfg["identification"] = dict(cfg["identification"])
    refit_cfg["identification"]["drift_polynomial_order"] = SLOW_BRANCH_REFIT_DRIFT_ORDER
    refit, _ = _select_model_order_single(segments, SLOW_BRANCH_REFIT_ORDERS, refit_cfg, refit_rng, fixed=True)
    refit_ratio = slow_branch_ratio(refit)
    if refit.success and refit_ratio <= SLOW_BRANCH_RATIO_THRESHOLD:
        note = (f"slow-branch refit: original order {selected.order} ratio {ratio:.4g} replaced by "
                f"constant-drift order {refit.order} ratio {refit_ratio:.4g}; {refit.message}")
        return replace(refit, message=note), fits
    note = (f"slow-branch check: ratio {ratio:.4g} > {SLOW_BRANCH_RATIO_THRESHOLD}; refit unresolved "
            f"(success={refit.success}, ratio {refit_ratio:.4g}); original kept; {selected.message}")
    return replace(selected, message=note), fits


def predict_polarization_drop(current_a: FloatArray, dt_s: float, fit: ECMFitResult) -> FloatArray:
    return _simulate_drop(
        np.asarray(current_a, dtype=float),
        dt_s,
        fit.r0_ohm,
        fit.rc_resistances_ohm,
        fit.time_constants_s,
        np.zeros(fit.order, dtype=float),
        np.zeros(1, dtype=float),
    )


def impedance_nrmse(
    fit: ECMFitResult,
    truth_model: ECMTruthModel,
    frequencies_hz: FloatArray,
    temperature_c: float,
    soh: float,
    soc: float,
) -> float:
    z_true = truth_model.impedance(frequencies_hz, temperature_c, soh, soc)
    z_fit = fit.impedance(frequencies_hz)
    return float(np.sqrt(np.mean(np.abs(z_fit - z_true) ** 2)) / (np.sqrt(np.mean(np.abs(z_true) ** 2)) + 1e-30))


def _moving_block_resample(residual: FloatArray, block_samples: int, rng: np.random.Generator) -> FloatArray:
    n = len(residual)
    if n == 0:
        return residual.copy()
    block_samples = max(1, min(block_samples, n))
    starts = rng.integers(0, max(1, n - block_samples + 1), size=int(np.ceil(n / block_samples)))
    sampled = np.concatenate([residual[s : s + block_samples] for s in starts])[:n]
    return sampled


def bootstrap_impedance_uncertainty(
    segments: list[FitSegment],
    selected_fit: ECMFitResult,
    frequencies_hz: FloatArray,
    cfg: dict,
    rng: np.random.Generator,
) -> BootstrapUncertainty:
    ident = cfg["identification"]
    replicates = int(ident["bootstrap_replicates"])
    confidence = float(ident["confidence_level"])
    samples: list[ComplexArray] = []
    for _ in range(replicates):
        boot_segments: list[FitSegment] = []
        for segment, residual in zip(segments, selected_fit.residuals_v, strict=True):
            block = max(1, int(round(float(ident["bootstrap_block_duration_s"]) / segment.dt_s)))
            synthetic = segment.voltage_drop_v + _moving_block_resample(residual, block, rng)
            boot_segments.append(FitSegment(segment.current_a, synthetic, segment.dt_s, segment.train_mask, segment.name))
        fit = fit_ecm_order(
            boot_segments,
            selected_fit.order,
            int(ident["drift_polynomial_order"]),
            bool(ident["transient_initial_states"]),
            max(100, int(ident["optimizer_max_nfev"]) // 2),
            float(ident["model_selection_bic_weight"]),
            float(ident["model_selection_voltage_weight"]),
            rng,
        )
        if fit.success and np.all(np.isfinite(fit.rc_resistances_ohm)):
            samples.append(fit.impedance(frequencies_hz))
    if not samples:
        z = selected_fit.impedance(frequencies_hz)
        nan = np.full(len(frequencies_hz), np.nan)
        return BootstrapUncertainty(frequencies_hz, z, nan, nan, nan, nan, nan, 0)
    arr = np.vstack(samples)
    alpha = 1.0 - confidence
    return BootstrapUncertainty(
        frequency_hz=np.asarray(frequencies_hz, dtype=float),
        impedance_mean_ohm=np.mean(arr, axis=0),
        impedance_std_ohm=np.std(np.abs(arr), axis=0, ddof=1) if len(arr) > 1 else np.zeros(len(frequencies_hz)),
        lower_real_ohm=np.quantile(arr.real, alpha / 2.0, axis=0),
        upper_real_ohm=np.quantile(arr.real, 1.0 - alpha / 2.0, axis=0),
        lower_imag_ohm=np.quantile(arr.imag, alpha / 2.0, axis=0),
        upper_imag_ohm=np.quantile(arr.imag, 1.0 - alpha / 2.0, axis=0),
        successful_replicates=len(arr),
    )
