from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import os
import math
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from .estimation import (
    bootstrap_impedance_uncertainty,
    impedance_nrmse,
    predict_polarization_drop,
    prepare_fit_segments,
    select_model_order,
    spectral_impedance,
    target_frequency_coverage,
)
from .fim import fisher_information
from .backends import build_truth_pack
from .models import ECMTruthModel, parameters_from_config, parameters_from_prior_config
from .plotting import (
    plot_chemistry_comparison,
    plot_duration_accuracy,
    plot_excitation_spectra,
    plot_metric_boxplot,
)
from .scheduler import CellDiagnosticState, PairSchedule, schedule_pairs
from .signals import (
    ExcitationDesign,
    candidate_segment_pool,
    concatenate_segments,
    spectral_coverage_score,
    synthetic_drive_current,
    validation_current_profile,
)
from .simulation import simulate_diagnostic_window
from .statistics import empirical_coverage, paired_method_comparisons, split_conformal_radius


DEFAULT_METHOD_SPECS = {
    "proposed": {"design": "proposed", "pairing": "mismatch_aware", "orientation": "optimized"},
    "random_pairing": {"design": "proposed", "pairing": "random", "orientation": "optimized"},
    "coverage_sequence": {"design": "spectral_baseline", "pairing": "mismatch_aware", "orientation": "optimized"},
    "short_sequence": {"design": "short_baseline", "pairing": "mismatch_aware", "orientation": "optimized"},
}


def _method_specs(cfg: dict) -> dict[str, dict[str, str]]:
    raw = cfg.get("methods")
    specs = DEFAULT_METHOD_SPECS if raw is None else raw
    out: dict[str, dict[str, str]] = {}
    for name, spec in specs.items():
        design = str(spec["design"])
        pairing = str(spec["pairing"])
        orientation = str(spec.get("orientation", "optimized"))
        if pairing not in {"mismatch_aware", "random"}:
            raise ValueError(f"Unsupported pairing method for {name}: {pairing}")
        if orientation not in {"optimized", "fixed_positive"}:
            raise ValueError(f"Unsupported orientation mode for {name}: {orientation}")
        out[str(name)] = {"design": design, "pairing": pairing, "orientation": orientation}
    if "proposed" not in out:
        raise ValueError("The method set must contain a 'proposed' method.")
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular base_config inheritance detected at {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    base_name = cfg.pop("base_config", None)
    if base_name is None:
        return cfg
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base_cfg = load_config(base_path, seen)
    return _deep_merge(base_cfg, cfg)


def _common_dt(cfg: dict) -> float:
    periods: list[float] = []
    for design_name in ("proposed", "short_baseline", "spectral_baseline"):
        design_cfg = cfg["excitation"][design_name]
        periods.extend(
            [float(design_cfg["low_band"]["bit_period_s"]), float(design_cfg["high_band"]["bit_period_s"])]
        )
    ticks = [max(1, int(round(period * 1_000_000))) for period in periods]
    common_ticks = ticks[0]
    for tick in ticks[1:]:
        common_ticks = math.gcd(common_ticks, tick)
    return common_ticks / 1_000_000.0


def _robust_joint_score(low_q: np.ndarray, high_q: np.ndarray, dt_s: float, target_f: np.ndarray, cfg: dict, current_amplitude: float) -> float:
    q = np.concatenate([low_q, high_q])
    scores = []
    for chemistry in cfg["study"]["chemistries"]:
        p = parameters_from_prior_config(chemistry, cfg)
        model = ECMTruthModel(p)
        nominal = model.effective_parameters(25.0, 0.90, 0.50)
        fim = fisher_information(
            q,
            dt_s,
            target_f,
            nominal,
            voltage_noise_std_v=float(max(cfg["study"]["voltage_noise_std_v"])),
            current_amplitude_a=max(current_amplitude / 2.0, 1.0),
            low_frequency_weight_power=float(cfg["excitation"]["proposed"]["low_frequency_weight_power"]),
            regularization=float(cfg["excitation"]["proposed"]["fim_regularization"]),
            objective=str(cfg["excitation"]["proposed"]["objective"]),
        )
        scores.append(fim.score)
    return float(min(scores))


def build_excitation_designs(cfg: dict) -> dict[str, ExcitationDesign]:
    common_dt = _common_dt(cfg)
    target = np.asarray(cfg["excitation"]["proposed"]["target_frequencies_hz"], dtype=float)
    target = target[(target > 0) & (target < 0.45 / common_dt)]
    n_candidates = int(cfg["excitation"]["proposed"].get("candidate_seeds_per_polynomial", 12))
    base_seed = int(cfg["project"]["random_seed"])
    current_amplitude = float(cfg["study"]["baseline_current_a"])

    designs: dict[str, ExcitationDesign] = {}

    proposed_cfg = cfg["excitation"]["proposed"]
    low_candidates = candidate_segment_pool(
        "low_band", proposed_cfg["low_band"], common_dt, n_candidates, base_seed + 100_000 * 0,
    )
    high_candidates = candidate_segment_pool(
        "high_band", proposed_cfg["high_band"], common_dt, n_candidates, base_seed + 100_000 * 1,
    )
    best_pair = None
    for low_c in low_candidates:
        for high_c in high_candidates:
            pair_score = _robust_joint_score(low_c.q, high_c.q, common_dt, target, cfg, current_amplitude)
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, low_c, high_c)
    best_score, best_low, best_high = best_pair
    designs["proposed"] = concatenate_segments(
        "proposed",
        [best_low, best_high],
        target,
        score=float(best_score),
        metadata={
            "criterion": "chemistry-robust maximin Fisher information (joint low/high pair search)",
            "candidate_count_per_band": n_candidates,
            "candidate_pairs_evaluated": n_candidates * n_candidates,
        },
    )

    baseline_n_candidates = int(cfg["excitation"].get("baseline_candidate_seeds_per_polynomial", 6))
    for design_name in ("spectral_baseline", "short_baseline"):
        design_cfg = cfg["excitation"][design_name]
        segments = []
        scores = []
        for band_idx, band_name in enumerate(("low_band", "high_band")):
            candidates = candidate_segment_pool(
                band_name,
                design_cfg[band_name],
                common_dt,
                max(4, baseline_n_candidates),
                base_seed + 200_000 + 10_000 * band_idx + (1 if design_name == "short_baseline" else 0),
            )
            scored = [(c, spectral_coverage_score(c.q, c.dt_s, target)) for c in candidates]
            best, score = max(scored, key=lambda item: item[1])
            segments.append(best)
            scores.append(score)
        designs[design_name] = concatenate_segments(
            design_name,
            segments,
            target,
            score=float(sum(scores)),
            metadata={"criterion": "spectral coverage heuristic"},
        )

    for name, design in list(designs.items()):
        logdets: list[float] = []
        conditions: list[float] = []
        by_chemistry: dict[str, dict[str, float]] = {}
        for chemistry in cfg["study"]["chemistries"]:
            p = parameters_from_prior_config(chemistry, cfg)
            nominal = ECMTruthModel(p).effective_parameters(25.0, 0.90, 0.50)
            fim = fisher_information(
                design.q,
                design.dt_s,
                target,
                nominal,
                voltage_noise_std_v=float(max(cfg["study"]["voltage_noise_std_v"])),
                current_amplitude_a=max(current_amplitude / 2.0, 1.0),
                low_frequency_weight_power=float(cfg["excitation"]["proposed"]["low_frequency_weight_power"]),
                regularization=float(cfg["excitation"]["proposed"]["fim_regularization"]),
                objective=str(cfg["excitation"]["proposed"]["objective"]),
            )
            logdets.append(fim.logdet)
            conditions.append(fim.condition_number)
            by_chemistry[chemistry] = {
                "logdet": float(fim.logdet),
                "condition_number": float(fim.condition_number),
            }
        metadata = dict(design.metadata)
        metadata.update({
            "fim_by_chemistry": by_chemistry,
            "worst_chemistry_logdet": float(min(logdets)) if logdets else float("nan"),
            "worst_condition_number": float(max(conditions)) if conditions else float("nan"),
        })
        designs[name] = ExcitationDesign(
            name=design.name,
            q=design.q,
            dt_s=design.dt_s,
            segments=design.segments,
            target_frequencies_hz=design.target_frequencies_hz,
            score=design.score,
            metadata=metadata,
        )
    return designs


def _prior_diagnostic_states(
    pack: list[ECMTruthModel],
    soc: np.ndarray,
    temperature_c: np.ndarray,
    soh: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    remaining: set[int],
    attempts: dict[int, int] | None = None,
) -> list[CellDiagnosticState]:
    f = np.asarray(cfg["pairing"]["impedance_frequencies_hz"], dtype=float)
    states: list[CellDiagnosticState] = []
    for idx in sorted(remaining):
        z_true = pack[idx].impedance(f, temperature_c[idx], soh[idx], soc[idx])
        magnitude_error = rng.normal(0.0, 0.05, size=len(f))
        phase_error = rng.normal(0.0, np.deg2rad(2.0), size=len(f))
        z_prior = z_true * (1.0 + magnitude_error) * np.exp(1j * phase_error)
        std = 0.06 * np.abs(z_prior) + 5e-6
        retry_boost = float((attempts or {}).get(idx, 0))
        priority = 1.0 + retry_boost + 0.5 * float(np.mean(std / np.maximum(np.abs(z_prior), 1e-9))) + rng.uniform(0.0, 0.05)
        states.append(
            CellDiagnosticState(
                index=idx,
                soc=float(soc[idx]),
                temperature_c=float(temperature_c[idx]),
                soh=float(soh[idx]),
                ocv_v=float(pack[idx].ocv(soc[idx])),
                impedance_ohm=np.asarray(z_prior, dtype=complex),
                impedance_std_ohm=np.asarray(std, dtype=float),
                priority=priority,
            )
        )
    return states


def _single_method_run(
    pack: list[ECMTruthModel],
    chemistry: str,
    pack_id: int,
    operating_point_id: int,
    operating_point: dict,
    noise_id: int,
    voltage_noise_std_v: float,
    current_noise_std_a: float,
    soc_bias_std: float,
    timing_jitter_std_s: float,
    method: str,
    design: ExcitationDesign,
    pairing_method: str,
    orientation_mode: str,
    cfg: dict,
    seed: int,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    scenario_seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    _t_method0 = perf_counter()
    print(f"[TIMING] method={method} START n_cells={len(pack)}", flush=True)
    rng = np.random.default_rng(seed)
    n_cells = len(pack)
    base_soc = float(operating_point["soc"])
    base_temp = float(operating_point["temperature_c"])
    base_soh = float(operating_point["soh"])
    if initial_state is None:
        soc = np.clip(rng.normal(base_soc, 0.012, size=n_cells), 0.05, 0.98)
        temperature = rng.normal(base_temp, 1.5, size=n_cells)
        soh = np.clip(rng.normal(base_soh, 0.012, size=n_cells), 0.70, 1.0)
    else:
        soc = np.asarray(initial_state[0], dtype=float).copy()
        temperature = np.asarray(initial_state[1], dtype=float).copy()
        soh = np.asarray(initial_state[2], dtype=float).copy()
        if not (len(soc) == len(temperature) == len(soh) == n_cells):
            raise ValueError("Initial-state arrays must match the pack size.")
    common_seed = int(seed if scenario_seed is None else scenario_seed)
    nominal_model = ECMTruthModel(parameters_from_prior_config(chemistry, cfg))
    remaining = set(range(n_cells))
    attempts = {i: 0 for i in range(n_cells)}
    accepted_cells: set[int] = set()
    rows: list[dict[str, Any]] = []
    all_point_errors_mv: list[float] = []
    round_metrics: list[dict[str, float]] = []
    operational_records = []
    round_index = 0
    bootstrap_limit = int(cfg["identification"].get("bootstrap_cells_per_pack", 0))
    bootstrapped = 0

    while len(remaining) >= 2:
        states = _prior_diagnostic_states(pack, soc, temperature, soh, cfg, rng, remaining, attempts)
        schedule_t0 = perf_counter()
        schedule = schedule_pairs(states, cfg, pairing_method, rng)
        if orientation_mode == "fixed_positive" and schedule.pairs:
            schedule = PairSchedule(
                pairs=schedule.pairs,
                orientations=tuple(1 for _ in schedule.pairs),
                total_cost=schedule.total_cost,
                method=schedule.method,
            )
        scheduling_time_ms = 1e3 * (perf_counter() - schedule_t0)
        if not schedule.pairs:
            break
        design_code = sum((idx + 1) * ord(ch) for idx, ch in enumerate(design.name))
        current_rng = np.random.default_rng(common_seed + 1_000_003 * round_index + design_code)
        pack_current = synthetic_drive_current(
            design.duration_s,
            design.dt_s,
            float(cfg["study"]["baseline_current_a"]),
            current_rng,
            minimum_current_a=float(cfg["architecture"]["minimum_current_a"]),
        )
        _t_diag0 = perf_counter()
        window = simulate_diagnostic_window(
            pack,
            schedule,
            design,
            pack_current,
            soc,
            temperature,
            soh,
            np.asarray([nominal_model.capacity_ah(float(h)) for h in soh], dtype=float),
            voltage_noise_std_v,
            current_noise_std_a,
            soc_bias_std,
            timing_jitter_std_s,
            float(cfg["architecture"]["switch_on_resistance_ohm"]),
            float(cfg["architecture"]["pack_power_reference_w"]),
            rng,
        )
        print(f"[TIMING] method={method} round={round_index} diagnostic_window "
              f"n_pairs={len(schedule.pairs)} elapsed={perf_counter()-_t_diag0:.2f}s", flush=True)
        operational_records.extend(dict(record,round_index=round_index) for record in window.operational_power_records)
        round_metrics.append(
            {
                "rms_power_disturbance_pct": window.relative_rms_power_disturbance_pct,
                "peak_power_disturbance_pct": window.relative_peak_power_disturbance_pct,
                "scheduling_time_ms": scheduling_time_ms,
            }
        )
        for measurement in window.measurements:
            cell_idx = measurement.cell_index
            _t_cell0 = perf_counter()
            spectral = spectral_impedance(
                measurement,
                design,
                nominal_model,
                float(cfg["identification"]["coherence_threshold"]),
                float(cfg["identification"]["minimum_excitation_snr_db"]),
            )
            segments = prepare_fit_segments(
                measurement,
                design,
                nominal_model,
                float(cfg["identification"]["blocked_validation_fraction"]),
            )
            _t_spectral = perf_counter()
            fit, all_fits = select_model_order(
                segments,
                cfg["identification"]["candidate_rc_orders"],
                cfg,
                rng,
            )
            _t_fit = perf_counter()
            per_order = ", ".join(f"o{f.order}={f.solve_time_s:.1f}s/nfev{f.nfev}" for f in all_fits)
            print(f"[TIMING] method={method} round={round_index} cell={cell_idx} "
                  f"spectral+prep={_t_spectral-_t_cell0:.2f}s select_model_order_total={_t_fit-_t_spectral:.2f}s "
                  f"[{per_order}]", flush=True)
            target_f = np.asarray(design.target_frequencies_hz, dtype=float)
            target_f = target_f[target_f < 0.45 / design.dt_s]
            z_nrmse = impedance_nrmse(fit, pack[cell_idx], target_f, temperature[cell_idx], soh[cell_idx], soc[cell_idx])
            rdc_true = pack[cell_idx].dc_resistance_ohm(float(temperature[cell_idx]), float(soh[cell_idx]), float(soc[cell_idx]))
            rdc_fit = fit.r0_ohm + float(np.sum(fit.rc_resistances_ohm))
            resistance_error_pct = 100.0 * abs(rdc_fit - rdc_true) / max(rdc_true, 1e-12)

            validation_dt = float(cfg["study"]["independent_validation_dt_s"])
            validation_rng = np.random.default_rng(common_seed + 7_000_001 + 10_007 * cell_idx)
            validation_current = validation_current_profile(
                float(cfg["study"]["independent_validation_duration_s"]),
                validation_dt,
                float(cfg["study"]["baseline_current_a"]),
                validation_rng,
            )
            _t_val0 = perf_counter()
            truth_validation = pack[cell_idx].simulate(
                validation_current,
                validation_dt,
                soc[cell_idx],
                temperature[cell_idx],
                soh[cell_idx],
            )
            print(f"[TIMING] method={method} round={round_index} cell={cell_idx} "
                  f"validation_simulate={perf_counter()-_t_val0:.2f}s", flush=True)
            measured_validation_current = validation_current + validation_rng.normal(
                0.0, current_noise_std_a, size=len(validation_current)
            )
            validation_soc_est = np.empty(len(validation_current), dtype=float)
            validation_soc_est[0] = float(np.clip(
                soc[cell_idx] + validation_rng.normal(0.0, soc_bias_std), 0.0, 1.0
            ))
            capacity_as = nominal_model.capacity_ah(float(soh[cell_idx])) * 3600.0
            for validation_k in range(len(validation_current) - 1):
                validation_soc_est[validation_k + 1] = np.clip(
                    validation_soc_est[validation_k]
                    - validation_dt * measured_validation_current[validation_k] / capacity_as,
                    0.0,
                    1.0,
                )
            predicted_drop = predict_polarization_drop(measured_validation_current, validation_dt, fit)
            predicted_voltage = nominal_model.ocv(validation_soc_est) - predicted_drop
            validation_error = predicted_voltage - truth_validation["voltage_v"]
            voltage_rmse_mv = 1e3 * float(np.sqrt(np.mean(validation_error**2)))
            voltage_mae_mv = 1e3 * float(np.mean(np.abs(validation_error)))
            voltage_q95_mv = 1e3 * float(np.quantile(np.abs(validation_error), 0.95))

            uncertainty_mean_std_mohm = np.nan
            interval_coverage = np.nan
            if bootstrapped < bootstrap_limit and int(cfg["identification"]["bootstrap_replicates"]) > 0:
                uncertainty = bootstrap_impedance_uncertainty(segments, fit, target_f, cfg, rng)
                uncertainty_mean_std_mohm = 1e3 * float(np.nanmean(uncertainty.impedance_std_ohm))
                if uncertainty.successful_replicates > 0:
                    z_true = pack[cell_idx].impedance(target_f, temperature[cell_idx], soh[cell_idx], soc[cell_idx])
                    covered_real = (z_true.real >= uncertainty.lower_real_ohm) & (z_true.real <= uncertainty.upper_real_ohm)
                    covered_imag = (z_true.imag >= uncertainty.lower_imag_ohm) & (z_true.imag <= uncertainty.upper_imag_ohm)
                    interval_coverage = float(np.mean(covered_real & covered_imag))
                bootstrapped += 1

            spectral_valid_fraction = target_frequency_coverage(spectral, target_f)
            acceptance_cfg = cfg["identification"].get("acceptance", {})
            reasons: list[str] = []
            if not fit.success:
                reasons.append("fit_failed")
            if spectral_valid_fraction < float(acceptance_cfg.get("minimum_valid_frequency_fraction", 0.25)):
                reasons.append("insufficient_frequency_coverage")
            if fit.validation_rmse_mv > float(acceptance_cfg.get("maximum_blocked_validation_rmse_mv", 30.0)):
                reasons.append("blocked_validation_error")
            if voltage_rmse_mv > float(acceptance_cfg.get("maximum_independent_voltage_rmse_mv", 40.0)):
                reasons.append("independent_validation_error")
            if not (np.all(np.isfinite(fit.rc_resistances_ohm)) and np.all(fit.rc_resistances_ohm > 0)
                    and np.all(np.isfinite(fit.time_constants_s)) and np.all(np.diff(fit.time_constants_s) > 0)):
                reasons.append("nonphysical_parameters")
            accepted_update = len(reasons) == 0
            print(f"[TIMING] method={method} round={round_index} cell={cell_idx} "
                  f"accepted={accepted_update} reasons={reasons} "
                  f"blocked_val_rmse_mv={fit.validation_rmse_mv:.2f} indep_val_rmse_mv={voltage_rmse_mv:.2f} "
                  f"spectral_valid_frac={spectral_valid_fraction:.2f} order={fit.order}", flush=True)
            attempts[cell_idx] += 1
            if accepted_update:
                accepted_cells.add(cell_idx)
                all_point_errors_mv.extend((1e3 * validation_error).tolist())
                remaining.discard(cell_idx)
            elif attempts[cell_idx] >= int(acceptance_cfg.get("maximum_attempts_per_cell", 2)):
                remaining.discard(cell_idx)
                reasons.append("attempt_limit")

            row = {
                "chemistry": chemistry,
                "pack_id": pack_id,
                "operating_point_id": operating_point_id,
                "noise_id": noise_id,
                "cell_index": cell_idx,
                "round_index": round_index,
                "method": method,
                "excitation_design": design.name,
                "pairing_method": pairing_method,
                "orientation_mode": orientation_mode,
                "pack_size": n_cells,
                "soc": soc[cell_idx],
                "temperature_c": temperature[cell_idx],
                "soh": soh[cell_idx],
                "voltage_noise_std_mv": 1e3 * voltage_noise_std_v,
                "current_noise_std_a": current_noise_std_a,
                "soc_bias_std_pct": 100.0 * soc_bias_std,
                "timing_jitter_std_us": 1e6 * timing_jitter_std_s,
                "diagnostic_window_s": design.duration_s,
                "reliable_lowest_frequency_hz": design.reliable_lowest_frequency_hz,
                "spectral_valid_fraction": spectral_valid_fraction,
                "accepted_update": accepted_update,
                "rejection_reason": "accepted" if accepted_update else ";".join(reasons),
                "attempt_number": attempts[cell_idx],
                "selected_order": fit.order,
                "fit_train_rmse_mv": fit.train_rmse_mv,
                "fit_blocked_validation_rmse_mv": fit.validation_rmse_mv,
                "validation_voltage_rmse_mv": voltage_rmse_mv,
                "validation_voltage_mae_mv": voltage_mae_mv,
                "validation_abs_error_q95_mv": voltage_q95_mv,
                "impedance_nrmse_pct": 100.0 * z_nrmse,
                "total_resistance_error_pct": resistance_error_pct,
                "fit_solve_time_s": fit.solve_time_s,
                "fit_nfev": fit.nfev,
                "bootstrap_mean_impedance_std_mohm": uncertainty_mean_std_mohm,
                "bootstrap_joint_interval_coverage": interval_coverage,
                "legacy_round_rms_power_disturbance_pct": window.relative_rms_power_disturbance_pct,
                "legacy_round_peak_power_disturbance_pct": window.relative_peak_power_disturbance_pct,
                "pairing_cost": schedule.total_cost,
                "round_scheduling_time_ms": scheduling_time_ms,
                "model_selection_scores": json.dumps({f.order: f.selection_score for f in all_fits}),
            }
            rows.append(row)
            soc[cell_idx] = measurement.true_soc[-1]
            temperature[cell_idx] = measurement.temperature_c[-1]
        round_index += 1
        if round_index > n_cells * int(cfg["identification"].get("acceptance", {}).get("maximum_attempts_per_cell", 2)):
            break

    refresh_time = round_index * design.duration_s
    for row in rows:
        row["diagnostic_rounds"] = round_index
        row["diagnostic_refresh_time_s"] = refresh_time
        row["legacy_pack_median_rms_power_disturbance_pct"] = float(np.median([m["rms_power_disturbance_pct"] for m in round_metrics])) if round_metrics else np.nan
        row["legacy_pack_max_peak_power_disturbance_pct"] = float(np.max([m["peak_power_disturbance_pct"] for m in round_metrics])) if round_metrics else np.nan
        row["pack_median_scheduling_time_ms"] = float(np.median([m["scheduling_time_ms"] for m in round_metrics])) if round_metrics else np.nan
        row["pack_p95_scheduling_time_ms"] = float(np.quantile([m["scheduling_time_ms"] for m in round_metrics], 0.95)) if round_metrics else np.nan
        row["pack_coverage_fraction"] = len(accepted_cells) / n_cells
        row["pack_attempted_fraction"] = len({r["cell_index"] for r in rows}) / n_cells
    metadata = {
        "method": method,
        "operational_power_records": operational_records,
        "rounds": round_index,
        "refresh_time_s": refresh_time,
        "coverage_fraction": len(accepted_cells) / n_cells if rows else 0.0,
        "attempted_fraction": len({r["cell_index"] for r in rows}) / n_cells if rows else 0.0,
    }
    print(f"[TIMING] method={method} END total={perf_counter()-_t_method0:.2f}s rounds={round_index}", flush=True)
    return rows, all_point_errors_mv, metadata


def _scenario_run(
    chemistry: str,
    pack_id: int,
    operating_point_id: int,
    operating_point: dict,
    noise_id: int,
    noise_tuple: tuple[float, float, float, float],
    pack_size: int,
    designs: dict[str, ExcitationDesign],
    cfg: dict,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    pack = build_truth_pack(chemistry, pack_size, cfg, rng)
    base_soc = float(operating_point["soc"])
    base_temp = float(operating_point["temperature_c"])
    base_soh = float(operating_point["soh"])
    initial_state = (
        np.clip(rng.normal(base_soc, 0.012, size=pack_size), 0.05, 0.98),
        rng.normal(base_temp, 1.5, size=pack_size),
        np.clip(rng.normal(base_soh, 0.012, size=pack_size), 0.70, 1.0),
    )
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {}
    metadata: list[dict[str, Any]] = []
    vnoise, inoise, socbias, jitter = noise_tuple
    for method_index, (method, spec) in enumerate(_method_specs(cfg).items()):
        design_name = spec["design"]
        pairing_method = spec["pairing"]
        orientation_mode = spec["orientation"]
        method_rows, method_errors, method_meta = _single_method_run(
            pack,
            chemistry,
            pack_id,
            operating_point_id,
            operating_point,
            noise_id,
            vnoise,
            inoise,
            socbias,
            jitter,
            method,
            designs[design_name],
            pairing_method,
            orientation_mode,
            cfg,
            seed + 10_000 * (method_index + 1),
            initial_state=initial_state,
            scenario_seed=seed,
        )
        rows.extend(method_rows)
        errors[method] = method_errors
        metadata.append(method_meta)
    return rows, errors, metadata


def _cartesian_noise(cfg: dict) -> list[tuple[float, float, float, float]]:
    explicit = cfg["study"].get("perturbation_conditions")
    if explicit:
        required = ("voltage_noise_std_v", "current_noise_std_a", "soc_bias_std", "switch_timing_jitter_std_s")
        conditions: list[tuple[float, float, float, float]] = []
        for condition in explicit:
            missing = [key for key in required if key not in condition]
            if missing:
                raise ValueError(f"Perturbation condition is missing {missing}: {condition}")
            conditions.append(tuple(float(condition[key]) for key in required))
        return conditions
    return [
        (float(v), float(i), float(s), float(j))
        for v in cfg["study"]["voltage_noise_std_v"]
        for i in cfg["study"]["current_noise_std_a"]
        for s in cfg["study"]["soc_bias_std"]
        for j in cfg["study"]["switch_timing_jitter_std_s"]
    ]


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["chemistry", "pack_id", "operating_point_id", "noise_id", "method", "pack_size"]
    accepted_metrics = [
        "validation_voltage_rmse_mv", "validation_voltage_mae_mv", "validation_abs_error_q95_mv",
        "impedance_nrmse_pct", "total_resistance_error_pct", "fit_blocked_validation_rmse_mv",
        "spectral_valid_fraction", "selected_order", "fit_solve_time_s",
        "bootstrap_joint_interval_coverage",
    ]
    conditioned = raw.copy()
    accepted_mask = conditioned["accepted_update"].astype(bool)
    conditioned.loc[~accepted_mask, accepted_metrics] = np.nan
    aggregation = {metric: ("mean" if metric == "bootstrap_joint_interval_coverage" else "median") for metric in accepted_metrics}
    aggregation.update({
        "diagnostic_window_s": "first",
        "diagnostic_refresh_time_s": "first",
        "diagnostic_rounds": "first",
        "legacy_pack_median_rms_power_disturbance_pct": "first",
        "legacy_pack_max_peak_power_disturbance_pct": "first",
        "pack_median_scheduling_time_ms": "first",
        "pack_p95_scheduling_time_ms": "first",
        "pack_coverage_fraction": "first",
        "pack_attempted_fraction": "first",
        "accepted_update": "mean",
    })
    out = conditioned.groupby(group_cols, as_index=False).agg(aggregation)
    return out.rename(columns={"accepted_update": "accepted_update_fraction"})


def _fmt(value: float, digits: int = 2) -> str:
    return "[RESULT]" if not np.isfinite(value) else f"{value:.{digits}f}"


def _write_results_macros(
    summary: pd.DataFrame,
    statistical: pd.DataFrame,
    conformal: dict[str, float],
    designs: dict[str, ExcitationDesign],
    output: Path,
) -> None:
    proposed = summary[summary["method"] == "proposed"]
    coverage = summary[summary["method"] == "coverage_sequence"]
    short = summary[summary["method"] == "short_sequence"]
    random_pair = summary[summary["method"] == "random_pairing"]

    def med(frame: pd.DataFrame, metric: str) -> float:
        return float(frame[metric].median()) if len(frame) else np.nan

    macros = {
        "NPackRealizations": int(proposed[["chemistry", "pack_id"]].drop_duplicates().shape[0]) if len(proposed) else 0,
        "NCellFits": int(len(proposed)),
        "ProposedDuration": med(proposed, "diagnostic_window_s"),
        "ProposedRefreshTime": med(proposed, "diagnostic_refresh_time_s"),
        "ShortRefreshTime": med(short, "diagnostic_refresh_time_s"),
        "ProposedSchedulingTime": med(proposed, "pack_p95_scheduling_time_ms"),
        "ProcessorName": platform.processor() or platform.machine() or "[RESULT]",
        "ProposedVoltageRMSE": med(proposed, "validation_voltage_rmse_mv"),
        "CoverageVoltageRMSE": med(coverage, "validation_voltage_rmse_mv"),
        "ShortVoltageRMSE": med(short, "validation_voltage_rmse_mv"),
        "ProposedImpedanceNRMSE": med(proposed, "impedance_nrmse_pct"),
        "CoverageImpedanceNRMSE": med(coverage, "impedance_nrmse_pct"),
        "ShortImpedanceNRMSE": med(short, "impedance_nrmse_pct"),
        "ProposedPowerRMS": med(proposed, "legacy_pack_median_rms_power_disturbance_pct"),
        "RandomPairPowerRMS": med(random_pair, "legacy_pack_median_rms_power_disturbance_pct"),
        "ProposedPowerPeak": med(proposed, "legacy_pack_max_peak_power_disturbance_pct"),
        "ProposedCoverage": 100.0 * med(proposed, "pack_coverage_fraction"),
        "ProposedResistanceError": med(proposed, "total_resistance_error_pct"),
        "ProposedValidFraction": 100.0 * med(proposed, "spectral_valid_fraction"),
        "ProposedOrderMedian": med(proposed, "selected_order"),
        "ProposedFitTime": med(proposed, "fit_solve_time_s"),
        "ProposedBootstrapCoverage": 100.0 * med(proposed, "bootstrap_joint_interval_coverage"),
        "CoverageVoltageReduction": 100.0 * (med(coverage, "validation_voltage_rmse_mv") - med(proposed, "validation_voltage_rmse_mv")) / max(med(coverage, "validation_voltage_rmse_mv"), 1e-12),
        "ShortVoltageReduction": 100.0 * (med(short, "validation_voltage_rmse_mv") - med(proposed, "validation_voltage_rmse_mv")) / max(med(short, "validation_voltage_rmse_mv"), 1e-12),
        "CoverageImpedanceReduction": 100.0 * (med(coverage, "impedance_nrmse_pct") - med(proposed, "impedance_nrmse_pct")) / max(med(coverage, "impedance_nrmse_pct"), 1e-12),
        "RandomPairPowerReduction": 100.0 * (med(random_pair, "legacy_pack_median_rms_power_disturbance_pct") - med(proposed, "legacy_pack_median_rms_power_disturbance_pct")) / max(med(random_pair, "legacy_pack_median_rms_power_disturbance_pct"), 1e-12),
        "ConformalRadius": conformal.get("radius_mv", np.nan),
        "ConformalCoverage": 100.0 * conformal.get("test_coverage", np.nan),
        "FIMLogDetGain": float(designs["proposed"].metadata.get("worst_chemistry_logdet", np.nan))
            - float(designs["spectral_baseline"].metadata.get("worst_chemistry_logdet", np.nan)),
        "FIMConditionReduction": 100.0 * (
            float(designs["spectral_baseline"].metadata.get("worst_condition_number", np.nan))
            - float(designs["proposed"].metadata.get("worst_condition_number", np.nan))
        ) / max(float(designs["spectral_baseline"].metadata.get("worst_condition_number", np.nan)), 1e-12),
    }
    for chemistry in ("NMC", "LFP"):
        frame = proposed[proposed["chemistry"] == chemistry]
        macros[f"{chemistry}VoltageRMSE"] = med(frame, "validation_voltage_rmse_mv")
        macros[f"{chemistry}ImpedanceNRMSE"] = med(frame, "impedance_nrmse_pct")

    lines = ["% Auto-generated by run_study.py. Do not edit manually."]
    for name, value in macros.items():
        if isinstance(value, str):
            rendered = (value.replace("\\", r"\textbackslash{}")
                             .replace("&", r"\&")
                             .replace("%", r"\%")
                             .replace("_", r"\_")
                             .replace("#", r"\#"))
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = _fmt(float(value), 2)
        lines.append(f"\\renewcommand{{\\{name}}}{{{rendered}}}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tables(summary: pd.DataFrame, statistical: pd.DataFrame, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "validation_voltage_rmse_mv",
        "impedance_nrmse_pct",
        "total_resistance_error_pct",
        "legacy_pack_median_rms_power_disturbance_pct",
        "diagnostic_refresh_time_s",
    ]
    grouped = summary.groupby(["chemistry", "method"])[metrics].median().reset_index()
    grouped.to_latex(
        table_dir / "main_results.tex",
        index=False,
        float_format=lambda x: f"{x:.2f}",
        caption="Median pack-level and cell-level results across the frozen Monte Carlo campaign.",
        label="tab:main_results",
        escape=True,
    )
    statistical.to_latex(
        table_dir / "statistical_tests.tex",
        index=False,
        float_format=lambda x: f"{x:.4g}",
        caption="Paired comparisons against the proposed method. Differences are proposed minus comparator.",
        label="tab:statistical_tests",
        escape=True,
    )


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _hash_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tasks(cfg: dict) -> list[tuple]:
    noise_grid = _cartesian_noise(cfg)
    tasks: list[tuple] = []
    base_seed = int(cfg["project"]["random_seed"])
    counter = 0
    for chemistry in cfg["study"]["chemistries"]:
        for pack_size in cfg["study"]["pack_sizes"]:
            for pack_id in range(int(cfg["study"]["n_packs_per_chemistry"])):
                for op_id, op in enumerate(cfg["study"]["operating_points"]):
                    for noise_id, noise in enumerate(noise_grid):
                        counter += 1
                        tasks.append((chemistry, pack_id, op_id, op, noise_id, noise, int(pack_size), base_seed + 100_003 * counter))
    return tasks


def run_scenario_shard(
    config_path: str | Path,
    task_index: int,
    task_count: int,
    shard_output_dir: str | Path,
    n_jobs: int = 1,
) -> Path:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    if not (0 <= task_index < task_count):
        raise ValueError(f"task_index {task_index} out of range for task_count {task_count}")
    designs = build_excitation_designs(cfg)
    tasks = build_tasks(cfg)
    subset = tasks[task_index::task_count]
    shard_output_dir = Path(shard_output_dir)
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{task_index:04d}"
    if not subset:
        (shard_output_dir / f"shard_{tag}.done").write_text("empty\n", encoding="utf-8")
        return shard_output_dir

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_scenario_run)(chem, pack_id, op_id, op, noise_id, noise, pack_size, designs, cfg, seed)
        for chem, pack_id, op_id, op, noise_id, noise, pack_size, seed in subset
    )
    rows = [row for scenario_rows, _, _ in results for row in scenario_rows]
    raw = pd.DataFrame(rows)
    raw.to_csv(shard_output_dir / f"shard_{tag}_raw.csv", index=False)

    error_records = []
    for (chem, pack_id, op_id, op, noise_id, noise, pack_size, seed), (_, error_dict, _) in zip(subset, results, strict=True):
        for method, errors in error_dict.items():
            for err in errors:
                error_records.append({
                    "chemistry": chem, "pack_id": pack_id, "operating_point_id": op_id,
                    "noise_id": noise_id, "pack_size": pack_size, "method": method,
                    "abs_error_mv": abs(float(err)),
                })
    error_columns = ["chemistry", "pack_id", "operating_point_id", "noise_id", "pack_size", "method", "abs_error_mv"]
    pd.DataFrame(error_records, columns=error_columns).to_csv(shard_output_dir / f"shard_{tag}_errors.csv", index=False)
    (shard_output_dir / f"shard_{tag}.done").write_text("ok\n", encoding="utf-8")
    return shard_output_dir


def merge_scenario_shards(
    config_path: str | Path,
    shard_output_dir: str | Path,
    expected_task_count: int,
) -> dict[str, Path]:
    t0 = perf_counter()
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    cfg = load_config(config_path)
    output_dir = root / cfg["project"].get("output_dir", "outputs")
    overleaf_dir = root / cfg["project"].get("overleaf_dir", "overleaf")
    write_overleaf = bool(cfg["project"].get("write_overleaf", True))
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_overleaf:
        (overleaf_dir / "figures").mkdir(parents=True, exist_ok=True)
        (overleaf_dir / "tables").mkdir(parents=True, exist_ok=True)

    designs = build_excitation_designs(cfg)
    if write_overleaf:
        plot_excitation_spectra(designs, overleaf_dir / "figures" / "excitation_spectra.pdf")

    tasks = build_tasks(cfg)
    shard_output_dir = Path(shard_output_dir)
    done_files = sorted(shard_output_dir.glob("shard_*.done"))
    if len(done_files) != expected_task_count:
        raise RuntimeError(
            f"Expected {expected_task_count} completed shards, found {len(done_files)} "
            f"in {shard_output_dir}. Wait for every Slurm array task to finish before merging."
        )

    raw_frames = [pd.read_csv(p) for p in sorted(shard_output_dir.glob("shard_*_raw.csv"))]
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if raw.empty:
        raise RuntimeError("The merged study produced no result rows.")
    if raw["accepted_update"].dtype == object:
        raw["accepted_update"] = raw["accepted_update"].map({"True": True, "False": False})
    raw["accepted_update"] = raw["accepted_update"].astype(bool)
    raw_path = output_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)
    rejected_path = output_dir / "rejected_windows.csv"
    raw.loc[~raw["accepted_update"]].to_csv(rejected_path, index=False)
    summary = summarize_results(raw)
    summary_path = output_dir / "summary_results.csv"
    summary.to_csv(summary_path, index=False)

    id_columns = ["chemistry", "pack_id", "operating_point_id", "noise_id", "pack_size"]
    base_seed = int(cfg["project"]["random_seed"])
    statistical = paired_method_comparisons(
        summary,
        metrics=[
            "validation_voltage_rmse_mv",
            "impedance_nrmse_pct",
            "legacy_pack_median_rms_power_disturbance_pct",
            "diagnostic_refresh_time_s",
        ],
        reference_method="proposed",
        comparison_methods=list(cfg.get("statistics", {}).get(
            "comparison_methods",
            [name for name in _method_specs(cfg) if name != "proposed"],
        )),
        id_columns=id_columns,
        confidence=float(cfg["statistics"]["confidence_level"]),
        n_bootstrap=int(cfg["statistics"]["bootstrap_replicates"]),
        multiplicity_method=str(cfg["statistics"]["multiplicity_method"]),
        rng=np.random.default_rng(base_seed + 999_999),
    )
    statistical_path = output_dir / "statistical_tests.csv"
    statistical.to_csv(statistical_path, index=False)

    error_columns = ["chemistry", "pack_id", "operating_point_id", "noise_id", "pack_size", "method", "abs_error_mv"]
    error_frames = []
    for p in sorted(shard_output_dir.glob("shard_*_errors.csv")):
        try:
            error_frames.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue
    errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=error_columns)
    calibration_errors: list[float] = []
    test_errors: list[float] = []
    if not errors_df.empty:
        proposed_errors = errors_df.loc[errors_df["method"] == "proposed"]
        calibration_errors = proposed_errors.loc[proposed_errors["pack_id"] % 2 == 0, "abs_error_mv"].tolist()
        test_errors = proposed_errors.loc[proposed_errors["pack_id"] % 2 != 0, "abs_error_mv"].tolist()
    radius = split_conformal_radius(np.asarray(calibration_errors), float(cfg["identification"]["confidence_level"]))
    coverage = empirical_coverage(np.asarray(test_errors), radius)
    conformal = {"radius_mv": radius, "test_coverage": coverage, "n_calibration": len(calibration_errors), "n_test": len(test_errors)}
    (output_dir / "conformal_calibration.json").write_text(json.dumps(conformal, indent=2), encoding="utf-8")

    if write_overleaf:
        plot_metric_boxplot(summary, "validation_voltage_rmse_mv", "Independent voltage RMSE (mV)", overleaf_dir / "figures" / "voltage_rmse_boxplot.pdf")
        plot_metric_boxplot(summary, "impedance_nrmse_pct", r"Impedance NRMSE (\%)", overleaf_dir / "figures" / "impedance_nrmse_boxplot.pdf")
        plot_chemistry_comparison(summary, "validation_voltage_rmse_mv", "Median voltage RMSE (mV)", overleaf_dir / "figures" / "chemistry_voltage_comparison.pdf")
        plot_duration_accuracy(summary, overleaf_dir / "figures" / "duration_accuracy.pdf")
        _write_tables(summary, statistical, overleaf_dir / "tables")
        _write_results_macros(summary, statistical, conformal, designs, overleaf_dir / "results_macros.tex")

    manifest = {
        "project": cfg["project"]["name"],
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": perf_counter() - t0,
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "n_scenarios": len(tasks),
        "n_raw_rows": len(raw),
        "designs": {
            name: {
                "duration_s": design.duration_s,
                "reliable_lowest_frequency_hz": design.reliable_lowest_frequency_hz,
                "score": design.score,
                "metadata": design.metadata,
            }
            for name, design in designs.items()
        },
        "truth_backend": cfg.get("truth_backend", {}),
        "bms_prior_section": cfg.get("bms_prior", {}).get("parameter_section"),
        "method_specs": _method_specs(cfg),
        "paired_design_controls": {
            "shared_initial_cell_states": True,
            "shared_validation_profile_by_cell": True,
            "shared_pack_current_for_equal_design_and_round": True,
        },
        "synthetic_truth_warning": cfg.get("synthetic_truth", {}).get("warning", ""),
        "write_overleaf": write_overleaf,
        "outputs": [str(raw_path), str(summary_path), str(statistical_path), str(rejected_path)],
        "merged_from_shards": True,
        "shard_count": expected_task_count,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "raw": raw_path,
        "summary": summary_path,
        "statistics": statistical_path,
        "manifest": manifest_path,
        "rejected": rejected_path,
        **({"macros": overleaf_dir / "results_macros.tex"} if write_overleaf else {}),
    }


def run_study(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    cfg = load_config(config_path)
    output_dir = root / cfg["project"].get("output_dir", "outputs")
    overleaf_dir = root / cfg["project"].get("overleaf_dir", "overleaf")
    write_overleaf = bool(cfg["project"].get("write_overleaf", True))
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_overleaf:
        (overleaf_dir / "figures").mkdir(parents=True, exist_ok=True)
        (overleaf_dir / "tables").mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    designs = build_excitation_designs(cfg)
    if write_overleaf:
        plot_excitation_spectra(designs, overleaf_dir / "figures" / "excitation_spectra.pdf")

    noise_grid = _cartesian_noise(cfg)
    tasks = []
    base_seed = int(cfg["project"]["random_seed"])
    counter = 0
    for chemistry in cfg["study"]["chemistries"]:
        for pack_size in cfg["study"]["pack_sizes"]:
            for pack_id in range(int(cfg["study"]["n_packs_per_chemistry"])):
                for op_id, op in enumerate(cfg["study"]["operating_points"]):
                    for noise_id, noise in enumerate(noise_grid):
                        counter += 1
                        tasks.append((chemistry, pack_id, op_id, op, noise_id, noise, int(pack_size), base_seed + 100_003 * counter))

    n_jobs = int(cfg["project"].get("n_jobs", -1))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_scenario_run)(chem, pack_id, op_id, op, noise_id, noise, pack_size, designs, cfg, seed)
        for chem, pack_id, op_id, op, noise_id, noise, pack_size, seed in tasks
    )
    rows = [row for scenario_rows, _, _ in results for row in scenario_rows]
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("The study produced no result rows.")
    raw_path = output_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)
    rejected_path = output_dir / "rejected_windows.csv"
    raw.loc[~raw["accepted_update"].astype(bool)].to_csv(rejected_path, index=False)
    summary = summarize_results(raw)
    summary_path = output_dir / "summary_results.csv"
    summary.to_csv(summary_path, index=False)

    id_columns = ["chemistry", "pack_id", "operating_point_id", "noise_id", "pack_size"]
    statistical = paired_method_comparisons(
        summary,
        metrics=[
            "validation_voltage_rmse_mv",
            "impedance_nrmse_pct",
            "legacy_pack_median_rms_power_disturbance_pct",
            "diagnostic_refresh_time_s",
        ],
        reference_method="proposed",
        comparison_methods=list(cfg.get("statistics", {}).get(
            "comparison_methods",
            [name for name in _method_specs(cfg) if name != "proposed"],
        )),
        id_columns=id_columns,
        confidence=float(cfg["statistics"]["confidence_level"]),
        n_bootstrap=int(cfg["statistics"]["bootstrap_replicates"]),
        multiplicity_method=str(cfg["statistics"]["multiplicity_method"]),
        rng=np.random.default_rng(base_seed + 999_999),
    )
    statistical_path = output_dir / "statistical_tests.csv"
    statistical.to_csv(statistical_path, index=False)

    calibration_errors: list[float] = []
    test_errors: list[float] = []
    for task, (_, error_dict, _) in zip(tasks, results, strict=True):
        _, pack_id, *_ = task
        proposed_errors = np.abs(np.asarray(error_dict.get("proposed", []), dtype=float))
        if pack_id % 2 == 0:
            calibration_errors.extend(proposed_errors.tolist())
        else:
            test_errors.extend(proposed_errors.tolist())
    radius = split_conformal_radius(np.asarray(calibration_errors), float(cfg["identification"]["confidence_level"]))
    coverage = empirical_coverage(np.asarray(test_errors), radius)
    conformal = {"radius_mv": radius, "test_coverage": coverage, "n_calibration": len(calibration_errors), "n_test": len(test_errors)}
    (output_dir / "conformal_calibration.json").write_text(json.dumps(conformal, indent=2), encoding="utf-8")

    if write_overleaf:
        plot_metric_boxplot(summary, "validation_voltage_rmse_mv", "Independent voltage RMSE (mV)", overleaf_dir / "figures" / "voltage_rmse_boxplot.pdf")
        plot_metric_boxplot(summary, "impedance_nrmse_pct", r"Impedance NRMSE (\%)", overleaf_dir / "figures" / "impedance_nrmse_boxplot.pdf")
        plot_chemistry_comparison(summary, "validation_voltage_rmse_mv", "Median voltage RMSE (mV)", overleaf_dir / "figures" / "chemistry_voltage_comparison.pdf")
        plot_duration_accuracy(summary, overleaf_dir / "figures" / "duration_accuracy.pdf")
        _write_tables(summary, statistical, overleaf_dir / "tables")
        _write_results_macros(summary, statistical, conformal, designs, overleaf_dir / "results_macros.tex")

    manifest = {
        "project": cfg["project"]["name"],
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_s": perf_counter() - t0,
        "config_path": str(config_path),
        "config_sha256": _hash_file(config_path),
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "n_scenarios": len(tasks),
        "n_raw_rows": len(raw),
        "designs": {
            name: {
                "duration_s": design.duration_s,
                "reliable_lowest_frequency_hz": design.reliable_lowest_frequency_hz,
                "score": design.score,
                "metadata": design.metadata,
            }
            for name, design in designs.items()
        },
        "truth_backend": cfg.get("truth_backend", {}),
        "bms_prior_section": cfg.get("bms_prior", {}).get("parameter_section"),
        "method_specs": _method_specs(cfg),
        "paired_design_controls": {
            "shared_initial_cell_states": True,
            "shared_validation_profile_by_cell": True,
            "shared_pack_current_for_equal_design_and_round": True,
        },
        "synthetic_truth_warning": cfg.get("synthetic_truth", {}).get("warning", ""),
        "write_overleaf": write_overleaf,
        "outputs": [str(raw_path), str(summary_path), str(statistical_path), str(rejected_path)],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "raw": raw_path,
        "summary": summary_path,
        "statistics": statistical_path,
        "manifest": manifest_path,
        "rejected": rejected_path,
        **({"macros": overleaf_dir / "results_macros.tex"} if write_overleaf else {}),
    }
