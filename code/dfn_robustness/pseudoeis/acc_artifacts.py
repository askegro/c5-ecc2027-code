from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccArtifactPaths:
    macros: Path
    information_figure: Path
    power_figure: Path
    primary_results: Path
    figure_source_data: Path
    manifest: Path


def _finite_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else float("nan")


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _fmt_number(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return r"\TBD{result}"
    return f"{value:.{digits}f}"


def _si(value: float, unit: str, digits: int = 2) -> str:
    return rf"\SI{{{_fmt_number(value, digits)}}}{{{unit}}}" if np.isfinite(value) else r"\TBD{result}"


def _duration_tex(seconds: float) -> str:
    if not np.isfinite(seconds):
        return r"\TBD{time}"
    if seconds >= 120.0:
        return _si(seconds / 60.0, r"\minute", 2)
    return _si(seconds, r"\second", 2)


def _method_frame(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    return summary.loc[summary["method"].astype(str) == method].copy()


def _design_metadata(run_manifest: dict[str, Any], design: str) -> dict[str, Any]:
    return dict(run_manifest.get("designs", {}).get(design, {}))


def _worst_logdet(run_manifest: dict[str, Any], design: str) -> float:
    metadata = _design_metadata(run_manifest, design).get("metadata", {})
    try:
        return float(metadata.get("worst_chemistry_logdet", np.nan))
    except (TypeError, ValueError):
        return float("nan")


def _low_rate_duration(cfg: dict[str, Any]) -> float:
    low = cfg["excitation"]["proposed"]["low_band"]
    return float((2 ** int(low["lfsr_order"]) - 1) * float(low["bit_period_s"]) * int(low.get("repeats", 1)))


def _paired_reduction(reference: pd.DataFrame, comparison: pd.DataFrame, metric: str) -> float:
    key = ["chemistry", "pack_id", "operating_point_id", "noise_id", "pack_size"]
    left = reference[key + [metric]].rename(columns={metric: "reference"})
    right = comparison[key + [metric]].rename(columns={metric: "comparison"})
    paired = left.merge(right, on=key, how="inner").dropna()
    if paired.empty:
        return float("nan")
    denominator = paired["comparison"].replace(0.0, np.nan)
    reductions = 100.0 * (paired["comparison"] - paired["reference"]) / denominator
    reductions = reductions[np.isfinite(reductions)]
    return float(reductions.median()) if len(reductions) else float("nan")


def _stat_row(statistics: pd.DataFrame, metric: str, comparison_method: str) -> dict[str, float] | None:
    match = statistics.loc[
        (statistics["metric"] == metric) & (statistics["comparison_method"] == comparison_method)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "median_difference": float(row["median_difference"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "p_adjusted": float(row["p_adjusted"]),
    }


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return r"\TBD{p}"
    if p >= 0.995:
        return "$p=1.00$"
    if p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$"


def _write_macros(
    summary: pd.DataFrame,
    cfg: dict[str, Any],
    run_manifest: dict[str, Any],
    output: Path,
    statistics: pd.DataFrame | None = None,
) -> dict[str, str]:
    proposed = _method_frame(summary, "proposed")
    coverage = _method_frame(summary, "coverage_sequence")
    short = _method_frame(summary, "short_sequence")
    random_pairing = _method_frame(summary, "random_pairing")

    unique_packs = proposed[["chemistry", "pack_id"]].drop_duplicates() if not proposed.empty else pd.DataFrame()
    proposed_design = _design_metadata(run_manifest, "proposed")
    lowest_hz = float(proposed_design.get("reliable_lowest_frequency_hz", np.nan))
    fim_gain = _worst_logdet(run_manifest, "proposed") - _worst_logdet(run_manifest, "spectral_baseline")
    truth_name = str(cfg.get("reporting", {}).get("truth_model_name", "validated truth model"))

    macros: dict[str, str] = {
        "NumPackRealizations": str(len(unique_packs)),
        "PackSize": _fmt_number(_finite_median(proposed, "pack_size"), 0),
        "TruthModelName": _tex_escape(truth_name),
        "LowRateDuration": _duration_tex(_low_rate_duration(cfg)),
        "LowestFrequency": _si(1e3 * lowest_hz, r"\milli\hertz", 2),
        "ProposedDuration": _duration_tex(float(proposed_design.get("duration_s", np.nan))),
        "FIMGain": _fmt_number(fim_gain, 2),
        "ProposedVoltageRMSE": _si(_finite_median(proposed, "validation_voltage_rmse_mv"), r"\milli\volt", 2),
        "CoverageVoltageRMSE": _si(_finite_median(coverage, "validation_voltage_rmse_mv"), r"\milli\volt", 2),
        "ShortVoltageRMSE": _si(_finite_median(short, "validation_voltage_rmse_mv"), r"\milli\volt", 2),
        "ProposedImpedanceNRMSE": _si(_finite_median(proposed, "impedance_nrmse_pct"), r"\percent", 2),
        "CoverageImpedanceNRMSE": _si(_finite_median(coverage, "impedance_nrmse_pct"), r"\percent", 2),
        "ShortImpedanceNRMSE": _si(_finite_median(short, "impedance_nrmse_pct"), r"\percent", 2),
        "ProposedPowerRMS": _si(_finite_median(proposed, "pack_median_rms_power_disturbance_pct"), r"\percent", 3),
        "RandomPowerRMS": _si(_finite_median(random_pairing, "pack_median_rms_power_disturbance_pct"), r"\percent", 3),
        "PairingReduction": _si(_paired_reduction(proposed, random_pairing, "pack_median_rms_power_disturbance_pct"), r"\percent", 1),
        "CoverageRate": _si(100.0 * _finite_median(proposed, "pack_coverage_fraction"), r"\percent", 1),
        "ProposedRefreshTime": _duration_tex(_finite_median(proposed, "diagnostic_refresh_time_s")),
        "CoverageRefreshTime": _duration_tex(_finite_median(coverage, "diagnostic_refresh_time_s")),
        "ShortRefreshTime": _duration_tex(_finite_median(short, "diagnostic_refresh_time_s")),
        "SelectedOrder": _fmt_number(_finite_median(proposed, "selected_order"), 0),
        "NMCVoltageRMSE": _si(_finite_median(proposed[proposed["chemistry"] == "NMC"], "validation_voltage_rmse_mv"), r"\milli\volt", 2),
        "LFPVoltageRMSE": _si(_finite_median(proposed[proposed["chemistry"] == "LFP"], "validation_voltage_rmse_mv"), r"\milli\volt", 2),
        "FitTime": _si(_finite_median(proposed, "fit_solve_time_s"), r"\second", 2),
        "ScheduleTime": _si(_finite_median(proposed, "pack_median_scheduling_time_ms"), r"\milli\second", 2),
    }

    if statistics is not None:
        power_vs_random = _stat_row(statistics, "pack_median_rms_power_disturbance_pct", "random_pairing")
        voltage_vs_random = _stat_row(statistics, "validation_voltage_rmse_mv", "random_pairing")
        voltage_vs_coverage = _stat_row(statistics, "validation_voltage_rmse_mv", "coverage_sequence")
        voltage_vs_short = _stat_row(statistics, "validation_voltage_rmse_mv", "short_sequence")
        power_vs_no_orientation = _stat_row(statistics, "pack_median_rms_power_disturbance_pct", "no_orientation")
        macros["PowerVsRandomSig"] = _fmt_p(power_vs_random["p_adjusted"]) if power_vs_random else r"\TBD{p}"
        macros["VoltageVsRandomSig"] = _fmt_p(voltage_vs_random["p_adjusted"]) if voltage_vs_random else r"\TBD{p}"
        macros["VoltageVsCoverageDiff"] = (
            _si(voltage_vs_coverage["median_difference"], r"\milli\volt", 3) if voltage_vs_coverage else r"\TBD{mV}"
        )
        macros["VoltageVsCoverageSig"] = _fmt_p(voltage_vs_coverage["p_adjusted"]) if voltage_vs_coverage else r"\TBD{p}"
        macros["VoltageVsShortDiff"] = (
            _si(voltage_vs_short["median_difference"], r"\milli\volt", 3) if voltage_vs_short else r"\TBD{mV}"
        )
        macros["VoltageVsShortSig"] = _fmt_p(voltage_vs_short["p_adjusted"]) if voltage_vs_short else r"\TBD{p}"
        macros["PowerVsNoOrientationDiff"] = (
            _si(power_vs_no_orientation["median_difference"], r"\percent", 3) if power_vs_no_orientation else r"\TBD{pct}"
        )
        macros["PowerVsNoOrientationSig"] = (
            _fmt_p(power_vs_no_orientation["p_adjusted"]) if power_vs_no_orientation else r"\TBD{p}"
        )
    else:
        for name in ("PowerVsRandomSig", "VoltageVsRandomSig", "VoltageVsCoverageDiff",
                     "VoltageVsCoverageSig", "VoltageVsShortDiff", "VoltageVsShortSig",
                     "PowerVsNoOrientationDiff", "PowerVsNoOrientationSig"):
            macros[name] = r"\TBD{stat}"

    if "operating_point_id" in proposed.columns and not proposed.empty:
        op_coverage = proposed.groupby("operating_point_id")["pack_coverage_fraction"].median()
        if len(op_coverage) > 1:
            worst_op_id = int(op_coverage.idxmin())
            worst_coverage = float(op_coverage.loc[worst_op_id])
            other_coverage = op_coverage.drop(index=worst_op_id)
            best_coverage = float(other_coverage.max())
            op_points = cfg.get("study", {}).get("operating_points", [])
            worst_op = op_points[worst_op_id] if 0 <= worst_op_id < len(op_points) else {}
            macros["WorstOpCoverageRate"] = _si(100.0 * worst_coverage, r"\percent", 1)
            macros["OtherOpCoverageRate"] = _si(100.0 * best_coverage, r"\percent", 1)
            macros["WorstOpTemperature"] = (
                _si(float(worst_op["temperature_c"]), r"\celsius", 0) if "temperature_c" in worst_op else r"\TBD{temp}"
            )
            macros["WorstOpSOC"] = (
                _si(100.0 * float(worst_op["soc"]), r"\percent", 0) if "soc" in worst_op else r"\TBD{SOC}"
            )
        else:
            for name in ("WorstOpCoverageRate", "OtherOpCoverageRate", "WorstOpTemperature", "WorstOpSOC"):
                macros[name] = r"\TBD{result}"
    else:
        for name in ("WorstOpCoverageRate", "OtherOpCoverageRate", "WorstOpTemperature", "WorstOpSOC"):
            macros[name] = r"\TBD{result}"

    lines = ["% Auto-generated by run_acc_study.py. Do not edit manually."]
    for name, rendered in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{rendered}}}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return macros


def _save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_information_and_prediction(
    summary: pd.DataFrame,
    run_manifest: dict[str, Any],
    output: Path,
) -> pd.DataFrame:
    design_labels = ["Proposed", "Coverage"]
    design_names = ["proposed", "spectral_baseline"]
    logdets = [_worst_logdet(run_manifest, name) for name in design_names]

    prediction_methods = ["proposed", "coverage_sequence", "short_sequence"]
    prediction_labels = ["Proposed", "Coverage", "Short"]
    prediction_data = [
        pd.to_numeric(_method_frame(summary, method)["validation_voltage_rmse_mv"], errors="coerce").dropna().to_numpy()
        for method in prediction_methods
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    axes[0].bar(design_labels, logdets)
    axes[0].set_ylabel("Worst-scenario log det(F)")
    axes[0].set_title("Offline information design")
    axes[0].grid(axis="y", alpha=0.25)

    valid_box_data = [values if len(values) else np.asarray([np.nan]) for values in prediction_data]
    axes[1].boxplot(valid_box_data, tick_labels=prediction_labels, showfliers=False)
    axes[1].set_ylabel("Held-out voltage RMSE (mV)")
    axes[1].set_title("Independent prediction")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output)

    rows: list[dict[str, Any]] = []
    for label, name, value in zip(design_labels, design_names, logdets, strict=True):
        rows.append({"panel": "information", "label": label, "method": name, "value": value})
    for label, name, values in zip(prediction_labels, prediction_methods, prediction_data, strict=True):
        for value in values:
            rows.append({"panel": "prediction", "label": label, "method": name, "value": float(value)})
    return pd.DataFrame(rows)


def _plot_power_transparency(summary: pd.DataFrame, output: Path) -> pd.DataFrame:
    methods = ["proposed", "no_orientation", "random_pairing"]
    labels = ["Proposed", "No orientation", "Random pairs"]
    rms_data = [
        pd.to_numeric(_method_frame(summary, method)["pack_median_rms_power_disturbance_pct"], errors="coerce").dropna().to_numpy()
        for method in methods
    ]
    peak_data = [
        pd.to_numeric(_method_frame(summary, method)["pack_max_peak_power_disturbance_pct"], errors="coerce").dropna().to_numpy()
        for method in methods
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    axes[0].boxplot([x if len(x) else np.asarray([np.nan]) for x in rms_data], tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("RMS power disturbance (%)")
    axes[0].set_title("RMS transparency")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot([x if len(x) else np.asarray([np.nan]) for x in peak_data], tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Peak power disturbance (%)")
    axes[1].set_title("Peak transparency")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output)

    rows: list[dict[str, Any]] = []
    for label, method, rms_values, peak_values in zip(labels, methods, rms_data, peak_data, strict=True):
        rows.extend({"metric": "rms", "label": label, "method": method, "value": float(v)} for v in rms_values)
        rows.extend({"metric": "peak", "label": label, "method": method, "value": float(v)} for v in peak_values)
    return pd.DataFrame(rows)


def _write_primary_results(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        "validation_voltage_rmse_mv",
        "impedance_nrmse_pct",
        "pack_median_rms_power_disturbance_pct",
        "pack_max_peak_power_disturbance_pct",
        "diagnostic_refresh_time_s",
        "pack_coverage_fraction",
        "selected_order",
        "fit_solve_time_s",
        "pack_median_scheduling_time_ms",
    ]
    available = [metric for metric in metrics if metric in summary]
    table = summary.groupby(["chemistry", "method"], as_index=False)[available].median(numeric_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


def write_acc_artifacts(
    summary_path: str | Path,
    config: dict[str, Any],
    run_manifest_path: str | Path,
    overleaf_dir: str | Path,
    output_dir: str | Path,
    statistics_path: str | Path | None = None,
) -> AccArtifactPaths:
    summary_path = Path(summary_path)
    run_manifest_path = Path(run_manifest_path)
    overleaf_dir = Path(overleaf_dir)
    output_dir = Path(output_dir)
    summary = pd.read_csv(summary_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    statistics = pd.read_csv(Path(statistics_path)) if statistics_path is not None else None

    macros_path = overleaf_dir / "results_macros.tex"
    info_figure = overleaf_dir / "figures" / "information_and_prediction.pdf"
    power_figure = overleaf_dir / "figures" / "power_transparency.pdf"
    primary_results = output_dir / "acc_primary_results.csv"
    figure_source_data = output_dir / "acc_figure_source_data.csv"
    artifact_manifest = output_dir / "acc_artifact_manifest.json"

    macros = _write_macros(summary, config, run_manifest, macros_path, statistics)
    info_source = _plot_information_and_prediction(summary, run_manifest, info_figure)
    power_source = _plot_power_transparency(summary, power_figure)
    source_data = pd.concat([info_source, power_source], ignore_index=True, sort=False)
    source_data.to_csv(figure_source_data, index=False)
    _write_primary_results(summary, primary_results)

    payload = {
        "summary_source": str(summary_path.resolve()),
        "run_manifest_source": str(run_manifest_path.resolve()),
        "macros": macros,
        "generated_files": [
            str(macros_path.resolve()),
            str(info_figure.resolve()),
            str(power_figure.resolve()),
            str(primary_results.resolve()),
            str(figure_source_data.resolve()),
        ],
    }
    artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return AccArtifactPaths(
        macros=macros_path,
        information_figure=info_figure,
        power_figure=power_figure,
        primary_results=primary_results,
        figure_source_data=figure_source_data,
        manifest=artifact_manifest,
    )
