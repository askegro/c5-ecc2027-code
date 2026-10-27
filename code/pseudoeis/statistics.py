from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    reference_method: str
    comparison_method: str
    n_pairs: int
    median_reference: float
    median_comparison: float
    median_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float


def bootstrap_ci(values: np.ndarray, confidence: float, n_bootstrap: int, rng: np.random.Generator, statistic=np.median) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    samples = rng.choice(x, size=(n_bootstrap, len(x)), replace=True)
    stats = np.apply_along_axis(statistic, 1, samples)
    alpha = 1.0 - confidence
    return float(np.quantile(stats, alpha / 2.0)), float(np.quantile(stats, 1.0 - alpha / 2.0))


def paired_method_comparisons(
    results: pd.DataFrame,
    metrics: list[str],
    reference_method: str,
    comparison_methods: list[str],
    id_columns: list[str],
    confidence: float,
    n_bootstrap: int,
    multiplicity_method: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        subset = results[id_columns + ["method", metric]].dropna()
        pivot = subset.pivot_table(index=id_columns, columns="method", values=metric, aggfunc="mean")
        for method in comparison_methods:
            if reference_method not in pivot or method not in pivot:
                continue
            paired = pivot[[reference_method, method]].dropna()
            if len(paired) == 0:
                continue
            ref = paired[reference_method].to_numpy(float)
            comp = paired[method].to_numpy(float)
            diff = ref - comp
            lo, hi = bootstrap_ci(diff, confidence, n_bootstrap, rng)
            try:
                p_value = float(wilcoxon(diff, zero_method="zsplit", alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "metric": metric,
                    "reference_method": reference_method,
                    "comparison_method": method,
                    "n_pairs": len(paired),
                    "median_reference": float(np.median(ref)),
                    "median_comparison": float(np.median(comp)),
                    "median_difference": float(np.median(diff)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_value,
                }
            )
    if not rows:
        return pd.DataFrame(columns=[f.name for f in PairedComparison.__dataclass_fields__.values()])
    out = pd.DataFrame(rows)
    _, adjusted, _, _ = multipletests(out["p_value"].to_numpy(), method=multiplicity_method)
    out["p_adjusted"] = adjusted
    return out


def split_conformal_radius(calibration_errors: np.ndarray, confidence: float) -> float:
    errors = np.sort(np.asarray(calibration_errors, dtype=float))
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return np.nan
    rank = int(np.ceil((len(errors) + 1) * confidence))
    rank = int(np.clip(rank, 1, len(errors)))
    return float(errors[rank - 1])


def empirical_coverage(errors: np.ndarray, radius: float) -> float:
    x = np.asarray(errors, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or not np.isfinite(radius):
        return np.nan
    return float(np.mean(np.abs(x) <= radius))
