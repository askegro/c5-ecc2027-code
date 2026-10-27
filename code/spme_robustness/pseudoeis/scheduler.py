from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class CellDiagnosticState:
    index: int
    soc: float
    temperature_c: float
    soh: float
    ocv_v: float
    impedance_ohm: ComplexArray
    impedance_std_ohm: FloatArray
    priority: float


@dataclass(frozen=True)
class PairSchedule:
    pairs: tuple[tuple[int, int], ...]
    orientations: tuple[int, ...]
    total_cost: float
    method: str


def _pair_cost(a: CellDiagnosticState, b: CellDiagnosticState, cfg: dict) -> float:
    w = cfg["pairing"]["weights"]
    arch = cfg["architecture"]
    z_scale = np.maximum(0.5 * (np.abs(a.impedance_ohm) + np.abs(b.impedance_ohm)), 1e-6)
    uncertainty = float(cfg["pairing"].get("uncertainty_inflation", 1.96)) * (
        a.impedance_std_ohm + b.impedance_std_ohm
    )
    impedance_mismatch = np.sqrt(np.mean((np.abs(a.impedance_ohm - b.impedance_ohm) / z_scale) ** 2))
    uncertainty_term = np.sqrt(np.mean((uncertainty / z_scale) ** 2))
    cost = (
        float(w["ocv"]) * abs(a.ocv_v - b.ocv_v) / 0.05
        + float(w["impedance"]) * impedance_mismatch
        + float(w["soc"]) * abs(a.soc - b.soc) / max(float(arch["admissible_soc_difference"]), 1e-9)
        + float(w["temperature"]) * abs(a.temperature_c - b.temperature_c) / max(float(arch["admissible_temperature_difference_c"]), 1e-9)
        + float(w["soh"]) * abs(a.soh - b.soh) / 0.10
        + float(w["uncertainty"]) * uncertainty_term
        - float(w.get("priority", 0.0)) * 0.5 * (a.priority + b.priority)
    )
    return float(cost)


def _eligible(a: CellDiagnosticState, b: CellDiagnosticState, cfg: dict) -> bool:
    arch = cfg["architecture"]
    return (
        abs(a.soc - b.soc) <= float(arch["admissible_soc_difference"])
        and abs(a.temperature_c - b.temperature_c) <= float(arch["admissible_temperature_difference_c"])
    )


def schedule_pairs(
    states: list[CellDiagnosticState],
    cfg: dict,
    method: str,
    rng: np.random.Generator,
) -> PairSchedule:
    max_pairs = min(int(cfg["architecture"]["max_simultaneous_pairs"]), len(states) // 2)
    if max_pairs < 1:
        return PairSchedule((), (), 0.0, method)

    n_pool = min(len(states), max(2 * max_pairs, 4 * max_pairs))
    pool = sorted(states, key=lambda s: s.priority, reverse=True)[:n_pool]

    if method == "random":
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        pairs: list[tuple[int, int]] = []
        total = 0.0
        while len(shuffled) >= 2 and len(pairs) < max_pairs:
            a = shuffled.pop()
            feasible_idx = next((j for j, b in enumerate(shuffled) if _eligible(a, b, cfg)), None)
            if feasible_idx is None:
                continue
            b = shuffled.pop(feasible_idx)
            pairs.append((a.index, b.index))
            total += _pair_cost(a, b, cfg)
    elif method == "mismatch_aware":
        graph = nx.Graph()
        lookup = {s.index: s for s in pool}
        graph.add_nodes_from(lookup)
        for i, a in enumerate(pool):
            for b in pool[i + 1 :]:
                if _eligible(a, b, cfg):
                    graph.add_edge(a.index, b.index, weight=_pair_cost(a, b, cfg))
        matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
        if len(matching) == 0 and len(pool) >= 2:
            for i, a in enumerate(pool):
                for b in pool[i + 1 :]:
                    graph.add_edge(a.index, b.index, weight=_pair_cost(a, b, cfg) + 1e3)
            matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
        ranked_edges = sorted(
            [(min(a, b), max(a, b), graph[a][b]["weight"]) for a, b in matching],
            key=lambda item: item[2],
        )[:max_pairs]
        pairs = [(a, b) for a, b, _ in ranked_edges]
        total = float(sum(weight for _, _, weight in ranked_edges))
    else:
        raise ValueError(f"Unsupported pairing method: {method}")

    state_lookup = {s.index: s for s in states}
    orientations = orient_pairs(pairs, state_lookup, cfg)
    return PairSchedule(tuple(pairs), tuple(orientations), total, method)


def _orientation_objective(
    orientations: tuple[int, ...],
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    states: dict[int, CellDiagnosticState],
    excitation_current_a: float,
) -> float:
    if not pairs:
        return 0.0
    aggregate = np.zeros_like(states[pairs[0][0]].impedance_ohm, dtype=complex)
    for rho, (a, b) in zip(orientations, pairs, strict=True):
        sa, sb = states[a], states[b]
        mismatch = (sa.ocv_v - sb.ocv_v) + excitation_current_a * (sa.impedance_ohm - sb.impedance_ohm)
        aggregate += float(rho) * mismatch
    return float(np.mean(np.abs(aggregate) ** 2))


def orient_pairs(
    pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    states: dict[int, CellDiagnosticState],
    cfg: dict,
) -> tuple[int, ...]:
    pairs = tuple(pairs)
    if not pairs:
        return ()
    excitation_current_a = abs(float(cfg.get("pairing", {}).get(
        "orientation_reference_current_a",
        0.5 * float(cfg.get("study", {}).get("baseline_current_a", 40.0)),
    )))
    exact_limit = int(cfg.get("pairing", {}).get("orientation_exact_pair_limit", 12))
    if len(pairs) <= exact_limit:
        best: tuple[int, ...] | None = None
        best_value = float("inf")
        for mask in range(1 << max(0, len(pairs) - 1)):
            candidate = (1,) + tuple(1 if (mask >> bit) & 1 else -1 for bit in range(len(pairs) - 1))
            value = _orientation_objective(candidate, pairs, states, excitation_current_a)
            if value < best_value:
                best_value = value
                best = candidate
        assert best is not None
        return best

    aggregate = np.zeros_like(states[pairs[0][0]].impedance_ohm, dtype=complex)
    orientations: list[int] = []
    for a, b in pairs:
        sa, sb = states[a], states[b]
        mismatch = (sa.ocv_v - sb.ocv_v) + excitation_current_a * (sa.impedance_ohm - sb.impedance_ohm)
        plus = float(np.mean(np.abs(aggregate + mismatch) ** 2))
        minus = float(np.mean(np.abs(aggregate - mismatch) ** 2))
        orientation = 1 if plus <= minus else -1
        aggregate += orientation * mismatch
        orientations.append(orientation)
    return tuple(orientations)
