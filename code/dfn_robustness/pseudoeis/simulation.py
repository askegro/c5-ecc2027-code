from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from numpy.typing import NDArray

from .models import ECMTruthModel
from .scheduler import PairSchedule
from .signals import ExcitationDesign

FloatArray = NDArray[np.float64]


@dataclass
class CellMeasurement:
    cell_index: int
    pair_index: int
    orientation: int
    time_s: FloatArray
    true_current_a: FloatArray
    measured_current_a: FloatArray
    true_voltage_v: FloatArray
    measured_voltage_v: FloatArray
    true_soc: FloatArray
    estimated_soc: FloatArray
    temperature_c: FloatArray
    q: FloatArray
    segment_slices: tuple[slice, ...]


@dataclass
class DiagnosticWindowResult:
    measurements: list[CellMeasurement]
    pack_voltage_disturbance_v: FloatArray
    pack_power_disturbance_w: FloatArray
    pack_power_reference_w: FloatArray
    relative_rms_power_disturbance_pct: float
    relative_peak_power_disturbance_pct: float
    duration_s: float
    operational_power_records: list[dict]


def _jitter_binary_sequence(q: FloatArray, dt_s: float, jitter_std_s: float, rng: np.random.Generator) -> FloatArray:
    if jitter_std_s <= 0:
        return q.copy()
    edge = np.flatnonzero(np.diff(q) != 0) + 1
    out = q.copy()
    max_shift = max(1, int(np.ceil(4.0 * jitter_std_s / dt_s)))
    for idx in edge:
        shift = int(np.clip(np.rint(rng.normal(0.0, jitter_std_s) / dt_s), -max_shift, max_shift))
        new_idx = int(np.clip(idx + shift, 1, len(q) - 1))
        if new_idx > idx:
            out[idx:new_idx] = q[idx - 1]
        elif new_idx < idx:
            out[new_idx:idx] = q[idx]
    return out


def _segment_slices(design: ExcitationDesign) -> tuple[slice, ...]:
    slices: list[slice] = []
    start = 0
    for segment in design.segments:
        stop = start + len(segment.q)
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def simulate_diagnostic_window(
    pack: list[ECMTruthModel],
    schedule: PairSchedule,
    design: ExcitationDesign,
    pack_current_a: FloatArray,
    soc0: FloatArray,
    temperature0_c: FloatArray,
    soh: FloatArray,
    estimated_capacity_ah: FloatArray,
    voltage_noise_std_v: float,
    current_noise_std_a: float,
    soc_bias_std: float,
    switch_timing_jitter_std_s: float,
    switch_on_resistance_ohm: float,
    pack_power_reference_w: float,
    rng: np.random.Generator,
) -> DiagnosticWindowResult:
    n = len(design.q)
    if len(pack_current_a) != n:
        raise ValueError("Pack-current profile length must equal excitation length.")
    if len(estimated_capacity_ah) != len(pack) or np.any(np.asarray(estimated_capacity_ah) <= 0):
        raise ValueError("Estimated capacities must be positive and match the pack size.")
    q_actual = _jitter_binary_sequence(design.q, design.dt_s, switch_timing_jitter_std_s, rng)
    time = np.arange(n, dtype=float) * design.dt_s
    measurements: list[CellMeasurement] = []
    pair_disturbances: list[FloatArray] = []
    inserted_voltage = np.zeros(n)
    fixed_voltage = {}
    segment_slices = _segment_slices(design)

    for pair_idx, ((a, b), orientation) in enumerate(zip(schedule.pairs, schedule.orientations, strict=True)):
        q_pair = orientation * q_actual
        s_a = 0.5 * (1.0 + q_pair)
        s_b = 0.5 * (1.0 - q_pair)
        ia = s_a * pack_current_a
        ib = s_b * pack_current_a
        sim_a = pack[a].simulate(ia, design.dt_s, soc0[a], temperature0_c[a], soh[a])
        sim_b = pack[b].simulate(ib, design.dt_s, soc0[b], temperature0_c[b], soh[b])
        va = sim_a["voltage_v"]
        vb = sim_b["voltage_v"]
        inserted_voltage += s_a * va + s_b * vb
        for cell_idx in (a, b):
            full = pack[cell_idx].simulate(pack_current_a, design.dt_s, soc0[cell_idx], temperature0_c[cell_idx], soh[cell_idx])
            values = np.asarray(full["voltage_v"])
            if values.shape != (n,) or not np.isfinite(values).all():
                raise RuntimeError("Invalid fixed-insertion reference trajectory")
            fixed_voltage[cell_idx] = values
        disturbance = 0.5 * q_pair * (va - vb)
        pair_disturbances.append(disturbance)

        for cell_idx, current, sim, q_local in (
            (a, ia, sim_a, q_pair),
            (b, ib, sim_b, -q_pair),
        ):
            measured_current = current + rng.normal(0.0, current_noise_std_a, size=n)
            measured_voltage = sim["voltage_v"] + rng.normal(0.0, voltage_noise_std_v, size=n)
            z0_est = float(np.clip(soc0[cell_idx] + rng.normal(0.0, soc_bias_std), 0.0, 1.0))
            estimated_soc = np.empty(n, dtype=float)
            estimated_soc[0] = z0_est
            capacity_as = float(estimated_capacity_ah[cell_idx]) * 3600.0
            for k in range(n - 1):
                estimated_soc[k + 1] = np.clip(
                    estimated_soc[k] - design.dt_s * measured_current[k] / capacity_as,
                    0.0,
                    1.0,
                )
            measurements.append(
                CellMeasurement(
                    cell_index=cell_idx,
                    pair_index=pair_idx,
                    orientation=int(orientation),
                    time_s=time,
                    true_current_a=current,
                    measured_current_a=measured_current,
                    true_voltage_v=sim["voltage_v"],
                    measured_voltage_v=measured_voltage,
                    true_soc=sim["soc"],
                    estimated_soc=estimated_soc,
                    temperature_c=sim["temperature_c"],
                    q=q_local,
                    segment_slices=segment_slices,
                )
            )

    if pair_disturbances:
        pack_voltage_disturbance = np.sum(np.vstack(pair_disturbances), axis=0)
    else:
        pack_voltage_disturbance = np.zeros(n, dtype=float)
    pack_power_disturbance = pack_current_a * pack_voltage_disturbance
    reference = np.full(n, float(pack_power_reference_w))
    denom_rms = np.sqrt(np.mean(reference**2)) + 1e-12
    rel_rms = 100.0 * np.sqrt(np.mean(pack_power_disturbance**2)) / denom_rms
    rel_peak = 100.0 * np.max(np.abs(pack_power_disturbance)) / (np.max(np.abs(reference)) + 1e-12)
    operational = []
    windows = [("whole_record", slice(0,n))] + [(seg.name, sl) for seg,sl in zip(design.segments,segment_slices)]
    if schedule.pairs:
        for choices in itertools.product((0,1),repeat=len(schedule.pairs)):
            units = [pair[choice] for pair,choice in zip(schedule.pairs,choices)]
            reference_voltage = sum(fixed_voltage[i] for i in units)
            power_delta = pack_current_a * (inserted_voltage-reference_voltage)
            ref_power = pack_current_a * reference_voltage
            for name,sl in windows:
                values=power_delta[sl];ref=ref_power[sl]
                total=float(np.sqrt(np.mean(values**2)))
                denominator=float(np.sqrt(np.mean(ref**2)))
                operational.append(dict(reference="fixed_insertion",window=name,
                    pairs=[list(pair) for pair in schedule.pairs],orientations=list(schedule.orientations),
                    fixed_units=units,mean_power_w=float(np.mean(values)),
                    total_rms_power_w=total,ac_rms_power_w=float(np.std(values)),
                    peak_abs_power_w=float(np.max(np.abs(values))),
                    reference_rms_power_w=denominator,
                    relative_total_rms_power_pct=100*total/denominator if denominator>0 else None,
                    scope="active diagnostic pairs; ideal switches; all fixed choices; matched initial states"))
    return DiagnosticWindowResult(
        measurements=measurements,
        pack_voltage_disturbance_v=pack_voltage_disturbance,
        pack_power_disturbance_w=pack_power_disturbance,
        pack_power_reference_w=reference,
        relative_rms_power_disturbance_pct=float(rel_rms),
        relative_peak_power_disturbance_pct=float(rel_peak),
        duration_s=design.duration_s,
        operational_power_records=operational,
    )
