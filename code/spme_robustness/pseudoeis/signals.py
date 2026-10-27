from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import max_len_seq

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExcitationSegment:
    name: str
    q: FloatArray
    dt_s: float
    bit_period_s: float
    lfsr_order: int
    seed_index: int

    @property
    def duration_s(self) -> float:
        return len(self.q) * self.dt_s


@dataclass(frozen=True)
class ExcitationDesign:
    name: str
    q: FloatArray
    dt_s: float
    segments: tuple[ExcitationSegment, ...]
    target_frequencies_hz: FloatArray
    score: float
    metadata: dict

    @property
    def duration_s(self) -> float:
        return len(self.q) * self.dt_s

    @property
    def reliable_lowest_frequency_hz(self) -> float:
        longest = max(segment.duration_s for segment in self.segments)
        return 2.0 / longest


def mls_bits(order: int, seed: int, invert: bool = False, shift: int = 0) -> NDArray[np.int8]:
    if order < 2:
        raise ValueError("LFSR order must be at least 2.")
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=order, dtype=np.int8)
    if not np.any(state):
        state[0] = 1
    seq, _ = max_len_seq(order, state=state)
    seq = np.roll(seq.astype(np.int8), int(shift) % len(seq))
    q = 2 * seq - 1
    if invert:
        q = -q
    return q.astype(np.int8)


def expand_bits(bits: NDArray[np.int8], bit_period_s: float, dt_s: float, repeats: int = 1) -> FloatArray:
    if bit_period_s <= 0 or dt_s <= 0:
        raise ValueError("Periods must be positive.")
    samples_per_bit = int(round(bit_period_s / dt_s))
    if samples_per_bit < 1 or not np.isclose(samples_per_bit * dt_s, bit_period_s, rtol=0, atol=1e-10):
        raise ValueError("bit_period_s must be an integer multiple of the common sampling interval.")
    tiled = np.tile(bits, int(repeats))
    return np.repeat(tiled, samples_per_bit).astype(float)


def make_segment(
    name: str,
    order: int,
    bit_period_s: float,
    dt_s: float,
    repeats: int,
    seed: int,
    invert: bool = False,
    shift: int = 0,
) -> ExcitationSegment:
    bits = mls_bits(order=order, seed=seed, invert=invert, shift=shift)
    q = expand_bits(bits, bit_period_s=bit_period_s, dt_s=dt_s, repeats=repeats)
    return ExcitationSegment(
        name=name,
        q=q,
        dt_s=dt_s,
        bit_period_s=bit_period_s,
        lfsr_order=order,
        seed_index=seed,
    )


def concatenate_segments(name: str, segments: Iterable[ExcitationSegment], target_frequencies_hz: Iterable[float], score: float, metadata: dict | None = None) -> ExcitationDesign:
    segs = tuple(segments)
    if not segs:
        raise ValueError("At least one segment is required.")
    dt = segs[0].dt_s
    if any(not np.isclose(s.dt_s, dt) for s in segs):
        raise ValueError("All segments must use the same sample interval after expansion.")
    return ExcitationDesign(
        name=name,
        q=np.concatenate([s.q for s in segs]),
        dt_s=dt,
        segments=segs,
        target_frequencies_hz=np.asarray(list(target_frequencies_hz), dtype=float),
        score=float(score),
        metadata={} if metadata is None else dict(metadata),
    )


def candidate_segment_pool(
    band_name: str,
    band_cfg: dict,
    common_dt_s: float,
    n_candidates: int,
    base_seed: int,
    swap_fraction_low: float = 0.01,
    swap_fraction_high: float = 0.05,
) -> list[ExcitationSegment]:
    order = int(band_cfg["lfsr_order"])
    period = float(band_cfg["bit_period_s"])
    repeats = int(band_cfg.get("repeats", 1))
    candidates: list[ExcitationSegment] = []
    base_bits = mls_bits(order=order, seed=base_seed + 17 * order)
    samples_per_bit = int(round(period / common_dt_s))
    if samples_per_bit < 1 or not np.isclose(samples_per_bit * common_dt_s, period, atol=1e-10):
        raise ValueError("Band bit period must be an integer multiple of the common sample interval.")
    for idx in range(n_candidates):
        rng = np.random.default_rng(base_seed + 1009 * idx + 17 * order)
        bits = base_bits.copy()
        if idx > 0:
            positive = np.flatnonzero(bits > 0)
            negative = np.flatnonzero(bits < 0)
            swap_fraction = swap_fraction_low + (swap_fraction_high - swap_fraction_low) * idx / max(
                n_candidates - 1, 1
            )
            n_swaps = max(1, int(swap_fraction * len(bits)))
            pos_sel = rng.choice(positive, size=min(n_swaps, len(positive)), replace=False)
            neg_sel = rng.choice(negative, size=min(n_swaps, len(negative)), replace=False)
            bits[pos_sel] = -1
            bits[neg_sel] = 1
            bits = np.roll(bits, int(rng.integers(0, len(bits))))
            if idx % 2:
                bits = -bits
        q = np.repeat(np.tile(bits, repeats), samples_per_bit).astype(float)
        candidates.append(
            ExcitationSegment(
                name=band_name,
                q=q,
                dt_s=common_dt_s,
                bit_period_s=period,
                lfsr_order=order,
                seed_index=base_seed + 1009 * idx + 17 * order,
            )
        )
    return candidates


def spectrum_energy(q: FloatArray, dt_s: float, frequencies_hz: FloatArray) -> FloatArray:
    x = np.asarray(q, dtype=float) - np.mean(q)
    n = len(x)
    fft = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, dt_s)
    power = np.abs(fft) ** 2 / max(n, 1)
    return np.interp(frequencies_hz, f, power, left=0.0, right=0.0)


def spectral_coverage_score(q: FloatArray, dt_s: float, target_frequencies_hz: FloatArray) -> float:
    e = spectrum_energy(q, dt_s, target_frequencies_hz)
    positive = e[e > 1e-15]
    if len(positive) == 0:
        return -np.inf
    loge = np.log(positive)
    coverage = len(positive) / len(target_frequencies_hz)
    uniformity = -float(np.std(loge))
    floor = float(np.percentile(loge, 10))
    return 10.0 * coverage + uniformity + 0.1 * floor


def synthetic_drive_current(
    duration_s: float,
    dt_s: float,
    baseline_current_a: float,
    rng: np.random.Generator,
    minimum_current_a: float = 0.0,
) -> FloatArray:
    t = np.arange(0.0, duration_s, dt_s)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4)
    envelope = (
        0.18 * np.sin(2 * np.pi * t / 53.0 + phases[0])
        + 0.12 * np.sin(2 * np.pi * t / 17.0 + phases[1])
        + 0.08 * np.sin(2 * np.pi * t / 7.3 + phases[2])
        + 0.04 * np.sin(2 * np.pi * t / 2.1 + phases[3])
    )
    knot_dt = max(1, int(round(2.5 / dt_s)))
    knots = rng.normal(0.0, 0.12, size=int(np.ceil(len(t) / knot_dt)) + 1)
    random_component = np.interp(np.arange(len(t)), np.arange(len(knots)) * knot_dt, knots)
    current = baseline_current_a * (1.0 + envelope + random_component)
    if minimum_current_a > 0:
        current = np.maximum(current, minimum_current_a)
    return current.astype(float)


def validation_current_profile(duration_s: float, dt_s: float, amplitude_a: float, rng: np.random.Generator) -> FloatArray:
    t = np.arange(0.0, duration_s, dt_s)
    current = 0.25 * amplitude_a * np.sin(2 * np.pi * t / 19.0)
    current += 0.18 * amplitude_a * np.sin(2 * np.pi * t / 4.7 + 0.8)
    block = max(1, int(round(4.0 / dt_s)))
    levels = rng.choice([-0.55, -0.25, 0.0, 0.35, 0.70, 1.0], size=int(np.ceil(len(t) / block)) + 1)
    step = np.repeat(levels, block)[: len(t)] * amplitude_a
    alpha = np.exp(-dt_s / 0.25)
    smooth = np.empty_like(step)
    smooth[0] = step[0]
    for k in range(1, len(step)):
        smooth[k] = alpha * smooth[k - 1] + (1.0 - alpha) * step[k]
    return (current + smooth).astype(float)
