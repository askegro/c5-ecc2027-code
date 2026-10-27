from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

FloatArray = NDArray[np.float64]


class TruthBackend(Protocol):

    def capacity_ah(self, soh: float) -> float:
        return float(self.parameters.capacity_ah * soh)

    def dc_resistance_ohm(self, temperature_c: float, soh: float, soc: float) -> float:
        p = self.effective_parameters(temperature_c, soh, soc)
        return float(p.r0_ohm + np.sum(p.rc_resistances_ohm))

    def simulate(
        self,
        current_a: ArrayLike,
        dt_s: float,
        soc0: float,
        temperature_c: float,
        soh: float,
        initial_rc_voltage_v: ArrayLike | None = None,
    ) -> dict[str, FloatArray]:
        pass

    def impedance(self, frequencies_hz: ArrayLike, temperature_c: float, soh: float, soc: float) -> NDArray[np.complex128]:
        pass

    def ocv(self, soc: ArrayLike) -> FloatArray:
        pass

    def capacity_ah(self, soh: float) -> float:
        pass

    def dc_resistance_ohm(self, temperature_c: float, soh: float, soc: float) -> float:
        pass


@dataclass(frozen=True)
class ECMParameters:
    chemistry: str
    capacity_ah: float
    r0_ohm: float
    rc_resistances_ohm: FloatArray
    time_constants_s: FloatArray
    ocv_soc_points: FloatArray
    ocv_v_points: FloatArray
    temperature_coefficient_per_c: float = -0.02
    ageing_resistance_gain_at_80pct_soh: float = 0.6
    thermal_capacity_j_per_k: float = 4500.0
    thermal_conductance_w_per_k: float = 2.5
    entropic_heat_coefficient_v_per_k: float = 0.0

    def __post_init__(self) -> None:
        r = np.asarray(self.rc_resistances_ohm, dtype=float)
        tau = np.asarray(self.time_constants_s, dtype=float)
        soc = np.asarray(self.ocv_soc_points, dtype=float)
        ocv = np.asarray(self.ocv_v_points, dtype=float)
        if r.ndim != 1 or tau.ndim != 1 or len(r) != len(tau):
            raise ValueError("RC resistances and time constants must be one-dimensional and equally sized.")
        if np.any(r <= 0) or np.any(tau <= 0) or self.r0_ohm <= 0 or self.capacity_ah <= 0:
            raise ValueError("All ECM parameters must be positive.")
        if len(soc) != len(ocv) or len(soc) < 3 or np.any(np.diff(soc) <= 0):
            raise ValueError("OCV support points must have equal length and strictly increasing SOC.")
        object.__setattr__(self, "rc_resistances_ohm", r)
        object.__setattr__(self, "time_constants_s", tau)
        object.__setattr__(self, "ocv_soc_points", soc)
        object.__setattr__(self, "ocv_v_points", ocv)

    @property
    def n_rc(self) -> int:
        return len(self.rc_resistances_ohm)


@dataclass
class ECMTruthModel:

    parameters: ECMParameters
    ambient_temperature_c: float = 25.0

    def __post_init__(self) -> None:
        self._ocv_interp = PchipInterpolator(
            self.parameters.ocv_soc_points,
            self.parameters.ocv_v_points,
            extrapolate=True,
        )

    def ocv(self, soc: ArrayLike) -> FloatArray:
        z = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return np.asarray(self._ocv_interp(z), dtype=float)

    def docv_dsoc(self, soc: ArrayLike) -> FloatArray:
        z = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return np.asarray(self._ocv_interp.derivative()(z), dtype=float)

    def _resistance_factor(self, temperature_c: float, soh: float, soc: float) -> float:
        p = self.parameters
        temp_factor = np.exp(p.temperature_coefficient_per_c * (temperature_c - 25.0))
        degradation = max(0.0, (1.0 - soh) / 0.20)
        ageing_factor = 1.0 + p.ageing_resistance_gain_at_80pct_soh * degradation
        edge_soc = ((soc - 0.5) / 0.5) ** 4
        soc_factor = 1.0 + 0.18 * edge_soc
        return float(temp_factor * ageing_factor * soc_factor)

    def effective_parameters(self, temperature_c: float, soh: float, soc: float) -> ECMParameters:
        factor = self._resistance_factor(temperature_c, soh, soc)
        tau_factor = 1.0 + 0.45 * max(0.0, (25.0 - temperature_c) / 25.0)
        return replace(
            self.parameters,
            capacity_ah=self.parameters.capacity_ah * soh,
            r0_ohm=self.parameters.r0_ohm * factor,
            rc_resistances_ohm=self.parameters.rc_resistances_ohm * factor,
            time_constants_s=self.parameters.time_constants_s * tau_factor,
        )

    def impedance(self, frequencies_hz: ArrayLike, temperature_c: float, soh: float, soc: float) -> FloatArray:
        f = np.asarray(frequencies_hz, dtype=float)
        if np.any(f <= 0):
            raise ValueError("Frequencies must be positive.")
        p = self.effective_parameters(temperature_c, soh, soc)
        jw = 1j * 2.0 * np.pi * f[:, None]
        z = p.r0_ohm + np.sum(p.rc_resistances_ohm[None, :] / (1.0 + jw * p.time_constants_s[None, :]), axis=1)
        return np.asarray(z, dtype=complex)

    def capacity_ah(self, soh: float) -> float:
        return float(self.parameters.capacity_ah * soh)

    def dc_resistance_ohm(self, temperature_c: float, soh: float, soc: float) -> float:
        p = self.effective_parameters(temperature_c, soh, soc)
        return float(p.r0_ohm + np.sum(p.rc_resistances_ohm))

    def simulate(
        self,
        current_a: ArrayLike,
        dt_s: float,
        soc0: float,
        temperature_c: float,
        soh: float,
        initial_rc_voltage_v: ArrayLike | None = None,
    ) -> dict[str, FloatArray]:
        i = np.asarray(current_a, dtype=float)
        if i.ndim != 1 or len(i) < 2:
            raise ValueError("Current must be a one-dimensional array with at least two samples.")
        if dt_s <= 0:
            raise ValueError("Sampling interval must be positive.")
        p0 = self.effective_parameters(temperature_c, soh, soc0)
        n = len(i)
        z = np.empty(n, dtype=float)
        temp = np.empty(n, dtype=float)
        v = np.empty(n, dtype=float)
        heat = np.empty(n, dtype=float)
        rc = np.zeros((n, p0.n_rc), dtype=float)
        if initial_rc_voltage_v is not None:
            x0 = np.asarray(initial_rc_voltage_v, dtype=float)
            if x0.shape != (p0.n_rc,):
                raise ValueError("Initial RC voltage shape does not match the model order.")
            rc[0] = x0
        z[0] = float(np.clip(soc0, 0.0, 1.0))
        temp[0] = float(temperature_c)

        for k in range(n - 1):
            pk = self.effective_parameters(temp[k], soh, z[k])
            alpha = np.exp(-dt_s / pk.time_constants_s)
            rc[k + 1] = alpha * rc[k] + pk.rc_resistances_ohm * (1.0 - alpha) * i[k]
            capacity_as = pk.capacity_ah * 3600.0
            z[k + 1] = np.clip(z[k] - dt_s * i[k] / capacity_as, 0.0, 1.0)
            q_ohmic = i[k] ** 2 * pk.r0_ohm
            q_polarization = float(np.sum(rc[k] ** 2 / np.maximum(pk.rc_resistances_ohm, 1e-12)))
            q_entropic = i[k] * (temp[k] + 273.15) * pk.entropic_heat_coefficient_v_per_k
            heat[k] = q_ohmic + q_polarization + q_entropic
            dtemp = (
                heat[k] - pk.thermal_conductance_w_per_k * (temp[k] - self.ambient_temperature_c)
            ) / pk.thermal_capacity_j_per_k
            temp[k + 1] = temp[k] + dt_s * dtemp
            v[k] = self.ocv(z[k]) - pk.r0_ohm * i[k] - np.sum(rc[k])

        plast = self.effective_parameters(temp[-1], soh, z[-1])
        heat[-1] = i[-1] ** 2 * plast.r0_ohm + float(
            np.sum(rc[-1] ** 2 / np.maximum(plast.rc_resistances_ohm, 1e-12))
        )
        v[-1] = self.ocv(z[-1]) - plast.r0_ohm * i[-1] - np.sum(rc[-1])
        return {
            "voltage_v": v,
            "soc": z,
            "temperature_c": temp,
            "rc_voltage_v": rc,
            "heat_w": heat,
            "current_a": i,
        }


def _parameters_from_mapping(chemistry: str, data: dict) -> ECMParameters:
    return ECMParameters(
        chemistry=chemistry,
        capacity_ah=float(data["nominal_capacity_ah"]),
        r0_ohm=float(data["r0_ohm"]),
        rc_resistances_ohm=np.asarray(data["rc_resistances_ohm"], dtype=float),
        time_constants_s=np.asarray(data["time_constants_s"], dtype=float),
        ocv_soc_points=np.asarray(data["ocv_soc_points"], dtype=float),
        ocv_v_points=np.asarray(data["ocv_v_points"], dtype=float),
        temperature_coefficient_per_c=float(data.get("temperature_coefficient_per_c", -0.02)),
        ageing_resistance_gain_at_80pct_soh=float(data.get("ageing_resistance_gain_at_80pct_soh", 0.6)),
    )


def parameters_from_config(chemistry: str, cfg: dict) -> ECMParameters:
    return _parameters_from_mapping(chemistry, cfg["synthetic_truth"][chemistry])


def parameters_from_prior_config(chemistry: str, cfg: dict) -> ECMParameters:
    prior_cfg = cfg.get("bms_prior", {})
    section_name = prior_cfg.get("parameter_section")
    if section_name:
        section = cfg.get(str(section_name), {})
        if chemistry not in section:
            raise KeyError(f"BMS prior section {section_name!r} has no {chemistry!r} entry.")
        data = section[chemistry]
    elif chemistry in prior_cfg:
        data = prior_cfg[chemistry]
    else:
        data = cfg["synthetic_truth"][chemistry]
    return _parameters_from_mapping(chemistry, data)


def sample_heterogeneous_pack(
    chemistry: str,
    pack_size: int,
    cfg: dict,
    rng: np.random.Generator,
) -> list[ECMTruthModel]:
    base = parameters_from_config(chemistry, cfg)
    chem_cfg = cfg["synthetic_truth"][chemistry]
    cv = float(chem_cfg.get("parameter_cv", 0.10))
    capacity_cv = float(chem_cfg.get("capacity_cv", 0.025))
    sigma = np.sqrt(np.log1p(cv**2))
    cap_sigma = np.sqrt(np.log1p(capacity_cv**2))
    batch_factor = rng.lognormal(mean=-0.5 * (0.35 * sigma) ** 2, sigma=0.35 * sigma)
    models: list[ECMTruthModel] = []
    for _ in range(pack_size):
        independent = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=1 + 2 * base.n_rc)
        cap_factor = rng.lognormal(mean=-0.5 * cap_sigma**2, sigma=cap_sigma)
        r0 = base.r0_ohm * batch_factor * independent[0]
        r = base.rc_resistances_ohm * batch_factor * independent[1 : 1 + base.n_rc]
        tau = base.time_constants_s * independent[1 + base.n_rc :]
        p = replace(base, capacity_ah=base.capacity_ah * cap_factor, r0_ohm=r0, rc_resistances_ohm=r, time_constants_s=tau)
        models.append(ECMTruthModel(p))
    return models


def ecm_impedance(
    frequencies_hz: ArrayLike,
    r0_ohm: float,
    rc_resistances_ohm: Sequence[float],
    time_constants_s: Sequence[float],
    inductance_h: float = 0.0,
) -> FloatArray:
    f = np.asarray(frequencies_hz, dtype=float)
    r = np.asarray(rc_resistances_ohm, dtype=float)
    tau = np.asarray(time_constants_s, dtype=float)
    jw = 1j * 2.0 * np.pi * f[:, None]
    z = r0_ohm + np.sum(r[None, :] / (1.0 + jw * tau[None, :]), axis=1)
    if inductance_h:
        z = z + 1j * 2.0 * np.pi * f * inductance_h
    return np.asarray(z, dtype=complex)


def simulate_identification_ecm(
    current_a: ArrayLike,
    dt_s: float,
    r0_ohm: float,
    rc_resistances_ohm: Sequence[float],
    time_constants_s: Sequence[float],
    ocv_v: float = 0.0,
    initial_rc_voltage_v: ArrayLike | None = None,
    drift_coefficients: Sequence[float] | None = None,
) -> FloatArray:
    i = np.asarray(current_a, dtype=float)
    r = np.asarray(rc_resistances_ohm, dtype=float)
    tau = np.asarray(time_constants_s, dtype=float)
    n_rc = len(r)
    x = np.zeros(n_rc, dtype=float) if initial_rc_voltage_v is None else np.asarray(initial_rc_voltage_v, dtype=float).copy()
    if x.shape != (n_rc,):
        raise ValueError("Initial RC voltage shape mismatch.")
    out = np.empty_like(i)
    alpha = np.exp(-dt_s / tau)
    t_norm = np.linspace(-1.0, 1.0, len(i))
    drift = np.zeros_like(i)
    if drift_coefficients is not None:
        for power, coefficient in enumerate(drift_coefficients):
            drift += float(coefficient) * t_norm**power
    for k, ik in enumerate(i):
        out[k] = ocv_v + drift[k] - r0_ohm * ik - np.sum(x)
        x = alpha * x + r * (1.0 - alpha) * ik
    return out
