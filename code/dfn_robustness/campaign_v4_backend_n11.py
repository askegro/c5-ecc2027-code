from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import pickle
import numpy as np
import pybamm
from scipy.interpolate import PchipInterpolator

_PARAMETER_SETS = {"NMC": "Chen2020", "LFP": "Prada2013"}
_TARGET_CAPACITY_AH = 60.0
_OCV_CACHE: dict[str, PchipInterpolator] = {}
_OCV_CACHE_DIR = Path(__file__).resolve().parent / ".pybamm_ocv_cache_dfn"


def _base_parameter_values(chemistry: str) -> pybamm.ParameterValues:
    name = _PARAMETER_SETS[chemistry]
    params = pybamm.ParameterValues(name)
    base_capacity = float(params["Nominal cell capacity [A.h]"])
    scale = _TARGET_CAPACITY_AH / base_capacity
    params["Number of electrodes connected in parallel to make a cell"] = scale
    return params


def _build_ocv_curve(chemistry: str, params: pybamm.ParameterValues) -> PchipInterpolator:
    if chemistry in _OCV_CACHE:
        return _OCV_CACHE[chemistry]
    cache_file = _OCV_CACHE_DIR / f"{chemistry}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as handle:
                interp = pickle.load(handle)
            _OCV_CACHE[chemistry] = interp
            return interp
        except Exception:
            pass
    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    p = params.copy()
    p["Ambient temperature [K]"] = 298.15
    p["Initial temperature [K]"] = 298.15
    p["Current function [A]"] = _TARGET_CAPACITY_AH / 20.0
    dt_s = 30.0
    n = int(round(21.5 * 3600.0 / dt_s)) + 1
    t_eval = np.arange(n) * dt_s
    sim = pybamm.Simulation(model, parameter_values=p, solver=pybamm.IDAKLUSolver())
    sol = sim.solve(t_eval=t_eval, initial_soc=1.0)
    voltage = np.asarray(sol["Terminal voltage [V]"].data, dtype=float)
    discharge_ah = np.asarray(sol["Discharge capacity [A.h]"].data, dtype=float)
    soc = np.clip(1.0 - discharge_ah / _TARGET_CAPACITY_AH, 0.0, 1.0)
    order = np.argsort(soc)
    soc_sorted, voltage_sorted = soc[order], voltage[order]
    keep = np.concatenate([[True], np.diff(soc_sorted) > 1e-6])
    interp = PchipInterpolator(soc_sorted[keep], voltage_sorted[keep], extrapolate=True)
    _OCV_CACHE[chemistry] = interp
    try:
        _OCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_file, "wb") as handle:
            pickle.dump(interp, handle)
        os.replace(tmp_file, cache_file)
    except Exception:
        pass
    return interp


def _zero_order_hold_interpolant(t_eval: np.ndarray, values: np.ndarray) -> "pybamm.Interpolant":
    n = len(t_eval)
    if n < 2:
        return pybamm.Interpolant(t_eval, values, pybamm.t, interpolator="linear")
    dt = float(t_eval[1] - t_eval[0])
    eps = dt * 1e-3
    x = np.empty(2 * n - 1, dtype=float)
    y = np.empty(2 * n - 1, dtype=float)
    x[0::2] = t_eval
    x[1::2] = t_eval[1:] - eps
    y[0::2] = values
    y[1::2] = values[:-1]
    return pybamm.Interpolant(x, y, pybamm.t, interpolator="linear")


def _classify_fast_regions(
    current: np.ndarray,
    dt_s: float,
    window_s: float = 2.0,
    change_frac: float = 0.02,
    density_threshold: float = 0.05,
    min_run_s: float = 1.0,
) -> list[tuple[int, int, bool]]:
    n = len(current)
    if n < 3:
        return [(0, n, True)]
    amp = max(float(np.max(np.abs(current))), 1e-9)
    threshold = change_frac * amp
    changed = np.abs(np.diff(current)) > threshold
    window = max(1, int(round(window_s / dt_s)))
    kernel = np.ones(window) / window
    density = np.convolve(changed.astype(float), kernel, mode="same")
    density = np.concatenate([density, density[-1:]])
    fast_mask = density > density_threshold

    runs: list[tuple[int, int, bool]] = []
    start = 0
    state = bool(fast_mask[0])
    for k in range(1, n):
        if bool(fast_mask[k]) != state:
            runs.append((start, k, state))
            start = k
            state = bool(fast_mask[k])
    runs.append((start, n, state))

    min_run = max(1, int(round(min_run_s / dt_s)))
    merged: list[tuple[int, int, bool]] = []
    for (s, e, is_fast) in runs:
        if merged and (e - s) < min_run:
            ps, pe, pstate = merged[-1]
            merged[-1] = (ps, e, pstate or is_fast)
        else:
            merged.append((s, e, is_fast))
    return merged


@dataclass
class PyBaMMSPMeCell:

    chemistry: str
    parameter_values: "pybamm.ParameterValues"
    ocv_interp: PchipInterpolator
    capacity_ah_base: float

    def ocv(self, soc):
        z = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        return np.asarray(self.ocv_interp(z), dtype=float)

    def capacity_ah(self, soh: float) -> float:
        return float(self.capacity_ah_base * float(soh))

    def _params_for_state(self, temperature_c: float, soh: float) -> "pybamm.ParameterValues":
        p = self.parameter_values.copy()
        t_k = 273.15 + float(temperature_c)
        p["Ambient temperature [K]"] = t_k
        p["Initial temperature [K]"] = t_k
        base_scale = float(self.parameter_values["Number of electrodes connected in parallel to make a cell"])
        p["Number of electrodes connected in parallel to make a cell"] = base_scale * float(soh)
        return p

    def impedance(self, frequencies_hz, temperature_c: float, soh: float, soc: float):
        p = self._params_for_state(temperature_c, soh)
        model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal", "surface form": "algebraic"})
        eis_sim = pybamm.EISSimulation(model, parameter_values=p)
        f = np.asarray(frequencies_hz, dtype=float)
        z0 = float(np.clip(soc, 0.03, 0.97))
        sol = eis_sim.solve(f, initial_soc=z0)
        z = np.asarray(sol["Impedance [Ohm]"].data).astype(complex).reshape(-1)
        return z

    def dc_resistance_ohm(self, temperature_c: float, soh: float, soc: float) -> float:
        z = self.impedance(np.asarray([0.008]), temperature_c, soh, soc)
        return float(np.real(z[0]))

    def simulate(
        self,
        current_a,
        dt_s: float,
        soc0: float,
        temperature_c: float,
        soh: float,
        initial_rc_voltage_v=None,
    ) -> dict[str, np.ndarray]:
        current = np.asarray(current_a, dtype=float)
        n = len(current)
        t_eval = np.arange(n) * float(dt_s)
        p = self._params_for_state(temperature_c, soh)
        p["Current function [A]"] = _zero_order_hold_interpolant(t_eval, current)
        model = pybamm.lithium_ion.DFN(options=_sim_options())
        z0 = float(np.clip(soc0, 0.03, 0.97))
        cap = self.capacity_ah(soh)

        regions = _classify_fast_regions(current, dt_s)
        voltage_parts: list[np.ndarray] = []
        discharge_parts: list[np.ndarray] = []
        ocv_parts: list[np.ndarray] = []
        prev_solution = None
        for region_idx, (start, end, is_fast) in enumerate(regions):
            seg_t_eval = t_eval[start:end]
            dt_max = dt_s if is_fast else min(0.25, max(dt_s, seg_t_eval[-1] - seg_t_eval[0]))
            solver = pybamm.CasadiSolver(mode="safe", dt_max=dt_max)
            sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
            if region_idx == 0:
                sol = sim.solve(t_eval=seg_t_eval, initial_soc=z0)
            else:
                relative_t_eval = np.concatenate(([0.0], seg_t_eval - prev_solution.t[-1]))
                sol = sim.step(
                    dt=float(relative_t_eval[-1]),
                    solver=solver,
                    t_eval=relative_t_eval,
                    starting_solution=prev_solution,
                )
            expected_t = seg_t_eval
            actual_t = sol.t[-len(expected_t):]
            if sol.termination != 'final time' or len(actual_t) != len(expected_t) or not np.allclose(actual_t, expected_t, rtol=0, atol=1e-9):
                raise RuntimeError('Corrected continuation did not reach the exact requested times')
            new_voltage = np.asarray(sol["Terminal voltage [V]"].data, dtype=float)
            new_discharge = np.asarray(sol["Discharge capacity [A.h]"].data, dtype=float)
            new_ocv = np.asarray(sol["Bulk open-circuit voltage [V]"].data, dtype=float)
            if region_idx > 0:
                new_voltage = new_voltage[-len(seg_t_eval):]
                new_discharge = new_discharge[-len(seg_t_eval):]
                new_ocv = new_ocv[-len(seg_t_eval):]
            voltage_parts.append(new_voltage)
            discharge_parts.append(new_discharge)
            ocv_parts.append(new_ocv)
            prev_solution = sol
            if len(new_voltage) < len(seg_t_eval):
                break

        voltage = np.concatenate(voltage_parts)
        discharge_ah = np.concatenate(discharge_parts)
        ocv_bulk = np.concatenate(ocv_parts)
        soc_trace = np.clip(z0 - discharge_ah / cap, 0.0, 1.0)
        if len(voltage) < n:
            pad = n - len(voltage)
            voltage = np.concatenate([voltage, np.full(pad, voltage[-1])])
            soc_trace = np.concatenate([soc_trace, np.full(pad, soc_trace[-1])])
            ocv_bulk = np.concatenate([ocv_bulk, np.full(pad, ocv_bulk[-1])])
        temperature_arr = np.full(n, float(temperature_c), dtype=float)
        return {
            "voltage_v": voltage[:n],
            "soc": soc_trace[:n],
            "ocv_bulk_v": ocv_bulk[:n],
            "temperature_c": temperature_arr,
            "current_a": current,
        }


def _spread() -> dict:
    import spread_arms
    return spread_arms.arm(os.environ.get("N6_SPREAD", "orig"))


def _sim_options() -> dict:
    opts = {"thermal": "isothermal"}
    sp = _spread()
    if sp["contact"] == "series":
        opts["contact resistance"] = "true"
    if sp.get("surface"):
        opts["surface form"] = sp["surface"]
    return opts


def build_pack(chemistry: str, pack_size: int, cfg: dict, rng: np.random.Generator):
    chemistry = str(chemistry)
    if chemistry not in _PARAMETER_SETS:
        raise ValueError(f"No validated PyBaMM parameter set mapped for chemistry {chemistry!r}.")
    base_params = _base_parameter_values(chemistry)
    ocv_interp = _build_ocv_curve(chemistry, base_params)
    base_scale = float(base_params["Number of electrodes connected in parallel to make a cell"])
    base_capacity_ah = float(base_params["Nominal cell capacity [A.h]"]) * base_scale

    sp = _spread()
    lnf = lambda s, z: math.exp(-0.5 * s**2 + s * z)
    batch_capacity_factor = lnf(sp["cap_batch"], rng.standard_normal())
    batch_resistance_factor = lnf(sp["res_batch"], rng.standard_normal())

    cells: list[PyBaMMSPMeCell] = []
    for _ in range(int(pack_size)):
        cell_capacity_factor = batch_capacity_factor * lnf(sp["cap_unit"], rng.standard_normal())
        z_res = rng.standard_normal()
        cell_resistance_factor = batch_resistance_factor * lnf(sp["res_unit"], z_res)
        p = base_params.copy()
        p["Number of electrodes connected in parallel to make a cell"] = base_scale * cell_capacity_factor
        contact_r = float(p.get("Contact resistance [Ohm]", 0.0) or 0.0)
        if sp["contact"] == "legacy":
            p["Contact resistance [Ohm]"] = max(0.0, contact_r + 1e-3 * (cell_resistance_factor - 1.0))
        else:
            p["Contact resistance [Ohm]"] = max(0.0, sp["rc_mean_ohm"] + sp["rc_sd_ohm"] * z_res)
        cells.append(
            PyBaMMSPMeCell(
                chemistry=chemistry,
                parameter_values=p,
                ocv_interp=ocv_interp,
                capacity_ah_base=base_capacity_ah * cell_capacity_factor,
            )
        )
    return cells
