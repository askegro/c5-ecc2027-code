from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .models import TruthBackend, sample_heterogeneous_pack


def _import_external_module(module_name: str, search_path: str | None) -> Any:
    if search_path:
        resolved = str(Path(search_path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    return importlib.import_module(module_name)


def build_truth_pack(
    chemistry: str,
    pack_size: int,
    cfg: dict,
    rng: np.random.Generator,
) -> list[TruthBackend]:
    backend_cfg = cfg.get("truth_backend", {"kind": "synthetic_ecm"})
    kind = str(backend_cfg.get("kind", "synthetic_ecm"))
    if kind == "synthetic_ecm":
        return list(sample_heterogeneous_pack(chemistry, pack_size, cfg, rng))
    if kind != "external_python":
        raise ValueError(f"Unsupported truth backend kind: {kind}")

    module_name = str(backend_cfg.get("module", "")).strip()
    if not module_name:
        raise ValueError("truth_backend.module is required for external_python.")
    module = _import_external_module(module_name, backend_cfg.get("search_path"))
    if not hasattr(module, "build_pack"):
        raise AttributeError(f"External backend module {module_name!r} has no build_pack function.")
    pack = module.build_pack(chemistry=chemistry, pack_size=pack_size, cfg=cfg, rng=rng)
    if len(pack) != pack_size:
        raise ValueError(f"External backend returned {len(pack)} cells; expected {pack_size}.")
    required = ("simulate", "impedance", "ocv", "capacity_ah", "dc_resistance_ohm")
    for index, cell in enumerate(pack):
        missing = [name for name in required if not hasattr(cell, name)]
        if missing:
            raise TypeError(f"External cell {index} is missing methods: {missing}")
    return list(pack)
