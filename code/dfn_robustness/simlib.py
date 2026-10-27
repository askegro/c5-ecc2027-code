import os, sys, json, time
for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ.setdefault(k, "1")
os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import warnings; warnings.filterwarnings("ignore")
from pseudoeis.signals import synthetic_drive_current, validation_current_profile

CONDS = dict(cold=dict(soc=0.2, temperature_c=10, soh=1.0), nominal=dict(soc=0.5, temperature_c=25, soh=1.0),
             hot=dict(soc=0.8, temperature_c=40, soh=1.0), aged=dict(soc=0.5, temperature_c=25, soh=0.85))
import spread_arms
SPREAD = spread_arms.arm(os.environ.get("N6_SPREAD", "orig"))
COND_OFFSET = dict(nominal=0, hot=1_000_000, cold=2_000_000, aged=3_000_000)

def load_cfg():
    cfg = json.loads((HERE / "config.json").read_text())
    cfg["truth_backend"]["module"] = "campaign_v4_backend_n11"; cfg["truth_backend"]["search_path"] = str(HERE)
    return cfg

def pack_state(seed_base, cond_name, pack_id, cfg):
    import pseudoeis.pipeline as pipe
    cond = CONDS[cond_name]
    seed = seed_base + COND_OFFSET[cond_name] + pack_id * 100003
    rng = np.random.default_rng(seed)
    pack = pipe.build_truth_pack("NMC", 4, cfg, rng)
    soc0 = np.clip(rng.normal(cond["soc"], .012, 4), .05, .98); temp0 = rng.normal(cond["temperature_c"], 1.5, 4)
    soh0 = np.clip(rng.normal(cond["soh"], SPREAD["soh_sd"], 4), .70, 1)
    return seed, pack, soc0, temp0, soh0

def pack_current(seed, n, dt, cfg):
    rng = np.random.default_rng(seed + 11)
    return synthetic_drive_current(n * dt, dt, float(cfg["study"]["baseline_current_a"]), rng,
                                   minimum_current_a=float(cfg["architecture"]["minimum_current_a"]))[:n]
