import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pseudoeis.signals import ExcitationDesign, ExcitationSegment, candidate_segment_pool, spectral_coverage_score, concatenate_segments
import pseudoeis.pipeline as pipe


def frozen_designs(ids=(4786, 9588)):
    frozen = json.loads((HERE / "frozen_designs.json").read_text())
    out = {}
    for cid in ids:
        f = frozen[str(cid)]
        segs = tuple(ExcitationSegment(s["name"], np.asarray(s["q"]), s["dt_s"], s["bit_period_s"], s["lfsr_order"], s["seed_index"]) for s in f["segments"])
        out[f"s{cid}"] = ExcitationDesign("controlled", np.asarray(f["q"]), f["dt_s"], segs, np.asarray(f["target_frequencies_hz"]), f["information_score"], dict(candidate_id=cid))
    return out


def short_design(cfg):
    common_dt = pipe._common_dt(cfg)
    target = np.asarray(cfg["excitation"]["proposed"]["target_frequencies_hz"], float); target = target[(target > 0) & (target < 0.45 / common_dt)]
    base_seed = int(cfg["project"]["random_seed"]); nb = int(cfg["excitation"].get("baseline_candidate_seeds_per_polynomial", 6))
    dcfg = cfg["excitation"]["short_baseline"]; segs = []
    for band_idx, band_name in enumerate(("low_band", "high_band")):
        cands = candidate_segment_pool(band_name, dcfg[band_name], common_dt, max(4, nb), base_seed + 200_000 + 10_000 * band_idx + 1)
        segs.append(max(cands, key=lambda c: spectral_coverage_score(c.q, c.dt_s, target)))
    return concatenate_segments("short_baseline", segs, target, 0.0)


def all_designs():
    cfg = json.loads((HERE / "config.json").read_text())
    d = frozen_designs()
    d["short74"] = short_design(cfg)
    return d
