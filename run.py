"""Entry points for the C5/ECC 2027 public research-code release.

The historical research scripts use paths relative to their working directory.
For reports, we stage the shipped reference files in a separate output directory
and execute the original calculations there. Source and reference data stay
unchanged, and reruns do not overwrite a previous report.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
REFERENCE = ROOT / "reference"


# Published directory -> original result directory expected by historical scripts.
REPORT_STAGING_MAP: dict[str, str] = {
    "main_study_results": "results_confirm",
    "matched_model_results": "results_e1",
    "comparator_results": "new",
    "native_step_results": "sup",
    "asymmetric_kinetics_results": "posthoc_mm",
    "spme_robustness_results": "robustness",
    "dfn_robustness_results": "robustness_dfn",
}

REPORT_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("statistical_analysis/analyze_confirm.py", ()),
    ("paper_reporting/make_numbers.py", ()),
    ("paper_reporting/make_numbers_sup.py", ()),
    ("paper_reporting/make_numbers_mm.py", ()),
    ("statistical_analysis/analyze_new.py", ("--e2", "results_confirm", "--raw", "sup/raw", "--lin", "posthoc_linear_rows.json",
                                 "--new", "new", "--e1raw", "sup/e1_raw.json")),
    ("paper_reporting/make_numbers_new.py", ("new_contrast.json", "new", "results_confirm", "posthoc_linear_rows.json", "sup/e1_raw.json")),
    ("paper_reporting/make_numbers_dep.py", ("results_confirm", "new", "results_cvxlin.json")),
    ("paper_reporting/make_numbers_dl.py", ("robustness/n6_results.tar.gz",)),
    ("paper_reporting/make_numbers_wltc.py", ("robustness/n6_results.tar.gz",)),
    ("paper_reporting/make_numbers_n9.py", ("robustness/n6_results.tar.gz",)),
    ("paper_reporting/make_numbers_rev.py", ("results_confirm", "posthoc_mm_contrast.json", str(CODE))),
    ("paper_reporting/make_numbers_timing.py", ("timing/timing_fits_idle_vera.json", "timing/timing_fits_idle_cloud.json")),
    ("paper_reporting/make_numbers_n15.py", ("robustness/n15_results.tar.gz",)),
    ("paper_reporting/make_numbers_noise.py", ("noise/noise_check.json", "noise/noise_resample.json")),
    ("paper_reporting/make_numbers_dfn.py", ("robustness_dfn/n11_results.tar.gz", "robustness_dfn/n13_results.tar.gz",
                                      "results_confirm", "new", "posthoc_linear_rows.json")),
    ("paper_reporting/make_numbers_n12.py", ("robustness_dfn/n11_results.tar.gz", "robustness_dfn/n12_results.tar.gz")),
    ("paper_reporting/make_table_rob.py", ("results_confirm", "new", "posthoc_linear_rows.json",
                                     "results_cvxlin.json", "posthoc_mm/asym",
                                     "robustness/n6_results.tar.gz", "robustness_dfn/n11_results.tar.gz",
                                     "robustness_dfn/n12_results.tar.gz", "robustness_dfn/n13_results.tar.gz",
                                     "robustness/n15_results.tar.gz")),
)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Preserve the old top-level imports (e.g. cvx_estimator and pseudoeis)
    # while separating the command-line scripts into topic directories.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(CODE), env.get("PYTHONPATH", ""))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def check_sources() -> None:
    sources = sorted(CODE.rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py")) + [ROOT / "run.py"]
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    expected = (
        REFERENCE / "main_study_results",
        REFERENCE / "matched_model_results",
        REFERENCE / "comparator_results",
        REFERENCE / "spme_robustness_results" / "n6_results.tar.gz",
        REFERENCE / "dfn_robustness_results" / "n11_results.tar.gz",
    )
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing reference inputs: " + ", ".join(missing))
    print(f"Syntax checked: {len(sources)} Python files.")
    print("Reference directories and main archives: present.")
    print("Note: main traces are available separately as a verified private archive; see docs/TRACE_REPRODUCTION.md. Per-unit results_baseline/ remains unavailable.")


def run_python(script: str, args: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(CODE / script), *args],
        cwd=cwd,
        env=child_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_core() -> None:
    result = run_python("tests/test_cvx_estimator.py", (), CODE)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"Estimator test failed (exit {result.returncode}).")


def report(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new --output directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REFERENCE, destination)
    # Original analysis scripts expect historical result-folder names. Stage the
    # clearly named repository directories under those historical names only in
    # the disposable reporting workspace, leaving published inputs unchanged.
    for release_name, script_name in REPORT_STAGING_MAP.items():
        (destination / release_name).rename(destination / script_name)
    log_file = destination / "run.log"
    lines: list[str] = []
    for index, (script, args) in enumerate(REPORT_STEPS, start=1):
        print(f"[{index:02}/{len(REPORT_STEPS)}] {script}", flush=True)
        result = run_python(script, args, destination)
        lines.extend((f"\n## {script} {' '.join(args)}\n", result.stdout, result.stderr))
        log_file.write_text("\n".join(lines), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{script} failed (exit {result.returncode}); see {log_file}")
    tex_files = sorted(destination.rglob("*.tex"))
    invalid = [str(path.relative_to(destination)) for path in tex_files
               if "nan" in path.read_text(encoding="utf-8").lower()]
    if invalid:
        raise RuntimeError(f"Non-finite macros in {invalid}; see {log_file}")
    print(f"Generated {len(tex_files)} TeX files; no NaN macros detected.")
    print(f"Output: {destination}")
    print("Figures that need the omitted raw simulation traces were not regenerated.")


def main() -> int:
    parser = argparse.ArgumentParser(description="C5/ECC 2027 code and reference-result reproduction")
    parser.add_argument("command", choices=("check", "test", "report"))
    parser.add_argument("--output", default="results/report",
                        help="New output directory for report (default: results/report)")
    args = parser.parse_args()
    try:
        if args.command == "check":
            check_sources()
        elif args.command == "test":
            test_core()
        else:
            check_sources()
            report(ROOT / args.output)
    except (FileNotFoundError, FileExistsError, RuntimeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
