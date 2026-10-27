# Convex Grid Estimator for Battery Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23699662.svg)](https://doi.org/10.5281/zenodo.23699662)

Code and reference results for **A Convex Grid Estimator for Battery Identification from Complementary Switching Records**.

The convex grid estimator (CGE) fits a battery equivalent-circuit model with a static nonlinear kinetic term using nonnegative least squares on fixed grids of time constants and current scales. The experiments compare CGE with conventional and alternative estimators using SPMe and DFN battery simulations.

## Quick start

Python 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py test
python run.py report
```

The report command regenerates statistics, tables, and LaTeX macros from the reference results. Outputs are written to `results/report/`, with a log in `run.log`. To use a different output directory:

```bash
python run.py report --output results/report-2
```

## Repository structure

| Path | Contents |
| --- | --- |
| `code/cvx_estimator.py` | Convex grid estimator |
| `code/pseudoeis/` | Conventional ECM estimation pipeline |
| `code/ms_estimator.py` | Malchau–Schmidt comparator |
| `code/statistical_analysis/` | Statistical analyses and contrasts |
| `code/paper_reporting/` | Figures, tables, and LaTeX macros |
| `code/spme_robustness/` | SPMe robustness studies |
| `code/dfn_robustness/` | DFN robustness studies |
| `code/tests/` | Numerical tests |
| `code/seqs/` | Identification sequences |
| `reference/` | Main-study, comparator, robustness, timing, and noise results |
| `docs/` | Reproduction instructions |
| `tools/` | Trace archive verification |

## Reproduction

See [the reproduction guide](docs/REPRODUCIBILITY.md) for reporting and simulation commands, and [trace inputs](docs/TRACE_REPRODUCTION.md) for verification of the complete 576-file main-study archive.

The main-study raw simulation traces are not included in this repository. They are required for refitting and raw-trace figures. Per-unit baseline fits are not included; their summary is available in `reference/results_baseline_summary.json`.

## License and citation

MIT License. Copyright © 2026 Albert Škegro.

Use [CITATION.cff](CITATION.cff) or the [Zenodo record](https://doi.org/10.5281/zenodo.23699662) to cite this software.
