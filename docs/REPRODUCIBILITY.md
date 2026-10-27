# Reproduction

## Tables and statistics

From the repository root:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py test
python run.py report
```

The report command runs 17 analysis and reporting steps using the data in `reference/`. Generated tables, LaTeX macros, and `run.log` are written to `results/report/`. Use a new output directory for each run:

```bash
python run.py report --output results/report-2
```

## Individual reporting scripts

After generating a report, run an individual reporting script from its output directory. On macOS/Linux:

```bash
cd results/report
PYTHONPATH=../../code python ../../code/paper_reporting/make_numbers_sup.py
```

On Windows, `python run.py report` configures the Python environment automatically.

## Main-study traces and fitting

See [trace inputs](TRACE_REPRODUCTION.md) to verify and extract the main-study trace archive. From `code/`:

```bash
python confirm_fit.py traces_confirm results_confirm
```

To generate the main-study traces from simulation instead:

```bash
cd code
bash run_confirm.sh
bash run_analysis.sh
```

The 576 simulations are specified in `code/confirm.list`. SPMe solves can require approximately 6 GB of memory per process.

## Additional experiments

- `code/e1_matched.py`: matched-model comparisons.
- `code/sim_mm.py` and `code/posthoc_mm.py`: asymmetric kinetics.
- `code/n1_ecm.py`, `code/n2_ms.py`, and `code/sup_raw.py`: comparator fits.
- `code/spme_robustness/`: SPMe robustness studies; see the study protocols.
- `code/dfn_robustness/`: DFN robustness studies; see the study protocols.

Run the robustness drivers from their respective study directories.

## Input requirements

Stored results support table and statistic generation. Refitting and raw-trace figures also require the [simulation trace dataset](https://doi.org/10.5281/zenodo.23701669). Per-unit `results_baseline/` fits are not included; their summary is in `reference/results_baseline_summary.json`.
