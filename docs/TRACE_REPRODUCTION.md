# Main-study trace inputs

Download the [main-study simulation dataset](https://doi.org/10.5281/zenodo.23701669) from Zenodo. It contains the trace archive, a file manifest, checksums, and loading instructions.

The archive contains **576 files for 96 units**: 24 units each under cold, nominal, hot, and aged conditions. Each unit has six records:

- `s4786.npz`: identification sequence.
- `s9588.npz`: coverage sequence.
- `short74.npz`: short sequence.
- `passive.npz`: passive record.
- `val.npz`: validation profile.
- `zref.json`: reference impedance.

## Verify the download

Download `traces_confirm.tgz`, `trace_manifest.txt`, `SHA256SUMS`, `README.txt`, and `LICENSE` into one directory.

On Linux/macOS:

```bash
sha256sum -c SHA256SUMS
```

On Windows PowerShell:

```powershell
Get-FileHash traces_confirm.tgz -Algorithm SHA256
```

The archive is 285,107,260 bytes. SHA-256:

```text
474093bb26dcf2686d0840e2b84f467547062e849cce4b6c8499e5bdcf34f88b
```

The file manifest matches all 576 output paths in `code/confirm.list`.

## Extract

From the repository root:

```bash
tar -xzf PATH_TO_DATA/traces_confirm.tgz -C code
```

This creates `code/traces_confirm/`. Use a working copy without an existing trace directory.

## Fit

From `code/`:

```bash
python confirm_fit.py traces_confirm results_confirm
```

Existing per-unit fit outputs are skipped. Use a fresh output directory to rerun all fits.

## Figures

First run `python run.py report` and copy the extracted `traces_confirm/` directory into `results/report/`. From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path code).Path
Push-Location results/report
try { python ../../code/paper_reporting/make_figs_ecc.py } finally { Pop-Location }
```

## Original split archive

For a local copy stored as `traces_confirm.tgz.part00` through `part15`, use Python 3.11 or later to verify the parts before extraction:

```text
python tools/verify_confirm_traces.py --parts "PATH_TO_PARTS" --manifest code/confirm.list --report local_reports/trace_verification.json
```

This checks gzip integrity, manifest membership, duplicate paths, record types, and unit counts. Use `--work-dir` to select a writable directory for the temporary joined archive.
