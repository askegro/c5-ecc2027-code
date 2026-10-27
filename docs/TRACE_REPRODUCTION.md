# Main-study trace inputs

The main-study archive contains **576 files for 96 units**: 24 units each under cold, nominal, hot, and aged conditions. Each unit has six records:

- `s4786.npz`: identification sequence.
- `s9588.npz`: coverage sequence.
- `short74.npz`: short sequence.
- `passive.npz`: passive record.
- `val.npz`: validation profile.
- `zref.json`: reference impedance.

The archive is distributed separately as `traces_confirm.tgz.part00` through `part15`.

## Verify

From the repository root, with Python 3.11 or later:

```text
python tools/verify_confirm_traces.py --parts "PATH_TO_PARTS" --manifest code/confirm.list --report local_reports/trace_verification.json
```

The verifier checks gzip integrity, exact membership against `confirm.list`, duplicate paths, record types, and unit counts. Use `--work-dir` to choose a writable directory for its temporary archive.

The assembled archive is 285,107,260 bytes. SHA-256:

```text
474093bb26dcf2686d0840e2b84f467547062e849cce4b6c8499e5bdcf34f88b
```

## Assemble and extract

On macOS/Linux:

```bash
cat PATH_TO_PARTS/traces_confirm.tgz.part* > traces_confirm.tgz
tar -xzf traces_confirm.tgz -C code
```

On Windows PowerShell:

```powershell
$partsDir = 'PATH_TO_PARTS'
$archivePath = 'traces_confirm.tgz'
$destination = [IO.File]::Create($archivePath)
try {
    0..15 | ForEach-Object {
        $partPath = Join-Path $partsDir ('traces_confirm.tgz.part{0:D2}' -f $_)
        $source = [IO.File]::OpenRead($partPath)
        try { $source.CopyTo($destination) } finally { $source.Dispose() }
    }
} finally { $destination.Dispose() }
tar -xzf $archivePath -C code
```

Extract into a working copy without an existing `code/traces_confirm/` directory. The archive contains that directory prefix.

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
